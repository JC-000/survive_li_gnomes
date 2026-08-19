#!/usr/bin/env python3
"""Exercise the whole 4c chain on the host, so the bench is not where it debuts.

    .venv/bin/python tools/test_tflm_cases.py

`tools/tflm_device_cases.py` is written for the board, but nothing in it is
board-specific except the `tflm` module it imports. So this stands in a `tflm`
built on the host -- the same `tflm_shim.cpp`, through ctypes -- runs the device
script exactly as MicroPython would, and feeds its output to
`tools/tflm_compare_cases.py`.

What that proves, and what it does not. It proves the driver runs, the case
files parse, the float-to-int8 recovery is exact against real outputs, the dump
format round-trips, and the comparator agrees when it should and **disagrees
when it should not** -- the negative control below, which matters more than the
positive one, because a gate that cannot fail is worse than no gate. It does not
prove anything about the RP2350: same source, different compiler, different
target. That is what the board is for, and it is step 4c.
"""

import ctypes
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import tflm_vs_tflite as tvt  # noqa: E402

CASES = os.path.join(REPO, "build", "tflm-cases-int8")
MODEL = os.path.join(REPO, "build", "si_real.tflite")


def host_tflm_module():
    """A stand-in for the firmware's `tflm`, backed by the host shared library.

    Deliberately implements only what tflm_device_cases.py uses, and with the
    same signatures the real module has. If the device script starts using
    something else, this fails loudly here rather than diverging quietly.
    """
    lib_path = os.path.join(REPO, "build", "tflm-host", "libtflm_host.dylib")
    if not os.path.exists(lib_path):
        lib_path = lib_path[:-6] + ".so"
    if not os.path.exists(lib_path):
        raise SystemExit("no host TFLM library; run ./tools/build_tflm_host.sh")
    lib = tvt.load_lib(lib_path)

    class Model:
        def __init__(self, blob, arena):
            self._m = tvt.Tflm(lib, bytes(blob), len(arena))

        def input_dimensions(self):
            return (1, self._m.n_in)

        def output_dimensions(self):
            return (1, self._m.n_out)

        def arena_used(self):
            return self._m.arena_used

        def run_int8(self, patch, scores):
            if len(patch) != self._m.n_in:
                raise ValueError("input is %d bytes, model wants %d"
                                 % (len(patch), self._m.n_in))
            n = self._m.n_out
            fbuf = (ctypes.c_float * n)()
            rc = lib.tflm_invoke(self._m.h, patch, len(patch), None, fbuf, n)
            if rc != 0:
                raise RuntimeError("tflm_invoke rc=%d" % rc)
            for i in range(n):
                scores[i] = fbuf[i]

    mod = types.ModuleType("tflm")
    mod.new = lambda blob, arena: Model(blob, arena)
    return mod


def run_device_script(workdir):
    """Run tflm_device_cases.py as MicroPython would: cwd = the filesystem root."""
    src = open(os.path.join(HERE, "tflm_device_cases.py")).read()
    sys.modules["tflm"] = host_tflm_module()
    cwd, out = os.getcwd(), io.StringIO()
    stdout = sys.stdout
    try:
        os.chdir(workdir)
        sys.stdout = out
        exec(compile(src, "tflm_device_cases.py", "exec"), {"__name__": "__main__"})
    finally:
        sys.stdout = stdout
        os.chdir(cwd)
        del sys.modules["tflm"]
    return out.getvalue()


def compare(reference, device_dump, expect=None):
    cmd = [sys.executable, os.path.join(HERE, "tflm_compare_cases.py"),
           reference, device_dump]
    if expect is not None:
        cmd += ["--expect", str(expect)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    if not os.path.isdir(CASES) or not os.path.exists(
            os.path.join(CASES, "reference.txt")):
        raise SystemExit("no cases; run .venv/bin/python tools/tflm_cases.py")

    work = tempfile.mkdtemp(prefix="tflm-cases-")
    try:
        shutil.copy(MODEL, os.path.join(work, "si_real.tflite"))
        os.mkdir(os.path.join(work, "cases"))
        n_cases = 0
        for f in sorted(os.listdir(CASES)):
            if f.endswith(".i8"):
                shutil.copy(os.path.join(CASES, f),
                            os.path.join(work, "cases", f))
                n_cases += 1

        print("running tflm_device_cases.py against the host library, "
              "%d cases" % n_cases)
        dump = run_device_script(work)
        n_lines = sum(1 for line in dump.splitlines() if line.startswith("CASE "))
        print("  produced %d CASE lines" % n_lines)

        dump_path = os.path.join(work, "device.txt")
        with open(dump_path, "w") as f:
            f.write(dump)

        reference = os.path.join(CASES, "reference.txt")

        print("\n-- positive control: it should agree --")
        rc, out = compare(reference, dump_path, expect=n_cases)
        print(out.rstrip())
        if rc != 0:
            print("\nFAIL: the comparator rejected a run it should have passed.")
            return 1

        print("\n-- negative control: one flipped count must be caught --")
        # One class of one case, off by a single count -- 1/256, the smallest
        # difference that exists and exactly the size TinyMaix's disagreement
        # hid behind. If the gate cannot see this, it cannot see anything.
        lines = dump.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("CASE "):
                tag, name, hexed = line.split()
                blob = bytearray(bytes.fromhex(hexed))
                blob[3] = (blob[3] + 1) & 0xFF
                lines[i] = "CASE %s %s" % (name, blob.hex())
                print("  perturbed %s class 3 by +1 count" % name)
                break
        tampered = os.path.join(work, "device-tampered.txt")
        with open(tampered, "w") as f:
            f.write("\n".join(lines) + "\n")
        rc, out = compare(reference, tampered, expect=n_cases)
        print("  comparator exit status %d" % rc)
        if rc == 0:
            print(out.rstrip())
            print("\nFAIL: the comparator passed a tampered dump. The gate is "
                  "useless as written.")
            return 1
        detail = [l for l in out.splitlines()
                  if "class  3" in l or "DIFFER" in l or l.startswith("FAIL")]
        for line in detail:
            print("  %s" % line.strip())

        print("\n-- negative control: a truncated run must be caught --")
        short = os.path.join(work, "device-short.txt")
        with open(short, "w") as f:
            kept = 0
            for line in dump.splitlines():
                if line.startswith("CASE "):
                    kept += 1
                    if kept > n_cases - 2:
                        continue
                f.write(line + "\n")
        rc, out = compare(reference, short, expect=n_cases)
        print("  dropped 2 cases; comparator exit status %d" % rc)
        if rc == 0:
            print("\nFAIL: the comparator passed a run that lost two cases.")
            return 1
        for line in out.splitlines():
            if line.startswith("FAIL"):
                print("  %s" % line)

        print("\nPASS: the 4c chain runs end to end on the host, and the gate "
              "fails when it should.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
