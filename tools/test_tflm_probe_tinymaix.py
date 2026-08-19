#!/usr/bin/env python3
"""Exercise `tools/tflm_probe.py`'s TinyMaix path on a host that has no TinyMaix.

    ./tools/test_tflm_probe_tinymaix.py

**This exists because that path shipped to a bench untested and failed there.**
The probe passed `bytes` straight from `read()` to `emlearn_cnn_int8.run()`,
which rejects anything whose typecode `mp_get_buffer` will not report --
`TypeError: object with buffer protocol required`, once per case, 30 times.
TFLM accepts the same object, so only half the run fell over and the half that
mattered was fine. The defect is documented in the probe's own header, in
`src/si_spot.py`, and in `docs/cnn-on-device.md`; it was walked into anyway,
because no host has `emlearn_cnn_int8` and so nothing exercised the branch.

A stub fixes that. It is written to be **strict in exactly the way emlearn is**
-- reject `bytes`, accept `array('B')`, and nothing else -- so that a probe
which would fail on the board fails here first, in two seconds, with no port
held.

The stub deliberately does not attempt to be TinyMaix. It returns arbitrary
scores. This tests the calling convention, which is the thing that broke; the
arithmetic is TinyMaix's own business and `docs/cnn-on-device.md` owns it.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
CASES = os.path.join(REPO, "build", "tflm-cases")

STUB = '''\
"""Stand-in for emlearn_cnn_int8, strict in the one way that matters.

`mp_get_buffer` does not report typecode 'B' for a `bytes` or a `bytearray`, so
the real wrapper raises for both and accepts only `array('B')`. Reproduced here
exactly, because a lenient stub would pass a probe that the board rejects --
which is the failure this file exists to prevent, not to imitate.
"""

from array import array

_ARRAY_TYPE = type(array("B", b""))


def _require_array_B(obj, what):
    if not isinstance(obj, _ARRAY_TYPE):
        raise TypeError("object with buffer protocol required")
    return obj


class _Model:
    def output_dimensions(self):
        return (22,)

    def run(self, patch, scores):
        _require_array_B(patch, "input")
        # Arbitrary but deterministic, so a capture is reproducible.
        total = 0
        for i in range(0, len(patch), 97):
            total += patch[i]
        for i in range(len(scores)):
            scores[i] = ((total + i * 7) % 256) / 256.0


def new(data):
    if not isinstance(data, _ARRAY_TYPE):
        raise ValueError("model should be bytes")
    return _Model()
'''


def main():
    if not os.path.isdir(os.path.join(CASES, "cases")):
        print("no staged cases at %s -- run ./tools/make_tflm_cases.py first"
              % CASES)
        return 1

    micropython = os.environ.get("MICROPYTHON")
    if not micropython or not os.path.exists(micropython):
        print("SKIP: set MICROPYTHON to a unix-port binary built with")
        print("      USER_C_MODULES=firmware/usermod (the `tflm` module must")
        print("      be present, or the probe has nothing to compare against).")
        return 0

    work = tempfile.mkdtemp(prefix="tflm-probe-test-")
    try:
        # The probe reads `<CASES>/manifest.txt` -- inside the cases directory,
        # because that is where the deploy instructions put it on the board.
        # make_tflm_cases.py writes it one level up, so the staging step copies
        # it in. Reproduced here rather than symlinking the parent, so this
        # test fails the same way a mis-staged board would.
        os.mkdir(os.path.join(work, "cases"))
        for name in os.listdir(os.path.join(CASES, "cases")):
            os.symlink(os.path.join(CASES, "cases", name),
                       os.path.join(work, "cases", name))
        shutil.copy(os.path.join(CASES, "manifest.txt"),
                    os.path.join(work, "cases", "manifest.txt"))
        os.symlink(os.path.join(CASES, "si_model.tflite"),
                   os.path.join(work, "si_model.tflite"))
        open(os.path.join(work, "emlearn_cnn_int8.py"), "w").write(STUB)
        # The probe loads this before it will enable the TinyMaix half. The
        # stub ignores the contents; only `array('B', blob)` reaching `new()`
        # is being tested.
        open(os.path.join(work, "si_model.tmdl"), "wb").write(b"\x00" * 64)

        source = open(os.path.join(HERE, "tflm_probe.py")).read()
        source = source.replace('CASES = "/cases"', 'CASES = "cases"')
        open(os.path.join(work, "probe.py"), "w").write(source)

        result = subprocess.run([micropython, "probe.py"], cwd=work,
                                capture_output=True, text=True)
        output = result.stdout + result.stderr

        tflm = sum(1 for l in output.splitlines()
                   if l.startswith("SCORE") and "runtime=tflm " in l)
        tinymaix = sum(1 for l in output.splitlines()
                       if l.startswith("SCORE") and "runtime=tinymaix " in l)
        probe_errors = [l for l in output.splitlines()
                        if l.startswith("PROBE-ERROR")]

        print("TinyMaix reported available :",
              "TinyMaix           : ready" in output)
        print("runtime=tflm SCORE lines    : %d" % tflm)
        print("runtime=tinymaix SCORE lines: %d" % tinymaix)
        print("PROBE-ERROR lines           : %d" % len(probe_errors))
        for line in probe_errors[:3]:
            print("   " + line)

        failures = []
        if "TinyMaix           : ready" not in output:
            failures.append("the stub was not picked up -- the TinyMaix half "
                            "never ran, so this test proved nothing")
        if probe_errors:
            failures.append("%d PROBE-ERROR line(s)" % len(probe_errors))
        if tinymaix != tflm or tflm == 0:
            failures.append("expected equal non-zero counts, got tflm=%d "
                            "tinymaix=%d" % (tflm, tinymaix))

        print()
        if failures:
            print("FAIL")
            for line in failures:
                print("  - " + line)
            if result.returncode != 0:
                print()
                print("probe exit %d, tail:" % result.returncode)
                for line in output.splitlines()[-15:]:
                    print("   " + line)
            return 1
        print("PASS -- both runtimes emit a SCORE line for every case, and the")
        print("        emlearn typecode rule is satisfied rather than assumed")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
