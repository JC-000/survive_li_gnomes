#!/usr/bin/env python3
"""Materialise the 30 comparison cases, and dump what the host computes for them.

The host half of step 4c in [docs/morning-runbook.md](../docs/morning-runbook.md).
`tools/tflm_vs_tflite.py` already proves **host TFLM == host reference TFLite**
on these cases; this proves nothing on its own. What it does is freeze the
cases as files the board can read, and record the host's answers in a form a
comparator can diff, so that the remaining claim -- **device TFLM == host
TFLM** -- can be measured rather than assumed.

    ./tools/build_tflm_host.sh                  # once, builds the host library
    .venv/bin/python tools/tflm_cases.py        # writes build/tflm-cases-int8/

Out of that directory:

    <case>.i8         2080 raw int8 bytes, exactly what run_int8() is fed
    reference.txt     the host's int8 output tensor for each case
    manifest.json     names, sizes, hashes, and the model this was run against

Then `tools/tflm_device_cases.py` runs on the board over the same `.i8` files
and `tools/tflm_compare_cases.py` diffs the two dumps.

**The cases.** Eight are the `build/kw_unknown_*.bin` patches -- the ones where
TinyMaix changes top-1 on 3 of 8, which is the whole reason this project stopped
trusting it. The rest are real recordings from `takes/` and `takes-oov/`, run
through `tools/si_features.py` exactly as the board's front end runs them; the
endpointer rejects some, and the rejections are printed rather than hidden,
because "22 takes" is a count somebody will want to reconcile.

**Why the input files are int8 and not uint8.** `si_patch.py` emits uint8 with
+128 applied, which is what `Model.run()` takes, and `run_int8()` takes the
signed form. Both exist and agree (`tools/test_tflm_module.py` checks that they
agree to the last bit). The signed form is used here because it is what the
host's `tflm_invoke` takes, so the two sides are fed byte-identical buffers and
the +128 transport is not silently part of what is being tested.
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import tflm_vs_tflite as tvt  # noqa: E402  (needs HERE on the path first)

# NOT build/tflm-cases: that belongs to tools/make_tflm_cases.py, fw-tflm's
# parallel staging of the same 30 cases, which owns reference.txt, manifest.txt
# and cases/ there. The two chains were written the same night and collided --
# this one overwrote that reference.txt, which would have made their
# check_tflm_device.py read a dump in a format it does not parse. Separate
# directories mean either chain can be run without disturbing the other.
OUT_DIR = os.path.join(REPO, "build", "tflm-cases-int8")
MODEL = os.path.join(REPO, "build", "si_real.tflite")
BINS = os.path.join(REPO, "build", "kw_unknown_*.bin")
TAKES = [os.path.join(REPO, "takes"), os.path.join(REPO, "takes-oov")]
ARENA = 64 * 1024

DUMP_VERSION = 1


def case_name(raw_name):
    """`kw_unknown_0.bin` and `alice_mother_3.wav` both become bare stems.

    The name is the join key between the two dumps and the filename on the
    board, so it has to survive both: no extension, no spaces.
    """
    stem = os.path.splitext(raw_name)[0]
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in stem)


def collect():
    cases = tvt.patches_from_bins(BINS) + tvt.patches_from_takes(TAKES, REPO)
    named = []
    seen = {}
    for raw_name, patch in cases:
        name = case_name(raw_name)
        if name in seen:
            # Two sources collapsing to one name would make the dumps disagree
            # about which case is which, and the comparator would be comparing
            # a case against a different case while reporting a pass.
            raise SystemExit("duplicate case name %r (from %r and %r)"
                             % (name, seen[name], raw_name))
        seen[name] = raw_name
        named.append((name, np.asarray(patch, dtype=np.int16)))
    return named


def main():
    if not os.path.exists(MODEL):
        raise SystemExit("no model at %s" % MODEL)

    model_bytes = open(MODEL, "rb").read()
    model_sha = hashlib.sha256(model_bytes).hexdigest()

    lib_path = os.path.join(REPO, "build", "tflm-host", "libtflm_host.dylib")
    if not os.path.exists(lib_path):
        lib_path = lib_path[:-6] + ".so"
    if not os.path.exists(lib_path):
        raise SystemExit("no host TFLM library; run ./tools/build_tflm_host.sh")
    lib = tvt.load_lib(lib_path)
    m = tvt.Tflm(lib, model_bytes, ARENA)

    print("model      %s" % os.path.relpath(MODEL, REPO))
    print("           sha256 %s" % model_sha)
    print("input      %d int8, scale %g zp %d" % (m.n_in, m.in_scale, m.in_zp))
    print("output     %d int8, scale %g zp %d" % (m.n_out, m.out_scale, m.out_zp))
    print("arena_used %d of %d" % (m.arena_used, ARENA))

    # The device cannot emit the raw int8 output tensor -- the MicroPython
    # binding's run_int8() writes float32 scores and the shim's int8 out
    # parameter is not plumbed through it. It does not need to: this model's
    # output scale is 2**-8 with zero point -128, so every score is
    # (q + 128) / 256, which is exactly representable in float32 for all 256
    # values of q. The int8 tensor is therefore recoverable from the scores
    # with no tolerance and no rounding policy -- see tflm_compare_cases.py,
    # which proves the round trip exhaustively before it uses it.
    #
    # If that ever stops being true -- a retrained model with an awkward scale,
    # or a second output tensor -- this assertion fails here, on the host,
    # rather than as a mysterious mismatch at the bench.
    if not (m.out_scale == 2.0 ** -8 and m.out_zp == -128):
        raise SystemExit(
            "output quantisation is scale=%r zp=%r, not 2**-8 / -128.\n"
            "The float scores no longer carry the int8 tensor losslessly, so\n"
            "the device dump needs the raw int8 path plumbed through\n"
            "firmware/usermod/tflm/modtflm.c before this comparison is valid."
            % (m.out_scale, m.out_zp))

    cases = collect()
    if not cases:
        raise SystemExit("no cases collected")

    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".i8"):
            os.remove(os.path.join(OUT_DIR, stale))

    manifest = {
        "dump_version": DUMP_VERSION,
        "model": os.path.relpath(MODEL, REPO),
        "model_sha256": model_sha,
        "n_in": m.n_in,
        "n_out": m.n_out,
        "out_scale": m.out_scale,
        "out_zero_point": m.out_zp,
        "arena_bytes": ARENA,
        "cases": [],
    }

    lines = ["# tflm case dump v%d" % DUMP_VERSION,
             "# side host",
             "# model_sha256 %s" % model_sha,
             "# n_out %d" % m.n_out]

    print("\n%-28s %6s  %s" % ("case", "bytes", "int8 output tensor"))
    for name, patch in cases:
        if patch.size != m.n_in:
            raise SystemExit("case %s is %d values, model wants %d"
                             % (name, patch.size, m.n_in))
        i8 = patch.astype(np.int8)
        path = os.path.join(OUT_DIR, name + ".i8")
        with open(path, "wb") as f:
            f.write(i8.tobytes())

        out = m.invoke(i8)
        hexed = out.astype(np.uint8).tobytes().hex()
        lines.append("CASE %s %s" % (name, hexed))
        manifest["cases"].append({
            "name": name,
            "file": name + ".i8",
            "sha256": hashlib.sha256(i8.tobytes()).hexdigest(),
        })
        print("%-28s %6d  %s" % (name, i8.size, hexed))

    lines.append("# end %d cases" % len(cases))
    with open(os.path.join(OUT_DIR, "reference.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    total = sum(len(open(os.path.join(OUT_DIR, c["file"]), "rb").read())
                for c in manifest["cases"])
    print("\n%d cases, %d bytes of input tensors -> %s"
          % (len(cases), total, os.path.relpath(OUT_DIR, REPO)))
    print("copy them to the board with:")
    print("    uvx mpremote connect $PORT mkdir :cases")
    print("    for f in %s/*.i8; do \\" % os.path.relpath(OUT_DIR, REPO))
    print("        uvx mpremote connect $PORT cp \"$f\" \":cases/$(basename $f)\"; done")


if __name__ == "__main__":
    main()
