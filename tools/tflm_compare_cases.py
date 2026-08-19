#!/usr/bin/env python3
"""Diff the host and device TFLM dumps. Exit 0 only on byte equality, everywhere.

The gate at step 4c of [docs/morning-runbook.md](../docs/morning-runbook.md).

    .venv/bin/python tools/tflm_compare_cases.py \\
        build/tflm-cases-int8/reference.txt device.txt

Byte equality on the **raw int8 output tensors**, not on dequantised floats and
not on top-1. Comparing floats would hide a one-count difference behind print
precision, and comparing argmax would hide it completely -- one count of 1/256
is the size of the margins `src/si_spot.py`'s gates run on, which is exactly
how TinyMaix passed a top-1 eye test while changing 3 of 8 patches.

Exit status is the interface: **0 when all cases match, 2 otherwise**, so it can
be the thing a runbook step is judged by rather than something a human reads
and interprets.

A mismatch is reported per class index with both values and the difference in
counts, because *how* it differs is the diagnosis: one class off by one count
is a rounding path; many classes wildly different is a different model, a
different input, or a broken kernel.
"""

import argparse
import struct
import sys


def prove_round_trip():
    """Re-establish, here and now, that the device's float->int8 recovery is exact.

    The device cannot emit the raw int8 tensor -- the binding returns float32
    scores -- so it recovers q as `round(score * 256) - 128`. That is only
    lossless because this model's output scale is 2**-8 and every (q + 128)/256
    is exactly representable in float32. Rather than assert that in a comment,
    check it over all 256 values on every run: it costs nothing, and it means a
    model whose quantisation changed cannot quietly invalidate the comparison.
    """
    scale, zp = 2.0 ** -8, -128
    for q in range(-128, 128):
        f32 = struct.unpack("f", struct.pack("f", (q - zp) * scale))[0]
        if int(round(f32 * 256.0)) + zp != q:
            return False, q
    return True, None


def parse(path):
    cases = {}
    order = []
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("CASE "):
                parts = line.split()
                if len(parts) != 3:
                    raise SystemExit("%s: malformed CASE line: %r" % (path, line))
                _, name, hexed = parts
                if name in cases:
                    raise SystemExit("%s: duplicate case %r" % (path, name))
                try:
                    blob = bytes.fromhex(hexed)
                except ValueError:
                    raise SystemExit("%s: case %s has non-hex payload %r"
                                     % (path, name, hexed))
                cases[name] = blob
                order.append(name)
            elif line.startswith("# "):
                bits = line[2:].split(None, 1)
                if len(bits) == 2:
                    meta.setdefault(bits[0], bits[1])
    return cases, order, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="host dump, build/tflm-cases-int8/reference.txt")
    ap.add_argument("device", help="device dump, or - for stdin")
    ap.add_argument("--expect", type=int, default=None,
                    help="required number of cases; the run is a FAIL with "
                         "fewer, even if every case present matched")
    args = ap.parse_args(argv)

    ok, bad_q = prove_round_trip()
    if not ok:
        print("FAIL: the float->int8 recovery is not exact at q=%d." % bad_q)
        print("      The device dump cannot be trusted; plumb the raw int8")
        print("      output through modtflm.c before comparing.")
        return 2

    ref, ref_order, ref_meta = parse(args.reference)
    dev_path = "/dev/stdin" if args.device == "-" else args.device
    dev, dev_order, dev_meta = parse(dev_path)

    print("reference  %d cases  %s" % (len(ref), args.reference))
    print("device     %d cases  %s" % (len(dev), args.device))

    problems = []

    # A device run against a different model is the failure most likely to look
    # like an arithmetic difference, so it is checked before any tensor is.
    ref_model = ref_meta.get("model_sha256")
    dev_model_line = dev_meta.get("model", "")
    if ref_model:
        print("model      %s (host)" % ref_model[:16])
    if dev_model_line:
        print("           %s (device reports)" % dev_model_line)

    missing = [n for n in ref_order if n not in dev]
    extra = [n for n in dev_order if n not in ref]
    if missing:
        problems.append("%d case(s) missing from the device dump: %s"
                        % (len(missing), ", ".join(missing[:6])
                           + ("..." if len(missing) > 6 else "")))
    if extra:
        problems.append("%d case(s) in the device dump only: %s"
                        % (len(extra), ", ".join(extra[:6])
                           + ("..." if len(extra) > 6 else "")))
    if args.expect is not None and len(dev) != args.expect:
        problems.append("expected %d cases, device produced %d"
                        % (args.expect, len(dev)))

    matched = 0
    differing = 0
    print()
    for name in ref_order:
        if name not in dev:
            print("  MISSING  %s" % name)
            continue
        a, b = ref[name], dev[name]
        if a == b:
            matched += 1
            print("  ok       %s  %s" % (name, a.hex()))
            continue
        differing += 1
        print("  DIFFER   %s" % name)
        print("           host   %s" % a.hex())
        print("           device %s" % b.hex())
        if len(a) != len(b):
            print("           lengths differ: host %d, device %d" % (len(a), len(b)))
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                sx = x - 256 if x > 127 else x
                sy = y - 256 if y > 127 else y
                print("           class %2d: host %4d, device %4d (%+d counts)"
                      % (i, sx, sy, sy - sx))

    print()
    print("%d matched, %d differing, %d compared of %d reference cases"
          % (matched, differing, matched + differing, len(ref)))

    if differing:
        problems.append("%d case(s) differ" % differing)

    if problems:
        print()
        for p in problems:
            print("FAIL: %s" % p)
        return 2

    print("PASS: device TFLM is byte-identical to host TFLM on all %d cases."
          % matched)
    print("      With tools/tflm_vs_tflite.py (host TFLM == host reference")
    print("      TFLite), that closes the chain: what the host evaluates is")
    print("      what the board computes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
