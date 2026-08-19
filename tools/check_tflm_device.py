#!/usr/bin/env python3
"""Decide the byte-for-byte question from a `tools/tflm_probe.py` capture.

    ./tools/check_tflm_device.py /tmp/device.txt

Exit 0 iff every `runtime=tflm` case in the capture matches
`build/tflm-cases/reference.txt` exactly, on all 22 integers, with no case
missing and no `PROBE-ERROR` line.

**This exists so nobody reads two columns of 22 integers at a bench.** The
whole TFLM case is "the board computes what the host computes"; a check that
depends on someone spotting a single count out of 5,280 is not a check.

It also reports what TinyMaix did on the same inputs, if the capture has it,
because that comparison is the one `docs/cnn-on-device.md` records as 3-of-8
disagreement and the one the morning is there to close.
"""

import argparse
import os
import sys


def parse(path, want_runtime=None):
    """-> {(runtime, name): (set, q, fields)}, plus probe errors."""
    rows = {}
    errors = []
    for line in open(path):
        line = line.strip()
        if line.startswith("PROBE-ERROR"):
            errors.append(line)
            continue
        if not line.startswith("SCORE"):
            continue          # diagnostics around the SCORE lines are ignored
        fields = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        if "q" not in fields or "name" not in fields:
            continue
        q = [int(v) for v in fields["q"].split(",")]
        if len(q) != 22:
            raise SystemExit(
                "%s: %s carries %d scores, expected 22. A device and a harness "
                "that disagree about the model is worth stopping on."
                % (path, fields["name"], len(q)))
        runtime = fields.get("runtime", "?")
        if want_runtime and runtime != want_runtime:
            continue
        rows[(runtime, fields["name"])] = (fields.get("set", "?"), q, fields)
    return rows, errors


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", help="output of tools/tflm_probe.py")
    ap.add_argument("--reference",
                    default=os.path.join(repo, "build", "tflm-cases",
                                         "reference.txt"))
    args = ap.parse_args(argv)

    reference, _ = parse(args.reference)
    device, probe_errors = parse(args.capture)

    host = {name: q for (runtime, name), (_s, q, _f) in reference.items()}
    tflm = {name: (s, q, f) for (runtime, name), (s, q, f) in device.items()
            if runtime == "tflm"}
    tinymaix = {name: (s, q, f) for (runtime, name), (s, q, f) in device.items()
                if runtime == "tinymaix"}

    print("host reference : %d cases (%s)" % (len(host), args.reference))
    print("device tflm    : %d cases" % len(tflm))
    print("device tinymaix: %d cases" % len(tinymaix))

    failures = []

    missing = sorted(set(host) - set(tflm))
    if missing:
        failures.append("%d case(s) missing from the capture: %s"
                        % (len(missing), ", ".join(missing)))
    extra = sorted(set(tflm) - set(host))
    if extra:
        failures.append("%d case(s) in the capture with no host reference: %s"
                        % (len(extra), ", ".join(extra)))

    print()
    print("== TFLM vs host, per case ==")
    exact = 0
    worst = 0
    for name in sorted(set(host) & set(tflm)):
        case_set, q, _fields = tflm[name]
        delta = [abs(a - b) for a, b in zip(q, host[name])]
        biggest = max(delta) if delta else 0
        worst = max(worst, biggest)
        if biggest == 0:
            exact += 1
            print("  %-22s %-9s exact" % (name, case_set))
        else:
            argmax_device = max(range(22), key=lambda i: q[i])
            argmax_host = max(range(22), key=lambda i: host[name][i])
            print("  %-22s %-9s DIFFERS max=%d  argmax device=%d host=%d"
                  % (name, case_set, biggest, argmax_device, argmax_host))
            failures.append("%s differs from the host by up to %d counts"
                            % (name, biggest))

    if tinymaix:
        print()
        print("== TinyMaix vs host, same inputs, same session ==")
        tm_exact = 0
        tm_top1 = 0
        for name in sorted(set(host) & set(tinymaix)):
            _case_set, q, _fields = tinymaix[name]
            delta = max(abs(a - b) for a, b in zip(q, host[name]))
            same_top1 = (max(range(22), key=lambda i: q[i])
                         == max(range(22), key=lambda i: host[name][i]))
            tm_exact += delta == 0
            tm_top1 += same_top1
            print("  %-22s max=%-4d top-1 %s"
                  % (name, delta, "agrees" if same_top1 else "DIFFERS"))
        n = len(set(host) & set(tinymaix))
        print("  TinyMaix bit-identical to host: %d/%d;  top-1 agreement %d/%d"
              % (tm_exact, n, tm_top1, n))

    if probe_errors:
        print()
        print("== PROBE-ERROR lines -- these are bugs in tflm_probe.py, not "
              "findings about the board ==")
        for line in probe_errors:
            print("  " + line)
        failures.append("%d PROBE-ERROR line(s)" % len(probe_errors))

    print()
    print("== verdict ==")
    print("bit-identical : %d / %d" % (exact, len(set(host) & set(tflm))))
    print("worst count   : %d" % worst)
    if failures:
        print("\nFAIL")
        for line in failures:
            print("  - " + line)
        return 2
    print("\nPASS -- the board computes what the host computes, exactly.")
    print("The operating point measured on the host under the *reference*")
    print("kernels therefore transfers to the board unchanged. See")
    print("docs/tflm-usermod.md, 'What the morning confirms'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
