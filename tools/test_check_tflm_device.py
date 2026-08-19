#!/usr/bin/env python3
"""Watch `tools/check_tflm_device.py` fail. Runs on the host, needs no board.

    ./tools/test_check_tflm_device.py

`check_tflm_device.py` is the thing that decides the morning's gating question,
and until this file existed **nobody had seen it say no**. A gate only ever
observed passing is indistinguishable from `return 0`, and this one guards a
claim -- "the board computes what the host computes" -- whose failure mode is a
single count out of 5,280. (fw-16mb asked for this; they were right that it was
the half hour worth spending.)

Each case corrupts a known-good capture in one specific way and asserts the
checker rejects it. The corruptions are chosen to be the ones that would
actually happen:

  * **one count off by one** -- +1/256 is exactly the size of the disagreement
    TinyMaix hid behind for a whole session. If the checker tolerates this it
    tolerates the entire defect TFLM exists to remove.
  * **a dropped case** -- a truncated capture, a serial hiccup, a `mpremote`
    interrupted mid-run. Silent under any check that only compares what is
    present against what is present.
  * **a PROBE-ERROR line** -- a bug in the probe, which must not be reported as
    a finding about the board.
  * **a wrong score count** -- a device and a harness disagreeing about the
    model, which si-model's parser also stops on.
  * **the unmodified capture** -- the control. If this does not pass, the other
    four prove nothing.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
CHECKER = os.path.join(HERE, "check_tflm_device.py")
REFERENCE = os.path.join(REPO, "build", "tflm-cases", "reference.txt")


def capture_from_reference():
    """A synthetic device capture that should pass: the host's own answers,
    relabelled `runtime=tflm`, with the diagnostics a real probe prints."""
    lines = ["some diagnostic chatter the checker must ignore",
             "=" * 20]
    for line in open(REFERENCE):
        line = line.strip()
        if line.startswith("SCORE"):
            lines.append(line.replace("runtime=host-tflm", "runtime=tflm"))
    lines.append("probe errors: 0")
    return lines


def run_checker(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        handle.write("\n".join(lines) + "\n")
        handle.close()
        result = subprocess.run(
            [sys.executable, CHECKER, handle.name, "--reference", REFERENCE],
            capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr
    finally:
        os.unlink(handle.name)


def bump_one_count(lines):
    """+1 on a single integer of a single case. The smallest possible lie."""
    out = []
    done = False
    for line in lines:
        if not done and line.startswith("SCORE") and "name=mother_01.wav" in line:
            head, _, q = line.partition("q=")
            values = [int(v) for v in q.split(",")]
            values[0] = (values[0] + 1) % 256
            line = head + "q=" + ",".join(str(v) for v in values)
            done = True
        out.append(line)
    assert done, "the fixture no longer contains mother_01.wav"
    return out


def main():
    if not os.path.exists(REFERENCE):
        print("no reference at %s -- run ./tools/make_tflm_cases.py first"
              % REFERENCE)
        return 1

    good = capture_from_reference()
    failures = []

    def expect(name, lines, want_pass, must_mention=None):
        code, output = run_checker(lines)
        passed = code == 0
        ok = passed == want_pass
        if ok and must_mention and must_mention not in output:
            ok = False
            note = "rejected, but never mentioned %r" % must_mention
        else:
            note = "exit %d" % code
        print("  %-34s %-4s  %s" % (name, "ok" if ok else "FAIL", note))
        if not ok:
            failures.append(name)

    print("control:")
    expect("unmodified capture passes", good, True)

    print("corruptions the checker must reject:")
    expect("one count off by one", bump_one_count(good), False, "mother_01.wav")
    expect("a case dropped", [l for l in good
                              if "name=problem_01.wav" not in l], False,
           "missing")
    expect("a PROBE-ERROR line",
           good + ["PROBE-ERROR name=wife_01.wav ValueError: nonsense"], False,
           "PROBE-ERROR")

    # A short q= is a structural disagreement about the model, and the checker
    # exits on it rather than scoring 21 of 22 integers.
    short = []
    trimmed = False
    for line in good:
        if not trimmed and line.startswith("SCORE"):
            head, _, q = line.partition("q=")
            line = head + "q=" + ",".join(q.split(",")[:21])
            trimmed = True
        short.append(line)
    expect("a q= with 21 scores", short, False, "expected 22")

    print()
    if failures:
        print("FAIL -- the checker did not reject: " + ", ".join(failures))
        return 1
    print("PASS -- the checker rejects every corruption and accepts the control")
    return 0


if __name__ == "__main__":
    sys.exit(main())
