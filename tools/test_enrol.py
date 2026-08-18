#!/usr/bin/env python3
"""Checks the enrolment session logic with no board attached.

    python3 tools/test_enrol.py

`tools/enrol.py` drives a real recording session: 24 words, 5 takes each, a
person saying them one at a time. The paths that matter are the ones someone
leans on at the worst moment -- halfway through, having already spent twenty
minutes on it. So this covers ordering, resume, skip, and what happens when a
capture comes back dead.

The transport underneath is not retested here; `tools/test_record_stream.py`
does that from both ends. `pull.capture` is replaced with a stub so this stays
about the session and not the wire.

Two behaviours are worth stating because they are easy to break by tidying:

- **A take that fails its signal check is deleted and retried, never kept.** The
  point of enrolling on the board's own microphone is defeated entirely by a
  directory that silently contains five seconds of a dead codec.
- **A manifest entry counts only if its file still exists.** Deleting a bad WAV
  by hand is how someone will fix a bad take, and the re-run has to notice.

This file must not hold a second copy of the vocabulary. An earlier version of
it asserted `len(VOCABULARY) == 25`, which did not detect drift from
`src/vocab.py` -- it enforced it, and failed against the change that corrected
it. See `test_vocabulary` and the gotchas memory.
"""

import builtins
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import types
from array import array
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FAILURES = []

# enrol imports pull_recording, which imports pyserial. Nothing here opens a
# port, so a stub keeps this runnable under the system python3 with no venv.
_serial = types.ModuleType("serial")


class _SerialException(Exception):
    pass


_serial.SerialException = _SerialException
_serial.Serial = object
sys.modules["serial"] = _serial

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import pull_recording as pull  # noqa: E402
import enrol  # noqa: E402
import vocab  # noqa: E402


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# --- fakes -----------------------------------------------------------------

def known_waveform(count, amplitude=9000):
    """Something with structure, loud enough to pass the signal check."""
    out = array("h", bytearray(2 * count))
    for i in range(count):
        out[i] = int(amplitude * ((i * 37 % 200) - 100) / 100.0)
    return out.tobytes()


DEAD = struct.pack("<%dh" % 800, *([600] * 800))  # a constant: codec never got BCLK


class FakePort:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Session:
    """Drives enrol.main() with scripted keypresses and a scripted capture."""

    def __init__(self, dead_takes=()):
        self.dead_takes = set(dead_takes)  # 1-based capture numbers to fail
        self.captures = 0

    def capture(self, port, seconds=2.0, rate=16000, timeout=30.0):
        self.captures += 1
        if self.captures in self.dead_takes:
            return rate, DEAD, 0.08
        return rate, known_waveform(int(rate * seconds)), 0.08

    def run(self, argv, keys):
        real_capture, real_open = pull.capture, pull.open_port
        real_input = builtins.input
        pull.capture = self.capture
        pull.open_port = lambda port, timeout: FakePort()
        answers = iter(keys)
        # enrol calls the builtin directly, so that is what has to be replaced.
        builtins.input = lambda prompt="": next(answers)
        sys.argv = ["enrol.py"] + argv
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                rc = enrol.main()
        except SystemExit as exc:
            rc = exc.code
        except StopIteration:
            rc = "ran out of scripted answers"
        finally:
            pull.capture, pull.open_port = real_capture, real_open
            builtins.input = real_input
        return rc, out.getvalue()


def read_manifest(outdir):
    with open(os.path.join(outdir, enrol.MANIFEST)) as handle:
        return json.load(handle)


# --- tests -----------------------------------------------------------------

def test_vocabulary():
    """Derived from src/vocab.py, so it cannot drift from what the device knows.

    FORMS and not LABELS: enrolment records what a person *says*, and SAD and
    SICK are two things to say even though the engine collapses them to one
    class. A word prompted here that the device cannot return is five takes of
    somebody's time spent on a template nothing will ever match.
    """
    print("vocabulary")
    check("derives from vocab.FORMS",
          enrol.VOCABULARY == [f.upper() for f in vocab.FORMS],
          "%r" % (enrol.VOCABULARY,))
    check("no duplicate forms",
          len(set(enrol.VOCABULARY)) == len(enrol.VOCABULARY))
    check("every form resolves to a label",
          all(vocab.label_of(w.lower()) for w in enrol.VOCABULARY))


def test_ordering():
    """Round-robin by default.

    Five takes of one word back to back are five copies of one delivery, at one
    level, with one prosody. Spreading repetitions across passes is what gives a
    template set something to average over.
    """
    print("ordering")
    check("round-robin spreads takes across passes",
          enrol.plan(["A", "B", "C"], 2, grouped=False)
          == [("A", 1), ("B", 1), ("C", 1), ("A", 2), ("B", 2), ("C", 2)])
    check("--grouped keeps a word's takes together",
          enrol.plan(["A", "B"], 2, grouped=True)
          == [("A", 1), ("A", 2), ("B", 1), ("B", 2)])
    check("plan covers every word and take exactly once",
          sorted(enrol.plan(["A", "B"], 3, False)) == sorted(enrol.plan(["A", "B"], 3, True)))


def test_session(outdir):
    """One pass: good takes, one dead capture, one skip."""
    print("session")
    # Capture 3 comes back as a constant -- the dead-codec case -- so WIFE take 1
    # is rejected and retried. FATHER take 2 is skipped by the operator.
    session = Session(dead_takes=[3])
    rc, out = session.run(
        ["--words", "MOTHER,FATHER,WIFE", "--reps", "2", outdir],
        ["", "", "", "", "", "s", ""],
    )
    check("session completes", rc == 0, "rc=%r" % (rc,))
    check("dead capture was rejected", "REJECTED" in out)
    check("dead capture was retried, not abandoned",
          any(e["label"] == "WIFE" and e["rep"] == 1 for e in read_manifest(outdir)["entries"]))

    doc = read_manifest(outdir)
    wavs = sorted(f for f in os.listdir(outdir) if f.endswith(".wav"))
    check("no WAV left behind for the rejected take",
          len(wavs) == len(doc["entries"]), "%d wavs, %d entries" % (len(wavs), len(doc["entries"])))
    check("every manifest entry has its file",
          all(os.path.exists(os.path.join(outdir, e["file"])) for e in doc["entries"]))
    check("skipped take is absent",
          not any(e["label"] == "FATHER" and e["rep"] == 2 for e in doc["entries"]))
    check("skip is reported so it is not silently lost", "still missing" in out)

    entry = next(e for e in doc["entries"] if e["label"] == "MOTHER")
    check("entry carries the label and rate the DSP side needs",
          entry["label"] == "MOTHER" and entry["rate"] == pull.DEFAULT_RATE,
          "%r" % (entry,))
    check("entry records signal stats, not just a path",
          {"rms", "peak", "samples"} <= set(entry))


def test_resume(outdir):
    """Re-running picks up where it stopped, and re-records a deleted file."""
    print("resume")
    before = read_manifest(outdir)
    session = Session()
    rc, out = session.run(["--words", "MOTHER,FATHER,WIFE", "--reps", "2", outdir], [""])
    check("resuming is announced", "resuming" in out, out[:120])
    check("only the outstanding take was recorded", session.captures == 1,
          "%d captures" % session.captures)
    check("nothing already recorded was re-prompted",
          len(read_manifest(outdir)["entries"]) == len(before["entries"]) + 1)

    victim = read_manifest(outdir)["entries"][0]
    os.remove(os.path.join(outdir, victim["file"]))
    session = Session()
    rc, out = session.run(
        ["--words", victim["label"], "--reps", str(victim["rep"]), outdir], [""]
    )
    check("a manifest entry whose file was deleted is re-recorded",
          os.path.exists(os.path.join(outdir, victim["file"])),
          victim["file"])

    session = Session()
    rc, out = session.run(["--words", "MOTHER,FATHER,WIFE", "--reps", "2", outdir], [])
    check("a complete set asks for nothing", "nothing to do" in out, out[:120])


def test_bad_input(outdir):
    print("bad input")
    session = Session()
    rc, _ = session.run(["--words", "BANANA", outdir], [])
    check("a word outside the vocabulary is refused before recording",
          rc and "not in the vocabulary" in str(rc), "rc=%r" % (rc,))
    check("nothing was captured for it", session.captures == 0)


def test_pull_diagnostics():
    """The capture diagnostics in `tools/pull_recording.py`.

    These belong to that module rather than to the session, and they are here
    because `enrol.py` is the only thing that acts on them and nothing else
    covers them at all -- `tools/test_record_stream.py` tests the wire, not the
    verdict on what came off it. A diagnostic only fires when something is
    already wrong, which is exactly when nobody wants to discover it was never
    exercised.

    The dead-codec and silent-signal cases are the fatal ones and are also
    reached through `test_session`; they are repeated here directly because
    that route only proves a take was rejected, not which check rejected it.
    """
    print("capture diagnostics")

    def stats_of(samples):
        return pull.stats(struct.pack("<%dh" % len(samples), *samples))

    fatal, advisory = pull.problems(stats_of([1234] * 500))
    check("a constant buffer is fatal", fatal and "constant" in fatal[0])
    check("and it names the wiring to check", fatal and "BCLK" in fatal[0])

    quiet = stats_of([int(30 * math.sin(i * 0.05)) for i in range(500)])
    fatal, advisory = pull.problems(quiet)
    check("a near-silent buffer is fatal", bool(fatal))
    check("but is not misreported as a dead channel", not quiet["flat"])

    fatal, advisory = pull.problems(stats_of([32767 if i % 2 else -20000 for i in range(500)]))
    check("clipping is advisory, not fatal",
          not fatal and any("rail" in a for a in advisory), "%r" % (advisory,))

    fatal, advisory = pull.problems(stats_of([9000 + int(4000 * math.sin(i * 0.05))
                                              for i in range(500)]))
    check("a large DC offset is advisory",
          any("DC offset" in a for a in advisory), "%r" % (advisory,))

    fatal, advisory = pull.problems(stats_of([int(9000 * math.sin(i * 0.05))
                                              for i in range(500)]))
    check("a clean loud capture raises nothing at all",
          not fatal and not advisory, "%r %r" % (fatal, advisory))

    # The header guard is the other diagnostic with no coverage. A desynced
    # stream yields a nonsense length, and without this the reader would sit in
    # `exactly()` until the timeout instead of failing immediately.
    class _Replay:
        def __init__(self, data):
            self.data, self.pos = data, 0

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        @property
        def in_waiting(self):
            return len(self.data) - self.pos

        def read(self, count):
            chunk = self.data[self.pos:self.pos + count]
            self.pos += len(chunk)
            return chunk

    port = _Replay(pull.MAGIC + struct.pack("<II", 16000, 99999999))
    try:
        pull.capture(port, seconds=0.1, rate=16000, timeout=0.5)
        check("an implausible sample count is refused", False, "no error raised")
    except pull.CaptureError as exc:
        check("an implausible sample count is refused", "implausible" in str(exc), str(exc))


def main():
    test_vocabulary()
    test_ordering()
    test_pull_diagnostics()
    outdir = tempfile.mkdtemp(prefix="enrol-test-")
    try:
        test_session(outdir)
        test_resume(outdir)
        test_bad_input(outdir)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
