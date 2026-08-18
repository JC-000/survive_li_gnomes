#!/usr/bin/env python3
"""Record the enrolment vocabulary through the board's own microphone.

Runs on the host. Prompts word by word, captures each utterance through the
ES8311 via tools/pull_recording.py, and writes labelled WAVs plus a manifest.

    uvx --from pyserial python tools/enrol.py recordings/
    uvx --from pyserial python tools/enrol.py recordings/ --reps 5
    PORT=/dev/cu.usbmodem1101 uvx --from pyserial python tools/enrol.py recordings/

Labels are correct by construction: the tool says which word to speak and names
the file after it. Nothing is transcribed, so there is no recogniser in the loop
to be wrong -- which matters, because Whisper on isolated single words is
unreliable in exactly this regime (a 0.5 s word padded to a 30 s input window is
the classic hallucination trigger).

Templates must come from THIS microphone, not the Mac's. The recogniser is
speaker-dependent DTW over MFCCs with one template per word, which has nothing
to absorb a channel mismatch: a different mic is a different frequency response,
noise floor and gain, and while cepstral mean normalisation is the right tool
for the convolutional part of that, the literature is clear that it degrades on
short utterances -- too little data to estimate the mean, and over a single word
the mean largely *is* the phonetic content. Hence recording through the board.

Nothing is written to the board's filesystem: audio is streamed out of RAM. See
the enrolment note in CLAUDE.md.

This produces audio and labels only. Template computation belongs to the MFCC
pipeline, which reads the manifest this writes.
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pull_recording as pull
import vocab

# Derived from src/vocab.py rather than written out, because a second copy of
# the vocabulary drifts and the drift is expensive in a way nothing reports: the
# list here decides what a person is asked to say into a microphone, five takes
# at a time. This was a hand-written list until it was checked, and by then it
# had gained NO -- a class vocab.py no longer has -- so an enrolment run would
# have spent five takes recording a word the device can never return, and
# produced a template for a label nothing matches.
#
# FORMS, not LABELS: enrolment records spoken words, and SAD/SICK and WANT/NEED
# are separate things to say even though the engine treats each pair as one
# class. tools/test_record_stream.py pins the two lists together.
VOCABULARY = [form.upper() for form in vocab.FORMS]

DEFAULT_REPS = 5
DEFAULT_SECONDS = 2.0  # generous for one word; halves the transfer vs 3 s
MANIFEST = "manifest.json"


def manifest_path(outdir):
    return os.path.join(outdir, MANIFEST)


def load_manifest(outdir):
    try:
        with open(manifest_path(outdir)) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {"entries": []}


def save_manifest(outdir, doc):
    with open(manifest_path(outdir), "w") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write("\n")


def already_have(doc, outdir, word, rep):
    """A (word, rep) counts as done only if its file is still on disk."""
    for entry in doc["entries"]:
        if entry["label"] == word and entry["rep"] == rep:
            return os.path.exists(os.path.join(outdir, entry["file"]))
    return False


def plan(words, reps, grouped):
    """Order of capture.

    Round-robin by default -- all words once, then all words again. Recording
    five takes of one word back to back gets five copies of the same delivery;
    spreading repetitions across passes varies level and prosody, which is what
    you want a template set to average over.
    """
    if grouped:
        return [(w, r) for w in words for r in range(1, reps + 1)]
    return [(w, r) for r in range(1, reps + 1) for w in words]


def record_one(port, outdir, word, rep, args):
    """Capture one utterance. Returns an entry dict, or None if skipped."""
    name = "%s_%02d.wav" % (word.lower(), rep)
    dest = os.path.join(outdir, name)

    while True:
        answer = input("    Enter to record, (s)kip, (q)uit: ").strip().lower()
        if answer.startswith("q"):
            raise KeyboardInterrupt
        if answer.startswith("s"):
            return None

        print("    recording %.1f s -- speak now" % args.seconds)
        try:
            rate, pcm, elapsed = pull.capture(
                port, seconds=args.seconds, rate=args.rate, timeout=args.timeout
            )
        except pull.CaptureError as exc:
            print("    capture failed: %s" % exc)
            if exc.preamble:
                text = exc.preamble.decode("utf-8", "replace").strip()
                if text:
                    print("    device said: %s" % text.splitlines()[-1])
            continue

        st = pull.stats(pcm)
        fatal, advisory = pull.problems(st)
        kbps = len(pcm) / 1024.0 / elapsed if elapsed > 0 else float("inf")

        # Written before the verdict, so a rejected take can still be listened
        # to. Hearing what arrived is usually how you find out why it failed.
        pull.write_wav(dest, rate, pcm)
        print(
            "    rms %.0f  peak %d  dc %.0f  %.2f s  %.0f KB/s"
            % (st["rms"], st["peak"], st["mean"], elapsed, kbps)
        )
        for note in advisory:
            print("    note: %s" % note)

        if fatal:
            for note in fatal:
                print("    REJECTED: %s" % note)
            os.remove(dest)
            continue

        return {
            "file": name,
            "label": word,
            "rep": rep,
            "rate": rate,
            "samples": st["samples"],
            "seconds": round(st["samples"] / float(rate), 4),
            "rms": round(st["rms"], 1),
            "peak": st["peak"],
            "dc": round(st["mean"], 1),
            "clipped": st["clipped"],
            "transfer_kbps": round(kbps, 1),
            "captured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", help="directory for the WAVs and manifest.json")
    ap.add_argument("--port", default=pull.DEFAULT_PORT)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--rate", type=int, default=pull.DEFAULT_RATE)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--words",
        help="comma-separated subset of the vocabulary, for re-recording a few",
    )
    ap.add_argument(
        "--allow-any",
        action="store_true",
        help="permit words outside the vocabulary, for recording negatives",
    )
    ap.add_argument(
        "--grouped",
        action="store_true",
        help="all repetitions of a word together, instead of round-robin passes",
    )
    args = ap.parse_args()

    words = [w.strip().upper() for w in args.words.split(",")] if args.words else list(VOCABULARY)
    unknown = [w for w in words if w not in VOCABULARY]
    if unknown and not args.allow_any:
        # On by default, and it has already earned its place: this list is
        # derived from vocab.FORMS precisely because an earlier hand-written
        # copy drifted and would have had someone record a retired word.
        sys.exit("not in the vocabulary: %s\n"
                 "(pass --allow-any if these are deliberate negatives)"
                 % ", ".join(unknown))
    if unknown:
        # Negatives are what make a real-speaker *precision* figure possible.
        # Without them the only measurable false fire is one keyword mistaken
        # for another, which is the rarer failure -- the one that matters is the
        # ball answering confidently when the user said something ordinary.
        # Record them somewhere they cannot be mistaken for template material.
        print("recording %d word(s) outside the vocabulary as negatives: %s"
              % (len(unknown), ", ".join(unknown)))

    os.makedirs(args.outdir, exist_ok=True)
    doc = load_manifest(args.outdir)
    doc.update(
        {
            "vocabulary": VOCABULARY,
            "rate": args.rate,
            "seconds": args.seconds,
            "source": "board ES8311 microphone",
        }
    )

    todo = [(w, r) for w, r in plan(words, args.reps, args.grouped)
            if not already_have(doc, args.outdir, w, r)]
    done = len(words) * args.reps - len(todo)
    if done:
        print("resuming: %d of %d already recorded" % (done, len(words) * args.reps))
    if not todo:
        print("nothing to do. Delete files or raise --reps to record more.")
        return 0

    print("%d to record at %d Hz into %s\n" % (len(todo), args.rate, args.outdir))

    try:
        port = pull.open_port(args.port, timeout=0.5)
    except pull.CaptureError as exc:
        print(exc, file=sys.stderr)
        return 1

    captured = 0
    # One open port for the whole session: reopening per capture costs a
    # re-enumeration wait for no benefit.
    with port:
        try:
            for index, (word, rep) in enumerate(todo, 1):
                print("[%d/%d] %s  (take %d)" % (index, len(todo), word, rep))
                entry = record_one(port, args.outdir, word, rep, args)
                if entry is None:
                    print("    skipped")
                    continue
                doc["entries"] = [
                    e for e in doc["entries"]
                    if not (e["label"] == word and e["rep"] == rep)
                ] + [entry]
                # Saved after every take: a session interrupted three quarters
                # of the way through should not lose three quarters of an hour.
                save_manifest(args.outdir, doc)
                captured += 1
        except KeyboardInterrupt:
            print("\nstopped.")

    save_manifest(args.outdir, doc)
    print(
        "\n%d captured this session, %d in %s"
        % (captured, len(doc["entries"]), manifest_path(args.outdir))
    )
    missing = [
        "%s take %d" % (w, r)
        for w, r in plan(words, args.reps, args.grouped)
        if not already_have(doc, args.outdir, w, r)
    ]
    if missing:
        print("still missing %d: %s%s" % (
            len(missing), ", ".join(missing[:6]), " ..." if len(missing) > 6 else ""))
        print("re-run the same command to pick up where this left off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
