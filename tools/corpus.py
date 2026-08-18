#!/usr/bin/env python3
"""Load the speaker-independence datasets, and keep the human takes out of training.

Two corpora, one loader, and a rule that is the whole point of the module:

    corpus-tts/   37 synthetic voices, split by voice     tools/train_corpus.py
    takes/        one real person through the board       tools/enrol.py

    python3 tools/corpus.py corpus-tts/          # what is in it
    python3 tools/corpus.py corpus-tts/ --check  # split integrity
    python3 tools/corpus.py --takes takes/       # the human held-out set

## The rule

`takes/` is real human speech recorded through the board's own microphone. It
is the only recording in this project made under the condition the device
actually operates in -- this speaker, this microphone, this room -- and it is
the only honest answer to "would a speaker-independent model work". Its value
is entirely in never having been trained on.

So it is loaded under the split name `human`, which `split_for_training()`
refuses to return, and `load()` will not read it at all: a caller has to name
`load_takes()` deliberately. There is no flag that mixes them. That is more
friction than a boolean, and it is the right amount, because the failure it
prevents is silent -- a model trained on its own test set reports a number
that looks like success.

## Degrading when the takes do not exist yet

They may not. The enrolment session may not have happened, may be half done,
or may have been recorded somewhere else. `load_takes()` returns an empty list
and sets `missing` on the result rather than raising, because everything else
here -- building the corpus, training, measuring held-out synthetic voices --
is work that can proceed without them. What it will not do is quietly report
zero human utterances as though that were a result: `describe()` says so, and
`--check` exits non-zero.
"""

import argparse
import array
import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import vocab  # noqa: E402

SYNTHETIC_SPLITS = ("train", "val", "test")
HUMAN_SPLIT = "human"
DEFAULT_TAKES = "takes"

# docs/speech.md fixes the feature contract at 16 kHz mono int16. A take at any
# other rate is not a take that needs resampling, it is a capture that went out
# at the wrong codec setting, and per CLAUDE.md that is exactly the class of
# fault that produces no exception anywhere.
EXPECTED_RATE = 16000


def read_wav(path):
    """int16 samples. A real parser, because `say` puts its PCM at offset 4096.

    `say` pads the header with a 4044-byte `FLLR` chunk. Anything that skips a
    fixed 44 bytes reads that padding as audio; `wave` walks the chunk list and
    lands in the right place. Board captures written by `tools/pull_recording.py`
    have an ordinary 44-byte header, so one reader covers both only if it is a
    parser.
    """
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("%s: expected mono int16, got %d ch %d bytes"
                             % (path, w.getnchannels(), w.getsampwidth()))
        if w.getframerate() != EXPECTED_RATE:
            raise ValueError("%s: %d Hz, expected %d -- see docs/speech.md"
                             % (path, w.getframerate(), EXPECTED_RATE))
        return array.array("h", w.readframes(w.getnframes()))


def load(root, split=None, category=None):
    """Records from a synthetic corpus built by `tools/train_corpus.py`.

    `split` is one of `train` / `val` / `test`, or None for all three.
    `category` is `word` / `unknown` / `variant`, or None for all.

    Each record is the manifest entry with `path` added. Audio is not read --
    a full corpus is 1.8 GB and a caller wants to decide when that happens.
    """
    manifest = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest):
        raise IOError("no manifest at %s -- run tools/train_corpus.py" % manifest)
    with open(manifest) as h:
        doc = json.load(h)

    if split is not None and split not in SYNTHETIC_SPLITS:
        # Naming `human` here is the mistake this module exists to catch, so it
        # gets its own message rather than an empty list.
        if split == HUMAN_SPLIT:
            raise ValueError("the human takes are not in the synthetic corpus; "
                             "call load_takes() and mean it")
        raise ValueError("unknown split %r" % split)

    out = []
    for rec in doc["entries"]:
        if split is not None and rec["split"] != split:
            continue
        if category is not None and rec["category"] != category:
            continue
        rec = dict(rec)
        rec["path"] = os.path.join(root, rec["file"])
        out.append(rec)
    return out


def split_for_training(root):
    """`(train, val)` only. `test` and `human` are not obtainable from here."""
    return load(root, "train"), load(root, "val")


def load_takes(root=DEFAULT_TAKES, require=False):
    """The real human held-out set, or an empty list if it is not there yet.

    Returns `(records, note)`. `note` is None when everything was found and a
    human-readable string when it was not, so a caller can print the reason
    rather than reporting an empty test set as a passing one.

    Labels come from `tools/enrol.py`, which names files after the word it
    asked for, so they are correct by construction -- nothing is transcribed
    and there is no recogniser in the loop to be wrong. They are upper-case
    spoken *forms*, so SICK maps to the SAD class here exactly as it does on
    the device; a form outside the vocabulary (recorded with `enrol.py
    --allow-any`) becomes an `unknown` record rather than being dropped,
    because human negatives are worth more than synthetic ones.
    """
    manifest = os.path.join(root, "manifest.json")
    if not os.path.isdir(root):
        note = "no %s directory -- the enrolment session has not run" % root
    elif not os.path.exists(manifest):
        note = "%s exists but has no manifest.json" % root
    else:
        note = None
    if note:
        if require:
            raise IOError(note)
        return [], note

    with open(manifest) as h:
        doc = json.load(h)

    out = []
    absent = 0
    for entry in doc.get("entries", []):
        path = os.path.join(root, entry["file"])
        if not os.path.exists(path):
            absent += 1
            continue
        form = entry["label"].lower()
        label = vocab.label_of(form)
        rec = dict(entry)
        rec.update({
            "path": path,
            "split": HUMAN_SPLIT,
            "category": "word" if label else "unknown",
            "label": label,
            "text": form,
            "voice": doc.get("speaker", "human"),
            "family": HUMAN_SPLIT,
            "locale": doc.get("locale", "unknown"),
            "source": doc.get("source", "board ES8311 microphone"),
        })
        out.append(rec)

    note = None
    if not out:
        note = "%s has a manifest but no readable takes" % root
    elif absent:
        note = "%d take(s) listed in the manifest are missing from disk" % absent
    return out, note


def check_split(root):
    """Every way the split could be leaking. Returns a list of complaints.

    The one that matters is a voice appearing under two splits. `train_corpus`
    takes the split from the roster and writes it into the path, so this can
    only happen if the roster was rebuilt between two partial builds -- which
    is exactly what a resumable builder makes easy, and which nothing else
    would notice.
    """
    problems = []
    records = load(root)
    splits = {}
    families = {}
    for rec in records:
        splits.setdefault(rec["voice"], set()).add(rec["split"])
        families.setdefault(rec["family"], set()).add(rec["split"])

    for voice, seen in sorted(splits.items()):
        if len(seen) > 1:
            problems.append("voice %r appears in %s"
                            % (voice, " and ".join(sorted(seen))))
    for family, seen in sorted(families.items()):
        if len(seen) > 1:
            problems.append("family %r spans %s -- one synthesiser on both "
                            "sides of the split" % (family, " and ".join(sorted(seen))))

    # Held-out voices are the only evidence for generalisation while `takes/`
    # is empty, so a test split of formant synthesisers would predict nothing
    # about a person. `tools/say_voices.py` holds out `natural` voices only.
    for split in ("val", "test"):
        tiers = set(r["tier"] for r in records
                    if r["split"] == split and r.get("tier"))
        wrong = sorted(tiers - {"natural"})
        if wrong:
            problems.append("split %s holds out %s voices; only `natural` "
                            "voices predict a human speaker"
                            % (split, ", ".join(wrong)))

    for rec in records:
        if rec["split"] == HUMAN_SPLIT:
            problems.append("%s is labelled human inside the synthetic corpus"
                            % rec["file"])
            break

    labelled = set(r["label"] for r in records
                   if r["category"] == "word" and r["label"])
    for split in SYNTHETIC_SPLITS:
        here = set(r["label"] for r in records
                   if r["split"] == split and r["category"] == "word")
        gaps = labelled - here
        if gaps:
            problems.append("split %s is missing %d class(es): %s"
                            % (split, len(gaps), ", ".join(sorted(gaps))))
    return problems


def describe(root, takes=DEFAULT_TAKES):
    records = load(root)
    print("%s: %d utterances" % (root, len(records)))
    print("\n%-6s %8s %8s %8s %8s %8s"
          % ("split", "voices", "families", "word", "unknown", "variant"))
    for split in SYNTHETIC_SPLITS:
        rows = [r for r in records if r["split"] == split]
        print("%-6s %8d %8d %8d %8d %8d"
              % (split,
                 len(set(r["voice"] for r in rows)),
                 len(set(r["family"] for r in rows)),
                 sum(1 for r in rows if r["category"] == "word"),
                 sum(1 for r in rows if r["category"] == "unknown"),
                 sum(1 for r in rows if r["category"] == "variant")))

    human, note = load_takes(takes)
    print("\n%s (%s): %d utterances" % (takes, HUMAN_SPLIT, len(human)))
    if note:
        print("  %s" % note)
        print("  the synthetic held-out voices measure whether a model "
              "memorised a synthesiser;")
        print("  only these measure whether it recognises a person.")
    else:
        words = [r for r in human if r["category"] == "word"]
        print("  %d in-vocabulary over %d classes, %d out-of-vocabulary"
              % (len(words), len(set(r["label"] for r in words)),
                 len(human) - len(words)))
    return human, note


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="corpus-tts",
                    help="synthetic corpus directory")
    ap.add_argument("--takes", default=DEFAULT_TAKES)
    ap.add_argument("--check", action="store_true",
                    help="verify the split and exit non-zero if it leaks")
    args = ap.parse_args(argv)

    try:
        human, note = describe(args.root, args.takes)
    except IOError as exc:
        # An unbuilt corpus is an ordinary state -- the build takes two hours
        # and is meant to be run separately -- so it gets a sentence and a
        # non-zero exit, not a traceback.
        print(exc, file=sys.stderr)
        human, note = load_takes(args.takes)
        print("%s (%s): %d utterances%s"
              % (args.takes, HUMAN_SPLIT, len(human),
                 "" if not note else " -- %s" % note), file=sys.stderr)
        return 1
    if not args.check:
        return 0

    problems = check_split(args.root)
    print()
    for p in problems:
        print("PROBLEM: %s" % p)
    if not human:
        print("PROBLEM: no human takes, so speaker independence is untested")
    if problems or not human:
        return 1
    print("split is clean: no voice or family spans two splits, "
          "every class present in all three, %d human utterances held out"
          % len(human))
    return 0


if __name__ == "__main__":
    sys.exit(main())
