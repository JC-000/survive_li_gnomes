#!/usr/bin/env python3
"""Checks the multi-voice corpus tooling without synthesising a corpus.

    python3 tools/test_corpus.py

Everything here is about the two properties that are worth more than the audio:

**The split holds.** A voice, or a whole synthesiser family, appearing on both
sides is the exact failure the speaker-independence experiment exists to
detect, and it is invisible in the result -- a leaked split reports a good
number, not an error. `check_split` is the guard and this exercises it against
a corpus deliberately built wrong.

**The human takes never reach training.** `takes/` is the only recording made
under the condition the device operates in, and its value is entirely in never
having been trained on. So `load` refuses to serve it and `split_for_training`
cannot return it, and both refusals are tested rather than assumed.

The third thing covered is that the tooling **degrades** when `takes/` is not
there yet, because for most of this experiment's life it will not be: the
enrolment session is a person sitting down with the board for an hour. Missing
takes must produce an empty set and a stated reason, never an exception and
never a silent zero that reads as a passing test.

No `say`, no board, no corpus: every fixture here is written by hand.
"""

import json
import os
import shutil
import sys
import tempfile
import wave
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FAILURES = []

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import corpus  # noqa: E402
import say_corpus as sc  # noqa: E402
import say_voices as sv  # noqa: E402
import train_corpus as tc  # noqa: E402
import vocab  # noqa: E402


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# --- fixtures --------------------------------------------------------------

def write_wav(path, samples, rate=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def tone(count=8000, amplitude=6000):
    out = array("h", bytearray(2 * count))
    for i in range(count):
        out[i] = amplitude if (i // 40) % 2 else -amplitude
    return out


def fake_corpus(root, entries):
    """A manifest and the WAVs it names. `entries` are partial records."""
    full = []
    for i, e in enumerate(entries):
        rec = {"split": "train", "category": "word", "label": "mother",
               "text": "mother", "voice": "V", "family": "F", "locale": "en_US",
               "rate_wpm": 175, "attacks": None, "gain_db": 0, "tilt": 0.0,
               "snr_db": 20, "lead_ms": 400, "noise": "lcg",
               "noise_dbfs": -40.0, "speech_dbfs": -24.0, "samples": 8000,
               "endpoint_ms": 400}
        rec.update(e)
        rec["file"] = rec.get("file", "%s/%s/%s/%03d.wav"
                              % (rec["split"], rec["category"], rec["voice"], i))
        write_wav(os.path.join(root, rec["file"]), tone())
        full.append(rec)
    with open(os.path.join(root, "manifest.json"), "w") as h:
        json.dump({"rate": 16000, "entries": full}, h)
    return full


def fake_takes(root, labels, rate=16000, drop=()):
    os.makedirs(root, exist_ok=True)
    entries = []
    for i, label in enumerate(labels):
        name = "%s_%02d.wav" % (label.lower(), i)
        if name not in drop:
            write_wav(os.path.join(root, name), tone(), rate)
        entries.append({"file": name, "label": label, "rep": i, "rate": rate,
                        "samples": 8000, "seconds": 0.5})
    with open(os.path.join(root, "manifest.json"), "w") as h:
        json.dump({"vocabulary": [f.upper() for f in vocab.FORMS],
                   "rate": rate, "seconds": 2.0,
                   "source": "board ES8311 microphone",
                   "entries": entries}, h)


# --- the vocabulary is not copied here -------------------------------------

def test_derived_from_vocab():
    """Nothing in the corpus tooling may hold its own copy of the word list.

    `tools/test_enrol.py` records why: an earlier version of *that* file
    asserted a word count, which did not detect drift from `src/vocab.py` but
    enforced it, and failed against the commit that fixed it. So these check
    relationships, never totals.
    """
    print("vocabulary")
    unknowns = tc.unknown_words()
    texts = set(sc.slug(t) for t, _, _ in unknowns)
    forms = set(sc.slug(f) for f in vocab.FORMS)
    check("no in-vocabulary form is also an unknown", not (texts & forms),
          sorted(texts & forms))

    varmap = tc.variants_map()
    stale = [text for text, label in varmap.items()
             if label not in vocab.LABELS]
    check("every inflection maps to a live class", not stale, stale)

    # A phrase that *contains* a keyword belongs in the variant category or
    # nowhere: firing SORRY on "sorry about that" is DOCTOR working, and
    # scoring it as a false positive would drag the rejection threshold down
    # to punish correct behaviour. "sorry about that" was in the silent set
    # until this check was written.
    import re
    embedded = []
    for text, cat, _ in unknowns:
        if cat == "variant":
            continue
        words = set(re.findall(r"[a-z']+", text.lower()))
        hit = [f for f in vocab.FORMS if f in words]
        if hit:
            embedded.append((text, hit))
    check("no silent-set text contains a keyword", not embedded, embedded)

    # WANTED and NEEDED are in say_corpus.OOV_VARIANTS and their class was
    # retired. They must come through as ordinary unknowns that have to stay
    # silent, not as variants whose firing is excused.
    cats = dict((sc.slug(t), c) for t, c, _ in unknowns)
    check("a variant of a retired class is not excused",
          cats.get("wanted") == "unknown", cats.get("wanted"))

    for word in sc.RETIRED:
        attacks = dict((sc.slug(t), a) for t, _, a in unknowns)
        check("retired word %r is in the unknown set" % word,
              attacks.get(sc.slug(word)) == "retired", attacks.get(sc.slug(word)))

    # Keyed by spoken *form*, not by class. SICK's rhymes -- six, thick, quick
    # -- threaten the SICK templates specifically, even though a fire there is
    # reported as the SAD class it shares with SAD. Recording which form is
    # under attack is the finer-grained fact and the class is recoverable from
    # it; the other direction is not.
    attacked = set(tc.NEAR_MISS)
    check("every near-miss group names a real spoken form",
          attacked <= set(vocab.FORMS), sorted(attacked - set(vocab.FORMS)))
    check("every spoken form has near-misses to defend against",
          set(vocab.FORMS) <= attacked, sorted(set(vocab.FORMS) - attacked))


def test_plan_shape():
    print("plan")
    unknowns = tc.unknown_words()
    plan = tc.utterance_plan("Samantha", vocab.FORMS, unknowns, tc.RATES, 3)
    words = [p for p in plan if p[1] == "word"]
    check("every form gets every rate",
          len(words) == len(vocab.FORMS) * len(tc.RATES), len(words))

    retired = [p for p in plan if p[4] and p[3] == "retired"] or \
              [p for p in plan if p[3] == "retired"]
    per = len(retired) / float(len(sc.RETIRED))
    check("retired words get the full rate set", per == len(tc.RATES), per)

    # The rotation is what keeps the corpus covering all five rates while no
    # single voice pays for all five. Two voices must disagree.
    other = tc.utterance_plan("Daniel", vocab.FORMS, unknowns, tc.RATES, 3)
    a = [p[4] for p in plan if p[3] is None and p[1] != "word"]
    b = [p[4] for p in other if p[3] is None and p[1] != "word"]
    check("the unknown rate rotates between voices", a != b)
    check("the rotation covers every rate", set(a) == set(tc.RATES), sorted(set(a)))


def test_pick_is_uniform():
    """The LCG's low bits collapse at a fixed stride. See `tc.pick_hi`."""
    print("augmentation draws")
    counts = {}
    rng = sc.Rng(999)
    for _ in range(2000):
        for _ in range(4):          # the four per-variant draws
            tc.pick_hi(rng, (0, 1, 2, 3, 4))
        v = tc.pick_hi(rng, tuple(range(6)))
        counts[v] = counts.get(v, 0) + 1
        rng.next()                  # the noise window draw
    check("all six noise files are reachable", len(counts) == 6, sorted(counts))
    low = min(counts.values())
    check("and roughly evenly", low > 2000 / 6 * 0.7, sorted(counts.items()))


# --- the split -------------------------------------------------------------

def test_split_integrity(tmp):
    print("split")
    good = os.path.join(tmp, "good")
    rows = []
    for split in ("train", "val", "test"):
        for label in vocab.LABELS:
            rows.append({"split": split, "voice": "voice-" + split,
                         "family": "fam-" + split, "label": label,
                         "text": label})
    fake_corpus(good, rows)
    check("a clean corpus has no complaints", corpus.check_split(good) == [],
          corpus.check_split(good))

    leaky = os.path.join(tmp, "leaky")
    rows = []
    for split in ("train", "val", "test"):
        for label in vocab.LABELS:
            # One voice, one family, on every side: the failure this exists for.
            rows.append({"split": split, "voice": "Flo", "family": "iOS",
                         "label": label, "text": label})
    fake_corpus(leaky, rows)
    problems = corpus.check_split(leaky)
    check("a voice on both sides is caught",
          any("Flo" in p for p in problems), problems)
    check("a family spanning the split is caught",
          any("iOS" in p for p in problems), problems)

    thin = os.path.join(tmp, "thin")
    rows = [{"split": "train", "voice": "A", "family": "fa",
             "label": l, "text": l} for l in vocab.LABELS]
    rows += [{"split": "val", "voice": "B", "family": "fb",
              "label": vocab.LABELS[0], "text": vocab.LABELS[0]}]
    rows += [{"split": "test", "voice": "C", "family": "fc",
              "label": l, "text": l} for l in vocab.LABELS]
    fake_corpus(thin, rows)
    check("a split missing classes is caught",
          any("missing" in p for p in corpus.check_split(thin)),
          corpus.check_split(thin))


def test_takes_are_never_training(tmp):
    print("human takes")
    root = os.path.join(tmp, "good")
    try:
        corpus.load(root, "human")
        check("load() refuses the human split", False, "no exception")
    except ValueError as exc:
        check("load() refuses the human split", "mean it" in str(exc), str(exc))

    train, val = corpus.split_for_training(root)
    both = train + val
    check("split_for_training yields no human record",
          all(r["split"] in ("train", "val") for r in both))
    check("split_for_training yields no test record",
          not any(r["split"] == "test" for r in both))


def test_takes_missing(tmp):
    print("takes, absent")
    records, note = corpus.load_takes(os.path.join(tmp, "nothing-here"))
    check("absent takes give an empty list", records == [], records)
    check("...and say why", note and "has not run" in note, note)

    empty = os.path.join(tmp, "empty-takes")
    os.makedirs(empty)
    records, note = corpus.load_takes(empty)
    check("a directory with no manifest is reported", note and "manifest" in note,
          note)

    try:
        corpus.load_takes(os.path.join(tmp, "nothing-here"), require=True)
        check("require=True raises", False, "no exception")
    except IOError:
        check("require=True raises", True)


def test_takes_present(tmp):
    print("takes, present")
    root = os.path.join(tmp, "takes")
    fake_takes(root, ["MOTHER", "SICK", "COFFEE"])
    records, note = corpus.load_takes(root)
    check("all takes load", len(records) == 3 and note is None, (len(records), note))

    by = dict((r["text"], r) for r in records)
    check("every take is in the human split",
          all(r["split"] == "human" for r in records))
    check("SICK maps to the SAD class, as the device does",
          by["sick"]["label"] == "sad", by["sick"]["label"])
    check("a word outside the vocabulary becomes an unknown",
          by["coffee"]["category"] == "unknown" and by["coffee"]["label"] is None,
          (by["coffee"]["category"], by["coffee"]["label"]))

    gone = os.path.join(tmp, "takes-partial")
    fake_takes(gone, ["MOTHER", "FATHER"], drop=("father_01.wav",))
    records, note = corpus.load_takes(gone)
    check("a manifest entry with no file is skipped", len(records) == 1, records)
    check("...and reported", note and "missing from disk" in note, note)


def test_rate_is_enforced(tmp):
    """A take at the wrong rate is a codec that ignored the register write.

    Per CLAUDE.md that produces no exception anywhere on the device, so the
    host is the only place it can be caught, and it has to be caught loudly:
    24 kHz audio read as 16 kHz is a corpus that trains on the wrong thing.
    """
    print("sample rate")
    root = os.path.join(tmp, "takes-24k")
    fake_takes(root, ["MOTHER"], rate=24000)
    records, _ = corpus.load_takes(root)
    try:
        corpus.read_wav(records[0]["path"])
        check("24 kHz audio is refused", False, "no exception")
    except ValueError as exc:
        check("24 kHz audio is refused", "24000" in str(exc), str(exc))


def test_voice_identity():
    """The split has leaked three times, each time a different sense of "same".

    All three are here because none of them errors: a leaked split reports a
    better number, not a failure. In order of discovery --

    1. `say -v <name>` renders an *uninstalled* voice in the default voice and
       returns 0, so several names become one speaker. Caught by digest.
    2. `Flo (English (UK))` and `Flo (English (US))` are one voice at two
       accents. Caught by name stem.
    3. `Samantha` and `Samantha (Enhanced)` are one voice at two quality
       tiers -- and they render *differently*, so the digest check passes them.
       Caught by name stem too, which is why that mechanism is the one worth
       guarding.
    """
    print("voice identity")
    check("a quality tier is stripped from the identity",
          sv.name_stem("Samantha (Enhanced)") == "Samantha",
          sv.name_stem("Samantha (Enhanced)"))
    check("a locale variant is stripped from the identity",
          sv.name_stem("Flo (English (UK))") == "Flo")
    check("quality is parsed from the name",
          (sv.quality_of("Allison (Enhanced)"), sv.quality_of("Samantha"))
          == ("Enhanced", "Compact"))

    # A Compact voice superseded by an Enhanced install of the same voice, and
    # a locale pair that must NOT be collapsed.
    voices = [{"name": "Samantha", "locale": "en_US"},
              {"name": "Samantha (Enhanced)", "locale": "en_US"},
              {"name": "Flo (English (UK))", "locale": "en_GB"},
              {"name": "Flo (English (US))", "locale": "en_US"}]
    kept, dropped = sv.prefer_quality(voices)
    names = sorted(v["name"] for v in kept)
    check("the Compact twin is dropped for the Enhanced one",
          dropped == ["Samantha"], dropped)
    check("...and the locale pair is kept",
          names == ["Flo (English (UK))", "Flo (English (US))",
                    "Samantha (Enhanced)"], names)

    try:
        sv.assert_distinct([{"name": "A", "pcm_sha256": "x"},
                            {"name": "B", "pcm_sha256": "x"}])
        check("identical renders raise", False, "no exception")
    except ValueError as exc:
        check("identical renders raise", "falling back" in str(exc), str(exc))
    check("distinct renders do not",
          sv.assert_distinct([{"name": "A", "pcm_sha256": "x"},
                              {"name": "B", "pcm_sha256": "y"}]) == 2)

    try:
        sv.assert_one_split_per_stem(
            [{"name": "Samantha", "split": "train"},
             {"name": "Samantha (Enhanced)", "split": "test"}])
        check("one speaker in two splits raises", False, "no exception")
    except ValueError as exc:
        check("one speaker in two splits raises",
              "two splits" in str(exc), str(exc))
    check("the same speaker in one split is fine",
          sv.assert_one_split_per_stem(
              [{"name": "Samantha", "split": "train"},
               {"name": "Samantha (Enhanced)", "split": "train"}]) is None)


def test_live_roster():
    """Whatever roster is on disk must satisfy the invariants. Skipped if absent."""
    path = os.path.join(ROOT, "corpus-tts", "roster.json")
    if not os.path.exists(path):
        print("live roster: none on disk, skipped")
        return
    print("live roster")
    with open(path) as h:
        roster = json.load(h)
    voices = roster["voices"]
    try:
        n = sv.assert_distinct(voices)
        check("every voice renders distinctly", n == len(voices))
    except ValueError as exc:
        check("every voice renders distinctly", False, str(exc))
    try:
        sv.assert_one_split_per_stem(voices)
        check("no speaker spans two splits", True)
    except ValueError as exc:
        check("no speaker spans two splits", False, str(exc))
    straddle = sv.check_straddle(voices, roster.get("close_pairs", []))
    check("no close pair straddles the split", not straddle, straddle)
    held = set(v["tier"] for v in voices if v["split"] in ("val", "test"))
    check("only natural voices are held out", held <= {"natural"}, sorted(held))


def main():
    test_derived_from_vocab()
    test_voice_identity()
    test_live_roster()
    test_plan_shape()
    test_pick_is_uniform()
    tmp = tempfile.mkdtemp(prefix="corpus-test-")
    try:
        test_split_integrity(tmp)
        test_takes_are_never_training(tmp)
        test_takes_missing(tmp)
        test_takes_present(tmp)
        test_rate_is_enforced(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
