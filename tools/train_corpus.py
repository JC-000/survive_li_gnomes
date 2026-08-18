#!/usr/bin/env python3
"""Build the multi-voice training corpus for the speaker-independent experiment.

`tools/say_corpus.py` builds an *evaluation* corpus: one voice, a handful of
rates, enough to rank front-end variants against each other. This builds a
*training* corpus: every usable `say` voice, split so that whole voices are
held out, with augmentation heavy enough that a model cannot win by memorising
a synthesiser.

    python3 tools/say_voices.py freeze corpus-tts/roster.json
    python3 tools/train_corpus.py corpus-tts/ --jobs 6
    python3 tools/train_corpus.py corpus-tts/ --plan        # size first, no audio

It shares `say_corpus`'s vocabulary, out-of-vocabulary list, channel model and
LCG rather than restating them, so a change there reaches both corpora.

## The question the corpus exists to answer

Today's spotter is speaker-*dependent*: three DTW templates per word in one
person's voice. It works and it is cheap. The open question is whether a
speaker-*independent* model is reachable without recording ten people for two
hours each, and the only data available for free is 37 synthetic voices.

So the number that matters is not accuracy on held-out `say` voices. It is
accuracy on **`takes/`** -- real human speech through the board's own
microphone. Everything here is arranged so that number is not contaminated:
`tools/corpus.py` will not return `takes/` under any split but `human`.

## What this corpus cannot do, restated because it is easy to forget

`docs/speech.md` already says it of the evaluation corpus and it is more true
here, not less. Synthetic speech varies in *session* -- rate, level, channel --
and not in *articulation*. Thirty-seven voices is thirty-seven consistent
talkers, not thirty-seven people having an off day. A model that scores well
across held-out voices here has proved it is not memorising a synthesiser. It
has **not** proved it will recognise a person, and the two are different
claims. `takes/` is the one that settles it.

## Augmentation, and why the noise is so loud

The board's own captures measure **mean|x| about 1200 of 32767, roughly
-28 dBFS** -- a high noise floor for a device held near a mouth, and a
mismatch in its own right, because clean TTS trained against a noisy capture
fails at exactly the point it is asked to work. With speech at a typical
mean|x| of 3000-6000 the board's working signal-to-noise is around 8-14 dB, so
the augmentation is centred there rather than on the -46..-34 dBFS floor
`say_corpus` uses for its quiet, close, dry evaluation channel.

Noise is specified as **SNR against the utterance**, not as an absolute level,
because the gain augmentation moves the utterance and an absolute floor would
then give an uncontrolled SNR -- loud renditions would come out clean and quiet
ones buried, with the label correlated to the level rather than to the word.
The achieved absolute dBFS is written into the manifest anyway, so it can be
compared against the board's -28.

Real recorded noise is used when Speech Commands' `_background_noise_` folder
is available (`tools/fetch_background_noise.sh`), since white noise is the one
kind of interference a spectral front end finds easiest. It falls back to the
LCG noise in `say_corpus.channel`, and the manifest records which was used, so
a run made without the download is visibly a weaker run rather than a silently
different one.

The word is also placed at a **random offset** in the buffer. A fixed lead
lets a convolutional model learn where in the frame the answer starts, which
is free accuracy here and none on the board, where the VAD decides.
"""

import argparse
import array
import json
import multiprocessing
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import say_corpus as sc  # noqa: E402
import vad  # noqa: E402
import vocab  # noqa: E402

RATE = sc.RATE

# Five rates rather than say_corpus's three-plus-four. It keeps enrolment and
# test renditions from being the same synthesis, which is the reason that file
# separates them, while giving a training set more of the rate spread that
# stands in for speaking-rate variation between people.
RATES = (145, 160, 175, 190, 205)

# Channel draws per synthesis. `say` is the expensive call -- about 150 ms --
# and the channel is cheap, so rendering once and drawing several channels off
# it is most of why a full build finishes in half an hour rather than three.
VARIANTS = 3

# The augmentation budget is not spread evenly, because the three populations
# are not equally informative and the corpus is disk-bound rather than
# idea-bound. Spending everything uniformly gave 79587 files and 3.3 GB, of
# which 83% were ordinary out-of-vocabulary nouns rendered five ways -- a lot
# of storage to establish that COFFEE is not MOTHER.
#
#   in-vocabulary   every rate, every draw. What the classes are learned from.
#   near-miss       two rates, two draws. Fewer than the keywords but far more
#                   than a random noun, because these are where a false fire
#                   comes from, and false fires are what the design trades
#                   recall away to avoid.
#   everything else one rate, two draws -- and the rate rotates per voice, so
#                   the *corpus* still covers all five even though no single
#                   voice does. Breadth of vocabulary is what the unknown class
#                   needs; depth per word is what the keywords need.
UNKNOWN_RATES = 1
NEAR_MISS_RATES = 2
UNKNOWN_VARIANTS = 2

# Wider than say_corpus's evaluation channel, which models one quiet room. A
# training set wants the range a hand-held board actually sees.
GAINS_DB = (-9, -6, -3, 0, 3)
TILTS = (-0.3, -0.15, 0.0, 0.15, 0.3)

# Signal-to-noise against the utterance's own mean|x|. Median 16 dB, which
# brackets the board's measured 8-14 dB working point; 30 dB is a quiet room
# and 8 dB is a bad one. See the module docstring.
#
# The floor was 6 dB until it was measured against `src/vad.py`, over 330
# in-vocabulary utterances from one voice:
#
#   SNR    30    24    18    14    11     8     6
#   VAD     0%    0%    0%    4%    0%   49%   83%   cannot find the word
#
# Nothing above 11 dB troubles the endpointer at all and 6 dB defeats it four
# times in five, so a 6 dB tier is mostly a tier of files the DTW path can
# never use. 8 dB is kept deliberately: it is inside the board's own range,
# and an utterance the VAD rejects is not a wasted file -- on the device that
# rejection *is* the designed behaviour, a deflection rather than a wrong
# answer, and a fixed-window classifier can still learn from the audio.
# `endpoint_ms` in the manifest says which side of that line each file fell,
# so neither consumer has to guess.
SNR_DB = (30, 24, 20, 16, 13, 10, 8)

# Silence around the word, drawn per utterance so the onset is not at a fixed
# offset. The floor is well clear of the 100 ms `src/vad.py` needs to estimate
# its background.
LEAD_MS = (200, 300, 400, 500, 600)
TAIL_MS = 400

BACKGROUND_DIR = "corpus_noise/_background_noise_"

CATEGORIES = ("word", "unknown", "variant")


# Out-of-vocabulary words beyond `say_corpus.OOV`, grouped by the keyword each
# group attacks. Grouping rather than listing flat, because "how often does the
# spotter fire on something that rhymes with FATHER" is the question the
# precision-over-recall design is built around, and a flat list cannot answer
# it. `say_corpus.OOV` already carries the general population -- what somebody
# actually says to a therapist toy -- and the dangerous three it named first:
# KNOW for the retired NO, and OTHER / ANOTHER / WONDER for FATHER.
#
# Chosen by rhyme and by initial-consonant confusion, not measured. Whether
# any of them actually fires is the output of the experiment, not an input.
NEAR_MISS = {
    "mother": ("smother", "mutter", "udder", "murder", "mumble"),
    "father": ("bother", "rather", "farther", "lather", "gather", "further"),
    "sister": ("mister", "blister", "resistor", "assist"),
    "brother": ("bothered", "rubber", "brotherly"),
    "wife": ("life", "knife", "wives", "why"),
    "husband": ("husbands", "husbandry"),
    "children": ("child", "childhood", "chill", "chilled"),
    "work": ("word", "walk", "worse", "worth", "world"),
    "money": ("monday", "many", "honey", "funny", "monkey"),
    "sleep": ("steep", "sweep", "asleep", "slip", "sleeve"),
    "death": ("deaf", "dead", "breath", "depth", "debt"),
    "love": ("glove", "above", "shove", "leave"),
    "sad": ("said", "sat", "bad", "mad", "sand", "sang"),
    "sick": ("six", "thick", "quick", "stick", "seek", "sink"),
    "happy": ("apply", "snappy", "hobby"),
    "angry": ("hungry", "anger", "angle", "ankle"),
    "afraid": ("parade", "raid", "afford"),
    "yes": ("guess", "less", "yet", "jazz", "yeah"),
    "dream": ("cream", "dreamt", "scream", "drink"),
    "computer": ("commuter", "compute", "computing", "commute"),
    "always": ("hallways", "away", "all right", "almost"),
    "sorry": ("story", "sore", "sorted", "sort"),
}

# Ordinary English that is neither a keyword nor a near-miss. The unknown class
# has to be *broad* as well as adversarial: a model that rejects only the
# things it was trained to reject has learned a 22nd class, not a rejection.
EXTRA_OOV = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "first", "last", "half",
    "monday", "friday", "weekend", "week", "month", "year", "hour", "minute",
    "house", "room", "door", "window", "table", "chair", "car", "street",
    "city", "country", "garden", "kitchen", "office", "shop", "train",
    "water", "food", "coffee", "dinner", "music", "book", "phone", "letter",
    "hand", "face", "eyes", "heart", "voice", "name", "story", "picture",
    "walk", "sit", "stand", "open", "close", "start", "stop", "wait",
    "give", "take", "make", "keep", "leave", "bring", "put", "hold",
    "good", "better", "best", "worse", "worst", "long", "short", "hard",
    "easy", "quiet", "loud", "warm", "cold", "young", "old", "new",
    "please", "thank you", "excuse me", "all right",
    "of course not", "i guess", "i mean", "you see", "sort of like",
    "what about you", "how are you", "that is right", "i am not sure",
    "it is fine", "never mind", "go on", "carry on", "let me think",
)

# Inflections of an actual keyword. Firing here is DOCTOR behaving correctly --
# "my mother's house" should reach the family rules -- so they are a third
# category, scored as neither a hit nor a false fire. `say_corpus.OOV_VARIANTS`
# has the ones its own corpus needed; these complete the set over the current
# vocabulary. `loves` is the case worth naming: the engine matches exact forms,
# so LOVE firing on "loves" is arguably right, and counting it as a false
# positive would push the threshold down for no good reason.
EXTRA_VARIANTS = {
    "loves": "love", "loved": "love", "loving": "love",
    "dreams": "dream", "dreamed": "dream",
    "sisters": "sister", "brothers": "brother", "mothers": "mother",
    "fathers": "father", "husbands": "husband", "wives": "wife",
    "deaths": "death", "sleeps": "sleep", "sleepy": "sleep",
    "works": "work", "monies": "money", "saddest": "sad",
    "happiest": "happy", "angrily": "angry", "sorrier": "sorry",
    "computed": "computer", "yes please": "yes",
}


def variants_map():
    """Every inflection that maps onto a live class, as `slug -> label`.

    Filtered against the vocabulary rather than trusted: `say_corpus`'s table
    still names WANTED and NEEDED, whose class was retired, and a variant whose
    class no longer exists is an ordinary out-of-vocabulary word that must stay
    silent -- exactly the trap `say_corpus.RETIRED` was written for. Letting it
    keep the `variant` label would excuse a real false fire.
    """
    out = {}
    for text, form in list(sc.OOV_VARIANTS.items()) + list(EXTRA_VARIANTS.items()):
        label = vocab.label_of(form)
        if label is not None:
            out[sc.slug(text.replace("_", " "))] = label
    return out


def unknown_words():
    """Every out-of-vocabulary utterance, as `(text, category, attacks)`.

    `attacks` names the keyword a near-miss was chosen to threaten, or None.
    Deduplicated across the three sources, first mention winning, so a word
    that appears in both `say_corpus.OOV` and `EXTRA_OOV` is synthesised once.
    """
    seen = set()
    out = []
    varmap = variants_map()

    def add(text, attacks=None):
        s = sc.slug(text)
        if s in seen:
            return
        seen.add(s)
        cat = "variant" if s in varmap else "unknown"
        out.append((text, cat, attacks))

    # Retired words first, so they cannot be crowded out by a later duplicate.
    # They were judged worth spotting once, so they are the likeliest to fire.
    for text in sc.RETIRED:
        add(text, attacks="retired")
    for keyword, words in sorted(NEAR_MISS.items()):
        for text in words:
            add(text, attacks=keyword)
    for text in sc.OOV:
        add(text.replace("_", " "))
    for text in EXTRA_OOV:
        add(text)
    for text in sorted(set(list(sc.OOV_VARIANTS) + list(EXTRA_VARIANTS))):
        add(text.replace("_", " "))
    return out


def load_background(path):
    """Speech Commands' `_background_noise_` WAVs, or an empty list.

    Empty is a supported outcome and not an error: the corpus still builds with
    the LCG noise, and the manifest records which was used. Failing here would
    make a 2.3 GB download a prerequisite for looking at the thing at all.
    """
    if not os.path.isdir(path):
        return []
    out = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".wav"):
            continue
        with wave.open(os.path.join(path, name), "rb") as w:
            if w.getframerate() != RATE or w.getnchannels() != 1:
                continue
            out.append((name, array.array("h", w.readframes(w.getnframes()))))
    return out


def pick_hi(rng, seq):
    """`rng.pick`, off the top bits, for a choice made at a fixed stride.

    `sc.Rng` is an LCG modulo 2^31, and its low bits carry the usual short
    period. That is harmless where `say_corpus` uses it -- a fresh draw each
    time, over lists of 3 and 4 -- and not harmless here, because the noise
    file is chosen at the *same offset* in the stream for every utterance, four
    picks after the previous one. *Measured* over 2000 utterances with the
    build's actual draw pattern and six noise files:

        rng.pick        {0: 616, 2: 708, 4: 676}         three files, never six
        top 15 bits     {0: 331, 1: 339, ..., 5: 317}    uniform

    The first corpus built before this was found used only `running_tap`,
    `exercise_bike` and `doing_the_dishes`; `dude_miaowing`, `pink_noise` and
    `white_noise` were unreachable. Nothing reported it -- the manifest said
    which noise each file got, and it took reading the histogram to notice
    that half the column was missing.
    """
    return seq[(rng.next() >> 16) % len(seq)]


def say_retry(text, path, voice, rate, attempts=3):
    """`sc.say_wav`, retried, because a 42000-file build meets a flaky `say`.

    Observed once during development: `say` returned SIGTERM for a single
    invocation that succeeded when repeated, while a second synthesis process
    was running. Concurrency itself is fine -- eight simultaneous `say -o`
    calls all returned 0 when tested -- so this is a rare transient rather
    than a contention limit, and losing forty minutes of build to it is the
    only real cost. Each retry is announced, so a voice that fails *every*
    time is visible rather than quietly absent.
    """
    import subprocess
    for attempt in range(attempts):
        try:
            sc.say_wav(text, path, voice, rate)
            return
        except subprocess.CalledProcessError as exc:
            print("  say failed (%s, %r, %d wpm): %s%s"
                  % (voice, text, rate, exc,
                     " -- retrying" if attempt + 1 < attempts else ""),
                  file=sys.stderr)
            if attempt + 1 == attempts:
                raise


def mean_abs(samples, start=0, count=None):
    if count is None:
        count = len(samples) - start
    if count <= 0:
        return 0.0
    total = 0
    for i in range(start, start + count):
        v = samples[i]
        total += v if v >= 0 else -v
    return total / float(count)


def dbfs(level):
    if level <= 0:
        return -120.0
    import math
    return 20.0 * math.log10(level / 32767.0)


def render(clean, rng, gain_db, tilt, snr_db, lead_ms, background):
    """One augmented utterance, and the record of how it was made.

    Order is gain, then tilt, then noise -- noise last, because the point of an
    SNR is the ratio at the microphone, and tilting the sum would change it.
    """
    gain_q15 = int(round(32768 * (10.0 ** (gain_db / 20.0))))
    tilt_q15 = int(round(tilt * 32768))
    lead = lead_ms * RATE // 1000
    tail = TAIL_MS * RATE // 1000
    n = lead + len(clean) + tail
    out = array.array("h", bytes(2 * n))

    prev = 0
    for i in range(len(clean)):
        x = clean[i]
        y = (x * gain_q15) >> 15
        y -= (tilt_q15 * prev) >> 15
        prev = x
        out[lead + i] = 32767 if y > 32767 else (-32768 if y < -32768 else y)

    speech = mean_abs(out, lead, len(clean))
    target = speech / (10.0 ** (snr_db / 20.0))

    if background:
        name, noise = pick_hi(rng, background)
        # A random window, so the same file does not always contribute the same
        # few seconds. The recordings are ~60 s, the window ~1.3 s.
        start = (rng.next() >> 8) % max(1, len(noise) - n)
        level = mean_abs(noise, start, n)
        scale = int(round(32768 * target / level)) if level > 0 else 0
        for i in range(n):
            v = out[i] + ((noise[start + i] * scale) >> 15)
            out[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
        source = name
    else:
        # sc.Rng.gauss_ish takes a peak-ish amplitude and returns something
        # whose mean magnitude is about a quarter of it; the factor keeps the
        # requested SNR honest for either noise source.
        amp = int(target * 4)
        for i in range(n):
            v = out[i] + rng.gauss_ish(amp)
            out[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
        source = "lcg"

    achieved = mean_abs(out, 0, lead)   # the lead is noise and nothing else

    # What `src/vad.py` makes of it, recorded per file rather than left to be
    # discovered later. The two consumers of this corpus want opposite things:
    # a template matcher can only use utterances the endpointer finds, while a
    # fixed-window classifier can use them all and should see the hard ones.
    # Measuring here costs 1.4 ms against the 650 ms the file already cost,
    # and it is the difference between a corpus with a documented property and
    # a corpus with a fifth of it quietly unusable.
    seg = vad.trim(out)
    return out, {
        "gain_db": gain_db,
        "tilt": tilt,
        "snr_db": snr_db,
        "lead_ms": lead_ms,
        "noise": source,
        "noise_dbfs": round(dbfs(achieved), 1),
        "speech_dbfs": round(dbfs(speech), 1),
        "samples": n,
        "endpoint_ms": None if seg is None else round(len(seg) * 1000.0 / RATE),
    }


def utterance_plan(voice, forms, unknowns, rates, variants):
    """Every `(text, category, label, attacks, rate, n_variants)` for a voice.

    Retired words get every rate, like the keywords do, for the reason
    `say_corpus.RETIRED` gives: they were judged worth spotting once, so they
    are the words likeliest to false-fire, and the evidence that they no longer
    do is the evidence that disappears first.

    The rotation for ordinary unknowns is offset by the voice name, so
    "coffee" is rendered at a different rate by each voice and the corpus
    covers all five rates without any voice paying for all five.
    """
    plan = []
    for form in forms:
        for rate in rates:
            plan.append((form, "word", vocab.label_of(form), None, rate, variants))

    varmap = variants_map()
    offset = sum(ord(c) for c in voice)
    for i, (text, cat, attacks) in enumerate(unknowns):
        label = varmap.get(sc.slug(text))
        if attacks == "retired":
            use = rates
        elif attacks:
            use = rates[:NEAR_MISS_RATES]
        else:
            use = (rates[(offset + i) % len(rates)],) * UNKNOWN_RATES
        for rate in use:
            plan.append((text, cat, label, attacks, rate, UNKNOWN_VARIANTS))
    return plan


def build_voice(job):
    """Render everything one voice owns. Runs in a worker process.

    Seeded from the voice name alone, so the augmentation a voice gets does not
    depend on how many workers ran or in what order they finished. A corpus
    that is only reproducible at `--jobs 1` is not reproducible.
    """
    (root, voice, split, family, locale, rates, variants,
     background_dir, resume) = job

    rng = sc.Rng(sum((i + 1) * ord(c) for i, c in enumerate(voice)) + 7)
    background = load_background(background_dir)
    forms = list(vocab.FORMS)
    unknowns = unknown_words()
    vslug = sc.slug(voice)
    raw = os.path.join(root, "_raw.%s.wav" % vslug)

    records = []
    for text, cat, label, attacks, rate, n_var in utterance_plan(
            voice, forms, unknowns, rates, variants):
        base = sc.slug(text)
        outdir = os.path.join(root, split, cat, vslug)
        os.makedirs(outdir, exist_ok=True)

        names = ["%s.r%d.%d.wav" % (base, rate, v) for v in range(n_var)]
        paths = [os.path.join(outdir, n) for n in names]
        # The RNG has to advance whether or not the files are written, or a
        # resumed build would augment the remainder differently from a fresh
        # one. So draw first, skip second.
        draws = [(pick_hi(rng, GAINS_DB), pick_hi(rng, TILTS),
                  pick_hi(rng, SNR_DB), pick_hi(rng, LEAD_MS))
                 for _ in range(n_var)]
        if resume and all(os.path.exists(p) for p in paths):
            continue

        say_retry(text, raw, voice, rate)
        clean = sc.read_wav(raw)
        for name, path, (gain, tilt, snr, lead) in zip(names, paths, draws):
            samples, how = render(clean, rng, gain, tilt, snr, lead, background)
            sc.write_wav(path, samples)
            rec = {"file": os.path.relpath(path, root), "split": split,
                   "category": cat, "label": label, "text": text,
                   "voice": voice, "family": family, "locale": locale,
                   "rate_wpm": rate, "attacks": attacks}
            rec.update(how)
            records.append(rec)

    if os.path.exists(raw):
        os.remove(raw)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="corpus directory; must contain roster.json")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--variants", type=int, default=VARIANTS)
    ap.add_argument("--voices", help="comma-separated subset, for a quick look")
    ap.add_argument("--background", default=BACKGROUND_DIR)
    ap.add_argument("--plan", action="store_true",
                    help="print the size and stop, without synthesising")
    ap.add_argument("--fresh", action="store_true",
                    help="re-render files that already exist")
    args = ap.parse_args(argv)

    roster_path = args.roster or os.path.join(args.root, "roster.json")
    if not os.path.exists(roster_path):
        sys.exit("no roster at %s -- run tools/say_voices.py freeze first"
                 % roster_path)
    with open(roster_path) as h:
        roster = json.load(h)

    rates = tuple(int(r) for r in args.rates.split(","))
    voices = roster["voices"]
    if args.voices:
        want = set(v.strip() for v in args.voices.split(","))
        voices = [v for v in voices if v["name"] in want]
        if not voices:
            sys.exit("none of those voices are in the roster")

    unknowns = unknown_words()
    per_voice = sum(e[5] for e in utterance_plan(
        "", vocab.FORMS, unknowns, rates, args.variants))
    total = per_voice * len(voices)
    background = load_background(args.background)

    print("%d voices x %d utterances = %d files" % (len(voices), per_voice, total))
    print("  vocabulary %d forms in %d classes, %d unknown texts "
          "(%d near-miss, %d inflection)"
          % (len(vocab.FORMS), len(vocab.LABELS), len(unknowns),
             sum(1 for _, _, a in unknowns if a and a != "retired"),
             sum(1 for _, c, _ in unknowns if c == "variant")))
    print("  rates %s, %d channel draws each" % (rates, args.variants))
    print("  noise: %s" % ("%d recorded files from %s"
                           % (len(background), args.background) if background
                           else "LCG only -- run tools/fetch_background_noise.sh"))
    # ~1.3 s at 32 KB/s. Worth printing: a full build is about a gigabyte and
    # finding that out afterwards is annoying.
    print("  roughly %.1f GB and %.0f min at --jobs %d"
          % (total * 42000 / 1e9, total * 0.20 / 60.0 / args.jobs, args.jobs))
    if args.plan:
        return 0

    os.makedirs(args.root, exist_ok=True)
    jobs = [(args.root, v["name"], v["split"], v["family"], v["locale"],
             rates, args.variants, os.path.abspath(args.background),
             not args.fresh)
            for v in voices]

    records = []
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            for i, out in enumerate(pool.imap_unordered(build_voice, jobs), 1):
                records.extend(out)
                print("  [%2d/%2d] %d files" % (i, len(jobs), len(out)),
                      file=sys.stderr)
    else:
        for i, job in enumerate(jobs, 1):
            out = build_voice(job)
            records.extend(out)
            print("  [%2d/%2d] %s: %d files" % (i, len(jobs), job[1], len(out)),
                  file=sys.stderr)

    manifest = os.path.join(args.root, "manifest.json")
    doc = {"generated_by": "tools/train_corpus.py",
           "rate": RATE, "rates_wpm": list(rates), "variants": args.variants,
           "template_format": None,
           "roster": os.path.relpath(roster_path, args.root),
           "entries": sorted(records, key=lambda r: r["file"])}
    if not args.fresh and os.path.exists(manifest):
        # A resumed build only re-renders what is missing, so its own records
        # cover only that. Merge, newest winning, or the manifest would shrink
        # to whatever the last run happened to touch.
        with open(manifest) as h:
            old = json.load(h)
        merged = dict((r["file"], r) for r in old.get("entries", []))
        merged.update((r["file"], r) for r in records)
        doc["entries"] = sorted(merged.values(), key=lambda r: r["file"])
    with open(manifest, "w") as h:
        json.dump(doc, h, indent=1, sort_keys=True)
        h.write("\n")

    counts = {}
    for r in doc["entries"]:
        counts[(r["split"], r["category"])] = \
            counts.get((r["split"], r["category"]), 0) + 1
    print("\n%s: %d utterances" % (manifest, len(doc["entries"])))
    print("%-6s %8s %8s %8s" % ("split", "word", "unknown", "variant"))
    for split in ("train", "val", "test"):
        print("%-6s %8d %8d %8d"
              % (split, counts.get((split, "word"), 0),
                 counts.get((split, "unknown"), 0),
                 counts.get((split, "variant"), 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
