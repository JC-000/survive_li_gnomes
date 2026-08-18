#!/usr/bin/env python3
"""Enumerate macOS `say` voices, prove they are distinct, and freeze a split.

The speaker-independent experiment rests on one assumption: that `say` gives us
many *different speakers* for free. This tool exists because that assumption is
false in three separate ways, each of which is silent, and each of which would
inflate the apparent speaker diversity of the training corpus without producing
a single error message.

    python3 tools/say_voices.py probe                    # the evidence table
    python3 tools/say_voices.py freeze corpus-tts/roster.json
    python3 tools/say_voices.py show corpus-tts/roster.json

## The three ways the voice list lies

**1. `say -v '?'` lists 184 voices and 43 of them are English.** Forty-three
sounds like a corpus. It is not forty-three speakers -- see below.

**2. The list is not stable, and it contains name collisions.** *Measured*
2026-08-18: two `say -v '?'` runs eleven minutes apart returned 184 and then
186 lines, the second containing `Aman (English (India))` and
`Tara (English (India))` **twice each**, with different sample phrases ("Hello!
My name is Aman." and "Hi, I'm Siri!"). Siri voices had finished downloading in
the background between the two runs. `say -v` takes a name, so the second entry
of a colliding pair is **unaddressable**: the flag resolves to one of them and
there is no way to ask for the other. A corpus built by iterating the live list
is therefore not reproducible even on the same machine on the same day, which is
why `freeze` writes the roster to JSON and the corpus builder reads *that*.

**3. Voices share a synthesiser.** This is the big one, and nothing about the
output announces it. *Measured*, per-word durations at `-r 175`:

    Eddy / Flo / Grandma / Grandpa / Reed / Rocko / Sandy / Shelley,
    all eight en_GB:   mother 830  father 911  computer 1113  yes 814  sad 975
    the same eight, en_US:      878        943           1065     814      975

Eight voices, one duration to the millisecond. They are one prosody model with
eight timbres, not eight speakers -- and a model that learns *timing* rather
than *timbre* would score perfectly across all eight while having learned
nothing that transfers to a person. The legacy MacinTalk voices do the same
thing in smaller groups: Fred / Trinoids / Zarvox / Bubbles share one tuple,
Junior / Kathy / Superstar / Ralph another, Albert / Bahh a third.

Two of them go further and are the *same voice under two names*. Long-term
average log-mel distance, Q8 log2 units, over 43 English voices (903 pairs):

    Eddy (UK)  <-> Reed (UK)       6.9  }  the median pair is 220.5
    Eddy (US)  <-> Reed (US)      10.0  }  and the widest is 643.7
    Grandma (US) <-> Shelley (US) 39.5
    Eddy (UK)  <-> Eddy (US)      63.1     <- the accent matters more than the name

So the split has to be taken over **timing families**, not over voice names.
Holding out Reed while training on Eddy would be a speaker-independence test
whose held-out speaker is in the training set.

## What this tool does not decide

Whether a voice is *worth training on*. Six of the 43 are singing rather than
speaking (Bells, Cellos, Organ, Jester, Bad News, Good News), and `--speech-only`
drops them on the objective grounds that their median word runs past
SPEECH_MAX_MS at a normal rate. The remainder includes robotic and processed
timbres (Zarvox, Bahh, Whisper) which are odd but do carry the phonetic
contrast: a DTW check of same-word-across-rates against different-word finds a
separation ratio of 1.8 to 3.8 for **every** English voice, so none is
degenerate. Whether the odd ones help or hurt a speaker-independent model is
the experiment, not an input to it.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import mfcc  # noqa: E402

# The locales a keyword spotter for English should be trained on. Deliberately
# wide: en_IN and en_ZA are as legitimate an accent of the target language as
# en_US, and accent breadth is the whole point of the exercise.
LOCALES = ("en_US", "en_GB", "en_AU", "en_IE", "en_ZA", "en_IN")

# Timing signature. Vocabulary words spanning one to three syllables, so a
# shared prosody model shows up as a shared *shape* and not just a shared
# overall speed. `sad` and `yes` are the short pair, `computer` the long one.
PROBE_WORDS = ("mother", "father", "computer", "yes", "sad")
PROBE_RATE = 175   # words per minute; the middle of `say`'s natural range

# Timbre fingerprint. One utterance, five words, long enough that the
# long-term average log-mel is a stable estimate of the voice rather than of
# whichever vowel happened to dominate.
FINGERPRINT_TEXT = "mother father dream sorry computer"

# Two voices count as sharing a prosody model when **every** probe word is
# within this. A tolerance rather than a quantised key, because a key puts
# Fred (`yes` 536 ms) and Bubbles (534 ms) in different buckets whenever the
# rounding boundary happens to fall between them, and 2 ms is 32 samples.
# Measured spread inside the families that exist: 0..2 ms. Measured gap to the
# nearest voice outside one: 15 ms (Daniel vs Karen on `computer`, and those
# two are 116 ms apart on `yes`, so the all-words test separates them anyway).
DURATION_TOL_MS = 8

# Below this, two voices are the same voice with a different label. Set an
# order of magnitude under the 220.5 median pair distance and well clear of the
# 39.5 third-closest pair, so it captures the Eddy/Reed collapse without
# reaching for anything arguable. Pairs under REPORT_DIST are printed anyway,
# because the number the reader wants is the gap, not the verdict.
TWIN_DIST = 25.0
REPORT_DIST = 100.0

# Median probe-word duration at 175 wpm. *Measured* over all 43 English
# voices, the median lands in 407..911 ms for every voice that speaks and in
# 1185..1940 ms for every voice that sings -- Cellos and Organ at 1185/1186,
# then Jester 1267, Bells 1348, Bad News 1534, Good News 1940. The cut sits in
# the empty band between 911 and 1185, so it is not a judgement call; note it
# has to be the *median*, since iOS voices take 1113 ms over `computer` alone.
SPEECH_MAX_MS = 1050

DEFAULT_SPLIT = ("train", "val", "test")

# Tiers, and the objective rule that assigns each. Not cosmetic: the split
# holds out `natural` voices and nothing else, because with no human takes the
# held-out voices are the *only* proxy for a real speaker, and a test set made
# of 1990s formant synthesisers predicts nothing about a person.
#
#   natural     a prosody family of one, median probe word <= 600 ms.
#               Measured, this selects exactly Samantha, Daniel, Karen, Moira,
#               Tessa, Rishi, Aman, Tara -- every modern concatenative voice
#               and nothing else. Their medians run 470..539 ms against the
#               next voice up at 703 (Whisper) and Boing at 750, so the cut
#               sits in a 160 ms gap rather than on a preference.
#   expressive  a family whose members span more than one locale. Measured,
#               exactly one family does: the sixteen iOS voices, eight names
#               in en_GB and en_US. Human-sounding, but one synthesiser.
#   novelty     everything else -- the MacinTalk formant and effect voices.
#               Real speech in the sense that they carry phonetic contrast,
#               and nothing like a human vocal tract.
NATURAL_MAX_WORD_MS = 600
TIERS = ("natural", "expressive", "novelty")

# `say -v '?'` reports a locale and nothing else, so gender is curated here.
# It is a **judgement from Apple's own voice documentation and the names**, not
# a measurement, and it is recorded rather than inferred because the split has
# to guarantee both genders in training: the user is male and wants the toy to
# work for women too, and nothing in the voice list would let that be checked.
#
# Anything absent is `unknown` and is treated as satisfying no gender
# constraint -- absent is not a third gender, it is missing information, and a
# constraint met by a guess is worse than one reported unmet.
VOICE_GENDER = {
    # en_US
    "Allison": "female", "Ava": "female", "Joelle": "female",
    "Nicky": "female", "Noelle": "female", "Samantha": "female",
    "Susan": "female", "Zoe": "female", "Kathy": "female",
    "Evan": "male", "Nathan": "male", "Tom": "male", "Bruce": "male",
    "Albert": "male", "Fred": "male", "Junior": "male", "Ralph": "male",
    # other English locales
    "Karen": "female", "Moira": "female", "Tessa": "female",
    "Tara": "female", "Fiona": "female", "Serena": "female",
    "Daniel": "male", "Rishi": "male", "Aman": "male", "Oliver": "male",
    # the iOS expressive set is deliberately absent: Apple does not present
    # Eddy, Flo, Reed, Rocko, Sandy or Shelley as gendered, and guessing from
    # the name would put a made-up fact into a constraint check.
}

# Voice quality, which `say` puts in the name: `Allison (Enhanced)`. Enhanced
# and Premium are much larger, much better models than the Compact default and
# are markedly closer to real speech, which is the entire reason for training
# on synthetic voices at all. Where the same voice is installed at two
# qualities, the better one wins -- they share a name stem, so `group_families`
# already keeps them on the same side of the split.
QUALITY_RANK = {"Premium": 3, "Enhanced": 2, "Compact": 1}

# Voices this close in long-term log-mel must not straddle a split.
#
# A backstop, not the primary defence: `group_families` already merges every
# same-name and twin pair, so the pairs this is aimed at cannot straddle by
# construction. It exists to catch a *future* change to the grouping, which is
# why it raises.
#
# The threshold sits between the two populations rather than on either. The
# dishonest pairs -- same voice, two accents -- run 61.5 (Sandy UK/US) to 73.3
# (Grandma UK/US). The closest genuinely-different pair that survives grouping
# is Aman and Tara at **84.7**: two en_IN Siri voices, different names,
# different prosody families, and different vocal tracts. They are allowed to
# straddle, and it is better that they do -- Aman trains and Tara tests, so the
# en_IN accent is represented on both sides. A test accent absent from training
# would measure accent transfer, which is a different and harder question than
# the speaker transfer being measured here.
#
# 75 therefore has 73.3 below it and 84.7 above it. The margin is thin on the
# upper side and worth knowing: a future voice landing between 75 and 84 would
# trip this and need a judgement rather than a threshold change.
STRADDLE_DIST = 75.0


def enumerate_voices():
    """Every English voice `say` currently offers, deduplicated by name.

    Returns `(voices, collisions)`. A collision is a name the list gives more
    than once: only the first is reachable through `say -v`, so the rest are
    dropped and reported rather than silently counted as extra speakers.
    """
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                         check=True).stdout
    voices = []
    seen = {}
    collisions = []
    for line in out.splitlines():
        # Columns are space-padded to a fixed width that long names overflow:
        # `Eddy (English (UK)) en_GB` has a **single** space before the locale
        # while `Samantha            en_US` has twelve. Anchoring on the `#`
        # that starts the sample phrase is the only reliable split.
        m = re.match(r"^(.*?)\s+([a-z]{2}_[A-Z]{2})\s*#", line)
        if not m:
            continue
        name, locale = m.group(1).strip(), m.group(2)
        if locale not in LOCALES:
            continue
        if name in seen:
            collisions.append(name)
            continue
        seen[name] = True
        voices.append({"name": name, "locale": locale})
    return voices, collisions


def prefer_quality(voices):
    """One entry per (name stem, locale), keeping the best quality installed.

    `Samantha` and `Samantha (Enhanced)` are both installed and both en_US.
    They are **the same speaker** at two quality tiers, and they render
    differently -- different model, different bytes -- so `assert_distinct`
    passes them as two voices and would go on passing them. Distinctness is not
    identity: two renders can differ byte-for-byte and still be one speaker.

    `group_families` already unions them, because `name_stem` strips the
    parenthesis, so they cannot straddle the split. Dropping the Compact one
    anyway does two further things: it stops the corpus carrying the same
    speaker twice with half the data at the worse quality, and it honours the
    instruction to prefer Enhanced and Premium, which are much better models.

    Keyed by stem **and locale**, so `Flo (English (UK))` and
    `Flo (English (US))` both survive -- those differ by accent, not by tier,
    and the measured distance between such a pair (61.5..73.3) is far larger
    than nothing.
    """
    best = {}
    for v in voices:
        key = (name_stem(v["name"]), v["locale"])
        rank = QUALITY_RANK.get(quality_of(v["name"]), 1)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, v)
    kept = [v for _, v in best.values()]
    dropped = [v["name"] for v in voices if v not in kept]
    kept.sort(key=lambda v: v["name"])
    return kept, sorted(dropped)


def assert_one_split_per_stem(voices):
    """No speaker identity may appear in two splits. Raises.

    A direct guard on the thing `group_families` is supposed to guarantee, kept
    separate from it so a future change to the grouping cannot quietly remove
    the property. Name stem is the identity: locale variants and quality tiers
    of one voice all share it.
    """
    seen = {}
    for v in voices:
        seen.setdefault(name_stem(v["name"]), set()).add(v["split"])
    bad = dict((k, sorted(s)) for k, s in seen.items() if len(s) > 1)
    if bad:
        raise ValueError(
            "the same speaker is in two splits: %s"
            % "; ".join("%s in %s" % (k, " and ".join(s))
                        for k, s in sorted(bad.items())))


def say_pcm(text, voice, rate, path):
    """Synthesise to `path` and return the samples.

    `--data-format=LEI16@16000` is the format docs/speech.md specifies, so
    nothing is resampled anywhere in this pipeline.

    `say` pads its header with a 4044-byte `FLLR` chunk, putting the PCM at
    offset **4096 rather than 44**. Anything that skips a fixed 44-byte header
    reads 4 kB of padding as audio -- a burst of zeros followed by a truncated
    word. `wave` walks the chunk list, so it lands in the right place; that is
    the reason to use a parser here rather than a slice.
    """
    subprocess.run(["say", "-v", voice, "-r", str(rate),
                    "--data-format=LEI16@16000", "--file-format=WAVE",
                    "-o", path, text], check=True)
    with wave.open(path, "rb") as w:
        assert w.getframerate() == mfcc.SAMPLE_RATE, w.getframerate()
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def fingerprint(pcm):
    """Long-term average log-mel, level removed: a voice's spectral signature.

    Level is subtracted because it is synthesis gain, which says nothing about
    who is speaking, and because the corpus builder is about to randomise it
    anyway. What is left is the shape of the average spectrum, which is what
    makes Reed sound like Eddy.

    Uses `mfcc.logmel_q8`, i.e. the project's own front end, so a voice pair
    this calls identical is identical *in the space the recogniser works in* --
    which is the only sense that matters here.
    """
    import array
    rows = mfcc.logmel_q8(array.array("h", pcm))
    if not rows:
        return None
    n = float(len(rows))
    avg = [sum(r[j] for r in rows) / n for j in range(len(rows[0]))]
    level = sum(avg) / len(avg)
    return [v - level for v in avg]


def fingerprint_distance(a, b):
    """Mean |difference| per mel band, in Q8 log2 units (1 = 1/256 octave)."""
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def probe(voices, scratch, verbose=True):
    """Measure every voice. Returns the list, each entry gaining measurements."""
    raw = os.path.join(scratch, "_probe.wav")
    for i, v in enumerate(voices):
        durations = []
        for word in PROBE_WORDS:
            pcm = say_pcm(word, v["name"], PROBE_RATE, raw)
            durations.append(round(len(pcm) / 2 * 1000.0 / mfcc.SAMPLE_RATE))
        pcm = say_pcm(FINGERPRINT_TEXT, v["name"], PROBE_RATE, raw)
        v["durations_ms"] = durations
        v["median_word_ms"] = sorted(durations)[len(durations) // 2]
        v["speech_like"] = v["median_word_ms"] <= SPEECH_MAX_MS
        v["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
        v["fingerprint"] = fingerprint(pcm)
        if verbose:
            print("  [%2d/%2d] %-24s %-6s median %4d ms%s"
                  % (i + 1, len(voices), v["name"], v["locale"],
                     v["median_word_ms"], "" if v["speech_like"] else "  (sings)"),
                  file=sys.stderr)
    if os.path.exists(raw):
        os.remove(raw)
    return voices


def name_stem(name):
    """`Flo (English (UK))` -> `Flo`. The character behind the locale variant.

    Also collapses `Allison (Enhanced)` onto `Allison`, so a voice installed at
    two qualities is one speaker for splitting purposes.
    """
    return name.split(" (")[0].strip()


def quality_of(name):
    """`Allison (Enhanced)` -> `Enhanced`. Compact is the unmarked default."""
    for q in QUALITY_RANK:
        if "(%s)" % q in name:
            return q
    return "Compact"


def gender_of(name):
    """`female` / `male` / `unknown`. Curated -- see VOICE_GENDER."""
    return VOICE_GENDER.get(name_stem(name), "unknown")


def same_prosody(a, b):
    return all(abs(x - y) <= DURATION_TOL_MS
               for x, y in zip(a["durations_ms"], b["durations_ms"]))


def group_families(voices):
    """Connected components over three ways of being the same speaker.

    Family, not voice, is the unit the split is taken over, and a family is a
    *component* rather than a bucket because the three relations overlap
    without nesting:

    - **same prosody** -- every probe word within `DURATION_TOL_MS`. One
      synthesiser. A classifier can learn timing, and a human held-out speaker
      shares timing with nobody, so this cue has to be removed from the test.
    - **same name stem** -- `Flo (English (UK))` and `Flo (English (US))`. One
      character rendered in two accents; long-term log-mel puts these pairs at
      61.5..73.3, against a 220.5 median over all pairs.
    - **twin timbre** -- fingerprint under `TWIN_DIST`. Eddy and Reed, at 6.9
      and 10.0, are one voice wearing two names.

    Taking components matters: the first pass of this tool grouped by prosody
    alone, which put the eight en_GB expressive voices in `train` and six of
    the eight en_US ones in `val` -- so `val` measured generalisation to Flo
    from having trained on Flo. That is the exact failure this experiment was
    built to detect, reproduced inside its own split.
    """
    n = len(voices)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = voices[i], voices[j]
            if same_prosody(a, b) or name_stem(a["name"]) == name_stem(b["name"]):
                union(i, j)
            elif (a["fingerprint"] and b["fingerprint"]
                  and fingerprint_distance(a["fingerprint"],
                                           b["fingerprint"]) < TWIN_DIST):
                union(i, j)

    groups = {}
    for i, v in enumerate(voices):
        groups.setdefault(find(i), []).append(v)
    out = []
    for members in groups.values():
        members.sort(key=lambda v: v["name"])
        out.append({"key": members[0]["name"],
                    "voices": [v["name"] for v in members],
                    "durations_ms": members[0]["durations_ms"]})
    out.sort(key=lambda f: (-len(f["voices"]), f["key"]))
    return out


def find_twins(voices, family_of):
    """Pairs whose long-term spectrum is close enough to be one voice.

    Reported rather than acted on, because `group_families` has already merged
    them. It is here so that the evidence for the merge is inspectable, and so
    a reader who later argues for a per-voice split can see what it would cost.
    """
    close = []
    for i, a in enumerate(voices):
        for b in voices[i + 1:]:
            if a["fingerprint"] is None or b["fingerprint"] is None:
                continue
            d = fingerprint_distance(a["fingerprint"], b["fingerprint"])
            if d < REPORT_DIST:
                close.append({"a": a["name"], "b": b["name"],
                              "distance": round(d, 1),
                              "twin": d < TWIN_DIST,
                              "same_family":
                                  family_of[a["name"]] == family_of[b["name"]]})
    close.sort(key=lambda p: p["distance"])
    return close


def _greedy(families, weights):
    """Largest family first, into whichever split is furthest below target."""
    total = sum(len(f["voices"]) for f in families)
    if not total:
        return
    targets = dict(zip(DEFAULT_SPLIT,
                       [total * w / float(sum(weights)) for w in weights]))
    got = dict((s, 0) for s in DEFAULT_SPLIT)
    for f in sorted(families, key=lambda f: (-len(f["voices"]), f["key"])):
        pick = max(DEFAULT_SPLIT,
                   key=lambda s: ((targets[s] - got[s]) / targets[s]
                                  if targets[s] else -1))
        f["split"] = pick
        got[pick] += len(f["voices"])


def assert_distinct(voices):
    """No two voice names may render byte-identically. Raises if they do.

    **`say -v <name>` does not fail on a voice that is not installed -- it
    renders in the system default voice and returns 0.** So a roster built
    from a written-down list of names silently becomes one voice under many
    names, those names land on both sides of the split, and the
    speaker-independent score becomes a voice matched against itself. It reads
    as a *good* result, which is why it has to raise rather than warn.

    This roster is built by probing `say -v '?'`, which lists only what is
    installed, so it should never trigger -- and it is checked anyway, because
    "should never" is what CLAUDE.md keeps a list of. *Measured* 2026-08-18
    over all 43 English voices: 43 distinct digests, no collisions.

    Called from `build_roster` and again from `tools/train_corpus.py` at the
    top of a build, so it cannot be the step somebody forgets.
    """
    seen = {}
    for v in voices:
        seen.setdefault(v["pcm_sha256"], []).append(v["name"])
    phantom = dict((k, n) for k, n in seen.items() if len(n) > 1)
    if phantom:
        raise ValueError(
            "these voice names render identically, so all but the first are "
            "`say` falling back to the default voice: %s"
            % "; ".join(", ".join(n) for n in phantom.values()))
    return len(seen)


def assign_tier(voices, families):
    """Label every voice `natural` / `expressive` / `novelty`. See TIERS."""
    size = {}
    locales = {}
    for f in families:
        size[f["key"]] = len(f["voices"])
        locales[f["key"]] = set()
    by_name = dict((v["name"], v) for v in voices)
    for f in families:
        for name in f["voices"]:
            locales[f["key"]].add(by_name[name]["locale"])
    for v in voices:
        fam = v["family"]
        if size[fam] == 1 and v["median_word_ms"] <= NATURAL_MAX_WORD_MS:
            v["tier"] = "natural"
        elif len(locales[fam]) > 1:
            v["tier"] = "expressive"
        else:
            v["tier"] = "novelty"
    for f in families:
        f["tier"] = by_name[f["voices"][0]]["tier"]
    return families


def check_straddle(voices, close_pairs):
    """Close voice pairs that ended up on opposite sides. Returns complaints.

    The first roster split by prosody alone, which put the eight en_GB
    expressive voices in `train` and six of the eight en_US ones in `val` --
    so `val` measured generalisation to Flo having trained on Flo. Seventeen
    of thirty-four close pairs straddled the boundary and nothing said so.
    """
    split_of = dict((v["name"], v["split"]) for v in voices)
    out = []
    for p in close_pairs:
        if p["distance"] >= STRADDLE_DIST:
            continue
        a, b = split_of.get(p["a"]), split_of.get(p["b"])
        if a and b and a != b:
            out.append("%s (%s) and %s (%s) are %.1f apart but split %s/%s"
                       % (p["a"], a, p["b"], b, p["distance"], a, b))
    return out


def assign_split(families, weights=(3, 1, 1)):
    """Whole families to train / val / test, in two strata.

    Balanced by **voice count**, not family count: families run from one voice
    to sixteen, and counting families would hand `val` sixteen voices while
    calling it one unit.

    The two strata are families of one voice and families of several, and they
    are assigned against separate budgets. Without that, greedy assignment puts
    the sixteen-voice expressive family and then the four-voice legacy families
    wherever the deficit is largest, and the nine one-voice families -- which
    are every modern natural-sounding voice this Mac has, Samantha, Daniel,
    Karen, Moira, Tessa, Rishi, Aman, Tara -- land wherever the arithmetic
    leaves them. A `test` split made entirely of 1990s formant synthesisers
    would answer a question nobody asked. Stratifying guarantees each split
    two or three of the natural voices.

    Deterministic rather than shuffled, so the roster is reproducible, and
    because with one family holding 43% of the voices a shuffle has a real
    chance of putting it in `test`.
    """
    for f in families:
        f["stratum"] = f["tier"]

    # Everything outside the natural tier trains and is never held out. The
    # held-out voices are the evidence for generalisation to a person, so they
    # have to be the voices that most resemble one -- a `test` split containing
    # Bahh predicts how the model does on a sheep noise.
    natural = [f for f in families if f["tier"] == "natural"]
    for f in families:
        if f["tier"] != "natural":
            f["split"] = "train"

    # The natural tier is assigned against explicit requirements rather than by
    # share, because the requirements are what the split is *for* and a 3:1:1
    # greedy pass satisfies them only by luck. The user speaks American English
    # and is male, and wants the toy to work for women too.
    #
    #   1. at least one en_US voice in train  -- train on the accent that will
    #      be spoken to it. The previous roster had none: every natural
    #      training voice was en_IN, en_IE or en_ZA and the only natural en_US
    #      voice sat in val, so the model had never heard the target accent and
    #      an accent gap could be reported as a speaker gap.
    #   2. both genders in train
    #   3. at least one en_US voice in test, so generalisation to an unseen
    #      American speaker is measured rather than assumed
    #   4. whatever is left spread 3:1:1
    #
    # Highest quality first within each step: Enhanced and Premium are much
    # better models than Compact, so when a requirement can be met by either,
    # the better voice is the one to spend on it.
    def rank(f):
        v = f["voices"][0]
        return (-QUALITY_RANK.get(quality_of(v), 1), v)

    us = sorted([f for f in natural if f["locale"] == "en_US"], key=rank)
    rest = sorted([f for f in natural if f["locale"] != "en_US"], key=rank)
    for f in natural:
        f["split"] = None

    # American voices are allocated by gender, test first and best first.
    #
    # Test gets one of each gender before train gets any, and gets the better
    # models, because `test` is where a handful of voices carry the entire
    # claim about an unseen American speaker while `train` has thirty others to
    # dilute a weak one. Train then takes one of each gender from what is left,
    # and the remainder trains.
    for want in ("male", "female"):
        for f in us:
            if f["split"] is None and gender_of(f["voices"][0]) == want:
                f["split"] = "test"
                break
    for want in ("male", "female"):
        for f in us:
            if f["split"] is None and gender_of(f["voices"][0]) == want:
                f["split"] = "train"
                break
    for f in us:
        if f["split"] is None:
            f["split"] = "train"

    def genders_in(split):
        return set(gender_of(v) for f in families if f["split"] == split
                   for v in f["voices"]) - {"unknown"}

    def take(pool, want_gender, split):
        """Assign the best unassigned voice of `want_gender` to `split`."""
        for f in pool:
            if f["split"] is None and gender_of(f["voices"][0]) == want_gender:
                f["split"] = split
                return True
        return False

    # Any gender still missing from train is filled from the other locales --
    # an accented voice of the right gender beats no voice of that gender.
    for want in ("male", "female"):
        if want not in genders_in("train"):
            take(rest, want, "train")

    # Then balance the held-out splits by gender. The user is male and wants
    # the toy to work for women too, so a `test` split that is entirely female
    # cannot answer half the question -- and the first version of this policy
    # produced exactly that, because it optimised for locale and never looked
    # at gender once the en_US test voice was placed.
    for split in ("test", "val"):
        for want in ("male", "female"):
            if want not in genders_in(split):
                take(rest, want, split) or take(us, want, split)

    _greedy([f for f in rest if f["split"] is None], weights)
    for f in natural:
        if f["split"] is None:
            f["split"] = "train"
    return families


def check_requirements(voices):
    """Which of the split's requirements hold. Returns a list of complaints.

    Reported rather than raised, because some of them cannot be met by any
    assignment -- if the machine has exactly one en_US male voice it can be
    trained on or held out, never both, and the right response is to say so
    rather than to pick one silently and call the split satisfied.
    """
    out = []
    by_split = {}
    for v in voices:
        by_split.setdefault(v["split"], []).append(v)

    train = by_split.get("train", [])
    if not any(v["locale"] == "en_US" for v in train):
        out.append("no en_US voice in train -- the model never hears the "
                   "accent it will be spoken to")
    genders = set(gender_of(v["name"]) for v in train) - {"unknown"}
    for want in ("male", "female"):
        if want not in genders:
            out.append("no %s voice in train" % want)

    for split in ("val", "test"):
        rows = by_split.get(split, [])
        if rows and not any(v["locale"] == "en_US" for v in rows):
            out.append("no en_US voice in %s -- generalisation to an unseen "
                       "American speaker is assumed, not measured" % split)
        seen = set(gender_of(v["name"]) for v in rows) - {"unknown"}
        missing = sorted({"male", "female"} - seen)
        if rows and missing:
            out.append("no %s voice in %s"
                       % (" or ".join(missing), split))
    return out


def build_roster(scratch, speech_only=True, weights=(3, 1, 1), verbose=True):
    voices, collisions = enumerate_voices()
    if verbose:
        print("%d English voices from `say -v '?'`%s"
              % (len(voices),
                 ", %d unaddressable duplicate name(s) dropped: %s"
                 % (len(collisions), ", ".join(sorted(set(collisions))))
                 if collisions else ""),
              file=sys.stderr)
    voices, superseded = prefer_quality(voices)
    if verbose and superseded:
        print("superseded by a better-quality install of the same voice: %s"
              % ", ".join(superseded), file=sys.stderr)
    probe(voices, scratch, verbose)

    dropped = []
    if speech_only:
        dropped = [v["name"] for v in voices if not v["speech_like"]]
        voices = [v for v in voices if v["speech_like"]]

    distinct = assert_distinct(voices)
    families = group_families(voices)
    family_of = {}
    for f in families:
        for name in f["voices"]:
            family_of[name] = f["key"]
    for v in voices:
        v["family"] = family_of[v["name"]]

    for v in voices:
        v["gender"] = gender_of(v["name"])
        v["quality"] = quality_of(v["name"])
    for f in families:
        f["locale"] = next(v["locale"] for v in voices
                           if v["name"] == f["voices"][0])
    assign_tier(voices, families)
    assign_split(families, weights)
    for v in voices:
        v["split"] = next(f["split"] for f in families
                          if v["name"] in f["voices"])
    assert_one_split_per_stem(voices)
    twins = find_twins(voices, family_of)
    unmet = check_requirements(voices)
    straddle = check_straddle(voices, twins)
    if straddle:
        raise ValueError("the split leaks: %s" % "; ".join(straddle))

    return {
        "generated_by": "tools/say_voices.py",
        "locales": list(LOCALES),
        "probe_words": list(PROBE_WORDS),
        "probe_rate": PROBE_RATE,
        "speech_only": speech_only,
        "distinct_digests": distinct,
        "tiers": list(TIERS),
        "dropped_not_speech": dropped,
        "duplicate_names": sorted(set(collisions)),
        "superseded_by_quality": superseded,
        "voices": voices,
        "families": families,
        "close_pairs": twins,
        "unmet_requirements": unmet,
    }


def print_report(roster, full=False):
    voices = roster["voices"]
    fam = roster["families"]
    print("%d usable voices in %d timing families" % (len(voices), len(fam)))
    if roster["dropped_not_speech"]:
        print("dropped, median word > %d ms (singing, not speaking): %s"
              % (SPEECH_MAX_MS, ", ".join(roster["dropped_not_speech"])))
    if roster.get("superseded_by_quality"):
        print("dropped, a better quality tier of the same voice is installed: %s"
              % ", ".join(roster["superseded_by_quality"]))
    if roster["duplicate_names"]:
        print("unaddressable duplicate names in `say -v '?'`: %s"
              % ", ".join(roster["duplicate_names"]))

    by_name = dict((v["name"], v) for v in voices)
    print("\n%-6s %-3s %-27s  %s"
          % ("split", "n", "family", " ".join("%6s" % w for w in PROBE_WORDS)))
    for f in sorted(fam, key=lambda f: (DEFAULT_SPLIT.index(f["split"]),
                                        -len(f["voices"]), f["key"])):
        print("%-6s %-11s %-3d %-27s  %s"
              % (f["split"], f["tier"], len(f["voices"]), f["key"],
                 " ".join("%6d" % d for d in f["durations_ms"])))
        for name in f["voices"]:
            v = by_name[name]
            print("           %-27s  %-6s %-7s %s"
                  % (name, v["locale"], v.get("gender", "?"),
                     v.get("quality", "?")))

    counts = {}
    for v in voices:
        counts[v["split"]] = counts.get(v["split"], 0) + 1
    famcount = {}
    for f in fam:
        famcount[f["split"]] = famcount.get(f["split"], 0) + 1
    print("\n%-6s %8s %10s" % ("split", "voices", "families"))
    for s in DEFAULT_SPLIT:
        print("%-6s %8d %10d" % (s, counts.get(s, 0), famcount.get(s, 0)))
    tiers = {}
    for v in voices:
        tiers[v["tier"]] = tiers.get(v["tier"], 0) + 1
    print("tiers: %s" % ", ".join("%s %d" % (t, tiers.get(t, 0)) for t in TIERS))
    print("%d voices, %d distinct probe digests -- no `say` fallbacks"
          % (len(voices), roster.get("distinct_digests", 0)))

    unmet = roster.get("unmet_requirements") or []
    if unmet:
        print("\nUNMET, and stated rather than worked around:")
        for u in unmet:
            print("  - %s" % u)

    close = roster["close_pairs"]
    if close:
        print("\nvoice pairs closer than %.0f in long-term log-mel "
              "(Q8 log2, 1 = 1/256 octave):" % REPORT_DIST)
        for p in close[:12] if not full else close:
            print("  %6.1f  %-24s %-24s %s%s"
                  % (p["distance"], p["a"], p["b"],
                     "SAME VOICE" if p["twin"] else "",
                     "" if p["same_family"] else "  [different families]"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    probe_cmd = sub.add_parser("probe", help="measure every voice, print the evidence")
    freeze_cmd = sub.add_parser("freeze", help="write the roster JSON")
    freeze_cmd.add_argument("out", help="path for roster.json")
    for q in (probe_cmd, freeze_cmd):
        q.add_argument("--all-voices", action="store_true",
                       help="keep the singing voices too")
        q.add_argument("--weights", default="3,1,1",
                       help="train,val,test share by voice count")
        q.add_argument("--scratch", default="/tmp")

    s = sub.add_parser("show", help="report an existing roster")
    s.add_argument("roster")
    s.add_argument("--full", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "show":
        with open(args.roster) as h:
            print_report(json.load(h), args.full)
        return 0

    weights = tuple(int(x) for x in args.weights.split(","))
    roster = build_roster(args.scratch, speech_only=not args.all_voices,
                          weights=weights)
    if args.cmd == "freeze":
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as h:
            json.dump(roster, h, indent=1, sort_keys=True)
            h.write("\n")
        print("wrote %s" % args.out)
    print_report(roster)
    return 0


if __name__ == "__main__":
    sys.exit(main())
