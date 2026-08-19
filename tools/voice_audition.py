#!/usr/bin/env python3
"""Render DOCTOR's replies through macOS `say` so a person can pick a voice.

The device cannot synthesise speech -- a MicroPython SAM takes ~1900 ms for two
words and espeak-ng wants PSRAM this board has not got -- so every line it will
ever speak has to be rendered here, on the Mac, and shipped as audio. That makes
"which voice, at what prosody" a decision taken once and then frozen into the
build, which is why it is worth auditioning properly rather than guessing.

    uv run tools/voice_audition.py corpus-voice/          # the shortlist
    uv run tools/voice_audition.py corpus-voice/ --budget # + corpus sizing

`corpus-voice/` is already gitignored by the `corpus*/` rule.

## What `say` actually honours, measured on this Mac

Not what the documentation implies. Every row below was established by rendering
the same line twice and comparing MD5s -- an ignored command leaves the output
byte-identical, which is the only way to tell it apart from one that worked.

    [[pbas n]]  WORKS.  n=44 is neutral for every voice tried; ~1.25 semitones
                per unit; clamps at the bottom around n=34 (Allison floors at
                101 Hz, Samantha at 90 Hz, and 30 sounds the same as 34).
    [[rate n]]  WORKS, and so does the -r flag, but see SLOW_IS_A_LIE below.
    [[volm f]]  WORKS.
    [[slnc n]]  WORKS. The only control that buys an arbitrary pause.
    [[pmod n]]  IGNORED. Byte-identical output at 0 and at 150, on Allison
                (Enhanced), Samantha AND Fred -- so this is not a
                modern-voice-only regression, it is gone across the board.
    [[emph +]]  IGNORED, same test.
    [[inpt PHON]] parsed and stripped, then the phoneme string is read out as
                letters. Worse than useless.

So pitch, rate, volume and silence are the whole toolbox. Pitch *modulation* --
the obvious lever for a flatter, warmer line-reading -- is not available.

## SLOW_IS_A_LIE

`-r` is nearly saturated on the slow side. Measured, "Tell me more about your
family" through Allison (Enhanced):

    -r 300 -> 1.01 s        -r 140 -> 1.70 s
    -r 220 -> 1.32 s        -r 100 -> 1.89 s
    -r 160 -> 1.61 s (dflt) -r  50 -> 2.14 s

Asking for a 3.2x slowdown (160 -> 50) delivers 1.33x, and 160/180 and 100/80
are quantised to the same output. Speeding *up* tracks the request fine. So a
languid delivery cannot be bought with `-r`; it has to come from inserted
silence. Hence PAUSE_MS and the phrasing rule below.

Punctuation is a fixed-size break, not a scalable one: a comma, an ellipsis and
an em dash all produce **byte-identical** output (+340 ms on the same line).

## Rate and pitch are not independent

Worth knowing before reading a pbas number as a pitch. Same line, Allison
(Enhanced), median F0 measured by `median_f0`:

    default                          200.5 Hz
    -r 100 alone                     193.4 Hz
    [[pbas 42]] alone                172.3 Hz
    [[pbas 42]] with -r 100          148.0 Hz

Slowing down lowers the pitch on its own, and the two compound rather more than
they add (172 x 193/200 would predict 166, not 148). So the presets below are
labelled with the pbas they ask for and the F0 they actually land on, because
those are different numbers.
"""

import argparse
import math
import os
import re
import struct
import subprocess
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import eliza_rules  # noqa: E402
import vocab  # noqa: E402

RATE = 22050          # audition only; the device corpus gets resampled later
NEUTRAL_PBAS = 44     # measured: this is "no change" for every voice tried

# Where a phrase break goes. `say` gives us no prosodic parse, so this is a
# rule, not an analysis: break before the last function word that has at least
# two words on each side of it. On DOCTOR's replies -- short, one clause, almost
# always ending in a prepositional phrase -- that lands on "about your family",
# "of your mother", "when you think", which is where a person would breathe.
# It is checked by ear on the audition lines and nowhere else.
BREAK_BEFORE = ("about", "of", "when", "with", "to", "from", "in", "over",
                "why", "what", "just")


def phrase(text, pause_ms):
    """Insert one [[slnc]] at the break point. Returns text unchanged at 0."""
    if not pause_ms:
        return text
    words = text.split()
    best = None
    for i, word in enumerate(words):
        if i >= 2 and len(words) - i >= 2 and word.lower().strip(",.") in BREAK_BEFORE:
            best = i
    if best is None:
        return text
    return " ".join(words[:best]) + "[[slnc %d]] " % pause_ms + " ".join(words[best:])


class Preset:
    """One prosody setting, and the `say` invocation it turns into."""

    def __init__(self, name, pbas=None, rate=None, pause_ms=0, volm=None,
                 lead_ms=0, note=""):
        self.name, self.note = name, note
        self.pbas, self.rate, self.pause_ms, self.volm = pbas, rate, pause_ms, volm
        self.lead_ms = lead_ms

    @property
    def semitones(self):
        """Pitch shift in semitones. ~1.25 st per pbas unit, measured."""
        return 0.0 if self.pbas is None else (self.pbas - NEUTRAL_PBAS) * 1.25

    def markup(self, text):
        head = ""
        if self.lead_ms:
            head += "[[slnc %d]]" % self.lead_ms
        if self.pbas is not None:
            head += "[[pbas %d]]" % self.pbas
        if self.volm is not None:
            head += "[[volm %.2f]]" % self.volm
        return head + (" " if head else "") + phrase(text, self.pause_ms)

    def render(self, voice, text, path):
        cmd = ["say", "-v", voice, "--data-format=LEI16@%d" % RATE, "-o", path]
        if self.rate is not None:
            cmd += ["-r", str(self.rate)]
        cmd += [self.markup(text)]
        subprocess.run(cmd, check=True, capture_output=True)
        return path

    def describe(self):
        bits = []
        bits.append("pitch %+.2f st (pbas %d)" % (self.semitones, self.pbas)
                    if self.pbas is not None else "pitch default")
        bits.append("rate %s" % (self.rate if self.rate is not None else "default (160)"))
        bits.append("phrase pause %d ms" % self.pause_ms if self.pause_ms else "no pause")
        if self.volm is not None:
            bits.append("volume %.2f" % self.volm)
        if self.lead_ms:
            bits.append("lead-in %d ms" % self.lead_ms)
        return ", ".join(bits)


# The shortlist's prosody axis. p0 is the control -- without it there is nothing
# to hear the others against, and "warmer" is only meaningful as a comparison.
PRESETS = (
    Preset("p0-neutral", note="control: no prosody markup at all"),
    Preset("p1-lower", pbas=42, note="pitch only, -2.5 st"),
    Preset("p2-slow", rate=100, note="rate only -- note how little it does"),
    Preset("p3-light", pbas=43, rate=100, pause_ms=220,
           note="half p3's pitch drop, for a voice that already starts low"),
    Preset("p3-warm", pbas=42, rate=100, pause_ms=220,
           note="the core recipe: -2.5 st, slower, one breath"),
    Preset("p4-darker", pbas=40, rate=100, pause_ms=260,
           note="-5 st. Past here the formants stop tracking and it turns male"),
    Preset("p5-intimate", pbas=41, rate=60, pause_ms=320, volm=0.75, lead_ms=150,
           note="everything at once, including a lead-in breath of silence"),
)

# Voices. The download queue completed during this session; all eleven the
# user asked for are installed, and every one below is verified distinct by
# `check_distinct` before it is used.
#
# The quality suffix is load-bearing. `Samantha` and `Samantha (Enhanced)` are
# both installed and are *different* voices (170.9 vs 180.7 Hz, different
# MD5s) -- ask for the bare name and you get the compact tier, which is what
# this exercise exists to avoid. Where only one tier is installed the bare name
# resolves to it (`Zoe` and `Zoe (Premium)` are byte-identical), so the suffix
# is harmless to include and dangerous to omit. Add it always.
#
# Natural pitch, no markup, on "Tell me more about your family":
#
#     Susan (Enhanced)    162.1 Hz     Samantha (Enhanced)  180.7 Hz
#     Nicky (Enhanced)    165.8 Hz     Allison (Enhanced)   200.5 Hz
#     Ava (Premium)       172.3 Hz     Zoe (Premium)        220.5 Hz
#     Samantha (compact)  170.9 Hz     Joelle (Enhanced)    237.1 Hz
#                                      Noelle (Enhanced)    268.9 Hz  EXCLUDED
#     Tom (Enhanced)      128.9 Hz     Evan (Enhanced)      121.2 Hz
#     Nathan (Enhanced)   107.0 Hz
#
# **Noelle is excluded on the user's ear** -- "sounds like a female child".
# That is a judgement about character, not quality, and no measurement here
# would have caught it. But it does give the shortlist an axis it was missing:
# Noelle is also the highest-pitched voice installed, and Joelle (237) and Zoe
# (220) are the next two. That is a hypothesis for the user's ear rather than a
# finding -- apparent age is not a function of F0 -- so both stay in, ordered by
# pitch and flagged, instead of being quietly dropped on a correlation.
#
# Ava carries the prosody sweep on two counts: it is one of only two Premium
# voices installed, a tier above the rest; and at 172 Hz it already sits low,
# which matters because [[pbas]] moves pitch without moving formants, so the
# further a voice is dragged the less it sounds like a woman speaking low and
# the more like a slowed recording. Susan and Nicky start lower but are only
# Enhanced. Every candidate also appears at p3-warm so this can be overruled by
# ear, which is the only instrument that settles it.
PRIMARY = "Ava (Premium)"
SHORTLIST = (
    # The prosody axis, on one voice.
    (PRIMARY, ("p0-neutral", "p1-lower", "p2-slow", "p3-light", "p3-warm",
               "p4-darker", "p5-intimate")),
    # The voice axis, at one setting, ordered low to high natural pitch so the
    # apparent-age question is walked deliberately rather than stumbled into.
    ("Susan (Enhanced)", ("p3-warm",)),        # 162 Hz, Siri-generation
    ("Nicky (Enhanced)", ("p3-warm",)),        # 166 Hz
    ("Samantha (Enhanced)", ("p3-warm",)),     # 181 Hz
    ("Allison (Enhanced)", ("p3-warm",)),      # 200 Hz
    ("Zoe (Premium)", ("p3-warm",)),           # 220 Hz, Siri-generation
    ("Joelle (Enhanced)", ("p3-warm",)),       # 237 Hz -- listen for age
    # Not a candidate for what was asked for. One render so that if a male
    # option is ever wanted, the best American male voice installed is a known
    # quantity rather than another download-and-wait.
    ("Tom (Enhanced)", ("p3-warm",)),
)

# Five real replies, one of each kind the corpus has to cover. Not invented
# lines: every one is in src/eliza_rules.py, spelled as `say` wants it.
LINES = (
    ("canned-family", "Tell me more about your family"),
    ("canned-goon", "Please go on"),
    ("literal-sad", "I am sorry to hear you are sad"),
    ("literal-mother", "What else comes to mind when you think of your mother"),
    ("noun-mother", "Why do you say your mother"),
)

# The seam test, and the most consequential comparison in the file: it decides
# whether the reply corpus is 132 clips or 379, which is the difference between
# fitting the filesystem and not.
#
# Rendering "why do you say your" once and gluing a separately rendered "mother"
# onto it is what makes a 12-noun slot cost one clip instead of twelve. But a
# sentence spoken whole carries one intonation contour across the join, and two
# clips butted together carry two. Normally you would flatten the pieces to
# match with [[pmod 0]] -- which is exactly the control that turns out to be
# ignored, so there is no way to smooth a seam after the fact. What is rendered
# is what ships.
#
# Each entry is (id, head, filler, tail). A **trailing** slot -- tail empty --
# has one join; a **medial** slot has two, and its tail fragment has to begin
# mid-clause.
#
# These were included expecting trailing to be the easy case, on the reasoning
# that the pitch is falling into a sentence end there anyway. **The measurement
# says otherwise** -- trailing steps 10.6 and 10.2 semitones, the two largest
# figures recorded, against 10.3/8.9 and 0.0/7.6 for medial. The damage is done
# by the head fragment taking a terminal contour, and it does that regardless of
# what follows; a medial slot merely adds a second join rather than worsening
# the first.
#
# That kills the obvious mitigation -- expand the five medial templates whole,
# splice the rest -- so the pair is kept here as the evidence for why.
SEAMS = (
    ("medial-thinking", "Does thinking of", "mother", "bring anything else to mind"),
    ("medial-remember", "Why do you remember", "mother", "just now"),
    ("trailing-say", "Why do you say your", "mother", ""),
    ("trailing-comes", "What else comes to mind when you think of your", "mother", ""),
)

XFADE_MS = 10   # over the splice; a hard cut clicks, same reason make_clip fades

# How much audio either side of a join to take a pitch reading from. Long enough
# to hold a few pitch periods at 130 Hz, short enough not to average the jump
# away.
SEAM_PROBE_MS = 140


def read_wav(path):
    with wave.open(path) as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return rate, list(struct.unpack("<%dh" % (len(raw) // 2), raw))


def write_wav(path, rate, samples):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % len(samples), *samples))


def trim(samples, rate, floor=200, pad_ms=30):
    """Strip `say`'s leading and trailing padding.

    It matters for the budget, not for the audition: `say` pads every render,
    and counting that padding 353 times would inflate the corpus by seconds of
    silence that the device would never store.
    """
    first, last = 0, len(samples) - 1
    while first < last and abs(samples[first]) < floor:
        first += 1
    while last > first and abs(samples[last]) < floor:
        last -= 1
    pad = int(pad_ms * rate / 1000)
    return samples[max(0, first - pad):min(len(samples), last + pad)]


def splice(head, tail, rate):
    """Glue two renders with a short crossfade. The seam this is testing."""
    n = min(int(XFADE_MS * rate / 1000), len(head), len(tail))
    if not n:
        return head + tail
    mixed = [(head[len(head) - n + i] * (n - i) + tail[i] * i) // n for i in range(n)]
    return head[:len(head) - n] + mixed + tail[n:]


def assemble(pieces, rate):
    """Splice several renders in order. Returns (samples, join offsets).

    The offsets are where the joins ended up, so `seam_jumps` can go and look
    at them afterwards.
    """
    out, joins = list(pieces[0]), []
    for piece in pieces[1:]:
        joins.append(len(out) - int(XFADE_MS * rate / 1000))
        out = splice(out, list(piece), rate)
    return out, joins


def seam_jumps(samples, rate, joins):
    """Pitch discontinuity either side of each join, in semitones.

    This does **not** decide whether a seam is acceptable -- that is an ear
    judgement and nothing here substitutes for it. What it does is put a number
    on the thing the ear is being asked about, so "the splice sounds fine" and
    "the splice sounds wrong" can each be checked against what is physically
    there. A join with a 0.2-semitone step and one with a 4-semitone step are
    different problems, and without measuring you cannot tell which you have.

    A reading of 0.0 means one side was unvoiced -- a stop or a silence -- where
    there is no pitch to be discontinuous, and those joins are the ones that
    tend to survive.
    """
    probe = int(SEAM_PROBE_MS * rate / 1000)
    out = []
    for at in joins:
        before = median_f0(samples[max(0, at - probe):at], rate)
        after = median_f0(samples[at:at + probe], rate)
        if before and after:
            out.append(abs(12 * math.log(after / before, 2)))
        else:
            out.append(0.0)
    return out


def duration(path):
    with wave.open(path) as handle:
        return handle.getnframes() / handle.getframerate()


def median_f0(samples, rate, lo=70, hi=400):
    """Median pitch over voiced frames, by autocorrelation.

    A changed MD5 proves `say` did *something*; it does not prove it lowered
    the pitch. This is what turned "[[pbas]] works" into the numbers in the
    module docstring, and it is what catches a setting that reads as a big
    change on paper and is inaudible -- or one that has silently clamped.
    """
    window, hop = int(0.040 * rate), int(0.020 * rate)
    lag_lo, lag_hi = rate // hi, rate // lo
    found = []
    for start in range(0, max(0, len(samples) - window), hop):
        frame = samples[start:start + window]
        if sum(v * v for v in frame) / window < 2_000_000:
            continue                                   # silence or unvoiced
        zero = sum(v * v for v in frame)
        best, best_r = 0, 0.0
        for lag in range(lag_lo, lag_hi):
            # Stride 2: halves the cost and does not move the peak, which is
            # broad at these lags.
            r = sum(frame[i] * frame[i + lag] for i in range(0, window - lag, 2))
            r /= (zero * (window - lag) / window + 1)
            if r > best_r:
                best_r, best = r, lag
        if best and best_r > 0.30:
            found.append(rate / best)
    found.sort()
    return found[len(found) // 2] if found else 0.0


def check_distinct(voices):
    """Assert every voice renders differently.

    `say -v <name>` renders an uninstalled voice in the system default and
    exits 0 -- see .serena/memories/gotchas_that_cost_time.md, where four
    "speakers" in a corpus shared an MD5. An audition of nine identical files
    is worse than no audition, because it looks like a result.
    """
    import hashlib
    import tempfile
    seen, bad = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        probe = os.path.join(tmp, "probe.wav")
        for voice in voices:
            # Not `-o /dev/stdout`: `say` writes nothing there and exits 0, which
            # would make every voice hash the empty string and fail this check
            # for the wrong reason -- the same class of silent no-op it is here
            # to catch.
            subprocess.run(["say", "-v", voice, "--data-format=LEI16@%d" % RATE,
                            "-o", probe, "tell me more about your family"],
                           check=True, capture_output=True)
            with open(probe, "rb") as handle:
                digest = hashlib.md5(handle.read()).hexdigest()
            if digest in seen:
                bad.append((voice, seen[digest]))
            seen[digest] = voice
    return bad


def build_shortlist(outdir):
    audio = os.path.join(outdir, "audition")
    os.makedirs(audio, exist_ok=True)
    presets = {p.name: p for p in PRESETS}
    combos = []

    for voice, names in SHORTLIST:
        for name in names:
            preset = presets[name]
            slug = "%s__%s" % (re.sub(r"\W+", "-", voice).strip("-").lower(), name)
            parts, total, pitch = [], 0.0, 0.0
            for line_id, text in LINES:
                path = os.path.join(audio, "%s__%s.wav" % (slug, line_id))
                preset.render(voice, text, path)
                total += duration(path)
                rate, samples = read_wav(path)
                parts.append(trim(samples, rate))
                if line_id == LINES[0][0]:
                    pitch = median_f0(parts[-1], rate)
            # One file per combination, so the audition is 15 plays not 75.
            gap = [0] * int(0.55 * RATE)
            reel = []
            for part in parts:
                reel += part + gap
            write_wav(os.path.join(audio, "%s__REEL.wav" % slug), RATE, reel)
            combos.append((slug, voice, preset, total, pitch))
            print("  %-44s %5.2f s  F0 %5.1f Hz" % (slug, total, pitch))

    # The seam, in the leading preset only -- one comparison, not a matrix.
    warm = presets["p3-warm"]
    seams = []
    for seam_id, head_text, filler, tail_text in SEAMS:
        texts = [t for t in (head_text, filler, tail_text) if t]
        rendered = []
        for i, text in enumerate(texts):
            part = os.path.join(audio, "seam__%s__part%d.wav" % (seam_id, i))
            warm.render(PRIMARY, text, part)
            rate, samples = read_wav(part)
            rendered.append(trim(samples, rate))

        joined, joins = assemble(rendered, rate)
        write_wav(os.path.join(audio, "seam__%s__ASSEMBLED.wav" % seam_id), rate, joined)

        whole_path = os.path.join(audio, "seam__%s__WHOLE.wav" % seam_id)
        warm.render(PRIMARY, " ".join(texts), whole_path)
        _, whole_s = read_wav(whole_path)
        whole_s = trim(whole_s, rate)

        # The same measurement on the whole rendering, at the point the join
        # would have fallen, scaled for the two clips being different lengths.
        # Without this control a 1.5-semitone step at a splice looks damning
        # when the intact sentence steps by nearly as much at the same word.
        scale = len(whole_s) / max(1, len(joined))
        control = seam_jumps(whole_s, rate, [int(at * scale) for at in joins])

        seams.append((seam_id, texts, seam_jumps(joined, rate, joins), control,
                      len(joined) / rate, len(whole_s) / rate))
        print("  seam %-22s %d join(s)  assembled %s  vs whole %s"
              % (seam_id, len(joins),
                 "/".join("%.1f" % v for v in seams[-1][2]),
                 "/".join("%.1f" % v for v in control)))

    return combos, seams


def corpus_lines():
    """Every line the device would have to ship, by how it gets built.

    Returns (canned, whole, pieces, medial).

    `canned` is the 79 replies with no slot at all, plus the four deflections
    and the greeting. Those ship as whole sentences under every plan, because
    there is nothing in them to assemble.

    `whole` is the fully expanded corpus -- `canned` plus every slotted
    template with every filler it can reach, each combination its own clip, no
    seams anywhere. `pieces` is what those slotted templates cost instead if
    they are assembled: the stems, plus the filler words spliced onto them.
    The shipping corpus is therefore `canned` + `whole` or `canned` + `pieces`,
    never `pieces` alone.

    `medial` is the part that spoils the neat version of the story. Not every
    slot is sentence-final: "DOES THINKING OF _ BRING ANYTHING ELSE TO MIND"
    and "WHAT MAKES YOU _ JUST NOW" put the user's word in the middle, so
    assembling them needs *two* splices and a fragment that trails off in the
    middle of a phrase. Those are the templates where the seam is most likely
    to be audible, and they are counted separately for that reason.
    """
    canned, literal, noun = [], [], []
    for entry in eliza_rules.RULES.values():
        if not isinstance(entry, tuple) or len(entry) != 3 or not entry[2]:
            continue
        for _, reasm in entry[2]:
            for kind, text in reasm:
                if kind == eliza_rules.CANNED:
                    canned.append(text)
                elif kind == eliza_rules.LITERAL:
                    literal.append(text)
                elif kind == eliza_rules.NOUN:
                    noun.append(text)

    canned = sorted(set(canned)) + list(eliza_rules.NONE) + [eliza_rules.GREETING]

    # Fillers the spotter can actually deliver. A slot the recogniser has no
    # word for is a slot the template never reaches, so it costs nothing.
    nouns = [w.lower() for w in vocab.NOUNS]
    feelings = [w.lower() for w in vocab.FEELINGS]

    canned = list(canned)
    whole, pieces, medial = list(canned), [], []

    def expand(text, fill):
        slotted = re.sub(r"\d", "%s", text)
        for word in fill:
            whole.append(slotted % word)
        head, _, tail = slotted.partition("%s")
        pieces.append(head.strip())
        if tail.strip():
            pieces.append(tail.strip())
            medial.append(text)

    for text in sorted(set(literal)) + sorted(set(noun)):
        if not re.search(r"\d", text):
            whole.append(text)
            continue
        # Which filler set the slot draws from, by what the template says about
        # it. The feeling branches are the only closed ones; everything else
        # takes a noun.
        if any(k in text for k in ("SORRY TO HEAR", "NOT TO BE", "PLEASANT TO BE",
                                   "MADE YOU", "HELPED YOU TO BE", "TREATMENT MADE",
                                   "MAKES YOU", "SUDDENLY")):
            expand(text, feelings)
        elif text.startswith("REALLY,") and "MY" not in text:
            continue          # EVERYONE / NOBODY -- not in the spotter's vocabulary
        elif "SURELY NOT" in text:
            continue
        else:
            expand(text, nouns)

    # MEMORY lines take a noun in the same way.
    for _, text in eliza_rules.MEMORY:
        expand(text, nouns)

    pieces = sorted(set(p for p in pieces if p)) + nouns + feelings
    return canned, whole, pieces, medial


# Bytes per second of mono audio, by candidate on-device format. The first row
# is what clips/ uses today (tools/make_clip.py packs 24 kHz stereo, left and
# right identical, because that is what the audio PIO consumes without decoding).
FORMATS = (
    ("24 kHz packed stereo (today's clip format)", 96000),
    ("16 kHz mono int16", 32000),
    ("16 kHz IMA ADPCM 4-bit", 8000),
    ("8 kHz mono int16", 16000),
    ("8 kHz IMA ADPCM 4-bit", 4000),
)
FILESYSTEM = 3 * 1024 * 1024


def budget(outdir, voice, preset):
    """Render the corpus once and report what it actually costs.

    Every second is measured, not estimated. The output is a *curve* rather
    than a single total, because the seam measurement (see `seam_jumps`) makes
    the small spliced corpus look unusable, and the question then becomes not
    "does the whole corpus fit" -- it does not -- but "how much of it fits".

    The curve is over vocabulary depth. Canned replies have no slot and always
    ship. Each additional noun the device can speak lights up every slotted
    template at once, at a roughly constant cost per noun, and a template that
    has no clip for the noun it heard can fall through to a deflection exactly
    as an unrecognised word already does. So the corpus can be truncated
    anywhere along this curve without the program breaking -- it just gets
    shyer, which this project has already decided is the good failure.
    """
    canned, whole, pieces, medial = corpus_lines()
    scratch = os.path.join(outdir, "budget-scratch")
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, "b.wav")
    done = [0]

    def seconds(text):
        # .capitalize() only so `say` does not read an all-caps line as an
        # initialism. It has no effect on duration, which is all we want.
        preset.render(voice, text.capitalize(), path)
        rate, samples = read_wav(path)
        done[0] += 1
        if done[0] % 50 == 0:
            print("    rendered %d" % done[0], flush=True)
        return len(trim(samples, rate)) / rate

    canned_s = sum(seconds(text) for text in canned)
    pieces_s = sum(seconds(text) for text in pieces)

    # Attribute every expanded sentence to the filler word it used, so the
    # curve can be summed without re-rendering anything.
    nouns = [w.lower() for w in vocab.NOUNS]
    feelings = [w.lower() for w in vocab.FEELINGS]
    by_word = {word: 0.0 for word in nouns + feelings}
    count = {word: 0 for word in nouns + feelings}
    canned_set = set(canned)
    for text in whole:
        if text in canned_set:
            continue
        last = text.rsplit(" ", 1)[-1].lower()
        word = last if last in by_word else next(
            (w for w in by_word if (" %s " % w) in text.lower()), None)
        word = word if word else nouns[0]
        by_word[word] += seconds(text)
        count[word] += 1

    # Feelings are not optional in the same way -- there are only four and they
    # gate the whole "I am sorry to hear you are ___" branch -- so they are
    # charged into the base rather than spread along the curve.
    base_s = canned_s + sum(by_word[w] for w in feelings)
    base_n = len(canned) + sum(count[w] for w in feelings)
    curve, running, tally = [], base_s, base_n
    for i, word in enumerate(nouns):
        running += by_word[word]
        tally += count[word]
        curve.append((i + 1, word, tally, running))
    # The spliced corpus still ships the canned replies whole -- there is
    # nothing in them to assemble -- so it is canned PLUS pieces, never pieces
    # alone. Counting only the pieces understates it by 83 clips.
    return ((len(canned), canned_s), (base_n, base_s), curve,
            (len(canned) + len(pieces), canned_s + pieces_s), len(medial))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--budget", action="store_true",
                    help="also render the full corpus and measure what it costs")
    args = ap.parse_args()

    voices = sorted({v for v, _ in SHORTLIST})
    clashes = check_distinct(voices)
    if clashes:
        print("REFUSING: these voices render identically, so at least one is not")
        print("installed and `say` is substituting the default:")
        for a, b in clashes:
            print("    %s == %s" % (a, b))
        return 1
    print("%d voices, all distinct." % len(voices))

    os.makedirs(args.outdir, exist_ok=True)
    print("rendering shortlist...")
    combos, seams = build_shortlist(args.outdir)

    lines = ["# Voice audition", "",
             "Rendered by `tools/voice_audition.py`. Play a `__REEL.wav` to hear",
             "all five lines in one go; the per-line files sit beside it.", "",
             "    afplay audition/<name>__REEL.wav", "",
             "The presets are built from the only four `say` controls that turned",
             "out to work here — pitch base, rate, volume and inserted silence.",
             "`[[pmod]]` and `[[emph]]` render byte-identical output at every value",
             "and are not used. `-r` barely slows these voices down (a 3.2x request",
             "buys 1.33x), so the pacing comes from the inserted pause, not the rate.",
             "The module docstring has the measurements.", "",
             "## Combinations", "",
             "Median F0 is measured on the first line, not assumed from the",
             "pbas number -- see `median_f0`.", "",
             "| Reel | Voice | Preset | Median F0 | Setting | Note |",
             "| --- | --- | --- | --- | --- | --- |"]
    for slug, voice, preset, _, pitch in combos:
        lines.append("| `%s__REEL.wav` | %s | %s | %.0f Hz | %s | %s |"
                     % (slug, voice, preset.name, pitch, preset.describe(), preset.note))
    lines += ["", "## Lines", ""]
    for line_id, text in LINES:
        lines.append("- `%s` — \"%s\"" % (line_id, text))
    lines += ["", "## The seam — the one comparison that decides the budget", "",
              "`__ASSEMBLED` is the sentence built from separately rendered pieces",
              "and spliced; `__WHOLE` is the same sentence rendered in one pass.",
              "Same voice, same preset. Play them back to back.", "",
              "`medial-*` have the slot mid-sentence and two joins; `trailing-*`",
              "have it at the end and one. Trailing was expected to be the easy",
              "case and **is not** — it produces the two largest steps in the table,",
              "because the damage comes from the head fragment taking a terminal",
              "contour and it does that either way.", "",
              "The step column is the pitch discontinuity measured either side of",
              "each join, in semitones, against the same measurement taken on the",
              "intact sentence at the same point. It does **not** say whether a",
              "seam is acceptable — only an ear does that — it says how big the",
              "thing being judged is. A 0.0 means one side was unvoiced, where",
              "there is no pitch to be discontinuous.", "",
              "| Files | Sentence | Joins | Step, assembled | Step, whole |",
              "| --- | --- | --- | --- | --- |"]
    for seam_id, texts, jumps, control, _, _ in seams:
        lines.append("| `seam__%s__{ASSEMBLED,WHOLE}.wav` | %s | %d | %s | %s |"
                     % (seam_id, " / ".join(texts), len(jumps),
                        ", ".join("%.1f st" % v for v in jumps),
                        ", ".join("%.1f st" % v for v in control)))

    if args.budget:
        preset = {p.name: p for p in PRESETS}["p3-warm"]
        print("measuring corpus budget for %s / p3-warm..." % PRIMARY)
        (n_canned, canned_s), (base_n, base_s), curve, (n_pieces, pieces_s), \
            n_medial = budget(args.outdir, PRIMARY, preset)

        def mb(seconds, bps):
            return seconds * bps / 1024 / 1024

        lines += ["", "## What fits — %s, p3-warm" % PRIMARY, "",
                  "Measured, not estimated: every line rendered and its `say`",
                  "padding trimmed before counting. Sizes are 4-bit IMA ADPCM,",
                  "which is the only format the corpus fits in at all (see below).",
                  "",
                  "The whole corpus does not fit. But it does not have to: a",
                  "template with no clip for the noun it heard can fall through to",
                  "a deflection, exactly as an unrecognised word already does, so",
                  "the corpus can be truncated anywhere on this curve and the",
                  "program still works — it just gets shyer.", "",
                  "| Vocabulary | Clips | Seconds | 16 kHz | 8 kHz |",
                  "| --- | --- | --- | --- | --- |",
                  "| canned replies only | %d | %.0f | %.2f MB | %.2f MB |"
                  % (n_canned, canned_s, mb(canned_s, 8000), mb(canned_s, 4000)),
                  "| + the 4 feelings | %d | %.0f | %.2f MB | %.2f MB |"
                  % (base_n, base_s, mb(base_s, 8000), mb(base_s, 4000))]
        for i, word, clips, seconds in curve:
            lines.append("| + %d noun%s (…%s) | %d | %.0f | %.2f MB | %.2f MB |"
                         % (i, "" if i == 1 else "s", word, clips, seconds,
                            mb(seconds, 8000), mb(seconds, 4000)))
        full = curve[-1][3]
        lines += ["", "Against a **3 MB filesystem**, and that is before the code,",
                  "the DTW templates and the existing shake/fart/laugh clips — so",
                  "the usable ceiling is well under 3 MB, not at it.", "",
                  "**The full noun set does not fit at any format.** At 8 kHz it is",
                  "3.38 MB and at 16 kHz 6.76 MB. Leaving ~1 MB for everything else,",
                  "the practical stop is around **7 nouns at 8 kHz** or **2 at",
                  "16 kHz** — which is the real choice: a wider vocabulary at",
                  "telephone quality, or a narrower one that sounds better.", "",
                  "### Why not the other formats", "",
                  "| Format | canned only | full corpus |", "| --- | --- | --- |"]
        for label, bps in FORMATS:
            def cell(seconds):
                size = seconds * bps
                return "%.2f MB%s" % (size / 1024 / 1024,
                                      "" if size <= FILESYSTEM else " **over**")
            lines.append("| %s | %s | %s |" % (label, cell(canned_s), cell(full)))
        lines += ["", "### The spliced alternative", "",
                  "Rendering the slotted templates as stems plus filler words and",
                  "joining them on the device — the canned replies still ship whole —",
                  "costs **%d clips, %.0f s, %.2f MB at 16 kHz**, which fits"
                  % (n_pieces, pieces_s, mb(pieces_s, 8000)),
                  "easily. The seam measurement above is why it is not the",
                  "recommendation; see the audition files before trusting either.",
                  "",
                  "%d of the slotted templates put the slot mid-sentence, needing"
                  % n_medial,
                  "two joins rather than one.", ""]

    with open(os.path.join(args.outdir, "INDEX.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote %s" % os.path.join(args.outdir, "INDEX.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
