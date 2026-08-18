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

# Voices.
#
# The `(Enhanced)` suffix is load-bearing: `Samantha` and `Samantha (Enhanced)`
# are both installed here and are different voices (170.9 Hz vs 180.7 Hz, and
# they do not share an MD5). Ask for the bare name and you get the compact one,
# which is the tier this exercise exists to avoid. `say -v` also renders an
# *uninstalled* name in the system default and exits 0, so `check_distinct`
# runs before anything else.
#
# Susan carries the prosody sweep because it has the lowest natural pitch of
# any female voice installed -- measured on "Tell me more about your family",
# no markup:
#
#     Susan (Enhanced)    162.1 Hz      Joelle (Enhanced)    237.1 Hz
#     Samantha (Enhanced) 180.7 Hz      Noelle (Enhanced)    268.9 Hz
#     Allison (Enhanced)  200.5 Hz      (Nathan, male        107.0 Hz)
#
# That matters because [[pbas]] is the control that costs naturalness -- it
# shifts pitch without moving the formants with it, so the further a voice is
# dragged the more it stops sounding like a woman speaking low and starts
# sounding like a recording slowed down. A voice that starts in the target
# register needs the least of it. This is a stated criterion, not a verdict:
# every candidate also appears at p3-warm so it can be overruled by ear.
PRIMARY = "Susan (Enhanced)"
SHORTLIST = (
    (PRIMARY, ("p0-neutral", "p1-lower", "p2-slow", "p3-light", "p3-warm",
               "p4-darker", "p5-intimate")),
    ("Allison (Enhanced)", ("p0-neutral", "p3-warm")),
    ("Samantha (Enhanced)", ("p3-warm",)),
    ("Joelle (Enhanced)", ("p3-warm",)),
    ("Noelle (Enhanced)", ("p3-warm",)),
    ("Moira", ("p3-warm",)),       # en_IE, for an accent that is not American
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

# The seam test. Rendering "why do you say your" once and gluing a separately
# rendered "mother" onto it is what makes a 12-noun slot cost 1 clip instead of
# 12 -- but a sentence spoken whole carries an intonation contour across that
# join, and two clips spliced together do not. There is no measurement that
# settles whether it matters; that is what ears are for, so both go in the
# shortlist side by side.
SEAMS = (
    ("noun-mother", "Why do you say your", "mother"),
    ("literal-mother", "What else comes to mind when you think of your", "mother"),
)

XFADE_MS = 10   # over the splice; a hard cut clicks, same reason make_clip fades


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
    for seam_id, head_text, tail_text in SEAMS:
        head = os.path.join(audio, "seam__%s__head.wav" % seam_id)
        tail = os.path.join(audio, "seam__%s__tail.wav" % seam_id)
        warm.render(PRIMARY, head_text, head)
        warm.render(PRIMARY, tail_text, tail)
        rate, head_s = read_wav(head)
        _, tail_s = read_wav(tail)
        joined = splice(trim(head_s, rate), trim(tail_s, rate), rate)
        write_wav(os.path.join(audio, "seam__%s__ASSEMBLED.wav" % seam_id), rate, joined)
        whole = os.path.join(audio, "seam__%s__WHOLE.wav" % seam_id)
        warm.render(PRIMARY, head_text + " " + tail_text, whole)
        print("  seam %-39s assembled vs whole" % seam_id)

    return combos


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
    """Render the whole corpus once and report what it actually costs.

    Every second here is measured, not estimated: the point of the exercise is
    to replace an arithmetic guess about corpus size with a real total for a
    real voice at a real prosody. Rendering ~400 lines takes a couple of
    minutes and settles it.
    """
    canned, whole, pieces, medial = corpus_lines()
    scratch = os.path.join(outdir, "budget-scratch")
    os.makedirs(scratch, exist_ok=True)

    def total_seconds(lines, tag):
        seconds = 0.0
        path = os.path.join(scratch, "%s.wav" % tag)
        for i, text in enumerate(lines):
            # .capitalize() only so `say` does not read an all-caps line as an
            # initialism. It has no effect on duration, which is all we want.
            preset.render(voice, text.capitalize(), path)
            rate, samples = read_wav(path)
            seconds += len(trim(samples, rate)) / rate
            if i % 50 == 0:
                print("    %s %d/%d" % (tag, i, len(lines)), flush=True)
        return seconds

    # `whole` already contains `canned`, so the canned pass is charged once and
    # subtracted rather than rendered twice.
    canned_s = total_seconds(canned, "canned")
    slotted = [line for line in whole if line not in set(canned)]
    slotted_s = total_seconds(slotted, "slotted")
    pieces_s = total_seconds(pieces, "pieces")
    return ((len(canned), canned_s),
            (len(whole), canned_s + slotted_s),
            (len(canned) + len(pieces), canned_s + pieces_s),
            len(medial))


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
    combos = build_shortlist(args.outdir)

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
    lines += ["", "## The seam", "",
              "`__ASSEMBLED` is a stem and a noun rendered separately and spliced;",
              "`__WHOLE` is the same sentence rendered in one pass. Same voice,",
              "same preset. The difference is the intonation contour across the",
              "join, and whether it matters is a judgement for ears, not a",
              "measurement.", ""]
    for seam_id, head, tail in SEAMS:
        lines.append("- `seam__%s__*.wav` — \"%s\" + \"%s\"" % (seam_id, head, tail))

    if args.budget:
        preset = {p.name: p for p in PRESETS}["p3-warm"]
        print("measuring corpus budget for %s / p3-warm..." % PRIMARY)
        (n_canned, s_canned), (n_whole, s_whole), (n_asm, s_asm), n_medial = \
            budget(args.outdir, PRIMARY, preset)
        lines += ["", "## Corpus budget — %s, p3-warm" % PRIMARY, "",
                  "Measured, not estimated: every line below was rendered and its",
                  "`say` padding trimmed before counting.", "",
                  "**canned** is the slotless replies alone -- the floor, and a",
                  "device that shipped only these would still hold a conversation,",
                  "because a reply it cannot speak can fall through to a deflection",
                  "exactly as an unrecognised word already does. **whole** adds every",
                  "slotted template with every filler, each its own clip, spliced",
                  "nowhere. **assembled** adds them as stems plus filler words,",
                  "joined on the device.", "",
                  "| | Clips | Seconds |", "| --- | --- | --- |",
                  "| canned only | %d | %.1f |" % (n_canned, s_canned),
                  "| + slots, whole | %d | %.1f |" % (n_whole, s_whole),
                  "| + slots, assembled | %d | %.1f |" % (n_asm, s_asm), "",
                  "| Format | canned | whole | assembled |",
                  "| --- | --- | --- | --- |"]
        for label, bps in FORMATS:
            def cell(seconds):
                size = seconds * bps
                mark = "" if size <= FILESYSTEM else " **over**"
                return "%.2f MB%s" % (size / 1024 / 1024, mark)
            lines.append("| %s | %s | %s | %s |"
                         % (label, cell(s_canned), cell(s_whole), cell(s_asm)))
        lines += ["", "Against a %d MB filesystem, and that is before the code, the"
                  % (FILESYSTEM // 1024 // 1024),
                  "DTW templates and the existing shake/fart/laugh clips.", "",
                  "%d of the slotted templates put the slot in the *middle* of the"
                  % n_medial,
                  "sentence rather than at the end, so assembling them needs two",
                  "splices and a fragment that stops mid-phrase. If the seam turns",
                  "out to be audible, those are the ones to expand whole and the",
                  "sentence-final ones to keep assembled.", ""]

    with open(os.path.join(args.outdir, "INDEX.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote %s" % os.path.join(args.outdir, "INDEX.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
