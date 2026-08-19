#!/usr/bin/env python3
"""Render everything DOCTOR can say, encode it, and pack it into one file.

    uv run tools/voice_pak.py corpus-voice/                    # the whole pak
    uv run tools/voice_pak.py corpus-voice/ --only-audition    # one clip, first
    uv run tools/voice_pak.py corpus-voice/ --only "Please go on."
    uv run tools/voice_pak.py corpus-voice/ --fixtures         # + decoder fixtures

Output is gitignored by the existing `corpus*/` rule. Regenerating it is a
`say` run and a couple of minutes; nothing here is hand-edited.

The board cannot synthesise speech (docs/speech-voice.md costs SAM and
espeak-ng and rules both out), so every line it will ever speak is rendered
here and shipped as audio. `talk.Conversation._clip_for` looks a clip up by
`sha1(reply_text)[:8]`, which makes the *exact* reply text the primary key --
one different comma and the device plays nothing. Hence `corpus()` below,
which does not spell any reply out.

## How many clips there are, and why the docs said 379

docs/speech-voice.md sizes the corpus at 379 sentences and concludes it cannot
fit. That number counts every reply template in `eliza_rules` crossed with
every filler it grammatically accepts. **The device cannot reach most of
them.** `talk.Conversation.reply` hands the engine a bag of exactly one spotted
word, and `Doctor._answer` walks the ranked keywords and returns the *first*
rule that produces a reply -- so for a bag of {MOTHER} the family rules answer
and the twenty-odd other templates that would also accept "mother" are never
consulted.

Enumerated properly, and counting every rotation state rather than a sample:

    111  replies reachable from a one-word bag       (`corpus()`)
    +2   the greeting and NOTHING_HEARD, which talk.py speaks directly
    ---
    113  clips  (measured: 234 s, 1.79 MB at 4-bit 16 kHz ADPCM)

Of those 111, only 68 were observed in a 300,000-turn random walk over the
engine; the other 43 need a rule-rotation state the walk never produced but
that nothing forbids. They are shipped anyway. A clip nobody plays costs
~16 KB of a 15 MB filesystem; a *missing* clip is a reply the toy mouths
silently, and there is no way to tell that apart from a decoder bug at the
desk. So `corpus()` returns the provable superset, and
`tools/test_voice_pak.py` asserts the observed set is inside it.

The consequence is that the budget question docs/speech-voice.md leaves open --
"a wider vocabulary at telephone quality, or a narrower one that sounds
better" -- is not a real choice. The whole corpus fits at 16 kHz with 13 MB
spare, and would fit uncompressed too (7.5 MB). ADPCM is kept because the
streaming decoder is already being built for it and 4x fewer flash reads is
free headroom, not because anything is tight.

## The recipe is not restated here

Voice, pitch and pause come from `tools/voice_audition.py` -- `PRESETS`
["p3-warm"] on `PRIMARY` -- because that is where they were measured and a
second copy would drift from it silently. Only what this file changes is
declared below: 16 kHz instead of the audition's 22050, and peak 15000.

## Peak 15000, and why not 30000

`tools/make_clip.py` normalises to 30000 and the laugh clip then had to be cut
by 6 dB by hand, because digital headroom is not analogue headroom: this amp
and speaker overdrive well below full scale. 15000 is the level the six 8 kHz
stopgap clips were bench-checked at, alongside codec volume 82 (90 overdrives;
`listen.speak` records both). Changing either is a bench measurement, not an
edit.

## Verifying this reached the glass, or rather the speaker

Per CLAUDE.md: nothing here has been played through the board. `say` exiting 0
proves a file was written and nothing about what a person hears, exactly as an
unpowered panel accepts SPI writes. The audition clip exists so a human settles
it before 113 renders are trusted -- `--only-audition` writes both the encoded
blob and a decoded WAV so `afplay` and the device can be compared to each
other, not just to expectation.
"""

import argparse
import hashlib
import os
import struct
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import eliza                # noqa: E402
import eliza_rules as rules # noqa: E402
import vocab                # noqa: E402
import voice_audition       # noqa: E402

RATE = 16000
PEAK = 15000
PRESET = "p3-warm"

# Spoken by talk.py itself rather than by the engine, so no sweep of the rule
# data will ever find them. Imported by value rather than by importing talk,
# which pulls in `board`, `epaper` and `listen` and cannot load on the host;
# tools/test_voice_pak.py imports talk under stubs and asserts these match.
GREETING_SOURCE = rules.GREETING
NOTHING_HEARD = "I did not hear anything. Hold the screen and speak."

# The one the user auditions before the other 112 are trusted. Chosen because
# it is the reply the whole voice investigation was measured on -- every F0
# figure in docs/speech-voice.md is this sentence -- so a surprise here is a
# surprise against a known number rather than against an impression.
AUDITION = "Tell me more about your family."


# --- what the device can actually say ---------------------------------------

def clip_id(text):
    """The name the device will look this clip up by.

    Must stay byte-identical to `talk.Conversation._clip_for`, which builds
    `say_<id>.pcmw` from `binascii.hexlify(sha1(text.encode()).digest())[:8]`.
    MicroPython's hashlib has no `hexdigest`, which is why _clip_for spells it
    the long way; the result is the same lowercase hex.
    """
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def corpus():
    """Every reply the device can put on the panel, as `_present` spells it.

    Returns a sorted list of (text, moods) where `moods` is the set of mood
    tags the reply's source templates carried -- QUESTION, STATEMENT or ECHO.
    The moods are recorded rather than acted on: the audition settled on one
    prosody for everything, and a second recipe is a second thing to A/B at
    the bench. They are here so that decision can be revisited from data.

    ## Why this is a superset and not a trace

    A trace -- call the engine a lot and collect what falls out -- can only
    ever report what it happened to see, and the engine rotates through each
    rule's replies on a per-(keyword, decomposition) counter that a caller
    cannot address. So this walks the rule data the way `Doctor._answer` walks
    it, but takes **every** branch instead of the first that answers and
    **every** template in `_usable` instead of the one at the current rotation
    position. That over-counts (43 of the 111 have no reachable rotation
    state) and cannot under-count, which is the direction that matters.

    Everything that decides *what a reply says* is the engine's own code --
    `_rank`, `_assumable`, `_captured_from_spotted`, `_usable`, `fill`,
    `sentence_case`. Only the control flow is re-expressed. That is the part
    tools/test_voice_pak.py pins, by asserting a long random walk over the real
    `talk.Conversation` produces nothing this did not predict.
    """
    doctor = eliza.Doctor(priority=vocab.PRIORITY)
    found = {}

    def note(text, mood):
        found.setdefault(text, set()).add(mood)

    # The bag is always zero or one word: `talk.Conversation.reply` appends at
    # most one spotted label. That is the fact that collapses 379 to 111.
    for label in list(vocab.LABELS) + [None]:
        spotted = [] if label is None else [vocab.ECHO[label].upper()]
        swapped = [rules.SUBS.get(w, w) for w in spotted]
        available = [rules.SUBS.get(w, w) for w in spotted if w in vocab.NOUNS]

        keywords = doctor._rank(spotted)
        if doctor._heard_content(spotted, vocab.NOUNS):
            for keyword in doctor._assumable(swapped, available):
                if keyword not in keywords:
                    keywords.append(keyword)

        # (keyword, words) rather than keyword alone: a PRE template rewrites
        # the word list before handing control on, and the rewritten list is
        # what the next rule's decomposition sees.
        pending = [(k, swapped) for k in keywords]
        seen = set()
        while pending:
            keyword, words = pending.pop(0)
            if (keyword, tuple(words)) in seen:
                continue
            seen.add((keyword, tuple(words)))
            entry = rules.RULES.get(keyword)
            if entry is None:
                continue
            _rank, goto, ruleset = entry
            if goto:
                pending.append((goto, words))
                continue
            for pattern, templates in ruleset:
                captured = doctor._captured_from_spotted(pattern, words, available)
                if captured is None:
                    continue
                # `_usable` drops the mood tag, and it also *filters*: control
                # kinds always survive, PHRASE never does on this path, and a
                # NOUN template is dropped when there is no noun to plant. So
                # the mood list has to be filtered identically rather than
                # taken whole -- zipping the unfiltered moods against the
                # filtered templates mislabels every reply after the first
                # dropped one. The assert below is what would catch that, and
                # what will catch `_usable` changing under this.
                usable = doctor._usable(templates, available, captured)
                moods = []
                for kind, mood, _payload in templates:
                    if kind in (rules.GOTO, rules.NEWKEY, rules.PRE):
                        moods.append(mood)
                    elif kind not in rules.SPOTTABLE:
                        continue
                    elif kind == rules.NOUN and not available:
                        continue
                    else:
                        moods.append(mood)
                if len(moods) != len(usable):
                    raise AssertionError(
                        "eliza._usable filtered %d templates where this "
                        "predicted %d -- its kind filter changed; update "
                        "corpus()" % (len(usable), len(moods)))
                for (kind, payload), mood in zip(usable, moods):
                    if kind == rules.NEWKEY:
                        continue
                    if kind == rules.GOTO:
                        pending.append((payload, words))
                        continue
                    if kind == rules.PRE:
                        rewrite, target = payload
                        pending.append((target, eliza.fill(rewrite, captured).split()))
                        continue
                    note(eliza.sentence_case(eliza.fill(payload, captured)), mood)

    # `_give_up`'s deflections. Reachable from any bag that matches nothing,
    # and the MEMORY queue that would otherwise pre-empt them cannot fill on
    # this path -- `_remember` is only called from `respond_to_words`, which
    # talk.py never calls. tools/test_talk.py pins that.
    for text in rules.NONE:
        note(eliza.sentence_case(text), rules.STATEMENT)

    # Spoken by talk.py, not by the engine.
    note(eliza.sentence_case(GREETING_SOURCE), rules.STATEMENT)
    note(NOTHING_HEARD, rules.STATEMENT)

    return sorted((text, "".join(sorted(moods))) for text, moods in found.items())


# --- rendering ---------------------------------------------------------------

def render(text, path, voice=None, preset=None):
    """One line through `say`, at the settled recipe, trimmed and normalised.

    Returns the sample list. The markup comes from the audition's preset object
    so pitch, pause and volume have exactly one home; only the rate flag and
    the data format are spelled here, because those are what this file changes.
    """
    voice = voice or voice_audition.PRIMARY
    preset = preset or {p.name: p for p in voice_audition.PRESETS}[PRESET]

    raw = path + ".say.wav"
    cmd = ["say", "-v", voice, "--data-format=LEI16@%d" % RATE, "-o", raw]
    if preset.rate is not None:
        cmd += ["-r", str(preset.rate)]
    cmd += [preset.markup(text)]
    subprocess.run(cmd, check=True, capture_output=True)

    got_rate, samples = voice_audition.read_wav(raw)
    if got_rate != RATE:
        raise AssertionError(
            "say returned %d Hz, asked for %d -- --data-format was ignored"
            % (got_rate, RATE))
    os.remove(raw)

    # `say` pads every render; 113 copies of that padding is seconds of
    # silence the device would store and play.
    samples = voice_audition.trim(samples, RATE)
    return normalise(samples)


def normalise(samples, peak=PEAK):
    """Scale to `peak`. See the module docstring on why peak is not 30000."""
    loudest = max((abs(v) for v in samples), default=0) or 1
    gain = peak / loudest
    return [max(-32768, min(32767, int(round(v * gain)))) for v in samples]


def write_wav(path, samples, rate=RATE):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % len(samples), *samples))


# --- IMA ADPCM ---------------------------------------------------------------
#
# The 1992 IMA/DVI tables, unmodified. They are a specification rather than a
# tuning parameter: the device decoder has to hold the identical numbers or the
# two drift apart within a few hundred samples, so these are not to be
# "cleaned up" any more than epaper.py's LUTs are.

STEP_TABLE = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41,
    45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209,
    230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876,
    963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749,
    3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
    9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767,
)

INDEX_TABLE = (-1, -1, -1, -1, 2, 4, 6, 8)

# Clip blob: n_samples, initial predictor, initial step index, one reserved
# byte, then the nibbles. Sample 0 is the predictor and is not encoded, which
# is what makes the decoder's first output exact and gives the adaptation a
# real starting point instead of a guess.
CLIP_HEADER = struct.Struct("<IhBB")

# Sample 0 is stored verbatim, so the encoder starts from a correct predictor
# and only the step size has to adapt. Left at 0: every clip begins in the
# 30 ms of near-silence `trim` leaves as padding, where step 7 is right and a
# larger opening step would inject a click into the quietest part of the clip.
# Measured over the corpus, forcing the best of {0,4,8,16,24,32} instead moves
# mean SNR by under 0.1 dB, so the search is not worth the format complexity.
INITIAL_STEP_INDEX = 0


def adpcm_encode(samples, step_index=INITIAL_STEP_INDEX):
    """PCM int16 -> one ADPCM blob.

    Closed loop: the predictor is advanced with the *decoded* value, never the
    input, so encoder and decoder see the same state at every step. An encoder
    that predicts from the input instead sounds fine on short clips and drifts
    audibly on long ones, and the round-trip test would still pass if the
    reference decoder made the same mistake -- which is why the fixtures go to
    the device decoder rather than staying here.
    """
    if not samples:
        return CLIP_HEADER.pack(0, 0, step_index, 0)

    predictor = max(-32768, min(32767, samples[0]))
    index = step_index
    nibbles = []

    for sample in samples[1:]:
        step = STEP_TABLE[index]
        delta = sample - predictor
        code = 0
        if delta < 0:
            code = 8
            delta = -delta
        diff = step >> 3
        if delta >= step:
            code |= 4
            delta -= step
            diff += step
        if delta >= (step >> 1):
            code |= 2
            delta -= step >> 1
            diff += step >> 1
        if delta >= (step >> 2):
            code |= 1
            diff += step >> 2
        predictor += -diff if code & 8 else diff
        predictor = max(-32768, min(32767, predictor))
        index = max(0, min(88, index + INDEX_TABLE[code & 7]))
        nibbles.append(code)

    packed = bytearray((len(nibbles) + 1) // 2)
    for i, code in enumerate(nibbles):
        # Low nibble first, the IMA/WAV order. A decoder that reads them the
        # other way round produces noise that still decodes to the right
        # *length*, so this is asserted against the device rather than assumed.
        if i % 2:
            packed[i // 2] |= code << 4
        else:
            packed[i // 2] |= code
    header = CLIP_HEADER.pack(len(samples), max(-32768, min(32767, samples[0])),
                              step_index, 0)
    return bytes(header) + bytes(packed)


def adpcm_decode(blob):
    """One ADPCM blob -> PCM int16. The reference the device must match.

    Deliberately the plainest possible transcription of the standard. It is not
    the decoder that ships -- that one is `@micropython.viper` and lives on the
    device -- it is the thing that decides whether the shipping one is right.
    """
    n_samples, predictor, index, _reserved = CLIP_HEADER.unpack_from(blob, 0)
    if not n_samples:
        return []
    out = [predictor]
    body = blob[CLIP_HEADER.size:]
    for i in range(n_samples - 1):
        byte = body[i // 2]
        code = (byte >> 4) if i % 2 else (byte & 0x0F)
        step = STEP_TABLE[index]
        diff = step >> 3
        if code & 4:
            diff += step
        if code & 2:
            diff += step >> 1
        if code & 1:
            diff += step >> 2
        predictor += -diff if code & 8 else diff
        predictor = max(-32768, min(32767, predictor))
        index = max(0, min(88, index + INDEX_TABLE[code & 7]))
        out.append(predictor)
    return out


def blob_length(n_samples):
    """Bytes an encoded clip of `n_samples` occupies. Index arithmetic."""
    if n_samples <= 1:
        return CLIP_HEADER.size
    return CLIP_HEADER.size + (n_samples // 2)


def snr_db(original, decoded):
    """Signal-to-noise of a round trip, in dB. Sanity, not a quality metric.

    4-bit IMA is nominally ~20 dB on speech. A number far above that means the
    clip was silence; far below means the encoder and decoder disagree, which
    is the failure this exists to catch.
    """
    import math
    signal = sum(v * v for v in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, decoded))
    if not noise:
        return float("inf")
    if not signal:
        return float("-inf")
    return 10 * math.log10(signal / noise)


# --- the container -----------------------------------------------------------
#
# One file, one upload, seekable. The index is a fixed stride at a fixed
# offset, so the device binary-searches it with ~7 twenty-byte reads and never
# holds it in RAM -- which is the whole reason it is sorted and the ids are
# stored as ascii hex rather than packed bytes: `talk._clip_for` already has
# the id as 8 ascii characters and can compare them without parsing anything.

MAGIC = b"VPAK"
VERSION = 1
PAK_HEADER = struct.Struct("<4sHHII")     # magic, version, flags, count, rate
PAK_ENTRY = struct.Struct("<8sIII")       # id, offset, length, n_samples
BLOB_ALIGN = 4                            # DMA-friendly, and free


def _align(value, to=BLOB_ALIGN):
    return (value + to - 1) // to * to


def write_pak(path, clips, rate=RATE):
    """`clips` is an iterable of (id, blob, n_samples). Returns the index."""
    clips = sorted(clips, key=lambda c: c[0])
    ids = [c[0] for c in clips]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate clip ids in the pak: two replies hashed "
                             "the same, or the same reply was rendered twice")

    start = _align(PAK_HEADER.size + PAK_ENTRY.size * len(clips))
    index, offset = [], start
    for clip_hex, blob, n_samples in clips:
        index.append((clip_hex, offset, len(blob), n_samples))
        offset = _align(offset + len(blob))

    with open(path, "wb") as handle:
        handle.write(PAK_HEADER.pack(MAGIC, VERSION, 0, len(clips), rate))
        for clip_hex, at, length, n_samples in index:
            handle.write(PAK_ENTRY.pack(clip_hex.encode(), at, length, n_samples))
        handle.write(b"\0" * (start - handle.tell()))
        for (clip_hex, at, length, _n), (_id, blob, _s) in zip(index, clips):
            assert handle.tell() == at, "index and writer disagree at %s" % clip_hex
            handle.write(blob)
            handle.write(b"\0" * (_align(length) - length))
    return index


def read_pak(path):
    """Walk a pak the way the device would. Returns (rate, [(id, blob, n)])."""
    with open(path, "rb") as handle:
        head = handle.read(PAK_HEADER.size)
        magic, version, _flags, count, rate = PAK_HEADER.unpack(head)
        if magic != MAGIC:
            raise AssertionError("not a voice pak: magic %r" % (magic,))
        if version != VERSION:
            raise AssertionError("pak version %d, this tool writes %d"
                                 % (version, VERSION))
        entries = []
        for i in range(count):
            handle.seek(PAK_HEADER.size + PAK_ENTRY.size * i)
            clip_hex, at, length, n_samples = PAK_ENTRY.unpack(
                handle.read(PAK_ENTRY.size))
            entries.append((clip_hex.decode(), at, length, n_samples))
        out = []
        for clip_hex, at, length, n_samples in entries:
            handle.seek(at)
            out.append((clip_hex, handle.read(length), n_samples))
    return rate, out


# --- fixtures for the device decoder -----------------------------------------

def fixtures(outdir):
    """Blobs and their expected PCM, for whoever writes the device decoder.

    Hand-built edge cases first, then a real rendered reply. The edge cases are
    the ones a fast viper decoder gets wrong: the dangling high nibble at an
    odd sample count, and both saturation rails, where a decoder that keeps the
    predictor in a 16-bit register wraps instead of clamping and the clip
    detonates rather than distorting.
    """
    os.makedirs(outdir, exist_ok=True)
    cases = [
        ("empty", []),
        ("one-sample", [1234]),
        ("two-samples", [0, 900]),
        # Odd sample count -> (n-1) even -> no dangling nibble; the next one
        # has (n-1) odd and does dangle. Both spelled out so neither is a
        # coincidence of whatever the corpus happened to contain.
        ("odd-samples", [0, 100, -100, 250, -250]),
        ("even-samples", [0, 100, -100, 250]),
        # Drive the predictor into both rails and hold it there. A decoder
        # without the clamp diverges permanently from here.
        ("rail-positive", [0] + [32767] * 200),
        ("rail-negative", [0] + [-32768] * 200),
        # Full-scale square wave: the step index pumps to the top of the table
        # and back down every half cycle.
        ("square", [(-20000 if (i // 40) % 2 else 20000) for i in range(400)]),
    ]
    written = []
    for name, samples in cases:
        blob = adpcm_encode(samples)
        decoded = adpcm_decode(blob)
        assert len(decoded) == len(samples), name
        with open(os.path.join(outdir, "%s.adpcm" % name), "wb") as handle:
            handle.write(blob)
        with open(os.path.join(outdir, "%s.pcm" % name), "wb") as handle:
            handle.write(struct.pack("<%dh" % len(decoded), *decoded))
        written.append((name, len(samples), len(blob)))
    return written


# --- the build ---------------------------------------------------------------

def build(outdir, texts, want_fixtures=False, pak_name="voice.pak"):
    os.makedirs(outdir, exist_ok=True)
    wavdir = os.path.join(outdir, "wav")
    os.makedirs(wavdir, exist_ok=True)

    clips, manifest, total_samples, worst = [], [], 0, (999.0, "")
    for i, (text, moods) in enumerate(texts):
        clip_hex = clip_id(text)
        samples = render(text, os.path.join(wavdir, clip_hex))
        blob = adpcm_encode(samples)
        decoded = adpcm_decode(blob)
        if len(decoded) != len(samples):
            raise AssertionError("round trip changed the length of %s" % clip_hex)
        if blob_length(len(samples)) != len(blob):
            raise AssertionError("blob_length disagrees with the encoder on %s"
                                 % clip_hex)
        snr = snr_db(samples, decoded)
        if snr < worst[0]:
            worst = (snr, text)
        clips.append((clip_hex, blob, len(samples)))
        manifest.append((clip_hex, len(samples), moods, snr, text))
        total_samples += len(samples)
        # A decoded WAV per clip, not just the encoded blob: this is what
        # `afplay` plays, and it is the only artefact here that a person can
        # judge. Written from the *decoded* samples deliberately -- auditioning
        # the pre-encode render would audition something the device never
        # plays.
        write_wav(os.path.join(wavdir, "%s.wav" % clip_hex), decoded)
        print("  %3d/%d  %s  %5.2f s  %5.1f dB  %s"
              % (i + 1, len(texts), clip_hex, len(samples) / RATE, snr,
                 text[:46]), flush=True)

    pak = os.path.join(outdir, pak_name)
    index = write_pak(pak, clips)

    with open(os.path.join(outdir, "voice_manifest.txt"), "w") as handle:
        handle.write("# id  samples  seconds  moods  round-trip SNR  text\n")
        handle.write("# Generated by tools/voice_pak.py. id = sha1(text)[:8],\n"
                     "# which is what talk.Conversation._clip_for looks up.\n")
        for clip_hex, n_samples, moods, snr, text in sorted(manifest):
            handle.write("%s  %7d  %6.2f  %-3s  %5.1f  %s\n"
                         % (clip_hex, n_samples, n_samples / RATE, moods or "-",
                            snr, text))

    size = os.path.getsize(pak)
    seconds = total_samples / RATE
    print()
    print("wrote %s" % pak)
    print("  clips            %d" % len(clips))
    print("  speech           %.1f s (%.1f min)" % (seconds, seconds / 60))
    print("  pak              %.2f MB (%d bytes)" % (size / 1024 / 1024, size))
    print("  index            %d bytes, %d per entry, blobs from %d"
          % (PAK_ENTRY.size * len(clips), PAK_ENTRY.size, index[0][1] if index else 0))
    print("  uncompressed     %.2f MB would have been the int16 cost"
          % (total_samples * 2 / 1024 / 1024))
    print("  worst round trip %.1f dB on %r" % (worst[0], worst[1][:50]))
    # 15 MB is the filesystem on the reflashed 16 MB board (docs/hardware.md);
    # the code, model and templates are the rest.
    budget = 15 * 1024 * 1024
    print("  budget           %.2f MB of %d MB, %.2f MB spare (%.1f%% used)"
          % (size / 1024 / 1024, budget // 1024 // 1024,
             (budget - size) / 1024 / 1024, 100.0 * size / budget))
    if size > budget:
        print("  OVER BUDGET")

    if want_fixtures:
        fixdir = os.path.join(outdir, "fixtures")
        for name, n_samples, n_bytes in fixtures(fixdir):
            print("  fixture %-14s %4d samples -> %4d bytes" % (name, n_samples,
                                                               n_bytes))
        print("wrote %s" % fixdir)
    return pak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--only", action="append", default=[],
                    help="render just this reply, by exact text or by clip id. "
                         "Repeatable.")
    ap.add_argument("--only-audition", action="store_true",
                    help="render only %r, the clip to judge before the "
                         "other 112 are trusted" % AUDITION)
    ap.add_argument("--fixtures", action="store_true",
                    help="also write decoder fixtures for the device side")
    ap.add_argument("--list", action="store_true",
                    help="print the corpus and its sizing, render nothing")
    args = ap.parse_args()

    texts = corpus()

    if args.list:
        for text, moods in texts:
            print("%s  %-3s  %s" % (clip_id(text), moods or "-", text))
        print("\n%d replies" % len(texts))
        return 0

    wanted = list(args.only)
    if args.only_audition:
        wanted.append(AUDITION)
    if wanted:
        by_id = {clip_id(t): (t, m) for t, m in texts}
        by_text = {t: (t, m) for t, m in texts}
        picked = []
        for want in wanted:
            if want in by_text:
                picked.append(by_text[want])
            elif want in by_id:
                picked.append(by_id[want])
            else:
                print("no such reply: %r" % want)
                print("(%d in the corpus; `--list` prints them)" % len(texts))
                return 1
        texts = picked

    # A subset never writes `voice.pak`. Auditioning one clip into the
    # directory that holds the built corpus would otherwise replace 113 clips
    # with one, and the result is a valid pak -- nothing downstream would
    # complain, the device would simply go quiet for everything except the
    # line that was auditioned.
    build(args.outdir, texts, want_fixtures=args.fixtures,
          pak_name="voice.pak" if not wanted else "voice-partial.pak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
