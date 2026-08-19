#!/usr/bin/env python3
"""Tests the voice corpus and its container.

    python3 tools/test_voice_pak.py

No `say`, no board, no rendered audio: every fixture here is synthesised, so
the suite runs in about a second and does not need the two minutes of speech
synthesis that `tools/voice_pak.py` does. What it covers is the three ways this
build can be wrong without looking wrong.

**The enumeration can be short.** A reply the corpus does not contain is a clip
the device does not have, and a missing clip is *silence* -- which is also what
a decoder bug, a codec misconfiguration and a flat battery produce. There is no
way to tell them apart at the desk. So the corpus is checked against the engine
itself, driven the way `talk.py` drives it, rather than against a list.

**The container arithmetic can be off by a little.** An index that points four
bytes early decodes to noise, not to an error. The pak is therefore walked
byte-wise here -- offsets, lengths, alignment, the lot -- rather than read back
with the same code that wrote it.

**The codec can be subtly non-standard.** An encoder and a decoder written by
the same person from the same misreading agree with each other perfectly. So
the encoder is checked against `audioop`, CPython's own IMA implementation,
which nothing in this project wrote.

The one thing no test here can cover is whether the result sounds like anything
worth listening to, and per CLAUDE.md and docs/speech-voice.md that is settled
by a person with `afplay` and then by the board's own speaker -- not by a
number in this file.
"""

import os
import shutil
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

# Imported before voice_pak, and for its side effects as much as its helpers:
# it installs the `machine`/`framebuf`/`epaper` stubs and loads the real
# `talk.py` under them. That is the only way to exercise `_clip_for` and
# `Conversation` on the host, and reusing it means the reply sweep here and the
# panel-budget sweep there cannot drift apart.
import test_talk                                    # noqa: E402
import voice_pak                                    # noqa: E402

eliza = test_talk.eliza
rules = test_talk.eliza_rules
talk = test_talk.talk
vocab = test_talk.vocab

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# --- 1. the enumeration ------------------------------------------------------

def test_the_corpus_is_what_the_engine_produces():
    """Drive the real `talk.Conversation` and assert it says nothing new.

    This is the assertion the whole build rests on. `voice_pak.corpus()` walks
    the rule data taking every branch; this walks the engine taking the branch
    the engine chooses, and the two have to agree in the direction that
    matters -- the engine must never produce a text the corpus lacks.

    Two sweeps, because they fail differently. The per-label sweep exhausts
    each rule's rotation from a clean session and would catch a corpus missing
    a whole reply bank. The interleaved walk shares one session across labels,
    which is what a real conversation does, and is the only thing that reaches
    the rotation states a solo sweep cannot: the turn counter is per
    (keyword, decomposition), so a rule can be entered at a position another
    keyword left behind.

    Measured: the interleaved walk stops finding new replies immediately -- 68
    distinct texts, all of them from the solo sweep, and nothing new in 30,000
    further turns. The remaining 43 in `corpus()` are shipped unobserved on
    purpose; see its docstring.
    """
    print("enumeration")
    import random

    known = set(text for text, _moods in voice_pak.corpus())
    labels = list(vocab.LABELS) + [None]
    seen, missing = set(), []

    def turn(session, label):
        text = session.reply(label)
        seen.add(text)
        if text not in known:
            missing.append((label, text))

    for label in labels:
        session = talk.Conversation()
        for _ in range(200):
            turn(session, label)

    rng = random.Random(20260819)
    session = talk.Conversation()
    for _ in range(30000):
        turn(session, rng.choice(labels))

    check("every reply the engine gives is in the corpus (%d distinct seen)"
          % len(seen), not missing, "missing %d, first: %r" % (len(missing),
                                                               missing[:2]))
    check("the sweep actually exercised the engine", len(seen) > 60,
          "only %d distinct replies -- the sweep is not reaching the rules"
          % len(seen))
    check("the greeting is in the corpus",
          talk.Conversation().greeting() in known)
    check("NOTHING_HEARD is in the corpus", talk.NOTHING_HEARD in known)
    check("voice_pak's copy of NOTHING_HEARD matches talk's",
          voice_pak.NOTHING_HEARD == talk.NOTHING_HEARD,
          "%r vs %r" % (voice_pak.NOTHING_HEARD, talk.NOTHING_HEARD))


def test_nothing_is_rendered_that_the_device_cannot_reach():
    """The other direction, against `test_talk`'s own template sweep.

    `corpus()` over-counts deliberately, but it must over-count *within the
    rule data* -- every clip has to trace back to a template the device can
    reach, expanded with a word the spotter can actually deliver. A text that
    does not is either a PHRASE template (needs a transcript the device never
    has) or invented, and both are bytes spent on a reply that can never play.

    The two exceptions are the two lines `talk.py` speaks itself rather than
    asking the engine for, and they are named here rather than filtered by
    pattern so that a third one cannot appear silently.
    """
    print("provenance")
    spoken_by_talk = {eliza.sentence_case(rules.GREETING), talk.NOTHING_HEARD}

    reachable = set()
    for echo in set(vocab.ECHO.values()):
        for is_reachable, template in test_talk._reply_templates():
            if is_reachable:
                reachable.add(test_talk._rendered(template, echo))

    stray = [text for text, _moods in voice_pak.corpus()
             if text not in reachable and text not in spoken_by_talk]
    check("every clip traces to a device-reachable template", not stray,
          "%d stray: %r" % (len(stray), stray[:3]))

    for text in spoken_by_talk:
        check("talk's own line is in the corpus: %r" % text[:32],
              any(t == text for t, _m in voice_pak.corpus()))


def test_clip_ids_are_what_the_device_looks_up():
    """`clip_id` and `talk.Conversation._clip_for` must agree exactly.

    Not compared as strings -- `_clip_for` is *run*, in a directory holding the
    files this tool would have named, and asked to find them. That covers the
    parts a string comparison would not: the `say_` prefix, the `.pcmw`
    suffix, the truncation to eight characters, and MicroPython's
    hexlify-instead-of-hexdigest spelling.
    """
    print("clip ids")
    corpus = voice_pak.corpus()
    ids = [voice_pak.clip_id(text) for text, _moods in corpus]
    check("no two replies share an id (%d clips)" % len(ids),
          len(set(ids)) == len(ids),
          "collisions: %d" % (len(ids) - len(set(ids))))
    check("every id is 8 lowercase hex characters",
          all(len(i) == 8 and all(c in "0123456789abcdef" for c in i)
              for i in ids))

    session = talk.Conversation()
    tmp = tempfile.mkdtemp(prefix="voice-pak-ids-")
    was = os.getcwd()
    try:
        os.chdir(tmp)
        for text, _moods in corpus[:20] + corpus[-20:]:
            name = "say_%s.pcmw" % voice_pak.clip_id(text)
            with open(name, "wb") as handle:
                handle.write(b"\0" * 4)
            check("_clip_for finds %s (%r)" % (name, text[:26]),
                  session._clip_for(text) == name,
                  "got %r" % (session._clip_for(text),))
    finally:
        os.chdir(was)
        shutil.rmtree(tmp, ignore_errors=True)


# --- 2. the codec ------------------------------------------------------------

def _tone(n, period=37, amplitude=9000, offset=0):
    """A cheap deterministic signal. Not speech; speech is not what is at
    stake here -- the encoder is arithmetic and arithmetic is checked with
    arithmetic."""
    import math
    return [max(-32768, min(32767,
                            int(amplitude * math.sin(2 * math.pi * (i + offset)
                                                     / period))))
            for i in range(n)]


def test_the_round_trip_preserves_length_and_sounds_like_the_input():
    print("round trip")
    cases = {
        "empty": [],
        "one sample": [1234],
        "two samples": [0, -900],
        "odd nibble count": [0, 100, -100, 250],
        "tone": _tone(4000),
        "quiet tone": _tone(4000, amplitude=200),
        "loud tone": _tone(4000, amplitude=32000),
        "step to the rail": [0] * 50 + [32767] * 200 + [-32768] * 200,
    }
    for name, samples in cases.items():
        blob = voice_pak.adpcm_encode(samples)
        decoded = voice_pak.adpcm_decode(blob)
        check("%s: length survives (%d)" % (name, len(samples)),
              len(decoded) == len(samples), "got %d" % len(decoded))
        check("%s: blob_length agrees with the encoder" % name,
              voice_pak.blob_length(len(samples)) == len(blob),
              "predicted %d, encoded %d"
              % (voice_pak.blob_length(len(samples)), len(blob)))
        if samples:
            check("%s: sample 0 is exact" % name, decoded[0] == samples[0],
                  "%d vs %d" % (decoded[0], samples[0]))
        check("%s: every decoded sample is a valid int16" % name,
              all(-32768 <= v <= 32767 for v in decoded))

    # 4-bit IMA is nominally ~20 dB on a signal that uses the range. Asserted
    # as a floor rather than a target: what a low number here means is that the
    # encoder and decoder have diverged, not that the audio is poor.
    for name in ("tone", "loud tone"):
        snr = voice_pak.snr_db(cases[name],
                               voice_pak.adpcm_decode(
                                   voice_pak.adpcm_encode(cases[name])))
        check("%s: round trip is %.1f dB, over the 18 dB floor" % (name, snr),
              snr > 18.0)


def test_the_encoder_matches_cpythons_own_ima():
    """Checked against `audioop`, which nobody here wrote.

    An encoder and a decoder built from the same reading of the same spec agree
    with each other whether or not the reading was right, and a round-trip test
    cannot see the difference. `audioop.lin2adpcm` is an independent IMA
    implementation shipped with CPython, so agreeing with it is evidence about
    the *format* rather than about internal consistency.

    Two known and deliberate differences, both accounted for below:

    - `audioop` encodes every sample; `voice_pak` stores sample 0 verbatim in
      the clip header and encodes the rest. So it is handed `samples[1:]` with
      an initial state of `(samples[0], 0)`, which is the same thing.
    - `audioop` packs the **high** nibble first. `voice_pak` packs the low
      nibble first, which is the IMA/WAV convention and what the device
      decoder expects. The bytes are therefore compared nibble-swapped -- and
      that swap being necessary *is* the pinned fact: if the encoder ever
      starts matching audioop byte-for-byte, the nibble order has silently
      flipped and every clip will decode to noise.

    Skipped, loudly, where `audioop` is gone: it is deprecated and was removed
    in Python 3.13. A skipped check is printed rather than passed, because a
    green line for a test that did not run is how a suite goes quietly vacuous.
    """
    print("codec conformance")
    import warnings
    try:
        with warnings.catch_warnings():
            # It warns that it is going away. Noted in the docstring; a
            # DeprecationWarning on stderr every run reads like a failure.
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop
    except ImportError:
        print("  SKIP audioop is not available (removed in Python 3.13); the "
              "independent check on the encoder did not run")
        return

    def swap_nibbles(data):
        return bytes(((b >> 4) | ((b & 0x0F) << 4)) for b in data)

    import random
    rng = random.Random(4242)
    signals = {
        "noise": [max(-32768, min(32767, int(rng.gauss(0, 4000))))
                  for _ in range(1001)],
        "tone": _tone(1001),
        "impulses": [(30000 if i % 97 == 0 else 0) for i in range(1001)],
        "silence": [0] * 1001,
    }
    for name, samples in signals.items():
        raw = struct.pack("<%dh" % (len(samples) - 1), *samples[1:])
        theirs, _state = audioop.lin2adpcm(raw, 2, (samples[0], 0))
        mine = voice_pak.adpcm_encode(samples)[voice_pak.CLIP_HEADER.size:]
        check("%s: nibbles match audioop's IMA" % name,
              swap_nibbles(theirs) == mine,
              "first difference at nibble %d"
              % next((i for i, (a, b) in enumerate(zip(swap_nibbles(theirs),
                                                       mine)) if a != b), -1))
        check("%s: and are NOT byte-identical (nibble order is low-first)"
              % name, theirs != mine or set(theirs) <= {0x00},
              "the encoder has started packing high-nibble-first")

    check("the step table is the 89-entry IMA one",
          len(voice_pak.STEP_TABLE) == 89
          and voice_pak.STEP_TABLE[0] == 7
          and voice_pak.STEP_TABLE[-1] == 32767)
    check("the index table is the 8-entry IMA one",
          voice_pak.INDEX_TABLE == (-1, -1, -1, -1, 2, 4, 6, 8))


def test_the_fixtures_cover_what_a_fast_decoder_gets_wrong():
    """The fixtures handed to the device decoder must contain the traps.

    A fixture set of four ordinary clips proves the two decoders agree on the
    easy path and says nothing about the rails or the dangling nibble, which
    are exactly where a viper decoder holding the predictor in a 32-bit
    register and skipping the clamp will differ.
    """
    print("decoder fixtures")
    tmp = tempfile.mkdtemp(prefix="voice-pak-fix-")
    try:
        written = voice_pak.fixtures(tmp)
        names = [name for name, _n, _b in written]
        for wanted in ("rail-positive", "rail-negative", "square",
                       "even-samples", "odd-samples", "one-sample", "empty"):
            check("fixture present: %s" % wanted, wanted in names)

        dangling = 0
        for name, n_samples, n_bytes in written:
            with open(os.path.join(tmp, "%s.adpcm" % name), "rb") as handle:
                blob = handle.read()
            with open(os.path.join(tmp, "%s.pcm" % name), "rb") as handle:
                pcm = handle.read()
            expected = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
            check("%s: fixture PCM is this decoder's output" % name,
                  list(expected) == voice_pak.adpcm_decode(blob))
            check("%s: fixture blob length is the documented arithmetic" % name,
                  len(blob) == voice_pak.blob_length(n_samples)
                  and len(blob) == n_bytes)
            if n_samples > 1 and (n_samples - 1) % 2:
                dangling += 1
                check("%s: the dangling high nibble is zero" % name,
                      blob[-1] >> 4 == 0)
        check("at least one fixture has a dangling nibble", dangling >= 1)

        rails = voice_pak.adpcm_decode(
            open(os.path.join(tmp, "rail-positive.adpcm"), "rb").read())
        check("the positive rail is reached and held", max(rails) == 32767)
        rails = voice_pak.adpcm_decode(
            open(os.path.join(tmp, "rail-negative.adpcm"), "rb").read())
        check("the negative rail is reached and held", min(rails) == -32768)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_deployable_fixtures_say_what_they_claim(tmp):
    """`src/voice_fixtures.py` is the contract with the device decoder.

    Two things are checked, and the second is the one worth having. The first
    is that the recorded head, tail and checksum are what this decoder
    produces -- which catches a stale generated file. The second is that the
    stress cases actually reach the states their `why` strings advertise: a
    vector named "index pins at 88" that quietly stops at 40 is a test that
    passes on both sides while covering nothing, and the name is the only
    thing that would still say otherwise.
    """
    print("deployable fixtures")
    path = voice_pak.FIXTURE_MODULE
    if not os.path.exists(path):
        print("  SKIP no %s -- run tools/voice_pak.py corpus-voice/ "
              "--fixture-module" % path)
        return
    fixtures = test_talk._load_source("voice_fixtures", path)

    def unpack(raw):
        return list(struct.unpack("<%dh" % (len(raw) // 2), raw))

    check("the module is a current one", fixtures.FORMAT == 1
          and fixtures.RATE == voice_pak.RATE)
    check("there are nibble cases and clip cases",
          len(fixtures.CASES) >= 6 and len(fixtures.CLIPS) >= 3,
          "%d cases, %d clips" % (len(fixtures.CASES), len(fixtures.CLIPS)))

    by_name = {}
    for name, _why, predictor, index, data, count, head, tail, total in \
            fixtures.CASES:
        out = voice_pak.decode_nibbles(data, count, predictor, index)
        by_name[name] = out
        check("%s: decodes to %d samples" % (name, count), len(out) == count)
        check("%s: head matches" % name,
              unpack(head) == out[:fixtures.EDGE])
        check("%s: tail matches" % name,
              unpack(tail) == out[-fixtures.EDGE:])
        check("%s: checksum matches" % name,
              fixtures.checksum(out) == total,
              "%d vs %d" % (fixtures.checksum(out), total))
        check("%s: nibble bytes are exactly half the nibble count" % name,
              len(data) == (count + 1) // 2)

    # The claims. Each is an observable consequence of the state the case is
    # named for, so none of them needs the decoder instrumented.
    check("tiny: uses all sixteen codes, so no code is untested",
          len({(b >> 4) for b in voice_pak._nibble_cases()[0][4]}
              | {(b & 0x0F) for b in voice_pak._nibble_cases()[0][4]}) == 16)
    check("index-to-max: the predictor reaches +32767, which needs the step "
          "to have grown", max(by_name["index-to-max"]) == 32767)
    check("index-to-min-value: the predictor reaches -32768",
          min(by_name["index-to-min-value"]) == -32768)
    # At index 0 the step is 7 and 7 >> 3 is zero, so a decoder that clamped
    # the index correctly stops moving. A decoder that let the index go
    # negative would read off the front of the table instead.
    tail_of = by_name["index-down-to-zero"][-40:]
    check("index-down-to-zero: the output goes flat, so the index clamped at 0",
          len(set(tail_of)) == 1, "%d distinct values in the tail"
          % len(set(tail_of)))
    check("held-at-the-rail: pushing further into the rail does not move it",
          set(by_name["held-at-the-rail"]) == {32767})

    known = {voice_pak.clip_id(text): text for text, _m in voice_pak.corpus()}
    for clip_hex, _why, text, blob, n_samples, head, tail, total in \
            fixtures.CLIPS:
        out = voice_pak.adpcm_decode(blob)
        check("clip %s: is a current reply" % clip_hex, clip_hex in known,
              "not in the corpus")
        check("clip %s: the text is the one that hashes to it" % clip_hex,
              voice_pak.clip_id(text) == clip_hex)
        check("clip %s: decodes to %d samples" % (clip_hex, n_samples),
              len(out) == n_samples)
        check("clip %s: head, tail and checksum match" % clip_hex,
              unpack(head) == out[:fixtures.EDGE]
              and unpack(tail) == out[-fixtures.EDGE:]
              and fixtures.checksum(out) == total)

    # The checksum is a pin only if it can fail. A plain sum could not tell
    # these apart, which is why it is a rolling hash.
    swapped = list(by_name["tiny"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    check("the checksum notices two samples swapped",
          fixtures.checksum(swapped) != fixtures.checksum(by_name["tiny"]))


def test_the_fixture_module_can_load_on_the_board():
    """It has to run under MicroPython, where most of the library is absent.

    The point of generating literals rather than reading the pak is that the
    module needs nothing at import but the interpreter. An `import struct` or
    an f-string would sail through here and fail on the device, where there is
    no second chance to notice.
    """
    print("fixture module portability")
    path = voice_pak.FIXTURE_MODULE
    if not os.path.exists(path):
        print("  SKIP no %s" % path)
        return
    import ast

    with open(path) as handle:
        source = handle.read()
    # Parsed, not grepped. A grep for "f'" matches the byte b'\x7f' in every
    # data literal in the file, which is how the first version of this check
    # failed on nothing at all.
    tree = ast.parse(source, path)
    imports = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("imports nothing at all", not imports,
          "imports on lines %r" % [n.lineno for n in imports])
    fstrings = [node for node in ast.walk(tree)
                if isinstance(node, ast.JoinedStr)]
    check("no f-strings, which MicroPython's compiler is picky about",
          not fstrings, "on lines %r" % [n.lineno for n in fstrings])
    check("it is a generated file and says so",
          "Generated" in source.splitlines()[0])


# --- 3. the container --------------------------------------------------------

def _synthetic_pak(path, count=40):
    """A pak over real clip ids and fake audio.

    Real ids, because the index is sorted by them and their distribution is
    what the device's binary search walks. Fake audio, because rendering forty
    lines through `say` would put a minute of speech synthesis in a unit test
    and prove nothing the round-trip checks above do not.
    """
    texts = [text for text, _moods in voice_pak.corpus()][:count]
    clips, expected = [], {}
    for i, text in enumerate(texts):
        samples = _tone(200 + i * 13, period=17 + i, amplitude=1000 + i * 300)
        blob = voice_pak.adpcm_encode(samples)
        clips.append((voice_pak.clip_id(text), blob, len(samples)))
        expected[voice_pak.clip_id(text)] = (text, samples, blob)
    voice_pak.write_pak(path, clips)
    return expected


def test_the_pak_walks(tmp):
    """Parse the file from its bytes, not with the writer's own bookkeeping.

    Every field is re-derived here: the header is unpacked by hand, entry `i`
    is read at the stride the device will use, and the blobs are checked to
    tile the file without gaps beyond alignment padding or any overlap at all.
    An index that is internally consistent but four bytes out decodes to noise
    rather than to an error, so "the reader agreed with the writer" is not
    evidence.
    """
    print("container")
    path = os.path.join(tmp, "voice.pak")
    expected = _synthetic_pak(path)

    with open(path, "rb") as handle:
        data = handle.read()

    magic, version, flags, count, rate = voice_pak.PAK_HEADER.unpack_from(data, 0)
    check("magic", magic == b"VPAK", repr(magic))
    check("version 1", version == 1)
    check("flags are zero", flags == 0)
    check("count matches what was packed", count == len(expected), str(count))
    check("sample rate is 16000", rate == 16000, str(rate))

    entries = []
    for i in range(count):
        at = voice_pak.PAK_HEADER.size + voice_pak.PAK_ENTRY.size * i
        clip_hex, offset, length, n_samples = voice_pak.PAK_ENTRY.unpack_from(
            data, at)
        entries.append((clip_hex.decode(), offset, length, n_samples))

    ids = [e[0] for e in entries]
    check("the index is sorted, so the device can binary-search it",
          ids == sorted(ids))
    check("every id in the pak resolves to a corpus reply",
          all(i in expected for i in ids),
          "unknown: %r" % [i for i in ids if i not in expected][:3])
    check("every corpus reply packed has an entry",
          set(ids) == set(expected))

    first_blob = voice_pak.PAK_HEADER.size + voice_pak.PAK_ENTRY.size * count
    check("the blobs start after the index",
          entries[0][1] >= first_blob,
          "first blob at %d, index ends at %d" % (entries[0][1], first_blob))

    end = None
    for clip_hex, offset, length, n_samples in entries:
        check("%s: blob is inside the file" % clip_hex,
              offset + length <= len(data),
              "%d + %d > %d" % (offset, length, len(data)))
        check("%s: blob is 4-byte aligned" % clip_hex, offset % 4 == 0,
              str(offset))
        if end is not None:
            check("%s: does not overlap the blob before it" % clip_hex,
                  offset >= end, "starts at %d, previous ended at %d"
                  % (offset, end))
            check("%s: no more than 3 bytes of padding before it" % clip_hex,
                  offset - end < 4, "gap of %d bytes" % (offset - end))
        end = offset + length

        text, samples, blob = expected[clip_hex]
        check("%s: length is the documented arithmetic" % clip_hex,
              length == voice_pak.blob_length(n_samples))
        check("%s: n_samples in the index matches the blob header" % clip_hex,
              n_samples == voice_pak.CLIP_HEADER.unpack_from(
                  data, offset)[0])
        check("%s: the bytes at the offset are this clip" % clip_hex,
              data[offset:offset + length] == blob)
        check("%s: and decode to the audio that was packed" % clip_hex,
              voice_pak.adpcm_decode(data[offset:offset + length])
              == voice_pak.adpcm_decode(blob))
    check("the file ends at the last blob (within alignment)",
          0 <= len(data) - end < 4, "%d trailing bytes" % (len(data) - end))

    got_rate, out = voice_pak.read_pak(path)
    check("read_pak returns every clip", len(out) == count and got_rate == rate)
    check("read_pak returns the same bytes as walking the file by hand",
          all(blob == expected[clip_hex][2] for clip_hex, blob, _n in out))


def test_a_duplicate_is_refused(tmp):
    """Two clips with one id would silently shadow each other in the index."""
    print("duplicate refusal")
    blob = voice_pak.adpcm_encode(_tone(100))
    try:
        voice_pak.write_pak(os.path.join(tmp, "dup.pak"),
                            [("aaaaaaaa", blob, 100), ("aaaaaaaa", blob, 100)])
        check("write_pak refuses a duplicate id", False, "it accepted one")
    except AssertionError:
        check("write_pak refuses a duplicate id", True)


def test_the_real_pak_if_one_has_been_built():
    """Validate `corpus-voice/voice.pak` when it exists, and say so when not.

    The build is a two-minute `say` run and its output is gitignored, so a
    fresh clone has no pak and this is not a failure. It is printed rather
    than passed over, because "the pak is fine" and "there was no pak" are
    different results and only one of them is worth anything.
    """
    print("the built pak")
    path = os.path.join(ROOT, "corpus-voice", "voice.pak")
    if not os.path.exists(path):
        print("  SKIP no pak at %s -- run tools/voice_pak.py corpus-voice/"
              % path)
        return
    rate, clips = voice_pak.read_pak(path)
    known = {voice_pak.clip_id(text): text for text, _m in voice_pak.corpus()}
    check("the built pak is 16 kHz", rate == 16000, str(rate))
    check("every id in the built pak is a current reply",
          all(clip_hex in known for clip_hex, _b, _n in clips),
          "stale: %r" % [c for c, _b, _n in clips if c not in known][:3])
    check("every current reply is in the built pak",
          set(known) <= set(c for c, _b, _n in clips),
          "missing %d" % len(set(known) - set(c for c, _b, _n in clips)))
    bad = [clip_hex for clip_hex, blob, n in clips
           if len(voice_pak.adpcm_decode(blob)) != n]
    check("every blob decodes to the sample count its index claims", not bad,
          "wrong: %r" % bad[:3])
    seconds = sum(n for _c, _b, n in clips) / float(rate)
    size = os.path.getsize(path)
    print("       %d clips, %.0f s, %.2f MB" % (len(clips), seconds,
                                                size / 1024 / 1024))
    check("the pak fits the 15 MB filesystem with room for the program",
          size < 12 * 1024 * 1024, "%.2f MB" % (size / 1024 / 1024))


def main():
    test_the_corpus_is_what_the_engine_produces()
    test_nothing_is_rendered_that_the_device_cannot_reach()
    test_clip_ids_are_what_the_device_looks_up()
    test_the_round_trip_preserves_length_and_sounds_like_the_input()
    test_the_encoder_matches_cpythons_own_ima()
    test_the_fixtures_cover_what_a_fast_decoder_gets_wrong()
    test_the_fixture_module_can_load_on_the_board()
    tmp = tempfile.mkdtemp(prefix="voice-pak-")
    try:
        test_the_deployable_fixtures_say_what_they_claim(tmp)
        test_the_pak_walks(tmp)
        test_a_duplicate_is_refused(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    test_the_real_pak_if_one_has_been_built()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES[:6])))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
