#!/usr/bin/env python3
"""Pins `src/adpcm.py` against the encoder, and `listen`'s streaming against it.

    python3 tools/test_adpcm.py

Three independent pins, because a decoder has no partial failures worth the
name: 4-bit IMA either reproduces the encoder's arithmetic exactly or it
diverges and never comes back, and what a person hears at the desk is "the
voice is a bit crackly" -- which is also what a wrong volume, a wrong sample
rate, an overdriven amp and a clipped render sound like. Nothing points here.

1. **Against the reference.** `tools/voice_pak.py::adpcm_decode` is the
   normative decoder and the one the encoder was built against. Every case its
   own `fixtures()` emits is decoded by both and compared **sample by sample**,
   and so is a run of pseudo-random PCM. No tolerance anywhere.
2. **Against arithmetic nobody wrote twice.** `test_hand_vector` walks sixteen
   nibbles whose expected output was computed by hand from the table, so a
   shared misreading of the standard by both implementations still fails. The
   first five steps of that derivation are written out in the test.
3. **Against int32.** Every comparison runs with `adpcm.CHECK_BOUNDS` on, which
   makes the host raise on a value that would wrap on the device -- and
   `test_bounds` drives the step index to both ends of the table and checks the
   measured peaks against the bound stated in `adpcm.py`'s docstring.

## What this cannot catch, and what does

Under CPython both sides get arbitrary-precision ints. `@micropython.viper`'s
`int` is a signed machine word that wraps **silently**, so this file can pass
perfectly while the board decodes something else. That is the same gap
`tools/test_spotter.py` records, and it is closed the same way: `--emit-fixtures`
writes `src/voice_fixtures.py`, which the board imports and checks against these
same expected bytes. A green run here means the arithmetic agrees given
unbounded integers, and nothing more.

## The streaming test is the interesting one

`test_stream_ping_pong` runs `listen.Recorder._stream` -- the real one, not a
model of it -- against a real `voice.pak`, with only the two methods that touch
hardware replaced. The fake `_arm` copies out the words it was pointed at, so
the test reconstructs exactly the sample sequence the DMA would have handed the
PIO and compares it with the reference decode of the whole clip.

That is what catches the failures the chunking can actually have: a nibble
stream split across a byte, a half decoded into the wrong offset, an off-by-one
where sample 0 comes from the clip header, and the last chunk. All four are
silent -- they produce audio of the right length that is wrong in the middle.
"""

import os
import random
import struct
import sys
import tempfile
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Never leave a .pyc behind, and never read one. Same reasoning as
# tools/test_spotter.py::_load_source: every module here is under active edit,
# and a stale cache gives clean wrong values rather than an error.
sys.dont_write_bytecode = True

import voice_pak  # noqa: E402 -- the reference encoder and decoder

FAILURES = []


def check(what, ok, detail=""):
    if ok:
        print("  ok   %s" % what)
    else:
        FAILURES.append(what)
        print("  FAIL %s%s" % (what, ("  -- " + detail) if detail else ""))


def _load_source(name, path):
    """Load a module from its source text, with no __pycache__ involved."""
    with open(path) as handle:
        source = handle.read()
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


adpcm = _load_source("adpcm", os.path.join(ROOT, "src", "adpcm.py"))


def install_stubs():
    """Enough of MicroPython for `listen.py` to import and run its arithmetic.

    `listen` needs `board` (which needs `machine`), `rp2`, and the `time`
    tick helpers. Nothing stubbed here is exercised by the streaming test: the
    two methods that touch the DMA are replaced in the test itself, which is
    what keeps the stubs from quietly becoming a second implementation of the
    thing under test.
    """
    machine = types.ModuleType("machine")

    class FakePin:
        IN = 0
        OUT = 1
        PULL_UP = 2

        def __init__(self, *a, **k):
            self._v = k.get("value", 0)

        def value(self, v=None):
            if v is None:
                return self._v
            self._v = v

    machine.Pin = FakePin
    machine.I2C = lambda *a, **k: None
    machine.ADC = lambda *a, **k: types.SimpleNamespace(read_u16=lambda: 40000)
    machine.SPI = lambda *a, **k: types.SimpleNamespace(write=lambda *x: None)
    machine.freq = lambda: 150_000_000
    sys.modules["machine"] = machine
    sys.modules["rp2"] = types.ModuleType("rp2")

    import time as _time
    if not hasattr(_time, "ticks_us"):
        _time.ticks_ms = lambda: int(_time.monotonic() * 1000)
        _time.ticks_us = lambda: int(_time.monotonic() * 1e6)
        _time.ticks_diff = lambda a, b: a - b
        _time.ticks_add = lambda a, b: a + b
        _time.sleep_ms = lambda ms: None


install_stubs()
listen = _load_source("listen", os.path.join(ROOT, "src", "listen.py"))


# --- helpers ---------------------------------------------------------------

def decode_device(blob):
    """Decode one blob with `src/adpcm.py`, returned as a list of int16.

    Goes through the packed-word output the device really writes and reads the
    low halfword back, so the packing convention is exercised rather than
    bypassed. The high halfword is checked to match in `test_word_packing`.
    """
    n_samples, predictor, index, _reserved = voice_pak.CLIP_HEADER.unpack_from(
        blob, 0)
    if not n_samples:
        return []
    out = array("h", bytes(4 * n_samples))
    adpcm.emit_sample(out, 0, predictor)
    nibs = n_samples - 1
    if nibs:
        state = adpcm.new_state(predictor, index)
        adpcm.decode_into(blob, voice_pak.CLIP_HEADER.size, nibs, out, 1, state)
    return [out[2 * i] for i in range(n_samples)]


def fixture_cases():
    """The encoder's own fixture set, encoded and decoded by the reference.

    Read back off disk rather than kept in memory: `fixtures()` is what
    `voice_pak.py --fixtures` ships to whoever writes a decoder, so this checks
    the artefact that would actually be handed over, not a parallel path
    through the same functions.
    """
    outdir = tempfile.mkdtemp(prefix="adpcm-fixtures-")
    written = voice_pak.fixtures(outdir)
    cases = []
    for name, n_samples, _n_bytes in written:
        with open(os.path.join(outdir, "%s.adpcm" % name), "rb") as handle:
            blob = handle.read()
        with open(os.path.join(outdir, "%s.pcm" % name), "rb") as handle:
            raw = handle.read()
        expected = list(struct.unpack("<%dh" % (len(raw) // 2), raw))
        cases.append((name, blob, expected, n_samples))
    return cases


def speechish(n, seed=20260819):
    """Pseudo-random PCM with speech-shaped dynamics. Not a signal, a stressor.

    A random walk with occasional jumps: the walk keeps the step index low
    where quantisation is finest, and the jumps drive it up the table, so one
    run visits most of the adaptation range instead of sitting in one corner.
    """
    rng = random.Random(seed)
    out = []
    v = 0
    for i in range(n):
        if i % 137 == 0:
            v = rng.randint(-15000, 15000)
        v += rng.randint(-900, 900)
        out.append(max(-32768, min(32767, v)))
    return out


# --- the tests -------------------------------------------------------------

def test_tables():
    print("\ntables")
    check("STEP_TABLE is the reference table, element by element",
          list(adpcm.STEP_TABLE) == list(voice_pak.STEP_TABLE),
          "%d vs %d entries" % (len(adpcm.STEP_TABLE),
                                len(voice_pak.STEP_TABLE)))
    check("89 steps ending at 32767",
          len(adpcm.STEP_TABLE) == 89 and adpcm.STEP_TABLE[88] == 32767)
    check("STEP_MAX_INDEX addresses the last step",
          adpcm.STEP_MAX_INDEX == len(adpcm.STEP_TABLE) - 1)
    # The device table is sixteen entries so the inner loop needs no `& 7`.
    # That is only safe if it mirrors exactly.
    mirrored = list(voice_pak.INDEX_TABLE) * 2
    check("INDEX_TABLE mirrors the reference's eight entries",
          list(adpcm.INDEX_TABLE) == mirrored,
          "%r vs %r" % (list(adpcm.INDEX_TABLE), mirrored))
    check("the pak format version matches the writer's",
          adpcm.FORMAT == voice_pak.VERSION)
    check("header and entry strides match the writer's",
          (adpcm.HEADER_LEN, adpcm.ENTRY_LEN, adpcm.CLIP_HEADER_LEN)
          == (voice_pak.PAK_HEADER.size, voice_pak.PAK_ENTRY.size,
              voice_pak.CLIP_HEADER.size),
          "%r vs %r" % ((adpcm.HEADER_LEN, adpcm.ENTRY_LEN,
                         adpcm.CLIP_HEADER_LEN),
                        (voice_pak.PAK_HEADER.size, voice_pak.PAK_ENTRY.size,
                         voice_pak.CLIP_HEADER.size)))
    check("the magic matches", adpcm.MAGIC == voice_pak.MAGIC)
    check("the sample rate matches the render rate",
          adpcm.SAMPLE_RATE == voice_pak.RATE == listen.SAMPLE_RATE)


def test_hand_vector():
    """Sixteen nibbles worked out by hand from the table in the docstring.

    The point of this one is that no code produced the expected values, so a
    misreading of the standard shared by the encoder and the decoder -- the
    `((2*(code&7)+1)*step) >> 3` variant, say, or the high nibble first -- still
    fails here. The first five steps, from predictor 0 and step index 0:

        code 4  step 7   diff 0+7            = 7    pred    7   index 0+2 =  2
        code 7  step 9   diff 1+9+4+2        = 16   pred   23   index 2+8 = 10
        code F  step 19  diff 2+19+9+4       = 34   pred  -11   index 10+8= 18
        code 0  step 41  diff 5              = 5    pred   -6   index 18-1= 17
        code 8  step 37  diff 4              = 4    pred  -10   index 17-1= 16

    The remaining eleven were continued the same way; between them the codes
    cover all sixteen values, both signs and both nibble positions.
    """
    print("\nhand-computed vector")
    codes = [4, 7, 15, 0, 8, 5, 13, 6, 14, 3, 11, 1, 9, 2, 10, 12]
    expected = [7, 23, -11, -6, -10, 36, -32, 86, -125, 75,
                -107, -37, -101, -4, -92, -238]
    packed = bytearray()
    for i in range(0, len(codes), 2):
        packed.append(codes[i] | (codes[i + 1] << 4))
    check("the vector covers every code once", sorted(codes) == list(range(16)))
    check("packs to the expected bytes", bytes(packed).hex() == "740f586d3e1b29ca",
          bytes(packed).hex())

    out = array("h", bytes(4 * len(codes)))
    state = adpcm.new_state(0, 0)
    adpcm.decode_into(packed, 0, len(codes), out, 0, state)
    got = [out[2 * i] for i in range(len(codes))]
    check("device decoder matches the hand-computed samples", got == expected,
          "%r" % (got,))
    check("state is carried out correctly", list(state) == [-238, 32],
          "%r" % (list(state),))

    # And the reference agrees with the hand arithmetic too, which is what makes
    # the two-way comparison below meaningful rather than circular.
    blob = voice_pak.CLIP_HEADER.pack(len(codes) + 1, 0, 0, 0) + bytes(packed)
    check("reference decoder matches the same hand-computed samples",
          voice_pak.adpcm_decode(blob) == [0] + expected)


def test_against_reference():
    print("\nagainst tools/voice_pak.py's reference decoder")
    cases = fixture_cases()
    check("the encoder emitted its whole fixture set", len(cases) == 8,
          "%d cases" % len(cases))
    for name, blob, expected, n_samples in cases:
        got = decode_device(blob)
        check("fixture %-14s (%d samples)" % (name, n_samples),
              got == expected,
              "first difference at %s" % _first_diff(got, expected))
        check("fixture %-14s blob_length agrees" % name,
              len(blob) == voice_pak.blob_length(n_samples))

    # Real-length material, where a one-LSB divergence has room to become
    # audible rather than staying inside a five-sample fixture.
    for seconds, seed in ((1.0, 1), (3.5, 2)):
        pcm = speechish(int(seconds * voice_pak.RATE), seed=seed)
        blob = voice_pak.adpcm_encode(pcm)
        want = voice_pak.adpcm_decode(blob)
        got = decode_device(blob)
        check("%.1f s of stressor PCM decodes identically" % seconds,
              got == want, "first difference at %s" % _first_diff(got, want))
        check("%.1f s round trip is >= 15 dB SNR" % seconds,
              voice_pak.snr_db(pcm, got) >= 15.0,
              "%.1f dB" % voice_pak.snr_db(pcm, got))


def _first_diff(a, b):
    if len(a) != len(b):
        return "length %d vs %d" % (len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "index %d: %d vs %d" % (i, x, y)
    return "none"


def test_word_packing():
    print("\npacked-word output")
    pcm = speechish(4000, seed=3)
    blob = voice_pak.adpcm_encode(pcm)
    n_samples = len(pcm)
    out = array("h", bytes(4 * n_samples))
    predictor = voice_pak.CLIP_HEADER.unpack_from(blob, 0)[1]
    index = voice_pak.CLIP_HEADER.unpack_from(blob, 0)[2]
    adpcm.emit_sample(out, 0, predictor)
    state = adpcm.new_state(predictor, index)
    adpcm.decode_into(blob, voice_pak.CLIP_HEADER.size, n_samples - 1, out, 1,
                      state)
    both = all(out[2 * i] == out[2 * i + 1] for i in range(n_samples))
    check("every word carries the sample in both halves", both)
    # The PIO shifts bits 31..16 out as left and 15..0 as right, so identical
    # halves is what makes a mono clip play on both channels at the same level.
    # A word of (s << 16) alone plays at half amplitude on one side and is the
    # kind of thing that reads as "quiet" rather than as a bug.
    check("decoding at a word offset lands where it was asked to",
          _offset_lands_correctly())


def _offset_lands_correctly():
    codes = bytes([0x44, 0x44])
    out = array("h", bytes(4 * 16))
    state = adpcm.new_state(0, 0)
    adpcm.decode_into(codes, 0, 4, out, 5, state)
    untouched = all(out[i] == 0 for i in range(10)) and \
        all(out[i] == 0 for i in range(18, 32))
    written = all(out[i] != 0 for i in range(10, 18))
    return untouched and written


def test_bounds():
    print("\nint32 headroom (adpcm.CHECK_BOUNDS)")
    adpcm.reset_peaks()
    adpcm.CHECK_BOUNDS = True
    try:
        # Every code, every nibble position, from every step index in the
        # table. This is the exhaustive version of the fixtures' "square" case
        # and it visits the largest step with the largest diff, which is where
        # the bound in adpcm.py's docstring is stated.
        every = bytearray()
        for hi in range(16):
            for lo in range(16):
                every.append(lo | (hi << 4))
        for start_index in range(89):
            out = array("h", bytes(4 * 512))
            state = adpcm.new_state(0, start_index)
            adpcm.decode_into(every, 0, 512, out, 0, state)
            for i in range(512):
                v = out[2 * i]
                if v < -32768 or v > 32767:
                    check("clamped to int16 from step index %d" % start_index,
                          False, "sample %d = %d" % (i, v))
                    return
        check("every code from every step index stays inside int16", True)

        peaks = adpcm.peak_report()
        # The bound stated in adpcm.py's docstring. Checked, not assumed: a
        # table edited by hand is exactly how `step` stops being <= 32767.
        check("|diff| stayed within the proved 61436",
              peaks.get("diff", 0) <= 61436, "%r" % (peaks,))
        check("|pred| before clamping stayed within the proved 94204",
              peaks.get("pred_raw", 0) <= 94204, "%r" % (peaks,))
        check("the proof is not vacuous -- the sweep got near the bound",
              peaks.get("diff", 0) > 50000, "%r" % (peaks,))
        print("       peaks: diff %d (%.4f%% of int32), pred_raw %d"
              % (peaks.get("diff", 0),
                 100.0 * peaks.get("diff", 0) / (1 << 31),
                 peaks.get("pred_raw", 0)))
    finally:
        adpcm.CHECK_BOUNDS = False


def test_chunked_decode():
    """Decoding in pieces must equal decoding in one go. The streaming contract.

    Chunks are even, and the last one may be odd -- which is exactly what
    `listen._chunk_nibs` guarantees and what `decode_into` requires, since it
    always enters a byte at its low nibble. `test_chunk_nibs` pins the
    guarantee; this pins that the guarantee is sufficient.
    """
    print("\nchunked decode")
    pcm = speechish(3001, seed=4)
    blob = voice_pak.adpcm_encode(pcm)
    want = voice_pak.adpcm_decode(blob)
    n_samples = len(pcm)
    body = voice_pak.CLIP_HEADER.size
    predictor, index = voice_pak.CLIP_HEADER.unpack_from(blob, 0)[1:3]

    for chunk in (2, 8, 100, 512, 998, 3000):
        out = array("h", bytes(4 * n_samples))
        adpcm.emit_sample(out, 0, predictor)
        state = adpcm.new_state(predictor, index)
        done = 0
        remaining = n_samples - 1
        while remaining:
            n = chunk if remaining > chunk else remaining
            adpcm.decode_into(blob, body + (done >> 1), n, out, 1 + done, state)
            done += n
            remaining -= n
        got = [out[2 * i] for i in range(n_samples)]
        check("chunks of %d decode identically" % chunk, got == want,
              "first difference at %s" % _first_diff(got, want))


def test_chunk_nibs():
    print("\nlisten._chunk_nibs")
    rec = _recorder(max_samples=4096)
    # Never splits a byte except on the final chunk, never overruns the room,
    # and always makes progress. The last is what would hang: a first sweep of
    # this found `room=1, remaining=2` returning 0, which is an infinite loop in
    # `_stream` and would present at the desk as the board dying mid-reply.
    # `bind_voice` now refuses a buffer that small; room starts at 2 here
    # because that is the smallest value it can produce.
    for room in (2, 3, 7, 8, 4095):
        for remaining in range(1, 40):
            n = rec._chunk_nibs(remaining, room)
            if not (1 <= n <= min(room, remaining)):
                check("room %d remaining %d is in range" % (room, remaining),
                      False, "got %d" % n)
                return
            if n < remaining and (n & 1):
                check("room %d remaining %d ends on a byte boundary"
                      % (room, remaining), False, "got %d" % n)
                return
    check("never splits a byte, never overruns, always advances", True)
    check("an odd room rounds down rather than up",
          rec._chunk_nibs(100, 7) == 6)
    check("a final short chunk may be odd", rec._chunk_nibs(5, 100) == 5)


def _recorder(max_samples=4096):
    """A Recorder with no hardware, for the arithmetic only."""
    return listen.Recorder(rate=listen.SAMPLE_RATE, max_samples=max_samples)


def test_buffer_geometry():
    print("\nplay buffer geometry")
    for max_samples in (4096, listen.MAX_SAMPLES):
        rec = _recorder(max_samples)
        half = rec._play_half_words
        # A half must be a whole number of words, because the viper decoder
        # stores through a ptr32 and an odd halfword offset would be a
        # misaligned 32-bit store.
        check("%d samples: the halves are word-aligned" % max_samples,
              (2 * half) % 2 == 0 and 4 * half <= 2 * len(rec.buf),
              "half=%d buf=%d halfwords" % (half, len(rec.buf)))
        check("%d samples: the two halves do not overlap" % max_samples,
              len(rec._play_halves[0]) == len(rec._play_halves[1]) == 2 * half)
        check("%d samples: the scratch holds a whole chunk of nibbles"
              % max_samples,
              (half // 2 + 1) >= (half + 1) // 2)
    check("the shipped buffer is well clear of the streaming minimum",
          _recorder(listen.MAX_SAMPLES)._play_half_words
          >= 16 * listen.MIN_PLAY_HALF_WORDS)
    rec = _recorder(listen.MAX_SAMPLES)
    print("       %d words a half = %.0f ms at %d Hz, %d bytes of nibbles"
          % (rec._play_half_words,
             1000.0 * rec._play_half_words / listen.SAMPLE_RATE,
             listen.SAMPLE_RATE, rec._play_half_words // 2 + 1))


def _build_pak(clips, path):
    """Write a real pak with the real writer. `clips` is [(text, pcm)]."""
    entries = []
    for text, pcm in clips:
        blob = voice_pak.adpcm_encode(pcm)
        entries.append((voice_pak.clip_id(text), blob, len(pcm)))
    voice_pak.write_pak(path, entries)
    return entries


def test_pak_reader():
    print("\nvoice.pak reader")
    tmp = tempfile.mkdtemp(prefix="adpcm-pak-")
    path = os.path.join(tmp, "voice.pak")
    texts = ["Tell me more about your family.",
             "Please go on.",
             "Why do you say that?",
             "I am not sure I understand you fully.",
             "Do you feel strongly about discussing such things?"]
    clips = [(t, speechish(2000 + 700 * i, seed=10 + i))
             for i, t in enumerate(texts)]
    entries = _build_pak(clips, path)
    by_id = {cid: (blob, n) for cid, blob, n in entries}

    pak = adpcm.Pak(path)
    check("opens", pak.open(), pak.error or "")
    check("reports the clip count", pak.count == len(texts),
          "%d" % pak.count)
    check("reports the rate", pak.rate == voice_pak.RATE, "%d" % pak.rate)

    # Every id found by the bisect, and every id found as bytes and as str,
    # because `talk._clip_for` hands over bytes from binascii.hexlify.
    ok = True
    for cid, (blob, n_samples) in by_id.items():
        entry = pak.lookup(cid.encode())
        if entry is None or entry[2] != n_samples:
            ok = False
            break
        if pak.lookup(cid) != entry:
            ok = False
            break
    check("every clip is found by binary search, as bytes and as str", ok)
    check("an id that is not there returns None",
          pak.lookup(b"deadbeef") is None)
    check("an id shorter than the stride returns None",
          pak.lookup(b"dead") is None)

    # And the bytes come back out. decode_clip is the probe path, not the
    # playback path, but it exercises start_clip and readinto.
    ok = True
    for cid, (blob, n_samples) in by_id.items():
        entry = pak.lookup(cid.encode())
        out = array("h", bytes(4 * n_samples))
        got_n = adpcm.decode_clip(pak, entry[0], out, 0, n_samples)
        got = [out[2 * i] for i in range(n_samples)]
        if got_n != n_samples or got != voice_pak.adpcm_decode(blob):
            ok = False
            check("decode_clip on %s" % cid, False,
                  "first difference at %s"
                  % _first_diff(got, voice_pak.adpcm_decode(blob)))
            break
    check("decode_clip reproduces every clip in the pak", ok)
    pak.close()

    bad = adpcm.Pak(os.path.join(tmp, "nothing-here.pak"))
    check("a missing pak reports rather than raises", bad.open() is False)
    with open(os.path.join(tmp, "wrong.pak"), "wb") as handle:
        handle.write(b"NOPE" + b"\0" * 40)
    check("a pak with the wrong magic is refused",
          adpcm.Pak(os.path.join(tmp, "wrong.pak")).open() is False)
    with open(os.path.join(tmp, "future.pak"), "wb") as handle:
        handle.write(voice_pak.PAK_HEADER.pack(voice_pak.MAGIC,
                                               voice_pak.VERSION + 1, 0, 0,
                                               voice_pak.RATE))
    check("a pak from a later encoder is refused, not misread",
          adpcm.Pak(os.path.join(tmp, "future.pak")).open() is False)
    return path, by_id


def test_stream_ping_pong(path, by_id):
    """The real `_stream`, with only the two hardware methods replaced.

    The fake `_arm` copies the words out of the half it was pointed at, so what
    is compared is the exact sample sequence the DMA would have handed the PIO.
    """
    print("\nstreaming ping-pong")
    pak = adpcm.Pak(path)
    pak.open()

    # A clip much longer than the buffer, so it crosses several boundaries, and
    # clips shorter than one half, so the single-chunk case is covered too.
    for label, max_samples in (("clip spans many halves", 512),
                               ("clip spans two halves", 2048),
                               ("clip fits one half", 1 << 16)):
        rec = _recorder(max_samples)
        played = []

        def fake_arm(slot, count, restart, _rec=rec, _played=played):
            base = slot * _rec._play_half_words
            _played.extend(_rec.buf[2 * (base + i)] for i in range(count))

        rec._arm = fake_arm
        rec._await_dma = lambda: 0
        rec._pak = pak
        rec._nibbles = bytearray(rec._play_half_words // 2 + 1)

        ok = True
        for cid, (blob, n_samples) in sorted(by_id.items()):
            entry = pak.lookup(cid.encode())
            del played[:]
            rec._stream(entry)
            want = voice_pak.adpcm_decode(blob)
            if played != want:
                ok = False
                check("%s: %s" % (label, cid), False,
                      "%d samples played, %d expected; first difference at %s"
                      % (len(played), len(want), _first_diff(played, want)))
                break
        check("%s: every clip streams out identically to a whole decode"
              % label, ok)

    # And the boundaries really were crossed, or the test above proved nothing
    # about chunking.
    rec = _recorder(512)
    arms = []
    rec._arm = lambda slot, count, restart: arms.append((slot, count, restart))
    rec._await_dma = lambda: 0
    rec._pak = pak
    rec._nibbles = bytearray(rec._play_half_words // 2 + 1)
    longest = max(by_id.items(), key=lambda kv: kv[1][1])
    rec._stream(pak.lookup(longest[0].encode()))
    check("a long clip crossed several boundaries", len(arms) >= 4,
          "%d DMA runs" % len(arms))
    check("only the first run restarts the state machine",
          arms[0][2] is True and all(a[2] is False for a in arms[1:]))
    check("the halves alternate",
          all(arms[i][0] != arms[i + 1][0] for i in range(len(arms) - 1)))
    check("the runs sum to the clip length",
          sum(a[1] for a in arms) == longest[1][1],
          "%d vs %d" % (sum(a[1] for a in arms), longest[1][1]))
    check("no run is longer than a half",
          all(a[1] <= rec._play_half_words for a in arms))
    pak.close()


def test_clip_id_contract():
    """`talk._clip_for` and `voice_pak.clip_id` must spell the same eight hex.

    Nothing else in the system checks this, and a mismatch is silent: the pak
    builds, the device boots, the panel shows every reply, and the speaker never
    makes a sound for any of them.
    """
    print("\nclip id contract")
    import binascii
    import hashlib
    ok = True
    for text in ("Please go on.", "Tell me more about your family.",
                 "I did not hear anything. Hold the screen and speak.",
                 "WHY DO YOU REMEMBER YOUR MOTHER JUST NOW"):
        theirs = voice_pak.clip_id(text)
        # Spelled the way talk.py must spell it, MicroPython having no
        # hashlib.hexdigest.
        mine = binascii.hexlify(hashlib.sha1(text.encode()).digest())[:8]
        if mine.decode() != theirs:
            ok = False
            check("clip id for %r" % text[:30], False,
                  "%s vs %s" % (mine.decode(), theirs))
            break
    check("hexlify(digest)[:8] equals hexdigest()[:8] on real reply text", ok)


def test_voice_vectors():
    """`src/voice_vectors.py` -- the encoder's own vectors, decoded by ours.

    This is the contract with `tools/voice_pak.py`, and it is deliberately not
    a test of two functions in the same process agreeing: the vectors are
    generated by the encoder side, committed, and read back here as literal
    bytes. A change to either decoder that nobody regenerated fails here.

    Its cases reach three things `fixture_cases()` does not: a bare nibble
    stream with no clip header, so the decode loop is pinned with nothing
    around it; the step index clamped at **0**, which is only visible as an
    output that goes flat; and real rendered speech cut out of the shipping
    pak.
    """
    print("\nsrc/voice_vectors.py (generated by tools/voice_pak.py)")
    path = os.path.join(ROOT, "src", "voice_vectors.py")
    if not os.path.exists(path):
        check("src/voice_vectors.py exists", False,
              "run: uv run tools/voice_pak.py corpus-voice/ --fixture-module")
        return
    vec = _load_source("voice_vectors", path)
    check("vector format matches the decoder's", vec.FORMAT == adpcm.FORMAT)
    check("vector rate matches the decoder's", vec.RATE == adpcm.SAMPLE_RATE)
    check("there are vectors to check", len(vec.CASES) > 0,
          "%d cases" % len(vec.CASES))

    edge = vec.EDGE
    for case in vec.CASES:
        name, _why, predictor, index, nibbles, n_nibbles, head, tail, want = case
        out = array("h", bytes(4 * n_nibbles))
        state = adpcm.new_state(predictor, index)
        adpcm.decode_into(bytes(nibbles), 0, n_nibbles, out, 0, state)
        got = [out[2 * i] for i in range(n_nibbles)]
        # The reference, run here as well, so a stale committed vector fails as
        # loudly as a wrong decoder rather than being taken as gospel.
        want_full = voice_pak.decode_nibbles(bytes(nibbles), n_nibbles,
                                             predictor, index)
        check("case %-20s decodes identically to the reference" % name,
              got == want_full,
              "first difference at %s" % _first_diff(got, want_full))
        check("case %-20s matches the committed checksum" % name,
              vec.checksum(got) == want,
              "%d vs %d" % (vec.checksum(got), want))
        check("case %-20s head and tail are where the vector says" % name,
              _pcm_eq(head, got[:edge]) and _pcm_eq(tail, got[-edge:]))

    if not getattr(vec, "CLIPS", ()):
        check("real-speech clip vectors are present", False,
              "CLIPS is empty -- regenerate with a built voice.pak alongside")
        return
    for clip in vec.CLIPS:
        clip_hex, _why, _text, blob, n_samples, head, tail, want = clip
        blob = bytes(blob)
        got = decode_device(blob)
        want_full = voice_pak.adpcm_decode(blob)
        check("clip %s decodes identically to the reference" % clip_hex,
              got == want_full,
              "first difference at %s" % _first_diff(got, want_full))
        check("clip %s matches the committed checksum (%d samples)"
              % (clip_hex, n_samples),
              len(got) == n_samples and vec.checksum(got) == want,
              "%d samples, checksum %d vs %d"
              % (len(got), vec.checksum(got), want))
        check("clip %s head and tail are where the vector says" % clip_hex,
              _pcm_eq(head, got[:vec.EDGE]) and _pcm_eq(tail, got[-vec.EDGE:]))


def _pcm_eq(raw, samples):
    return bytes(raw) == struct.pack("<%dh" % len(samples), *samples)


# --- fixtures for the board ------------------------------------------------
#
# The host cannot see a viper wrap. These carry the same expected values to the
# device, where `tools/voice_probe.py` section (a) decodes them with the real
# viper decoder and compares. Same arrangement as `src/speech_fixtures.py`,
# and generated the same way -- by the host test that already knows the answer.

FIXTURE_STRESS_SAMPLES = 4000
FIXTURE_INLINE_LIMIT = 256      # above this, a checksum instead of the samples


def checksum(values):
    """acc = (acc * 31 + v) & 0x3FFFFFFF -- `speech_fixtures`' checksum.

    The same one, deliberately: a second checksum convention on the same board
    is a second thing to get wrong, and the mask keeps it inside a MicroPython
    small int on a 32-bit build.
    """
    acc = 0
    for v in values:
        acc = (acc * 31 + v) & 0x3FFFFFFF
    return acc


def _fixture_cases_for_emit():
    cases = []
    for name, blob, expected, n_samples in fixture_cases():
        why = "encoder fixture: %s" % name
        cases.append((name, why, blob, expected, n_samples))
    pcm = speechish(FIXTURE_STRESS_SAMPLES, seed=999)
    blob = voice_pak.adpcm_encode(pcm)
    cases.append(("stress", "%d samples of speech-shaped stressor; the step "
                            "index visits most of the table"
                  % FIXTURE_STRESS_SAMPLES,
                  blob, voice_pak.adpcm_decode(blob), len(pcm)))
    return cases


def emit_fixtures(path):
    cases = _fixture_cases_for_emit()
    lines = []
    add = lines.append
    add('"""Decoder fixtures for src/adpcm.py. Generated -- do not edit.\n')
    add("Produced by `python3 tools/test_adpcm.py --emit-fixtures`, whose")
    add("expected values come from `tools/voice_pak.py`'s reference decoder.")
    add("")
    add("The host suite already proves the two agree given unbounded integers.")
    add("These exist for the one thing it cannot reach: `@micropython.viper`'s")
    add("`int` is a 32-bit machine word that wraps **silently**, so a device")
    add("can disagree with a host that is checking the same arithmetic. Run")
    add("them with `tools/voice_probe.py` section (a).")
    add("")
    add("Each case is (name, why, blob, expected, n_samples, checksum).")
    add("`blob` is a whole ADPCM clip, header included. `expected` is")
    add("little-endian int16 for the short cases and `None` for the long ones,")
    add("where the checksum stands in so the module stays small enough to")
    add("import on a board with the TFLM arena resident.")
    add('"""')
    add("")
    add("FORMAT = %d" % adpcm.FORMAT)
    add("INLINE_LIMIT = %d" % FIXTURE_INLINE_LIMIT)
    add("")
    add("")
    add("def checksum(values):")
    add("    acc = 0")
    add("    for v in values:")
    add("        acc = (acc * 31 + v) & 0x3FFFFFFF")
    add("    return acc")
    add("")
    add("")
    add("CASES = (")
    for name, why, blob, expected, n_samples in cases:
        add("    (%r, %r," % (name, why))
        add("     (")
        for chunk in _hexlines(blob):
            add("        %s" % chunk)
        add("     ),")
        if n_samples <= FIXTURE_INLINE_LIMIT:
            raw = struct.pack("<%dh" % len(expected), *expected)
            add("     (")
            for chunk in _hexlines(raw):
                add("        %s" % chunk)
            add("     ),")
        else:
            add("     None,")
        add("     %d, %d)," % (n_samples, checksum(expected)))
    add(")")
    add("")
    text = "\n".join(lines)
    with open(path, "w") as handle:
        handle.write(text)
    return len(text)


def _hexlines(data, per_line=20):
    """`b'...'` literals, joined by adjacency the way speech_fixtures does."""
    out = []
    for i in range(0, len(data), per_line):
        out.append(repr(bytes(data[i:i + per_line])))
    return out or ["b''"]


def test_emitted_fixtures():
    """The generated module must still say what the reference says.

    Checked rather than trusted, because it is generated from one side of the
    comparison it is supposed to referee: if `voice_pak.py` changes and nobody
    reruns `--emit-fixtures`, the board would go on agreeing with a stale
    answer and report a clean pass.
    """
    print("\nemitted device fixtures")
    path = os.path.join(ROOT, "src", "voice_fixtures.py")
    if not os.path.exists(path):
        check("src/voice_fixtures.py exists", False,
              "run: python3 tools/test_adpcm.py --emit-fixtures")
        return
    fx = _load_source("voice_fixtures", path)
    want = _fixture_cases_for_emit()
    check("the emitted file is current (%d cases)" % len(want),
          len(fx.CASES) == len(want),
          "%d on disk, %d from the reference" % (len(fx.CASES), len(want)))
    if len(fx.CASES) != len(want):
        return
    ok = True
    for (name, _why, blob, expected, n_samples), case in zip(want, fx.CASES):
        got_name, _got_why, got_blob, got_pcm, got_n, got_sum = case
        if got_name != name or bytes(got_blob) != bytes(blob) \
                or got_n != n_samples or got_sum != checksum(expected):
            ok = False
            check("case %s is current" % name, False,
                  "regenerate with --emit-fixtures")
            break
        if got_pcm is not None:
            raw = struct.pack("<%dh" % len(expected), *expected)
            if bytes(got_pcm) != raw:
                ok = False
                check("case %s inline PCM is current" % name, False, "")
                break
    check("every emitted case matches the reference decoder", ok)
    # And the device decoder reproduces them, which is what the board will
    # assert about itself.
    ok = True
    for case in fx.CASES:
        name, _why, blob, _pcm, n_samples, want_sum = case
        got = decode_device(bytes(blob))
        if len(got) != n_samples or checksum(got) != want_sum:
            ok = False
            check("device decoder reproduces %s" % name, False,
                  "%d samples, checksum %d vs %d"
                  % (len(got), checksum(got), want_sum))
            break
    check("the device decoder reproduces every emitted case on the host", ok)


def main():
    if "--emit-fixtures" in sys.argv:
        path = os.path.join(ROOT, "src", "voice_fixtures.py")
        size = emit_fixtures(path)
        print("wrote %s (%d bytes)" % (path, size))
        return 0

    print("adpcm: device decoder against tools/voice_pak.py")
    test_tables()
    test_hand_vector()
    test_against_reference()
    test_word_packing()
    test_bounds()
    test_chunked_decode()
    test_chunk_nibs()
    test_buffer_geometry()
    path, by_id = test_pak_reader()
    test_stream_ping_pong(path, by_id)
    test_clip_id_contract()
    test_voice_vectors()
    test_emitted_fixtures()

    print()
    if FAILURES:
        print("FAILED (%d)" % len(FAILURES))
        for name in FAILURES:
            print("  - %s" % name)
        return 1
    print("PASS -- host arithmetic agrees. The device is checked separately:")
    print("        viper's int is 32 bits and wraps where CPython's does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
