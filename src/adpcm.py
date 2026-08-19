"""IMA ADPCM decoding, and reading the `voice.pak` container clips live in.

4-bit IMA ADPCM, mono, 16 kHz. The host renders every line the device will ever
speak (`docs/speech-voice.md` says why runtime synthesis was ruled out),
encodes it, and ships one `voice.pak`. This module is the only thing on the
device that knows either format.

**`tools/voice_pak.py` is the normative side of both.** It writes the file and
carries the reference encoder and decoder; this is a port of its `adpcm_decode`
and `read_pak`, not a second design -- the same relationship `spotter.py` has
with `tools/mfcc.py`. `tools/test_adpcm.py` holds them together on the host
against the encoder's own fixtures, and `Pak.selftest()` re-runs those fixtures
on the board, which is the only run that can see the failure that matters.

**Why a module of its own rather than more of `listen.py`.** Everything here is
integer arithmetic and file offsets: no `machine`, no `rp2`, no codec. That is
what lets the real decoder run under CPython with no board attached and be
compared byte for byte with the encoder's, rather than two prose descriptions
of IMA ADPCM being trusted to have meant the same thing. Folding it into
`listen.py` would put it behind a `machine` import and cost exactly that test.

## The decode step, exactly

There are two IMA ADPCM variants in the wild and they are **not** bit-identical:
the sum-of-shifts form below, and `diff = ((2 * (code & 7) + 1) * step) >> 3`,
which differs by an LSB on many codes because this one truncates per term.

    step = STEP_TABLE[index]
    diff = step >> 3
    if code & 4: diff += step
    if code & 2: diff += step >> 1
    if code & 1: diff += step >> 2
    pred = pred - diff if code & 8 else pred + diff
    clamp pred to int16
    index = clamp(index + INDEX_TABLE[code & 7], 0, 88)

Nibbles are **low first**: sample 2k in bits 0-3 of byte k, 2k+1 in bits 4-7.
A decoder that reads them the other way round produces noise of exactly the
right length, which is why the order is asserted rather than assumed.

**Sample 0 of a clip is the header's predictor and is not encoded**, so a clip
of n samples carries n-1 nibbles. `listen` writes that first sample with
`emit_sample()` and then decodes the rest.

## The output is PIO words, not samples

`decode_into` writes `(s << 16) | s` -- the packed 32-bit stereo frame
`audio_pio_out` pulls, per docs/hardware.md. Decoding straight into that shape
means one pass instead of decode-then-pack, and it means the decode target *is*
the play buffer with no intermediate. Both halfwords hold the same sample, so
the low halfword of word `i` is sample `i` and a test reads the output as plain
int16 without unpacking anything.

## int32 headroom -- the part CPython cannot check

The device runs the `@micropython.viper` transcription at the bottom of this
file, where `int` is a 32-bit machine word that wraps **silently**. CPython's
ints are unbounded, so neither the host test nor a plain-path device run can
see a wrap. The proof, stated rather than assumed:

    step                  <= 32767                 STEP_TABLE[88], the largest
    diff  = (step >> 3) + step + (step >> 1) + (step >> 2)
                          <= 4095 + 32767 + 16383 + 8191 = 61436
    pred before clamping  in [-32768 - 61436, 32767 + 61436]
                          =  [-94204, 94203]

which is 0.0044% of int32. Nothing here is tight -- unlike the FFT in
`spotter.py`, which runs at 99.8% of its bound and needed its shifts kept in
order. `CHECK_BOUNDS` asserts it anyway on the host, because a table edited by
hand is exactly how `step` stops being <= 32767.

**One value wraps on purpose.** `(s << 16) | s` at full scale is 0xFFFF_FFFF,
negative as a signed machine word. It is a bit pattern stored through a
`ptr32`, not an arithmetic result, and the store keeps the low 32 bits either
way. The plain path writes the same bytes as two halfwords, which is why it
takes an `array("h")` and not an `array("I")`.
"""

from array import array

# Mirrors tools/voice_pak.py's VERSION. `Pak.open` refuses a file it does not
# know: a decoder quietly mis-reading a newer container would present as "the
# voice sounds like static sometimes", which is not a symptom that points here.
FORMAT = 1
SAMPLE_RATE = 16000

# The standard IMA tables, byte-identical to tools/voice_pak.py's. 89 steps;
# the last is 32767, which is what the headroom proof above rests on.
STEP_TABLE = array("i", (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
))
STEP_MAX_INDEX = 88

# Sixteen entries, not the reference's eight indexed by `code & 7`. Same values
# mirrored: it removes one mask from the inner loop and cannot disagree with
# the eight-entry form, which `tools/test_adpcm.py` checks element by element.
INDEX_TABLE = array("i", (
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
))


def new_state(predictor=0, index=0):
    """The decoder state, as the two ints viper can reach through a ptr32.

    `array("i")` and not a tuple because `decode_into` has to write it back: a
    viper function cannot return a pair, which is what carries the state across
    the chunk boundaries streaming playback decodes in.
    """
    return array("i", (predictor, index))


def emit_sample(out, out_off, value):
    """Write one already-decoded sample as a packed word. Sample 0's home.

    Here rather than in `listen` so the `(s << 16) | s` convention lives in one
    file: two places that pack for the PIO is two places to get an octave-low
    clip out of, which docs/hardware.md records happening once already.
    """
    o = 2 * out_off
    out[o] = value
    out[o + 1] = value


# --- The host-only overflow check ------------------------------------------
#
# Same device as `tools/mfcc.py`'s: CPython cannot notice a value that would
# wrap on the board, so this makes it notice. Off by default, because the plain
# path below is also the fallback on a device without viper and must not pay
# for it. `tools/test_adpcm.py` turns it on for every comparison it makes.

CHECK_BOUNDS = False
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_peak = {}


def _chk(name, v):
    if v < _INT32_MIN or v > _INT32_MAX:
        raise OverflowError("%s = %d does not fit int32" % (name, v))
    a = v if v >= 0 else -v
    if a > _peak.get(name, 0):
        _peak[name] = a
    return v


def peak_report():
    """Largest magnitude reached at each checked stage since `reset_peaks`."""
    return dict(_peak)


def reset_peaks():
    _peak.clear()


# --- The decoder, portable -------------------------------------------------
#
# This is the specification, and the only version CPython ever sees. The viper
# transcription at the bottom shadows it on the device.


def decode_into(src, src_off, nib_count, out, out_off, state):
    """Decode `nib_count` nibbles into packed PIO words. Returns nothing.

    `src`      bytes-like holding the nibble stream
    `src_off`  byte offset of the first nibble's byte (nibble 0 is its LOW half)
    `out`      an `array("h")`; word `i` occupies elements `2i` and `2i+1`
    `out_off`  destination in **words**, so `out[2 * out_off]` is the first
               sample decoded. Words, not samples: the DMA counts this buffer in
               words and MicroPython counts it in halfwords, and mixing those
               two units is exactly how the "burst of static" bug happened.
    `state`    from `new_state()`, read at entry and written back at exit --
               which is what lets one clip be decoded in chunks of any size.

    Nothing here bounds-checks `out`. A decode that ran off the end of the
    capture buffer would corrupt the heap silently, so `listen.speak` derives
    every count from the buffer's own length and never from the file's.
    """
    pred = state[0]
    index = state[1]
    o = 2 * out_off
    for i in range(nib_count):
        byte = src[src_off + (i >> 1)]
        if i & 1:
            code = (byte >> 4) & 15
        else:
            code = byte & 15

        step = STEP_TABLE[index]
        diff = step >> 3
        if code & 4:
            diff += step
        if code & 2:
            diff += step >> 1
        if code & 1:
            diff += step >> 2
        if CHECK_BOUNDS:
            _chk("diff", diff)
        if code & 8:
            pred -= diff
        else:
            pred += diff
        if CHECK_BOUNDS:
            _chk("pred_raw", pred)
        if pred > 32767:
            pred = 32767
        elif pred < -32768:
            pred = -32768

        index += INDEX_TABLE[code]
        if index < 0:
            index = 0
        elif index > STEP_MAX_INDEX:
            index = STEP_MAX_INDEX

        # Both halves of the packed word, as two int16s. Identical bytes to the
        # viper path's single `(s << 16) | s` store on a little-endian machine,
        # and unlike that store it fits an array("h").
        out[o] = pred
        out[o + 1] = pred
        o += 2

    state[0] = pred
    state[1] = index


# --- The container ---------------------------------------------------------
#
# Written by tools/voice_pak.py::write_pak, whose comments are normative:
#
#   header  <4sHHII>  magic "VPAK", version, flags, count, rate      16 bytes
#   index   <8sIII>   id (8 ascii hex), offset, length, n_samples    20 bytes
#   blobs   4-byte aligned, in index order; the index is sorted by id
#   blob    <IhBB>    n_samples, predictor, step index, reserved      8 bytes
#                     then (n_samples // 2) nibble bytes
#
# The index is **bisected on flash and never held in RAM**, which is the reason
# the ids are 8 ascii characters rather than 4 packed bytes: `talk._clip_for`
# already has the id in that form and can compare it without parsing. At 113
# clips a lookup is ~7 reads of 20 bytes; flash reads at 9,320 kB/s
# (docs/hardware.md), so it costs microseconds and no heap at all.
#
# **Do not "optimise" this into a dict read once at bind.** That is the obvious
# cleanup, it is what this file's author proposed first, and it is wrong here:
# a dict of 113 entries is tens of KB of small objects on a heap that never
# compacts and that `talk.reserve()` already carves by hand to place three
# blocks. The cost it would save is microseconds per press. The encoder's
# author chose the flash bisect and was right; the format is theirs
# (`tools/voice_pak.py` is normative) and this is a port of it, not a design.
# The same heap has already eaten the activation chirp twice and the TFLM
# arena once -- see the reservation ordering in `talk.reserve()`.

MAGIC = b"VPAK"
HEADER_LEN = 16
ENTRY_LEN = 20
CLIP_HEADER_LEN = 8


def _u16(b, i):
    return b[i] | (b[i + 1] << 8)


def _u32(b, i):
    return b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24)


class Pak:
    """`voice.pak`: the index stays on flash, the audio streams off it.

    Opened once and held open for the life of the program -- `open()` costs one
    16-byte read, not a scan.

    Never raises out of `open()` or `lookup()`. No pak, a truncated pak, or a
    pak from a future encoder all mean the same thing to the program: the
    device answers on the panel and says nothing, which `talk` already treats
    as the normal case rather than as an error.
    """

    def __init__(self, path="voice.pak"):
        self.path = path
        self.rate = SAMPLE_RATE
        self.count = 0
        self.error = None
        self._fh = None
        # One 20-byte scratch, reused by every bisect step. Small, but a fresh
        # bytearray per probe would be ~7 allocations per press on a heap that
        # never compacts.
        self._entry = bytearray(ENTRY_LEN)

    def open(self):
        """Read and validate the header. Returns True if the pak is usable."""
        try:
            fh = open(self.path, "rb")
        except OSError as exc:
            self.error = "no %s (%s)" % (self.path, exc)
            return False
        try:
            head = fh.read(HEADER_LEN)
            if len(head) != HEADER_LEN or bytes(head[0:4]) != MAGIC:
                raise ValueError("not a voice.pak")
            version = _u16(head, 4)
            if version != FORMAT:
                raise ValueError("pak version %d, this decoder reads %d"
                                 % (version, FORMAT))
            self.count = _u32(head, 8)
            self.rate = _u32(head, 12)
            self._fh = fh
            self.error = None
            return True
        except Exception as exc:  # noqa: BLE001 -- a bad pak must not be fatal
            self.error = "%s: %s" % (type(exc).__name__, exc)
            print("voice.pak unusable (%s)" % self.error)
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
            self.count = 0
            self._fh = None
            return False

    def lookup(self, clip_id):
        """(offset, length, n_samples) for an 8-char id, or None.

        Binary search over the on-flash index, which `write_pak` sorts by id.
        `clip_id` is `bytes` or `str`; it is compared as bytes because that is
        what a read returns and what `talk._clip_for` can produce cheapest.
        """
        if self._fh is None:
            return None
        if not isinstance(clip_id, bytes):
            clip_id = clip_id.encode()
        try:
            lo = 0
            hi = self.count - 1
            fh = self._fh
            ent = self._entry
            mv = memoryview(ent)
            while lo <= hi:
                mid = (lo + hi) >> 1
                fh.seek(HEADER_LEN + ENTRY_LEN * mid)
                if fh.readinto(mv) != ENTRY_LEN:
                    return None
                here = bytes(ent[0:8])
                if here == clip_id:
                    return (_u32(ent, 8), _u32(ent, 12), _u32(ent, 16))
                if here < clip_id:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return None
        except Exception as exc:  # noqa: BLE001 -- silence beats a broken turn
            print("voice.pak lookup failed (%s: %s)" % (type(exc).__name__, exc))
            return None

    def start_clip(self, offset):
        """Seek to a clip and consume its 8-byte header.

        Returns `(state, n_samples)`. The file is then positioned at nibble 0,
        and `state[0]` is **sample 0**, which the caller emits itself before
        decoding anything -- see the module docstring.
        """
        fh = self._fh
        fh.seek(offset)
        head = fh.read(CLIP_HEADER_LEN)
        if len(head) != CLIP_HEADER_LEN:
            raise ValueError("clip header truncated")
        n_samples = _u32(head, 0)
        pred = _u16(head, 4)
        if pred > 32767:
            pred -= 65536
        return new_state(pred, head[6]), n_samples

    def readinto(self, buf, n):
        """Read up to `n` nibble bytes into `buf`. Returns how many arrived."""
        return self._fh.readinto(memoryview(buf)[:n])

    def close(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        self._fh = None


def decode_clip(pak, offset, out, out_off, limit_words):
    """Decode a whole clip into `out`, for tests and probes rather than for
    playback. Returns the sample count written, or 0.

    `listen.speak` deliberately does NOT use this -- it streams, and a clip
    longer than the buffer is normal there. This exists so the on-device
    fixture run can decode a known blob and hand the bytes back for comparison
    without also involving the codec, the DMA and the PIO.
    """
    state, n_samples = pak.start_clip(offset)
    if n_samples == 0:
        return 0
    if n_samples > limit_words:
        raise ValueError("clip is %d samples, buffer holds %d"
                         % (n_samples, limit_words))
    emit_sample(out, out_off, state[0])
    nibs = n_samples - 1
    if nibs:
        src = bytearray((nibs + 1) // 2)
        got = pak.readinto(src, len(src))
        if got != len(src):
            raise ValueError("clip body truncated: %d of %d" % (got, len(src)))
        decode_into(src, 0, nibs, out, out_off + 1, state)
    return n_samples


# --- The same decoder again, in viper --------------------------------------
#
# Speed only, and a transcription of the loop above rather than a second
# implementation: same expressions, same order. The three viper details that
# are silent when wrong are the three `spotter.py` records, and they apply here
# unchanged:
#
#   - a ptr8 load is zero-extended, which is what the nibble stream wants, and
#     a ptr32 load is a whole machine word that `int()` types signed;
#   - `>>` on a signed viper int is arithmetic and floors, matching CPython --
#     which matters here because `pred` goes negative and `diff` is compared
#     against it;
#   - `ptr32(GLOBAL)` casts a module-level array("i") without passing it in,
#     which keeps STEP_TABLE and INDEX_TABLE out of the signature.
#
# `out` is cast to ptr32 from an `array("h")`. That is safe only because
# `out_off` counts **words**: an odd halfword offset would be a misaligned
# 32-bit store. `listen` derives both half offsets from the buffer length so
# they cannot be odd, and `tools/test_adpcm.py` asserts that arithmetic.
#
# Only ImportError is caught, so this falls back to the portable path on the
# host and nowhere else. A viper compile error on the device is meant to be
# loud: silently decoding at bytecode speed is the failure this port removes.

# Reported by `tools/voice_probe.py`, not consulted by anything. A board
# silently running the bytecode decoder would still sound correct, only slower,
# which is the one failure mode here that has no symptom until a clip stutters.
VIPER = False

try:  # pragma: no cover -- device only
    import micropython

    @micropython.viper
    def decode_into(src: ptr8, src_off: int, nib_count: int,  # noqa: F811
                    out: ptr32, out_off: int, state: ptr32):
        steps = ptr32(STEP_TABLE)
        idxtab = ptr32(INDEX_TABLE)
        pred = int(state[0])
        index = int(state[1])
        o = int(out_off)
        i = 0
        while i < nib_count:
            byte = int(src[src_off + (i >> 1)])
            if i & 1:
                code = (byte >> 4) & 15
            else:
                code = byte & 15

            step = int(steps[index])
            diff = step >> 3
            if code & 4:
                diff += step
            if code & 2:
                diff += step >> 1
            if code & 1:
                diff += step >> 2
            if code & 8:
                pred -= diff
            else:
                pred += diff
            if pred > 32767:
                pred = 32767
            elif pred < -32768:
                pred = -32768

            index += int(idxtab[code])
            if index < 0:
                index = 0
            elif index > 88:
                index = 88

            # Wraps to a negative machine word at full scale, deliberately:
            # this is the PIO's frame layout, not an arithmetic result.
            v = pred & 0xFFFF
            out[o] = (v << 16) | v
            o += 1
            i += 1

        state[0] = pred
        state[1] = index

    VIPER = True

except ImportError:
    pass
