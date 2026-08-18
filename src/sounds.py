"""Waveform synthesis for the press sounds. Pure DSP -- no hardware, no I/O.

Kept free of `board` and `machine` imports so the maths can be exercised on a
host with plain CPython.

Everything is integer-only: MicroPython floats would roughly triple generation
time and none of this needs the precision. Trig is used once per clip at setup
to derive fixed-point filter coefficients, never per sample.

The shake is synthesised at the full 24 kHz output rate. The fart is synthesised
at 8 kHz and sample-and-held up by 3: it is a low-frequency sound, so generating
at full rate would cost three times as long for no audible gain, and the imaging
the hold adds above 4 kHz is inaudible on it.

Not everything here is synthesised. Sampled clips (see tools/make_clip.py) are
converted on the host and read straight off the filesystem into a buffer.
"""

import math
from array import array

SAMPLE_RATE = 24000
SYNTH_RATE = 8000
_UP = SAMPLE_RATE // SYNTH_RATE

_PEAK = 30000  # leave a little headroom below full scale

# Clip lengths, and the only place they are defined -- Shaker reserves output
# buffers from these before generating anything.
ALTERNATE_MS = {"fart": 650}


def output_frames(duration_ms):
    return SAMPLE_RATE * duration_ms // 1000


def allocate_bytes(n_bytes):
    """Reserve an output buffer.

    Worth calling early, and largest first. Two traps:

    - These are 60-105 KB contiguous and MicroPython's heap does not compact, so
      the free total can look ample while no single block that large remains.
    - `array("I", bytearray(n))` holds the bytearray *and* the array at once, so
      the transient peak is **twice** the final size. Measured: 174 KB was the
      largest free block after the shake clip, which is plenty for a 105 KB sigh
      and still not enough to build one.
    """
    return array("I", bytearray(n_bytes))


def allocate(duration_ms):
    return allocate_bytes(4 * output_frames(duration_ms))


def _xorshift(seed):
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= seed >> 17
    seed ^= (seed << 5) & 0xFFFFFFFF
    return seed


def _pack(sample):
    """int16 sample -> one packed 32-bit stereo frame (same signal both sides)."""
    if sample > 32767:
        sample = 32767
    elif sample < -32768:
        sample = -32768
    half = sample & 0xFFFF
    return (half << 16) | half


def _normalise_and_upsample(raw, out, peak=_PEAK):
    """Scale to `peak`, sample-and-hold by _UP, pack into 32-bit stereo frames.

    Normalising afterwards means the synthesis routines above can use whatever
    internal scaling keeps their maths clean, without hand-tuning a gain.

    DC is removed first. The fart's asymmetric saw leaves a large offset (~3500
    of 30000), which wastes headroom and pushes the speaker cone off centre for
    the whole clip -- audible as a thump at each end.
    """
    mean = sum(raw) // len(raw)
    if mean:
        for i in range(len(raw)):
            raw[i] -= mean

    largest = 1
    for value in raw:
        magnitude = value if value >= 0 else -value
        if magnitude > largest:
            largest = magnitude
    gain = (peak << 12) // largest

    if out is None:
        out = array("I", bytearray(4 * len(raw) * _UP))
    index = 0
    for value in raw:
        word = _pack((value * gain) >> 12)
        for _ in range(_UP):
            out[index] = word
            index += 1
    return out


def shake(duration_ms=540, bursts=3):
    """Three decaying bursts of low-passed noise: a die tumbling in liquid.

    Synthesised at the full output rate, unlike the alternates -- this one has
    real high-frequency content and the user has already signed off on how it
    sounds, so it is left exactly as it was.
    """
    frames = SAMPLE_RATE * duration_ms // 1000
    buf = array("I", bytearray(4 * frames))
    burst = frames // bursts
    seed = 0x1234567
    low = 0

    for i in range(frames):
        seed = _xorshift(seed)
        white = (seed & 0xFFFF) - 32768
        low += (white - low) >> 3

        pos = i % burst
        if pos < burst >> 4:
            env = (pos << 8) // max(1, burst >> 4)
        else:
            env = ((burst - pos) << 8) // burst
            env = (env * env) >> 8

        env = (env * (256 - (i // burst) * 60)) >> 8

        # _pack inlined: a call per sample at 24 kHz costs ~580 ms of
        # first-press latency, and this is the one clip generated at full rate.
        sample = (low * env) >> 7
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768
        half = sample & 0xFFFF
        buf[i] = (half << 16) | half

    return buf


# --- Alternates ------------------------------------------------------------
# Both are deliberately parameterised with named constants at the top of each
# function: these are novelty sounds and tuning them is a listening exercise,
# not a calculation.

def fart(out=None, duration_ms=None):
    """A descending buzz with an irregular sputter, heavily low-passed."""
    if duration_ms is None:
        duration_ms = ALTERNATE_MS["fart"]
    START_HZ = 100
    END_HZ = 55
    WOBBLE_HZ = 16
    WOBBLE_EVERY = 40  # samples between sputter updates

    frames = SYNTH_RATE * duration_ms // 1000
    raw = array("i", bytearray(4 * frames))
    seed = 0xBEEF
    phase = 0
    low1 = 0
    low2 = 0
    wobble = 0

    for i in range(frames):
        if i % WOBBLE_EVERY == 0:
            seed = _xorshift(seed)
            wobble = ((seed >> 8) % (2 * WOBBLE_HZ + 1)) - WOBBLE_HZ

        freq = START_HZ - ((START_HZ - END_HZ) * i) // frames + wobble
        if freq < 25:
            freq = 25
        phase = (phase + (freq * 65536) // SYNTH_RATE) & 0xFFFF

        # Asymmetric saw: squashing the negative half adds the buzzy harmonics
        # that make it read as a raspberry rather than a hum.
        saw = phase - 32768
        buzz = saw if saw > 0 else saw >> 1

        seed = _xorshift(seed)
        noise = (seed & 0x1FFF) - 4096

        low1 += ((buzz + (noise >> 1)) - low1) >> 2
        low2 += (low1 - low2) >> 2

        if i < frames >> 5:                      # fast attack
            env = (i << 8) // max(1, frames >> 5)
        elif i > (frames * 13) >> 4:             # decay over the last ~19%
            env = ((frames - i) << 8) // (frames - ((frames * 13) >> 4))
        else:
            env = 256

        raw[i] = (low2 * env) >> 8

    return _normalise_and_upsample(raw, out)
