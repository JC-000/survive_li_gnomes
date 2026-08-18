#!/usr/bin/env python3
"""Fixed-point MFCC front end -- the reference implementation.

This is the authority. `docs/speech.md` describes what this file does, and the
device-side `@micropython.viper` port must produce *bit-identical* output for
identical input. Anything that would make the two disagree is a bug here, not a
liberty there.

Three rules follow from that, and they are why this file looks the way it does:

1. **No floats on the feature path.** Not for speed on the host -- it is for
   agreement. MicroPython builds `float` as single precision on RP2, so any
   `math.cos` in the signal path would round differently to CPython's double and
   the two implementations would silently diverge. Floats appear only in
   `build_tables()`, which runs on the host and freezes its results into
   `src/speech_tables.py`.
2. **No numpy, no list comprehensions, no generator tricks in the core.** Every
   loop here has a direct viper transcription. `mfcc_float()` at the bottom is a
   separate, deliberately independent float model used only to check that the
   fixed-point path is doing the arithmetic anyone would expect; nothing calls
   it in anger.
3. **Every shift is an arithmetic shift.** CPython's `>>` floors, ARM's ASR
   floors, and viper's `int` is a signed machine word -- so `-3 >> 1 == -2` on
   both sides. Never write `// 2` where `>> 1` is meant; they differ on negatives
   only in the direction that bites later.

Usage:

    python3 tools/mfcc.py --selftest          # fixed vs float, and bound checks
    python3 tools/mfcc.py --emit-tables       # regenerate src/speech_tables.py
    python3 tools/mfcc.py --describe          # print every constant

No third-party packages: this runs under the system python3.
"""

import array
import math
import os
import sys

# --- The contract ----------------------------------------------------------
# Changing any of these invalidates every recorded template. Bump
# TEMPLATE_FORMAT and re-record.

SAMPLE_RATE = 16000
FRAME_LEN = 400          # 25 ms
FRAME_STRIDE = 160       # 10 ms, so 100 frames per second
FFT_SIZE = 512           # next power of two above FRAME_LEN
N_BINS = FFT_SIZE // 2 + 1   # 257 usable magnitude bins, DC to Nyquist

PREEMPH_Q15 = 31785      # round(0.97 * 32768)

N_MEL = 26
# Full band, after measuring the alternative rather than assuming it.
#
# Band-limiting to roughly the telephone band is the standard channel-robustness
# move: the extremes are where two microphones disagree most. Measured
# cross-microphone it does work -- 200-5000 recovers 41/70 top-1 against the
# full band's 36/70, and 300-3400 gives 42/70.
#
# But **deltas subsume it entirely**. With deltas enabled, 100-7600 and
# 200-5000 both score exactly 43/70 cross-microphone. The band-limit adds
# nothing on top, and it costs something real: it discards the 5-8 kHz region
# where /s/ and /f/ live, which is what separates FATHER from MOTHER and from
# "other" -- and "other" -> FATHER is the most dangerous false fire in the
# corpus. The ZCR endpointing pass exists specifically to keep those fricative
# onsets in the segment; discarding them in frequency afterwards would undo it.
MEL_LOW_HZ = 100.0       # below this is mains hum and handling rumble
MEL_HIGH_HZ = 7600.0     # above this is the codec's anti-alias roll-off

N_CEPS = 12              # c1..c12; c0 is dropped, see docs/speech.md
LOG_Q = 8                # log-mel is Q8 log2: 1 LSB = 1/256 octave
FEAT_SHIFT = 4           # features are Q4 log2: 1 LSB = 1/16 octave ~= 0.19 dB
LIFTER_L = 0             # 0 disables cepstral liftering; see --selftest notes
MEL_FLOOR_SHIFT = 0      # 0 disables; else floor each band at peak >> this
DELTA_WIDTH = 2          # 0 disables deltas; else the regression half-window
DELTA_SHIFT = 2          # deltas use FEAT_SHIFT - this, to equalise their
                         # weight against the statics in an L1 distance:
                         # measured raw, mean |delta| is 0.26x mean |static|,
                         # so an unweighted L1 would let the channel-robust
                         # half of the vector carry a quarter of the vote.
                         # Two bits brings 8 up to 32 against a static 34.
DELTA_Q15 = 0            # reciprocal of 2*sum(n^2), filled in by _init_delta()


def n_feat():
    """Features per frame: statics, plus deltas if they are enabled."""
    return N_CEPS * (2 if DELTA_WIDTH else 1)


def _delta_recip():
    """Q15 reciprocal of the regression denominator 2*sum(n^2), n=1..D.

    A multiply by a frozen reciprocal rather than a divide: viper has no
    integer divide worth using, and this way host and device round identically.
    """
    if not DELTA_WIDTH:
        return 0
    denom = 2 * sum(n * n for n in range(1, DELTA_WIDTH + 1))
    return (32768 + denom // 2) // denom

TEMPLATE_FORMAT = 2

_TABLES_HEADER = "speech_tables"


# --- Table construction (host only, floats allowed) ------------------------

def hz_to_mel(f):
    """HTK mel scale. Slaney's is piecewise-linear below 1 kHz and gives
    different filter edges; the device must use the same one as this file."""
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_window():
    """Hamming in Q15. Symmetric (N-1 denominator), matching HTK and scipy's
    `sym=True`, not the periodic form numpy's `np.hamming` shares with it."""
    out = array.array("H", bytes(2 * FRAME_LEN))
    for n in range(FRAME_LEN):
        w = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (FRAME_LEN - 1))
        out[n] = int(round(w * 32767.0))
    return out


def build_twiddles():
    """Forward-DFT twiddles W_k = exp(-2*pi*i*k/N) in Q15, k < N/2.

    32767 stands in for 1.0. That costs 0.003% of amplitude and buys the
    guarantee that no coefficient overflows an int16, which the overflow proof
    in docs/speech.md leans on.
    """
    half = FFT_SIZE // 2
    re = array.array("h", bytes(2 * half))
    im = array.array("h", bytes(2 * half))
    for k in range(half):
        angle = 2.0 * math.pi * k / FFT_SIZE
        re[k] = int(round(32767.0 * math.cos(angle)))
        im[k] = int(round(-32767.0 * math.sin(angle)))
    return re, im


def build_melbank():
    """Triangular mel filterbank as (start bin, length, flat Q15 weights).

    Stored per filter rather than per bin because the device loop is then a
    plain contiguous walk over both the magnitude array and the weight array,
    which is what viper is good at.
    """
    m_low = hz_to_mel(MEL_LOW_HZ)
    m_high = hz_to_mel(MEL_HIGH_HZ)
    edges = []
    for i in range(N_MEL + 2):
        edges.append(mel_to_hz(m_low + (m_high - m_low) * i / (N_MEL + 1)))

    bin_hz = float(SAMPLE_RATE) / FFT_SIZE
    starts = array.array("H", bytes(2 * N_MEL))
    lens = array.array("H", bytes(2 * N_MEL))
    weights = array.array("H")
    for i in range(N_MEL):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        first, last = None, None
        row = []
        for k in range(N_BINS):
            f = k * bin_hz
            if f <= lo or f >= hi:
                continue
            if f <= mid:
                w = (f - lo) / (mid - lo)
            else:
                w = (hi - f) / (hi - mid)
            q = int(round(w * 32768.0))
            if q <= 0:
                continue
            if q > 32768:
                q = 32768
            if first is None:
                first = k
            last = k
            row.append(q)
        if first is None:  # a triangle narrower than one bin; must not happen
            raise ValueError("mel filter %d spans no FFT bin" % i)
        starts[i] = first
        lens[i] = last - first + 1
        for q in row:
            weights.append(q)
    return starts, lens, weights, edges


def build_log2_table():
    """65 entries of round(256 * log2(1 + i/64)), for linear interpolation.

    Linear interpolation of log2 over a 1/64 interval errs by at most
    (h^2/8)*max|f''| = 4.4e-5 octaves = 0.011 Q8 LSB, so the table's own
    +/-0.5 LSB rounding dominates and the interpolation is effectively free.
    """
    out = array.array("H", bytes(2 * 65))
    for i in range(65):
        out[i] = int(round((1 << LOG_Q) * math.log2(1.0 + i / 64.0)))
    return out


def build_dct():
    """DCT-II rows for c1..c12 in Q15. Unnormalised: the orthonormal scaling is
    a constant per row and DTW never compares one coefficient against another,
    so it would only cost precision."""
    out = array.array("h", bytes(2 * N_CEPS * N_MEL))
    for j in range(1, N_CEPS + 1):
        for i in range(N_MEL):
            v = math.cos(math.pi * j * (i + 0.5) / N_MEL)
            out[(j - 1) * N_MEL + i] = int(round(32767.0 * v))
    return out


def build_lifter():
    """Cepstral lifter gains in Q12 (4096 == unity).

    Unity is exact through the fixed-point path: (c * 4096) >> 12 == c for
    every c, negatives included, because the shift is arithmetic. So leaving
    LIFTER_L at 0 costs nothing but a no-op multiply, and the stage stays in
    place for anyone who wants to measure it.
    """
    out = array.array("H", bytes(2 * N_CEPS))
    for j in range(1, N_CEPS + 1):
        if LIFTER_L <= 0:
            g = 1.0
        else:
            g = 1.0 + (LIFTER_L / 2.0) * math.sin(math.pi * j / LIFTER_L)
        out[j - 1] = int(round(g * 4096.0))
    return out


class Tables(object):
    """Everything the fixed-point path reads. Built once."""

    def __init__(self):
        self.window = build_window()
        self.tw_re, self.tw_im = build_twiddles()
        self.mel_start, self.mel_len, self.mel_w, self.mel_edges = build_melbank()
        self.log2 = build_log2_table()
        self.dct = build_dct()
        self.lifter = build_lifter()
        self.brev = build_bitrev()


def build_bitrev():
    """Bit-reversal permutation for FFT_SIZE, as a table.

    A table rather than the usual reversal loop: it is 512 bytes, and it keeps
    the device's inner loop free of the bit-twiddling that viper would have to
    do per element anyway.
    """
    bits = FFT_SIZE.bit_length() - 1
    out = array.array("H", bytes(2 * FFT_SIZE))
    for i in range(FFT_SIZE):
        r = 0
        v = i
        for _ in range(bits):
            r = (r << 1) | (v & 1)
            v >>= 1
        out[i] = r
    return out


_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = Tables()
    return _TABLES


# --- Bound checking --------------------------------------------------------
# CPython ints never overflow, so the host cannot notice a value that would
# wrap on the device. This flag makes it notice.

CHECK_BOUNDS = False
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_peak = {}


def _chk(name, v):
    if not CHECK_BOUNDS:
        return v
    if v < _INT32_MIN or v > _INT32_MAX:
        raise OverflowError("%s = %d does not fit int32" % (name, v))
    a = v if v >= 0 else -v
    if a > _peak.get(name, 0):
        _peak[name] = a
    return v


def peak_report():
    """Largest magnitude seen at each checked stage, as a fraction of int32."""
    rows = []
    for name in sorted(_peak):
        rows.append((name, _peak[name], _peak[name] / float(_INT32_MAX)))
    return rows


# --- The fixed-point front end ---------------------------------------------

def preemphasise(samples, out=None):
    """y[n] = sat16(x[n] - (0.97 * x[n-1])), y[-1] taken as x[0].

    Saturating to int16 is the load-bearing part. Without it a full-scale
    signal alternating at 8 kHz produces |y| up to 64551, and the very next
    stage (window, Q15) would then reach 2.115e9 -- inside int32 by 1.5% and
    not somewhere to be standing. Saturation bounds it at 32768*32767 = 1.07e9
    with a factor of two spare. Real speech never comes near either: the
    selftest reports how often saturation actually fires (it is zero).
    """
    n = len(samples)
    if out is None:
        out = array.array("h", bytes(2 * n))
    prev = samples[0] if n else 0
    for i in range(n):
        x = samples[i]
        y = x - ((PREEMPH_Q15 * prev) >> 15)
        prev = x
        if y > 32767:
            y = 32767
        elif y < -32768:
            y = -32768
        out[i] = y
    return out


def isqrt(v):
    """floor(sqrt(v)) for v >= 0, exactly. math.isqrt is exact, and the
    device's bitwise restoring square root is exact, so the two agree."""
    return math.isqrt(v)


def fft512(re, im, t):
    """In-place radix-2 decimation-in-time FFT, one right shift per stage.

    The unconditional shift keeps |X| <= 32767 at every stage: if |A| and |B|
    are within the disc of radius M then so is |A +- W*B| / 2. That in turn
    bounds the twiddle products at 32767*32767*(|cos|+|sin|) <= 1.519e9, which
    fits int32 with 29% to spare. The price is 9 bits of the input's dynamic
    range, which `mfcc_frame` pays back by block-normalising the frame first
    and folding the shift into the log.
    """
    tw_re, tw_im = t.tw_re, t.tw_im
    n = FFT_SIZE
    size = 2
    while size <= n:
        half = size >> 1
        step = n // size
        i = 0
        while i < n:
            k = 0
            j = i
            end = i + half
            while j < end:
                l = j + half
                wr = tw_re[k]
                wi = tw_im[k]
                lr = re[l]
                li = im[l]
                tr = _chk("fft.tw", wr * lr - wi * li + 16384) >> 15
                ti = _chk("fft.tw", wr * li + wi * lr + 16384) >> 15
                jr = re[j]
                ji = im[j]
                re[l] = (jr - tr + 1) >> 1
                im[l] = (ji - ti + 1) >> 1
                re[j] = (jr + tr + 1) >> 1
                im[j] = (ji + ti + 1) >> 1
                k += step
                j += 1
            i += size
        size <<= 1


def log2_q8(v, t):
    """round-free Q8 log2 of a positive int, by table plus linear interpolation.

    v is clamped to 1 rather than special-cased, so a digitally silent mel bin
    lands at exactly 0 instead of needing a sentinel.
    """
    if v < 1:
        v = 1
    e = v.bit_length() - 1
    if e >= 16:
        m = v >> (e - 16)
    else:
        m = v << (16 - e)
    # m is 1.xxxx in Q16, so m >> 10 is 64..127
    idx = (m >> 10) - 64
    frac = m & 1023
    tab = t.log2
    lo = tab[idx]
    hi = tab[idx + 1]
    return (e << LOG_Q) + lo + (((hi - lo) * frac + 512) >> 10)


def mfcc_frame(frame, t, work=None):
    """One 400-sample frame of pre-emphasised int16 -> 12 raw cepstra (Q8 log2).

    Raw means before liftering and before mean normalisation, both of which
    need the whole utterance.
    """
    if work is None:
        work = new_work()
    re, im, mag, mel = work

    # Window, and find the block-floating-point shift in the same pass.
    win = t.window
    peak = 0
    for n in range(FRAME_LEN):
        v = (frame[n] * win[n] + 16384) >> 15
        re[n] = v
        a = v if v >= 0 else -v
        if a > peak:
            peak = a
    for n in range(FRAME_LEN, FFT_SIZE):
        re[n] = 0
    for n in range(FFT_SIZE):
        im[n] = 0

    # g left-shifts the frame so its peak lands in [16384, 32767]. Without it a
    # quiet frame would lose most of its bits to the FFT's nine stage shifts.
    if peak == 0:
        g = 0
    else:
        g = 14 - (peak.bit_length() - 1)
        if g < 0:
            g = 0
    if g:
        for n in range(FRAME_LEN):
            re[n] = re[n] << g

    # Bit-reversal permutation into place, then the transform.
    brev = t.brev
    for n in range(FFT_SIZE):
        b = brev[n]
        if b > n:
            re[n], re[b] = re[b], re[n]
    fft512(re, im, t)

    # Magnitude, not power. Power would need 15 bits of headroom the mel
    # accumulator does not have; magnitude keeps the quiet bins alive and only
    # halves the log, which is a constant factor DTW never sees.
    for k in range(N_BINS):
        r = re[k]
        i2 = im[k]
        p = _chk("mag.sq", r * r + i2 * i2)
        m = isqrt(p)
        if p - m * m > m:   # round to nearest; the floor's -0.5 bias is
            m += 1          # 17% of a quiet bin and it does not cancel
        # The FFT's complex-magnitude invariant, checked rather than asserted.
        # |X| bounds re and im *jointly*; treating them as independently able
        # to reach 32767 gives re^2+im^2 = 2.147e9 (99.99% of int32), which is
        # not reachable -- it would need |X| = 46341. The real ceiling is
        # 400 non-zero inputs of at most 32767 each, divided by 512: 25599.
        _chk("fft.absmax", m)
        mag[k] = m

    # Triangular mel filterbank.
    starts, lens, mw = t.mel_start, t.mel_len, t.mel_w
    ofs = 0
    top = 0
    for i in range(N_MEL):
        s = starts[i]
        n = lens[i]
        acc = 0
        for k in range(n):
            acc += _chk("mel.mul", mag[s + k] * mw[ofs + k] + 128) >> 8
        _chk("mel.acc", acc)
        ofs += n
        mel[i] = acc
        if acc > top:
            top = acc

    # Optional spectral floor. The FFT's stage shifts leave a noise floor that
    # is a large fraction of a band 45 dB or more below the frame's peak, and
    # taking the log of that amplifies pure quantisation noise into the
    # cepstrum. Clamping costs real detail, so it is off unless measured to
    # help -- see the sweep in tools/dtw.py --tune.
    if MEL_FLOOR_SHIFT:
        floor = top >> MEL_FLOOR_SHIFT
        for i in range(N_MEL):
            if mel[i] < floor:
                mel[i] = floor

    for i in range(N_MEL):
        # 2^(2-g) undoes the FFT's nine stage shifts, the block normalisation
        # and the Q7 the accumulator carries, so the log is of the true
        # magnitude and frames of different loudness stay comparable.
        mel[i] = log2_q8(mel[i], t) + ((2 - g) << LOG_Q)

    # DCT-II. Shifting each term rather than the sum keeps the accumulator
    # inside int32; the cost is under a thousandth of a Q8 LSB.
    dct = t.dct
    out = [0] * N_CEPS
    for j in range(N_CEPS):
        row = j * N_MEL
        acc = 0
        for i in range(N_MEL):
            acc += _chk("dct.mul", mel[i] * dct[row + i] + 16384) >> 15
        out[j] = _chk("dct.acc", acc)
    return out


def new_work():
    return (array.array("i", bytes(4 * FFT_SIZE)),
            array.array("i", bytes(4 * FFT_SIZE)),
            array.array("i", bytes(4 * N_BINS)),
            array.array("i", bytes(4 * N_MEL)))


def frame_count(n_samples):
    if n_samples < FRAME_LEN:
        return 0
    return 1 + (n_samples - FRAME_LEN) // FRAME_STRIDE


def mfcc(samples, t=None):
    """int16 samples -> feature rows. The whole pipeline.

    `mfcc_q8` stops one step earlier, at the post-CMN Q8 statics, which is what
    a statics-only template stores.
    """
    return features_from_q8(mfcc_q8(samples, t))


def mfcc_q8(samples, t=None):
    """int16 samples -> post-CMN statics in Q8 log2, before the Q4 rescale.

    Pre-emphasis, framing, per-frame cepstra, liftering, mean normalisation.
    Deltas and the Q8 -> Q4 rescale happen in `features_from_q8`.
    """
    if t is None:
        t = tables()
    n_frames = frame_count(len(samples))
    if n_frames == 0:
        return []

    pre = preemphasise(samples)
    work = new_work()
    frame = array.array("h", bytes(2 * FRAME_LEN))
    raw = []
    for f in range(n_frames):
        base = f * FRAME_STRIDE
        for n in range(FRAME_LEN):
            frame[n] = pre[base + n]
        raw.append(mfcc_frame(frame, t, work))

    lift = t.lifter
    for f in range(n_frames):
        row = raw[f]
        for j in range(N_CEPS):
            row[j] = _chk("lift", row[j] * lift[j] + 2048) >> 12

    # Per-utterance mean normalisation. Truncating toward zero, not flooring,
    # because that is what a hardware divide gives and the device has no
    # cheap floor-divide.
    for j in range(N_CEPS):
        total = 0
        for f in range(n_frames):
            total += raw[f][j]
        _chk("cmn.sum", total)
        if total >= 0:
            mean = total // n_frames
        else:
            mean = -((-total) // n_frames)
        for f in range(n_frames):
            raw[f][j] = raw[f][j] - mean
    return raw


def features_from_q8(q8):
    """Q8 post-CMN statics -> the 24-wide Q4 feature rows.

    Split out of `mfcc()` because the device can rebuild the entire feature
    vector from the statics alone, and this is the function its
    `expand_in_place()` has to reproduce exactly.

    Deltas are taken here, from the *Q8* statics. That is the whole reason a
    statics-only template stores Q8 and not the Q4 the features use: deltas
    computed from Q4 values are quantised twice and do not match this. The
    mean normalisation having already happened is harmless -- CMN subtracts a
    constant per coefficient and a constant cancels identically in
    `c[t+n] - c[t-n]`.
    """
    n_frames = len(q8)
    dif = deltas(q8) if DELTA_WIDTH else None
    shift = FEAT_SHIFT - DELTA_SHIFT
    out = []
    for f in range(n_frames):
        row = [0] * N_CEPS
        for j in range(N_CEPS):
            row[j] = (q8[f][j] + (1 << (FEAT_SHIFT - 1))) >> FEAT_SHIFT
        if dif is not None:
            # Statics first, then deltas, so a 12-wide reader sees the same
            # first half.
            for j in range(N_CEPS):
                row.append((dif[f][j] + (1 << (shift - 1))) >> shift)
        out.append(row)
    return out


def deltas(raw):
    """Regression deltas over a +/-DELTA_WIDTH window, in place-safe form.

    d[t] = sum(n * (c[t+n] - c[t-n])) / (2 * sum(n^2))

    Deltas are computed from the *pre-CMN* statics deliberately, and it costs
    nothing to do so: cepstral mean normalisation subtracts a constant per
    coefficient, and a constant cancels exactly in `c[t+n] - c[t-n]`. Which is
    the whole point of having them -- a fixed channel or gain offset vanishes
    from a frame-to-frame difference *exactly*, with no estimate involved,
    whereas CMN has to estimate the offset from half a second of audio in which
    the offset and the phonetic content are not separable.

    Edge frames replicate rather than shrink the window, which is what HTK
    does; the alternative is a delta whose scale changes at the edges.
    """
    n_frames = len(raw)
    recip = _delta_recip()
    out = []
    for t in range(n_frames):
        row = [0] * N_CEPS
        for j in range(N_CEPS):
            acc = 0
            for n in range(1, DELTA_WIDTH + 1):
                a = t + n
                b = t - n
                if a > n_frames - 1:
                    a = n_frames - 1
                if b < 0:
                    b = 0
                acc += n * (raw[a][j] - raw[b][j])
            row[j] = _chk("delta", acc * recip + 16384) >> 15
        out.append(row)
    return out


def pack_template(frames):
    """Frames -> little-endian uint16 blob, biased by +32768.

    The bias is what lets the device read templates with `ptr16` and subtract
    directly: the offset cancels in a difference, so no sign extension is
    needed in the DTW inner loop.
    """
    width = n_feat()
    out = bytearray(2 * width * len(frames))
    p = 0
    for row in frames:
        for j in range(width):
            v = row[j] + 32768
            if v < 0:
                v = 0
            elif v > 65535:
                v = 65535
            out[p] = v & 0xFF
            out[p + 1] = v >> 8
            p += 2
    return bytes(out)


def pack_statics(q8):
    """Q8 post-CMN statics -> little-endian uint16 blob, biased by +32768.

    Half the size of a full template, because the deltas are derivable. See
    `expand_all()`. Returns (blob, n_clamped) -- a non-zero clamp count means
    the recording was loud enough to overflow int16 in Q8 and the template is
    no longer bit-exact, which the caller must not swallow.

    Headroom measured over the enrolment corpus: largest |value| 14511 against
    the 32767 limit, so 2.3x. Comfortable, not enormous, which is why the count
    comes back rather than being asserted away.
    """
    out = bytearray(2 * N_CEPS * len(q8))
    clamped = 0
    p = 0
    for row in q8:
        for j in range(N_CEPS):
            v = row[j]
            if v > 32767:
                v = 32767
                clamped += 1
            elif v < -32768:
                v = -32768
                clamped += 1
            v += 32768
            out[p] = v & 0xFF
            out[p + 1] = v >> 8
            p += 2
    return bytes(out), clamped


def statics_bytes(n_frames):
    return 2 * N_CEPS * n_frames


def expand_all(buf, index, scratch=None):
    """Grow a blob of packed statics into full feature rows, inside one buffer.

    `buf` is `2 * n_feat() * total_frames` bytes with all the packed statics
    already read into its front. `index` is `(label, frame_offset, n_frames)`
    per template, in ascending frame order. On return `buf` holds the full
    template layout and nothing else was ever allocated.

    This is what lets the device store 74 KB and match on 148 KB without
    holding both. The correctness argument is short on purpose, because the
    clever version was not:

    - Templates are expanded **last to first**. Template `k` writes to
      `[48*F_k, 48*(F_k+n_k))` while the statics still needed belong to
      templates `0..k` and live below `24*F_k + 24*n_k`. Writing starts at
      `48*F_k`, and `48*F_k >= 24*F_k` always, so nothing below is touched.
    - Each template's statics are copied into `scratch` first, so the source
      of the frame being written is never the buffer being written to. That
      removes the within-template overlap entirely rather than reasoning about
      it -- deltas reach two frames either side, and an in-place version has to
      stash the low frames and argue about where the boundary falls. 1.5 KB of
      scratch buys the argument away.

    Deltas must not bleed across template boundaries, which is the other reason
    this is per-template rather than one pass over the whole blob: each
    template's edges replicate its own first and last frame.

    Verified against `mfcc()` on every corpus template by
    `tools/test_templates.py`. Do not change it without re-running that.
    """
    if scratch is None:
        longest = 0
        for _label, _off, n in index:
            if n > longest:
                longest = n
        scratch = bytearray(statics_bytes(longest))

    recip = _delta_recip()
    shift = FEAT_SHIFT - DELTA_SHIFT
    width = n_feat()

    for k in range(len(index) - 1, -1, -1):
        _label, frame_off, n_frames = index[k]
        src = statics_bytes(frame_off)
        count = statics_bytes(n_frames)
        for i in range(count):
            scratch[i] = buf[src + i]

        for i in range(n_frames - 1, -1, -1):
            p = 2 * width * (frame_off + i)
            for j in range(N_CEPS):
                q = 2 * (N_CEPS * i + j)
                v = (scratch[q] | (scratch[q + 1] << 8)) - 32768
                v = (v + (1 << (FEAT_SHIFT - 1))) >> FEAT_SHIFT
                v += 32768
                buf[p] = v & 0xFF
                buf[p + 1] = v >> 8
                p += 2
            if not DELTA_WIDTH:
                continue
            for j in range(N_CEPS):
                acc = 0
                for d in range(1, DELTA_WIDTH + 1):
                    a = i + d
                    b = i - d
                    if a > n_frames - 1:
                        a = n_frames - 1
                    if b < 0:
                        b = 0
                    qa = 2 * (N_CEPS * a + j)
                    qb = 2 * (N_CEPS * b + j)
                    acc += d * ((scratch[qa] | (scratch[qa + 1] << 8))
                                - (scratch[qb] | (scratch[qb + 1] << 8)))
                v = (acc * recip + 16384) >> 15
                v = (v + (1 << (shift - 1))) >> shift
                v += 32768
                if v < 0:
                    v = 0
                elif v > 65535:
                    v = 65535
                buf[p] = v & 0xFF
                buf[p + 1] = v >> 8
                p += 2
    return buf


def unpack_template(blob):
    width = n_feat()
    n = len(blob) // (2 * width)
    frames = []
    p = 0
    for _ in range(n):
        row = [0] * width
        for j in range(width):
            row[j] = (blob[p] | (blob[p + 1] << 8)) - 32768
            p += 2
        frames.append(row)
    return frames


# --- Float model, for checking only ----------------------------------------

def mfcc_float(samples):
    """The same pipeline in floating point, modelling the same constants.

    Deliberately reproduces the fixed-point path's *choices* -- Q15 window,
    magnitude mel, log2, unnormalised DCT -- but none of its quantisation, so
    the difference between the two is exactly the fixed-point error and nothing
    else. If this and `mfcc()` diverge by more than a couple of Q4 LSBs, the
    integer path has a bug.
    """
    t = tables()
    n_frames = frame_count(len(samples))
    if n_frames == 0:
        return []

    pre = []
    prev = float(samples[0])
    for i in range(len(samples)):
        x = float(samples[i])
        y = x - (PREEMPH_Q15 / 32768.0) * prev
        prev = x
        pre.append(max(-32768.0, min(32767.0, y)))

    out = []
    for f in range(n_frames):
        base = f * FRAME_STRIDE
        re = [0.0] * FFT_SIZE
        im = [0.0] * FFT_SIZE
        for n in range(FRAME_LEN):
            re[n] = pre[base + n] * (t.window[n] / 32768.0)
        _fft_float(re, im)
        mag = []
        for k in range(N_BINS):
            mag.append(math.hypot(re[k], im[k]))
        mel = []
        ofs = 0
        for i in range(N_MEL):
            s, n = t.mel_start[i], t.mel_len[i]
            acc = 0.0
            for k in range(n):
                acc += mag[s + k] * (t.mel_w[ofs + k] / 32768.0)
            ofs += n
            mel.append((1 << LOG_Q) * math.log2(max(acc, 1.0)))
        row = []
        for j in range(1, N_CEPS + 1):
            acc = 0.0
            for i in range(N_MEL):
                acc += mel[i] * math.cos(math.pi * j * (i + 0.5) / N_MEL)
            row.append(acc * (t.lifter[j - 1] / 4096.0))
        out.append(row)

    for j in range(N_CEPS):
        mean = sum(out[f][j] for f in range(n_frames)) / n_frames
        for f in range(n_frames):
            out[f][j] = (out[f][j] - mean) / (1 << FEAT_SHIFT)
    return out


def _fft_float(re, im):
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    size = 2
    while size <= n:
        half = size >> 1
        ang = -2.0 * math.pi / size
        for i in range(0, n, size):
            for k in range(half):
                wr = math.cos(ang * k)
                wi = math.sin(ang * k)
                l = i + k + half
                tr = wr * re[l] - wi * im[l]
                ti = wr * im[l] + wi * re[l]
                re[l] = re[i + k] - tr
                im[l] = im[i + k] - ti
                re[i + k] += tr
                im[i + k] += ti
        size <<= 1


# --- Table emission --------------------------------------------------------

def _blob(arr):
    return bytes(memoryview(arr).cast("B"))


def _fmt_bytes(name, data, width=18):
    lines = ["%s = (" % name]
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        lines.append("    b'" + "".join("\\x%02x" % b for b in chunk) + "'")
    lines.append(")")
    return "\n".join(lines)


def emit_tables(path):
    t = tables()
    parts = ['"""Frozen DSP tables for the keyword spotter. Generated -- do not edit.',
             "",
             "Produced by tools/mfcc.py --emit-tables. They are frozen rather than",
             "computed at import because MicroPython's float is single precision on",
             "RP2: cos() there does not round the way CPython's does, and the device",
             "must agree with the host bit for bit.",
             "",
             "All blobs are little-endian and are read with ptr16/ptr8 in viper.",
             '"""',
             "",
             "SAMPLE_RATE = %d" % SAMPLE_RATE,
             "FRAME_LEN = %d" % FRAME_LEN,
             "FRAME_STRIDE = %d" % FRAME_STRIDE,
             "FFT_SIZE = %d" % FFT_SIZE,
             "N_BINS = %d" % N_BINS,
             "N_MEL = %d" % N_MEL,
             "N_CEPS = %d" % N_CEPS,
             "PREEMPH_Q15 = %d" % PREEMPH_Q15,
             "LOG_Q = %d" % LOG_Q,
             "FEAT_SHIFT = %d" % FEAT_SHIFT,
             "TEMPLATE_FORMAT = %d" % TEMPLATE_FORMAT,
             "",
             "# The delta stage. DELTA_Q15 is the Q15 reciprocal of the",
             "# regression denominator 2*sum(n^2); a multiply by a frozen",
             "# reciprocal rather than a divide, so both sides round alike.",
             "DELTA_WIDTH = %d" % DELTA_WIDTH,
             "DELTA_SHIFT = %d" % DELTA_SHIFT,
             "DELTA_Q15 = %d" % _delta_recip(),
             "",
             "# Derived, but emitted rather than recomputed: a device that",
             "# calculates N_FEAT from DELTA_WIDTH itself is a second copy of",
             "# the rule, and this project has spent a week removing those.",
             "N_FEAT = %d" % n_feat(),
             "",
             "MEL_FLOOR_SHIFT = %d" % MEL_FLOOR_SHIFT,
             "LIFTER_L = %d" % LIFTER_L,
             "",
             "# Informational: the filterbank is already baked into MEL_START,",
             "# MEL_LEN and MEL_W_Q15 above. These are here so a device can",
             "# report what front end it is running.",
             "MEL_LOW_HZ = %g" % MEL_LOW_HZ,
             "MEL_HIGH_HZ = %g" % MEL_HIGH_HZ,
             ""]
    for name, arr in (("WINDOW_Q15", t.window),
                      ("TWIDDLE_RE_Q15", t.tw_re),
                      ("TWIDDLE_IM_Q15", t.tw_im),
                      ("BITREV", t.brev),
                      ("MEL_START", t.mel_start),
                      ("MEL_LEN", t.mel_len),
                      ("MEL_W_Q15", t.mel_w),
                      ("LOG2_Q8", t.log2),
                      ("DCT_Q15", t.dct),
                      ("LIFTER_Q12", t.lifter)):
        parts.append(_fmt_bytes(name, _blob(arr)))
        parts.append("")
    text = "\n".join(parts)
    with open(path, "w") as fh:
        fh.write(text)
    return len(text)


# --- Selftest --------------------------------------------------------------

def _test_signal(n=6400, seed=12345):
    """A deterministic pseudo-speech signal: three formant-ish tones under an
    amplitude envelope, plus a little noise. Not speech, but it exercises the
    same dynamic range and does not need an audio file to exist."""
    out = array.array("h", bytes(2 * n))
    s = seed
    for i in range(n):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        noise = (s >> 16) % 401 - 200
        env = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / n)
        v = (7000.0 * math.sin(2.0 * math.pi * 320.0 * i / SAMPLE_RATE)
             + 3500.0 * math.sin(2.0 * math.pi * 1180.0 * i / SAMPLE_RATE)
             + 1500.0 * math.sin(2.0 * math.pi * 2600.0 * i / SAMPLE_RATE))
        out[i] = max(-32768, min(32767, int(v * env) + noise))
    return out


def describe():
    t = tables()
    print("MFCC contract (TEMPLATE_FORMAT %d)" % TEMPLATE_FORMAT)
    print("  %d Hz mono int16, frame %d (%d ms), stride %d (%d ms)"
          % (SAMPLE_RATE, FRAME_LEN, FRAME_LEN * 1000 // SAMPLE_RATE,
             FRAME_STRIDE, FRAME_STRIDE * 1000 // SAMPLE_RATE))
    print("  FFT %d -> %d bins of %.2f Hz" % (FFT_SIZE, N_BINS,
                                              SAMPLE_RATE / FFT_SIZE))
    print("  pre-emphasis %d/32768 = %.5f" % (PREEMPH_Q15, PREEMPH_Q15 / 32768))
    print("  window: Hamming Q15, min %d max %d" % (min(t.window), max(t.window)))
    print("  mel: %d filters, %.0f-%.0f Hz, HTK scale" % (N_MEL, MEL_LOW_HZ,
                                                          MEL_HIGH_HZ))
    print("  mel weights stored: %d (%d bytes)" % (len(t.mel_w), 2 * len(t.mel_w)))
    widths = [t.mel_len[i] for i in range(N_MEL)]
    print("  filter widths in bins: min %d, max %d" % (min(widths), max(widths)))
    print("  filter edges (Hz): " + ", ".join("%.0f" % e for e in t.mel_edges[:4])
          + " ... " + ", ".join("%.0f" % e for e in t.mel_edges[-3:]))
    print("  cepstra kept: c1..c%d, c0 dropped" % N_CEPS)
    print("  log-mel Q%d log2, features Q%d log2 (1 LSB = %.3f dB)"
          % (LOG_Q, LOG_Q - FEAT_SHIFT,
             20 * math.log10(2 ** (1 / (1 << (LOG_Q - FEAT_SHIFT))))))
    print("  lifter L = %d (%s)" % (LIFTER_L, "off" if LIFTER_L <= 0 else "on"))
    print("  mel floor: %s" % ("off" if not MEL_FLOOR_SHIFT
                               else "peak >> %d (%.0f dB)"
                               % (MEL_FLOOR_SHIFT, 6.02 * MEL_FLOOR_SHIFT)))
    total = (2 * len(t.window) + 2 * len(t.tw_re) + 2 * len(t.tw_im)
             + 2 * len(t.brev) + 2 * N_MEL * 2 + 2 * len(t.mel_w)
             + 2 * len(t.log2) + 2 * len(t.dct) + 2 * len(t.lifter))
    print("  frozen tables: %d bytes" % total)
    print("  deltas: %s" % ("off" if not DELTA_WIDTH
                            else "D=%d, weighted x%d" % (DELTA_WIDTH,
                                                        1 << DELTA_SHIFT)))
    print("  features per frame: %d (%d bytes)" % (n_feat(), 2 * n_feat()))


# Provable ceilings, as fractions of int32. See the overflow proof in
# docs/speech.md. `fft.tw` is bounded by 32767.71 * 32767 via Cauchy-Schwarz
# and the FFT's complex-magnitude invariant; `mag.sq` by |X| <= 400*32767/512.
CEILINGS = {"fft.tw": 32768 * 32767,
            "mag.sq": 25599 * 25599,
            "fft.absmax": 25599}


def _load_source(path, name):
    """Execute a generated module from its **source text**, every time.

    Deliberately not `importlib`. These loads exist to detect stale generated
    files, and `importlib` consults `__pycache__` -- so the check meant to
    catch stale state could be answered by stale state, and silently pass.
    That is not hypothetical: it happened once here, and the tell was a
    selftest that failed, then passed, with the file restored in between.

    `exec(compile(source))` reads the bytes on disk on every call and writes
    no cache. Slower by microseconds, and it cannot lie.
    """
    with open(path, "r") as fh:
        source = fh.read()
    namespace = {"__name__": name, "__file__": path}
    exec(compile(source, path, "exec"), namespace)
    return _Namespace(namespace)


class _Namespace(object):
    """Attribute access over an exec'd module namespace."""

    def __init__(self, mapping):
        self.__dict__.update(mapping)


def _selftest_tables_current():
    """Fail if src/speech_tables.py is stale against this file.

    The emitted tables are a second copy of everything in here, and the failure
    mode is the quiet one: tables regenerated from a different `mfcc.py` than
    the device compares against produce features that are wrong by a little,
    everywhere, forever. Nothing downstream notices -- the device's own tests
    would compare a stale table against a stale table and agree.

    So this rebuilds every table in memory and compares. It is the mechanism
    that makes the device-side constant pins redundant: they can only catch a
    mistyped constant, and there is nothing left to mistype.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "src", _TABLES_HEADER + ".py")
    if not os.path.exists(path):
        print("no src/%s.py; run tools/mfcc.py --emit-tables" % _TABLES_HEADER)
        return True

    st = _load_source(path, "_st")

    t = tables()
    bad = []
    for name, arr in (("WINDOW_Q15", t.window),
                      ("TWIDDLE_RE_Q15", t.tw_re),
                      ("TWIDDLE_IM_Q15", t.tw_im),
                      ("BITREV", t.brev),
                      ("MEL_START", t.mel_start),
                      ("MEL_LEN", t.mel_len),
                      ("MEL_W_Q15", t.mel_w),
                      ("LOG2_Q8", t.log2),
                      ("DCT_Q15", t.dct),
                      ("LIFTER_Q12", t.lifter)):
        want = _blob(arr)
        got = getattr(st, name, None)
        if got != want:
            bad.append("%s (%d bytes emitted, %d rebuilt)"
                       % (name, len(got) if got else 0, len(want)))
    for name, want in (("SAMPLE_RATE", SAMPLE_RATE), ("FRAME_LEN", FRAME_LEN),
                       ("FRAME_STRIDE", FRAME_STRIDE), ("FFT_SIZE", FFT_SIZE),
                       ("N_BINS", N_BINS), ("N_MEL", N_MEL),
                       ("N_CEPS", N_CEPS), ("PREEMPH_Q15", PREEMPH_Q15),
                       ("LOG_Q", LOG_Q), ("FEAT_SHIFT", FEAT_SHIFT),
                       ("TEMPLATE_FORMAT", TEMPLATE_FORMAT),
                       ("DELTA_WIDTH", DELTA_WIDTH),
                       ("DELTA_SHIFT", DELTA_SHIFT),
                       ("DELTA_Q15", _delta_recip()),
                       ("N_FEAT", n_feat()),
                       ("MEL_FLOOR_SHIFT", MEL_FLOOR_SHIFT),
                       ("LIFTER_L", LIFTER_L)):
        if getattr(st, name, None) != want:
            bad.append("%s (%r emitted, %r here)"
                       % (name, getattr(st, name, None), want))
    if bad:
        print("src/%s.py is STALE against tools/mfcc.py:" % _TABLES_HEADER)
        for b in bad:
            print("    %s" % b)
        print("  run: python3 tools/mfcc.py --emit-tables")
        print("  then regenerate fixtures and re-enrol -- every template built")
        print("  against the emitted tables is invalid.")
        return False
    print("src/%s.py is current with this file (10 tables, 17 constants)"
          % _TABLES_HEADER)
    return True


def _selftest_fixtures():
    """Run the generated fixtures through the bounds check.

    They exist for the device port, but they are also the only inputs here that
    reach the tight part of the FFT. Returns True if every stage stayed inside
    both int32 and its own proved ceiling.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "src", "speech_fixtures.py")
    if not os.path.exists(path):
        print("no src/speech_fixtures.py; run tools/make_fixtures.py")
        return True

    fx = _load_source(path, "_fx")

    _peak.clear()
    for name, _why, pcm, _expected in fx.CASES:
        buf = array.array("h", bytes(len(pcm)))
        for i in range(0, len(pcm), 2):
            v = pcm[i] | (pcm[i + 1] << 8)
            buf[i // 2] = v - 65536 if v > 32767 else v
        mfcc(buf)

    ok = True
    print("fixtures: worst stage over %d cases" % len(fx.CASES))
    for name in sorted(_peak):
        peak = _peak[name]
        frac = peak / float(_INT32_MAX)
        ceiling = CEILINGS.get(name)
        note = ""
        if ceiling:
            note = "  ceiling %d (%.2f%%), using %.1f%% of it" % (
                ceiling, 100.0 * ceiling / _INT32_MAX, 100.0 * peak / ceiling)
            if peak > ceiling:
                ok = False
                note += "  FAIL: past its proved ceiling"
        print("  %-10s %12d  %5.2f%% of int32%s" % (name, peak, 100 * frac, note))
        if frac > 0.95:
            ok = False
            print("    FAIL: no headroom left")
    return ok


def selftest():
    global CHECK_BOUNDS
    CHECK_BOUNDS = True
    t = tables()
    ok = True

    # Tables are self-consistent.
    assert len(t.window) == FRAME_LEN
    assert len(t.tw_re) == FFT_SIZE // 2
    assert sum(t.mel_len[i] for i in range(N_MEL)) == len(t.mel_w)
    assert len(t.dct) == N_CEPS * N_MEL
    for i in range(N_MEL):
        assert t.mel_start[i] + t.mel_len[i] <= N_BINS

    # log2_q8 against math.log2.
    worst = 0
    for v in [1, 2, 3, 7, 255, 256, 1000, 65535, 65536, 1 << 20, (1 << 21) - 1]:
        got = log2_q8(v, t)
        want = (1 << LOG_Q) * math.log2(v)
        worst = max(worst, abs(got - want))
    print("log2_q8 worst error: %.3f Q8 LSB (%.5f octaves)" % (worst, worst / 256))
    if worst > 1.0:
        ok = False
        print("  FAIL: log2 table is not accurate enough")

    # Pre-emphasis saturation should never fire on plausible audio.
    sig = _test_signal()
    pre = preemphasise(sig)
    sat = sum(1 for i in range(len(pre)) if pre[i] in (32767, -32768))
    print("pre-emphasis saturated on %d of %d samples of the test signal"
          % (sat, len(sig)))

    # Fixed vs float.
    fx = mfcc(sig, t)
    fl = mfcc_float(sig)
    assert len(fx) == len(fl)
    worst = 0.0
    total = 0.0
    n = 0
    for f in range(len(fx)):
        for j in range(N_CEPS):   # statics only; mfcc_float has no deltas
            d = abs(fx[f][j] - fl[f][j])
            worst = max(worst, d)
            total += d
            n += 1
    print("fixed vs float over %d frames x %d coeffs: mean |err| %.3f, worst %.3f"
          % (len(fx), N_CEPS, total / n, worst))
    if DELTA_WIDTH:
        print("  (statics only -- the float model has no delta stage; deltas"
              " are exact\n   differences of the statics, so they add no new"
              " quantisation)")
    print("  (units are Q4 log2 LSBs; typical |feature| is %d)"
          % (sum(abs(v) for row in fx for v in row) // n))
    # The residual is the scaled FFT's noise floor reaching the log of a
    # low-energy mel band, not a coding error, and it is *deterministic*: the
    # device runs the same integer path, so templates and queries carry the
    # same distortion and most of it cancels in the DTW distance. What matters
    # is the end-to-end number from tools/dtw.py --eval, which is measured
    # both ways. This bound only catches gross breakage.
    if worst > 40.0:
        ok = False
        print("  FAIL: fixed-point path disagrees with the float model")

    # Round trip through the packed template form.
    blob = pack_template(fx)
    back = unpack_template(blob)
    assert back == fx, "template pack/unpack is not lossless"
    print("template pack round-trips: %d frames, %d bytes (%d bytes/frame)"
          % (len(fx), len(blob), 2 * N_CEPS))

    # The synthetic signal above is not the worst case for overflow -- real
    # speech drives the FFT twiddle far harder, because pre-emphasis saturation
    # clamps a pathological input before the transform ever sees it. So the
    # fixtures, which include the worst frame found in the whole corpus, are
    # run through the same bounds check. Without this the selftest would report
    # a comfortable 41.9% for a stage that actually reaches 49.9%.
    ok = _selftest_fixtures() and ok
    ok = _selftest_tables_current() and ok

    # Overflow headroom.
    print("int32 headroom by stage:")
    for name, peak, frac in peak_report():
        print("  %-10s peak %12d  %5.1f%% of int32" % (name, peak, 100 * frac))
        if frac > 0.95:
            ok = False
            print("    FAIL: too close to the limit")

    CHECK_BOUNDS = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if "--emit-tables" in argv:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "src", _TABLES_HEADER + ".py")
        size = emit_tables(path)
        print("wrote %s (%d bytes)" % (path, size))
        return 0
    if "--describe" in argv:
        describe()
        return 0
    if "--selftest" in argv or len(argv) == 1:
        describe()
        print()
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
