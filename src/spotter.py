"""Keyword spotting on the device: MFCC front end, banded DTW, rejection gate.

Audio buffer in, one vocabulary label out, or None. `docs/speech.md` is
normative and `tools/mfcc.py` + `tools/dtw.py` are the reference
implementations; this is a port of them, not a second design.

## Bit-exactness is the requirement, not a goal

Templates are computed on the host by `tools/record_templates.py` and matched
here. If the two front ends disagree by one LSB anywhere, every template picks
up an offset that the DTW distances quietly absorb, and it presents as "the
recogniser is a bit poor" -- indistinguishable from bad enrolment, wrong mic
gain, a wrong sample rate, or the channel mismatch the whole enrolment design
was rearranged to avoid. There is no symptom that points here.

So `tools/test_spotter.py` feeds identical input to this module and to
`tools/mfcc.py` and asserts the features are **identical, not close**, and the
same for DTW distances and the template expansion. Run it after any change.
Every constant below is pinned against the reference by that test as well; none
of them is a judgement call made here.

## Written twice: plain, then in viper

Every stage of the front end and the DTW inner loop exist twice -- a plain
version, which is the specification and the only one CPython ever sees, and a
`@micropython.viper` transcription in the `try: import micropython` blocks that
shadows it on the device. The plain one was measured at **122.9 ms per frame**
on this board, so a one-second utterance would have spent ~12 s in the front end
alone. The latency budget is the pause after the user lets go of the screen,
against a panel that already takes ~583 ms to redraw.

Measured on the board, per frame:

    prepare  4.5 ms -> 0.72     fft  80.9 ms -> 3.66     mag  14.0 ms -> 0.77
    mel      4.0 ms -> 0.30     dct   2.5 ms -> 0.14     whole frame 122.9 -> 6.2

The FFT is 65% of what remains and is at viper's floor for this butterfly:
13 statements, none of them removable without changing the arithmetic. What is
left to spend is `BAND` in the matcher, which costs far more than the front end
does -- see the timings under "What a turn costs" below.

**The bounds stopped being free at that point.** CPython and MicroPython both
use arbitrary-precision ints, so nothing in the plain version can wrap; viper's
`int` is a 32-bit machine word and wraps silently, and neither the host test nor
a plain-path device run can see it happen. The proof does not carry over by
itself -- what carries it is `src/speech_fixtures.py`, which pins ten stages
per case, and `tw_stress`, a real frame at 99.8% of the twiddle bound. Run that
one first. All five cases currently agree with the host at every pinned stage:
`saturated`, `preemph`, `peak`, `g`, `windowed`, `fft`, `mag`, `mel`,
`cepstra`, `features`, plus `DTW_0_1` for the matcher.

The tightest stage is the **FFT twiddle**, and it is tighter than it looks:

    ceiling  1 073 709 056   50.00% of int32   (Cauchy-Schwarz on wr,wi / lr,li)
    measured 1 071 628 352   49.90% of int32   -- 99.8% of the ceiling used

and the measurement is real speech (`speech_fixtures.tw_stress`, `husband.1`
frame 31), not a synthetic signal. **Nothing synthetic gets near it**: an
alternating full-scale input reaches only 41.5%, because pre-emphasis saturation
clamps it before the FFT ever sees it. So a hostile test input is *milder* than
ordinary loud speech here and would give false confidence. Run `tw_stress`
first.

Two numbers, and they measure different things -- quote both or neither. The
analysis is **tight**: 99.8% of the proved ceiling is used. The hardware margin
is **a full factor of two**: that ceiling is 50.00% of int32. A correct port
does not wrap. What would break the bound:
removing or skipping one of the nine stage shifts, raising the block-float
target above 32767, or reordering `wr*lr - wi*li` so a partial sum exceeds the
final value. What would *not*: recording gain, window, mel range or sample rate,
because the block-float normalisation makes the FFT input scale-invariant.

`mag.sq` is **not** a concern. Its ceiling is **30.52%** and it is one of the
roomiest stages. The reasoning that makes it look like the tightest is recorded
here because it is a natural misreading and it will be made again:

    WRONG:  |re| <= 32767 and |im| <= 32767, so re*re + im*im <= 2.147e9,
            which is 99.99% of int32.

Three ways to see that it is wrong, the first of which needs no invariant at
all:

1. `X[k]` is a sum of 400 non-zero inputs (the frame is 400 of 512 samples, the
   rest zero-padded), each at most 32767, and the transform divides by 2^9. So
   `|X| <= 400 * 32767 / 512 = 25599` and `re^2 + im^2 <= 25599^2`, which is
   30.52% of int32. That follows from the DFT being a bounded sum.
2. The two component bounds are not independent -- both are *consequences* of
   `|X| <= 32767`. Multiplying two consequences of one constraint and treating
   the product as attainable is the error: `re^2 + im^2 = |X|^2`, and reaching
   2.147e9 would need `|X| = 46341`, which the constraint that produced both
   bounds forbids.
3. The component-wise reading contradicts a bound this pipeline already relies
   on. If `|X|` could reach 46341 then so could `mag`, and the mel stage's
   `mag * 32768 <= 1.074e9` breaks at `46341 * 32768 = 1.52e9`, 70.7% of int32.
   A bound predicting that a *different* stage overflows first, when that stage
   measures 5.16%, is not conservative. It is wrong.

Measured: `mag.sq` peaks at 0.86% over 1130 real frames, 8.86% over the
fixtures, and 8.85% under a directed search over 256 full-scale bin-centre
sinusoids -- the inputs that concentrate the most energy into one bin. Max `|X|`
after any stage is 18944 against the 25599 ceiling.

The error was reasoning about the reference implementation's *types* instead of
the pipeline's *invariant*.

None of this has to be believed: `mfcc.py` guards the invariant rather than
asserting it, and `--selftest` fails a stage past its proved ceiling:

    fft.absmax        13 793   ceiling      25 599              53.9% of it
    mag.sq       190 246 849   ceiling 655 308 801 (30.52%)     29.0% of it
    fft.tw     1 071 628 352   ceiling 1 073 709 056 (50.00%)   99.8% of it

Run that rather than assuming a port inherits any of it.

## What a turn costs

Timed on the board, viper throughout, against a synthetic 66-template set with
the corpus's own length distribution (28..61 frames, 44 mean):

| query | front end | matching | turn |
| --- | --- | --- | --- |
| 44 frames (a typical word) | 273 ms | 616 ms | ~889 ms |
| 48 frames (0.5 s of audio) | 297 ms | 672 ms | 969 ms |
| 98 frames (1.0 s of audio) | 603 ms | 562 ms | 1165 ms |

The 98-frame row matches *faster* than the 48-frame one because the duration
gate rejects 40 of the 66 templates outright, which is the gate doing its job.

**Matching dominates, as the operation counts said it would** -- roughly two to
one here rather than eight to one, because the front end's FFT is heavier per
operation than the matcher's L1 loop. The remaining levers, in order:

- **`BAND`.** Measured, 66 templates against a 50-frame query:
  **band 20 = 1146 ms, band 10 = 696 ms, band 5 = 409 ms** -- very nearly linear
  in `2*BAND + 1`, as the shape predicts. The host corpus scored 5, 10 and 20
  identically, so halving it looks free, but that corpus is saturated and the
  claim wants re-measuring against templates enrolled through this board before
  the 287 ms is banked. It is a one-line change when somebody has that evidence.
- **The duration gate**, already in place and already free.
- The DTW inner loop is unrolled four ways and is otherwise at viper's floor;
  the front end is 20x down from where it started. Neither has much left.
"""

from array import array

import speech_tables as T

# --- Constants, all read from the generated tables -------------------------
# Nothing here is written out by hand any more: `--emit-tables` carries every
# one, so the device cannot drift from the host by a typo. tools/test_spotter.py
# still asserts them against tools/mfcc.py, which now catches a *stale*
# speech_tables.py rather than a mistyped constant -- a smaller job than it was,
# and still worth having.

SAMPLE_RATE = T.SAMPLE_RATE
FRAME_LEN = T.FRAME_LEN
FRAME_STRIDE = T.FRAME_STRIDE
FFT_SIZE = T.FFT_SIZE
N_BINS = T.N_BINS
N_MEL = T.N_MEL
N_CEPS = T.N_CEPS
PREEMPH_Q15 = T.PREEMPH_Q15
LOG_Q = T.LOG_Q
FEAT_SHIFT = T.FEAT_SHIFT
TEMPLATE_FORMAT = T.TEMPLATE_FORMAT

DELTA_WIDTH = T.DELTA_WIDTH          # regression half-window; 0 disables deltas
DELTA_SHIFT = T.DELTA_SHIFT          # deltas use FEAT_SHIFT - this, weighting x4
MEL_FLOOR_SHIFT = T.MEL_FLOOR_SHIFT  # 0 disables the spectral floor

# Both emitted rather than derived, deliberately. N_FEAT is N_CEPS * 2 and
# DELTA_Q15 is the Q15 reciprocal of 2*sum(n^2) -- but a device that recomputes
# either from DELTA_WIDTH is a second copy of the rule, and a second copy is
# what this project has spent a week removing. A frozen reciprocal rather than a
# divide also means host and device round identically, and viper has no integer
# divide worth using.
N_FEAT = T.N_FEAT
DELTA_Q15 = T.DELTA_Q15

# --- Matcher parameters, mirrored from tools/dtw.py ------------------------

# Sakoe-Chiba radius in frames, about the ratio line. Measured non-binding on
# the host corpus: bands of 5, 10 and 20 scored identically, so 5 is free there.
# 10 is kept because TTS rate variation is globally uniform and a person's local
# timing is not -- but if probe section (g) says a turn is too slow, halving this
# is the first thing to try and the corpus says it costs nothing.
BAND = 10
DUR_RATIO_PCT = 200  # reject outright beyond a 2:1 length difference
INF = 1 << 29        # far above any real cost, far below int32

# The operating point, from the host sweep over the enrolment corpus:
# precision 1.000, recall 0.966 at 21 classes. Both must be re-measured against
# templates enrolled through this board's ES8311 -- which is why `spot_scored`
# exists, since a device that only prints its verdict gives nobody the numbers
# to tune with.
THRESHOLD = 750
MARGIN = 120

# Templates are stored biased by +32768 so they can be read as unsigned 16-bit
# and subtracted without sign extension. The query is biased the same way and
# the bias cancels in the difference, which is the whole point of it.
BIAS = 32768


def _decode(blob):
    """A frozen table of little-endian int16 -> array("i").

    Signed: the twiddles and the DCT are, and nothing else in the tables
    reaches 32768, so one decoder covers all of them. Costs about 10 KB across
    every table, once, and keeps the inner loops free of byte arithmetic.
    """
    n = len(blob) // 2
    out = array("i", bytes(4 * n))
    for i in range(n):
        v = blob[2 * i] | (blob[2 * i + 1] << 8)
        out[i] = v - 65536 if v > 32767 else v
    return out


WINDOW = _decode(T.WINDOW_Q15)
TW_RE = _decode(T.TWIDDLE_RE_Q15)
TW_IM = _decode(T.TWIDDLE_IM_Q15)
BITREV = _decode(T.BITREV)
MEL_START = _decode(T.MEL_START)
MEL_LEN = _decode(T.MEL_LEN)
MEL_W = _decode(T.MEL_W_Q15)
LOG2 = _decode(T.LOG2_Q8)
DCT = _decode(T.DCT_Q15)
LIFTER = _decode(T.LIFTER_Q12)


# --- The fixed-point front end ---------------------------------------------

def _isqrt(v):
    """floor(sqrt(v)), exactly. Newton, so it agrees with math.isqrt.

    The reference uses math.isqrt, which is exact; this has to be exact too or
    the magnitude spectrum differs by an LSB in the bins where it rounds.
    """
    if v <= 0:
        return 0
    guess = v
    step = (v + 1) // 2
    while step < guess:
        guess = step
        step = (guess + v // guess) // 2
    return guess


def frame_count(n_samples):
    if n_samples < FRAME_LEN:
        return 0
    return 1 + (n_samples - FRAME_LEN) // FRAME_STRIDE


def preemphasise(samples, start, count, out):
    """y[n] = sat16(x[n] - 0.97*x[n-1]), with y[-1] taken as x[0].

    The saturation is load-bearing, not defensive: without it a full-scale
    signal alternating at 8 kHz reaches |y| = 64551, and the window stage would
    then land at 2.115e9 -- inside int32 by 1.5%, which is not somewhere to be
    standing. Real speech never comes near it.
    """
    prev = samples[start] if count else 0
    for i in range(count):
        x = samples[start + i]
        y = x - ((PREEMPH_Q15 * prev) >> 15)
        prev = x
        if y > 32767:
            y = 32767
        elif y < -32768:
            y = -32768
        out[i] = y
    return out


def _fft512(re, im):
    """In-place radix-2 DIT FFT, one right shift per stage.

    The unconditional per-stage shift keeps |X| <= 32767 throughout: if |A| and
    |B| are inside the disc of radius M then so is |A +- W*B|/2, so the
    recurrence contracts rather than growing. The price is nine bits of dynamic
    range, which the block-float shift in _mfcc_frame pays back and folds into
    the log.

    **That shift is the whole overflow proof.** Two numbers, and they are
    different quantities -- state both or neither:

    - the twiddle products reach 1 071 628 352 against a proved ceiling of
      1 073 709 056, so **the analysis is tight**, 99.8% of it used;
    - that ceiling is 50.00% of int32, so **there is a full factor of two before
      anything overflows**. Tight analysis, no risk.

    Removing a stage shift, or making one conditional on the data, breaks the
    bound rather than costing precision -- and breaks it silently in viper.
    Measured max |X| after any stage is 18944 (stage 1, falling to 4300 by stage
    9), so it looks comfortable from outside: the margin is in the products, not
    the values.
    """
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
                wr = TW_RE[k]
                wi = TW_IM[k]
                lr = re[l]
                li = im[l]
                tr = (wr * lr - wi * li + 16384) >> 15
                ti = (wr * li + wi * lr + 16384) >> 15
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


def _log2_q8(v):
    """Q8 log2 of a positive int, by table plus linear interpolation.

    v is clamped to 1 rather than special-cased, so a digitally silent mel band
    lands at exactly 0 instead of needing a sentinel.
    """
    if v < 1:
        v = 1
    e = _bit_length(v) - 1
    if e >= 16:
        m = v >> (e - 16)
    else:
        m = v << (16 - e)
    idx = (m >> 10) - 64          # m is 1.xxx in Q16, so this is 0..63
    frac = m & 1023
    lo = LOG2[idx]
    hi = LOG2[idx + 1]
    return (e << LOG_Q) + lo + (((hi - lo) * frac + 512) >> 10)


def _bit_length(v):
    """int.bit_length() is present in MicroPython, but not on every build, and
    this runs 26 times per frame. Explicit beats a version check."""
    n = 0
    while v:
        v >>= 1
        n += 1
    return n


def new_work():
    """The scratch the front end needs. ~9 KB, allocated once and reused."""
    return (array("i", bytes(4 * FFT_SIZE)),   # re
            array("i", bytes(4 * FFT_SIZE)),   # im
            array("i", bytes(4 * N_BINS)),     # mag
            array("i", bytes(4 * N_MEL)),      # mel
            array("h", bytes(2 * FRAME_LEN)),  # frame
            array("i", bytes(4 * N_CEPS)))     # cepstra


# The front end is split into one function per stage of docs/speech.md rather
# than written as one long body, because that is the seam the viper port
# replaces along: each stage is overridden on its own and `speech_fixtures`
# pins each one, so a mismatch names the stage. The stages are otherwise
# unchanged, and the plain versions here stay the specification.

def _prepare(src, base, re, im):
    """Stages 2-3: window, block-float shift, bit-reverse. Returns `g`.

    Reads FRAME_LEN samples from `src` starting at `base` rather than taking a
    frame that has already been copied out. The copy was 400 assignments per
    frame for nothing -- the stride is a plain offset and every consumer of a
    frame is this function.
    """
    peak = 0
    for n in range(FRAME_LEN):
        v = (src[base + n] * WINDOW[n] + 16384) >> 15
        re[n] = v
        a = v if v >= 0 else -v
        if a > peak:
            peak = a
    for n in range(FRAME_LEN, FFT_SIZE):
        re[n] = 0
    for n in range(FFT_SIZE):
        im[n] = 0

    # g left-shifts the frame so its peak lands in [16384, 32767]; without it a
    # quiet frame loses most of its bits to the FFT's nine stage shifts.
    if peak == 0:
        g = 0
    else:
        g = 14 - (_bit_length(peak) - 1)
        if g < 0:
            g = 0
    if g:
        for n in range(FRAME_LEN):
            re[n] = re[n] << g

    for n in range(FFT_SIZE):
        b = BITREV[n]
        if b > n:
            re[n], re[b] = re[b], re[n]
    return g


def _magnitudes(re, im, mag):
    """Stage 5. Magnitude, not power: power would need 15 bits of headroom the
    mel accumulator does not have, and magnitude only halves the log, which is a
    constant factor DTW never sees."""
    for k in range(N_BINS):
        r = re[k]
        i2 = im[k]
        p = r * r + i2 * i2
        m = _isqrt(p)
        if p - m * m > m:   # round to nearest; the floor's -0.5 bias is 17% of
            m += 1          # a quiet bin and it does not cancel
        mag[k] = m


def _melbank(mag, mel, g):
    """Stages 6-7: filterbank, optional floor, Q8 log2 with the `g` correction."""
    ofs = 0
    top = 0
    for i in range(N_MEL):
        s = MEL_START[i]
        n = MEL_LEN[i]
        acc = 0
        for k in range(n):
            acc += (mag[s + k] * MEL_W[ofs + k] + 128) >> 8
        ofs += n
        mel[i] = acc
        if acc > top:
            top = acc

    if MEL_FLOOR_SHIFT:
        floor = top >> MEL_FLOOR_SHIFT
        for i in range(N_MEL):
            if mel[i] < floor:
                mel[i] = floor

    for i in range(N_MEL):
        # 2^(2-g) undoes the FFT's nine stage shifts, the block normalisation
        # and the Q7 the accumulator carries, so the log is of the true
        # magnitude and frames of different loudness stay comparable.
        mel[i] = _log2_q8(mel[i]) + ((2 - g) << LOG_Q)


def _dct(mel, out):
    """Stage 8. Shifting each term rather than the sum keeps the accumulator
    inside int32, at a cost under a thousandth of a Q8 LSB."""
    for j in range(N_CEPS):
        row = j * N_MEL
        acc = 0
        for i in range(N_MEL):
            acc += (mel[i] * DCT[row + i] + 16384) >> 15
        out[j] = acc


def _mfcc_at(src, base, work, out):
    """FRAME_LEN pre-emphasised int16 at `src[base:]` -> 12 raw cepstra.

    `out` must be an array("i"); `mfcc_frame` is the wrapper that accepts a
    plain list. Raw meaning before liftering and before mean normalisation,
    both of which need the whole utterance.
    """
    re, im, mag, mel = work[0], work[1], work[2], work[3]
    g = _prepare(src, base, re, im)
    _fft512(re, im)
    _magnitudes(re, im, mag)
    _melbank(mag, mel, g)
    _dct(mel, out)
    return out


def mfcc_frame(frame, work, out):
    """One 400-sample frame of pre-emphasised int16 -> 12 raw cepstra, Q8 log2.

    `out` may be any indexable of length N_CEPS, including a plain list, which
    is what `tools/test_spotter.py` passes; the arithmetic happens in
    `work[5]` because the viper DCT writes through a ptr32.
    """
    ceps = work[5]
    _mfcc_at(frame, 0, work, ceps)
    for j in range(N_CEPS):
        out[j] = ceps[j]
    return out


def _lifter_row(ceps, q8, off):
    """Stage 9, folded in per frame rather than as a second pass over the rows."""
    for j in range(N_CEPS):
        q8[off + j] = (ceps[j] * LIFTER[j] + 2048) >> 12


def _cmn(q8, n_frames):
    """Stage 10, in place. Truncating toward zero, not flooring, because that is
    what a hardware divide gives and the device has no cheap floor-divide -- and
    because the host does the same, which is the only reason that matters."""
    for j in range(N_CEPS):
        total = 0
        for f in range(n_frames):
            total += q8[f * N_CEPS + j]
        if total >= 0:
            mean = total // n_frames
        else:
            mean = -((-total) // n_frames)
        for f in range(n_frames):
            q8[f * N_CEPS + j] -= mean


def mfcc_q8(samples, start, count, work=None):
    """int16 samples -> post-CMN statics in Q8 log2, as a flat array("i").

    Stops one step before the features: this is what a statics-only template
    stores, and what deltas are taken from.
    """
    n_frames = frame_count(count)
    if n_frames == 0:
        return None, 0
    if work is None:
        work = new_work()
    ceps = work[5]

    pre = array("h", bytes(2 * count))
    preemphasise(samples, start, count, pre)

    q8 = array("i", bytes(4 * N_CEPS * n_frames))
    for f in range(n_frames):
        _mfcc_at(pre, f * FRAME_STRIDE, work, ceps)
        _lifter_row(ceps, q8, f * N_CEPS)
    _cmn(q8, n_frames)
    return q8, n_frames


def _feature_rows(q8, out, n_frames):
    """Stages 10-11: Q8 statics -> the biased uint16 rows DTW reads."""
    shift = FEAT_SHIFT - DELTA_SHIFT
    half = 1 << (FEAT_SHIFT - 1)
    dhalf = 1 << (shift - 1)

    for f in range(n_frames):
        src = f * N_CEPS
        dst = f * N_FEAT
        for j in range(N_CEPS):
            out[dst + j] = ((q8[src + j] + half) >> FEAT_SHIFT) + BIAS
        if not DELTA_WIDTH:
            continue
        # Deltas come off the *Q8* statics, which is the entire reason a
        # statics-only template stores Q8 and not the Q4 the features use:
        # deltas taken from Q4 values are quantised twice and do not match.
        # CMN having already happened is harmless -- it subtracts a constant per
        # coefficient, and a constant cancels identically in c[t+n] - c[t-n].
        for j in range(N_CEPS):
            acc = 0
            for d in range(1, DELTA_WIDTH + 1):
                a = f + d
                b = f - d
                if a > n_frames - 1:
                    a = n_frames - 1
                if b < 0:
                    b = 0
                acc += d * (q8[a * N_CEPS + j] - q8[b * N_CEPS + j])
            v = ((acc * DELTA_Q15 + 16384) >> 15)
            v = ((v + dhalf) >> shift) + BIAS
            if v < 0:
                v = 0
            elif v > 65535:
                v = 65535
            out[dst + N_CEPS + j] = v


def features(samples, start, count, work=None):
    """int16 samples -> (biased uint16 feature rows, n_frames).

    The rows come back biased by +32768 in an array("H"), which is the form the
    DTW inner loop wants: templates are stored the same way and the bias
    cancels in the difference, so neither side needs sign extension.
    """
    q8, n_frames = mfcc_q8(samples, start, count, work)
    if not n_frames:
        return None, 0

    out = array("H", bytes(2 * N_FEAT * n_frames))
    _feature_rows(q8, out, n_frames)
    return out, n_frames


# --- The same front end again, in viper ------------------------------------
#
# Speed only. Every expression below is the same expression as the portable
# stage it replaces, in the same order, and the portable version above stays
# the specification -- this is a transcription, not a second implementation.
# `src/speech_fixtures.py` is what holds the two together: it pins the pipeline
# stage by stage, so a wrong transcription names the stage it broke instead of
# only disagreeing at the output.
#
# **The overflow proof in docs/speech.md stops being free here.** CPython and
# MicroPython both use unbounded ints, so neither the host test nor an
# unported device run can see a value that would wrap; viper's `int` is a
# 32-bit machine word and wraps silently. Two things the proof rests on, both
# preserved literally below:
#
#   - one unconditional right shift per FFT stage, which is what keeps
#     |X| <= 32767 and so the twiddle product inside 50.00% of int32;
#   - `wr*lr - wi*li` in that order, because a reordering can make a partial
#     sum exceed the bounded final value.
#
# `speech_fixtures.tw_stress` is a real frame sitting at 99.8% of that bound
# and it is the case to run first -- the deliberately hostile full-scale input
# is *milder*, because pre-emphasis saturation clamps it before the FFT.
#
# Three viper details, each silent when wrong (all probed on this build,
# MicroPython 1.28.0 / armv7emsp, not assumed):
#
#   - a ptr16 load is zero-extended, so int16 input needs the explicit
#     `- 65536`; a ptr32 load is a whole machine word, and `int()` is what
#     types it signed, so every load is wrapped in it;
#   - `>>` on a signed viper int is arithmetic and floors, matching CPython on
#     negatives, and `//` is available and does the same;
#   - `ptr32(GLOBAL)` inside a viper function casts a module-level array("i"),
#     which is what keeps the frozen tables out of the argument list.
#
# Only ImportError is caught, so this falls back to the portable path on the
# host and nowhere else. A viper compile error on the device is meant to be
# loud: silently running at 120 ms a frame is the failure this port exists to
# remove.

try:  # pragma: no cover -- device only
    import micropython

    @micropython.viper
    def _v_preemph(samples: ptr16, start: int, count: int, out: ptr16):
        if count <= 0:
            return
        pe = int(PREEMPH_Q15)
        prev = int(samples[start])
        if prev > 32767:
            prev = prev - 65536
        i = 0
        while i < count:
            x = int(samples[start + i])
            if x > 32767:
                x = x - 65536
            y = x - ((pe * prev) >> 15)
            prev = x
            if y > 32767:
                y = 32767
            elif y < -32768:
                y = -32768
            out[i] = y
            i += 1

    @micropython.viper
    def _v_prepare(src: ptr16, base: int, re: ptr32, im: ptr32) -> int:
        wnd = ptr32(WINDOW)
        brv = ptr32(BITREV)
        flen = int(FRAME_LEN)
        fsz = int(FFT_SIZE)
        peak = 0
        n = 0
        while n < flen:
            x = int(src[base + n])
            if x > 32767:
                x = x - 65536
            v = (x * int(wnd[n]) + 16384) >> 15
            re[n] = v
            if v < 0:
                v = 0 - v
            if v > peak:
                peak = v
            n += 1
        while n < fsz:
            re[n] = 0
            n += 1
        n = 0
        while n < fsz:
            im[n] = 0
            n += 1

        g = 0
        if peak != 0:
            b = 0
            t = peak
            while t != 0:
                t >>= 1
                b += 1
            g = 14 - (b - 1)
            if g < 0:
                g = 0
        if g != 0:
            n = 0
            while n < flen:
                re[n] = int(re[n]) << g
                n += 1

        n = 0
        while n < fsz:
            b = int(brv[n])
            if b > n:
                t = int(re[n])
                re[n] = int(re[b])
                re[b] = t
            n += 1
        return g

    @micropython.viper
    def _v_fft512(re: ptr32, im: ptr32):
        twr = ptr32(TW_RE)
        twi = ptr32(TW_IM)
        n = int(FFT_SIZE)
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
                    wr = int(twr[k])
                    wi = int(twi[k])
                    lr = int(re[l])
                    li = int(im[l])
                    tr = (wr * lr - wi * li + 16384) >> 15
                    ti = (wr * li + wi * lr + 16384) >> 15
                    jr = int(re[j])
                    ji = int(im[j])
                    re[l] = (jr - tr + 1) >> 1
                    im[l] = (ji - ti + 1) >> 1
                    re[j] = (jr + tr + 1) >> 1
                    im[j] = (ji + ti + 1) >> 1
                    k += step
                    j += 1
                i += size
            size <<= 1

    @micropython.viper
    def _v_magnitudes(re: ptr32, im: ptr32, mag: ptr32):
        # Restoring binary square root rather than the portable Newton loop:
        # exact floor either way, but no divide, which viper has no cheap form
        # of. `rem` ends as p - res*res, so the round-to-nearest test that the
        # portable version writes as `p - m*m > m` is `rem > res` here.
        nb = int(N_BINS)
        # 2**30 is one past MicroPython's small-int range, so written as a
        # literal inside the loop viper takes it for an object and refuses the
        # comparison. int() once, outside, is the whole fix.
        hibit = int(1 << 30)
        k = 0
        while k < nb:
            r = int(re[k])
            i2 = int(im[k])
            p = r * r + i2 * i2
            rem = p
            res = 0
            bit = hibit
            while bit > rem:
                bit >>= 2
            while bit != 0:
                t = res + bit
                if rem >= t:
                    rem -= t
                    res = (res >> 1) + bit
                else:
                    res >>= 1
                bit >>= 2
            if rem > res:
                res += 1
            mag[k] = res
            k += 1

    @micropython.viper
    def _v_melbank(mag: ptr32, mel: ptr32, g: int):
        ms = ptr32(MEL_START)
        ml = ptr32(MEL_LEN)
        mw = ptr32(MEL_W)
        lg = ptr32(LOG2)
        nmel = int(N_MEL)
        logq = int(LOG_Q)
        ofs = 0
        top = 0
        i = 0
        while i < nmel:
            s = int(ms[i])
            ln = int(ml[i])
            acc = 0
            k = 0
            while k < ln:
                acc += (int(mag[s + k]) * int(mw[ofs + k]) + 128) >> 8
                k += 1
            ofs += ln
            mel[i] = acc
            if acc > top:
                top = acc
            i += 1

        fsh = int(MEL_FLOOR_SHIFT)
        if fsh != 0:
            fl = top >> fsh
            i = 0
            while i < nmel:
                if int(mel[i]) < fl:
                    mel[i] = fl
                i += 1

        # (2 - g) << LOG_Q, written as a multiply because g can exceed 2 and
        # shifting a negative left is not something to rely on.
        corr = (2 - g) * (1 << logq)
        i = 0
        while i < nmel:
            v = int(mel[i])
            if v < 1:
                v = 1
            e = 0
            t = v
            while t != 0:
                t >>= 1
                e += 1
            e -= 1
            if e >= 16:
                m = v >> (e - 16)
            else:
                m = v << (16 - e)
            idx = (m >> 10) - 64
            frac = m & 1023
            lo = int(lg[idx])
            hi = int(lg[idx + 1])
            mel[i] = (e << logq) + lo + (((hi - lo) * frac + 512) >> 10) + corr
            i += 1

    @micropython.viper
    def _v_dct(mel: ptr32, out: ptr32):
        dct = ptr32(DCT)
        nmel = int(N_MEL)
        nceps = int(N_CEPS)
        j = 0
        while j < nceps:
            row = j * nmel
            acc = 0
            i = 0
            while i < nmel:
                acc += (int(mel[i]) * int(dct[row + i]) + 16384) >> 15
                i += 1
            out[j] = acc
            j += 1

    @micropython.viper
    def _v_lifter_row(ceps: ptr32, q8: ptr32, off: int):
        lf = ptr32(LIFTER)
        nceps = int(N_CEPS)
        j = 0
        while j < nceps:
            q8[off + j] = (int(ceps[j]) * int(lf[j]) + 2048) >> 12
            j += 1

    @micropython.viper
    def _v_cmn(q8: ptr32, n_frames: int):
        nceps = int(N_CEPS)
        j = 0
        while j < nceps:
            total = 0
            f = 0
            while f < n_frames:
                total += int(q8[f * nceps + j])
                f += 1
            # Toward zero, as the portable version and the host both do.
            if total >= 0:
                mean = total // n_frames
            else:
                mean = 0 - ((0 - total) // n_frames)
            f = 0
            while f < n_frames:
                q8[f * nceps + j] = int(q8[f * nceps + j]) - mean
                f += 1
            j += 1

    @micropython.viper
    def _v_feature_rows(q8: ptr32, out: ptr16, n_frames: int):
        nceps = int(N_CEPS)
        nfeat = int(N_FEAT)
        bias = int(BIAS)
        fshift = int(FEAT_SHIFT)
        dwidth = int(DELTA_WIDTH)
        dq15 = int(DELTA_Q15)
        shift = fshift - int(DELTA_SHIFT)
        half = 1 << (fshift - 1)
        dhalf = 1 << (shift - 1)
        last = n_frames - 1
        f = 0
        while f < n_frames:
            src = f * nceps
            dst = f * nfeat
            j = 0
            while j < nceps:
                out[dst + j] = ((int(q8[src + j]) + half) >> fshift) + bias
                j += 1
            if dwidth != 0:
                j = 0
                while j < nceps:
                    acc = 0
                    dn = 1
                    while dn <= dwidth:
                        a = f + dn
                        b = f - dn
                        if a > last:
                            a = last
                        if b < 0:
                            b = 0
                        acc += dn * (int(q8[a * nceps + j]) - int(q8[b * nceps + j]))
                        dn += 1
                    v = (acc * dq15 + 16384) >> 15
                    v = ((v + dhalf) >> shift) + bias
                    if v < 0:
                        v = 0
                    elif v > 65535:
                        v = 65535
                    out[dst + nceps + j] = v
                    j += 1
            f += 1

    # Bound to the portable names rather than wrapped in one. A wrapper is a
    # Python frame per call and the front end makes six of them per frame; at
    # ~0.6 ms a frame that was 10% of the front end, for nothing. Both names
    # stay live on purpose -- `_prepare` is whichever implementation is in use,
    # `_v_prepare` is always the viper one and the portable body is still up
    # there under its own name, so a device-side run can compare the two
    # directly rather than only against the fixtures.
    def preemphasise(samples, start, count, out):  # noqa: F811
        """See the portable version above, which is the specification.

        The one stage that keeps its wrapper: it is public and documented as
        returning `out`, a viper function returns None, and it runs once per
        utterance rather than once per frame.
        """
        _v_preemph(samples, start, count, out)
        return out

    _prepare = _v_prepare
    _fft512 = _v_fft512
    _magnitudes = _v_magnitudes
    _melbank = _v_melbank
    _dct = _v_dct
    _lifter_row = _v_lifter_row
    _cmn = _v_cmn
    _feature_rows = _v_feature_rows

except ImportError:
    pass

# --- Template expansion ----------------------------------------------------

def expand(buf, index, scratch=None):
    """Grow packed 12-wide Q8 statics into 24-wide features, inside one buffer.

    Called by `templates.load(buf, expand)` when `PACKED == "statics"`. A port
    of `mfcc.expand_all`; `tools/test_spotter.py` asserts the two agree byte for
    byte, and `tools/test_templates.py` proves the host side against `mfcc()`.

    The correctness argument is deliberately short, because the clever version
    was not:

    - Templates expand **last to first**. Template k writes to
      [48*F_k, 48*(F_k+n_k)) while the statics still needed live below
      24*F_k + 24*n_k, and 48*F_k >= 24*F_k always, so nothing below is touched.
    - Each template's statics are copied into scratch first, so the source of
      the frame being written is never the buffer being written to. That deletes
      the within-template overlap question rather than answering it -- deltas
      reach two frames either side, and an in-place version has to stash the low
      frames and argue about where the boundary falls. 1.5 KB buys the argument
      away.

    Deltas replicate at each template's *own* edges and must not read into the
    neighbour, which is the other reason this is per template rather than one
    pass over the blob.
    """
    if scratch is None:
        longest = 0
        for entry in index:
            if entry[2] > longest:
                longest = entry[2]
        scratch = bytearray(2 * N_CEPS * longest)

    shift = FEAT_SHIFT - DELTA_SHIFT
    half = 1 << (FEAT_SHIFT - 1)
    dhalf = 1 << (shift - 1)

    for k in range(len(index) - 1, -1, -1):
        frame_off = index[k][1]
        n_frames = index[k][2]
        src = 2 * N_CEPS * frame_off
        count = 2 * N_CEPS * n_frames
        for i in range(count):
            scratch[i] = buf[src + i]

        for i in range(n_frames - 1, -1, -1):
            p = 2 * N_FEAT * (frame_off + i)
            for j in range(N_CEPS):
                q = 2 * (N_CEPS * i + j)
                v = (scratch[q] | (scratch[q + 1] << 8)) - BIAS
                v = ((v + half) >> FEAT_SHIFT) + BIAS
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
                v = (acc * DELTA_Q15 + 16384) >> 15
                v = ((v + dhalf) >> shift) + BIAS
                if v < 0:
                    v = 0
                elif v > 65535:
                    v = 65535
                buf[p] = v & 0xFF
                buf[p + 1] = v >> 8
                p += 2
    return buf


# --- Dynamic time warping --------------------------------------------------

def dtw(query, n, buf, frame_off, m, band=BAND):
    """Normalised L1 distance between a query and one template, or INF.

    `query` is the biased array("H") from `features`; the template lives in the
    big buffer as biased uint16 from `frame_off` for `m` frames. Both are
    biased by the same constant, so the bias cancels in every difference and
    neither side needs sign extension -- which is what the packing was designed
    for.

    Symmetric Sakoe-Chiba: (1,0) and (0,1) cost d, the diagonal (1,1) costs 2d,
    and the total is divided by n+m. That weighting is why no path length has to
    be tracked: every monotone path from (0,0) to (n-1,m-1) accumulates exactly
    n+m units of step weight, so the normaliser is a constant and comparing a
    long template against a short one stays honest.

    L1 rather than L2: no multiply, and squaring only sharpens the influence of
    whichever coefficient happened to be worst.
    """
    if n == 0 or m == 0:
        return INF
    if n * 100 > m * DUR_RATIO_PCT or m * 100 > n * DUR_RATIO_PCT:
        return INF

    # The two cost rows are reused across every call in a turn. Allocating them
    # here would be 132 small allocations per utterance -- 66 templates, two
    # rows each -- on a heap that never compacts, which is how fragmentation
    # arrives without anything looking wrong. bind() sizes them once from the
    # longest template; the fallback is for callers that never bind, i.e. tests.
    if _prev is not None and len(_prev) >= m:
        prev, cur = _prev, _cur
    else:
        prev = array("i", bytes(4 * m))
        cur = array("i", bytes(4 * m))
    for j in range(m):
        prev[j] = INF

    base = 2 * N_FEAT * frame_off
    for i in range(n):
        c = (i * m) // n
        lo = c - band
        hi = c + band
        if lo < 0:
            lo = 0
        if hi > m - 1:
            hi = m - 1
        qi = i * N_FEAT
        for j in range(m):
            cur[j] = INF
        for j in range(lo, hi + 1):
            p = base + 2 * N_FEAT * j
            d = 0
            for k in range(N_FEAT):
                v = query[qi + k] - (buf[p] | (buf[p + 1] << 8))
                d += v if v >= 0 else -v
                p += 2
            if i == 0 and j == 0:
                cur[0] = 2 * d
                continue
            best = INF
            if i > 0:
                a = prev[j] + d                      # (1,0)
                if a < best:
                    best = a
                if j > 0:
                    a = prev[j - 1] + 2 * d          # (1,1)
                    if a < best:
                        best = a
            if j > 0:
                a = cur[j - 1] + d                   # (0,1)
                if a < best:
                    best = a
            cur[j] = best
        prev, cur = cur, prev

    total = prev[m - 1]
    if total >= INF:
        return INF
    return total // (n + m)


try:  # pragma: no cover -- device only
    import micropython

    @micropython.viper
    def _v_dtw(query: ptr16, buf: ptr16, rows: ptr32,
               n: int, m: int, band: int, frame_off: int) -> int:
        """The portable `dtw` body, transcribed. Same recurrence, same order.

        `rows` is one array holding both cost rows end to end -- viper cannot
        swap two array objects, so the two halves are swapped by their offsets
        `p0` and `p1` instead. It must be at least 2*m long; `bind()` sizes it
        from the longest template.

        `query` and `buf` are both read as unsigned 16-bit, which is what the
        +32768 bias is for: the bias cancels in the difference and neither side
        needs sign extension. Indices are in uint16 units, so the byte offsets
        of the portable version lose their factors of two.
        """
        nfeat = int(N_FEAT)
        inf = int(INF)
        p0 = 0
        p1 = m
        j = 0
        while j < m:
            rows[j] = inf
            j += 1

        base = nfeat * frame_off
        i = 0
        while i < n:
            c = (i * m) // n
            lo = c - band
            hi = c + band
            if lo < 0:
                lo = 0
            if hi > m - 1:
                hi = m - 1
            qi = i * nfeat
            j = 0
            while j < m:
                rows[p1 + j] = inf
                j += 1
            j = lo
            while j <= hi:
                p = base + nfeat * j
                d = 0
                k = 0
                # Unrolled four ways, with a tail for an N_FEAT that is not a
                # multiple of four. Measured on this build: 9.56 -> 7.53 us per
                # band cell, 21%. viper spills every local to the stack, so what
                # the unrolling buys is fewer loop-variable statements, not
                # fewer loads -- which is also why `if/else` beats a branchless
                # abs here and why walking two indices is *slower* than
                # recomputing `qi + k`. All three were measured, not assumed.
                stop = nfeat & -4
                while k < stop:
                    v = int(query[qi + k]) - int(buf[p + k])
                    if v < 0:
                        d -= v
                    else:
                        d += v
                    v = int(query[qi + k + 1]) - int(buf[p + k + 1])
                    if v < 0:
                        d -= v
                    else:
                        d += v
                    v = int(query[qi + k + 2]) - int(buf[p + k + 2])
                    if v < 0:
                        d -= v
                    else:
                        d += v
                    v = int(query[qi + k + 3]) - int(buf[p + k + 3])
                    if v < 0:
                        d -= v
                    else:
                        d += v
                    k += 4
                while k < nfeat:
                    v = int(query[qi + k]) - int(buf[p + k])
                    if v < 0:
                        d -= v
                    else:
                        d += v
                    k += 1
                if i == 0 and j == 0:
                    best = 2 * d
                else:
                    best = inf
                    if i > 0:
                        a = int(rows[p0 + j]) + d          # (1,0)
                        if a < best:
                            best = a
                        if j > 0:
                            a = int(rows[p0 + j - 1]) + 2 * d   # (1,1)
                            if a < best:
                                best = a
                    if j > 0:
                        a = int(rows[p1 + j - 1]) + d      # (0,1)
                        if a < best:
                            best = a
                rows[p1 + j] = best
                j += 1
            t = p0
            p0 = p1
            p1 = t
            i += 1

        total = int(rows[p0 + m - 1])
        if total >= inf:
            return inf
        return total // (n + m)

    def dtw(query, n, buf, frame_off, m, band=BAND):  # noqa: F811
        """See the portable version above, which is the specification.

        The two early rejections stay out here rather than in the kernel: a
        template rejected on duration never pays a viper call, and the
        duration gate is what rejects most of them.
        """
        if n == 0 or m == 0:
            return INF
        if n * 100 > m * DUR_RATIO_PCT or m * 100 > n * DUR_RATIO_PCT:
            return INF
        rows = _rows
        if rows is None or len(rows) < 2 * m:
            rows = array("i", bytes(8 * m))
        return _v_dtw(query, buf, rows, n, m, band, frame_off)

except ImportError:
    pass

# --- The spotter itself ----------------------------------------------------

_buf = None       # the template buffer; owned by the caller, not by this module
_index = None     # (label, frame_offset, n_frames) per template
_work = None      # front-end scratch, allocated on first use
_prev = None      # DTW cost rows, sized to the longest template by bind()
_cur = None
_rows = None      # the viper matcher's two rows, in one array; see _v_dtw


def bind(buf, index):
    """Point the spotter at a loaded template buffer.

    Called once, by `talk.reserve_templates()`, after `templates.load()`. This
    module deliberately does **not** import `templates` and load them itself:
    that buffer is the largest allocation in the program and the order it is
    made in is load-bearing, so `main()` owns it. See talk.reserve_templates.

    Note this keeps a reference, but the caller must too -- if `main()` drops
    its name, this one keeps the memory alive but nothing else does, and the
    intent of the ownership becomes impossible to read.
    """
    global _buf, _index, _prev, _cur, _rows
    _buf = buf
    _index = index
    longest = 0
    for entry in index:
        if entry[2] > longest:
            longest = entry[2]
    # Both shapes, because whichever `dtw` is bound wants its own: the portable
    # one swaps two array objects, the viper one swaps two offsets into a
    # single array. Half a kilobyte between them, against 137 KB of templates.
    _prev = array("i", bytes(4 * longest))
    _cur = array("i", bytes(4 * longest))
    _rows = array("i", bytes(8 * longest))


def ready():
    return _buf is not None and _index is not None


def scores(query, n):
    """[(distance, label), ...] ascending, one entry per class.

    A class may own several templates -- three takes per spoken form, and SAD
    owns both SAD and SICK -- and the class score is the minimum over all of
    them. Nothing downstream cares which template matched, only which class.
    """
    best = {}
    for entry in _index:
        label = entry[0]
        s = dtw(query, n, _buf, entry[1], entry[2])
        if s < best.get(label, INF):
            best[label] = s
    out = []
    for label in best:
        out.append((best[label], label))
    out.sort()
    return out


def spot_scored(samples, start, end, threshold=THRESHOLD, margin=MARGIN):
    """(label or None, best score, runner-up score).

    The scores come out because `threshold` and `margin` are tuned numbers that
    must be re-measured against templates enrolled through this board's own
    microphone, and a device that only reports its verdict gives nobody anything
    to tune with.

    Two gates. The absolute one asks whether this is a good enough match to act
    on at all. The margin asks whether it is *distinctly* the best -- if MOTHER
    and BROTHER both score 190, the right answer is to say nothing, whatever the
    absolute score. The margin turns out to be the cheaper half of the
    precision: near-miss confusions cluster tightly while genuine matches stand
    clear.
    """
    global _work
    if not ready():
        return None, INF, INF
    count = end - start
    if frame_count(count) == 0:
        return None, INF, INF
    if _work is None:
        _work = new_work()

    query, n = features(samples, start, count, _work)
    if not n:
        return None, INF, INF

    ranked = scores(query, n)
    if not ranked:
        return None, INF, INF
    best, label = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else INF
    if best > threshold:
        return None, best, runner_up
    if margin and runner_up - best < margin:
        return None, best, runner_up
    return label, best, runner_up


def spot(samples, start, end):
    """The vocabulary label that was said, or None. `spot_scored` with the
    scores dropped."""
    return spot_scored(samples, start, end)[0]
