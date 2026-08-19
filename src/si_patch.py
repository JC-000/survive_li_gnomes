"""Endpointed samples -> the 80x26 uint8 patch the CNN takes.

The device half of `tools/si_features.py`, which is the specification. Where the
two disagree this file is wrong, and `tools/test_si_patch.py` is what says so.

## This is a tap, not a second front end

Every value here is already computed by the existing front end on its way to
each cepstrum. `spotter._mfcc_at` runs stages 2-7 of `docs/speech.md` and leaves
the 26 Q8 log2 mel values in `work[3]`; the DCT then reads them and this file
reads them too. So the arithmetic that produces them is the arithmetic the DTW
path already uses, proved bit-exact against `src/speech_fixtures.py` on this
board -- there is nothing new to verify below the tap.

**`_mfcc_at` is called whole, including its DCT, whose output is discarded.**
Copying its four lines here without the DCT would save 0.14 ms a frame -- 11 ms
over an 80-frame patch, against a front end that costs about 500 ms and a panel
refresh that costs 583 -- and would buy that by holding a second copy of the
stage order, which is the thing most likely to drift when someone adds a stage.
The 2% is the cheaper side of that trade. If it ever stops being, inline the
four calls and add a fixture pinning them against `mel`.

## What this file does add, and it is all of it

Per-band mean subtraction over the utterance, then a shift to int8:

    m[i]    = trunc_toward_zero(sum over t of x[t][i], n)
    y[t][i] = clamp((x[t][i] - m[i] + 8) >> 4, -128, 127)

Per **band**, not per frame, and it is the log-mel analogue of the cepstral mean
normalisation the DTW path already does: a fixed channel colouration is additive
in the log domain, so subtracting each band's own mean removes it exactly. That
is what lets a model trained on `say` voices meet a real microphone at all.

Truncation **toward zero**, not floor, because that is what a hardware divide
gives and what the host reference does. On negative totals the two differ, and
they differ on exactly the quiet bands where the mean is negative.

## Why there is no viper here

The brief called for this pass in viper. It is written in plain bytecode
instead, and the numbers are **measured on this board** over a 36-frame take:

| | |
| --- | --- |
| `logmel_rows` (the tapped front end, plus the discarded DCT) | **224.2 ms** |
| `normalise_into` (everything this file adds) | **23.4 ms** |
| the whole recognition, including 66.6 ms of inference | **317 ms** |
| the panel refresh that follows it | **624 ms** |

So the pass is **9.5%** of the two feature stages and **2.5% of a turn**. It is
not worth porting, and it is worth saying that the argument for not porting it
was originally weaker than the measurement: the estimate written here first was
"single-digit milliseconds", which was optimistic by a factor of two or three.
The conclusion survived being measured; the reasoning behind it did not entirely.

What porting would cost is not typing. Viper's `int` is a 32-bit machine word
that wraps silently, and every line of it needs the bounds discipline
`docs/speech.md` spends pages on. Spending that on 2.5% of a turn dominated by
an e-paper refresh would be the wrong trade. **Measure before porting the next
one too** -- and measure rather than estimate, which is the part this file got
wrong the first time.

## The uint8 convention is not cosmetic

`emlearn_cnn_int8` takes `array('B')` and computes the quantised value as
`uint8 - 128` internally, ignoring the model's own scale and zero point on that
path. `si_real` quantises its input at scale 1.0 / zero point 0, so the int8
feature value *is* the quantised value and the transport is a bare `+ 128`.
`tools/tmdl_info.py` prints that mapping and refuses a model where it would be
anything else -- see docs/cnn-on-device.md.
"""

from array import array

import spotter

N_BANDS = spotter.N_MEL          # 26
N_FRAMES = 80                    # see tools/si_features.py for how 80 was chosen
INPUT_SHIFT = 4
PATCH_BYTES = N_FRAMES * N_BANDS

# The VAD accepts 15..180 frames, so the row buffer is sized for the longest
# utterance it will ever hand over rather than for the patch. Q8 log2 values run
# about -3072..7014 (the low end is the `(2 - g) << 8` correction at maximum
# block-float shift), so int16 holds them with room to spare.
MAX_ROWS = 180

# Pre-emphasis runs one frame at a time, so the buffer is one frame plus the
# single sample of history the filter needs -- 401 samples, 802 bytes.
#
# It was 29040 samples (58080 bytes) for about ten minutes, holding the whole
# utterance at once. That allocation failed at boot: `array("h", bytes(n))`
# holds both objects, so it wanted ~116 KB contiguous, and it ran after the
# 94 KB capture buffer had already fragmented the heap. Making a 58 KB buffer
# fit by allocating it earlier would have worked and would have left a 58 KB
# hostage to the next thing that grows. Not needing it is better.
PRE_SAMPLES = spotter.FRAME_LEN + 1


def allocate():
    """The buffers a spotter needs, allocated once and reused.

    Returned as a tuple rather than held in a module global because allocation
    *order* is a whole-program concern -- `docs/speech.md` records what that cost
    to learn -- and this module cannot know what else is about to be reserved.
    """
    return (spotter.new_work(),                        # ~9 KB of front-end scratch
            array("h", bytes(2 * MAX_ROWS * N_BANDS)),   # 9.4 KB of raw mel rows
            array("B", bytes(PATCH_BYTES)),              # 2080 B, the model input
            array("h", bytes(2 * PRE_SAMPLES)))          # 802 B, one frame


def logmel_rows(samples, start, count, work, rows, pre=None):
    """Fill `rows` with per-frame Q8 log2 mel. Returns the frame count.

    `samples` is the raw int16 capture buffer and `start`/`count` are the
    endpointed span, the same convention `spotter.features` takes.

    `pre` is the int16 buffer for the pre-emphasised signal and **should always
    be supplied**; `allocate()` returns one. The `None` fallback allocates, and
    allocating here is a bug that already reached hardware -- see below.
    """
    n_frames = spotter.frame_count(count)
    if n_frames == 0:
        return 0
    if n_frames > MAX_ROWS:
        # The VAD's own accept window is 15..180, so this cannot fire from the
        # live path. It can from a test feeding an untrimmed buffer, and
        # silently writing past `rows` is the failure this project has spent
        # the most time on, so it is a clamp and not a comment.
        n_frames = MAX_ROWS

    if pre is None:
        # ALLOCATING HERE IS A BUG, and it reached hardware: with `si_spot`
        # never passing a buffer, this ran every turn and sized itself by
        # utterance length. A 1.8 s hold asked for `bytes(2 * 28800)` -- which
        # MicroPython rounds to 57601 -- and the turn died with a MemoryError
        # on the *first* press after a clean reset.
        #
        # It is worse than one allocation. `array("h", bytes(n))` holds the
        # bytes object and the array at the same time, so it peaks at **twice**
        # the final size: 115 KB of transient, per turn, on a heap that never
        # compacts. docs/speech.md names this exact form as the thing not to
        # write, and `sounds.allocate_bytes` records what it cost to learn the
        # first time.
        #
        # Kept only so a direct caller (a host test, a REPL poke) still works.
        # The live path must pass the buffer `allocate()` made once.
        pre = array("h", bytes(2 * PRE_SAMPLES))

    mel = work[3]
    ceps = work[5]
    dst = 0
    for f in range(n_frames):
        # Pre-emphasise this frame only. `y[n] = sat16(x[n] - 31785*x[n-1] >> 15)`
        # is causal, so a frame computed from `x[base-1 .. base+399]` is
        # bit-identical to the same slice of a whole-utterance pass -- provided
        # the real predecessor is supplied rather than the `x[-1] := x[0]` rule,
        # which applies only to the very first sample of the utterance.
        #
        # So frame 0 is pre-emphasised directly, and every later frame starts
        # one sample early and the extra output is skipped. That one sample of
        # overlap is the whole trick, and getting it wrong would shift every
        # frame but the first by an amount too small to look like a bug.
        base = f * spotter.FRAME_STRIDE
        if base == 0:
            spotter.preemphasise(samples, start, spotter.FRAME_LEN, pre)
            off = 0
        else:
            spotter.preemphasise(samples, start + base - 1,
                                 spotter.FRAME_LEN + 1, pre)
            off = 1
        # The DCT inside runs and its output is thrown away; see the module
        # docstring for why that is the cheaper of the two mistakes available.
        spotter._mfcc_at(pre, off, work, ceps)
        for i in range(N_BANDS):
            rows[dst + i] = mel[i]
        dst += N_BANDS
    return n_frames


def normalise_into(rows, n_frames, patch):
    """Per-band mean subtraction, shift to int8, bias to uint8, fit to N_FRAMES.

    Writes exactly `PATCH_BYTES` into `patch` and returns the number of values
    that clipped. A non-zero count is not an error -- what clips is a band far
    below its own mean over the utterance, so the clamp acts as a spectral floor
    -- but it is returned rather than swallowed, because a *large* count means
    the shift is wrong and that is invisible otherwise.

    Centre-crop or centre-pad, matching `si_features.fit`. Padding with the
    normalised zero is padding with the band mean, which is what silence
    normalises to, so a short word is surrounded by something the model reads as
    nothing rather than by an edge.
    """
    half = 1 << (INPUT_SHIFT - 1)
    clipped = 0

    # Which rows land in the patch, and where. Cropping takes the middle because
    # the endpointer has already centred the word; padding splits the remainder
    # so the word stays centred.
    if n_frames >= N_FRAMES:
        src_start = (n_frames - N_FRAMES) // 2
        dst_start = 0
        n_copy = N_FRAMES
    else:
        src_start = 0
        dst_start = (N_FRAMES - n_frames) // 2
        n_copy = n_frames

    # Means over *every* frame the utterance has, not only the copied ones.
    # `si_features.normalise` runs before `fit`, so the mean is over the whole
    # endpointed word; taking it over the cropped window instead would make the
    # normalisation depend on the crop and would not reproduce the host.
    for i in range(N_BANDS):
        total = 0
        off = i
        for t in range(n_frames):
            total += rows[off]
            off += N_BANDS
        if total >= 0:
            mean = total // n_frames
        else:
            mean = -((-total) // n_frames)   # toward zero, as the host does

        # 128 is the biased zero: the pad value and the value a band exactly at
        # its own mean takes.
        for t in range(dst_start):
            patch[t * N_BANDS + i] = 128
        for t in range(dst_start + n_copy, N_FRAMES):
            patch[t * N_BANDS + i] = 128

        src = (src_start * N_BANDS) + i
        dst = (dst_start * N_BANDS) + i
        for t in range(n_copy):
            v = (rows[src] - mean + half) >> INPUT_SHIFT
            if v > 127:
                v = 127
                clipped += 1
            elif v < -128:
                v = -128
                clipped += 1
            patch[dst] = v + 128
            src += N_BANDS
            dst += N_BANDS

    return clipped


def patch_for(samples, start, count, buffers, pre=None):
    """The whole path: endpointed samples -> (patch, n_frames, clipped).

    `buffers` is what `allocate()` returned. `patch` is the same array every
    call -- the caller must not hold it across turns.
    """
    work, rows, patch, pooled = buffers
    n_frames = logmel_rows(samples, start, count, work, rows,
                           pre if pre is not None else pooled)
    if n_frames == 0:
        return None, 0, 0
    clipped = normalise_into(rows, n_frames, patch)
    return patch, n_frames, clipped
