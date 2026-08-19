# The feature contract

Everything a keyword has to survive between the microphone and a match. This
document is **normative**: `tools/mfcc.py` is the reference implementation and
the device's `@micropython.viper` port must produce *bit-identical* output for
identical input. Where the two disagree, this document and `tools/mfcc.py` are
right and the port is wrong.

[speech-design.md](speech-design.md) is the companion and argues *why* — why
DTW rather than a trained classifier, why 16 kHz, why precision over recall.
This file only pins down *what*, to the last integer.

> ## Every accuracy figure in this document is synthetic
>
> The corpus is macOS `say` at varied speaking rates through a channel model.
> That reproduces **session** mismatch — a different day, a different level —
> and **not articulation** mismatch: the mumbled vowel, the swallowed final
> consonant, the day with a cold. The cross-microphone figures are worse than
> that: the "microphone difference" is one **I chose**, not one anybody
> measured.
>
> These numbers did the job they were built for, which was **ranking design
> choices against each other** — deltas against band-limiting, NO in or out,
> which words collide. They are not a prediction of what the board will do, and
> a recall of 0.966 should not be quoted as though they were.
>
> **`tools/mic_margin.py record` is the first thing to run when a board is
> connected.** Nothing here is confirmed until it has.

## Why bit-identical, and why that forces integers

Templates are enrolled on the host and matched on the device. A DTW distance is
a sum of absolute differences between the two, so any arithmetic difference
between the two implementations shows up as a distance that nobody can account
for — not as a crash, and not as anything either side's own tests would catch.

That rules out floating point outright, and not for speed. MicroPython builds
`float` as **single** precision on RP2, so `math.cos` there does not round the
way CPython's double does; a window table computed on each side would differ in
the last bits and the two front ends would drift apart silently. Every table
below is therefore computed once on the host and frozen into
`src/speech_tables.py` (`python3 tools/mfcc.py --emit-tables`, 4258 bytes).

Integers also happen to be the only thing viper accelerates: the ~16x it gives
over bytecode is for integer and bit operations, and it does
[essentially nothing for floats](https://docs.micropython.org/en/latest/reference/speed_python.html).
So this is a case where the correctness argument and the speed argument point
the same way.

**All shifts are arithmetic.** CPython's `>>` floors, ARM's `ASR` floors, and
viper's `int` is a signed machine word, so `-3 >> 1 == -2` on both sides. Never
substitute `// 2` for `>> 1`; they differ on negatives, which is where it will
not be noticed.

## Audio format

| | |
| --- | --- |
| Rate | 16000 Hz |
| Channels | 1 |
| Sample format | signed 16-bit, native (little) endian |
| Codec setting | ES8311 with MCLK = 4096000 = 256 x fs |

The vendor coefficient table in `src/es8311.py` carries a `[4096000, 16000, ...]`
row, so this is a supported combination and not an extrapolation. The
microphone is verified at **both** rates (`docs/hardware.md`): 24 kHz first, and
then 16 kHz measured at **15990 Hz** against wall clock, −0.1 %. Resampling
24 kHz down is not an escape route — 24000/16000 is 3:2 and needs a real filter —
so configure the codec for 16 kHz directly.

24 kHz would also cost 50% more frames for nothing: speech energy above 8 kHz
does not distinguish these words. Going the other way to 8 kHz would be actively
harmful, because it discards the 4-8 kHz band where /s/ and /f/ live, and those
are what separate SISTER from MOTHER and FATHER from BROTHER.

## Endpointing

Runs on the raw samples, before anything below. **`src/vad.py` is the single
implementation**; `tools/vad.py` loads it by file path and adds WAV reading and
a CLI, so host and device cannot disagree.

They started as two implementations kept in step by `tools/test_vad.py`, and
drifted twice in one afternoon anyway, in both directions. A word trimmed
differently on the two sides is compared against a differently shaped template,
and the DTW distances then drift for a reason neither file's own tests can see —
so agreeing by construction is worth more than agreeing by assertion.
`tools/test_vad.py` keeps its job: guarding the `@micropython.viper`
`_frame_stats` against its portable twin, which is the part a shared module
cannot protect.

Rabiner & Sambur (1975), short-time energy plus zero-crossing rate:

| | |
| --- | --- |
| Frame | 160 samples (10 ms), no overlap |
| Energy | `sum(abs(x))` over the frame, **not** squared |
| ZCR | sign changes per frame, 0..159 |
| Background estimate | first 10 frames (100 ms) |
| `ITL` | `min(3*(IMX-IMN)/100 + IMN, 4*IMN)`, floored at 1 |
| `ITU` | `5 * ITL` |
| `IZCT` | `max(25, zmean + 2*zstd)` — see below |
| ZCR lookback | 25 frames, needs 3 hits |
| ZCR energy gate | hit also requires `energy > IMN + 4*sigma`, floored at `IMN + IMN>>3` |
| Margin kept | 3 frames (30 ms) each side |
| Accept | 15..180 frames (150 ms .. 1.8 s) |

Sum-of-absolute-value rather than squared energy is deliberate: squared energy
over a 160-sample frame of full-scale int16 reaches 1.7e11 and would need 64-bit
accumulation, which viper does not have. `sum(abs(x))` peaks at 5242880 and fits
int32 nine times over, and the two rank frames near enough identically for a
threshold test.

### IZCT inverts the paper's clamp

R&S take `IZCT = min(25, zmean + 2*sigma)`. That assumes a background whose
crossing rate is **low**, true of a 1975 analog lead-in and false of any
digitised one — broadband noise crosses zero on nearly every other sample.
Measured here: silence runs **75-90** crossings per 10 ms frame while a voiced
vowel runs **3-14**. Taking the minimum puts the threshold far below the
background, every silent frame reads as frication, and the back-off walks the
endpoint out to the full 25-frame lookback in both directions. Before the fix
every utterance in the evaluation corpus endpointed to 930-1090 ms *regardless
of the word*; after it, they come out 430-590 ms with COMPUTER correctly the
longest.

So 25 becomes a lower bound and the adaptive term rises above it. Even then,
crossing rate alone cannot separate frication from a noise floor — both are
broadband — so a hit must also clear twice the background energy. That is about
twelve standard deviations of a silent frame's variation, while the /s/ of SORRY
measures 2.5x to 6.6x background.

The ZCR pass is not a refinement. Energy alone finds the voiced core and clips
the unvoiced onset — the /f/ of FATHER, the /s/ of SORRY — and those onsets are
exactly the cue that separates FATHER from MOTHER. Cutting them makes the pair
*more* confusable.

### The energy gate must sit below ITL, or the pass cannot fire at all

The back-off only ever inspects frames *outside* the energy-detected region,
which means frames below ITL. So it can only ever act in the band between its
own energy gate and ITL — and if the gate sits at or above ITL that band is
empty and the pass is inert, however high the crossing rate.

The first version gated at `2 * IMN`. Since
`ITL = min(3% of (IMX - IMN) + IMN, 4 * IMN)`, that band is empty unless
`IMX > ~34 * IMN`, i.e. unless the loudest frame is about 31 dB above
background. Measured over 175 corpus utterances, **37 of them (21%) had an
empty band**, and the predicted cutoff matched the observed one exactly. The
pass was switching itself off in the quieter recordings — precisely where it is
wanted. (Found by the device-side agent from the algebra, not from a failing
test, which is the only way this one surfaces: it degrades output rather than
breaking it.)

Gating on **sigma** rather than on a multiple of the mean fixes it, and is the
more honest test anyway — the question is "is this frame above the noise", and
the noise has a measurable spread. A silent frame's summed magnitude varies by
about 8% of its mean, so four sigma lands near `1.3 * IMN` and the band stays
open down to `IMX > ~12 * IMN`. The `IMN >> 3` floor covers a pathological
zero-variance background where sigma alone would admit every frame.

| | before (`2 * IMN`) | after (`IMN + 4*sigma`) |
| --- | --- | --- |
| gate / IMN | flat 2.00 | 1.12–1.34, median 1.21 |
| ZCR band empty | 37 of 175 | **0 of 175** |
| pass moved an endpoint | 8 of 175 | **20 of 175** |
| end-to-end precision / recall | 1.000 / 0.990 | 1.000 / 0.990 |

Two and a half times more effective at the job it exists for, at no cost.

## Front end, stage by stage

Reference: `tools/mfcc.py`. Every constant below is also emitted into
`src/speech_tables.py`.

### 1. Pre-emphasis

```
y[n] = sat16(x[n] - ((31785 * x[n-1]) >> 15))     x[-1] := x[0]
```

`31785 = round(0.97 * 32768)`. Output saturates to `[-32768, 32767]`.

The saturation is load-bearing, not tidiness. Without it a full-scale signal
alternating at 8 kHz gives `|y|` up to 64551, and the windowing multiply below
would then reach 2.115e9 — inside int32 by 1.5%, which is not a margin. With it
the bound is `32768 * 32767 = 1.074e9`, a factor of two clear. Real speech never
approaches either: measured on the test signal, saturation fires on **0 of 6400
samples**.

### 2. Framing and window

| | |
| --- | --- |
| Frame length | 400 samples (25 ms) |
| Stride | 160 samples (10 ms), so 100 frames/s |
| Frame count | `1 + (n_samples - 400) // 160`, zero if shorter than 400 |
| Window | Hamming, **symmetric**: `0.54 - 0.46*cos(2*pi*n/399)` |
| Stored as | `WINDOW_Q15[n] = round(w * 32767)`, range 2621..32767 |

Symmetric (`N-1` denominator), matching HTK and scipy's `sym=True` — *not* the
periodic form. They differ by one sample of phase and that is enough to break
bit-identity.

```
v[n] = (y[n] * WINDOW_Q15[n] + 16384) >> 15
```

The `+16384` rounds instead of truncating. Every shift in this pipeline rounds
that way; truncation biases each stage low, and nine stages of consistent
downward bias is not noise, it is an offset that survives to the cepstrum.

### 3. Block-floating-point normalisation

```
peak = max |v[n]| over the frame
g    = 0                      if peak == 0
     = max(0, 14 - floor(log2(peak)))    otherwise
a[n] = v[n] << g              n < 400
a[n] = 0                      400 <= n < 512
```

`g` puts the frame's peak in `[16384, 32767]`, buying back the dynamic range the
FFT's nine stage shifts are about to spend. `g` is carried to stage 6 and folded
into the log, so frames of different loudness stay comparable — this is a
*scaling* trick, not a normalisation of the features.

### 4. FFT

512-point radix-2 **decimation in time**, in place, on two int32 arrays, with
one right shift per stage.

Input is permuted by the frozen `BITREV` table (a table rather than the usual
reversal loop: 1024 bytes, and it keeps viper's inner loop free of per-element
bit twiddling).

Twiddles are `W_k = exp(-2*pi*i*k/512)` for `k < 256`:

```
TWIDDLE_RE_Q15[k] = round(32767 * cos(2*pi*k/512))
TWIDDLE_IM_Q15[k] = round(-32767 * sin(2*pi*k/512))
```

32767 stands in for 1.0. Butterfly, for stage size `s`, half `h = s/2`, twiddle
step `512/s`:

```
tr = (wr*re[l] - wi*im[l] + 16384) >> 15
ti = (wr*im[l] + wi*re[l] + 16384) >> 15
re[l] = (re[j] - tr + 1) >> 1
im[l] = (im[j] - ti + 1) >> 1
re[j] = (re[j] + tr + 1) >> 1
im[j] = (im[j] + ti + 1) >> 1
```

**Overflow proof.** The tightest stage in the whole pipeline is this twiddle
product. Two numbers describe it and they are different quantities, so state
both or neither:

- **How close to the proof:** real speech reaches 1071611968 against a proved
  ceiling of 1073709056 — **99.8% of the bound.** The analysis is tight; there
  is almost nothing between what speech does and what the algebra permits.
- **How far from failure:** that ceiling is **50.00% of int32**, so the stage
  sits a **full factor of two** below overflow. Nothing here is near wrapping.

The first number says the proof is not slack, which is why the *bound* is worth
defending. The second says there is no risk to mitigate. A reader who sees only
"99.8%" will either spend a bit that is not needed or optimise away a margin
that is doing real work; both would be wrong.

The invariant is on **complex magnitude, not on each component separately**,
and the difference is the whole argument:

- If `|A| <= M` and `|B| <= M` then `|W*B| <= M` (since `|W| < 1`), so
  `|A ± W*B| <= 2M`, and after the `>> 1` the bound is `M` again.
- The input is real with `|a[n]| <= 32767` and zero imaginary part, so
  `M = 32767` at the start and the recurrence contracts:
  `M' <= 0.99996*M + 1.05`, which is below `M` at `M = 32767`. So
  **`|X| <= 32767` at every stage**, rounding included.
- The product is then bounded by Cauchy-Schwarz:
  `|wr*lr - wi*li| <= sqrt(wr² + wi²) * sqrt(lr² + li²)`. Twiddle rounding gives
  `sqrt(wr² + wi²) <= 32767.71`, so the product is at most
  `32767.71 * 32767 = 1.0737e9` — **50.0% of int32, a factor of two clear.**

> **The component-wise reading is a trap, and it is the natural one.** Taking
> `|re| <= 32767` and `|im| <= 32767` *independently* gives
> `re² + im² <= 2 * 32767² = 2.147e9`, which is 99.99% of int32 and looks
> terrifying. It is not reachable: it would require `|X| = 32767*sqrt(2) =
> 46341`, which the magnitude invariant above forbids. Components are bounded
> *jointly*, by the magnitude, not one at a time.

Measured, over 1130 real frames:

| | measured | bound |
| --- | --- | --- |
| `fft.tw` twiddle product | **1071611968 — 49.90%** | 50.00% |
| max `|X|` after any stage | 18944 (stage 1, falling to 4300 by stage 9) | 32767 |
| `mag.sq` = `re² + im²` | 18495425 — **0.86%** | 30.52% |

`mag.sq` is nowhere near the limit. Its own ceiling is not 50% either: only 400
of the 512 inputs are non-zero and each is at most 32767, so
`|X| <= 400 * 32767 / 512 = 25599` and `re² + im² <= 25599² = 30.52%` of int32.

**What would break the twiddle bound, and what would not.** It rests on
`|X| <= 32767` inside the FFT, which rests on one right shift per stage. So:

- **Would break it:** removing or skipping a stage shift, or raising the
  block-float target above 32767. Both are changes to the FFT itself.
- **Would not break it:** a recording gain change, a different window, a wider
  or narrower mel range, a different sample rate. The block-floating-point
  normalisation in stage 3 makes the FFT input **scale-invariant** — the frame
  peak always lands in `[16384, 32767]` whatever the recording level — so the
  twiddle product cannot be pushed past its bound from outside the transform.

If the invariant ever changes so that the input scale *can* be pushed from
outside, the remedy is Q14 twiddles instead of Q15: the product halves to 25%
of int32, at the cost of one bit of twiddle precision and a full re-enrolment.
It is not needed today, and buying it now would spend precision against a risk
that cannot occur.

Output represents the true spectrum scaled by `2^(g-9)`.

### 5. Magnitude

```
p = re[k]*re[k] + im[k]*im[k]      k = 0..256
m = isqrt(p)                        floor
if p - m*m > m: m += 1              round to nearest
mag[k] = m
```

**Magnitude, not power.** Power needs 15 bits of headroom the mel accumulator
does not have; magnitude keeps quiet bins alive and only halves the log, which
is a constant factor DTW never sees. The round-to-nearest correction matters
because a floor's -0.5 bias is 17% of a quiet bin and does not cancel.

`isqrt` must be exact floor on both sides — `math.isqrt` on the host, a bitwise
restoring square root on the device. `p <= 1.074e9`, so it fits int32.

### 6. Mel filterbank

24 triangular filters, **HTK** mel scale, 200 Hz to 5000 Hz:

```
mel(f)  = 2595 * log10(1 + f/700)
imel(m) = 700 * (10^(m/2595) - 1)
```

HTK, not Slaney. Slaney's scale is piecewise-linear below 1 kHz and puts the
filter edges somewhere else entirely; picking the wrong one is a silent
mismatch, not an error.

26 edges are placed at equal mel spacing between `mel(200)` and `mel(5000)`;
filter `i` spans `edges[i] .. edges[i+1] .. edges[i+2]`.

**Band-limited on purpose.** The extremes are where two microphones disagree
most, so discarding them is cheap channel robustness: measured
cross-microphone, 200–5000 recovers 41/70 top-1 against the full band's 36/70.

200–5000 rather than the telephone band 300–3400, which was the original
proposal: measured, the two are indistinguishable cross-microphone (41 versus
42 of 70, well inside the ±4-word standard error at n=70), and 3400 Hz discards
the entire 3.4–8 kHz region where /s/ and /f/ live. Those are what separate
SISTER from MOTHER and FATHER from BROTHER, and the ZCR endpointing pass exists
specifically to keep those onsets *in* the segment — preserving them in time
and then discarding them in frequency would be perverse.

Filter widths run 4 bins to 25 bins; 292 weights, 584 bytes.

Stored per filter as `MEL_START[i]`, `MEL_LEN[i]` and a flat `MEL_W_Q15` array
— a contiguous walk over both `mag` and the weights, which is what viper is
good at.

```
acc = 0
for k in 0..MEL_LEN[i]-1:
    acc += (mag[MEL_START[i]+k] * MEL_W_Q15[ofs+k] + 128) >> 8
```

The per-term shift is **8, not 15**, leaving `acc` in Q7 — the accumulator
carries 128x the filter output. Shifting by 15 per term truncated each of up to
43 terms independently and biased a filter low by ~20 counts on a value of a few
hundred; that is a 6% error, 0.09 octaves, and it was the dominant fixed-point
error in the whole pipeline. Bound: `32776 * 32768 >> 8 = 4197376` per term,
43 terms max, `acc <= 1.8e8` = 8.4% of int32.

### 7. Log

```
mel[i] = log2_q8(max(acc, 1)) + ((2 - g) << 8)
```

Q8 log2: 1 LSB = 1/256 octave. The `2 - g` correction undoes the FFT's nine
stage shifts (-9), the block normalisation (`+g` inverted) and the Q7 the
accumulator carries (+7): `-(g - 9) - 7 = 2 - g`.

`log2_q8` is a 65-entry table of `round(256 * log2(1 + i/64))` with linear
interpolation:

```
e = bit_length(v) - 1
m = v << (16-e)  if e < 16  else  v >> (e-16)      # 1.xxxx in Q16
idx  = (m >> 10) - 64                              # 0..63
frac = m & 1023
return (e << 8) + LOG2_Q8[idx] + (((LOG2_Q8[idx+1]-LOG2_Q8[idx])*frac + 512) >> 10)
```

Interpolation error over a 1/64 interval is at most
`(h^2/8)*max|f''| = 4.4e-5` octaves = 0.011 Q8 LSB, so the table's own rounding
dominates. **Measured worst error: 0.446 Q8 LSB = 0.0017 octaves.**

Clamping `acc` to 1 rather than special-casing zero puts a digitally silent
band at exactly 0 with no sentinel.

`MEL_FLOOR_SHIFT` exists in the reference as a tunable (floor each band at
`peak >> n`) and is **0, disabled**, by measurement — see the variants table
below.

### 8. DCT-II

12 coefficients, `c1..c12`. **`c0` is dropped.**

```
DCT_Q15[(j-1)*26 + i] = round(32767 * cos(pi*j*(i+0.5)/26))     j = 1..12

c[j] = sum over i of ((mel[i] * DCT_Q15[...] + 16384) >> 15)
```

Unnormalised: the orthonormal scaling is a constant per row, and DTW never
compares one coefficient against another, so it would only cost precision.

Shifting each term rather than the accumulated sum is what keeps this inside
int32 — the unshifted sum of 26 terms would reach 6.5e9. The cost is under a
thousandth of a Q8 LSB. **Measured: `dct.mul` peaks at 153575749, 7.2% of
int32.**

`c0` is dropped because it is overall log energy, which here is dominated by how
far the board was from the speaker's mouth. Keeping it after mean normalisation
would leave a useful syllable-envelope contour, but with a dynamic range several
times that of `c1..c12`, so it would dominate an L1 distance and would need a
weight nobody has measured. The energy contour is not entirely lost: endpointing
has already aligned the utterances.

### 9. Lifter

```
c[j] = (c[j] * LIFTER_Q12[j] + 2048) >> 12
```

Q12 gains, 4096 = unity. `LIFTER_L = 0` (off) by measurement. Unity is *exact*
through the fixed-point path — `(c*4096 + 2048) >> 12 == c` for every `c`,
negatives included — so the stage costs one no-op multiply and stays in place
for anyone who wants to measure it again.

### 10. Per-utterance cepstral mean normalisation

```
mean[j] = trunc_toward_zero(sum over frames of c[j][t], n_frames)
feat[j][t] = (c[j][t] - mean[j] + 8) >> 4
```

Truncation **toward zero**, not floor — that is what a hardware divide gives and
the device has no cheap floor-divide. On the host:
`mean = s // n if s >= 0 else -((-s) // n)`.

The final `>> 4` takes Q8 log2 down to **Q4 log2**: 1 LSB = 1/16 octave =
0.376 dB. `sum` peaks at 204195, far inside int32.

CMN over a single 500 ms word is a compromise and worth knowing about: it is the
right tool for the convolutional part of a channel mismatch, but it is estimated
from too little data here, and over one word the cepstral mean substantially
*is* the phonetic content. It helps; it does not rescue. This is the main reason
templates must be enrolled through the board's own microphone.

### 11. Delta coefficients

```
d[t][j] = sum over n=1..D of ( n * (c[t+n][j] - c[t-n][j]) )
d[t][j] = (d[t][j] * DELTA_RECIP_Q15 + 16384) >> 15
feat[12+j][t] = (d[t][j] + (1 << (s-1))) >> s      s = FEAT_SHIFT - DELTA_SHIFT
```

`D = DELTA_WIDTH`, and `DELTA_RECIP_Q15 = round(32768 / (2 * sum(n^2)))` — a
multiply by a frozen reciprocal rather than a divide, because viper has no
integer divide worth using and this way both sides round identically. Edge
frames replicate rather than shrink the window, as HTK does; shrinking it would
change the delta's scale at the edges.

**Deltas are computed from the pre-CMN statics, and that is exact rather than
merely convenient.** Cepstral mean normalisation subtracts a constant per
coefficient, and a constant cancels identically in `c[t+n] - c[t-n]`. Which is
the entire argument for having them: a fixed channel or gain offset vanishes
from a frame-to-frame difference *exactly*, with no estimate involved, whereas
CMN has to estimate that offset from half a second of audio in which the offset
and the phonetic content are not separable. For a 0.5 s word, deltas are the
more reliable channel-invariance mechanism of the two.

`DELTA_SHIFT = 2` gives the deltas four times the weight the plain `FEAT_SHIFT`
would. Measured raw, `mean |delta|` is **0.26x** `mean |static|` (8 against 34),
so an unweighted L1 distance would let the channel-robust half of the vector
carry a quarter of the vote. Two bits brings 8 up to 32 against a static 34.

Cost: the feature vector doubles to 24, so template storage and the DTW inner
loop both double. See the variants table for whether that is worth paying.

### Feature summary

**24 int16 per frame** — 12 statics `c1..c12` then 12 deltas — **48 bytes per
frame**, 100 frames per second.

Measured over the enrolment corpus (statics, 38700 values): mean `|feat|` **67**,
median **42**, 99th percentile **411**, max **859**. That is what fixes the
container at int16: the p99 alone overflows an int8, and dropping to Q2 log2 to
fit one would quantise at 1.5 dB.

Templates are stored as little-endian **uint16 biased by +32768**, so the DTW
inner loop reads them with `ptr16` and subtracts without sign extension — the
bias cancels in a difference.

### Fixed-point accuracy, measured

`python3 tools/mfcc.py --selftest` compares the integer path against a float
model of the same pipeline (same Q15 window, same magnitude mel, same log2, same
unnormalised DCT — so the difference is purely quantisation):

```
fixed vs float over 38 frames x 12 coeffs: mean |err| 2.624, worst 11.425
  (units are Q4 log2 LSBs; typical |feature| is 34)
```

The residual is the scaled FFT's noise floor reaching the log of a low-energy
mel band. It is not a coding error and it is **deterministic**: the device runs
the same integer path, so templates and queries carry the same distortion and
most of it cancels in the distance. The number that matters is the end-to-end
one below.

### Measured and proved are different claims

The prose above states **proved** bounds; the table below is **measured**. They
must not be allowed to blur, because they fail differently: a proof survives a
gain change, a measurement over a handful of signals does not.

The clearest evidence for keeping them apart is that the deliberately hostile
signal is *milder* than ordinary speech. Alternating full scale reaches 41.5% of
int32; a real utterance of "husband" reaches **49.90%**. Pre-emphasis saturation
clamps the pathological input before the FFT ever sees it, so the stress case
built specifically to probe overflow probes less than the corpus does. Anyone
reasoning from "we tested the worst case" would draw exactly the wrong
conclusion — which is also why the pre-emphasis saturation is load-bearing
rather than defensive, and must not be tidied away as a redundant clamp.

Peak int32 occupancy by stage, **measured** over 1130 frames of corpus speech:

| stage | peak | % of int32 | ceiling |
| --- | --- | --- | --- |
| `fft.tw` | 1071611968 | **49.90%** | 50.00% |
| `dct.mul` | 153575749 | 7.2% | |
| `mel.mul` | 80286176 | 3.7% | |
| `mag.sq` | 18495425 | 0.86% | 30.52% |
| `mel.acc` | 991754 | 0.05% | |
| `cmn.sum` | 204195 | 0.01% | |

`tools/mfcc.py` has a `CHECK_BOUNDS` mode that asserts every one of these fits
int32 and reports the peaks; CPython's unbounded ints otherwise hide exactly the
bug the device would hit.

## Matching

`tools/dtw.py` is the reference.

### Distance

L1 over the 12 coefficients — no multiply, and squaring only sharpens the
influence of whichever coefficient happened to be worst.

```
d(i,j) = sum over k of |query[i][k] - template[j][k]|
```

### Banded DTW

Symmetric Sakoe-Chiba:

```
D(0,0) = 2*d(0,0)
D(i,j) = min( D(i-1,j)   + d(i,j),        # (1,0)
              D(i,j-1)   + d(i,j),        # (0,1)
              D(i-1,j-1) + 2*d(i,j) )     # (1,1), weight 2
score  = D(N-1,M-1) // (N + M)
```

The diagonal's weight of 2 is what makes `N+M` a valid normaliser: every
monotone path from `(0,0)` to `(N-1,M-1)` accumulates exactly `N+M` units of
step weight, so the divisor is path-independent and **no path length has to be
tracked**. That is the whole reason for the symmetric form.

Band: for query frame `i`, the template frames considered are

```
c  = (i * M) // N
lo = max(0, c - 10)
hi = min(M-1, c + 10)
```

`BAND = 10` frames = 100 ms. The band is centred on the **ratio line**, not the
diagonal, so a globally faster or slower delivery is already absorbed and the
radius only has to cover *local* deviation.

**Duration prefilter:** reject outright (score = infinity) if the two lengths
differ by more than 2:1. Cheap, and it removes the collisions where a long word
warps onto a short one.

`INF = 1 << 29` — far above any real path cost, far below int32, so adding a
distance to it cannot wrap.

### Decision

A class holds every template of every spoken form it owns (SAD owns SAD and
SICK, so six), and its score is the minimum over all of them. Two gates:

| | |
| --- | --- |
| `THRESHOLD` | fire only if the best class score is at or below it |
| `MARGIN` | fire only if the second-best *different* class is at least this much worse |

Both are needed, and they catch different failures. The threshold stops the
utterance that resembles nothing in the vocabulary. The margin stops the
utterance that resembles two things equally — if MOTHER and BROTHER both score
190, the right answer is silence whatever the absolute score is. Measured, the
margin is the more valuable of the two: near-miss confusions cluster tightly
while genuine matches stand clear.

**Recommended: `THRESHOLD = 750`, `MARGIN = 120`.** See below.

### Rejection is a feature, not a safety net

The absolute-distance test is the single most important line in the matcher and
it is worth being explicit about why. `argmin` alone *always* returns a word.
Point a nearest-neighbour classifier at a sound outside its vocabulary and it
does not hesitate — it confidently names whichever of its twenty-two templates
happens to be least unlike a cough. Every one of those is a wrong answer
delivered with the same certainty as a right one.

For this toy that is the whole difference between charming and broken. A
rejection produces "PLEASE GO ON", which is what DOCTOR says anyway and which
nobody can tell from intended behaviour. A confident misfire produces "DO YOU
OFTEN THINK OF MONEY" when the user said "morning", and the illusion does not
survive it.

So the threshold is not a tuning parameter bolted on at the end; it is the
mechanism that converts "I have no idea" into "didn't catch that". It also
means the honest metric is precision and recall separately — a single accuracy
figure averages together the one error that matters and the one that does
not.

## What it measures

`tools/say_corpus.py` synthesises a labelled corpus with macOS `say`, which
writes 16 kHz mono int16 WAV directly. One voice (Samantha), enrolment at
172-178 wpm, test at 150-205 wpm, plus a per-utterance channel model (gain
-6..+3 dB, additive noise at -46..-34 dBFS, first-order tilt ±0.15) and 400 ms
of background either side.

**These numbers are optimistic and it matters by how much** — see the banner at
the top of this document. Varying rate and channel reproduces *session*
mismatch, not *articulation* mismatch. Treat them as an upper bound and as a
way to rank design choices against each other. Re-run the harness against
templates enrolled through the board before trusting any absolute figure.

Corpus: 21 classes / 22 spoken forms, 3 enrolment takes each (66 templates,
44 frames mean, 28..61), 88 in-vocabulary test utterances, and 180
out-of-vocabulary ones chosen as what somebody actually says to a therapist toy,
including deliberate near-misses.

The out-of-vocabulary set is scored in two parts, because lumping them together
makes the front end look worse than it is and hides the collision that matters:

- **Morphological variants** — MOTHER'S, WORKED, DREAMING, COMPUTERS.
  Firing here is *correct* behaviour for DOCTOR, so it counts as neither a hit
  nor a miss.
- **Must stay silent** — everything else, including every **retired** word.

### Retired words stay in the corpus deliberately

`say_corpus.RETIRED` names every word that was once in the vocabulary and is
now cut — currently NO, WANT and NEED — and generates them into the silent set
with the full four test rates rather than the two an ordinary out-of-vocabulary
noun gets. They were judged worth spotting once, so they are the words most
likely to false-fire, and they deserve more evidence than an arbitrary noun.

It is there because of a trap. The corpus is generated by enumerating
`vocab.FORMS`, so the moment a word is cut it stops being recorded — and the
utterances proving the spotter stays quiet on it vanish from the test set at
exactly the moment they start to matter. The sweep would go on reporting
precision 1.000 while no longer testing the thing the cut was made for.

**Deriving a test set from the system under test is right until the thing being
tested is the system's *absence* of something.** The same shape bit an ELIZA
rule test: it built word combinations from `vocab.LABELS`, filtered to those
containing WANT, and after the cut every iteration was skipped — green, in the
count, and cited as the evidence for the removal.

The harness prints how many retired-word utterances it scored, so a corpus that
has quietly lost them is visible rather than merely unchanged.

Measured: all 12 retired utterances stay well clear. The nearest is "want" ->
WIFE at **968**, against an operating threshold of 750.

### Precision and recall

Precision = hits / fires. Recall = hits / in-vocabulary utterances. A miss costs
nothing — ELIZA deflects, in character — while a false fire answers "morning"
with "DO YOU OFTEN THINK OF MONEY", and that is where the illusion dies.

| threshold | margin | recall | false fires |
| --- | --- | --- | --- |
| 700 | 120 | 0.943 | 0 |
| **750** | **120** | **0.966** | **0** |
| 800 | 120 | 0.977 | 1 |
| 850 | 120 | 0.989 | 1 |
| 900 | 120 | 0.989 | 1 |

**Recommended: threshold 750, margin 120 — precision 1.000, recall 0.966**, 85
of 88. The first genuine false fire is "other" → FATHER at 727, which the
margin gate rejects (it sits only 98 from its rival); the next is at 800, so the
operating point has 6.7% of headroom. Distances roughly doubled against the
12-wide feature vector, because the L1 sum now runs over 24 terms.

Both gates still earn their place, and the margin gate is doing the work here:
at margin 0 the threshold has to sit at 706 to stay clean, and its cushion is
3%.

### Front-end variants, and what this corpus cannot tell you

`python3 tools/dtw.py --tune corpus/`, every variant against the same audio,
recall at the best zero-false-fire setting:

(Run before deltas were adopted; the absolute thresholds are for the 12-wide
feature vector and are not comparable with the table above.)

| variant | precision | recall | threshold | margin | bytes |
| --- | --- | --- | --- | --- | --- |
| **statics only, 100–7600** | 1.000 | 0.990 | 316 | 80 | 75696 |
| deltas D=2 | 1.000 | 0.979 | 706 | 0 | 151392 |
| deltas D=2, unweighted | 1.000 | 0.990 | 448 | 120 | 151392 |
| mel 300–3400 x20 | 1.000 | 0.990 | 242 | 80 | 75696 |
| mel 300–3400 + deltas | 1.000 | 0.990 | 627 | 120 | 151392 |
| mel 200–5000 x24 | 1.000 | 0.990 | 294 | 80 | 75696 |
| cepstral lifter L=22 | 1.000 | **0.927** | 2119 | 0 | 75696 |
| mel floor 36 dB | 1.000 | 0.990 | 309 | 80 | 75696 |
| band 5 | 1.000 | 0.990 | 316 | 80 | 75696 |
| band 20 | 1.000 | 0.990 | 316 | 80 | 75696 |

**Nine of the ten land on exactly 0.990, and that is the finding.** This corpus
is saturated: one voice, one channel, cross-session only. It has enough
resolution to reject liftering and to have killed NO, and no resolution at all
on anything else.

That is not a defect in the corpus, it is what it is for. Deltas and a
band-limited filterbank are **channel-robustness** measures. A test set with no
channel mismatch in it cannot price them, and reading "no improvement" here as
"no benefit" would be exactly the wrong conclusion. The condition that prices
them is cross-microphone, which is `tools/mic_margin.py compare`.

What this table does settle:

- **Liftering makes it worse**, costing 6 points of recall — the one variant
  everyone argues about from first principles. HTK's lifter equalises
  coefficients for a diagonal-covariance Gaussian, where scaling is irrelevant;
  inside an L1 distance it multiplies c11 and c12 by twelve, and those are the
  noisiest. `LIFTER_L = 0`.
- **The mel floor buys nothing**, so it stays off rather than paying real
  spectral detail for it. `MEL_FLOOR_SHIFT = 0`.
- **Band 5, 10 and 20 are identical**, so the Sakoe-Chiba radius is not binding
  here and band 5 would halve DTW cost for free. Kept at 10: TTS rate variation
  is globally uniform and the ratio line absorbs all of it, while a person's
  local timing wanders in a way this corpus cannot show. The cheapest available
  speed-up if it survives real audio.
- **Deltas must be weighted.** Unweighted they match baseline; at `DELTA_SHIFT`
  the plain shift they lose a point. Either way they double storage and DTW
  cost, and on same-microphone audio they buy nothing — as expected.

### Confusions

Inter-word confusion is essentially **zero**: across the entire threshold sweep
the "fired as the wrong class" count never exceeds 1. Every false positive in
the corpus is an out-of-vocabulary word, not a keyword mistaken for another
keyword.

Separation from the nearest rival class, mean over test takes (higher is
safer; the operating threshold is 300):

| gap | class | nearest rival |
| --- | --- | --- |
| 97 | mother | father |
| 145 | father | brother |
| 151 | brother | mother |
| 173 | want | love |
| 198 | death | yes |
| 202 | wife | want |
| 215 | sad | death |
| 221 | love | want |
| 256 | money | sorry |
| 288+ | everything else | |

MOTHER / FATHER / BROTHER are the tight cluster, exactly as predicted — they
share `/-ʌðər/`, which is 60% of each word. **They are nonetheless separable**
and none of them needed to be dropped: the initial consonants differ about as
much as English consonants can (nasal murmur, fricative, stop-plus-liquid), and
the ZCR endpointing pass is what keeps those onsets in the segment. This is the
concrete payoff from that pass.

The words that attack FATHER are not MOTHER and BROTHER but *other* and
*another* and *wonder* — same rime, no keyword involved. That is what the margin
gate is for.

### One word had to be dropped: NO

NO is /noʊ/. So is KNOW. Homophones, not near-misses; no acoustic matcher
separates them, and "I don't know" and "you know" are among the commonest things
anyone says to a therapist. Measured: "know" matched the NO templates at
distance **172**, while the best genuine in-vocabulary match in the entire set
scored **143**. There is no threshold between them. THOUGH and FELT attacked NO
the same way.

Keeping NO capped the whole vocabulary at **7% recall** for zero false fires,
because one class's false alarms set the threshold for all twenty-three.
Dropping it gave **89.6%** immediately, and **99.0%** with the margin gate.

The lesson generalises: a short, unstressed, vowel-dominated token is the worst
thing to put in a small-vocabulary spotter, because English is full of function
words that sound like it. Long polysyllables — COMPUTER, CHILDREN — are nearly
free. "no" now falls through to a deflection, which is in character.

## What a template set costs, and how it must be shipped

Measured, 21 classes / 22 forms / 3 takes = **66 templates, 2918 frames**;
templates run 28..61 frames (280..610 ms), 44 mean. Expanded, that is
**140064 bytes — 137 KB**, on a ~490 KB heap, alongside the ELIZA rule set,
the 200x200 framebuffer and the capture buffer. It is the largest single
allocation the program makes.

### Ship a blob, not a module

`tools/record_templates.py --format bin` (the default) writes `templates.bin`
plus a ~4 KB loader. **Do not use `--format py`.** An escaped bytes literal
costs four characters per byte, so the same data is ~600 KB of Python source,
and MicroPython has to compile that inside the heap it is about to fill. Even
cross-compiled, the constant lands in RAM in one block at a time nobody
chooses. `--format py` survives only for host-side inspection and prints a
warning.

The loader does one allocation and reads straight into it:

```python
buf = templates.load(bytearray(templates.BUFFER_BYTES), spotter.expand)
```

`load(buf=None, expand=None)` allocates `BUFFER_BYTES` itself if given nothing,
but `main()` passing its own buffer is the better shape: allocation *order* is
a whole-program concern and the templates module cannot know what else is about
to be reserved.

**`load()` keeps no reference.** The returned buffer is the only one, so
whatever holds it must go on holding it for the life of the program — the name
looks dead after the spotter is constructed and deleting it would free 137 KB
at some unrelated later collection.

**`expand` is required when `PACKED == "statics"`, and omitting it raises.**
That is deliberate rather than defensive: matching against unexpanded statics
does not crash, it reads half a template followed by a block of zeros and
returns confident nonsense. A silent wrong answer is the failure mode this
whole document is arranged against, so the one call that could produce it is
made impossible to write.

Never `array("h", bytearray(n))` — that holds the bytearray *and* the array at
once and peaks at twice the size. `sounds.allocate_bytes` records what taught
this project the rule: 419 KB free, largest block 174 KB, and a 140 KB clip
still would not allocate, because building it briefly needed 280 KB.

### Allocation order: largest *transient*, not largest steady

"Largest first", the rule `Shaker._setup` follows, is a shorthand for something
more precise, and here the shorthand gives the wrong answer.

The templates are the largest thing at rest (137 KB against the capture
buffer's 94 KB). But `listen.allocate_samples` builds the capture buffer as
`array("h", bytearray(2 * count))`, which holds the bytearray **and** the array
at once — so it briefly needs **188 KB** to end up with 94. The template buffer
has no such transient: `bytearray(BUFFER_BYTES)` is one allocation, and the
expansion works inside it through 1.4 KB of scratch.

So the capture buffer is the bigger *spike* even though it is the smaller
*resident*, and it should be allocated first, when the heap is emptiest:

| order | resident | **peak** |
| --- | --- | --- |
| templates, then capture | 235.4 KB | **324.3 KB** |
| **capture, then templates** | 235.4 KB | **235.4 KB** |

Capture-first is **92.3 KB** cheaper at the peak, and its peak never exceeds
its resident figure at all — the 188 KB transient happens while only 0 KB is
held, so it costs nothing. Templates-first pays the transient on top of a
already-held 137 KB.

Worked, in bytes:

```
capture first:  bytearray(96000)+array = 192000 peak, 96000 resident
                bytearray(140064)      = 236064
                + 1464 scratch         = 237528 peak, freed to 236064
                framebuffer 5000       = 241064
templates first: bytearray(140064)     = 140064
                + 1464 scratch         = 141528 peak, freed to 140064
                capture 140064+192000  = 332064 peak
```

The rule to carry forward is **order by transient peak, not by final size** —
they differ exactly when one allocation has a hidden double and another does
not. And on a heap that does not compact, the peak is what fails.

**The transient cannot simply be avoided, and it is worth writing down why so
nobody re-derives it.** `AudioPIO.dma_record_into` sets `count=len(buf)` with
16-bit transfers, so `len()` has to be a count of samples: a bare `bytearray`
would set a count of *bytes* and run the DMA past the end of the buffer.
MicroPython has no `memoryview.cast` to bridge that, and every form that
produces an `array("h")` holds both objects at once. So the 188 KB is
structural.

It could be removed by giving `dma_record_into` an explicit sample count, but
that edits `src/audio_pio_mpy.py`, which the working Magic 8-Ball shares, for no
gain — capture-first already makes the spike free, because it lands when
nothing else is held. Ordering solves it; leave the allocation alone.

### Statics-only packing

`--pack statics` (the default) stores **12-wide Q8 statics** — 70128 bytes,
exactly half — and the device expands them to the 24-wide features in place at
start-up. That halves the flash footprint and the `mpremote` transfer. It does
**not** reduce match-time RAM: `BUFFER_BYTES` is still allocated up front and
the blob is read into its front.

Two things make this exact rather than approximate, and both are easy to get
wrong:

- **The blob stores Q8, not the Q4 the features use.** Deltas are regressions
  over the statics, so computing them from Q4 values quantises twice and does
  not reproduce `mfcc()`. Storing Q8 and shifting afterwards does. Headroom is
  the cost: largest `|Q8 static|` measured over the corpus is 14511 against the
  int16 limit, **2.3x**, versus 38x for Q4. `pack_statics()` returns a clamp
  count and `record_templates.py` exits non-zero if it is ever non-zero, since
  a clamped template is silently slightly wrong forever.
- **Expansion is per template, last to first, through a scratch buffer.**
  Template `k` writes to `[48*F_k, ...)` while the statics still needed lie
  below `24*F_k + 24*n_k`, and `48*F_k >= 24*F_k` always, so nothing below is
  disturbed. Copying each template's statics into `SCRATCH_BYTES` (1464, the
  longest template) first means the source of the frame being written is never
  the buffer being written to — which removes the within-template overlap
  argument instead of getting it right. Deltas also replicate at each
  template's own edges and must not read into the neighbour, which is the
  other reason this is per template and not one pass.

**For the first hardware session, use `--pack full`.** It writes the finished
24-wide features, so `templates.load(buf)` needs no expansion pass and
`talk.py` works the moment `spotter.spot` exists. RAM is identical either way
and the probe's heap figure is unchanged; the cost is a 140 KB file instead of
70 KB and a slower `mpremote cp`. The reason is not convenience: the expansion
is pure integer arithmetic already proven bit-exact on the host over 66 real
templates and every awkward synthetic shape, so running it on the board for the
first time buys nothing — while a bug in its viper port would corrupt templates
in a way indistinguishable from a bad sample rate, a mis-set mic gain, bad
endpointing or an uncalibrated threshold, all of which *are* being tested for
the first time in that session. Do not debug a known-good routine alongside
five unknowns. Switch to `statics` once the recogniser works end to end; it is
one flag and a re-run.

`tools/test_templates.py` checks the reconstruction against `mfcc()` on all 175
corpus templates plus short and multi-template synthetic blobs (1, 2, 3 frames;
`(40, 3)`; `(1, 1, 1)`), because the boundary cases are where an off-by-one
would hide. **Re-run it after any change to the packing or the expansion.**

If RAM is still the binding constraint after that, the remaining levers are two
takes per word instead of three, and dropping deltas — a one-line revert, but
the last thing to spend, since deltas are what measurement says carries the
channel-robust information.

## The margin experiment

`tools/mic_margin.py` — the experiment that should be run before trusting any
of the above, because it is the one that can invalidate the approach rather
than tune it. It asks two questions:

1. **Same microphone: is there margin at all?** If same-word distances are not
   clearly smaller than different-word distances, DTW over MFCCs is the wrong
   tool and no threshold rescues it.
2. **Cross microphone: can templates be enrolled anywhere but the board?**

It records **both microphones simultaneously**, so both hear the same
utterance and the only difference is the channel. Recording into each in turn
would confound the channel with how differently the word was said the second
time, which is the quantity being measured.

Reference from the synthetic corpus, same channel, cross-session: mean
within-word 256, mean between-word 733, **ratio 2.86**. Synthetic speech is
more repeatable than a person, so real same-microphone numbers should sit below
that; it is an upper reference, not a target.

### Predicted, not yet measured

`tools/mic_margin.py simulate` passes corpus audio through two different
channels — the "microphone difference" is one I chose (+6 dB, brighter tilt,
quieter floor), so this **predicts** and does not measure:

| condition | within | between | ratio | top-1 |
| --- | --- | --- | --- | --- |
| board → board | 297 | 691 | 2.33 | 69/70 |
| mac → mac | 325 | 491 | 1.51 | 67/70 |
| **mac templates → board queries** | 524 | 722 | **1.38** | **36/70** |

Top-1 collapses from 69/70 to **36/70** and 59% of the margin is gone. If the
real run reproduces anything like that, Mac-enrolled templates are finished —
and note the failure mode: at ratio 1.38 the right word and the wrong word are
about equally far away, so the mismatch does not produce silence, it produces
**confident errors**. That is the worst possible failure for this toy, and it
is why enrolment goes through the board.

### What the cross-microphone condition decided

`tools/mic_margin.py compare` re-scores one session under several front ends.
This is the measurement that priced deltas and the band-limited filterbank,
because the same-microphone corpus could not (everything scored 0.990 there).

| front end | same-mic ratio | cross-mic ratio | kept | cross-mic top-1 |
| --- | --- | --- | --- | --- |
| statics, 100–7600 | 2.33 | 1.38 | 59% | 36/70 |
| **+ deltas** | 2.22 | 1.35 | 61% | **43/70** |
| statics, 300–3400 x20 | 2.22 | 1.38 | 62% | 42/70 |
| statics, 200–5000 x24 | 2.33 | 1.40 | 60% | 41/70 |
| deltas + 200–5000 x24 | — | 1.36 | — | **43/70** |
| deltas + 300–3400 x20 | 2.05 | 1.33 | 65% | **43/70** |

Two conclusions, and the second is the one that changed a decision:

1. **Deltas earn their cost.** Top-1 goes from 36/70 to 43/70 under channel
   mismatch — a 19% relative gain, about 1.7 standard errors at n=70, so
   suggestive rather than conclusive, but pointing the way the theory says it
   should.
2. **Band-limiting is subsumed by deltas and was therefore dropped.** On its
   own it helps (36 → 41 or 42). Combined with deltas it adds *nothing*:
   100–7600 and 200–5000 both score exactly 43/70. Meanwhile it costs the
   5–8 kHz fricative band that separates FATHER from MOTHER and from "other" —
   and "other" → FATHER is the most dangerous false fire in the corpus. Keeping
   the fricative onsets through the ZCR endpointing pass and then discarding
   them in frequency would have undone that work for no measured gain.

None of it rescues a channel mismatch: 43/70 is still far below the 69/70 that
same-microphone enrolment gets. Deltas reduce the damage; they do not repair
it. Enrol on the board.

## Cost on the device

**Measured**, on the RP2350 at MicroPython 1.28.0, native arch `armv7emsp`.
`src/spotter.py` carries the front end twice: a plain version, which is the
specification, and a `@micropython.viper` transcription that shadows it on the
device. Per frame:

| stage | plain | viper | |
| --- | --- | --- | --- |
| window, block-float, bit-reverse | 4.50 ms | **0.72 ms** | |
| FFT | 80.9 ms | **3.66 ms** | 22x, and 65% of what remains |
| magnitude | 14.0 ms | **0.77 ms** | restoring sqrt, no divide |
| mel filterbank + log | 4.03 ms | **0.30 ms** | |
| DCT | 2.46 ms | **0.14 ms** | |
| **whole frame** | **122.9 ms** | **6.2 ms** | 20x |

Pre-emphasis runs at 0.598 us/sample, 9.6 ms for a second of audio. The
per-utterance tail — CMN and the delta/pack pass — is 0.73 ms and 3.20 ms over
98 frames.

A turn, against a 66-template set with the corpus's length distribution:

| query | front end | matching | turn |
| --- | --- | --- | --- |
| 44 frames (a typical word) | 273 ms | 616 ms | ~889 ms |
| 48 frames (0.5 s of audio) | 297 ms | 672 ms | 969 ms |
| 98 frames (1.0 s of audio) | 603 ms | 562 ms | 1165 ms |

The 98-frame row matches faster than the 48-frame one because the duration gate
rejects 40 of the 66 templates before the inner loop — the gate doing its job.

### The one end-to-end run of the whole program

Those rows are the recogniser measured in isolation. The program itself —
capture, endpointing, the engine, the panel — has been run on the board **once**,
on 2026-08-18 via `mpremote run src/talk.py`, and this is the only record of it:

```
-> speech 960..9280 (520 ms), heard -
   I am not sure I understand you fully [full, 3707 ms]
-> speech 6400..35200 (1800 ms), heard -
   Please go on [partial, 3213 ms]
```

Two turns, both reaching the glass. The timings are press-to-reply wall clock
and include however long the screen was held, so they are not comparable with
the table above; what they establish is that `listen_once` → `vad.Endpointer` →
`Conversation` → `screen.render` → `Panel.show` runs, and that the
partial-refresh policy works on the second turn as `docs/design.md` says it
should.

This is the same run behind the "spans of 520 ms and 1800 ms off actual
utterances" line in `1bca7f5`, which dates it: **it predates the chirp-settle
fix (`12a942e`) and the `MIC_GAIN` 3 → 1 change (`61967bc`)**, so the capture
path as it stands today has not been through it. And `heard -` on both turns is
not a rejection — there were no templates on the device, and there still are
none. **No reply beyond a deflection has ever been spoken to this program.**

**Matching dominates, about two to one** rather than the eight to one the
operation counts suggested: the front end's FFT is heavier per operation than
the matcher's L1 loop, so counting operations under-priced it. `BAND` is
therefore the lever that is left, and it is very nearly linear in `2*BAND + 1`:

| band | 66 templates, 50-frame query |
| --- | --- |
| 20 | 1146 ms |
| **10** (current) | **696 ms** |
| 5 | 409 ms |

The corpus scored 5, 10 and 20 identically, so halving it looks free — but that
corpus is saturated (see the variants table above) and the claim wants
re-measuring against templates enrolled through this board before the 287 ms is
banked.

### What the viper port cost to get right

Three things about viper that are silent when wrong, all probed on this build
rather than assumed:

- a `ptr16` load is zero-extended, so int16 input needs an explicit `- 65536`;
  a `ptr32` load is a whole machine word and `int()` is what types it *signed*,
  which is why every load in the port is wrapped in it;
- `2**30` is one past MicroPython's small-int range, so a bare `1 << 30`
  literal inside a viper function is an *object* and the comparison against it
  fails to compile. `int()` once, outside the loop, is the fix;
- viper spills every local to the stack, so statement count is the thing that
  costs, not loads. Measured on the DTW inner loop: unrolling four ways with
  `if/else` took a band cell from 9.56 to 7.53 us (21%), while a branchless
  `abs` was *slower* (10.68) and walking two indices instead of recomputing
  `qi + k` was slower still (10.84).

Only `ImportError` is caught around the viper blocks, so they fall back to the
plain path on the host and nowhere else. A compile error on the device is meant
to be loud: silently running at 122.9 ms a frame is the failure the port exists
to remove.

## Proving the device port is bit-exact

`src/speech_fixtures.py`, generated by `tools/make_fixtures.py`, exists because
drift does not announce itself. A device front end that is *nearly* right
produces distances that are slightly wrong, which reads as "the recogniser is a
bit poor" — indistinguishable from bad enrolment, a channel mismatch, or an
untuned threshold. By the time the front end is suspected, four other things
have been changed.

Four cases, each pinning the pipeline **stage by stage** rather than only at the
output, so a mismatch localises:

| stage | pinned as |
| --- | --- |
| `saturated` | count of pre-emphasis saturations |
| `preemph` | checksum of the pre-emphasised signal |
| `peak`, `g` | frame 0's block-float peak and shift |
| `windowed` | checksum after window and `<< g` |
| `fft` | checksum of re[] and im[] after the transform |
| `mag` | checksum of the 257 magnitudes |
| `mel` | all 26 Q8 log2 values, in full |
| `cepstra` | all 12 raw cepstra, in full |
| `features` | all 5 frames x 24, in full |
| `DTW_0_1` | the distance between two cases, pinning the matcher too |

A mismatch at `cepstra` with `mel` matching is a DCT bug. At `mel` with `mag`
matching, a filterbank bug. At `fft` alone, twiddles or scaling. Without the
intermediates every one of those presents identically and finding which means
bisecting 42000 operations by hand.

Checksums are `acc = (acc * 31 + v) & 0x3FFFFFFF` — order-sensitive, no sign
games, three lines of viper. Mel, cepstra and features are stored whole because
knowing *which* coefficient is wrong is worth the bytes.

### The overflow case

CPython's ints are unbounded, so the host cannot notice a value that would wrap
on the device; viper's `int` is a 32-bit machine word that wraps **silently**.
`mfcc.CHECK_BOUNDS` catches this on the host and nothing catches it on the
board, so one fixture is a deliberately hostile alternating full-scale signal,
and **every case records the peak int32 magnitude reached at each guarded
expression**. A device that agrees on the quiet cases and disagrees on
`fullscale` has an overflow, and the recorded peaks say which stage.

Measured peaks, as a fraction of int32:

| case | peak | pre-emphasis saturations |
| --- | --- | --- |
| speech (real MOTHER) | 947900160 — **44.1%** | 0 |
| fullscale (hostile) | 890721744 — 41.5% | 1039 |
| quiet | 805265408 — 37.5% | 0 |
| silence | 16793088 — 0.8% | 0 |

Note that **real speech uses more headroom than the hostile case**, because
saturation clamps the hostile one before it reaches the FFT. That is the
saturation stage doing its job, and it is why the worst case for overflow is
ordinary loud speech rather than the pathological signal — worth knowing before
anyone "optimises away" the saturation as defensive.

## Changing any of this

`TEMPLATE_FORMAT` in `tools/mfcc.py` is 1. Every constant in this document is
part of it. Change one and every enrolled template is invalid: bump the number,
re-run `python3 tools/mfcc.py --emit-tables`, and re-enrol.
