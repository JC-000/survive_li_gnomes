# A speaker-independent spotter — design notes and where it got to

Can the ball recognise **anybody**, without a human recording a keyword
campaign? [speech-design.md](speech-design.md) costed a trained CNN against DTW
and chose DTW on one row of the table — **recordings per word**, 50-100 against
DTW's 3. The idea under test here is that macOS `say` makes that row free: train
on synthetic voices, and the spotter stops being speaker-dependent.

This document records what was built, what was measured, and what is still
unknown. It worked, on the one speaker it has been tried on.

> ## The result, in three numbers
>
> Trained on 8 `say` voices, tested on the person who will use it:
>
> | | top-1 | precision 1.000 at recall |
> | --- | --- | --- |
> | CNN, synthetic held-out voices | 0.626 | 0.013 |
> | **CNN, real speaker** | **0.700** | **0.500** |
> | DTW control, same real speaker | 0.300 | **0.000** |
>
> **The model recognises a real human better than it recognises held-out
> synthetic voices**, and it clears the incumbent's bar decisively: at a
> threshold where it never once fires wrongly, it catches half the keywords,
> where DTW with synthetic templates catches none at any threshold.
>
> **n=10 keywords and 12 negatives, one speaker, one session, one room.** Half
> of ten is five utterances; the error bar is enormous. This answers "does a
> synthetically-trained model recognise a real person at all" — which was the
> question — and nothing finer.

## The endpointer looked broken, and was being fed a chirp

This section is kept in the order it happened, because the wrong diagnosis was
reached twice on good evidence and the shape of that is worth more than the
conclusion.

**The symptom.** Ten utterances captured through the board — `src/vad.py` found
no word at all in **6 of them**, and the four it did find came back at 150-310 ms
against 300-640 ms for the same words synthetically: catching the vowel and
losing the onset, which is exactly what `docs/speech.md` warns stops separating
FATHER from MOTHER.

**The mechanism, and this part survived.** Writing `IMN` for the background
estimate (first 10 frames) and `IMX` for the loudest frame, `docs/speech.md`
gives `ITL = min(3*(IMX-IMN)/100 + IMN, 4*IMN)` and `ITU = 5*ITL`. The `4*IMN`
branch only binds above `IMX/IMN = 101`, which nothing approaches, so in
practice `ITU = 0.15*IMX + 4.85*IMN`. Speech is found only if `IMX > ITU`:

> **The endpointer requires the loudest frame to exceed
> `4.85 / 0.85 = 5.71` times the background.**

That predicted cutoff separated the ten takes **perfectly** — rejected at
3.6, 3.7, 4.4, 4.9, 5.5, 5.7 and found at 6.3, 7.1, 7.9, 8.6, with `father_01`
sitting exactly on the boundary and rejected.

**Two wrong conclusions drawn from it.** First, that the cause was clipping:
refuted by `dream_01`, which had zero clipped samples and was rejected anyway.
Then, that the cause was a loud room and the fix was to lower the capture gain:
refuted by arithmetic — `IMX/IMN` is **gain-invariant**, and simulating a 6 dB
reduction across all ten takes returned every ratio identical to three figures.
That left "the board's noise floor is 24-28 dB above the ES8311's own -44 dBFS,
so it is the room", and a recommendation to change `ITU`'s multiplier.

**All of which was measuring a bug.** `Recorder.chirp()` played a 300 ms
activation tone and waited on `play_finished()` — which reports that the DMA has
drained into the PIO FIFO, not that the speaker has stopped. Capture opened
while the tone was still sounding, so **every recording began with the tail of a
chirp**, landing precisely on the frames `IMN` is estimated from. A contaminated
background collapses `IMX/IMN`, which is the ratio the algebra identifies. The
mechanism was right; the thing inflating it was not the room.

Found and fixed by the lead in parallel with the analysis
(`CHIRP_SETTLE_MS = 140` in `src/listen.py`, timed from the start of playback
rather than from the DMA finishing). Their evidence, discarding the first N ms
of 22 pre-fix takes:

| skip | endpoints successfully |
| --- | --- |
| 0 ms | 8 / 22 |
| **300 ms** | **22 / 22** |
| 400 ms | 21 / 22 |
| 500 ms | 14 / 22 |

**Re-measured here on ten post-fix captures, independently:**

| | before | after |
| --- | --- | --- |
| endpointed | 4 / 10 | **10 / 10** |
| `IMX/IMN` | 3.6 – 8.6 | **22.6 – 49.3** |
| `IMN` (background) | 280k – 390k | **73k – 90k** |
| endpointed length | 150 – 310 ms | **250 – 440 ms** |

The background fell about fourfold and the ratio rose four to sixfold. **`ITU`
is nowhere near binding on a clean capture** — the margin is now 4x to 9x the
requirement — so do not touch it. The 5.71x rule has still never been tested
against this microphone; every measurement that appeared to test it was taken
through the bug.

### What this cost, and the rule that would have saved it

Two confident diagnoses, both from correct measurements, both wrong, because
every number came through a fault upstream of the thing being measured. The
algebra was right the whole way and pointed at `IMN` from the start; what was
missing was asking **why `IMN` was large** rather than assuming the answer.

`CLAUDE.md`'s standing rule is that "the code ran" never proves the hardware
did anything. The version of it that applies here: **a measurement of a signal
is not a measurement of its source.** A background estimate is a number about
the first 100 ms of a buffer, not about the room, and the two only coincide when
nothing else is in that buffer.

### The real-speaker result

`tools/si_eval.py build/si_real.tflite corpus-tts/manifest.json --takes ...`,
int8 model, 22 real utterances (10 keywords, 12 negatives):

| | top-1 | precision 1.000 at recall | at 95% precision |
| --- | --- | --- | --- |
| CNN, synthetic held-out voices (2) | 0.626 | 0.013 | 0.013 |
| **CNN, real speaker** | **0.700** | **0.500** | 0.500 |
| DTW control, same real speaker | 0.300 | 0.000 | — |
| DTW control, held-out synthetic voices *(dry-run)* | 0.753 | 0.233 | 0.301 |
| DTW, same speaker (`docs/speech.md`) | — | 0.966 | — |

At threshold 0.598 the model fires on MOTHER, FATHER, HAPPY, SORRY and one of
the two SAD-class words, is silent on the other five keywords, and is **silent
on all twelve negatives** — including the near-rime attackers *other*,
*another*, *wonder*, *mothers*, *brothers* that set the DTW threshold, and the
retired *know*, *want*, *need*. Zero false fires of either kind.

**The synthetic and real precision figures are not comparable, and the
difference is not the speaker.** The synthetic set has 1386 must-stay-silent
utterances chosen adversarially against 621 keywords; the real set has 12 against
10. A precision-1.000 operating point is far harder to reach against 1386
attackers than against 12, so the 0.013 is a much harsher test than the 0.500,
not evidence that synthetic voices are harder to recognise. Read the **top-1**
column for the speaker comparison and the **precision** column only within a row.

**Why the real speaker beats held-out synthetic voices** is worth stating rather
than celebrating: the two held-out voices are Daniel and Samantha, one family
each, and with n=2 the synthetic figure has an error bar of its own. The honest
reading is that the real-speaker result is *not worse*, which is the surprise —
`docs/speech-design.md` predicted a large fall and the brief for this work said
to expect one.

### What is still true

- **si-corpus's SNR result stands and is independent of this bug.** The same
  endpointer fails on **49% of utterances at 8 dB SNR and 83% at 6 dB**, against
  0% at 11 dB and above. The margin is thin even when the capture is clean, and
  `endpoint_ms: null` remains a real population in the corpus.
- **The clipping is fixed at source, and the corpus gap it revealed is not.**
  `MIC_GAIN` was 3, at which every real utterance pinned full scale — 2415
  saturated samples across ten keywords, in runs of 2-3, which is a waveform
  riding the rail at its peaks rather than isolated glitches. At gain 1 the
  peaks are 6723-23008 and **zero samples clip**. Lowering it was safe precisely
  because `IMX/IMN` is gain-invariant, so it could not cost the endpointer
  anything — and it did not: zero rejections either way.

  What survives is that **saturation appears nowhere in the training corpus**,
  which draws gain and tilt but never clipping. This particular microphone no
  longer produces it, but a louder speaker or a closer one will, and the corpus
  has no example of it. Still worth an augmentation axis; no longer urgent.
- **The feature dynamic-range gap is real, and it is not the capture.** Three
  versions of the same 22 words are kept on disk, so this is a controlled
  comparison rather than an inference:

  | set | n | std | p01 | p99 | cells at -128 |
  | --- | --- | --- | --- | --- | --- |
  | real, chirp-contaminated | 8 | 11.07 | -43 | 31 | 0.00% |
  | real, clipped (mic gain 3) | 22 | 17.91 | -51 | 50 | 0.00% |
  | **real, clean (mic gain 1)** | 22 | **16.88** | -53 | 50 | 0.00% |
  | synthetic, dry-run (no noise) | 400 | **43.12** | -128 | 89 | 3.27% |

  **Clipping was not the cause: removing it changed nothing** — 17.91 clipped
  against 16.88 clean, with the clipped set marginally the *higher* of the two.
  The chirp was worth about six points. What remains is a **~2.6x** gap that
  belongs to the signal chain rather than to any capture fault, and both capture
  faults are now fixed so it cannot be attributed to them again.

  One component is an artefact of the comparison rather than of speech:
  **3.27% of synthetic cells sit at the -128 clamp against 0.00% of real ones**,
  because `say` renders true digital silence in the lead and tail, which after
  per-band mean subtraction is a large negative excursion a microphone with a
  noise floor cannot produce. Excluding those takes synthetic from 43.12 to
  36.50 — some of the gap, not most of it.

  The remaining hypothesis is that the **dry-run corpus has no additive noise at
  all** while the generated corpus does, and noise raises the floor and
  compresses exactly this contrast. Testable the moment the corpus exists, and
  worth doing first: it decides whether a model would read inputs from a third
  of the range it trained on.

## Labels, used as [speech-design.md](speech-design.md) uses them

| Label | Means |
| --- | --- |
| *dry-run* | Measured on a **throwaway 10-voice corpus I generated to prove the harness**. Not a result. Never quote these as findings. |
| *real corpus* | Measured on `corpus-tts/`, the voice-disjoint corpus with a channel model. **There are none of these yet — the roster is frozen but the audio was never generated.** |
| *verified* | Checked by running the thing, on the host |
| *unknown* | Not established — see [What is still unknown](#what-is-still-unknown) |

## Why synthetic voices at all

Training data is the whole difficulty. Google Speech Commands is 105,829
utterances from 2,618 speakers and overlaps this vocabulary by **two words**
(YES, and NO which has since been dropped) — no MOTHER, no FATHER, no DREAM, no
SAD. Recording real people costs hours each and we would want ten or more.
macOS `say` ships ~184 voices for free. That is the whole hypothesis.

Frame any model result against the DTW control below rather than against the
shipped 0.966: **a speaker-independent recogniser that reaches half the shipped
recall is doing well, not badly.**

## What the model sees, and why it is not a second front end

`tools/si_features.py`. Endpointing is `src/vad.py` through `tools/vad.py` —
the device's own. Features are `mfcc.logmel_q8()`, an accessor added to
`tools/mfcc.py` that returns **the 26 Q8 log2 mel values the existing front end
already computes on its way to every cepstrum**, taken from `work[3]`
immediately before the DCT.

A convolutional net wants the filterbank, not the cepstrum: neighbouring mel
bins are correlated in exactly the way a kernel exploits, and the DCT exists to
destroy that correlation — right for DTW's L1 distance, wrong here.

**Nothing new has to be verified on hardware.** `python3 tools/mfcc.py
--selftest` checks the accessor two ways, both *verified*: applying the DCT to
what it returns reproduces the raw cepstra exactly, and it matches the `mel`
entry of all five cases in `src/speech_fixtures.py` — the array the viper port
is already required to reproduce bit for bit — **5 of 5**. The device side is a
tap on an array it is already filling.

### The one new stage: normalisation to int8

```
m[i]    = trunc_toward_zero(sum over t of x[t][i], n_frames)     per band
y[t][i] = clamp((x[t][i] - m[i] + (1 << (S-1))) >> S, -128, 127)  S = 4
```

Per-**band** mean subtraction, which is the log-mel analogue of the cepstral
mean normalisation the DTW path already does, and there for the same reason: a
fixed channel colouration is additive in the log domain, so subtracting each
band's own mean removes it exactly.

**Measured consequence, and it redirected the corpus effort.** If per-band mean
subtraction removes any fixed linear filter, then gain and spectral tilt
augmentation are a no-op for these features. Checked on one utterance
("computer", Samantha, 150 wpm), patch values spanning -128..97:

| transform | mean abs diff | max |
| --- | --- | --- |
| gain -6 dB | 0.68 LSB | 15 |
| gain +3 dB | 0.32 LSB | 9 |
| tilt +0.15 | 0.34 LSB | 8 |
| tilt -0.15 | 0.35 LSB | 14 |

A fifth of a decibel — quantisation and the endpointer landing a frame
differently, not the transform surviving. Two of `say_corpus.py`'s three channel
axes therefore buy this path nothing. What *does* survive is **additive noise**,
**reverberation**, **speaking rate**, and far above all of them **more voices**.
The corpus generator was redirected accordingly and now mixes real background
noise at controlled SNR.

### Sizes, chosen by measurement

`N_FRAMES = 80`, `N_BANDS = 26`, `INPUT_SHIFT = 4`.

Endpointed lengths over 880 `say` utterances of all 22 forms (*dry-run*): min
35, median 55, p95 79, max 130 frames.

| N_FRAMES | centre-cropped |
| --- | --- |
| 64 | 23.3% |
| **80** | **4.3%** |
| 96 | 1.7% |
| 112 | 0.8% |

80 is where the curve flattens; 96 buys 2.6 points for 20% more
multiply-accumulates, and what is still cropped at 80 is the old formant voices,
on which the endpointer runs long, not genuinely long words.

**This number is due a re-derivation and is probably a little too large.** The
probe included the singing voices, which the frozen roster drops, and covered
~11 distinct voices rather than the 20 names it appeared to, so the 130-frame
tail is largely gone.

Real speech does **not** pull hard the other way, and an earlier draft of this
file said it did. That claim rested on the pre-fix captures, where the
endpointer was truncating words it had barely found — 15-31 frames. Measured on
the ten post-fix takes, all of which endpoint, the range is **25-44 frames**,
which sits inside the synthetic distribution rather than a third of it. A
word-dependent duration is an endpointer working; the constant one is the
failure `docs/speech.md` warns about, and that is what the old numbers were.

So 80 frames is generous but not absurd, and the case for shrinking it is now
about compute rather than about padding drowning the signal. Re-measure on the
natural tier before re-picking — si-corpus notes the 8 natural voices build in
about 15 minutes, and that is the population this choice has to fit.

Note `docs/speech.md`'s "28..61 frames" is **one voice at 172-178 wpm**. Twenty
voices across a wider rate range are half again as long. Do not reuse that range
for anything else.

The shift trades resolution against clipping (150 utterances, *dry-run*):

| shift | 1 LSB | clipped |
| --- | --- | --- |
| 3 | 0.188 dB | 20.2% |
| **4** | **0.376 dB** | **2.3%** |
| 5 | 0.753 dB | 0.0% |

4, because what clips is a band more than 48 dB below its own mean over the
utterance — the noise floor inside a silent frame, not any part of the word — so
the clip acts as a spectral floor, which `docs/speech.md` measured neutral for
DTW. Shift 5 spends half the int8 range on a value nothing depends on.

## The model

`tools/si_train.py`. Two architectures, both shaped by what TinyMaix can run
rather than by taste, and both **inside every budget by a wide margin**:

| | params | MMAC | output elements | weights | `.tmdl` |
| --- | --- | --- | --- | --- | --- |
| `dscnn` w1.0 | 13142 | 1.106 | 45142 | 11.4 KB | 21.1 KB padded |
| `dscnn` w2.0 | 42134 | 3.359 | 90262 | 38.2 KB | ~52 KB padded |
| `plain` w1.0 | 41222 | 2.237 | **16662** | 39.8 KB | 41.5 KB |

Against DTW's measured 616-672 ms of matching, all of them are cheap. Size will
not decide this.

**Output element count is in that table because it, not MACs, tracks the
requantisation cost** both back ends pay per output element — a fixed charge
that MAC counts miss entirely. `plain` w1.0 has twice `dscnn` w1.0's MACs and a
third of its output elements, so the two orderings disagree.

**It does not flip this decision, and an earlier draft of this file said it
did.** Costed on the real counts (si-device's model, ~65 cycles per output
element against ~36 per MAC), requantisation is **18.9% of `dscnn` w1.0 on
TinyMaix and 6.9% on viper** — so it is largest on TinyMaix, not on the viper
fallback, and halving the MACs beats tripling the output elements at these
ratios. `dscnn` w1.0 wins on both back ends, on viper by nearly 2x
(~285 ms against `plain`'s ~544 ms, both *predicted*). **Build `dscnn`.**

Keep the column: at a different shape it does flip the answer (2.8x the output
elements for 0.72x the MACs, in si-device's earlier comparison). One unmeasured
caveat in `plain`'s favour — a depthwise 3x3 has a contiguous inner run of 3
where a dense conv has `kw*chi`, so depthwise amortises viper's loop overhead
worse than a flat cost-per-MAC assumes. That penalty would have to be about 2x
to close the gap.

Three training choices are about precision rather than accuracy:

- **`unknown` is a trained class**, not a threshold. A softmax over keywords
  alone cannot represent "none of these" and produces confident nonsense at
  exactly the moment it matters.
- **Label smoothing is off.** It improves accuracy and flattens the softmax, and
  a flattened softmax is what a confidence threshold cannot work with. The usual
  advice points the wrong way here.
- **`--unknown-weight`** multiplies the unknown class's training weight, trading
  recall for precision directly. Sweep it; do not assume it.

## Measurement

`tools/si_eval.py` reports three things separately and never averages them:
synthetic held-out **voices**, real speaker, and precision swept against a
threshold and a margin. The gates map onto DTW's:

| `tools/dtw.py` | here |
| --- | --- |
| best distance <= THRESHOLD | top-1 probability >= `thresh` |
| second-best - best >= MARGIN | top-1 - top-2 >= `margin` |
| (no equivalent) | top-1 class is not `unknown` |

`measure()` is deliberately the same counting rules as `dtw.measure`, including
the three-way truth — in-vocabulary, benign morphological variant, must stay
silent — so the two are comparable. The table is printed as *highest recall at
each precision* rather than one row per threshold: DTW's integer scores have
plateaus and collapse to a few dozen rows, softmax probabilities are all
distinct and print a thousand rows of noise.

### The validation set is two voices, and that caps what can be concluded

The frozen roster holds 43 installed English voices, but only **8 are
independent natural speakers** — the rest are 16 iOS voices that form a single
family and 13 MacinTalk formant/novelty voices. Holding out a fifth of the
*voices* would put a sheep noise in the validation set, so the split holds out
half the *natural tier* instead:

| split | voices |
| --- | --- |
| train | 33 (16 expressive, 13 novelty, 4 natural) |
| val | 2 — Daniel, Samantha |
| test | 2 — Karen, Tara |

**Treat val as n=2 and do not report architecture differences from it.** Two
voices cannot separate a one- or two-point difference between `dscnn` and
`plain`, or between widths, from noise. The honest uses of this corpus are the
large effects — does the approach work at all, does the DTW control beat the CNN
— and nothing finer. Every manifest row carries a `tier` tag, so a different
split can be tried without regenerating audio.

This is a ceiling on the *idea*, not on the corpus tooling: `say` does not
supply enough independent vocal tracts on this machine to validate a
speaker-independent model properly, and that is worth knowing before anyone
spends another session on it.

### The control that decides how to read a bad result

`tools/si_dtw_control.py` runs **the incumbent, unchanged, speaker-independently**
— templates from `say` voices in the training split, queries from held-out
voices. Without it, "the CNN did badly" has two explanations that look identical
and lead to opposite decisions: the model is too small, or synthetic voices do
not transfer at all.

**Measured, *dry-run*** — 66 templates from 3 voices, 300 queries from 3 unseen
voices:

| | top-1 | precision 1.000 at recall | 95% precision at recall |
| --- | --- | --- | --- |
| DTW, same speaker (`docs/speech.md`) | — | 0.966 | — |
| **DTW, held-out speakers** *(dry-run)* | **0.753** | **0.233** | 0.301 |
| CNN, 7 training voices, no noise *(dry-run)* | 0.490 | 0.076 | 0.121 |

Read the first two rows and stop. **Changing speaker costs DTW 73 points of
recall at zero false fires** — that is the size of the problem, measured rather
than assumed, and it is the bar the CNN has to clear. The third row is a
deliberately impoverished harness test and **is not a result**; the corpus it
used has 7 voices and no channel augmentation, against the real corpus's 23 and
a full noise model.

## Four traps, all of which fail silently

Grouped here because they are the same family as the entries in
`.serena/memories/gotchas_that_cost_time.md`: every one of them produces a
plausible number rather than an error.

### 1. `say -v` renders an uninstalled voice as the default, without failing

In a throwaway corpus, `Samantha`, `Fiona`, `Tom` and `Alex` had the **same
MD5**. Only Samantha was installed. Ten of twenty names were phantoms.

If phantoms land on both sides of a voice-disjoint split, the
speaker-independent score becomes a voice matched against itself and reads
**high**. Caught only because the DTW control returned distances of **exactly
0**, which real audio never produces.

`si_features.check_distinct_voices()` hashes every WAV and **raises**; it runs at
the top of `si_train.py`, `si_eval.py` and `si_dtw_control.py`.

### 2. The voice split leaked through locale variants

`roster.json`'s first split put every val voice's same-named counterpart in
train — Flo, Grandma, Grandpa, Rocko, Sandy and Shelley, US in val and UK in
train. Six of eight val voices, and **17 of 34 of the roster's own recorded
`close_pairs` straddled a split**. Those are one synthetic voice with a
different accent setting, not two speakers. Group by voice *name* before
grouping by acoustic fingerprint.

### 3. Batch norm's default momentum makes a model that is constant on held-out data

The first dry-run model reported **0.70 training accuracy and a flat 0.055 top
probability — 1/18 — on every held-out utterance**. Keras defaults
`BatchNormalization(momentum=0.99)`, which assumes thousands of steps; this
corpus gives a few tens per epoch, so the inference-time moving statistics were
still near their initial values when training stopped. Training metrics use
batch statistics and look fine throughout, so nothing points at batch norm.

`BN_MOMENTUM = 0.9` took keyword accuracy on held-out voices from **0.000 to
0.490** with no other change.

### 4. The int8 input contract holds by luck unless it is asserted

The device hands TinyMaix raw bytes, and the wrapper computes `quantised =
uint8 - 128` while **ignoring the model's own input scale and zero point**. So
the model's input quantisation must come out at exactly `scale 1.0, zero 0`. It
does — but only because the patches span the full int8 range, which is a
property of the *data*: a corpus that never clips would silently change the
scale, the device would feed numbers the model was not trained on, and the model
would load, run, and be quietly wrong.

`si_train.to_int8_tflite` now reads the quantisation back and **raises** unless
it is 1.0 / 0.

## The device path, verified on the host

Keras -> int8 `.tflite` -> `.tmdl`, all *verified* by running it. Findings that
constrain the model, from si-device's reading of the runtime and from
conversion:

- **No MAX_POOL_2D, no AVERAGE_POOL_2D, no REDUCE_MAX.** Only `MEAN`, as the
  global-average head. Every downsample is a strided convolution. A global *max*
  pool crashes the converter — reproduced.
- **The dense layer must be per-tensor.** TinyMaix's `tml_fc` reads `ws[0]` and
  applies it to every logit while the converter writes all 22 scales. It
  converts cleanly and returns wrong numbers for 21 of 22 classes. The
  converter flag exists on **TF 2.20 and not on TF 2.13**, so: **export the
  `.tflite` from the 2.20 venv, convert to `.tmdl` in the 2.13 one.**
- **`out_deq` must be 1**, or TinyMaix leaves the output int8 while emlearn
  reads it as float32.
- **The `.tmdl` must be padded, and this is now confirmed rather than
  suspected.** emlearn sizes TinyMaix's activation scratch as the model file's
  length, which is a guess, and `tm_load` does not check it against the model's
  own `buf_size`. `dscnn` w1.0 is **15776 bytes of file against a 21120-byte
  buffer** — it would write 5.2 KB past a heap allocation, silently, on a heap
  that also holds the ELIZA rules and the capture buffer. Depthwise-separable
  networks are structurally the shape that triggers it: few weights, large
  intermediate tensors. `tools/tmdl_info.py --pad` appends zeros until the file
  covers the buffer; the trailing bytes are never read. The model in `build/`
  is padded and `tmdl_info.py` exits 0 on it.
- **The module is `emlearn_cnn_int8`, not `tinymaix_cnn`.** It was renamed at
  emlearn-micropython 0.6.0 and split at 0.8.0; the old path is a 404. It is
  still the TinyMaix wrapper and is published prebuilt for `armv7emsp`.
- **Single-layer mode is not reachable.** `sub_size` is hardcoded to 0 in the
  converter, so the 3.6 KB figure TinyMaix's own tooling prints is what that
  mode *would* need, not what any `.tmdl` will ask for. Plan on
  `2 x max(file, buf_size)` resident, plus ~12 KB for the module. **Padding
  costs RAM as well as flash**, because both buffers are sized to the file
  length: `dscnn` w1.0 goes from 31552 B resident / 47328 B peak unpadded to
  **42240 B resident / 63360 B peak** padded. That is the trade, and it is
  still nothing against 489 KB contiguous.

**Two virtual environments are needed and both are `.venv*`-gitignored:**

```
uv venv --python 3.11 .venv     && uv pip install --python .venv/bin/python "tensorflow>=2.16,<2.21"
uv venv --python 3.10 .venv-tm  && uv pip install --python .venv-tm/bin/python "tensorflow-macos==2.13.0" "numpy<2" pillow
git clone --depth 1 https://github.com/sipeed/TinyMaix.git
touch TinyMaix/tools/__init__.py       # tflite2tmdl.py uses a relative import
cd TinyMaix && python -m tools.tflite2tmdl in.tflite out.tmdl int8 1 80,26,1 22
```

## What is still unknown

- **Whether any of this works on a human.** *unknown*, and it is the experiment.
  Ten clean post-fix takes now exist and all ten endpoint, but ten utterances of
  one speaker with no negatives cannot measure precision, which is the metric
  this design turns on.
- **Whether the 5.71x `IMX/IMN` requirement is the right design.** Derived here
  and still never tested against this microphone — every measurement that looked
  like a test was taken through the chirp bug. On clean captures it is not
  binding, so this is a question for a noisier room rather than a live problem.
- **Whether the remaining 2.2x feature contrast gap is real or is the
  clipping.** Halved by the chirp fix; the rest is unexplained.
- **Whether `emlearn_cnn_int8` imports on this board.** *unknown* — si-device
  owns it and had no board time. The prebuilt `.mpy` decodes to mpy 6.3 /
  `armv7emsp` / 31-bit small ints, which matches this board on all three, so the
  ABI question is settled in our favour; whether the ARM code links and computes
  correctly is not. A clean import is not proof either — a native module can
  import and still generate wrong code, so the probe classifies emlearn's ten
  shipped MNIST digits, and `build/kw_unknown_*.bin` are eight real patches with
  host predictions in their filenames to prove it on *this* topology.
- **Inference time on the device.** *unknown*. si-device predicts 25-50 ms on
  TinyMaix and 225-450 ms on a viper fallback, against DTW's measured 616-672 ms
  of matching.
- **Every accuracy number on the real corpus.** The corpus was still generating.
- **Whether TinyMaix's unclamped int8 layer outputs wrap on real device
  features.** Its `tm_postprocess_sum` has a bare C cast where TFLite clamps, so
  a representative set calibrated on clean synthetic features could let real
  ones wrap into confident nonsense. A host-side counter for this is not built.

## Resuming

0. **The endpointer is fixed — confirm it stays fixed, then move on.** The
   chirp-bleed bug is repaired and post-fix captures endpoint 10 of 10 with 4x
   to 9x margin. **Do not change `ITU`.** What is left from that investigation is
   the clipping: every capture still saturates, and saturation is absent from the
   training corpus, so it needs an augmentation axis rather than a knob.
1. **Get ~3 minutes of the user's voice.** Keywords through
   `tools/enrol.py takes/`, then negatives through
   `tools/enrol.py takes-oov/ --allow-any`: **other, another, wonder, mother's,
   brother's** first (the "other" -> FATHER collision at 727 sets the whole DTW
   threshold), then **no, know, want, need**, then conversational filler. Without
   negatives, real-speaker *precision* — the number that matters most — cannot
   be measured at all.
2. **Generate the corpus** — `python3 tools/train_corpus.py corpus-tts/ --jobs 7`,
   about 2 hours and resumable, or the 8 natural voices alone in ~15 minutes.
   Then `tools/si_features.py corpus-tts/manifest.json --cache <dir> --stats` for
   the length distribution and per-class counts. Note `endpoint_ms: null` rows
   are utterances the VAD could not find and are a real population, not an
   error.
3. `tools/si_dtw_control.py corpus-tts/manifest.json` — **the bar, first.**
4. `tools/si_train.py corpus-tts/manifest.json --arch dscnn` and again with
   `--arch plain`; sweep `--width` and `--unknown-weight`.
5. `tools/si_eval.py build/si.tflite corpus-tts/manifest.json --takes takes/`.
6. Convert, `tools/tmdl_info.py --pad`, hand to the device side.

**The decision rule, agreed in advance so it cannot be argued backwards:** if
the CNN does not beat the DTW control's precision-1.000 recall on *real* speech,
ship DTW with synthetic templates and stop. Given how forgiving ELIZA is of a
miss, a model that fires rarely and is never wrong is the target; one that is
merely more accurate on average is not.
