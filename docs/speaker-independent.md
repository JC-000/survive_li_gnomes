# Speaker-independent keyword spotting

The shipped recogniser is speaker-**dependent**: DTW against three templates
per word, recorded in one person's voice by `tools/enrol.py`. It is cheap,
needs no training, and only works for the person who enrolled.

On 2026-08-18 the user asked for a **speaker-independent** recogniser instead —
one anybody can talk to, with no enrolment. This file records what that costs,
what was measured, and what is still unknown.

**Status: unfinished.** The session paused before any result. Everything below
is either a measurement on synthetic data or an explicit gap. Nothing here has
been validated against a real human speaker.

## The cost of speaker independence, measured

The number that frames the whole problem, from `tools/si_dtw_control.py` —
DTW unchanged, but with templates from `say` voices and queries from *held-out*
voices:

| | recall at precision 1.000 |
| --- | --- |
| DTW, same speaker (`docs/speech.md`) | **0.966** |
| DTW, held-out speakers | **0.233** |

**Changing speaker costs 73 points of recall at zero false fires.** That is the
size of the problem, and it applies to any approach, not just DTW.

Read that before any model result. A speaker-independent recogniser that reaches
half the shipped recall is doing well, not badly.

## Why synthetic voices

Training data is the whole difficulty. Google Speech Commands has 105,829
utterances from 2,618 speakers and overlaps our vocabulary by **two words**
(YES, NO) — no MOTHER, FATHER, DREAM, SAD. Recording real humans costs 2-3 hours
each and we would want ten or more.

macOS `say` ships ~184 voices. The hypothesis is that training on those plus
augmentation generalises to a real human. **That hypothesis is untested** — see
the gaps below.

## The device path

Verified on the host, not on the board: Keras -> int8 TFLite -> TinyMaix `.tmdl`
converts cleanly.

| | |
| --- | --- |
| Flash | 15.4 KB |
| TinyMaix RAM | 20.6 KB (3.6 KB single-layer mode) |
| Parameters | 13,142 |
| MACs | 1.11 M |

Size will not decide this. Two constraints came from the converter rather than
from choice: it has **no `MAX_POOL_2D` and no `REDUCE_MAX`**, so every downsample
is a strided convolution and the head is a global average pool.

**The int8 input contract is exact and is asserted, not assumed.** The converter
chose input scale 1.0 and zero point 0, so the device hands TinyMaix the raw
int8 log-mel patch with no rescaling — but only because the patches span the
full int8 range. A future corpus that never clips would silently change the
scale, and the model would load, run, and be quietly wrong.
`si_train.to_int8_tflite` raises if the scale is not exactly 1.0 / 0.

The front end is **not** re-implemented. `mfcc.logmel_q8()` returns the Q8
log-mel the existing pipeline already computes on its way to every cepstrum, so
there is no second front end needing its own bit-exactness proof against the
device. `mfcc.py --selftest` checks that applying the DCT to what it returns
reproduces the raw cepstra exactly.

## Three traps found while building this

**`say -v <name>` does not fail on an uninstalled voice — it silently renders in
the system default.** Four supposedly distinct speakers shared an MD5. If such
phantoms land on both sides of a voice-disjoint split, the speaker-independent
score becomes a voice matched against itself and reads *high*. Caught only
because a DTW control returned distances of exactly zero.
`si_features.check_distinct_voices()` now hashes every WAV and raises.

**The voice split leaked.** macOS ships same-named US and UK variants — Flo,
Grandma, Grandpa, Rocko, Sandy, Shelley — and the original `roster.json` put 17
of 34 close pairs on opposite sides. Training on UK Flo and testing on US Flo is
not speaker independence.

**Batch normalisation defaults defeat a small corpus.** Keras defaults its
moving-average momentum to 0.99, which assumes thousands of steps; this corpus
gives a few tens per epoch, so the inference-time statistics were still near
their initial values when training stopped. The symptom is a model reporting 0.70
*training* accuracy while emitting a constant near-uniform softmax on every
held-out utterance — training metrics use batch statistics and look fine
throughout. Momentum 0.9 took held-out keyword accuracy from **0.000 to 0.490**
with no other change.

## What is NOT known

Listed plainly because each one could change the answer.

- **No real-speaker test set exists.** The user did not enrol. Every number here
  is synthetic-trained and synthetic-tested. A model that recognises `say` voices
  while failing on people would look identical to success in the current harness.
  **This is the single biggest gap and the first thing to fix.**
- **`tinymaix_cnn` has never been imported on this board.** emlearn-micropython
  publishes prebuilt `.mpy` for `armv7emsp`, which matches, but the project only
  claims testing on x64 and xtensawin. If it does not load, the fallback is
  hand-rolled int8 convolution in viper, which favours a simpler topology — so
  this gates the architecture choice.
- **A possible heap-corruption issue with the DS-CNN at width 1.0** was raised by
  the device agent and never written up before the session ended. Unproven,
  possibly inferred rather than observed. Treat as a warning to re-derive, not a
  fact.
- **No inference timing on hardware.** Op counts suggest a CNN's fixed cost may
  beat DTW, which scales with template count and already dominates the front end
  8:1 — but that is arithmetic, not measurement.
- The CNN dry-run figures (top-1 0.490, recall 0.076 at precision 1.000) come
  from a deliberately impoverished 7-voice corpus with no channel augmentation,
  built to prove the harness. **They are not results and must not be quoted as
  such.** Note only that DTW-SI currently leads the CNN; if that survives the
  real corpus, the recommendation is "ship DTW with synthetic templates".

## Resuming

1. **Get ~3 minutes of real speech from a human.** A handful of keywords plus
   the negatives list, via `tools/enrol.py takes-oov/ --allow-any`. Without a
   held-out human there is no verdict, only a self-consistent synthetic loop.
   Negatives matter as much as keywords: without them the only measurable false
   fire is one keyword mistaken for another, which is the rarer failure — the one
   that ruins the toy is the ball answering confidently when someone said
   something ordinary.
2. **Settle `tinymaix_cnn` on the board.** Highest-value unknown, gates the
   architecture. `tools/cnn_probe.py` exists for this.
3. Finish the corpus with the regenerated, leak-free voice split.
4. Compare against `tools/si_dtw_control.py`'s 0.233, not against the shipped
   0.966. Different question.

Known board facts, so they are not re-derived: MicroPython 1.28.0, arch
`armv7emsp`, mpy 6.3, clean-boot heap 492 KB free / 489 KB largest block,
`emlearn_fft` not installed and no networking to fetch it, viper measured at
12.8-35.6x and bit-identical to the portable path.
