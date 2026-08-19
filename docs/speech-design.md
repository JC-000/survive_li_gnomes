# Speech input — design notes

Giving the ball an ear: the user speaks, ELIZA answers. This document records
*why* the speech path is built the way it is. The feature format, frame sizes and
wire details live in [speech.md](speech.md); this is the reasoning behind them.

The governing constraint is the same one as everywhere else, plus a new one:
**everything runs on the device.** No network, no host at runtime. That leaves
~490 KB of RAM, 3 MB of filesystem and a 150 MHz M33 to do it in.

## Nothing in this document has been measured on this board

That needs saying at the top, because every other document in `docs/` is a record
of things that were actually driven, and this one mostly is not. It began as
research, done 2026-08-17, before any speech code existed. As the host pipeline
comes up, findings are being folded back in — but those are measured on a Mac
against recorded audio, which is not the same as measured on the board.

Four labels are used throughout and they are not interchangeable:

| Label | Means |
| --- | --- |
| *measured* | Measured in this project — on the **host** unless it says otherwise |
| *published* | Someone else measured it, on hardware named in the citation |
| *estimate* | Arithmetic from a *published* anchor, which is always named |
| *unknown* | Not established either way — listed under [Still open](#still-open) |

`CLAUDE.md` exists because "the code ran" was once mistaken for "the panel
updated". The equivalent mistake here is reading an *estimate* as a budget. Run
`tools/speech_probe.py` and replace them.

## Transcription is not an option

Someone will eventually propose running Whisper. The gap is not a tuning problem
or a quantisation problem — it is two to three orders of magnitude.

Against our ~490 KB of free RAM and 3 MB filesystem — every row *published*
except the one marked:

| | Model file | Runtime RAM | Over our RAM by |
| --- | --- | --- | --- |
| whisper tiny / tiny.en (f16) | 75 MiB | ~273 MB | **557×** |
| whisper tiny, optimistic 4× quant *(estimate)* | ~19 MB | ~68 MB | ~140× |
| Moonshine tiny (27 M params, int8) | 28.2 MB | ≥28 MB | ≥58× |
| Vosk small-en-us-0.15 | 40 MB | ~300 MB | **612×** |
| Coqui / DeepSpeech `.tflite` EN | 47 MB | hundreds of MB | ~100× |
| sherpa-onnx streaming zipformer int8 | 67 MB encoder | tens of MB | >100× |

The quantised row is deliberately generous to the idea — a straight 4× scaling of
the f16 figures, better than quantisation actually achieves — and it still lands
140× out.

Sources: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) memory table;
[Vosk models](https://alphacephei.com/vosk/models) ("requires about 300Mb of
memory in runtime"); [Moonshine](https://huggingface.co/UsefulSensors/moonshine);
[DeepSpeech 0.9.3](https://deepspeech.readthedocs.io/en/r0.9/USING.html);
[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html).

Size is not even the binding constraint. Each of these needs a runtime — ggml,
ONNX Runtime, Kaldi, full TFLite — that has no microcontroller port at all, and
assumes an operating system, an allocator and a memory-mappable filesystem.
There is no quantisation trick, no streaming trick and no second-core trick that
closes a 500× gap.

Kaldi is not a separate option: Vosk *is* Kaldi, with the same figures.

**So the ball does not transcribe. It spots keywords.** That turns out to suit
ELIZA far better than transcription would, for reasons in the next two sections.

## Why template matching rather than a trained classifier

Two approaches fit in the RAM budget. Both were costed; the decision was not
close, and it was not decided on accuracy.

### The comparison

A trained classifier means a small depthwise-separable CNN over log-mel features
— the shape of TFLM's `micro_speech`, scaled up from its two words. The reference
numbers come from [Hello Edge (arXiv:1711.07128)](https://arxiv.org/abs/1711.07128),
which is the canonical sizing paper for this: DS-CNN-S is **38.6 KB, 5.4 MOps,
94.4 % accuracy on 12 classes**, running **~12 ms per inference** on a Cortex-M7
at 216 MHz with CMSIS-NN (all *published*).

Template matching means DTW over MFCC frames against recordings of the user's
own voice.

Both columns are quoted at the same **25 words**, so they can be read against
each other:

| | Trained CNN | DTW templates |
| --- | --- | --- |
| Model / templates | ~39 KB *(estimate)* | 47 KB *(estimate)*, in flash |
| RAM at inference | ~80–100 KB *(estimate)* | **<5 KB** *(estimate)* |
| Inference | 60–150 ms *(estimate)* | 50–200 ms *(estimate)* |
| **Recordings per word** | **50–100+** | **3** |
| Accuracy, one speaker, quiet | 70–85 % *(estimate)* | 80–90 % *(estimate)* |
| Time to a working demo | weeks | days |

The CNN's model size is *estimated* by growing DS-CNN-S's final layer from 64→12
to 64→25, which adds ~0.8 KB of int8 weights; at 50 words it would be ~41 KB.
Its RAM is *estimated* from the largest activation (49 × 10 × 64 int8 ≈ 31 KB,
double-buffered) plus 1960 B of features. Its inference time scales Hello Edge's
12 ms figure for a slower clock and for TinyMaix's generic C in place of
CMSIS-NN — the softest number in the table. DTW's are derived in
[The arithmetic](#the-arithmetic) below.

Note how little separates them on anything except the bolded row. Neither is
close to a RAM limit, and both answer inside the time the panel takes to refresh.

### What actually decided it

Read the recordings-per-word row, not the accuracy row.

[Edge Impulse's own guidance](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting)
is at least 50 samples per keyword, 100 better, 1000+ best, plus ~50 recorded in
silence for the noise class, and 12–25 minutes of varied collection. At 25 words
that is roughly half an hour of one person saying the same words into a
microphone, and it is a *floor*, not a target. Three recordings per word is about
two minutes.

The usual escape from that cost is to start from a public dataset. It does not
work here. Google Speech Commands v0.02 is
[105,829 utterances of 35 words from 2,618 speakers](https://service.tib.eu/ldmservice/en/dataset/bibtex/google-speech-commands-dataset-version-ii),
and the 35 words are:

> yes, no, up, down, left, right, on, off, stop, go, zero–nine, bed, bird, cat,
> dog, happy, house, marvin, sheila, tree, wow, backward, forward, follow, learn,
> visual

Against an ELIZA vocabulary — mother, father, dream, sorry, remember, always,
computer — the overlap is **yes and no. Two words out of ~25.** So essentially
100 % of the training data would be recorded by hand either way. The dataset
buys nothing except its `_background_noise_` folder, which is worth having for
tuning rejection whichever path is taken.

### Why few-shot learning does not rescue the CNN

It is the obvious objection, and the published result is genuinely good:
[Few-Shot Keyword Spotting in Any Language](https://arxiv.org/abs/2104.01454)
reports **87.4 % streaming accuracy at a 4.3 % false-accept rate across 440
keywords in 22 languages, from five examples per keyword** (*published*).

It does not help, because it works by fine-tuning a small head on a large
multilingual embedding model — and **that embedding has to run at inference time
too.** It does not fit in 490 KB, and dropping it drops the result with it.

DTW is the few-shot method whose inference-time cost is a few kilobytes. That is
the whole argument.

### The arithmetic

Worth showing, so it can be checked rather than believed. Assumes 16 kHz, 40 ms
frames on a 20 ms stride, a ~1 s utterance, 13 MFCC coefficients, 25 words × 3
templates:

| | |
| --- | --- |
| Frames per utterance | 1000 ms ÷ 20 ms = **49** |
| Template storage | 25 × 3 × 49 × 13 bytes int8 = **47 KB** |
| Query features in RAM | 49 × 13 = 637 B |
| DTW working set | two rows of 49 int32 = 392 B |
| Cells, full matrix | 49 × 49 = 2401, × 13 ops = 31.2 k per template |
| Cells, Sakoe-Chiba band ±10 | ~49 × 21 = 1029, × 13 ops = **13.4 k** per template |
| Total, 75 templates, banded | **~1.0 M integer ops** |

Templates live in flash as frozen `bytes`, which MicroPython memory-maps on rp2,
so they cost **no RAM at all** — only the one being compared needs to be walked.
Growing to 50 words costs 96 KB of flash and nothing in RAM.

The 1.0 M op count is firm. Turning it into milliseconds is not: it needs a
viper throughput figure this project does not have. See
[Still open](#still-open).

## Precision over recall

This is the load-bearing idea of the whole design, and it is easy to miss because
it inverts the usual target.

ELIZA is unusually forgiving of a recogniser that hears nothing, and unusually
unforgiving of one that mishears. When DOCTOR finds no keyword it emits a
deflection — "Please go on", "What does that suggest to you?" — and **that is a
real DOCTOR response, not a failure mode.** It is what the program does with
genuinely keyword-free input. A user cannot tell a missed word from a sentence
that happened to contain nothing interesting.

A *wrong* match has no such cover. Answering "Tell me more about your family"
when the user said "computer" is not a lesser version of working; it breaks the
illusion in one turn, and the illusion is the entire product.

So the recogniser should be shy. Set the DTW rejection threshold high and let
most utterances fall through to a deflection:

| | Feels like |
| --- | --- |
| 40 % recall, 95 % precision | a therapist who is listening but reticent |
| 85 % recall, 75 % precision | a broken toy |

The consequence for tuning is concrete: **measure how often it fires wrongly, not
how often it fires.** A build that recognises less but never misfires is the
better build, and the usual accuracy metric will rank it last.

It also changes the vocabulary ceiling. A general keyword spotter on this
hardware would honestly stop at 15–25 words. Because the uncertain tail costs
nothing here, ~25 words at a high threshold is a better bet than 15 tuned for
recall.

## What the vocabulary buys, and what it costs

The instinct is to spend the vocabulary on ELIZA's *trigger* keywords. The
ELIZA-side research says otherwise, and it reshaped the word list.

Of DOCTOR's 215 reassembly templates:

| | Templates | Needs |
| --- | --- | --- |
| Survive a bag of keywords as-is | **115** | the keyword alone |
| Need the user's own words back | 100 | verbatim input we do not have |
| — of those, rescuable by one spotted noun | 23 | one echoable content word |

So **138 of 215 templates (64 %) are reachable** from keyword spotting alone, and
the 23 rescuable ones are unlocked not by recognising more *triggers* but by
recognising more *nouns the response can echo*. The vocabulary budget therefore
goes on echoable content words — mother, father, dream, computer — rather than on
function words.

That also happens to be the phonetically easier choice. Function words (I, you,
my) are short, unstressed and reduced in running speech, which makes them the
worst possible DTW targets; content words are longer and more distinct. The two
arguments point the same way, which is rare enough to be worth noting.

**The honest consequence: this keeps the oracle and loses the therapist.** The
100 templates that need verbatim input are the ones that echo your own words back
at you, and those are the moments people quote when they talk about ELIZA. What
remains is a machine that recognises what you are talking about and responds in
character — much closer to the ball it is bolted to than to Weizenbaum's program.
That is a deliberate trade, not an oversight.

One pair to watch: **mother and father differ only in the initial consonant.**
For 13 MFCC coefficients against three templates that is a genuinely hard
discrimination, and they are two of the highest-value words in the script.
Expect them to be the first confusion the threshold has to suppress.

## The front end

### 16 kHz, not the 24 kHz the microphone was verified at

`docs/hardware.md` records the microphone verified at **24 kHz**. Speech capture
should reconfigure the codec to **16 kHz** instead. The ES8311 supports 8–48 kHz
in single-speed mode ([user guide, hosted by
Waveshare](https://files.waveshare.com/wiki/common/ES8311.user.Guide.pdf)), so
this is a register change, not a resampler.

Three reasons:

1. Every published model, dataset and benchmark in this document is 16 kHz.
   Staying at 24 kHz means either retraining the front end or resampling on every
   utterance.
2. It deletes a 3:2 polyphase resampler from the MicroPython budget — the most
   expensive stage we would otherwise be paying for and getting nothing from.
3. Buffers shrink by a third: 1 s of mono int16 is **32 KB** rather than 48 KB,
   2 s is 64 KB rather than 96 KB. Against a ~490 KB heap that already holds
   ~213 KB of audio clips, that is the difference between comfortable and not.

Nothing is lost. Speech energy above 8 kHz does not contribute to keyword
discrimination.

**Verify the rate change electrically.** Count samples actually delivered against
wall-clock, or scope BCLK. Per `CLAUDE.md`, a codec that ignored the register
write will look exactly like one that accepted it — `dma_record_into` will fill
the buffer either way.

### Fixed point, not float — viper does nothing for floats

The front end must be integer arithmetic end to end. This is the single most
important implementation constraint in the document, and it is not obvious.

MicroPython's emitters, measured on ESP32 by
[luvsheth](https://luvsheth.com/p/making-micropython-computations-run) (*published*):

| | Time | vs bytecode |
| --- | --- | --- |
| bytecode | 487 ms | 1× |
| `@micropython.native` | 203.8 ms | 2.4× |
| `@micropython.viper` | **30.03 ms** | **16×** |
| C module | 4.48 ms | 109× |

That 16× is the only thing making a MicroPython front end viable. But
[the MicroPython docs](https://docs.micropython.org/en/latest/reference/speed_python.html)
are explicit that viper's speed comes from *integer arithmetic and bit
manipulation*, where it is "almost as fast as assembler". Viper works in machine
words and pointers. **It does essentially nothing for float math.**

So a float mel filterbank in viper is not a faster float mel filterbank — it is
the same slow one. Q15/Q31 fixed point throughout, or write C.

There is an irony worth recording: the RP2350's M33 does have a hardware
single-precision FPU (FPv5) and DSP instructions, unlike the RP2040. We cannot
reach it from MicroPython, so the chip's headline advantage over its predecessor
is unavailable on the path we are taking, and only appears if the front end ever
moves to C.

### What the FFT costs

The one hard timing anchor on this exact silicon:
[micropython-fourier](https://github.com/peterhinch/micropython-fourier) does a
**1024-point single-precision FFT in 6.97 ms on a Pico 2 (RP2350)** (*published*),
in inline ARM assembler using the FPU.

Scaling by N log N gives **~3.1 ms for 512-point, ~1.4 ms for 256-point**
(*estimate*). At 49 frames that is ~150 ms per second of audio, which fits.

Treat that as a conservative floor. 6.97 ms at 150 MHz is about a million cycles,
which is slow for a hardware FPU — CMSIS-DSP-class code on a comparable part is
roughly an order of magnitude quicker. The routine is a general
MicroPython-callable, not a tuned one.

`emlearn_fft` should remove the need to hand-roll this at all. Its native modules
are published prebuilt for `armv7emsp`, and MicroPython tags **RP2350 as
`armv7emsp`** ([discussion #16538](https://github.com/orgs/micropython/discussions/16538)),
so it should install with `mip` — no firmware rebuild, no flashing. *Unverified
on this board;* see [Still open](#still-open).

What emlearn does **not** provide is the rest of the front end. Verified from its
[module listing](https://github.com/emlearn/emlearn-micropython/tree/master/src):
FFT, IIR, arrayutils, trees, KNN, k-means, and a TinyMaix CNN wrapper — and **no
mel-spectrogram, no MFCC, no audio module.** Its only audio example is a sound
level meter. The underlying C library does have
[`eml_audio.h`](https://raw.githubusercontent.com/emlearn/emlearn/master/emlearn/eml_audio.h)
with `eml_audio_melspectrogram`, but it is not bound to MicroPython. So the mel
filterbank, log and DCT are ours to write in viper regardless.

### Endpointing

Use Rabiner & Sambur (1975),
["An Algorithm for Determining the Endpoints of Isolated Utterances"](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1975.tb02840.x)
— short-time energy plus zero-crossing rate, with three thresholds: ITU (upper
energy), ITL (lower energy) and IZCT (zero-crossing rate). It is *published* as
working at signal-to-noise ratios of about 30 dB or better, which a person
speaking at a device on a desk comfortably clears.

**The zero-crossing pass is not optional here.** Energy alone finds the voiced
part of a word and clips the unvoiced onset — the /f/ of *father*, the /s/ of
*sorry*. Both are words we specifically need, and truncating their first
consonant is exactly what makes *father* collide with *mother*.

Shape:

- 10 ms frames. Energy as `sum(abs(x))` rather than sum of squares — no overflow
  to reason about and it stays in viper's integer domain.
- Calibrate the noise floor from the first ~100 ms after the press, before the
  user has started speaking. The touch press already gives us that moment for
  free.
- Start on energy above ITU, backtrack to the ITL crossing, then extend backwards
  by the zero-crossing rule. End after 300–500 ms continuously below ITL.
- **Hard-cap the utterance at 2 s (64 KB).** Without a cap a cough, a passing
  truck or a stuck threshold consumes the heap.

### IZCT: do not take the paper's clamp literally

The one place the 1975 algorithm is actively wrong on digitised audio, and it
cost a debugging session before anyone spotted it.

R&S set `IZCT = min(25, zmean + 2*zstd)`. That **minimum** assumes a background
whose zero-crossing rate is *low* — true of a 1975 analog lead-in, false of
anything sampled. A broadband noise floor changes sign on nearly every other
sample, so the background crosses *more* often than speech does, not less.
*Measured* on the host corpus:

| | Crossings per 10 ms frame |
| --- | --- |
| Silence (room noise floor) | 75–90 |
| Voiced vowel | 3–14 |
| Frication — the /s/ of *sorry* | 105–126 (background 73–92) |

Taking the minimum therefore pins IZCT far *below* the background, every silent
frame reads as frication, and the back-off walks the endpoint out to the full
lookback in both directions. **The fix is to invert the clamp** — treat the fixed
25 as a lower bound and let the adaptive term rise above it.

A crossing-rate test alone still is not enough, because frication and a noise
floor are both broadband. What separates them is that frication is *audible*, so
a ZCR hit must also clear the background energy (2× measures ~12σ of a silent
frame's variation, while the /s/ of *sorry* runs 2.5–6.6×).

The symptom is worth memorising, because nothing errors:

| | Endpointed length |
| --- | --- |
| Before the fix | 930–1090 ms — **every word, regardless of which** |
| After | 430–590 ms, with *computer* correctly the longest |

An endpointer that returns the full buffer every time looks like a working
endpointer. The tell is that the durations do not vary with the word.

## Traps

Two failure modes that will not announce themselves.

### Templates are recorded on the host, not on the device

The obvious design has the user enrol words by speaking to the ball. **That
requires writing to flash, which `CLAUDE.md` forbids outright** — and for a good
reason that applies here unchanged: a flash write interrupted by a power cut is
the one thing that can corrupt the filesystem, and this device loses power
without warning by design.

So enrolment happens on the host. The user speaks into the Mac, a host script
computes the MFCC templates, and `./tools/deploy.sh` ships them as a frozen
module alongside the code. The user still records their own voice in their own
acoustic conditions; it just goes through the laptop. The rule survives intact
and the speaker-dependent accuracy is unaffected.

The tempting alternative — an "enrol" mode that writes once and never runs during
normal operation — reintroduces precisely the risk the rule exists to prevent, in
exchange for saving a cable. Don't.

### The VAD will trigger on the ball's own audio

The board plays a shake, a fart or sampled laughter through **the same ES8311**
the microphone hangs off. An energy-based endpointer pointed at that will hear
it and treat it as speech.

**Gate the VAD across playback**, from `shaker.start()` to `shaker.finish()` plus
a margin for the room's decay.

This one is nasty in the same way the unpowered-panel bug was. `shake.py`
swallows its own exceptions by design — correctly, because a silent ball still
works — so a self-trigger produces no traceback and no log line. The symptom is
ELIZA answering a question nobody asked, intermittently, with nothing anywhere to
explain it.

## Still open

Everything below is *estimate* or *unknown* and should be replaced by a
measurement from `tools/speech_probe.py`. Listed with the anchor each was derived
from, so it is clear what is being replaced.

- **Viper throughput on this chip — `unknown`, and it blocks the DTW timing.**
  The op counts in [The arithmetic](#the-arithmetic) are firm (~1.0 M integer ops
  for 25 words, banded); converting them to milliseconds needs an operations-per-
  second figure that no source gives credibly. The 16× emitter ratio is solid;
  the *absolute* rate behind it cannot be recovered from the published benchmark.
  A viper loop of known op count, timed on the board, settles this and half the
  numbers in this document with it.
- **`emlearn_fft` on this board — `unknown`.** The architecture match
  (`armv7emsp`) is sound and third parties report success on RP2350, but
  emlearn's own README claims testing only on x64 and ESP32. If it does not load,
  the fallback is a hand-rolled radix-2 int32 FFT in viper — roughly 150 lines,
  well-trodden. This is the cheapest experiment in the list and the one that
  unblocks the most.
- **Per-frame front-end cost — `estimate`, from the 6.97 ms micropython-fourier
  FFT.** The FFT rows are extrapolated from real Pico 2 hardware and are the more
  trustworthy half; the mel filterbank, log and DCT rows are op-count arithmetic
  with no hardware behind them at all. **No measured MicroPython MFCC benchmark
  on RP2040 or RP2350 appears to exist** — this is a gap in the literature, not
  in the search.
- **DTW accuracy at 25 words — `estimate`, from a literature band of 59.7–93.3 %**
  across
  [several](https://www.researchgate.net/publication/269303996_Isolated_Words_Recognition_Using_a_Low_Cost_Microcontroller)
  [studies](https://www.researchgate.net/publication/346140785_Speech_Recognition_Implementation_Using_MFCC_and_DTW_Algorithm_for_Home_Automation).
  The most comparable one ran on a dsPIC30F4013 and only ever did **7 words**, at
  ~5 s per response. The 80–90 % figure quoted above is a judgement call inside
  that band and is **the number most likely to be wrong.**
- **CNN inference time — `estimate`, from Hello Edge's 12 ms on a Cortex-M7 with
  CMSIS-NN.** Scaled for clock and for TinyMaix's generic C. TinyMaix publishes
  no Cortex-M benchmark table. Only matters if the DTW path is abandoned.
- **The 16 kHz codec reconfiguration — `verified`.** The ES8311 datasheet says
  8–48 kHz, and this board, which had only ever been driven at 24 kHz, measured
  **15990 Hz** at the 16 kHz setting — −0.1 %, counted against wall clock rather
  than inferred from the absence of an exception. See `docs/hardware.md`.
- **Rejection threshold — `unknown` and unknowable in advance.** It is the one
  number that has to come from a person listening to the thing, and per
  [Precision over recall](#precision-over-recall) it should be tuned against
  false fires rather than hit rate.
