# Running a trained CNN on this board

What can actually execute on the RP2350, for the speaker-independent recogniser.
Companion to [speaker-independent.md](speaker-independent.md), which owns the
model and the corpus; this file owns the runtime.

> ## Nothing in this document has run on the board
>
> Not one line. The hardware was in use for the whole session and was never
> handed over, so **every figure here is either read out of source code or
> predicted from a cost model.** Where something is measured, it was measured on
> the host or is quoted from an earlier session and labelled as such.
>
> That matters more than usual, because the central claim — that
> `emlearn_cnn_int8` loads and runs on `armv7emsp` — is precisely the sort of
> thing that source reading cannot settle. Treat "Ready to run" below as a plan,
> not a result.

## Already established — do not re-derive

Measured in earlier sessions. These cost real time to get right.

| | |
| --- | --- |
| MicroPython | **1.28.0** |
| Native arch | **`armv7emsp`**, mpy **6.3** |
| MCU | RP2350A, Cortex-M33, 150 MHz, FPv5 single-precision FPU + Armv8-M DSP |
| Clean-boot heap | **492 KB free, 489 KB largest contiguous** |
| Filesystem | 3 MB (the stock `RPI_PICO2` build assumes a 4 MB part) |
| Networking | **none** — RP2350A, not a -W. It cannot `mip install` anything itself |
| `emlearn_fft` | **not installed** |
| viper speedup | **12.8x** on VAD frame stats, **35.6x** on the 512-point FFT, both bit-identical to the portable path |
| DTW cost today | ~**616-672 ms** of matching against 66 templates, plus ~273 ms of front end |

An earlier probe reported the capture buffer "would not fit". That was the probe
fragmenting its own heap. **Budget against 489 KB contiguous.**

## The module is not called `tinymaix_cnn`

It was renamed twice upstream and the old name now 404s:

| release | module |
| --- | --- |
| 0.4.0, 0.5.0 | `tinymaix_cnn.mpy` |
| 0.6.0, 0.7.0 | `emlearn_cnn.mpy` |
| **0.8.0 → 0.11.1 (`latest`)** | **`emlearn_cnn_int8.mpy`** + `emlearn_cnn_fp32.mpy` |

```
https://emlearn.github.io/emlearn-micropython/builds/latest/armv7emsp_6.3/emlearn_cnn_int8.mpy
```

5470 bytes, `sha256 60492a0bf6fb618edb2fb7ea5a2d55e49b5ad3a0ed67682cd07e96914a002012`,
byte-identical to the `0.11.1` pin. Its header decodes to **mpy 6.3, arch
`armv7emsp`, 31-bit small ints**, matching this board on all three. `latest` and
`master` are different directories and `master` accumulates stale files —
`tinymaix_cnn.mpy` survives only there, at unknown vintage. Pin `latest` or a
version, never `master`.

The old name is what made this look like the highest-risk unknown. It is not
missing; it is renamed.

**The board cannot fetch it.** Download on the Mac and `cp`. (`mpremote mip
install <url>` also works — mip resolves on the host and copies over — but a
plain `cp` of a file already on disk is one less thing to be wrong about.)

## What TinyMaix supports

Seven operators. `tflite2tmdl.py` asserts on anything else:

    CONV_2D, DEPTHWISE_CONV_2D, MEAN (= GlobalAveragePooling2D),
    FULLY_CONNECTED, SOFTMAX, RESHAPE, ADD

- **No MaxPooling2D, no AveragePooling2D.** Downsample with strided
  convolutions. Both candidate architectures originally used max pooling and
  both had to be rebuilt.
- **No dilation** — returns `TM_ERR_TODO`.
- **ReLU / ReLU6 only.**
- **BatchNorm is fine**, TFLite folds it into the preceding convolution.
- **Avoid `ADD`.** Upstream's own comment: *"it is dirty implement for ADD, only
  suit for simple resnet like network, no error warning for other network."*
- **One output, 1-D.** The wrapper raises otherwise.
- Layout is **NHWC**: `TM_MATP(mat,y,x,ch) = data + ((y*w + x)*c + ch)`.
- The prebuilt module is `TM_ARCH_CPU` / `TM_OPT0` — the **generic C path, not
  the ARM SIMD one**. The M33's DSP instructions are left on the table.

### Hard limits, from `tm_port.h`

| limit | value | on breach |
| --- | --- | --- |
| `kh * kw` | ≤ 25 | checked — `TM_ERR_KSIZE`, clean |
| `kh * kw * chi` (conv), `cho * kh * kw` (dwconv) | ≤ 2304 | **unchecked — overruns `sbuf`** |
| output channels | ≤ 1000 | **unchecked — overruns `sumscale`** |

## Four ways this fails silently

All four are checked by `tools/tmdl_info.py`, which exits non-zero. Run it on
every `.tmdl` before anything is copied to the board.

**1. The activation scratch is sized by guesswork.** `mod_cnn_new()` allocates
`data_buffer` with the *file length* and hands it to `tm_load()`, which never
compares it against the model's own `buf_size`. A model with `buf_size >
file_size` writes past a heap allocation with nothing raised. The model reads
back what it wrote, so its own answers stay self-consistent — the damage is
elsewhere, and it is allocation-order dependent, so a test can pass repeatedly
and the program still fail. Fix: `tools/tmdl_info.py --pad`. Full write-up in
`.serena/memories/gotchas_that_cost_time.md`.

**2. Dense layers must be per-tensor.** `tml_fc()` requantises with `ws[0]` —
the *first* channel's scale — applied to every output, while the converter
writes the full per-channel array. Conv and depthwise-conv use `ws[c]` properly;
dense does not. A per-channel dense layer converts cleanly and returns wrong
logits.

**3. `out_deq` must be 1.** With 0 on an int8 model, TinyMaix leaves the output
quantised while emlearn's `run()` reads `out.dataf[i]` as float32
unconditionally — class scores come back as reinterpreted bytes.

**4. TinyMaix never clamps its int8 layer outputs.** `tm_postprocess_sum` ends
with a bare C cast where TFLite's reference saturates to [-128, 127]. So
quantisation ranges calibrated on the wrong distribution **wrap** rather than
saturate. This one is *not* statically checkable and remains open: it needs a
host-side counter of how often `|sumf*out_s_inv + out_zp| > 127` over a
representative set. It matters here because the representative set is clean
synthetic features and the device will feed it something else.

## Memory cost

    resident = 2 x max(file_size, buf_size)      model copy + activation scratch
    peak     = 3 x file_size during new()        the caller's array('B') is alive too
    plus     ~12 KB at import                    5.5 KB code + sbuf 2304 + sumscale 4000 + k_oft 100

For the delivered `si_dscnn_w1.tmdl` (21120 B, already padded): **42240 B
resident, 63360 B peak.** Against 489 KB contiguous, with the 94 KB capture
buffer, ~37 KB of ELIZA rules and the 5 KB framebuffer, memory is nowhere near
binding — even with the 137 KB DTW template set still resident as the fallback.

(Note: 31552 / 47328 are the figures for the *unpadded* 15776-byte file. Padding
is what makes it safe, and padding is what doubles into the resident cost.)

## Predicted inference cost

**Predicted, not measured.** TinyMaix: ~3.5 cycles/MAC for the generic C
`tm_dot_prod`, plus ~20 cycles per output element for the float requant. Viper:
~36 cycles/MAC and ~65 cycles/output, anchored on this project's own two
measurements, which both work out at ~8 cycles per viper statement — the
512-point FFT at 218 cycles/butterfly over ~26 statements, and the DTW band cell
at 7.53 us over 144.

| model | MMAC | output elems | TinyMaix | viper |
| --- | --- | --- | --- | --- |
| **dscnn 1.0** (delivered) | 1.106 | 45142 | **~32 ms** | **~285 ms** |
| dscnn 2.0 | 3.359 | 90262 | ~90 ms | ~845 ms |
| plain 1.0 | 2.237 | 16662 | ~54 ms | ~544 ms |
| plain 2.0 | 8.528 | 33302 | ~203 ms | ~2061 ms |

Break-even against DTW is ~600 ms; ~200 ms is a clear win. **On TinyMaix every
candidate wins comfortably.** On the viper fallback, `dscnn 1.0` at ~285 ms is
the only one that wins clearly.

**Output-element count is real but secondary at these ratios.** It was worth
adding to the cost model — it is a fixed per-element charge that MAC counts miss
— but it turns out to be 19% of the total at worst (dscnn 1.0 on TinyMaix) and
under 7% on viper, where the per-MAC cost is high enough to dominate everything.
So `dscnn` wins on both back ends despite having 2.7x the output elements of
`plain`, because it has half the MACs. Pick on MACs here; keep the
output-element column so the next architecture is judged on both.

The remaining viper caveat is shape, not count: a depthwise 3x3 has a contiguous
inner run of `kw*chi = 3` where a dense conv has `kw*chi`, so depthwise MACs
amortise loop overhead worse than the flat 36 cycles assumes. That penalty is
unmeasured.

## The fallback, if the module does not load

Hand-rolled int8 convolution in `@micropython.viper`.

`ptr8` loads are zero-extended, exactly as `ptr16` loads are, so sign-extending
in the innermost loop in the program would be the wrong trade. Activations and
weights are stored **biased +128 as uint8** and the bias is unwound
algebraically — the same manoeuvre the DTW inner loop already makes with its
+32768 templates:

    sum (Wu-128)(Xu-128) = sum Wu*Xu - 128*sum Xu - 128*sum W

`sum W` over the *signed* weights folds into the bias at build time. There is no
`128*128*n` term: writing the expansion in terms of the stored uint8 values
makes it look like there should be one, but `sum Wu = sum W + 128*n` already
carries a `-128*128*n` that cancels it. Carrying both leaves every accumulator
wrong by 1179648 for a 3x24x1 filter, without crashing.

**The accumulator is not the tight stage.** Each term is bounded by 16384, so
`|acc| <= 16384 * kh*kw*chi`, which at TinyMaix's own widest (2304) is 1.76% of
int32. The tight stage is the requantisation: a Q15 multiplier gives
`65535 * 32767 = 2147385345`, **100.00% of int32**. So the accumulator is
pre-shifted into 16 bits and the multiplier is **Q14**:
`65535 * 16383 = 1073659905`, **50.00% of int32** — the same factor of two the
FFT twiddle keeps, and the same remedy `docs/speech.md` names for that stage.

`tools/test_conv_int8.py` proves this on the host over eight shapes and four
input patterns including the algebraic worst case (both factors at -128, which
reaches 16384 where the obvious +127 choice reaches 16129 and is 1.6% short).
It found the bias-fold bug above.

The host cannot prove the *port*, because viper wraps at int32 silently while
CPython and MicroPython bytecode both carry unbounded ints. Section 4 of the
probe drives the same worst case on the device for that reason.

## Ready to run: the ordered board session

One pass, cheapest and most decisive first, each step gating the next.
`tools/cnn_probe.py` is that sequence. It touches no peripheral — no panel, no
codec, no I2C — so it is safe against a board mid-way through anything else.

```sh
STAGE=<staging dir>              # see "Staging" below
P=/dev/cu.usbmodem101            # confirm: one report said usbmodem1101
uvx mpremote connect $P cp $STAGE/emlearn_cnn_int8.mpy :
uvx mpremote connect $P cp -r $STAGE/cnn :
uvx mpremote connect $P run tools/cnn_probe.py
```

| # | question | why it is in this position |
| --- | --- | --- |
| **0** | mpy version and native arch, read off the device | one `exec`; invalidates everything below if it disagrees |
| **1** | **does `emlearn_cnn_int8` import**, and what does it cost the heap | the highest-value unknown; gates the architecture choice |
| **2** | does it load emlearn's own MNIST model | known-good upstream, so a failure is the port's fault and nothing else's |
| **3** | **do the ten shipped digits classify correctly** | a clean import is *not* the test — native code can load and generate wrong code for this core with nothing raised |
| **3b** | the same two questions for `si_dscnn_w1.tmdl`, with its eight patches | an architecture inside the stated limits can still reach a path nobody exercises; failure here is a wrong answer, not a crash |
| **4** | viper fallback: dot product, three whole layers, and the int32 bound at its algebraic worst case | runs whether or not 1 succeeded — the two numbers are only useful side by side |
| **5** | coexistence with the 94 KB capture buffer, ELIZA rules, framebuffer and the 137 KB templates | sections 1-3 run on an empty heap, which is not the heap the program has |
| **6** | canary test for the scratch-buffer overrun | **opt-in**, needs a deliberately unpadded `.tmdl`; corrupts memory on purpose — reset the board afterwards |

Section 3b compares against `si-model`'s host TFLite predictions, which ride in
the filenames (`kw_<class>_<n>.bin`) so no schema is shared between agents and a
renamed file is obviously wrong rather than quietly mismatched. Two of the eight
are deliberately marginal (top-1 probability 0.125, margins 0.027 and 0.039) —
those are where an arithmetic fault shows first.

**If section 1 fails, that is a result, not a dead end**, and section 4 gives the
number that says how large a model the fallback can carry. Report it upstream
either way; it is worth an issue on emlearn-micropython, whose README claims
testing on x64 and xtensawin only.

### Staging

Everything except the two lines below is already fetched. If the scratch
directory is gone, re-fetch:

```sh
curl -O https://emlearn.github.io/emlearn-micropython/builds/latest/armv7emsp_6.3/emlearn_cnn_int8.mpy
B=https://raw.githubusercontent.com/emlearn/emlearn-micropython/master/examples/mnist_cnn
curl -o cnn/mnist_cnn_int8.tmdl $B/mnist_cnn_int8.tmdl
for i in 0 1 2 3 4 5 6 7 8 9; do curl -o cnn/mnist_$i.bin $B/data/mnist_example_$i.bin; done
cp build/si_dscnn_w1.tmdl cnn/model.tmdl
cp build/kw_unknown_*.bin cnn/
```

`build/` is gitignored, so the model and patches do not survive a clean
checkout. Regenerating them is one `si_train.py` run plus `tflite2tmdl.py`; the
recipe is in [speaker-independent.md](speaker-independent.md).

## Host-side conversion, verified by si-model

Keras → int8 `.tflite` → `.tmdl`. Two venvs, and the split is not optional:

- **Export the `.tflite` from TF 2.20**, because
  `_experimental_disable_per_channel_quantization_for_dense_layers` exists there
  and not on 2.13. A 2.13-produced tflite yields a per-channel dense and 21
  wrong logits.
- **Convert to `.tmdl` under Python 3.10 + `tensorflow-macos==2.13.0`**, because
  `tflite2tmdl.py` imports `tflite_reader.py`, which uses `tf.lite.Interpreter`,
  `tensorflow.lite.python.schema_py_generated`, `tensorflow.python.keras` and
  `keras.preprocessing.image` — the last two are gone in modern TF and Keras 3.
  (`tflite2tmdl.py` itself imports `tensorflow` and `PIL` and uses neither; the
  dependency is entirely in the reader.)
- `tflite2tmdl.py` does `from .tflite_reader import read_tflite`, so it must be
  run as a package module, not as a script.
- **Single-layer mode is unreachable**: `sub_size` is hardcoded to 0 in
  `pack_tmdl`. The wrapper would cope with a non-zero value (`tm_load` mallocs
  it), so this is a converter limitation, not an emlearn one.

## The input and output contract

- **Input**: `array.array('B')`, uint8, length `h*w*c`, NHWC. The wrapper
  computes `quantised = uint8 - 128` and **ignores the model's own `in_s`/`in_zp`
  on that path**. `si_dscnn_w1` quantises the input at scale 1.0 / zero point 0,
  so the device sends `uint8 = int8_feature + 128` and nothing else.
  `si_train.to_int8_tflite` now raises unless the converter lands on exactly
  that, so it is safe by construction rather than by luck.
- **Output**: `array.array('f')`, **float32**, one per class. `out_deq = 1` means
  TinyMaix dequantises the softmax before emlearn copies it out. The softmax is
  computed inside the model; the device only needs `argmax`.
- **Shape here**: 80 frames x 26 mel bands x 1, 22 classes (21 keywords +
  unknown). Class order is in `si_dscnn_w1.json`.

## Rejection is still unsolved, and it is the important one

`docs/speech.md` argues the case at length: the DTW matcher fires only if the
best score clears an absolute `THRESHOLD` **and** beats the runner-up by a
`MARGIN`, and the measured operating point is precision **1.000**, recall 0.966.
Precision is pinned at 1.000 because a false fire answers "morning" with "DO YOU
OFTEN THINK OF MONEY" and the illusion dies, whereas a miss produces a
deflection nobody can distinguish from intended behaviour.

**A softmax argmax always fires.** The 22nd `unknown` class is the right start,
but softmax probabilities are famously overconfident and a bare probability
threshold is unlikely to be enough. Whatever replaces `THRESHOLD` and `MARGIN`
has to be tuned the way `tools/dtw.py --tune` tunes them, and reported as
precision and recall **separately** — a single accuracy figure averages together
the one error that matters and the one that does not.

Nothing about this has been measured. It is the largest open question after the
import.
