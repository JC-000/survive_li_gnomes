# Running a trained CNN on this board

What can actually execute on the RP2350, for the speaker-independent recogniser.
Companion to [speaker-independent.md](speaker-independent.md), which owns the
model and the corpus; this file owns the runtime.

> ## Measured on the board, 2026-08-18
>
> **`emlearn_cnn_int8` imports, loads and computes correctly on this board.**
> The central unknown is closed. Every figure below marked *measured* came off
> the hardware in that session; the predictions this document previously carried
> are kept next to them, because two of them were wrong by enough to matter.
>
> **Verified on hardware** (`/dev/cu.usbmodem1401`): the import; correct
> classification of emlearn's ten MNIST digits; inference time for that model
> and for `si_real`; RAM at import, peak and rest; coexistence with the capture
> buffer, rules, framebuffer and DTW templates; the viper fallback's throughput
> and its int32 bound at the algebraic worst case; the scratch-buffer overrun,
> with a control; that `bytearray` is rejected as model input; and that the
> device disagrees with host TFLite on 3 of 8 keyword patches.
>
> **Updated 2026-08-18 (second session):** the CNN is now wired into `talk.py`
> and a full turn has run on the glass. The operating point has been measured
> through the device path. See *The recogniser that ships* below.
>
> **Still analysis, not measured**: whether the divergence comes from the unclamped
> int8 outputs or from the requant rounding; the front end feeding the CNN on
> the device; depthwise cost in the viper fallback; and every figure for the
> `dscnn 2.0` / `plain` candidates, which were never built as `.tmdl` and are
> scaled from the measured 9 cycles/MAC.

## Already established — do not re-derive

Measured in earlier sessions. These cost real time to get right.

| | |
| --- | --- |
| MicroPython | **1.28.0** |
| Native arch | **`armv7emsp`**, mpy **6.3** |
| MCU | RP2350A, Cortex-M33, 150 MHz, FPv5 single-precision FPU + Armv8-M DSP |
| Clean-boot heap | **492 KB free, 489 KB largest contiguous** |
| Filesystem | **15,728,640 bytes** since 2026-08-18 (`tools/build_firmware.sh`; it was 3 MB, the stock `RPI_PICO2` build assuming a 4 MB part). The heap figures above were taken before that change and still hold: 493,040 bytes free on the new firmware |
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
saturate. Not statically checkable, and still open: it needs a host-side counter
of how often `|sumf*out_s_inv + out_zp| > 127` over a representative set.

This is one candidate explanation for the divergence measured below, though not
the only one, and it has not been isolated.

## A `.tmdl` is not numerically the `.tflite` it came from

**The most important thing this session measured, and it was not on anyone's
list.** TinyMaix is an independent reimplementation, not a TFLite runtime: it
requantises in float and casts, where TFLite uses fixed-point rounding
multipliers with saturation. Those are different arithmetic, and the outputs
differ.

Measured on emlearn's own MNIST model, host TFLite against this board:

| digit | host p | device p | delta |
| --- | --- | --- | --- |
| 0, 1, 3, 8 | 0.9961 | 0.9960 | -0.0001 |
| 2 | 0.9453 | 0.9610 | **+0.0157** |
| 4 | 0.8906 | 0.8440 | **-0.0466** |
| 5 | 0.9141 | 0.8870 | -0.0271 |
| 7 | 0.9727 | 0.9570 | -0.0157 |
| 9 | 0.9531 | 0.9410 | -0.0121 |

Argmax survives all ten, because MNIST's margins are enormous. **`si_real`'s are
not.** On the eight keyword patches, top-1 differed from the host TFLite on
**three of eight** — and the device was systematically more confident in the
`unknown` class:

| patch | host top1 / p | device top1 / p |
| --- | --- | --- |
| 0 | **18** / 0.5664 | **21** / 0.9492 |
| 6 | 21 / 0.5000 (margin 0.0000) | 21 / 0.6875 |
| 7 | **11** / 0.5078 | **21** / 0.6445 |

**Consequence: `si-model`'s operating point does not transfer.** Precision 1.000
at recall 0.500 was measured on the `.tflite`. The threshold and margin gates
that replace DTW's `THRESHOLD` and `MARGIN` must be tuned on **the `.tmdl`, on
the device** — or on a host build of TinyMaix, which does not currently exist in
this project. Evaluating on TFLite and shipping the `.tmdl` is measuring one
thing and deploying another.

The bias direction is at least the safe one for this toy: more `unknown` means
more deflections and fewer confident misfires, which is what `docs/speech.md`
argues to optimise for. But it is unmeasured, on eight patches from a model
whose corpus was still generating.

### Which comparisons survive this, and one that does not

Comparisons *within* the CNN — architecture against architecture, real speaker
against synthetic, voice against voice — are relative measures taken under the
same arithmetic on both sides, so a systematic bias largely cancels and the
ranking should hold.

**The CNN-against-DTW comparison is the exception, and it is the one the whole
project rests on.** Only one side of it is affected: DTW is unaffected by
TinyMaix's arithmetic, while the CNN's numbers move. On the three patches that
disagreed, two flipped from a keyword to `unknown` — the direction that costs
**recall**. So a CNN recall figure measured on the `.tflite` is an **upper
bound** on what the device will do, and setting it beside DTW's measured recall
flatters the CNN by an unknown margin.

That does not overturn the case for the CNN — 66.6 ms against 616-672 ms is not
a margin arithmetic drift can close, and speaker independence is the reason for
doing this at all. It means the *accuracy* half of the comparison has to be
re-measured under TinyMaix before it is quoted against DTW, and the *cost* half
already has been.

### The drift is noise, not a shift, and that is the worse case

Worth being precise about, because the two have different remedies. On the MNIST
control the deltas run **both ways** — `+0.0157` on digit 2, `-0.0466` on digit
4. So this is not a monotone bias that a retuned threshold would absorb; it is
scatter added to the scores.

A monotone shift moves the operating point along the precision/recall frontier
and re-tuning recovers it. **Scatter degrades the frontier itself**, because it
blurs the separation between the classes the threshold is meant to divide. So
re-tuning on the device is necessary but may not be sufficient: the achievable
precision/recall curve under TinyMaix can be genuinely worse than the one
measured under TFLite, not merely differently placed on it.

### What the eight patches do and do not establish

They establish the *direction* of the disagreement — three of eight changed
top-1, two of them from a keyword to `unknown`. They do **not** establish whether
that is more accurate or less, and it would be easy to read them as if they did.

The patches carry no ground truth. Their filenames encode what an **earlier**
model (`si_dscnn_w1`) predicted, not what was said — which is why the host
references here were recomputed rather than taken from the labels. If the true
label on those two was in fact `unknown`, the device was the more accurate of
the two. Nothing available says which.

So: "the device leans toward `unknown`" is measured. "The device has lower
recall" is an inference from that lean at a **fixed** threshold, and it is the
part that needs the host build to settle.

## Memory cost

    resident = 2 x max(file_size, buf_size)      model copy + activation scratch
    peak     = 3 x file_size during new()        the caller's array('B') is alive too
    plus     ~12 KB at import                    5.5 KB code + sbuf 2304 + sumscale 4000 + k_oft 100

**Measured** for `si_real.tmdl` (21120 B, padded), by `gc.mem_alloc()` deltas:

| | measured | formula |
| --- | --- | --- |
| module import | **12096 B** | ~12 KB predicted |
| caller's `array('B')` of the file | **43472 B** | 2.06x the file — see below |
| peak during `new()` | **64624 B** | 3.06x the file |
| resident after dropping the array | **43536 B** | 2.06x the file |

The 2x and 3x formulas are right. **Use `gc.mem_alloc()` and not
`gc.mem_free()`** — a `mem_free` delta read 85824 B resident for the same model,
because free-list fragmentation from building the array is not resident use.
That over-reported the cost by 97%.

The surprise is the caller's own array: `array.array('B', blob)` costs **2.06x**
the bytes it holds, because MicroPython grows the array by doubling. It cannot
be avoided — `bytearray` is **rejected** by the wrapper with
`ValueError: model should be bytes`, since `mp_get_buffer` does not report
typecode `'B'` for it. Verified on the board.

Total with the model resident: **~55.6 KB**. Nowhere near binding.

### Coexistence, measured

Everything held at once, on the board:

| held | free | largest contiguous |
| --- | --- | --- |
| nothing (clean boot) | 492672 | ~489 KB |
| + 94 KB capture buffer | 261840 | |
| + 37 KB ELIZA rules | 223936 | |
| + 5 KB framebuffer | 218912 | 165408 |
| + 137 KB DTW templates | 174112 | 101936 |

**Inference under that full load: 3218 us against 3205 us on an empty heap** —
no measurable penalty. The CNN, the capture buffer, the rules, the framebuffer
and the DTW fallback all fit together with ~170 KB spare.

## Inference cost — measured

| | MACs | measured | cycles/MAC |
| --- | --- | --- | --- |
| MNIST (emlearn's own) | 51228 | **3205 us** | 9.38 |
| **`si_real` dscnn w1.0** | 1106048 | **66604 us — 66.6 ms** | 9.03 |

So TinyMaix costs **~9 cycles/MAC**, against the ~3.5 predicted from reading
`tm_dot_prod`. The prediction counted the multiply-accumulate and not the
`sbuf` im2col gather that precedes every output pixel, nor the per-channel
`sumscale` setup. **Predictions of this kind were 2.6x optimistic; use 9.**

Even so the conclusion is unchanged and large: **66.6 ms against DTW's
616-672 ms of matching.** The CNN is about **10x cheaper than the matcher it
replaces**, and it is flat in class count where DTW is linear in template count.
A turn goes from ~889 ms to ~340 ms, and the front end becomes the dominant
cost.

Scaling the other candidates at 9 cycles/MAC: `dscnn` 2.0 ~202 ms, `plain` 1.0
~134 ms, `plain` 2.0 ~512 ms. All still inside break-even on TinyMaix.

### The viper fallback, measured

| | measured | predicted |
| --- | --- | --- |
| `uint8` dot product, plain loop | **41.0 cycles/MAC** | 40 |
| same, 4x unrolled | **31.2 cycles/MAC** | 32 |
| **whole conv layers, gather + MAC + requant** | **62.1 cycles/MAC** | 36 |

The isolated inner loop matched prediction almost exactly — the ~8-cycles-per-
viper-statement model this project already had is sound. The **whole-layer**
figure did not: 62.1 against 36, because the conv inner loop carries two extra
address additions per iteration that the isolated dot product does not, and
because the patch-sum pass and requant scaffolding cost more than allowed for.

At 62.1 cycles/MAC the fallback would run `si_real` in **458 ms** — still inside
DTW's 616 ms, but with a third of the headroom the earlier estimate suggested,
and `plain` 1.0 would land at ~926 ms and **miss break-even entirely**. If the
fallback is ever needed, `dscnn` is the only candidate that fits it.

**The int32 bound held exactly.** Section 4 drove the accumulator to its
algebraic worst case on the device: `peak |acc>>r|` came back 36864 and 49152,
matching the host algebra to the unit, so viper did not wrap where CPython could
not have shown it. Requant products peaked at 28.12% and 37.50% of int32,
against the 50.00% the Q14 design allows.

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

## The board session, as run

Executed 2026-08-18 on `/dev/cu.usbmodem1401`. **Result: sections 0-6 all ran;
`emlearn_cnn_int8` imports, loads and classifies correctly.** The board was left
on the Magic 8-Ball with every test file removed and its file list identical to
how it was found.

| # | question | result |
| --- | --- | --- |
| 0 | build identity | MicroPython 1.28.0, mpy 6.3, arch 7 = `armv7emsp`, 492480 B free — all as expected |
| 1 | **does it import** | **yes**, 12096 B of heap |
| 2 | does it load a model | yes, both emlearn's MNIST and `si_real` |
| 3 | **do the ten digits classify correctly** | **10/10**, 3205 us each |
| 3b | `si_real` on eight patches | runs, 66.6 ms each; **top-1 differs from host TFLite on 3 of 8** — see the divergence section |
| 4 | viper fallback + int32 bound | 62.1 cycles/MAC end to end; bound held exactly (36864, 49152) |
| 5 | coexistence | everything fits, no slowdown under load |
| 6 | scratch-buffer overrun canary | **confirmed: 1164 bytes overwritten unpadded, 0 padded** |

What remains unknown is listed at the end of this document.

### The sequence, for next time

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

Section 1 did not fail. **`armv7emsp` works, and emlearn-micropython's README
claims testing only on x64 and xtensawin — that is worth telling them**, since
this board is now a data point they do not have.

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


## The recogniser that ships

The CNN is `talk.py`'s recogniser. DTW is parked, not deleted: `spotter` is
still deployed, is still the front end the CNN taps, and is still the
specification the port is proved against -- `_spot_keyword` simply asks the CNN
first and falls through to DTW only if the native module misbehaves. On the
workshop build `templates.bin` is absent, so DTW has nothing to match and that
fall-through returns None, which is the intended shape rather than a gap.

### The path

    capture -> vad.endpoints -> si_patch (tap spotter's mel, normalise, fit)
            -> emlearn_cnn_int8 -> argmax + three gates -> vocab label -> ELIZA

`src/si_patch.py` is the device half of `tools/si_features.py`. It calls
`spotter._mfcc_at` whole and reads the 26 Q8 log2 mel values out of `work[3]`,
so the arithmetic below the tap is the arithmetic the DTW path already uses and
is already proved bit-exact against `src/speech_fixtures.py`. Its own additions
-- per-band mean subtraction, the `>> 4` to int8, centre crop/pad, and the
`+ 128` uint8 transport -- are pinned byte-identical against the host reference
by `tools/test_si_patch.py`, over real takes and over the crop/pad boundaries.

### The operating point, measured through this path

10 real takes of 9 keyword classes and 12 real out-of-vocabulary takes, all
endpointed by `src/vad.py` and scored on the board through `si_patch` plus
TinyMaix -- **not** through the `.tflite`.

| | |
| --- | --- |
| argmax correct on positives | 7/10 |
| argmax `unknown` on negatives | **12/12** |
| **precision** | **1.000** |
| **recall** | **0.600** |
| `THRESHOLD` | 0.35 |
| `MARGIN` | 2/256 |

**Re-measuring was not optional, and the placeholder proves it.** Before this
sweep `si_spot.py` carried 0.90 / 0.50, chosen to look conservative. Measured,
that pair fires on **nothing at all** -- recall 0.000. A plausible-looking guess
at a gate is a spotter that never speaks.

**The gates are very nearly inert, and that is the finding.** No negative ever
produced a keyword argmax and no positive was ever labelled as the *wrong*
keyword, so precision is 1.000 at any setting and the gates only cost recall.
The trained `unknown` class is doing all the rejection work, which is the
argument for having trained one rather than thresholding a 21-way softmax.

They are non-zero anyway for one measured reason and one principled one. The
`father` take came back correct with a margin of **exactly 0.0000** -- top-1 and
top-2 equal, a coin flip that landed right. `docs/speech.md` is explicit that an
utterance resembling two things equally should produce silence whatever its
absolute score, and that argument does not depend on which way the coin came
down. `MARGIN = 2/256` is two output quantisation steps, the smallest gate that
means anything, and it costs exactly one take (recall 0.700 -> 0.600).
`THRESHOLD = 0.35` sits below every correct fire in the set, so it costs nothing
measurable and is prudence rather than evidence.

**What this is not** is an accuracy figure. Ten positives means recall moves in
steps of 0.1, and precision 1.000 rests on twelve negatives never firing. It is
enough to ship a toy whose failure mode is a deflection. The honest next step is
more negatives, not more tuning.

One encouraging detail: the words that most attacked DTW all reject cleanly
here. `other` -> unknown at 0.879, `mothers` -> unknown at 0.984, `brothers`,
`know`, `want`, `need` all unknown. `docs/speech.md` records "other" -> FATHER
at 727 as the most dangerous false fire in the DTW corpus.

### A full turn, on the glass

Driven from real recordings through the shipping path -- `_spot_keyword`, the
ELIZA session, and the panel:

| input | heard | reply | panel |
| --- | --- | --- | --- |
| `mother_01` | **mother** | *"Tell me more about your family"* | 1727 ms, full |
| `father_01` | none (tie rejected) | *"I am not sure I understand you fully"* | 624 ms, partial |
| `other_01` | none | *"Please go on"* | 609 ms, partial |

The panel figures are the point of that last column. `docs/hardware.md` measures
a real full refresh at 1715 ms and a partial at 612 ms, and an unpowered panel
returns in about zero while every SPI write still succeeds -- so those wall
times, with BUSY asserted, are what says the glass actually changed rather than
that the code completed.

### Turn budget, measured

| stage | 36-frame take |
| --- | --- |
| front end (`spotter._mfcc_at` x 36, plus its discarded DCT) | 224.2 ms |
| normalisation and packing (`si_patch.normalise_into`) | 23.4 ms |
| inference | 66.6 ms |
| **recognition total** | **317 ms** |
| panel refresh (partial) | 624 ms |
| **turn** | **~940 ms** |

The front end dominates recognition at 71%, inference is 21%, and the panel
dominates the turn. Nothing here is a candidate for optimisation before the
e-paper is.

## What is still unknown

Closed this session: the import, the arithmetic, RAM, inference time, and the
scratch-buffer overrun.

Still open, in the order they matter:

1. **Negatives.** Precision 1.000 rests on twelve out-of-vocabulary takes. That
   is the thinnest part of the whole result, and the gates cannot be tuned
   against a false fire that has never been observed. More negatives is worth
   more than anything else on this list.
2. **A live human voice.** Every turn measured has been driven from a recording
   through the microphone-onward path. Nobody has pressed the screen and spoken
   to it. The capture path is exercised by enrolment and the recogniser by these
   takes, but not the two joined at the actual microphone.
3. **Recall is 0.600 on ten takes**, so three of nine classes were missed and
   the granularity is 0.1. Whether that is the model, the operating point, or
   this speaker is not separable at this sample size.
4. **The unclamped-output counter**, item 4 above. It is one candidate
   explanation for the divergence and has not been isolated from the others.
5. **Whether the divergence is TinyMaix-generic or model-specific.** Ruling on
   this needs a host build of TinyMaix to compare against, which would also give
   `si-model` somewhere to tune the gates without holding the board.
6. **Depthwise cost in viper**, if the fallback is ever needed — a depthwise 3x3
   has a 3-element contiguous inner run and will do worse than the 62.1
   cycles/MAC measured on dense convolutions.
