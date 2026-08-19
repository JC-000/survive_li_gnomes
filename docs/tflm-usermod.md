# TensorFlow Lite Micro as a user C module — scoping verdict

Can the board execute `si_real.tflite` itself, with TFLite's own arithmetic,
instead of TinyMaix's approximation of it? Companion to
[cnn-on-device.md](cnn-on-device.md), which owns the TinyMaix runtime and
records the divergence this document exists to remove.

> ## Verdict: **GO**, on one measurement
>
> **Host TFLM and host TFLite produce bit-identical int8 output tensors on all
> 30 cases** — the 8 divergence patches plus all 22 real-speaker takes. Not
> "close", not "same argmax": the same 22 bytes, every time. TinyMaix manages
> 5 of 8 on top-1 alone.
>
> That was the gate, and everything else came in cheaper than feared:
>
> | | measured | against |
> | --- | --- | --- |
> | bit-identical output tensors | **30 / 30** | TinyMaix: 3 of 8 patches change top-1 |
> | flash, **measured in a real rp2 image** | **+79,444 B** stripped, +127,768 B with diagnostics | 710,192 B was free in the 1 MB reserve; **630,748 B still is** |
> | static RAM (`.bss`), same image | **+8 B** stripped, +392 B with diagnostics | — |
> | tensor arena, 64-bit host | **28,664 B** | TinyMaix path: ~55.6 KB resident |
> | inference time on the board | **not measured** | TinyMaix: 66.6 ms |
>
> **The one thing not measured is the one that needs the board**: nothing here
> has *run* on an RP2350. What exists is a **complete rp2 firmware image
> containing the module**, built through `tools/build_firmware.sh` against
> fw-16mb's board definition and pinned toolchain — `build/mpy-16mb-tflm.uf2`,
> 911,872 bytes, and a stripped variant at 815,616. It has not been flashed.
> The module is separately proved to load a model and classify, correctly and
> bit-exactly, under the **unix** port of MicroPython v1.28.0.
>
> The 16 MB image currently flashed on the hardware does **not** contain this
> module, and nothing here changed it.
>
> **And one existing figure is not what it says it is.** The recorded `.tflite`
> operating point — threshold 0.598, precision 1.000 at recall **0.500** — was
> measured through a default `tf.lite.Interpreter`, which installs the XNNPACK
> delegate. Under the reference kernels, which are what TFLM computes and
> therefore what the board will compute, the same 22 takes and the same scorer
> give **0.637 at recall 0.300**. Top-1 is 0.700 under all three host runtimes:
> the model is unchanged, and what moved is where a threshold can sit. The
> retune everyone expected to be a no-op is required. See *What the morning
> confirms* below.

## Why this was worth asking

`docs/cnn-on-device.md` measures it: TinyMaix is an independent reimplementation
of TFLite's int8 arithmetic, not a TFLite runtime. It requantises in float and
casts where TFLite uses fixed-point multipliers with defined rounding and
saturation. The consequences are all recorded there and none of them are
theoretical — top-1 differs from host TFLite on 3 of 8 patches, the operating
point had to be re-derived on the device rather than transferred from the
`.tflite`, and four separate silent-failure modes are guarded on the host by
`tools/tmdl_info.py` because the runtime will not report them.

The proposal was: compile TFLM into the firmware, run the `.tflite` directly,
and the divergence stops existing rather than being managed. That is only worth
doing if TFLM really does compute what TFLite computes. This document is
mostly the answer to that.

## 1. Does a maintained MicroPython TFLM usermod exist?

**No.** Surveyed 2026-08-18.

| candidate | last real commit | MicroPython | rp2 | licence | usable |
| --- | --- | --- | --- | --- | --- |
| [mocleiri/tensorflow-micropython-examples](https://github.com/mocleiri/tensorflow-micropython-examples) | **2025-01-26** | pins **v1.19.1** | RP2040 only | MIT | no |
| [openmv/libtflm](https://github.com/openmv/libtflm) | 2026-04-03 | n/a — not a usermod | no M33 lib | Apache-2.0 (it is TFLM) | as a reference |
| `mocktix/micropython-tflite` | — | — | — | — | **does not exist** (404) |
| emlearn-micropython | 2026-07-30 | current | yes | MIT | **not TFLM** — it is TinyMaix, which is what we have |

Details worth carrying, because each cost a lookup:

- **mocleiri's is the closest thing to prior art, and it has drifted twice
  over.** Its last real commit is January 2025, nineteen months ago, and the
  MicroPython its submodule pins — `9b48634` — is **v1.19.1, June 2022**, four
  years old and nine minor releases behind ours. Its rp2 CI builds
  seven boards, all RP2040, and `prepare-tflm-rp2.sh` generates the TFLM tree
  with `TARGET_ARCH=cortex-m0`. It is a genuinely useful *reference*: its
  `micropython.cmake` is a working C++-on-rp2 USER_C_MODULES file and several
  decisions here were taken from it. It is not something to depend on.
- **OpenMV embeds TFLM in a MicroPython fork**, and `openmv/libtflm` publishes
  prebuilt static libraries from upstream. The libraries are cortex-m0plus,
  m4+fp, m7+fp, m55 and m55+u55 — **no m33**, and the m4+fp library is
  hardfloat where the rp2 port is softfp, so it could not be linked even
  setting the architecture aside. What is worth taking from OpenMV is the shape
  of the C facade, which is what `openmv-libtf.cpp` is and what mocleiri lifted.
- MicroPython's own [discussion #18273](https://github.com/orgs/micropython/discussions/18273)
  (Oct–Nov 2025) ends with a collaborator saying "it would be great to have a
  TFLM module that one could just include as a USER_C_MODULE in a typical
  build". That is the state of the art: wanted, not written.

**So: we would own a port.** Costed below, and the cost turned out to be small,
because TFLM is designed to be dropped into a foreign build system and
MicroPython officially supports C++ user modules (`examples/usercmodule/cppexample`,
and `docs/develop/cmodules.rst` documents `SRC_USERMOD_CXX` / `CXXFLAGS_USERMOD`).
What we own is under 800 lines: a C facade over the C++ interpreter (312), a
MicroPython binding (269), and two build files (198).

## 2. The bit-exactness proof

This is the whole case, so it is the part with the most measurement behind it.

    ./tools/fetch_tflm.sh
    ./tools/build_tflm_host.sh
    .venv/bin/python tools/tflm_vs_tflite.py --model build/si_real.tflite \
        --classes build/si_real.json --bins 'build/kw_unknown_*.bin' \
        --takes takes takes-oov

`tools/build_tflm_host.sh` compiles TFLM's reference kernels **and
`firmware/usermod/tflm/tflm_shim.cpp`** — the same file the firmware compiles —
into a host shared library. `tools/tflm_vs_tflite.py` drives it through ctypes
and compares it against `tf.lite.Interpreter` on the same inputs.

**The comparison is on the raw int8 output tensor**, not on dequantised floats.
Comparing floats would hide a one-count difference behind print precision, and
one count of 1/256 is exactly the size of `si_spot.MARGIN`.

    bit-identical int8 output tensors : 30 / 30
    top-1 agreement                   : 30 / 30
    worst single-count difference     : 0

The 30 are the 8 `build/kw_unknown_*.bin` patches — the ones on which the device
and the host disagreed — and the 22 real takes, featurised through
`tools/si_features.py`, which is the same front end `src/si_patch.py` is pinned
against.

### Which TFLite, and why this matters beyond TFLM

`tf.lite.Interpreter` is three different runtimes depending on how it is
constructed, and **they do not agree with one another**:

| `--resolver` | what runs | bit-identical to TFLM | top-1 agreement |
| --- | --- | --- | --- |
| **`ref`** | TFLite's reference kernels | **30 / 30** | 30 / 30 |
| `nodelegate` | TFLite's optimised CPU kernels | 10 / 30 (max 8 counts) | 30 / 30 |
| `default` | the same, plus the XNNPACK delegate | 6 / 30 (max 8 counts) | 30 / 30 |

None of that is a defect: optimised int8 kernels are permitted to reassociate,
and on this model none of the reassociation reaches the argmax. But two things
follow, and the second is the uncomfortable one.

**First, `--resolver ref` is the only setting under which "host equals device"
is a true statement.** Any evaluation of a model destined for this board has to
use it. `tools/tflm_vs_tflite.py` defaults to it for that reason.

**Second, "host TFLite" in `docs/cnn-on-device.md` and
`docs/speaker-independent.md` means the *default* interpreter**, since that is
what a plain `tf.lite.Interpreter(model_path=...)` gives. So the host references
those documents compare TinyMaix against carry up to 8 counts of runtime
variation of their own. That does not overturn the TinyMaix finding — a max
delta of 8 counts is small beside three changed top-1 decisions — but it does
mean the *size* of the TinyMaix divergence was measured against a moving
reference, and any re-measurement should pin `BUILTIN_REF`.

### Does bit-exactness carry from this host to the M33?

**Not measured, and it cannot be from here** — running M33 code needs the board
or an emulator, and neither was available. What can be said is where the risk
is and is not:

- **The int8 eval paths are integer end to end.** `reference/integer_ops/conv.h`
  contains no `float` or `double` at all; depthwise and fully-connected use
  float only in their hybrid (float-activation) variants, which an int8-in /
  int8-out model never reaches; softmax's int8 path is gemmlowp fixed point.
  Integer arithmetic is identical on any conforming C++ implementation.
- **The quantisation multipliers are computed in `double` at
  `AllocateTensors()`.** The M33 has a single-precision FPU, so those doubles
  are software-emulated — but IEEE-754 double is IEEE-754 double, correctly
  rounded either way, so they land on the same `int32` multiplier and shift.
- **The one way to break this is a compiler flag.** `-ffast-math`,
  `-ffp-contract=fast` or `-Ofast` at that stage would be enough. **fw-16mb
  checked the generated build and none of the three appears anywhere** in the
  port, the SDK or the firmware target; the C++ flags are
  `-Os -fno-exceptions -fno-unwind-tables -fno-rtti -fno-use-cxa-atexit`,
  linked with `--gc-sections`. The usermod adds none of them either, and both
  build files carry a comment saying why.
- `docs/speaker-independent.md` records that the host TinyMaix build landed
  within 2–8 of 256 quantisation levels of the device's, "likely genuine
  ARM-versus-host float rounding in the requantisation". **That is precisely the
  failure mode TFLM does not have**, because TinyMaix requantises in float per
  output element and TFLM does not requantise in float at all.

So: high confidence, unmeasured, and **the first thing to check on the board is
that these 30 cases reproduce byte for byte.** If they do not, this document is
wrong and the reason will be a build flag.

### CMSIS-NN: faster, and not proved equal

The optimised ARM kernels are the obvious next thought on a 150 MHz M33, and
this build deliberately does not use them.

**TFLM's own kernel tests compare CMSIS-NN against the reference goldens with
`1.0 /* tolerance */`** — one quantisation count, in `conv_test.cc` and its
siblings. That is a statement that they agree to within a count, not that they
are equal. One count of 1/256 is the whole of `si_spot.MARGIN`, and
`docs/speaker-independent.md` records three live answers decided by exact ties.

Inference is 21% of recognition and recognition is a third of a turn
(`docs/cnn-on-device.md`); the panel refresh is 624 ms. **There is nothing here
that speed buys.** Reference kernels, and the cmake says why in a comment
beside the glob that would otherwise pick up `kernels/cmsis_nn/`.

## 3. Footprint

### Flash, measured twice — and the standalone estimate was 21% low

**Take these numbers, because they are the real image.** Two firmwares built
through `tools/build_firmware.sh` with the board's own definition and pinned
compiler, sized against the plain image that is on the board:

| image | `.text` | delta | `.bss` | delta | free in the 1 MB reserve |
| --- | --- | --- | --- | --- | --- |
| plain 16 MB (flashed, verified) | 338,384 | — | 5,052 | — | 710,192 |
| **+ `tflm`, stripped** | **417,828** | **+79,444** | 5,060 | **+8** | **630,748** |
| + `tflm`, diagnostics on | 466,152 | +127,768 | 5,444 | +392 | 582,424 |

**`-ffile-prefix-map` is not optional, and it failed silently once.** TFLM's
assertion macros bake `__FILE__` into the image: 19 absolute host paths in the
diagnostics-on build. The wasted flash is the minor half — the major half is
that the same sources from a different checkout directory produce a different
binary, which voids any byte-for-byte reproducibility claim about the firmware.
(fw-16mb found it in a string diff of the two images.)

The trap in the fix is worth more than the fix. `-ffile-prefix-map` matches
`__FILE__` as a **literal string prefix**, and `__FILE__` carries the path cmake
handed the compiler via `-I`, which cmake normalises. The default `TFLM_DIR` was
`.../usermod/tflm/../../../vendor/tflm`, so the flag appeared on every compile
line, matched nothing, and the paths stayed in the image — a build log that says
the fix is applied and a binary that says it is not. `get_filename_component(...
ABSOLUTE)` first. Worth 944 bytes, and the reproducibility.

**And `strings firmware.elf | grep vendor/tflm` is necessary but not
sufficient** — it says the paths are gone, not that the *binary* stopped
depending on where the tree sits, and those are different claims. The check
that separates them is to stage the whole tree in a different directory, build
it there, and compare: fw-16mb did that and got a **byte-identical image**,
same `e7b1069a…`. That is the check the first attempt would have failed, and
it is the one to run when this is ever touched again. The grep is 0 where it
was 19, and on its own it would have been 0 for the wrong reason.

The stripped image is byte-identical before and after that fix, which confirms
fw-16mb's diagnosis that the paths arrive with the diagnostics.

**Static RAM is eight bytes.** Not a rounding of something larger — with the
diagnostic strings gone, so are the statics behind them. TFLM allocates
everything else out of the arena, which is the property the whole design rests
on, and this is that property showing up in a link map.

**And the standalone estimate below was 21% optimistic** — 62,332 B against the
79,444 B the real link costs. The gap is C++ runtime and linker behaviour that a
tiny `--gc-sections` probe does not reproduce. That is the same shape of error
`docs/cnn-on-device.md` records for the cycles/MAC prediction ("2.6x optimistic;
use 9"), so it is worth naming rather than quietly correcting: **a synthetic
link under-counts, and the number to quote is the one from the real image.**

The standalone measurement is kept below because it is the one that can be run
without a full firmware build, and because it is what isolates the `-O2`/`-Os`
and op-count choices from everything else in the image.

### The standalone measurement

`tools/size_tflm_m33.sh` cross-compiles with **the firmware's own compiler and
flags** — Arm GNU 14.2.rel1 out of `build/toolchain/`, `-mcpu=cortex-m33 -mthumb
-march=armv8-m.main+fp+dsp -mfloat-abi=softfp -mcmse` — and links a probe that
loads `si_real.tflite` and invokes it, with `--gc-sections`. The flags are read
off fw-16mb's generated build rather than reconstructed; an earlier pass here
used `-mfpu=fpv5-sp-d16` under GCC 14.3 and came out ~1 KB different.

The figure is the **difference** against an otherwise identical link, so
newlib's own baseline is not charged to TFLM, and the model is linked in as a
`const` array so the collector sees which kernels the graph actually reaches. A
stub model would undercount.

| configuration | flash delta | of which model | TFLM itself |
| --- | --- | --- | --- |
| 11 ops, `-O2`, diagnostics on | 138,644 B | 30,072 B | 108,572 B (106.0 KiB) |
| 11 ops, `-O2`, `TF_LITE_STRIP_ERROR_STRINGS` | 92,404 B | 30,072 B | **62,332 B (60.9 KiB)** |
| 11 ops, `-Os`, diagnostics on | 122,272 B | 30,072 B | 92,200 B (90.0 KiB) |
| 11 ops, `-Os`, stripped | 76,456 B | 30,072 B | **46,384 B (45.3 KiB)** |

**`-O2` costs 15,948 bytes over the port's own `-Os`**, and the build files take
it, because inference time on the board is the one number this module still
lacks and flash is the resource there is most of. If the board turns out fast
enough at `-Os`, that is 15.6 KiB back for one line.

A sixth-op trim — registering only what `si_real` uses instead of eleven
builtins — was measured at 45,344 B in a scratch build under the earlier
toolchain. Not repeated here, because at these sizes it is not worth the
resolver being narrower than the next model.

**The 46 KB that stripping saves is mostly newlib.** TFLM's diagnostic strings
drag in `_vfprintf_r` (8.9 KB), `_dtoa_r` (4.1 KB) and `_vfiprintf_r` (4.6 KB).
In the real image MicroPython may already carry some of that, so **60.9 KiB is
the pessimistic number and the true marginal cost is at most that**. Keep the
strings during bring-up; strip them to ship.

Do not be alarmed by the archive: `libtflm_m33.a` is 1.18 MB, and OpenMV's
prebuilt is 1.3 MB. Almost all of it is kernels nothing links.

**Against the image now on the board**: the reserve ahead of the
15,728,640-byte filesystem is 1 MB, of which 710,192 B was free. The stripped
module leaves **630,748 B** of it free. Not a budget question. Past the reserve
the line in `mpconfigboard.h` would have to grow, costing filesystem — but the
filesystem is there for the 6.8 MB corpus, so that would be a conversation and
not an edit.

### RAM

**Static: 332 bytes of `.bss`.** That is the entire fixed cost. TFLM allocates
nothing outside the arena, which is the property the whole design rests on.

**Arena: 28,664 bytes**, bisected on the 64-bit host — the smallest buffer for
which `AllocateTensors()` succeeds. `RecordingMicroInterpreter` breaks it down:

| | bytes | on a 32-bit target |
| --- | --- | --- |
| head — planned activations | 21,120 | **identical**, it is tensor data |
| tail — eval tensors, persistent tensors, quantisation, node registrations | 6,544 | smaller; every entry holds pointers |
| the shim's own handle (mostly the 11-entry op resolver table) | ~1,000 | roughly half |
| **total** | **28,664** | **~25.5 KB predicted, not measured** |

The 21,120-byte head is the same number `tools/tmdl_info.py` pads the `.tmdl` to.
Both are the model's own peak activation requirement; they agree because they
are the same quantity.

**And TFLM is the cheaper runtime in RAM, which was not expected.** The current
path costs a measured **~55.6 KB resident** (`docs/cnn-on-device.md`): 2.06x the
file for the model copy plus scratch, plus 12 KB at import. TFLM with the model
read from the filesystem costs 28.7 KB of arena plus a 30 KB copy of the model
inside it, so ~59 KB — about the same. But a **custom firmware makes the model a
`const` array in flash**, and then FlatBuffers reads it in place: aligned,
immovable, free. The arena holds activations only and the whole runtime costs
**~25 KB of heap**, less than half of today's. `modtflm.c` carries that path
behind `TFLM_BUILTIN_MODEL`, off by default — freezing the model means
reflashing to retrain, and the model is still changing.

### Speed

**Not measured on the board.** For scale, the host (arm64 M-series, reference
kernels, `-O2`) runs `si_real` in **0.838 ms**. That says nothing useful about
a 150 MHz M33 except that the reference kernels are not pathological.

The budget is loose either way: TinyMaix's measured 66.6 ms is 21% of
recognition and the panel refresh alone is 624 ms. TFLM's reference conv is a
straightforward im2col-free nested loop and is unlikely to be dramatically
worse than TinyMaix's 9 cycles/MAC; if it turns out to be, CMSIS-NN is the
lever, at the cost above.

## 4. The API

Shaped to match `emlearn_cnn_int8`'s so that `src/si_spot.py` changes an import
and a constructor call, not its logic.

```python
import tflm
from array import array

arena = bytearray(32 * 1024)                 # allocate once, up front
model = tflm.new(open("si_model.tflite", "rb").read(), arena)

model.input_dimensions()                     # (80, 26, 1)
model.output_dimensions()                    # (22,)
model.arena_used()                           # bytes actually needed

scores = array("f", (0.0 for _ in range(22)))
model.run(patch, scores)                     # patch: uint8 = int8 + 128
model.run_int8(patch, scores)                # or int8 directly
```

`Spotter.bind()` becomes:

```python
data = array("B", blob)                      #  before: 2.06x the file, and a
self.model = _cnn.new(data)                  #  bytearray is rejected outright

self.model = tflm.new(blob, self.arena)      #  after: bytes are fine
```

Three deliberate departures, each one a defect this project has already paid
for and each one checked by `tools/test_tflm_module.py`:

1. **The arena is the caller's.** emlearn sizes TinyMaix's scratch by the model
   file's length and never compares it against the model's own `buf_size` —
   confirmed on this board at 1164 bytes written past a heap allocation with
   nothing raised, which is why `tools/tmdl_info.py --pad` exists and why
   `deploy.sh` refuses an unpadded model. Here the caller passes a buffer, TFLM
   plans into it, and a buffer that is too small **raises at construction**.
   `arena_used()` turns sizing from a guess into a measurement. The whole class
   of fault is gone, along with the tool that guards it.
2. **Signedness is in the method name, not sniffed from the buffer.** emlearn
   rejects `bytearray` because `mp_get_buffer` does not report typecode `'B'`
   for one; sniffing typecodes is where that goes wrong. `run()` takes the
   uint8 transport `si_patch.py` already emits and `run_int8()` takes int8. Any
   buffer object works for either — `bytes`, `bytearray` and `array('B')` are
   all tested.
3. **The model is copied into the arena.** FlatBuffers reads the model in place
   with aligned loads and a `bytes` object off the MicroPython heap is not
   guaranteed 16-byte aligned. Copying costs the model's size once and removes
   a fault that would present as a wrong answer rather than a crash.

### What the module does not do, and should not

It exposes one input and one output tensor, both int8. Multi-input models,
float models and int16 models all fail loudly at `run()` with `TFLM_ERR_TYPE`
rather than being half-supported. `si_spot.py` needs exactly this shape, and a
runtime that quietly accepts a model it will get wrong is the failure mode this
whole exercise is about.

## 5. Build integration

Both mechanisms, both written, one of them exercised end to end:

| file | for | state |
| --- | --- | --- |
| `firmware/usermod/tflm/micropython.cmake` | the **rp2** port | **configures** — fw-16mb ran it through the hook and cmake reached this file's own `TFLM_DIR` guard. Not yet compiled or linked. |
| `firmware/usermod/tflm/micropython.mk` | the **unix** port | **built and run**, MicroPython v1.28.0 |
| `firmware/usermod/tflm/modtflm.c` | the binding | 269 lines |
| `firmware/usermod/tflm/tflm_shim.{h,cpp}` | the C facade | 312 lines, shared with the host proof |
| `tools/fetch_tflm.sh` | the source tree, from two pinned commits | run |
| `tools/test_tflm_module.py` | the binding's self-test | passes under the unix port |

The rp2 build goes through `tools/build_firmware.sh`, which owns the pinned
toolchain, the v1.28.0 version check and the mpy-cross ordering:

```sh
./tools/fetch_tflm.sh
USER_C_MODULES=$PWD/firmware/usermod/tflm/micropython.cmake \
BUILD_DIR=$PWD/build/mpy-16mb-tflm \
OUT=build/mpy-16mb-tflm.uf2 \
./tools/build_firmware.sh

# and to strip the diagnostic strings, which is 49,268 bytes of the image:
CMAKE_ARGS=-DTFLM_STRIP_ERROR_STRINGS=1 \
USER_C_MODULES=$PWD/firmware/usermod/tflm/micropython.cmake \
BUILD_DIR=$PWD/build/mpy-16mb-tflm-strip \
OUT=build/mpy-16mb-tflm-strip.uf2 \
./tools/build_firmware.sh
```

**Both paths must be absolute**, and that is the first of the traps below: the rp2 port's
Makefile passes `USER_C_MODULES` through to cmake unchanged and cmake resolves
it relative to `ports/rp2`, so a repo-relative path fails with

    USER_C_MODULES doesn't exist:
    .../ports/rp2/firmware/usermod/tflm/micropython.cmake

naming a path nobody wrote. A relative `BUILD_DIR` fails later and more
mildly — the image builds and only the final copy misses it, leaving the `.uf2`
under `build/micropython/ports/rp2/build/`.

`BUILD_DIR` and `OUT` are distinct from the defaults on purpose, so the plain
image that is currently flashed and verified stays untouched. `TFLM_DIR` is not
passed because this file defaults it to `vendor/tflm`, which is where
`fetch_tflm.sh` puts it.

**The compiler is pinned to Arm GNU 14.2.rel1** and that pin is not a
preference: Homebrew's `arm-none-eabi-gcc` ships without newlib and cannot link
pico-sdk's boot stage 2, and Arm 15.3 fails on mbedtls under
`-Werror=array-bounds`. The size figures above were taken with 14.2.rel1 for
that reason.

### Six traps, all of which fail quietly

Every one of these produced a *successful build* with the module simply absent,
or a failure naming the wrong file — the shape of failure this project keeps
meeting. The first is the relative-path one above; the rest follow.

**2. `USER_C_MODULES` means two different things.** For CMake it is the path to
a `.cmake` file. For Make it is the **parent directory** that gets globbed for
`*/micropython.mk`. Pointing Make at the module's own directory builds and
links cleanly and then fails at run time with `ImportError: no module named
'tflm'`, because nothing was ever added.

**3. The same variable reaches mpy-cross, which only knows the Make rule.**
Found by fw-16mb and fixed in `tools/build_firmware.sh`. `USER_C_MODULES` set
in the environment for an rp2 build is inherited by the **mpy-cross host
build**, which goes through `py/py.mk` and applies the *directory* convention
to a `.cmake` file:

    py/py.mk:37: *** USER_C_MODULES doesn't exist: .../micropython.cmake

The file plainly exists, and the failure arrives before the port is configured
at all, so it reads as a broken usermod rather than a build-script bug. The
script now clears the variable for that one command; **build through the script
rather than by hand.**

**4. MicroPython's Make ports do not compile `.cc`.** `py.mk` pattern-matches
`SRC_USERMOD_CXX` against `%.cpp` only, so a `.cc` file is accepted into the
variable, never compiled, never linked — and the build succeeds. This is why
upstream's `create_tflm_tree.py` has a `--rename_cc_to_cpp` flag and why
`tools/fetch_tflm.sh` renames the whole tree. Nothing in TFLM `#include`s a
`.cc` file, so the rename is safe.

**5. TFLM's sources belong in the `LIB` variables, and in CMake in a real
library.** `py.mk` adds `SRC_USERMOD_C`/`_CXX` to `SRC_QSTR`, which
preprocesses every listed file in a single `clang -E` invocation to harvest
`MP_QSTR_` names. Four hundred TFLM translation units contain no QSTRs;
scanning them costs minutes and enough memory that clang died with
`Broken pipe` here. `SRC_USERMOD_LIB_CXX` compiles and links without being
scanned. The CMake file does the equivalent by building TFLM as a `STATIC`
library and leaving only `modtflm.c` in the usermod target.

**6. A generated TFLM tree needs more than `tensorflow/lite`.**
`schema_utils.cc` includes `tensorflow/compiler/mlir/lite/kernels/internal/`,
and `micro_ops.h` includes `signal/micro/kernels/*.h` unconditionally even when
no signal operator is registered. `tools/fetch_tflm.sh` copies both.

### The tree is fetched, not vendored

~20 MB of upstream sources plus three third-party header trees, all
reproducible, so `vendor/tflm/` is gitignored on the same reasoning as
`build/`. The **pins are the point**: TFLM's arithmetic is what is being bought,
so the tree the firmware builds from has to be the tree the proof ran against.
`tools/fetch_tflm.sh` names them, and bumping them means re-running
`tools/tflm_vs_tflite.py`.

    tflite-micro   f8c117b558b0cf25cd8bff7143f4c361fbeced4f
    flatbuffers    v25.9.23   + TFLM's own no-dynamic-allocation patch
    gemmlowp       719139ce755a0f31cbf1c37f7f98adcc7fc9f425
    ruy            d37128311b445e758136b8602d1bbd2a755e115d

Licences are all permissive: TFLM Apache-2.0, flatbuffers and gemmlowp and ruy
Apache-2.0. `fetch_tflm.sh` copies each `LICENSE` into the tree.

Applying the flatbuffers patch is not cosmetic. It removes flatbuffers' fallback
to a default allocator; without it the interpreter can reach the C heap, which
on the board is not MicroPython's heap.

## What the morning confirms, and what it cannot

Written the night before, so that nobody re-derives the logic at a bench.

### The measurement is the byte check, and it is one command

    ./tools/make_tflm_cases.py                    # host: 30 patches + answers
    uvx mpremote connect $PORT cp build/tflm-cases/si_model.tflite :
    uvx mpremote connect $PORT cp -r build/tflm-cases/cases :
    uvx mpremote connect $PORT cp build/tflm-cases/manifest.txt :cases/
    uvx mpremote connect $PORT run tools/tflm_probe.py | tee /tmp/device.txt
    ./tools/check_tflm_device.py /tmp/device.txt   # exit 0 = confirmed

`check_tflm_device.py` compares all 22 output integers of all 30 cases against
`reference.txt` and exits non-zero on a single count. **That comparison is the
whole confirmation.** If it passes, the board computes what the host computes,
and every host number measured under the reference kernels is a device number
by construction -- no re-derivation, no separate accuracy run, nothing to
interpret.

If it fails, the failure is specific and the first suspect is a build flag; see
*Does bit-exactness carry* above. It is not a reason to retune anything.

**A `PROBE-ERROR` line is a bug in `tools/tflm_probe.py`, not a finding about
the board**, and `check_tflm_device.py` fails the run on one. That distinction
is the direct lesson of `tools/cnn_probe.py`, where every section's broad
`except` could dress a probe typo as a hardware fault.

### The operating point does **not** transfer from the recorded number, and this is the finding

The plan of record was: TFLM is bit-exact with the `.tflite`, so the `.tflite`
operating point -- **threshold 0.598, precision 1.000 at recall 0.500** on the
22 takes -- becomes the device operating point and the retune is a no-op.

**It is not a no-op, because that recorded number came from XNNPACK.** Scored
through `tools/si_eval.py --device-scores`, which is si-model's own scorer, the
same 22 takes and the same `si_real.tflite`:

| host runtime | recommended threshold | precision | recall |
| --- | --- | --- | --- |
| `default` (XNNPACK delegate) | 0.598 | 1.000 | 0.500 |
| `nodelegate` (optimised CPU) | 0.594 | 1.000 | 0.500 |
| **`ref` (reference kernels) = what TFLM computes** | **0.637** | **1.000** | **0.300** |

The recorded 0.598 / 0.500 reproduces exactly — under the delegate. The runtime
that will actually ship reaches precision 1.000 only at **0.637, recall 0.300**.

**The mechanism is small and specific.** `problem_01.wav` -- an
out-of-vocabulary take -- comes back `brother` at p = 0.625 under the reference
kernels. The threshold that must exclude it therefore sits above 0.625, and two
correct answers live between 0.598 and 0.637. The whole difference is a handful
of quantisation counts moving one negative across one threshold, which is
precisely the size of the runtime disagreement measured in *Which TFLite*
(up to 8 counts of 1/256).

Three things follow, and the order matters:

1. **The model has not got worse.** Top-1 is 0.700 under all three runtimes,
   identically. What moved is where a threshold can be placed, not what the
   network computes.
2. **0.300 against 0.500 is two utterances out of ten**, which si-model's own
   A/B criteria call noise at this sample size. It is not evidence that TFLM
   classifies worse; it is evidence that **a threshold tuned on one runtime does
   not transfer to another**, which is the same lesson `docs/speech.md` is built
   around and the same one `docs/cnn-on-device.md` recorded for TinyMaix.
3. **So the retune is required, and it is cheap.** It does not need the board:
   `tools/make_tflm_cases.py` already produces the reference-kernel scores, and
   `si_eval.py --device-scores build/tflm-cases/reference.txt` sweeps them. What
   the board adds is confirmation that those host scores *are* the device's,
   which is the byte check above.

**Filter to `set=takes` before scoring.** Every `SCORE` line carries `set=`, and
the eight `set=bitexact` patches have **no ground truth** -- their filenames
encode what an *earlier* model predicted. A scorer that does not filter treats
all eight as must-stay-silent negatives; run over all 30 it recommends threshold
0.637 for a partly fictional reason, and the agreement with the correct answer
is a coincidence.

### What the morning cannot settle

- **Whether TFLM is more accurate than TinyMaix.** Ten positives and twelve
  negatives; recall moves in steps of 0.1. The paired run
  (`runtime=tflm` and `runtime=tinymaix` on identical inputs in one session)
  removes the session variable, which is worth having, but not the sample size.
- **Whether either operating point survives a stranger.** Every take is one
  speaker, who is also the person the gates were tuned against.
- **Anything about a live microphone.** These are staged input tensors,
  deliberately: it puts the front end outside the experiment so that a byte
  difference has exactly one possible cause.

## 6. What this does not settle

In the order they matter.

1. **Nothing has run on an RP2350.** The module is proved under the unix port,
   proved to link for M33, and its cmake is proved to configure inside the real
   rp2 build — but no rp2 image containing it has been compiled, and the 16 MB
   image now on the board does not contain it. The first board session should run
   `tools/test_tflm_module.py` and then re-run the 30 cases and check them byte
   for byte against the host — `docs/cnn-on-device.md`'s standing rule is that
   a clean import is not the test.
2. **Inference time on the board is unknown.** If TFLM's reference kernels come
   in far above TinyMaix's 66.6 ms, the decision is CMSIS-NN versus a slower
   turn, and section 2 argues that trade is not obviously worth taking.
3. **The operating point moves, and this time in a direction that can be
   predicted.** `si_spot.py`'s `THRESHOLD`, `MARGIN` and `TIE_FLOOR` were tuned
   on TinyMaix, on the board. Under TFLM the right values are the ones tuned on
   the `.tflite` — which is what `si-model` measured originally, precision 1.000
   at recall 0.500 — because host and device would finally be the same thing.
   That is the *benefit*, but it is still a re-tune, and it must be re-measured
   rather than assumed.
4. **The arena on a 32-bit target is predicted, not measured** — ~25.5 KB
   against the host's 28,664 B. `arena_used()` reports it on the board in one
   line.
5. **The 6-op flash figure** came from a scratch build with a hand-trimmed
   resolver. If 61.5 KiB ever matters, trim the resolver properly and re-measure.
6. **The binding does not expose the raw int8 output tensor**, and one thing
   currently depends on that being unnecessary. `tflm_shim.h` has the int8
   pointer and both output quantisation parameters; `modtflm.c` binds none of
   them, so `run()` returns dequantised floats only. Every device dump
   therefore recovers the model's output bytes as `round(p * 256)`, which is
   exact **only because `si_real` quantises its output at scale 2^-8, zero
   point -128** -- a property of this model, not of the module. A retrain onto
   an awkward scale would silently invalidate it, and the byte comparison would
   fail while blaming the board.
   Mitigated rather than fixed: `tools/make_tflm_cases.py` checks the round
   trip against the library's raw int8 on all 30 cases, and
   `tools/tflm_probe.py` section 2b checks on the device that every score lands
   on a multiple of 1/256 and refuses to emit a dump otherwise (measured worst
   departure: exactly 0). **The fix is to bind the int8 pointer**, which makes
   it structural instead of fortunate and costs a firmware rebuild -- so it is
   the first thing to do after the bench, not before it. (Raised by fw-16mb.)
7. **`emlearn_cnn_int8` does not have to go**, and the version pin is what
   keeps that true. The vendored `.mpy` is a **native** module at mpy 6.3 /
   `armv7emsp`, and native modules are rejected unless `MPY_VERSION` and
   `MPY_SUB_VERSION` both match. v1.28.0 is 6.3, so it stays loadable in this
   image — but neither this module nor anything else can bump the MicroPython
   version without costing the board its existing CNN runtime, and the board
   has no networking to fetch a replacement. (fw-16mb, unprompted, and it is
   the kind of thing that is only ever noticed after the fact.)
   Keeping both makes the fallback chain TFLM → TinyMaix → DTW for the price of
   the 12 KB TinyMaix import. Whether that is prudence or clutter is a call for
   whoever ships this.
