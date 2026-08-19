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
> | flash, Cortex-M33, linked delta | **61.5 KiB** stripped, 106.9 KiB with diagnostics | ~670 KB spare in the 1 MB firmware reserve |
> | static RAM (`.bss`) | **332 B** | — |
> | tensor arena, 64-bit host | **28,664 B** | TinyMaix path: ~55.6 KB resident |
> | inference time on the board | **not measured** | TinyMaix: 66.6 ms |
>
> **The one thing not measured is the one that needs the board**: nothing here
> has run on an RP2350. The module is proved to compile, load a model and
> classify under the **unix** port of MicroPython v1.28.0, and proved to
> cross-compile and link for Cortex-M33. Between those two there is no rp2
> firmware build, because the board is in use.
>
> **And one existing figure is now suspect.** `si-model`'s host references were
> taken from a default `tf.lite.Interpreter`, which installs the XNNPACK
> delegate. Measured here, that runtime differs from TFLM by up to 8 counts of
> 1/256 on these same 30 cases. See *Which TFLite* below — it changes what
> "host TFLite said" means in three documents.

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
- **The one way to break this is a compiler flag.** `-ffast-math` or
  `-ffp-contract=fast` at that stage would be enough. The build files say so in
  a comment, and neither is set.
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

### Flash, measured

`tools/size_tflm_m33.sh` cross-compiles for `-mcpu=cortex-m33 -mthumb
-mfloat-abi=softfp -mfpu=fpv5-sp-d16` — the rp2 port's own flags for RP2350 —
and links a probe that loads `si_real.tflite` and invokes it, with
`--gc-sections`. The figure is the **difference** against an otherwise identical
link, so newlib's own baseline is not charged to TFLM, and the model is linked
in as a `const` array so the collector sees which kernels the graph actually
reaches. A stub model would undercount.

| configuration | flash delta | of which model | TFLM itself |
| --- | --- | --- | --- |
| 11 ops, diagnostics on | 139,528 B | 30,072 B | **109,456 B (106.9 KiB)** |
| 11 ops, `TF_LITE_STRIP_ERROR_STRINGS` | 93,000 B | 30,072 B | **62,928 B (61.5 KiB)** |
| 6 ops (`si_real`'s own), stripped | 75,416 B | 30,072 B | 45,344 B (44.3 KiB) † |

† measured in a scratch build with a hand-trimmed resolver, not with the
shipped one.

**The 47 KB that stripping saves is mostly newlib.** TFLM's diagnostic strings
drag in `_vfprintf_r` (8.9 KB), `_dtoa_r` (4.1 KB) and `_vfiprintf_r` (4.6 KB).
In the real image MicroPython may already carry some of that, so **61.5 KiB is
the pessimistic number and the true marginal cost is at most that**. Keep the
strings during bring-up; strip them to ship.

Do not be alarmed by the archive: `libtflm_m33.a` is 1.18 MB, and OpenMV's
prebuilt is 1.3 MB. Almost all of it is kernels nothing links.

Against fw-16mb's image — ~330 KB in a 1 MB firmware reserve, with 15 MB of
filesystem beyond it — 61.5 KiB is not a budget question.

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
| `firmware/usermod/tflm/micropython.cmake` | the **rp2** port | written, **not built** — needs cmake and the board's toolchain |
| `firmware/usermod/tflm/micropython.mk` | the **unix** port | **built and run**, MicroPython v1.28.0 |
| `firmware/usermod/tflm/modtflm.c` | the binding | 269 lines |
| `firmware/usermod/tflm/tflm_shim.{h,cpp}` | the C facade | 312 lines, shared with the host proof |
| `tools/fetch_tflm.sh` | the source tree, from two pinned commits | run |
| `tools/test_tflm_module.py` | the binding's self-test | passes under the unix port |

The rp2 hook, to be run against `fw-16mb`'s tree:

```sh
cmake -S <micropython>/ports/rp2 -B build-tflm \
      -DMICROPY_BOARD=WAVESHARE_RP2350_TOUCH_EPAPER_154 \
      -DMICROPY_BOARD_DIR=$PWD/firmware/boards/WAVESHARE_RP2350_TOUCH_EPAPER_154 \
      -DUSER_C_MODULES=$PWD/firmware/usermod/tflm/micropython.cmake \
      -DTFLM_DIR=$PWD/vendor/tflm
```

### Four traps, all of which fail quietly

Each of these produced a *successful build* with the module simply absent or
uncompilable, which is the shape of failure this project keeps meeting.

**1. `USER_C_MODULES` means two different things.** For CMake it is the path to
a `.cmake` file. For Make it is the **parent directory** that gets globbed for
`*/micropython.mk`. Pointing Make at the module's own directory builds and
links cleanly and then fails at run time with `ImportError: no module named
'tflm'`, because nothing was ever added.

**2. MicroPython's Make ports do not compile `.cc`.** `py.mk` pattern-matches
`SRC_USERMOD_CXX` against `%.cpp` only, so a `.cc` file is accepted into the
variable, never compiled, never linked — and the build succeeds. This is why
upstream's `create_tflm_tree.py` has a `--rename_cc_to_cpp` flag and why
`tools/fetch_tflm.sh` renames the whole tree. Nothing in TFLM `#include`s a
`.cc` file, so the rename is safe.

**3. TFLM's sources belong in the `LIB` variables, and in CMake in a real
library.** `py.mk` adds `SRC_USERMOD_C`/`_CXX` to `SRC_QSTR`, which
preprocesses every listed file in a single `clang -E` invocation to harvest
`MP_QSTR_` names. Four hundred TFLM translation units contain no QSTRs;
scanning them costs minutes and enough memory that clang died with
`Broken pipe` here. `SRC_USERMOD_LIB_CXX` compiles and links without being
scanned. The CMake file does the equivalent by building TFLM as a `STATIC`
library and leaving only `modtflm.c` in the usermod target.

**4. A generated TFLM tree needs more than `tensorflow/lite`.**
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

## 6. What this does not settle

In the order they matter.

1. **Nothing has run on an RP2350.** The module is proved under the unix port
   and proved to link for M33. The first board session should run
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
6. **`emlearn_cnn_int8` does not have to go.** Keeping both means the fallback
   chain becomes TFLM → TinyMaix → DTW, which costs the 12 KB TinyMaix import
   and nothing else. Whether that is prudence or clutter is a call for whoever
   ships this.
