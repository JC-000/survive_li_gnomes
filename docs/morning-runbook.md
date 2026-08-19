# Morning runbook: the TFLM image on live silicon

Written the night of 2026-08-18, to be executed by whoever is holding the board
with the user at the desk. It is steps, not intent — the intent is in
[tflm-usermod.md](tflm-usermod.md) (why TFLM at all) and
[restore-factory-firmware.md](restore-factory-firmware.md) (what the images are).

**State the board is in right now.** Powered down, carrying the verified 16 MB
image and a working ELIZA deploy. Nothing is half-done, so there is nothing to
finish before starting.

| | |
| --- | --- |
| firmware | `WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2` |
| filesystem | 15,728,640 bytes, 27 files deployed |
| port last night | `/dev/cu.usbmodem1401` — **do not assume it, see step 1** |

## What is being tested, in gating order

Each step is a gate: if it fails, the next one tells you nothing. Stop, and go
to *Rollback*.

1. The image boots and the filesystem is intact.
2. `import tflm` returns a module.
3. `tools/test_tflm_module.py` — the binding behaves.
4. The 30 cases reproduce byte for byte — **the whole reason for the exercise**.
5. Inference timing, against TinyMaix's 66.6 ms.
6. The `si_spot` A/B, if fw-tflm has the swap path ready.

Steps 1–3 are mechanical. Step 4 is the one worth the bench session: TinyMaix
changes top-1 on 3 of 8 patches, TFLM does not, and that claim is only proved on
the host so far.

---

## 0. Before touching anything

```sh
shasum -a 256 -c firmware/backup/SHA256SUMS
```

Both `OK` before proceeding. This is the factory net and it is checked first
because it is the only thing here that cannot be rebuilt.

Confirm the images are the ones this document means:

| file | sha256 |
| --- | --- |
| `firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2` | `facdd8c5d1133407cf7f179d42b601aa6ae9a4ccb912b04b88e046ffcdaddd8b` |
| `firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2` | `aeea05f47a60af1f3de75fd4649aaf53930905559ec808f5a0efdf849d40417e` |
| `firmware/RPI_PICO2-20260406-v1.28.0.uf2` | `e65ad62ae886a4f56da8ef2c07904fe504b92de69e5ae6489acf881bcf30b6ae` |

There is also a **stripped** variant — same build with TFLM's diagnostic
strings compiled out, 49,268 bytes smaller:

| file | sha256 |
| --- | --- |
| `build/mpy-16mb-tflm-strip.uf2` | `d43152045c953f01343936d37e7c8dde9b35667fd261e565cc245b012628b837` |

Verified for this board (`pico_board: waveshare_rp2350_touch_epaper_154`,
`embedded drive 0x10100000-0x11000000`), but **it is not the one to flash this
morning**. During bring-up, "Failed to allocate tail memory. Requested: 816,
available 8" is the difference between a diagnosis and a guess, and 49 KB out
of 581 KB of headroom buys nothing today. Strip it once the arena is sized and
the model is settled. Note it lives in `build/`, which is disposable — if it
matters, move it to `firmware/` and give it a name.

Both `firmware/*.uf2` files are gitignored. If the combined one is missing,
rebuild it — about six minutes:

```sh
USER_C_MODULES=firmware/usermod/tflm/micropython.cmake \
BUILD_DIR=build/mpy-16mb-tflm \
OUT=firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2 \
./tools/build_firmware.sh
```

## 1. Find the port by UID, do not assume it

It re-enumerated under the same name last night, and there is a second,
unrelated serial device on this Mac (`/dev/cu.usbmodem246802461`, a USB modem).
Identify the board by its UID rather than by which port appeared:

```sh
ioreg -r -c IOUSBHostDevice -l -w0 | grep -B4 -A4 9717825cf3bcf97c
ls /dev/cu.usbmodem*
```

The board is VID `2E8A`, PID `0005`, serial `9717825cf3bcf97c`, product name
"Board in FS mode". Export the port once and use it everywhere below:

```sh
PORT=/dev/cu.usbmodem1401     # whatever step 1 actually found
```

## 2. Nothing to rescue — but check, do not remember

Last night's answer was "nothing on the board exists only on the board", and
that was true of last night's filesystem. It may not be true of this morning's,
so the listing is repeated rather than recalled:

```sh
uvx mpremote connect $PORT ls
```

Everything expected is reproducible: the `.py` modules from `src/`, `laugh.raw`
from `clips/`, `si_model.tmdl` from `models/si_real.tmdl`,
`emlearn_cnn_int8.mpy` from `vendor/`. Anything else — recordings, captures,
a `templates.bin` from an enrolment done since — **comes off the board first**,
because the flash reformats it:

```sh
uvx mpremote connect $PORT cp :whatever.bin ./whatever.bin
```

Record the before figure, so the after figure means something:

```sh
uvx mpremote connect $PORT exec "import os; s=os.statvfs('/'); print(s[0]*s[2])"
# expect 15728640
```

## 3. Flash the combined image

```sh
uvx mpremote connect $PORT bootloader
# wait for it to appear; picotool info should print the *current* image
picotool info
picotool load firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2 -x
```

About five seconds. The board reboots itself and the serial device comes back —
re-run step 1 if the name changed.

### Verification ladder

```sh
uvx mpremote connect $PORT exec "
import os, sys
s = os.statvfs('/')
print('filesystem', s[0]*s[2])
print(os.uname())
"
```

Expect **15,728,640** again, `_build='WAVESHARE_RP2350_TOUCH_EPAPER_154'`, and
an empty filesystem — the reformat happens on this flash too, for the same
block-count reason, and it prints nothing while doing it.

**The capacity test is optional this time, and here is why.** `flash_capacity.py`
answers "does the part really answer above 4 MB", which was settled on
2026-08-18: 10,485,760 bytes written and read back with zero bad chunks. The
part has not changed and neither has the flash geometry — same
`embedded drive: 0x10100000-0x11000000 (15360K)`, confirmed in the combined
image before flashing. Run it only if `statvfs` disagrees with the number above,
in which case something is wrong that it will characterise.

## 4. TFLM on the board, in gating order

Copy the two fixtures 4b needs — they live in `build/`, which is gitignored,
and the board has no networking. 4c copies its own, 30 more files:

```sh
uvx mpremote connect $PORT cp build/si_real.tflite :si_real.tflite
uvx mpremote connect $PORT cp build/kw_unknown_0.bin :kw_unknown_0.bin
```

### 4a. Does the module exist

```sh
uvx mpremote connect $PORT exec "import tflm; print(tflm)"
```

This is the step that the whole host-side verification cannot reach. It is
proved in the image — `tflm_user_cmodule` is linked at `0x1006cc44`, the qstr
`tflm` is in the table, `b'tflm'` is in the UF2 payload and absent from the
plain image — but a module that links is not a module that imports.

`ImportError` here with everything else green means the module was built and
registered but not reached; that is a build question, not a board question, and
the board should go back to the plain image while it is answered.

### 4b. The binding

```sh
uvx mpremote connect $PORT run tools/test_tflm_module.py
```

Proves what TinyMaix gets wrong and this module is written not to: a too-small
arena **raises** instead of overrunning the heap (TinyMaix wrote 1164 bytes past
its allocation on this board and raised nothing), `bytes`/`bytearray`/`array('B')`
are all accepted, a wrong-length input raises, and `run()` and `run_int8()`
agree.

### 4c. The 30 cases, byte for byte

**The gate.** Everything before it is plumbing; this is the measurement the
TFLM decision rests on. `tools/tflm_vs_tflite.py` has already shown host TFLM
equals host *reference* TFLite on these 30 cases; this shows the board computes
the same bytes as the host, which closes the chain from training to glass.

The cases are eight `kw_unknown_*` patches — the ones where TinyMaix changes
top-1 on 3 of 8 — plus 22 real takes from `takes/` and `takes-oov/`, featurised
by the same `tools/si_features.py` the board's front end uses. They are already
built at `build/tflm-cases-int8/` (30 files, 62,400 bytes). Rebuild them only if the
model changed:

```sh
.venv/bin/python tools/tflm_cases.py
```

Copy the model and the cases over — about 62 KB, a few seconds on a 15 MB
filesystem:

```sh
uvx mpremote connect $PORT cp build/si_real.tflite :si_real.tflite
uvx mpremote connect $PORT mkdir :cases
for f in build/tflm-cases-int8/*.i8; do
    uvx mpremote connect $PORT cp "$f" ":cases/$(basename $f)"
done
```

Run it, keeping the output, and diff:

```sh
uvx mpremote connect $PORT run tools/tflm_device_cases.py | tee /tmp/device.txt
.venv/bin/python tools/tflm_compare_cases.py \
    build/tflm-cases-int8/reference.txt /tmp/device.txt --expect 30
```

**Read the exit status, not the prose**: 0 only when all 30 match byte for byte,
2 otherwise. `--expect 30` makes a run that dies at case 19 a failure rather
than a pass over the cases that survived.

A `DIFFER` line names the class index and the difference in counts. One class
off by one count is a rounding path; many classes far apart is a different
model, a different input, or a broken kernel — check the `# model` line in the
device dump against `model_sha256` in the reference before assuming arithmetic.

**There are two chains for this step, written the same night.** fw-tflm built
`tools/make_tflm_cases.py` + `tools/tflm_probe.py` + `tools/check_tflm_device.py`
(uint8 transport, si-model's `SCORE` line format, and it runs **TinyMaix beside
TFLM on the same inputs**, which substantiates the 3-of-8 claim on the device
rather than only on the host). The commands above are the other one. They stage
into separate directories -- `build/tflm-cases/` is theirs, `build/tflm-cases-int8/`
is this one -- so either can be run without disturbing the other.

Their two host dumps were cross-checked on 2026-08-18 and **agree on all 30
cases, all 660 integers**, having been written independently from the same
model. Run either at the bench; if there is time, run both, because agreement
between two implementations is worth more than either alone. If only one is run,
theirs carries the TinyMaix comparison and is the better use of the window.

Two things worth knowing before reading a result:

- **The comparison is on the raw int8 output tensor.** The device's binding
  returns float32 scores, and the driver recovers the int8 from them. That is
  exact rather than approximate: this model's output scale is 2^-8 with zero
  point -128, so every score is (q + 128)/256, all 256 of which are exactly
  representable in float32. `tflm_cases.py` refuses to write a dump if the
  model's quantisation ever stops matching, and the comparator re-proves the
  round trip over all 256 values on every run.
- **The whole chain was exercised on the host on 2026-08-18** by
  `tools/test_tflm_cases.py`, which runs the same device script against the host
  library and checks that the gate *fails* when a single count is flipped and
  when two cases go missing. A gate that cannot fail is worse than no gate, so
  that control exists. What it cannot test is the RP2350: same source, different
  compiler, different target. That is this step.

### 4d. Inference time

**Already measured by 4c** — `tflm_device_cases.py` times every invoke and
prints `# time <case> <ms>` per case and a mean at the end, so this step is
reading numbers you already have rather than a separate run.

The figure to beat is TinyMaix's **66.6 ms** for `si_real` (docs/cnn-on-device.md).
TFLM's reference kernels are not the optimised CMSIS-NN ones — that was a
deliberate choice for bit-exactness — so slower is expected and acceptable; the
budget is the pause after the button, against a panel that takes ~583 ms to
redraw regardless.

### 4e. The si_spot A/B

Only if fw-tflm has the swap path ready. Otherwise it is the next session's
work, and the runbook ends at 4d.

## 5. Rollback ladder

Each rung is one command pair plus a redeploy, and each is named by file. Going
down a rung reformats the filesystem again — that is expected, and it is why
step 2 exists.

| rung | when | command |
| --- | --- | --- |
| 1. plain 16 MB | TFLM misbehaves; keeps the 15 MB filesystem | `picotool load firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2 -x` |
| 2. stock RPI_PICO2 | the custom build itself is suspect; back to 3 MB | `picotool load firmware/RPI_PICO2-20260406-v1.28.0.uf2 -x` |
| 3. factory C firmware | MicroPython itself is suspect | `picotool load firmware/backup/factory-main.uf2 -x` |
| 4. whole factory flash | last resort, 16 MB, ~1 min | `picotool load firmware/backup/factory-flash-16MB.bin -o 0x10000000 -x` |

Each preceded by `uvx mpremote connect $PORT bootloader`, and rungs 1 and 2
followed by `PORT=$PORT ./tools/deploy.sh eliza`.

Rung 1 is the one to expect to use. Rung 4 exists only on this Mac — see
[restore-factory-firmware.md](restore-factory-firmware.md).

## 6. End state: hand the board back working

```sh
PORT=$PORT ./tools/deploy.sh eliza
```

Skip the `ava_*.pcmw` audition clips; they were dropped deliberately.
`templates.bin` will warn and that is the known state — no enrolment has ever
been completed.

### Then leave the board in the friendly REPL, not the raw one

**This cost an hour on 2026-08-18.** `mpremote exec` and `mpremote run` drive
the board through the *raw* REPL. If a session ends without leaving it — an
interrupted command, a tool that exits early — the board stays parked there,
`main.py` is not running, and the next connection fails with:

    mpremote.transport.TransportError: could not enter raw repl

which reads exactly like dead hardware and is not. It also means the deployed
program is not running, so the user pressing the screen gets nothing.

The fix is a soft reboot, which drops out of raw REPL and runs `main.py` from
the top:

```sh
uvx mpremote connect $PORT soft-reset
```

Then witness the boot rather than assuming it:

```sh
uvx mpremote connect $PORT exec "print('alive')"
```

If that prints, the board is at a REPL that answers. If it does not, use
`uvx mpremote connect $PORT reset` (a hard reset) and try again — and if the
program is genuinely running and busy in a turn, the REPL may legitimately not
answer, which is why the last check is a human one.

### One human press

Hold the screen or POWER, say "mother", let go. That is the only end-to-end
test that exists — panel, touch, codec, capture, spotter and rules in one path —
and no amount of green above substitutes for it.

Watch for the panel: a full refresh holds BUSY ~1290 ms and flashes; a partial
takes ~583 ms and does not. A redraw that returns instantly did not reach the
glass.
