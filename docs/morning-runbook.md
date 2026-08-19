# Morning runbook: the TFLM image on live silicon

Written the night of 2026-08-18, to be executed by whoever is holding the board
with the user at the desk. **Executed 2026-08-19**: the flash took, 4a-4d all
passed, `check_tflm_device.py` exited 0 on 30/30 bit-identical with worst count
0, and TinyMaix differed on top-1 in 5 of the 30 — the 3 documented
`kw_unknown` patches plus `problem_01` and `wonder_01`, which are real
recordings. TFLM 245.6 ms per inference against TinyMaix's 66.6 ms, both
measured in that one session. The two corrections marked *2026-08-19* below
came out of running it. It is steps, not intent — the intent is in
[tflm-usermod.md](tflm-usermod.md) (why TFLM at all) and
[restore-factory-firmware.md](restore-factory-firmware.md) (what the images are).

**State the board is in as of 2026-08-19, after the run.** Carrying the
combined TFLM image and a fresh ELIZA deploy, with every staged test file
removed. Nothing is half-done.

| | |
| --- | --- |
| firmware | `WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2` (`e7b1069a…`) |
| filesystem | 15,728,640 bytes, 20 files — the ELIZA program and its data, nothing else |
| port | `/dev/cu.usbmodem1401` — **do not assume it, see step 1** |
| `import tflm` | works; the module is in this image |

Re-running this document from step 0 is safe and repeats the measurement. Note
that flashing the *same* image again does **not** reformat the filesystem: the
block count is unchanged, so littlefs mounts what is already there. Only a
change of flash geometry — this image or the plain one against the stock
`RPI_PICO2` build — triggers the reformat step 2 protects against.

## What is being tested, in gating order

Each step is a gate: if it fails, the next one tells you nothing. Stop, and go
to *Rollback*.

1. The image boots and the filesystem is intact.
2. `import tflm` returns a module.
3. `tools/test_tflm_module.py` — the binding behaves.
4. `tools/tflm_probe.py` + `check_tflm_device.py` — the 30 cases reproduce byte
   for byte. **The whole reason for the exercise.**
5. Inference timing, against TinyMaix's 66.6 ms — falls out of step 4.
6. The `si_spot` A/B, if fw-tflm has the swap path ready.

Steps 1–3 are mechanical. Step 4 is the one worth the bench session: TinyMaix
changes top-1 on 3 of 8 patches, TFLM does not, and that claim has so far only
been proved on the host and against a unix-port MicroPython standing in for the
board. A pass there makes every host figure measured under reference kernels a
device figure, which is why there is no separate accuracy run in this list.

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
| `firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2` | `e7b1069af4eba02b1aadcf8a115f26f4b1b1d36e25eec6d8ddbd51d99cafe63c` |
| `firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2` | `aeea05f47a60af1f3de75fd4649aaf53930905559ec808f5a0efdf849d40417e` |
| `firmware/RPI_PICO2-20260406-v1.28.0.uf2` | `e65ad62ae886a4f56da8ef2c07904fe504b92de69e5ae6489acf881bcf30b6ae` |

A **stripped** variant exists — the same build with TFLM's diagnostic strings
compiled out, 49,268 bytes smaller, 815,616 B,
sha256 `d43152045c953f01343936d37e7c8dde9b35667fd261e565cc245b012628b837`.
It is verified for this board and it is deliberately **not** in `firmware/`:

```sh
EXTRA_CMAKE_ARGS=-DTFLM_STRIP_ERROR_STRINGS=1 \
USER_C_MODULES=firmware/usermod/tflm/micropython.cmake \
BUILD_DIR=build/mpy-16mb-tflm-strip OUT=build/mpy-16mb-tflm-strip.uf2 \
./tools/build_firmware.sh
```

`firmware/` holds exactly the three images this runbook flashes — the target
and the two rollback rungs — because a fourth one whose name differs from the
target by `-strip` is something a tired hand tab-completes into at 9am, and the
cost of that mistake is losing the diagnostics precisely when diagnosing. It
rebuilds in six minutes and reproduces the hash above exactly.

It should not be the first flash regardless: "Failed to allocate tail memory.
Requested: 816, available 8" is the difference between a diagnosis and a guess
— it is what identified a 40 KB-arena mistake in one line during development —
and 49 KB out of 581 KB of headroom buys nothing today. Strip it once the arena
is sized and the model has settled.

All the `firmware/*.uf2` files are gitignored. If the combined one is missing,
rebuild it — about six minutes:

```sh
USER_C_MODULES=firmware/usermod/tflm/micropython.cmake \
BUILD_DIR=build/mpy-16mb-tflm \
OUT=firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2 \
./tools/build_firmware.sh
```

**A rebuild reproduces the hash above exactly**, so a mismatch means something
really did change rather than "builds differ, they always do". That is true
because of a fix made the same night: TFLM's assertion macros were baking
`__FILE__` into the image, 19 absolute host paths and 2,133 bytes of them,
which made the binary depend on where the repo sat on disk. `-ffile-prefix-map`
on `tflm_lib` removed them. Checked rather than assumed — the whole tree was
built a second time from a different directory, and the two images are
byte-identical. Add `EXTRA_CMAKE_ARGS=-DTFLM_STRIP_ERROR_STRINGS=1` for the
stripped variant, whose bytes the path fix does not change at all, because the
paths only ever existed inside the diagnostics that variant compiles out.

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
and the board has no networking. 4c copies its own, 30 more files.

**They go in `build/` on the device, not at the root.** `test_tflm_module.py`
opens them by the relative paths `build/si_real.tflite` and
`build/kw_unknown_0.bin`, so root copies fail with a bare `OSError: ENOENT` at
the open, which reads like a missing file rather than a misplaced one:

```sh
uvx mpremote connect $PORT mkdir :build
uvx mpremote connect $PORT cp build/si_real.tflite :build/si_real.tflite
uvx mpremote connect $PORT cp build/kw_unknown_0.bin :build/kw_unknown_0.bin
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

The cases are eight `kw_unknown_*` patches — the ones `docs/cnn-on-device.md`
records TinyMaix disagreeing on — plus all 22 takes from `takes/` (10) and
`takes-oov/` (12), none dropped by the endpointer. They are staged as tensors
rather than WAVs on purpose: if the board featurised the audio itself, a byte
difference would have two possible causes, and this step is about the model.
The front end is proved separately, by `tools/test_si_patch.py` and by
`speech_probe` section (h) on this very firmware.

#### Run this one

fw-tflm's chain, verified end to end against a unix-port MicroPython carrying
the usermod — 30 cases, 30/30 bit-identical, exit 0. It runs **TinyMaix beside
TFLM on the same inputs**, so the 3-of-8 claim gets made on the board rather
than quoted at the bench from a host result.

```sh
./tools/make_tflm_cases.py                    # only if the model changed
uvx mpremote connect $PORT cp build/tflm-cases/si_model.tflite :
uvx mpremote connect $PORT cp -r build/tflm-cases/cases :
uvx mpremote connect $PORT cp build/tflm-cases/manifest.txt :cases/
uvx mpremote connect $PORT run tools/tflm_probe.py | tee /tmp/device.txt
./tools/check_tflm_device.py /tmp/device.txt
```

**Read the exit status, not the prose.** Zero means every case matched on all
22 integers, with no case missing and no `PROBE-ERROR` line. Nobody should be
reading two columns of 22 integers at a bench.

#### The second opinion, if there is time

A second chain exists, written the same night, independently, from the same
model. It stages into `build/tflm-cases-int8/` and does not disturb the files
above.

```sh
.venv/bin/python tools/tflm_cases.py          # only if the model changed
uvx mpremote connect $PORT cp build/si_real.tflite :si_real.tflite
uvx mpremote connect $PORT mkdir :cases
for f in build/tflm-cases-int8/*.i8; do
    uvx mpremote connect $PORT cp "$f" ":cases/$(basename $f)"
done
uvx mpremote connect $PORT run tools/tflm_device_cases.py | tee /tmp/device-i8.txt
.venv/bin/python tools/tflm_compare_cases.py \
    build/tflm-cases-int8/reference.txt /tmp/device-i8.txt --expect 30
```

Same rule: exit 0 or it did not pass. `--expect 30` makes a run that dies at
case 19 a failure rather than a pass over the cases that survived. A `DIFFER`
line names the class index and the difference in counts — one class off by one
count is a rounding path, many classes far apart is a different model, a
different input, or a broken kernel.

**The two host dumps were cross-checked on 2026-08-18 and agree on all 30
cases, all 660 integers.** Two implementations, different transport
conventions, same numbers — so the host reference does not rest on one
person's featurisation call. If the two chains ever disagree *at the bench*,
that disagreement is itself the finding and neither result should be reported
until it is understood.

#### What a pass means, and what it does not

A pass makes every host number measured under **reference kernels** a device
number, by construction. There is no separate accuracy run to schedule.

That matters this morning because the recorded operating point moved, and not
because of anything TFLM did. fw-tflm's measurement: the `.tflite` figures on
record (threshold 0.598, precision 1.000 at recall 0.500) were taken through
XNNPACK; under the reference kernels TFLM computes, the same takes and the same
scorer give **0.637 at recall 0.300**. Top-1 is 0.700 under all three host
runtimes, so the model did not change — what changed is where a threshold can
sit. It is si-model's number to own; see `docs/tflm-usermod.md`, *What the
morning confirms*.

### 4d. Inference time

**Already measured by 4c** — both probes time every invoke, so this step is
reading numbers already in the capture rather than a separate run.

The figure to beat is TinyMaix's **66.6 ms** for `si_real` (docs/cnn-on-device.md).
TFLM's reference kernels are not the optimised CMSIS-NN ones — that was a
deliberate choice for bit-exactness — so slower is expected and acceptable; the
budget is the pause after the button, against a panel that takes ~583 ms to
redraw regardless.

### 4e. The si_spot A/B

Only if fw-tflm has the swap path ready. Otherwise it is the next session's
work, and the runbook ends at 4d.

**Do not use the recorded 0.598 threshold for it.** That number was measured
through XNNPACK, and the board computes reference kernels; on the same 22 takes
with the same scorer, reference kernels need **0.637 for precision 1.000, and
recall falls to 0.300**. Top-1 is 0.700 under all three host runtimes, so this
is a threshold moving, not a model changing.

The retune is host-side work and it is si-model's: they hold the reference-kernel
dump and run the sweep. **The A/B uses whatever threshold that sweep blesses**,
not the constant currently in the tree. If the sweep has not landed by the time
you reach this step, run 4a–4d, report them, and leave 4e — an A/B against a
threshold known to be measured on the wrong kernels would produce a number
somebody would later have to unpublish.

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

### `voice.pak` is verified on the device, and here is why it has to be

1.9 MB over `mpremote cp`, and **nothing downstream would notice a truncated
transfer**. A short pak still has a valid header, `Pak.open()` still reports the
right clip count and rate (both live in the first 16 bytes), and the index still
bisects — so the boot line `voice.pak bound (113 clips, 16000 Hz)` prints
exactly as it does on a good upload. The first clip whose blob starts past the
cut is the one that fails, mid-reply, and it presents as a decoder bug.

`deploy.sh` therefore hashes the file **on the board** after copying it and
refuses the deploy on a mismatch. Reading the count back is not a substitute and
neither is `ls`: both are satisfied by the first 16 bytes.

To check a board that is already deployed, without redeploying:

```sh
uvx mpremote connect $PORT exec "
import hashlib, binascii
h = hashlib.sha256(); buf = bytearray(4096)
with open('voice.pak', 'rb') as f:
    while True:
        n = f.readinto(buf)
        if not n: break
        h.update(memoryview(buf)[:n])
print(binascii.hexlify(h.digest()).decode())"
shasum -a 256 corpus-voice/voice.pak
```

For the 2026-08-19 corpus both must read
`2b4bf8a5b3165e86609a8fb950f4497eefd8c5b8d7b0f8e0dac3a8d3be8197fe`.

### Then leave the board in the friendly REPL, not the raw one

**This cost an hour on 2026-08-18.** `mpremote exec` and `mpremote run` drive
the board through the *raw* REPL. If a session ends without leaving it — an
interrupted command, a tool that exits early — the board stays parked there,
`main.py` is not running, and the next connection fails with:

    mpremote.transport.TransportError: could not enter raw repl

which reads exactly like dead hardware and is not. It also means the deployed
program is not running, so the user pressing the screen gets nothing.

**Check the host first — this bit us on 2026-08-19.** Both fixes below need the
port, and neither can get it if something on this Mac is already holding it. A
stale `mpremote`, or a `pyserial` script that hung in `close()`, produces
*exactly* the same `could not enter raw repl`, and so does `picotool`
failing to reach the device. The board is fine and nothing will tell you so:

```sh
lsof /dev/cu.usbmodem1401
```

**Identify what it names before killing it.** Not "kill whatever holds the
port" — that rule is wrong and cost a near-miss on 2026-08-19.

**The commonest thing it names is one of ours**, and there are two kinds that
look *identical* in `lsof` and `ps`:

- **A stuck orphan.** `uvx --from pyserial python -` fed by a heredoc orphans
  under this shell: the parent returns, the child is reparented to PID 1, and
  it sits forever on a stdin that will never arrive. It cannot make progress,
  so killing it costs nothing. Three of these on 2026-08-19.
- **A deliberate read-only tap.** Someone monitoring a live user session:
  pyserial reading the port into a log so every turn is visible. Also
  `python -`, also reparented to 1, also with a closed stdin — because its work
  is the read loop, not the stdin. Killing it destroys someone's session
  record and blinds them mid-run.

`ps -o ppid= -p <pid>` returning `1` therefore proves nothing either way. **Ask
before killing** if a session might be in progress. The tell that separates them
after the fact: once the port is free, a board that answers *immediately with no
reset* was never hung, so whatever held it was doing no harm.

Avoid creating the first kind at all: write the script to a file and
`python thatfile.py` rather than piping a heredoc into `python -`. The
scratchpad is the place for it. On the morning this was written the board
answered `alive` immediately afterwards **with no reset at all** — which is the
proof it had never hung, since a hung program would still have been hung.

If the port is genuinely free, the fix is a soft reboot, which drops out of raw
REPL and runs `main.py` from the top:

```sh
uvx mpremote connect $PORT soft-reset
```

On a connection you are already holding — a `mpremote repl` session, or a
terminal that was interrupted mid-run — the same thing by hand is **CTRL-B**
(leave raw REPL for the friendly one) then **CTRL-D** (soft reboot). Worth
knowing because that is the case where there is no separate shell to run the
command above from. `tools/tflm_probe.py` prints this reminder when it
finishes, for exactly this reason.

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
