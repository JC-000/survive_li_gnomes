"""Run the staged cases through TFLM on the board, and TinyMaix beside it.

    ./tools/make_tflm_cases.py
    P=/dev/cu.usbmodem1401
    uvx mpremote connect $P cp build/tflm-cases/si_model.tflite :
    uvx mpremote connect $P cp -r build/tflm-cases/cases :
    uvx mpremote connect $P cp build/tflm-cases/manifest.txt :cases/
    uvx mpremote connect $P run tools/tflm_probe.py | tee /tmp/device.txt
    ./tools/check_tflm_device.py /tmp/device.txt

Touches no peripheral -- no panel, no codec, no I2C -- so it is safe against a
board mid-way through anything else, exactly as `tools/cnn_probe.py` is.

## What it answers, in order

1. does `tflm` import, and what does it cost the heap
2. does it load `si_model.tflite`, and what arena does it actually need
3. **do the 30 staged cases come back byte-identical to the host** -- this is
   the test the whole TFLM case rests on, and `check_tflm_device.py` is the
   thing that decides it, not a human reading two columns
4. what does one inference cost in microseconds, against TinyMaix's measured
   66604 us
5. the same 30 through TinyMaix, in the same session, so the comparison has no
   session variable in it

## Two things `tools/cnn_probe.py` learned the hard way

**A probe section with its own broad `except` converts probe bugs into
plausible-looking hardware failures.** So the structure here is deliberate and
different: **setup does not catch anything.** If the import fails, the model
will not load, or the manifest is missing, the traceback is the report -- those
are unambiguous and a swallowed exception would only disguise them. Only the
per-case loop catches, and what it prints is tagged `PROBE-ERROR`, which
`check_tflm_device.py` treats as a failure of *this file* rather than a finding
about the board. A probe that cannot tell you which of the two it found is
worse than no probe.

**Nothing is `del`'d that a later section needs.** Both runtimes are loaded up
front and both stay bound for the whole run, which costs ~100 KB of heap and
buys a run that cannot fail halfway through for a reason that is not about the
hardware. Section 5 reports the heap so that cost is visible rather than
assumed.

## Output format

One `SCORE` line per case per runtime, in si-model's ingest format, so a whole
unedited capture can be pasted into `tools/si_eval.py --device-scores`:

    SCORE runtime=tflm set=takes name=mother_01.wav frames=36 clipped=0 us=... q=...

`q=` is 22 integers 0..255 with `probability = q / 256`. **Integers, not
floats**: comparing a device float against a host float through two different
`printf` implementations is a worse test than comparing the bytes the model
produced. The module returns dequantised floats, so `q` is recovered as
`round(p * 256)` -- exact, because the output scale is 2^-8 and the numerator is
an integer, and `tools/make_tflm_cases.py` checks that round trip on the host
rather than asserting it here.

**`set=` matters and is not decoration.** `set=takes` are the 22 real
utterances and carry ground truth in their filenames. `set=bitexact` are the 8
patches from `docs/cnn-on-device.md`, which carry **no** ground truth -- their
names encode what an earlier model predicted. Scoring those as accuracy is
wrong, and a scorer that does not filter on `set=` will silently treat all 8 as
must-stay-silent negatives.
"""

import gc
import sys
import time

CASES = "/cases"
MODEL = "si_model.tflite"
N_CLASSES = 22
# 64 KB, and the first draft of this file said 40 -- which is the interpreter's
# own requirement (28,664 B measured) with the model copy left out. `tflm.new()`
# puts the model inside the arena, so the buffer has to hold both: ~58.8 KB on a
# 64-bit host, less on the board where the metadata pointers halve. It raised
# cleanly rather than overrunning, which is the whole design difference from
# TinyMaix -- but the probe should not need that mercy, so this is generous and
# `arena_used()` below reports what was actually needed.
ARENA_BYTES = 64 * 1024
REPEATS = 5             # for the timing figure; the byte check needs only one


def rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def heap():
    gc.collect()
    return gc.mem_alloc()


def read_manifest(path):
    """-> [(set, name, frames, clipped)]. Missing file is a hard failure."""
    out = []
    handle = open(path)
    try:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError("bad manifest line: %r" % line)
            case_set, name, frames, clipped = parts
            out.append((case_set, name,
                        None if frames == "-" else int(frames),
                        None if clipped == "-" else int(clipped)))
    finally:
        handle.close()
    return out


def quantised(scores, n):
    """float scores -> si-model's `q`, integers 0..255. Exact; see the header."""
    out = []
    for i in range(n):
        v = int(scores[i] * 256 + 0.5)
        if v < 0:
            v = 0
        elif v > 255:
            v = 255
        out.append(v)
    return out


def score_line(runtime, case_set, name, q, frames, clipped, micros):
    parts = ["SCORE", "runtime=" + runtime, "set=" + case_set, "name=" + name]
    if frames is not None:
        parts.append("frames=%d" % frames)
    if clipped is not None:
        parts.append("clipped=%d" % clipped)
    parts.append("us=%d" % micros)
    parts.append("q=" + ",".join(str(v) for v in q))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 0. identity, and the manifest
# ---------------------------------------------------------------------------
# Nothing below this point is caught. A failure here is unambiguous -- the
# module is absent, the model was not copied, the staging is wrong -- and the
# traceback names it exactly. Wrapping it would turn "you forgot to copy the
# model" into something that reads like a hardware fault.

rule("0. build identity")
print("sys.implementation :", sys.implementation)
print("sys.version        :", sys.version)
base = heap()
print("heap at start      : %d B allocated" % base)

manifest = read_manifest(CASES + "/manifest.txt")
print("cases              : %d (%d takes, %d bitexact)"
      % (len(manifest),
         sum(1 for c in manifest if c[0] == "takes"),
         sum(1 for c in manifest if c[0] == "bitexact")))

rule("1. import tflm")
before = heap()
import tflm                                              # noqa: E402
after = heap()
print("import cost        : %d B" % (after - before))

rule("2. load si_model.tflite into a caller-owned arena")
handle = open(MODEL, "rb")
try:
    blob = handle.read()
finally:
    handle.close()
print("model file         : %d B" % len(blob))

arena = bytearray(ARENA_BYTES)
before = heap()
model = tflm.new(blob, arena)
after = heap()
print("input_dimensions   :", model.input_dimensions())
print("output_dimensions  :", model.output_dimensions())
print("arena_used         : %d B of %d B offered" % (model.arena_used(),
                                                     ARENA_BYTES))
print("new() heap cost    : %d B (the arena itself is %d B, allocated above)"
      % (after - before, ARENA_BYTES))

dims = model.output_dimensions()
if dims[0] != N_CLASSES:
    raise SystemExit("model has %d outputs, expected %d -- wrong model staged"
                     % (dims[0], N_CLASSES))

# `blob` is deliberately still bound. tflm.new() copies the model into the
# arena, so it is not needed -- but cnn_probe.py has twice been bitten by a
# `del` of something a later section wanted, and 30 KB is not worth the risk of
# repeating that. It is dropped at the end, where nothing follows it.

# ---------------------------------------------------------------------------
# 3. TinyMaix beside it, if it is still deployed
# ---------------------------------------------------------------------------
# Loaded here rather than in its own section further down, so that both models
# are resident for the whole run and the timing figures in section 4 are taken
# under the same heap. Absence is a finding, not an error: a board flashed with
# the TFLM image and never given a .tmdl is a perfectly ordinary state.

rule("3. TinyMaix, for the paired comparison")
from array import array                                  # noqa: E402

tinymaix = None
tinymaix_why = None
try:
    import emlearn_cnn_int8 as _tm
except ImportError as exc:
    _tm = None
    tinymaix_why = "emlearn_cnn_int8 not in this firmware (%s)" % exc

if _tm is not None:
    try:
        tm_handle = open("si_model.tmdl", "rb")
        try:
            tm_blob = tm_handle.read()
        finally:
            tm_handle.close()
        tm_data = array("B", tm_blob)
        tinymaix = _tm.new(tm_data)
        print("si_model.tmdl      : %d B, loaded" % len(tm_blob))
    except OSError as exc:
        tinymaix_why = "si_model.tmdl not on the board (%s)" % exc
    except ValueError as exc:
        tinymaix_why = "TinyMaix rejected the model (%s)" % exc

if tinymaix is None:
    print("TinyMaix           : not available -- %s" % tinymaix_why)
    print("                     TFLM numbers below stand alone; the paired")
    print("                     comparison si-model asked for needs both.")
else:
    print("TinyMaix           : ready, paired comparison enabled")

# ---------------------------------------------------------------------------
# 4. the cases, through both runtimes
# ---------------------------------------------------------------------------

rule("4. cases -- SCORE lines follow, one per case per runtime")
print("(paste this whole capture into tools/si_eval.py --device-scores;")
print(" filter to set=takes for tuning -- set=bitexact has no ground truth)")
print()

scores = array("f", (0.0 for _ in range(N_CLASSES)))
tm_scores = array("f", (0.0 for _ in range(N_CLASSES)))
errors = 0
tflm_total = 0
tflm_n = 0
tm_total = 0
tm_n = 0

for case_set, name, frames, clipped in manifest:
    try:
        handle = open(CASES + "/" + name + ".bin", "rb")
        try:
            patch = handle.read()
        finally:
            handle.close()

        model.run(patch, scores)                 # warm, so timing excludes any
        best = time.ticks_us()                   # first-call effects
        for _ in range(REPEATS):
            model.run(patch, scores)
        micros = time.ticks_diff(time.ticks_us(), best) // REPEATS
        tflm_total += micros
        tflm_n += 1
        print(score_line("tflm", case_set, name, quantised(scores, N_CLASSES),
                         frames, clipped, micros))

        if tinymaix is not None:
            tinymaix.run(patch, tm_scores)
            t0 = time.ticks_us()
            for _ in range(REPEATS):
                tinymaix.run(patch, tm_scores)
            tm_micros = time.ticks_diff(time.ticks_us(), t0) // REPEATS
            tm_total += tm_micros
            tm_n += 1
            print(score_line("tinymaix", case_set, name,
                             quantised(tm_scores, N_CLASSES),
                             frames, clipped, tm_micros))
    except Exception as exc:      # noqa: BLE001
        # Tagged so it cannot be mistaken for a finding about the board.
        # check_tflm_device.py fails the run on any of these.
        errors += 1
        print("PROBE-ERROR name=%s %s: %s" % (name, type(exc).__name__, exc))

# ---------------------------------------------------------------------------
# 5. what it cost
# ---------------------------------------------------------------------------

rule("5. summary")
if tflm_n:
    print("TFLM     : %d cases, mean %d us/inference" % (tflm_n, tflm_total // tflm_n))
if tm_n:
    print("TinyMaix : %d cases, mean %d us/inference" % (tm_n, tm_total // tm_n))
    print("           (docs/cnn-on-device.md measured 66604 us for this model)")
print("probe errors: %d" % errors)
print("heap now    : %d B allocated, %d B above the start"
      % (heap(), heap() - base))

del blob                                     # nothing follows this line

print()
print("The byte-for-byte verdict is NOT in this output -- run")
print("    ./tools/check_tflm_device.py <this capture>")
print("which diffs every q= against the host reference and exits non-zero on")
print("any difference. Reading two columns of 22 integers by eye is how a")
print("one-count difference gets agreed to at the bench.")
print()
print("This probe touched no peripheral and left no files behind.")
print("It does NOT leave the board in raw REPL by itself -- but `mpremote run`")
print("does, if it is interrupted mid-run. A board in raw REPL is")
print("indistinguishable from a dead one: buttons do nothing and there is no")
print("output. Recovery is CTRL-B then CTRL-D over a held serial connection.")
