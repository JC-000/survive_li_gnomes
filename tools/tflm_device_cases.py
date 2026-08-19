"""Run the 30 comparison cases through TFLM **on the board**. Runs under MicroPython.

Step 4c of [docs/morning-runbook.md](../docs/morning-runbook.md), and the one
measurement the whole TFLM decision rests on. `tools/tflm_vs_tflite.py` proves
host TFLM equals host reference TFLite; this produces the device's answers for
the same cases so `tools/tflm_compare_cases.py` can show they are the same
bytes. A clean `import tflm` proves nothing about arithmetic.

    .venv/bin/python tools/tflm_cases.py          # writes build/tflm-cases-int8/
    uvx mpremote connect $PORT cp build/si_real.tflite :si_real.tflite
    uvx mpremote connect $PORT mkdir :cases
    for f in build/tflm-cases-int8/*.i8; do
        uvx mpremote connect $PORT cp "$f" ":cases/$(basename $f)"; done
    uvx mpremote connect $PORT run tools/tflm_device_cases.py | tee device.txt
    .venv/bin/python tools/tflm_compare_cases.py \\
        build/tflm-cases-int8/reference.txt device.txt

Output is one `CASE <name> <hex>` line per case, in the same format the host
dump uses, interleaved with `# ` comment lines the comparator ignores.

**Printed as they compute, not collected and printed at the end.** A run that
dies at case 19 -- out of memory, a kernel fault, a board that resets -- should
leave nineteen results behind and a clear stopping point, not an empty
traceback. The same reason `tools/speech_probe.py` prints as it goes.

**Nothing here catches exceptions.** If the module raises, that traceback is
the finding, and converting it into a printed verdict would turn a driver bug
into a hardware conclusion. The only `except` is the ImportError for the
module, which is an environment fact with an actionable message.

## Why it prints int8 when the binding returns floats

`Model.run_int8()` writes float32 scores; the C shim's raw int8 output pointer
is not plumbed through the MicroPython binding. It does not have to be for this
comparison to be exact. `si_real`'s output quantisation is **scale 2^-8, zero
point -128**, so every score is (q + 128) / 256 for an int8 q, and every one of
those 256 values is exactly representable in float32. The multiply is
exact, so recovering q by `round(score * 256) - 128` is lossless -- not a
tolerance, not a rounding convention, an identity.

`tools/tflm_cases.py` refuses to write a dump if the model's quantisation ever
stops matching that, and `tools/tflm_compare_cases.py` re-proves the round trip
over all 256 values every time it runs. If a retrained model breaks it, the
fix is to plumb the int8 pointer through `modtflm.c` -- the shim already has it.
"""

import gc
import os
import sys
from array import array

try:
    import tflm
except ImportError:
    print("FAIL: no `tflm` module in this firmware.")
    print("      Flash the combined image:")
    print("      firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB-tflm.uf2")
    sys.exit(1)

try:
    from time import ticks_diff, ticks_ms
except ImportError:
    # CPython, for the host-side test of this script's own logic
    # (tools/test_tflm_cases.py). Not a fallback the board ever takes.
    from time import monotonic as _monotonic

    def ticks_ms():
        return int(_monotonic() * 1000)

    def ticks_diff(a, b):
        return a - b

# gc.mem_free() is MicroPython's, and the board is what these numbers are for.
# Absent under CPython, where tools/test_tflm_cases.py runs this same script
# against the host library; the heap lines are simply not printed there rather
# than being faked.
_mem_free = getattr(gc, "mem_free", None)


def heap_note(label):
    if _mem_free is not None:
        print("# heap %s %d" % (label, _mem_free()))


# Over the REPL each print already goes out on its own, so this costs nothing
# on the board; it is here so that a run piped through `tee` on the host still
# leaves the finished cases behind when the run dies in the middle of the set.
_flush = getattr(sys.stdout, "flush", None)


def flush():
    if _flush is not None:
        _flush()


MODEL = "si_real.tflite"
CASE_DIR = "cases"
ARENA_BYTES = 64 * 1024

# The two constants that make the float scores exact int8. Asserted against the
# model on the host by tools/tflm_cases.py; hard-coded here because the binding
# does not expose the output quantisation and a wrong guess must not be silent.
OUT_SCALE_RECIP = 256      # 1 / 2**-8
OUT_ZERO_POINT = -128


def case_files():
    names = [f for f in os.listdir(CASE_DIR) if f.endswith(".i8")]
    names.sort()
    return names


def main():
    blob = open(MODEL, "rb").read()
    arena = bytearray(ARENA_BYTES)
    model = tflm.new(blob, arena)

    n_out = model.output_dimensions()[-1]
    scores = array("f", (0.0 for _ in range(n_out)))

    print("# tflm case dump v1")
    print("# side device")
    print("# model %s %d bytes" % (MODEL, len(blob)))
    print("# n_out %d" % n_out)
    print("# input_dimensions %s" % (model.input_dimensions(),))
    print("# arena_used %d of %d" % (model.arena_used(), ARENA_BYTES))
    heap_note("free")

    names = case_files()
    print("# cases %d" % len(names))

    total_ms = 0
    for name in names:
        patch = open(CASE_DIR + "/" + name, "rb").read()
        t0 = ticks_ms()
        model.run_int8(patch, scores)
        ms = ticks_diff(ticks_ms(), t0)
        total_ms += ms

        # Recover the int8 output tensor. Exact -- see the module docstring.
        out = bytearray(n_out)
        for i in range(n_out):
            q = int(round(scores[i] * OUT_SCALE_RECIP)) + OUT_ZERO_POINT
            if q < -128 or q > 127:
                # Not reachable for a well-formed int8 tensor, and if it
                # happens the dump must not silently clamp it into agreement.
                raise ValueError("case %s class %d: score %r -> q %d, "
                                 "outside int8" % (name, i, scores[i], q))
            out[i] = q & 0xFF

        stem = name[:-3]
        print("CASE %s %s" % (stem, "".join("%02x" % b for b in out)))
        print("# time %s %d ms" % (stem, ms))
        flush()

    print("# end %d cases" % len(names))
    if names:
        print("# inference mean %d ms over %d cases (TinyMaix was 66.6 ms)"
              % (total_ms // len(names), len(names)))
    heap_note("free after")


main()
