"""Self-test for the `tflm` user C module. Run under MicroPython, not CPython.

It proves the binding, not the arithmetic -- `tools/tflm_vs_tflite.py` owns the
arithmetic. What it checks is the set of things emlearn's TinyMaix wrapper gets
wrong and this module is written not to (docs/tflm-usermod.md has the history):

  - an arena that is too small **raises** instead of overrunning the heap.
    TinyMaix sizes its scratch by the model file's length and never compares it
    against the model's own requirement; measured on this board, 1164 bytes
    written past the allocation with nothing raised.
  - `bytes`, `bytearray` and `array('B')` are all accepted as input. emlearn
    rejects `bytearray` outright, because it sniffs a typecode that
    `mp_get_buffer` does not report for one.
  - a wrong-length input raises rather than reading past the buffer.
  - `run()` (uint8 + 128 transport, what `src/si_patch.py` emits) and
    `run_int8()` produce exactly the same scores.

    micropython tools/test_tflm_module.py               # unix port, repo root
    uvx mpremote connect $PORT run tools/test_tflm_module.py

Needs `build/si_real.tflite` and `build/kw_unknown_0.bin` in the working
directory, so on the board those two have to be copied over first.
"""

import sys
from array import array

try:
    import tflm
except ImportError:
    # The module only exists where it has been compiled in: a MicroPython
    # unix-port binary built with USER_C_MODULES=firmware/usermod, or an rp2
    # firmware carrying the usermod. Under CPython, or under a stock
    # MicroPython, that is an environment fact and not a failure -- skip
    # loudly rather than fail, so a suite sweep stays meaningful. (The
    # convention here: FAIL means code is wrong.)
    #
    #   make -C <micropython>/ports/unix \
    #        USER_C_MODULES=<repo>/firmware/usermod TFLM_DIR=<repo>/vendor/tflm
    print("SKIP: tflm module not compiled into this interpreter "
          "(build the unix port with USER_C_MODULES, or run on the board)")
    sys.exit(0)

MODEL = "build/si_real.tflite"
PATCH = "build/kw_unknown_0.bin"
N_CLASSES = 22


def main():
    blob = open(MODEL, "rb").read()
    arena = bytearray(64 * 1024)
    model = tflm.new(blob, arena)

    print("input_dimensions ", model.input_dimensions())
    print("output_dimensions", model.output_dimensions())
    print("arena_used       ", model.arena_used())

    patch = open(PATCH, "rb").read()
    scores = array("f", (0.0 for _ in range(N_CLASSES)))
    model.run(patch, scores)
    best = 0
    for i in range(1, N_CLASSES):
        if scores[i] > scores[best]:
            best = i
    print("argmax           ", best, "p=%.4f" % scores[best])

    # The two entry points must agree to the last bit, not merely to the
    # argmax: the whole reason for this module is that a count of 1/256 is the
    # size of si_spot's MARGIN.
    signed = bytearray((b - 128) & 0xFF for b in patch)
    other = array("f", (0.0 for _ in range(N_CLASSES)))
    model.run_int8(signed, other)
    same = all(scores[i] == other[i] for i in range(N_CLASSES))
    print("run == run_int8  ", same)

    # An arena below what arena_used() reported must fail loudly.
    try:
        tflm.new(blob, bytearray(31 * 1024))
        print("small arena       FAILED -- no exception")
    except ValueError as exc:
        print("small arena       raises:", exc)

    try:
        model.run(b"short", scores)
        print("short input       FAILED -- no exception")
    except ValueError as exc:
        print("short input       raises:", exc)

    for kind, buf in (("bytes", patch),
                      ("bytearray", bytearray(patch)),
                      ("array('B')", array("B", patch))):
        got = array("f", (0.0 for _ in range(N_CLASSES)))
        model.run(buf, got)
        print("accepts %-11s %s" % (kind, got[best] == scores[best]))


main()
