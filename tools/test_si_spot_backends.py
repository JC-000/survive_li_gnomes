"""Exercise `si_spot`'s two runtimes. Run under MicroPython, not CPython.

    micropython tools/test_si_spot_backends.py          # unix port, repo root
    uvx mpremote connect $PORT run tools/test_si_spot_backends.py

`tools/tflm_probe.py` compares the two runtimes' *numbers*. This checks the
thing above them: that `si_spot.Spotter` can be pointed at either one, that the
selection rules do what they say, and that a board carrying only one of them
still binds. The gates are not re-tested here because they are literally the
same code on both paths -- that is the point of putting the backend inside
`bind()` rather than writing a second Spotter.

Four selection rules, each of which could plausibly have been written the other
way and each of which would be a bad surprise at a bench:

  1. `bind()` with no argument takes `BACKEND_DEFAULT`, which is TinyMaix.
  2. `bind()` falls through to the other runtime if the default's module is not
     in the firmware -- so the TFLM-only image is not a silent spotter.
  3. `bind(backend="tflm")` forces TFLM.
  4. **An explicit backend that is unavailable is an error, not a
     fall-through.** A board that reports TinyMaix numbers when asked for TFLM
     ones would poison the A/B, and this is the morning's A/B.

Needs `si_model.tflite` and/or `si_model.tmdl` in the working directory, plus
`vocab`. It binds models but runs no audio, so it touches no peripheral.
"""

import sys

sys.path.insert(0, "src")

# `si_spot` imports `si_patch`, which imports `spotter`, which is full of
# `@micropython.viper` -- and the unix port rejects that decorator outright
# ("invalid micropython decorator"), because viper is an ARM/Xtensa code
# generator. So on a host this file cannot import the real front end, and it
# substitutes a stub *before* si_spot is imported.
#
# That is honest for what this file tests and would not be for anything else:
# the stub stands in for the feature path, which is not exercised here at all,
# and `bind()` -- the whole subject -- never touches it. On the board the real
# module is present and the stub is not installed. Which happened is printed,
# because a test that silently ran against a stub is worse than no test.
STUBBED = False
try:
    import si_patch                                       # noqa: F401
except SyntaxError:
    class _StubPatch:
        """Enough of si_patch for `bind()`; every call raises if reached."""

        @staticmethod
        def allocate():
            return None

        @staticmethod
        def patch_for(*args, **kwargs):
            raise AssertionError("this test does not exercise the front end")

    sys.modules["si_patch"] = _StubPatch()
    STUBBED = True

import si_spot


def report(label, spotter, ok):
    state = "bound backend=%s" % spotter.backend if ok else "not bound"
    print("  %-46s %s" % (label, state))
    if not ok and spotter.error:
        print("  %-46s   error: %s" % ("", spotter.error))
    return ok


def main():
    if STUBBED:
        print("NOTE: si_patch is stubbed -- the real one needs viper, so it")
        print("      cannot import on a host. bind() does not use it; the")
        print("      feature path is tested by tools/test_si_patch.py.")
        print()
    print("runtimes visible to si_spot:")
    print("  emlearn_cnn_int8 (TinyMaix) :", si_spot._cnn is not None)
    print("  tflm (TFLM)                 :", si_spot._tflm is not None)
    print("  BACKEND_DEFAULT             :", si_spot.BACKEND_DEFAULT)
    print("  classes                     :", len(si_spot.CLASSES))
    print()

    have_tm = si_spot._cnn is not None
    have_tflm = si_spot._tflm is not None
    failures = []

    print("1. bind() with no argument")
    spotter = si_spot.Spotter()
    ok = spotter.bind()
    report("default", spotter, ok)
    if have_tm:
        if not ok or spotter.backend != "tinymaix":
            failures.append("default did not choose tinymaix")
    elif have_tflm:
        # Rule 2. The fall-through is what keeps a TFLM-only image useful.
        if not ok or spotter.backend != "tflm":
            failures.append("default did not fall through to tflm")
    else:
        if ok:
            failures.append("bound with no runtime present")

    print("2. bind(backend='tflm')")
    spotter = si_spot.Spotter()
    ok = spotter.bind(backend="tflm")
    report("explicit tflm", spotter, ok)
    if have_tflm:
        if not ok or spotter.backend != "tflm":
            failures.append("explicit tflm did not bind tflm")
        else:
            print("  %-46s arena_used=%d" % ("", spotter.model.arena_used))
    elif ok:
        failures.append("explicit tflm bound without the module")

    print("3. bind(backend='tinymaix')")
    spotter = si_spot.Spotter()
    ok = spotter.bind(backend="tinymaix")
    report("explicit tinymaix", spotter, ok)
    if have_tm:
        if not ok or spotter.backend != "tinymaix":
            failures.append("explicit tinymaix did not bind tinymaix")
    elif ok:
        # Rule 4, and the one that matters most for the A/B.
        failures.append("explicit tinymaix bound something else -- an "
                        "explicit request must not fall through")

    print("4. bind(backend='nonsense')")
    spotter = si_spot.Spotter()
    ok = spotter.bind(backend="nonsense")
    report("unknown backend", spotter, ok)
    if ok:
        failures.append("an unknown backend name bound something")

    print("5. a missing model file degrades rather than raising")
    spotter = si_spot.Spotter()
    ok = spotter.bind(path="no_such_model.bin",
                      backend="tflm" if have_tflm else "tinymaix")
    report("missing file", spotter, ok)
    if ok:
        failures.append("bound a model file that does not exist")

    print()
    if failures:
        print("FAIL")
        for line in failures:
            print("  - " + line)
        return 1
    print("PASS -- all selection rules behave as documented")
    return 0


sys.exit(main())
