#!/usr/bin/env python3
"""Host-side wrapper around the device endpointer. **The algorithm lives in
`src/vad.py`** -- this file adds WAV loading and a CLI and nothing else.

    python3 tools/vad.py somefile.wav        # report the endpoints it finds

## Why this is a wrapper and not a second implementation

Templates are enrolled on the host and matched on the device. If the two sides
trim a word differently -- 30 ms of margin here, none there -- then every
template is compared against a differently shaped runtime segment, and the DTW
distances drift for a reason that shows up in neither file's own tests.

This started as a real second implementation kept in step by
`tools/test_record_stream.py` and `tools/test_vad.py`. It drifted twice in one
afternoon anyway, in both directions, so the two agree by construction now
instead of by assertion. A test that catches drift is worth much less than a
structure that cannot drift.

`src/vad.py` cannot simply be imported by path from the host in the obvious way,
because it must stay importable on the board: no `wave`, no `argparse`, no
`sys.path` assumptions. So it is loaded here by explicit file path rather than
by name. That also removes a trap this file used to have -- `import vad` picked
up whichever of `src/` and `tools/` came first on `sys.path`, and with a warm
feature cache the wrong choice ran to completion and only failed later, on the
first run that had to decode a WAV.

`tools/test_vad.py` keeps its value after the conversion: it guards the
`@micropython.viper` `_frame_stats` against its portable twin, which is the one
part a shared module cannot protect. Its empty-band tripwire is worth keeping
alive too -- see the sigma gate note in `src/vad.py`.
"""

# Private aliases, every one of them. The wholesale re-export below copies
# *all* of src/vad.py's module namespace into this one, and that namespace
# contains its imports too -- `from array import array` there would land the
# array *class* on top of a plain `import array` here, and the failure
# ("type object 'array.array' has no attribute 'array'") points at this file
# rather than at the re-export that caused it.
import array as _array_mod
import importlib.util as _importlib_util
import os as _os
import sys as _sys
import wave as _wave

_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "src", "vad.py")

if not _os.path.exists(_SRC):
    raise ImportError("cannot find the shared endpointer at %s" % _SRC)

_spec = _importlib_util.spec_from_file_location("_device_vad", _SRC)
_device = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_device)

# Re-export wholesale rather than by name. A hand-written list is one more
# thing to forget to update when src/vad.py grows a constant, and forgetting
# would reintroduce exactly the drift this file exists to prevent.
for _name, _value in vars(_device).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value

device = _device   # for anything that wants to be explicit about the source


def read_wav(path):
    """16 kHz mono int16 WAV -> array('h').

    Anything else is an error, loudly: a silently resampled template would be
    worse than a missing one. Stays here rather than in src/vad.py because a
    module importing `wave` will not load on the board.
    """
    with _wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("%s: need mono 16-bit, got %d ch / %d bytes"
                             % (path, w.getnchannels(), w.getsampwidth()))
        if w.getframerate() != 16000:
            raise ValueError("%s: need 16000 Hz, got %d"
                             % (path, w.getframerate()))
        data = w.readframes(w.getnframes())
    return _array_mod.array("h", data)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        samples = read_wav(path)
        n_frames = len(samples) // VAD_FRAME             # noqa: F821
        energy, zcr = frame_stats(samples, n_frames)     # noqa: F821
        itl, itu, izct, imn, egate = thresholds(energy, zcr)  # noqa: F821
        span = endpoints(samples)                        # noqa: F821
        print("%s: %d samples (%.0f ms)" % (path, len(samples),
                                            1000.0 * len(samples) / 16000))
        imx = max(energy) if n_frames else 0
        print("  ITL %d  ITU %d  IZCT %d  background %d  zcr gate %d%s"
              % (itl, itu, izct, imn, egate,
                 "  [BAND EMPTY]" if egate >= itl else ""))
        print("  imx/imn %.1f%s"
              % (imx / imn if imn else 0.0,
                 "  <-- below ~12x, the ZCR pass cannot help here"
                 if imn and imx < 12 * imn else ""))
        if span is None:
            print("  no utterance found")
        else:
            print("  speech %d..%d  (%.0f..%.0f ms, %.0f ms long)"
                  % (span[0], span[1], span[0] / 16.0, span[1] / 16.0,
                     (span[1] - span[0]) / 16.0))
    return 0


if __name__ == "__main__":
    _sys.exit(main(_sys.argv))
