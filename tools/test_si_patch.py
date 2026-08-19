#!/usr/bin/env python3
"""Prove `src/si_patch.py` reproduces `tools/si_features.py` exactly.

    python3 tools/test_si_patch.py [corpus_dir]

The device builds the model's input; the host built the model's training data.
If the two disagree the model sees, at run time, something it was never trained
on -- and the failure is not a crash but a recogniser that is mysteriously worse
than the evaluation said it would be. That is the single most expensive failure
shape in this project's history, and `docs/speech.md` exists largely because of
it.

So this asserts **byte-identical patches**, not close ones, over real audio and
over the shapes where an off-by-one hides.

## What is actually at risk

The tap itself is not. `si_patch.logmel_rows` calls `spotter._mfcc_at` and reads
`work[3]`, and those values are already pinned bit-exact against
`src/speech_fixtures.py`. What is new, and therefore what this file is really
about, is the arithmetic bolted on after the tap:

- **the per-band mean**, whose truncation must be toward zero and not floor.
  They differ only when the total is negative, which happens in exactly the
  quiet bands, and a floor there is off by one -- a 1/16-octave error in a
  feature, on some bands, in some utterances. Nothing would ever look wrong.
- **the mean's window**: over every endpointed frame, *before* cropping. Taking
  it over the cropped window would be the natural way to write it and would
  make normalisation depend on the crop.
- **centre crop and centre pad**, where the halving is `//` on both sides and an
  odd remainder has to land the same way.
- **the +128 uint8 bias**, including on the pad value, which must be the biased
  zero and not zero.

## The corpus is optional and the synthetic cases are not

Real audio exercises the values; the constructed cases exercise the boundaries.
A 79-frame and an 81-frame utterance are one row either side of the crop/pad
branch, and a 1-frame one makes the mean equal the sample. Those are where the
`//` rounding shows, and no corpus is guaranteed to contain them.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

# src/si_patch.py imports `spotter`, which on the host falls back to the plain
# path because the viper block raises ImportError. That is the intended host
# behaviour and is what makes this comparison meaningful: the plain path is the
# specification and the device proves itself against the fixtures separately.
import si_patch      # noqa: E402
import si_features   # noqa: E402
import mfcc          # noqa: E402

try:
    import vad as _hostvad_unused  # noqa: F401
except ImportError:
    pass

FAILURES = []


def check(name, ok, detail=""):
    if ok:
        print("  ok    %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


def device_patch(samples):
    """Run the device path over a plain list of int16, as the board would."""
    from array import array
    buf = array("h", samples)
    buffers = si_patch.allocate()
    patch, n_frames, clipped = si_patch.patch_for(buf, 0, len(buf), buffers)
    if patch is None:
        return None, 0, 0
    return list(patch), n_frames, clipped


def host_patch(samples):
    """Run the host path, which is the specification."""
    rows = mfcc.logmel_q8(list(samples))
    if not rows:
        return None, 0, 0
    norm, clipped = si_features.normalise(rows)
    fitted = si_features.fit(norm)
    flat = []
    for row in fitted:
        for v in row:
            flat.append(v + 128)     # the device's uint8 transport
    return flat, len(rows), clipped


def compare(name, samples):
    dev, dev_n, dev_clip = device_patch(samples)
    host, host_n, host_clip = host_patch(samples)

    if host is None or dev is None:
        check("%s both reject" % name, host is None and dev is None,
              "device %r host %r" % (dev is None, host is None))
        return

    if len(dev) != len(host):
        check("%s length" % name, False, "device %d host %d" % (len(dev), len(host)))
        return

    diffs = [i for i in range(len(dev)) if dev[i] != host[i]]
    detail = ""
    if diffs:
        i = diffs[0]
        detail = "%d of %d differ; first at frame %d band %d: device %d host %d" % (
            len(diffs), len(dev), i // si_patch.N_BANDS, i % si_patch.N_BANDS,
            dev[i], host[i])
    check("%s patch identical (%d raw frames)" % (name, host_n), not diffs, detail)
    check("%s frame count" % name, dev_n == host_n,
          "device %d host %d" % (dev_n, host_n))
    check("%s clip count" % name, dev_clip == host_clip,
          "device %d host %d" % (dev_clip, host_clip))


# ---------------------------------------------------------------------------
# Synthetic signals, chosen for the boundaries rather than for realism
# ---------------------------------------------------------------------------

def tone(n_samples, freq=440, amp=8000, rate=16000):
    import math
    return [int(amp * math.sin(2 * math.pi * freq * t / rate)) for t in range(n_samples)]


def samples_for_frames(n_frames):
    """Exactly `n_frames` frames, per speech.md's `1 + (n - 400) // 160`."""
    return mfcc.FRAME_LEN + (n_frames - 1) * mfcc.FRAME_STRIDE


def test_synthetic():
    print("synthetic, at the crop/pad boundaries")
    import random
    rng = random.Random(20260818)

    cases = []
    # One frame either side of N_FRAMES, and N_FRAMES exactly: the three
    # branches of fit(), including the one where nothing happens.
    for n in (1, 2, 15, 79, 80, 81, 130):
        cases.append(("tone %d frames" % n, tone(samples_for_frames(n))))
    # Odd and even padding remainders land the word differently.
    for n in (77, 78):
        cases.append(("tone %d frames (odd pad)" % n, tone(samples_for_frames(n))))
    # Noise has no structure for a mean to cancel, and negative band means are
    # where truncation-toward-zero differs from floor.
    for n in (40, 80, 120):
        cases.append(("noise %d frames" % n,
                      [rng.randint(-6000, 6000) for _ in range(samples_for_frames(n))]))
    # Digital silence: every band at its floor, every mean equal to it, so the
    # normalised patch should be uniformly the biased zero.
    cases.append(("silence 80 frames", [0] * samples_for_frames(80)))
    # Full scale, where pre-emphasis saturation fires.
    cases.append(("fullscale 60 frames",
                  [32767 if t % 2 else -32768
                   for t in range(samples_for_frames(60))]))

    for name, samples in cases:
        compare(name, samples)


def test_silence_is_biased_zero():
    """Silence must normalise to exactly 128 everywhere, not to 0.

    Worth its own assertion because passing the *unbiased* value through is the
    natural bug, it survives the host comparison if the host has it too, and it
    would shift every input the model ever sees by half the int8 range.
    """
    print()
    print("silence normalises to the biased zero")
    dev, _, _ = device_patch([0] * samples_for_frames(80))
    check("all 128", dev is not None and set(dev) == {128},
          "distinct values: %s" % (sorted(set(dev))[:8] if dev else None))


def test_pad_value():
    """A short utterance's padding must also be the biased zero."""
    print()
    print("padding is the biased zero")
    dev, n, _ = device_patch(tone(samples_for_frames(20)))
    if dev is None:
        check("short utterance produced a patch", False)
        return
    pad_rows = (si_patch.N_FRAMES - n) // 2
    head = dev[:pad_rows * si_patch.N_BANDS]
    check("%d leading pad rows are 128" % pad_rows,
          set(head) == {128} if head else False,
          "distinct: %s" % sorted(set(head))[:8])


def test_corpus(corpus_dir):
    print()
    print("real audio from %s" % corpus_dir)
    import glob
    wavs = sorted(glob.glob(os.path.join(corpus_dir, "*.wav")))[:40]
    if not wavs:
        print("  (no wavs found; skipped)")
        return
    import vad as hostvad_mod  # noqa: F401
    sys.path.insert(0, HERE)
    import vad as toolvad
    n_done = 0
    for path in wavs:
        samples = toolvad.read_wav(path)
        trimmed = toolvad.trim(samples)
        if trimmed is None:
            continue
        compare(os.path.basename(path), list(trimmed))
        n_done += 1
    print("  compared %d of %d files (the rest were rejected by the VAD)"
          % (n_done, len(wavs)))


def main(argv):
    print("N_FRAMES=%d N_BANDS=%d INPUT_SHIFT=%d"
          % (si_patch.N_FRAMES, si_patch.N_BANDS, si_patch.INPUT_SHIFT))
    check("device and host agree on N_BANDS",
          si_patch.N_BANDS == si_features.N_BANDS)
    check("device and host agree on N_FRAMES",
          si_patch.N_FRAMES == si_features.N_FRAMES)
    check("device and host agree on INPUT_SHIFT",
          si_patch.INPUT_SHIFT == si_features.INPUT_SHIFT)
    print()

    test_synthetic()
    test_silence_is_biased_zero()
    test_pad_value()

    corpus = argv[1] if len(argv) > 1 else "takes"
    if os.path.isdir(corpus):
        test_corpus(corpus)

    print()
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for name in FAILURES:
            print("  %s" % name)
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
