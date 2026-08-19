#!/usr/bin/env python3
"""Prove the device patch path carries no state between utterances.

    python3 tools/test_si_patch_stateless.py

Written to answer one live observation: the same word said three times in a row
scored monotonically worse each turn (mother 0.352 -> unknown 0.605 -> unknown
0.977). Monotonic decay across near-identical inputs is the signature of state
accumulating somewhere, and `si_spot.Spotter` binds its buffers **once** at
start-up and reuses them for every turn, so the buffers are the obvious suspect.

This does not argue about whether they are reused correctly. It reuses them, the
way the live path does, and compares bytes.

Three questions, and the second is the one that matters:

1. **Repetition.** The same samples through the same buffers three times must
   give three identical patches. Anything that accumulates -- a running mean, a
   counter, an un-cleared row -- shows up here.

2. **A long utterance followed by a short one.** This is the dangerous shape and
   the reason the test exists. `logmel_rows` writes only the first `n_frames`
   rows of a buffer sized for 180, so a short word leaves the tail of a previous
   long word in `rows`. If `normalise_into` read past `n_frames` -- when
   computing a per-band mean, or when centre-padding 30 live frames out to 80 --
   it would pad with the previous word instead of with silence, and the model
   would see a chimera. The patch after a long word must equal the patch that
   same short word produces on buffers that have never been used.

3. **A different word in between.** Same as (2) but with the content changed as
   well as the length, in case anything keys on shape rather than values.

If all three pass, the patch path is not where the decay comes from, and the
search moves to the capture path -- `listen.Recorder`'s buffer and
`vad.endpoints`, which have their own per-turn lifecycle -- or to the audio
genuinely differing turn to turn.

A passing result here is evidence about **this** module only. It cannot say
anything about the codec, the recorder, or the model's own internals.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from array import array   # noqa: E402
import math               # noqa: E402
import random             # noqa: E402

import si_patch           # noqa: E402
import mfcc               # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    if ok:
        print("  ok    %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


def samples_for_frames(n_frames):
    return mfcc.FRAME_LEN + (n_frames - 1) * mfcc.FRAME_STRIDE


def word(n_frames, freq, amp=7000, seed=0):
    """A crude but deterministic stand-in for an utterance."""
    rng = random.Random(seed)
    n = samples_for_frames(n_frames)
    out = array("h", bytes(2 * n))
    for t in range(n):
        env = 0.4 + 0.6 * math.sin(math.pi * t / n)          # a rough syllable
        v = amp * env * math.sin(2 * math.pi * freq * t / 16000)
        v += rng.randint(-250, 250)
        out[t] = max(-32768, min(32767, int(v)))
    return out


def run(samples, buffers):
    patch, n_frames, clipped = si_patch.patch_for(samples, 0, len(samples), buffers)
    return (list(patch) if patch is not None else None), n_frames, clipped


def first_difference(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return "first differs at frame %d band %d: %d vs %d" % (
                i // si_patch.N_BANDS, i % si_patch.N_BANDS, a[i], b[i])
    return "lengths %d vs %d" % (len(a), len(b))


def test_repetition():
    """The same input, three times, through one set of buffers."""
    print("the same utterance three times through one set of buffers")
    bufs = si_patch.allocate()
    spoken = word(44, 320, seed=1)
    got = [run(spoken, bufs) for _ in range(3)]
    base = got[0][0]
    for i in (1, 2):
        check("run %d identical to run 1" % (i + 1), got[i][0] == base,
              first_difference(base, got[i][0]))
        check("run %d same frame count" % (i + 1), got[i][1] == got[0][1])
        check("run %d same clip count" % (i + 1), got[i][2] == got[0][2])


def test_long_then_short():
    """The shape that would leave a previous word in the padding."""
    print()
    print("a long utterance, then a short one, through the same buffers")
    long_word = word(120, 260, seed=2)     # more rows than a patch holds
    short_word = word(30, 320, seed=3)     # far fewer, so it is centre-padded

    dirty = si_patch.allocate()
    run(long_word, dirty)                  # leaves 120 rows behind
    after_long, n_after, clip_after = run(short_word, dirty)

    clean = si_patch.allocate()
    fresh, n_fresh, clip_fresh = run(short_word, clean)

    check("short word after a long one equals the same word on fresh buffers",
          after_long == fresh, first_difference(fresh, after_long))
    check("frame count unaffected", n_after == n_fresh,
          "%d vs %d" % (n_after, n_fresh))
    check("clip count unaffected", clip_after == clip_fresh,
          "%d vs %d" % (clip_after, clip_fresh))

    # And the padding really is the biased zero, not the tail of the long word.
    pad_rows = (si_patch.N_FRAMES - n_fresh) // 2
    if pad_rows > 0:
        head = after_long[:pad_rows * si_patch.N_BANDS]
        check("the %d padding rows are all 128" % pad_rows, set(head) == {128},
              "distinct values in the padding: %s" % sorted(set(head))[:8])
    else:
        print("  --    utterance was not short enough to pad; padding unchecked")


def test_alternating():
    """Different content as well as different length, three turns running."""
    print()
    print("alternating words, to catch anything keyed on shape not values")
    a = word(44, 300, seed=4)
    b = word(70, 500, seed=5)

    bufs = si_patch.allocate()
    run(b, bufs)
    a_after_b, _, _ = run(a, bufs)
    run(b, bufs)
    a_after_b2, _, _ = run(a, bufs)

    clean = si_patch.allocate()
    a_clean, _, _ = run(a, clean)

    check("A after B equals A on fresh buffers", a_after_b == a_clean,
          first_difference(a_clean, a_after_b))
    check("A after B twice is still identical", a_after_b2 == a_clean,
          first_difference(a_clean, a_after_b2))


def test_every_patch_byte_written():
    """No byte of the patch may survive from the previous call.

    Stronger than the comparisons above and independent of them: fill the patch
    with a sentinel by hand, run, and assert the sentinel is gone everywhere.
    A comparison test can pass while both runs share the same stale byte; this
    cannot.
    """
    print()
    print("every byte of the patch is written on every call")
    bufs = si_patch.allocate()
    work, rows, patch, _pre = bufs
    spoken = word(30, 320, seed=6)
    for sentinel in (0x5A, 0xA5):
        for i in range(len(patch)):
            patch[i] = sentinel
        si_patch.patch_for(spoken, 0, len(spoken), bufs)
        left = sum(1 for i in range(len(patch)) if patch[i] == sentinel)
        # The sentinel can legitimately reappear as a real value, so the test is
        # that the patch is not *mostly* sentinel and that it matches a clean run.
        clean = si_patch.allocate()
        want, _, _ = run(spoken, clean)
        check("sentinel 0x%02X: patch matches a clean run" % sentinel,
              list(patch) == want, first_difference(want, list(patch)))
        print("        (%d of %d bytes coincidentally equal the sentinel)"
              % (left, len(patch)))


def test_no_allocation_per_turn():
    """The live path must not allocate per turn, and must survive the longest
    utterance the endpointer will ever hand it.

    This is a regression test with a date on it. `logmel_rows` used to build its
    pre-emphasis buffer per call, sized by utterance length, and `si_spot` never
    passed one -- so a 1.8 s hold asked for 57601 bytes and the turn died with a
    MemoryError on the first press after a clean reset. Worse, the form used
    (`array("h", bytes(n))`) holds both objects at once and so peaked at twice
    that, per turn, on a heap that never compacts.

    Counting allocations portably is awkward, so this checks the two properties
    that matter instead: the pooled buffer is big enough for the longest
    accepted utterance, and a maximal utterance produces a patch without the
    fallback path being reached.
    """
    print()
    print("no per-turn allocation, and the longest utterance fits")
    bufs = si_patch.allocate()
    check("allocate() returns a pooled pre-emphasis buffer", len(bufs) == 4)
    pooled = bufs[3]
    check("pooled buffer is one frame plus history (%d samples)" % si_patch.PRE_SAMPLES,
          len(pooled) == si_patch.PRE_SAMPLES, "have %d" % len(pooled))
    # The point of the redesign: the buffer must NOT scale with utterance
    # length. A 58 KB one fitted the arithmetic and would not fit the heap.
    check("pooled buffer is small enough to be ordering-independent",
          2 * len(pooled) < 4096, "%d bytes" % (2 * len(pooled)))

    # The longest utterance src/vad.py will accept: 180 frames, 1.8 s.
    longest = word(si_patch.MAX_ROWS, 300, seed=7)
    patch, n_frames, _ = si_patch.patch_for(longest, 0, len(longest), bufs)
    check("a %d-frame utterance produces a patch" % si_patch.MAX_ROWS,
          patch is not None and len(patch) == si_patch.PATCH_BYTES)
    check("frame count clamped to MAX_ROWS", n_frames == si_patch.MAX_ROWS,
          "got %d" % n_frames)

    # And one longer still, which the VAD would reject but a caller might not.
    over = word(si_patch.MAX_ROWS + 40, 300, seed=8)
    patch2, n2, _ = si_patch.patch_for(over, 0, len(over), bufs)
    check("an over-long utterance is clamped rather than overrunning",
          patch2 is not None and n2 == si_patch.MAX_ROWS, "got %s" % n2)


def main():
    test_repetition()
    test_long_then_short()
    test_alternating()
    test_every_patch_byte_written()
    test_no_allocation_per_turn()
    print()
    if FAILURES:
        print("%d FAILED -- the patch path DOES carry state between turns" % len(FAILURES))
        for name in FAILURES:
            print("  %s" % name)
        return 1
    print("all passed -- no state survives a turn in si_patch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
