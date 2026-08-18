#!/usr/bin/env python3
"""Prove the statics-only template format reconstructs exactly, on the host.

    python3 tools/test_templates.py [corpus_dir]

A statics-only template stores 12 values per frame and the device rebuilds the
other 12 by expanding the buffer in place, so it never holds the
74 KB source and the 148 KB result at the same time. That saves half the flash
and half the `mpremote` transfer, and it is exactly the kind of pointer
arithmetic that is easy to get subtly wrong in a way no runtime check would
catch -- a template that is 2 LSBs off in four frames still matches, just
slightly worse, forever.

So it is checked here, against `mfcc()` itself, on real audio, before it is
ever run on a board.

Two things this pins in particular:

- **The overlap boundary.** The expansion writes into the same buffer it reads
  statics from, and deltas reach two frames either side, so a one- or two-frame
  template is where an off-by-one shows. A test using only 40-frame templates
  would never reach it, so short ones are deliberate.
- **Template boundaries.** Deltas replicate at each template's own edges and
  must not read into the neighbour, which only multi-template blobs catch.
- **Q8 storage, not Q4.** Deltas are regressions over the statics, and
  computing them from the Q4 features quantises twice. The blob therefore holds
  Q8 and the reconstruction shifts afterwards. If someone "simplifies" that to
  store Q4, this test fails loudly rather than the recogniser getting quietly
  worse.
"""

import array
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import mfcc  # noqa: E402
import vad   # noqa: E402

_fails = []


def check(name, ok, detail=""):
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", name,
                           "" if ok else "  " + detail))
    if not ok:
        _fails.append(name)


def roundtrip(q8):
    """One template: pack statics, expand, unpack."""
    return roundtrip_many([q8])[0], roundtrip_many.clamped


def roundtrip_many(templates):
    """Several templates in one blob, the way the device actually holds them.

    Multi-template is the case worth testing: deltas must not bleed across a
    template boundary, and the expansion writes into the same buffer it is
    reading statics from.
    """
    blob = bytearray()
    index = []
    frame_off = 0
    clamped = 0
    for k, q8 in enumerate(templates):
        packed, n_clamped = mfcc.pack_statics(q8)
        clamped += n_clamped
        blob += packed
        index.append(("t%d" % k, frame_off, len(q8)))
        frame_off += len(q8)
    roundtrip_many.clamped = clamped

    buf = bytearray(2 * mfcc.n_feat() * frame_off)
    buf[0:len(blob)] = blob
    mfcc.expand_all(buf, index)

    out = []
    for _label, off, n in index:
        start = 2 * mfcc.n_feat() * off
        end = start + 2 * mfcc.n_feat() * n
        out.append(mfcc.unpack_template(bytes(buf[start:end])))
    return out


roundtrip_many.clamped = 0


def compare(q8, label):
    want = mfcc.features_from_q8(q8)
    got, clamped = roundtrip(q8)
    if clamped:
        check(label, False, "%d values clamped at int16" % clamped)
        return
    if len(got) != len(want):
        check(label, False, "%d frames back, wanted %d" % (len(got), len(want)))
        return
    worst = 0
    at = None
    for f in range(len(want)):
        for j in range(len(want[f])):
            d = abs(got[f][j] - want[f][j])
            if d > worst:
                worst, at = d, (f, j)
    check(label, worst == 0,
          "worst difference %d at frame %d coeff %d" % (worst, at[0], at[1])
          if at else "")


def synthetic(n_frames, seed=7):
    """Q8 static rows with a wide dynamic range.

    Short lengths are the point: deltas reach two frames either side, so a
    one- or two-frame template is where an off-by-one in the expansion shows,
    and a 40-frame one never reaches it.
    """
    s = seed
    rows = []
    for _ in range(n_frames):
        row = []
        for _ in range(mfcc.N_CEPS):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            row.append((s >> 11) % 24001 - 12000)
        rows.append(row)
    return rows


def main(argv):
    corpus = argv[1] if len(argv) > 1 else None
    print("statics-only template format (TEMPLATE_FORMAT %d, %d features/frame)"
          % (mfcc.TEMPLATE_FORMAT, mfcc.n_feat()))

    print("\nsynthetic, short templates (the expansion overlap boundary)")
    for n in (1, 2, 3, 4, 5, 6, 9, 17, 40):
        compare(synthetic(n), "%d frame%s" % (n, "" if n == 1 else "s"))

    print("\nseveral templates in one blob (deltas must not cross a boundary)")
    for shape in ((3, 40), (40, 3), (1, 1, 1), (5, 61, 2, 28), (61, 61, 61)):
        group = [synthetic(n, seed=100 + n) for n in shape]
        want = [mfcc.features_from_q8(q8) for q8 in group]
        got = roundtrip_many(group)
        ok = got == want
        detail = ""
        if not ok:
            for gi, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    detail = "template %d of %s differs" % (gi, shape)
                    break
        check("shape %s" % (shape,), ok, detail)

    print("\nascending order is not assumed to be the only safe one")
    group = [synthetic(n, seed=7 * n) for n in (2, 3, 4)]
    check("short-first blob", roundtrip_many(group)
          == [mfcc.features_from_q8(q) for q in group])

    print("\nextreme values (the int16 clamp must not fire below the limit)")
    edge = [[32767 if (f + j) % 2 else -32768 for j in range(mfcc.N_CEPS)]
            for f in range(8)]
    got, clamped = roundtrip(edge)
    check("full-scale Q8 statics survive", clamped == 0,
          "%d clamped" % clamped)
    compare([[v // 2 for v in row] for row in edge], "half-scale alternating")

    if corpus:
        print("\nreal audio from %s" % corpus)
        n = 0
        worst_seen = 0
        for sub in ("enrol", "test"):
            d = os.path.join(corpus, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".wav"):
                    continue
                samples = vad.trim(vad.read_wav(os.path.join(d, fn)))
                if samples is None:
                    continue
                q8 = mfcc.mfcc_q8(samples)
                if not q8:
                    continue
                want = mfcc.features_from_q8(q8)
                got, clamped = roundtrip(q8)
                if clamped or got != want:
                    check(fn, False, "clamped=%d" % clamped)
                n += 1
                for row in q8:
                    for v in row:
                        worst_seen = max(worst_seen, abs(v))
        check("%d corpus templates reconstruct exactly" % n, not _fails)
        print("       largest |Q8 static| seen: %d of 32767 (%.1fx headroom)"
              % (worst_seen, 32767.0 / worst_seen if worst_seen else 0))
        print("       a clamp here would mean the recording was loud enough to")
        print("       overflow Q8; pack_statics returns the count so callers")
        print("       cannot miss it.")
    else:
        print("\n(pass a corpus directory to also check real audio)")

    print()
    if _fails:
        print("FAILED: %s" % ", ".join(_fails))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
