#!/usr/bin/env python3
"""Generate bit-exactness fixtures for the device front end.

    python3 tools/make_fixtures.py [corpus_dir]     # writes src/speech_fixtures.py
    python3 tools/make_fixtures.py --verify         # re-check the emitted file

The device port must produce **identical** features and distances, not close
ones. Drift does not announce itself: it is absorbed into the DTW distances and
comes out as "the recogniser is a bit poor", which is indistinguishable from
bad enrolment, a channel mismatch, or a threshold that wants tuning. By the
time anyone suspects the front end, four other things have been changed.

So the fixtures pin **every stage, not just the output**. A mismatch at
`cepstra` with `mel` matching is a DCT bug; a mismatch at `mel` with `mag`
matching is a filterbank bug; a mismatch at `fft` alone is a twiddle or a
scaling bug. Without the intermediate stages all of those present identically,
and finding which one it is means bisecting a 42000-operation pipeline by hand.

## The overflow case is the important one

CPython's ints are unbounded, so the host cannot notice a value that would wrap
on the device, and viper's `int` is a 32-bit machine word that wraps **without
raising**. `mfcc.CHECK_BOUNDS` catches this on the host; nothing catches it on
the device. So one fixture is a deliberately hostile full-scale signal chosen to
drive the FFT twiddle product toward its int32 bound, and every case records the
peak int32 magnitude reached at each stage. If the device disagrees on that
case and agrees on the others, the answer is an overflow, and the recorded peaks
say which stage to look at.

## Checksums

Whole intermediate arrays would be 512 complex int32s per frame, so the FFT and
magnitude stages are pinned by checksum:

    acc = 0
    for v in values:
        acc = (acc * 31 + v) & 0x3FFFFFFF

Order-sensitive, stays inside int32 with no sign games, and transcribes to
viper in three lines. Mel, cepstra and features are small enough to store whole,
and those are the stages where knowing *which* coefficient is wrong is worth
the bytes.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import mfcc  # noqa: E402
import vad   # noqa: E402

FIXTURE_FRAMES = 5
FIXTURE_SAMPLES = mfcc.FRAME_LEN + (FIXTURE_FRAMES - 1) * mfcc.FRAME_STRIDE
OUT_NAME = "speech_fixtures"


def checksum(values):
    acc = 0
    for v in values:
        acc = (acc * 31 + v) & 0x3FFFFFFF
    return acc


def _lcg(seed):
    s = seed & 0x7FFFFFFF or 1
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s


def case_speech(corpus):
    """A real endpointed utterance, truncated. The ordinary path."""
    if not corpus:
        return None
    d = os.path.join(corpus, "enrol")
    if not os.path.isdir(d):
        return None
    for fn in sorted(os.listdir(d)):
        if fn.startswith("mother."):
            samples = vad.trim(vad.read_wav(os.path.join(d, fn)))
            if samples is not None and len(samples) >= FIXTURE_SAMPLES:
                return list(samples[:FIXTURE_SAMPLES])
    return None


def case_tw_stress(corpus):
    """The real frames that come closest to the FFT twiddle's int32 bound.

    Found by searching the enrolment set: `husband.1` frame 31 drives
    `wr*lr - wi*li` to 1071611968, which is **99.8% of the provable bound** of
    32767.71 * 32767 = 1.0737e9. Nothing synthetic came close -- alternating
    full scale reaches only 41.5%, because pre-emphasis saturation clamps it
    before the FFT ever sees it.

    That is the case worth handing a viper port. If int32 wraps anywhere, it
    wraps here first, and it takes ordinary loud speech to get there.
    """
    if not corpus:
        return None
    path = os.path.join(corpus, "enrol", "husband.1.wav")
    if not os.path.exists(path):
        return None
    samples = vad.trim(vad.read_wav(path))
    if samples is None:
        return None
    # Frames 27..31, so the worst frame is the last one analysed.
    start = 27 * mfcc.FRAME_STRIDE
    if len(samples) < start + FIXTURE_SAMPLES:
        return None
    return list(samples[start:start + FIXTURE_SAMPLES])


def case_fullscale():
    """Hostile. Drives pre-emphasis into saturation and the FFT toward its bound.

    Alternating full scale is the worst case the overflow proof in
    docs/speech.md is written against: it is what makes the un-saturated
    pre-emphasis output reach 64551, and it puts the most energy possible into
    the top FFT bins. Not speech, and not meant to be.
    """
    out = []
    for i in range(FIXTURE_SAMPLES):
        out.append(32767 if (i & 1) else -32768)
    return out


def case_quiet():
    """Amplitude a few LSBs above nothing, to exercise the block-float shift.

    `g` reaches its maximum here, which is the path where the `2 - g` log
    correction earns its keep and where an off-by-one in the shift is invisible
    on any louder signal.
    """
    rng = _lcg(4321)
    out = []
    for _ in range(FIXTURE_SAMPLES):
        out.append((next(rng) >> 16) % 9 - 4)
    return out


def case_silence():
    """Digital zero. The log floor and the g == 0 branch."""
    return [0] * FIXTURE_SAMPLES


def stages_for(samples):
    """Run the pipeline, recording what each stage produced.

    Deliberately re-implements the frame loop rather than calling `mfcc()`, so
    the intermediates can be captured. It must therefore stay in step with
    `mfcc.mfcc_q8`; `--verify` checks the final features against the real
    function, which is what catches it drifting.
    """
    import array

    t = mfcc.tables()
    mfcc._peak.clear()
    was = mfcc.CHECK_BOUNDS
    mfcc.CHECK_BOUNDS = True
    try:
        buf = array.array("h", bytes(2 * len(samples)))
        for i, v in enumerate(samples):
            buf[i] = v
        pre = mfcc.preemphasise(buf)
        saturated = sum(1 for i in range(len(pre)) if pre[i] in (32767, -32768))

        work = mfcc.new_work()
        re, im, mag, mel = work
        frame = array.array("h", bytes(2 * mfcc.FRAME_LEN))
        for n in range(mfcc.FRAME_LEN):
            frame[n] = pre[n]

        # Frame 0, stage by stage, mirroring mfcc.mfcc_frame.
        win = t.window
        peak = 0
        for n in range(mfcc.FRAME_LEN):
            v = (frame[n] * win[n] + 16384) >> 15
            re[n] = v
            a = v if v >= 0 else -v
            if a > peak:
                peak = a
        for n in range(mfcc.FRAME_LEN, mfcc.FFT_SIZE):
            re[n] = 0
        for n in range(mfcc.FFT_SIZE):
            im[n] = 0
        g = 0 if peak == 0 else max(0, 14 - (peak.bit_length() - 1))
        if g:
            for n in range(mfcc.FRAME_LEN):
                re[n] = re[n] << g
        windowed = checksum([re[n] for n in range(mfcc.FFT_SIZE)])

        brev = t.brev
        for n in range(mfcc.FFT_SIZE):
            b = brev[n]
            if b > n:
                re[n], re[b] = re[b], re[n]
        mfcc.fft512(re, im, t)
        fft_sum = checksum([re[n] for n in range(mfcc.FFT_SIZE)]
                           + [im[n] for n in range(mfcc.FFT_SIZE)])

        mags = []
        for k in range(mfcc.N_BINS):
            p = re[k] * re[k] + im[k] * im[k]
            m = mfcc.isqrt(p)
            if p - m * m > m:
                m += 1
            mags.append(m)
        mag_sum = checksum(mags)

        cepstra = mfcc.mfcc_frame(frame, t, mfcc.new_work())
        # mfcc_frame overwrites its own work arrays; re-derive mel from a
        # clean run so the stored values are the ones it actually used.
        mel_vals = _mel_of_frame(frame, t)

        q8 = mfcc.mfcc_q8(buf)
        feats = mfcc.features_from_q8(q8)
        peaks = dict(mfcc._peak)
    finally:
        mfcc.CHECK_BOUNDS = was

    return {
        "saturated": saturated,
        "preemph": checksum([pre[i] for i in range(len(pre))]),
        "peak": peak,
        "g": g,
        "windowed": windowed,
        "fft": fft_sum,
        "mag": mag_sum,
        "mel": mel_vals,
        "cepstra": cepstra,
        "features": feats,
        "peaks": peaks,
    }


def _mel_of_frame(frame, t):
    """The 26 Q8 log2 mel values for one frame, by re-running the front half."""
    import array

    re = array.array("i", bytes(4 * mfcc.FFT_SIZE))
    im = array.array("i", bytes(4 * mfcc.FFT_SIZE))
    win = t.window
    peak = 0
    for n in range(mfcc.FRAME_LEN):
        v = (frame[n] * win[n] + 16384) >> 15
        re[n] = v
        a = v if v >= 0 else -v
        if a > peak:
            peak = a
    g = 0 if peak == 0 else max(0, 14 - (peak.bit_length() - 1))
    if g:
        for n in range(mfcc.FRAME_LEN):
            re[n] = re[n] << g
    for n in range(mfcc.FFT_SIZE):
        b = t.brev[n]
        if b > n:
            re[n], re[b] = re[b], re[n]
    mfcc.fft512(re, im, t)

    mags = []
    for k in range(mfcc.N_BINS):
        p = re[k] * re[k] + im[k] * im[k]
        m = mfcc.isqrt(p)
        if p - m * m > m:
            m += 1
        mags.append(m)

    out = []
    ofs = 0
    for i in range(mfcc.N_MEL):
        s, n = t.mel_start[i], t.mel_len[i]
        acc = 0
        for k in range(n):
            acc += (mags[s + k] * t.mel_w[ofs + k] + 128) >> 8
        ofs += n
        out.append(mfcc.log2_q8(acc, t) + ((2 - g) << mfcc.LOG_Q))
    return out


def _pcm(values):
    out = bytearray()
    for v in values:
        u = v & 0xFFFF
        out.append(u & 0xFF)
        out.append(u >> 8)
    return bytes(out)


def _blob_literal(data, width=20, indent="        "):
    lines = []
    for i in range(0, len(data), width):
        lines.append(indent + "b'"
                     + "".join("\\x%02x" % b for b in data[i:i + width]) + "'")
    return "\n".join(lines)


def build(corpus):
    cases = []
    speech = case_speech(corpus)
    if speech is not None:
        cases.append(("speech", "a real endpointed MOTHER, first %d frames"
                      % FIXTURE_FRAMES, speech))
    stress = case_tw_stress(corpus)
    if stress is not None:
        cases.append(("tw_stress", "real speech at 99.8% of the FFT twiddle "
                                   "bound; the overflow case that matters",
                      stress))
    cases.append(("fullscale", "alternating full scale; saturation clamps it, "
                               "so it is NOT the worst case", case_fullscale()))
    cases.append(("quiet", "a few LSBs; maximum block-float shift",
                  case_quiet()))
    cases.append(("silence", "digital zero; log floor and the g == 0 branch",
                  case_silence()))
    return [(name, why, samples, stages_for(samples))
            for name, why, samples in cases]


def emit(cases, path):
    lines = ['"""Bit-exactness fixtures for the speech front end. Generated.',
             "",
             "Produced by tools/make_fixtures.py. The device port must",
             "reproduce every value here exactly -- see docs/speech.md.",
             "",
             "Each case is (name, why, pcm, expected). `pcm` is little-endian",
             "int16. `expected` pins the pipeline stage by stage so a mismatch",
             "says which stage broke instead of only that something did.",
             "",
             "Checksums are:  acc = (acc * 31 + v) & 0x3FFFFFFF",
             "",
             "`peaks` is the largest magnitude each guarded expression reached",
             "on the host. viper's int is a 32-bit machine word and wraps",
             "without raising, so a device that disagrees only on the",
             "fullscale case has an overflow, and these say where.",
             '"""',
             "",
             "FORMAT = %d" % mfcc.TEMPLATE_FORMAT,
             "FRAMES = %d" % FIXTURE_FRAMES,
             "SAMPLES = %d" % FIXTURE_SAMPLES,
             "N_MEL = %d" % mfcc.N_MEL,
             "N_CEPS = %d" % mfcc.N_CEPS,
             "N_FEAT = %d" % mfcc.n_feat(),
             "",
             "",
             "def checksum(values):",
             "    acc = 0",
             "    for v in values:",
             "        acc = (acc * 31 + v) & 0x3FFFFFFF",
             "    return acc",
             "",
             "",
             "CASES = ("]
    for name, why, samples, st in cases:
        lines.append("    (%r, %r," % (name, why))
        lines.append("     (")
        lines.append(_blob_literal(_pcm(samples)))
        lines.append("     ),")
        lines.append("     {")
        lines.append("      'saturated': %d," % st["saturated"])
        lines.append("      'preemph': %d," % st["preemph"])
        lines.append("      'peak': %d," % st["peak"])
        lines.append("      'g': %d," % st["g"])
        lines.append("      'windowed': %d," % st["windowed"])
        lines.append("      'fft': %d," % st["fft"])
        lines.append("      'mag': %d," % st["mag"])
        lines.append("      'mel': %r," % (tuple(st["mel"]),))
        lines.append("      'cepstra': %r," % (tuple(st["cepstra"]),))
        lines.append("      'features': %r," % (tuple(tuple(r)
                                                      for r in st["features"]),))
        lines.append("      'peaks': %r," % (dict(sorted(st["peaks"].items())),))
        lines.append("     }),")
    lines.append(")")
    lines.append("")

    # A DTW distance too, so the matcher is pinned as well as the front end.
    import dtw as dtw_mod
    if len(cases) >= 2:
        a = cases[0][3]["features"]
        b = cases[1][3]["features"]
        lines.append("# dtw(CASES[0].features, CASES[1].features) with the")
        lines.append("# shipped band and step pattern. Pins the matcher, not")
        lines.append("# just the front end.")
        lines.append("DTW_0_1 = %d" % dtw_mod.dtw(a, b))
        lines.append("DTW_BAND = %d" % dtw_mod.BAND)
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def verify(path, corpus):
    """Re-derive everything and compare against what is on disk."""
    # Source text, not importlib: this is a staleness check and importlib
    # consults __pycache__, so it could be answered by the stale state it
    # exists to detect. See mfcc._load_source.
    fx = mfcc._load_source(path, "_fx")

    fails = []
    print("%d cases in %s" % (len(fx.CASES), path))
    for name, why, pcm, expected in fx.CASES:
        samples = []
        for i in range(0, len(pcm), 2):
            v = pcm[i] | (pcm[i + 1] << 8)
            samples.append(v - 65536 if v > 32767 else v)
        st = stages_for(samples)
        bad = []
        for key in ("saturated", "preemph", "peak", "g", "windowed", "fft",
                    "mag"):
            if st[key] != expected[key]:
                bad.append("%s (%r != %r)" % (key, st[key], expected[key]))
        if tuple(st["mel"]) != tuple(expected["mel"]):
            bad.append("mel")
        if tuple(st["cepstra"]) != tuple(expected["cepstra"]):
            bad.append("cepstra")
        if tuple(tuple(r) for r in st["features"]) != tuple(
                tuple(r) for r in expected["features"]):
            bad.append("features")

        # The independent check that matters: the stage-by-stage re-run above
        # shares code with the emitter, so it would agree with itself even if
        # both were wrong. mfcc.mfcc() does not.
        import array
        buf = array.array("h", bytes(2 * len(samples)))
        for i, v in enumerate(samples):
            buf[i] = v
        direct = mfcc.mfcc(buf)
        if tuple(tuple(r) for r in direct) != tuple(
                tuple(r) for r in expected["features"]):
            bad.append("features vs mfcc.mfcc()")

        print("  %-4s %-10s %s%s" % ("FAIL" if bad else "ok", name, why,
                                     "\n       " + "; ".join(bad) if bad else ""))
        if bad:
            fails.append(name)
        peak = max(expected["peaks"].values()) if expected["peaks"] else 0
        print("       peak int32 use %d (%.1f%%), pre-emphasis saturated %d"
              % (peak, 100.0 * peak / (2 ** 31 - 1), expected["saturated"]))
    if fails:
        print("\nFAILED: %s" % ", ".join(fails))
        return 1
    print("\nall fixtures reproduce exactly")
    return 0


def main(argv):
    path = os.path.join(HERE, "..", "src", OUT_NAME + ".py")
    if "--verify" in argv:
        corpus = None
        for a in argv[1:]:
            if not a.startswith("--"):
                corpus = a
        return verify(path, corpus)
    corpus = None
    for a in argv[1:]:
        if not a.startswith("--"):
            corpus = a
    cases = build(corpus)
    emit(cases, path)
    print("wrote %s (%d bytes, %d cases)"
          % (path, os.path.getsize(path), len(cases)))
    if corpus is None:
        print("  (no corpus given, so no real-speech case)")
    return verify(path, corpus)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
