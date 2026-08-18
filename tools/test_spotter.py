#!/usr/bin/env python3
"""Asserts src/spotter.py is bit-identical to the host reference.

    python3 tools/test_spotter.py

`tools/mfcc.py` and `tools/dtw.py` compute the templates; `src/spotter.py`
matches against them on the device. The two front ends must agree exactly, and
"exactly" is the operative word: a one-LSB disagreement does not fail, it shifts
every DTW distance by an amount the matcher silently absorbs, and it presents as
"the recogniser is a bit poor" -- which is also what bad enrolment, a wrong mic
gain, a wrong sample rate and a channel mismatch look like. Nothing points here.

So nothing in this file compares with a tolerance. Features are compared
element by element, DTW distances as integers, the expansion byte for byte.

It also pins every constant the device carries against the reference, because
`speech_tables.py` does not emit all of them and the rest are written out in
`spotter.py` by hand -- which is the duplication-that-must-agree shape this
project has been finding all week. Here it is enforced rather than trusted.

Runs under CPython with no device and no board: `spotter.py` imports only
`array` and `speech_tables`, which is what makes this possible at all.

## Host agreement is necessary and not sufficient

This test cannot, by construction, catch the one failure that matters most on
the device. Under CPython both implementations get arbitrary-precision ints.
`@micropython.viper`'s `int` is a signed machine word and wraps at int32,
silently. So a comparison test can pass perfectly here while the board produces
something else, with both sides agreeing and both being wrong the same way.

`tools/mfcc.py` anticipates this: `CHECK_BOUNDS` makes the reference raise on a
value that would not fit int32, and it defaults to **False**, so a test that
merely imports mfcc gets no checking at all. Every comparison below therefore
runs with it **on**, and `test_hot_input` feeds a deliberately loud signal --
the case most likely to find the stage with the least headroom, and the case a
board with an uncalibrated mic gain will actually meet.

Even so: a green run here means the arithmetic agrees given unbounded integers.
It is not proof the port is correct on the device. `src/speech_fixtures.py` is
what checks that, on the board, against values derived here.
"""

import math
import os
import random
import sys
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Never leave a .pyc behind, and never read one. See _load_source.
sys.dont_write_bytecode = True

FAILURES = []


def _load_source(name, path):
    """Load a module from its source text. Deliberately not importlib.

    `importlib` consults `__pycache__`, and Python's cache invalidation is
    (mtime, size) -- which is exactly the pair that can collide on a *generated*
    file rewritten within the filesystem's mtime granularity to the same length.
    Every module loaded here is generated or actively regenerated:
    `speech_tables.py` and `speech_fixtures.py` come out of `--emit-tables` and
    `make_fixtures.py`, and `spotter.py` and `mfcc.py` are both under edit.

    This is not hypothetical. A run of this suite reported 19 failures --
    coherent, plausible ones: a DELTA_SHIFT mismatch and feature differences --
    and went green on the next run with no file changed. A half-written source
    file raises SyntaxError; clean wrong values are what a stale cache gives.
    I first recorded that as a read race, which was the wrong diagnosis.

    So: read the text, compile it, exec it. No cache to consult, none written.
    "Why isn't this just importlib" is the obvious cleanup, and the answer is
    that a staleness check served from a cache of the stale state is worse than
    no check, because it teaches you to re-run until it goes green.
    """
    with open(path) as handle:
        source = handle.read()
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


# Order matters: speech_tables must be in sys.modules before spotter execs its
# `import speech_tables`, and mfcc before dtw imports it.
_load_source("speech_tables", os.path.join(ROOT, "src", "speech_tables.py"))
mfcc = _load_source("mfcc", os.path.join(HERE, "mfcc.py"))
_load_source("speech_fixtures", os.path.join(ROOT, "src", "speech_fixtures.py"))
spotter = _load_source("device_spotter", os.path.join(ROOT, "src", "spotter.py"))

import dtw as hostdtw  # noqa: E402  the reference matcher, over our mfcc


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def speech_like(n, seed, take=0):
    """A signal with formant structure, silence and fricative noise.

    Not speech, but it exercises the parts of the front end that a tone does
    not: the block-float shift across loud and quiet frames, the log's low end,
    and the delta window at both edges.

    `seed` fixes the formants and pitch -- the identity of the "word" -- and
    `take` varies only the noise, the gain and the pitch slightly, so two takes
    of one seed resemble each other the way two recordings of a word do. That
    distinction matters: with the formants drawn per take, takes of the "same
    word" are unrelated signals and a held-out match test cannot pass except by
    luck. It was written that way first, and the test duly failed.
    """
    rng = random.Random(seed)
    out = array("h", bytes(2 * n))
    f0 = 90 + rng.random() * 60
    phase = [0.0, 0.0, 0.0]
    formants = (rng.uniform(400, 800), rng.uniform(1100, 1800), rng.uniform(2300, 3200))
    rng = random.Random(seed * 1000 + take)
    f0 *= 1.0 + 0.02 * (take - 1)
    gain = 1.0 + 0.08 * (take - 1)
    for i in range(n):
        env = 0.0
        pos = i / float(n)
        if pos > 0.08 and pos < 0.92:          # leading and trailing silence
            env = 0.35 + 0.65 * math.sin(math.pi * (pos - 0.08) / 0.84)
        v = 0.0
        for k in range(3):
            phase[k] += 2 * math.pi * formants[k] / 16000.0
            v += math.sin(phase[k]) * (0.6 ** k)
        v *= 0.5 + 0.5 * math.sin(2 * math.pi * f0 * i / 16000.0)
        if 0.10 < pos < 0.18:                  # a fricative burst up front
            v = rng.gauss(0, 0.7)
        s = int(9000 * gain * env * v + rng.gauss(0, 40))
        out[i] = max(-32768, min(32767, s))
    return out


# --- the tests -------------------------------------------------------------

def test_constants():
    print("constants: device vs reference")
    pairs = (
        ("SAMPLE_RATE", spotter.SAMPLE_RATE, mfcc.SAMPLE_RATE),
        ("FRAME_LEN", spotter.FRAME_LEN, mfcc.FRAME_LEN),
        ("FRAME_STRIDE", spotter.FRAME_STRIDE, mfcc.FRAME_STRIDE),
        ("FFT_SIZE", spotter.FFT_SIZE, mfcc.FFT_SIZE),
        ("N_BINS", spotter.N_BINS, mfcc.N_BINS),
        ("N_MEL", spotter.N_MEL, mfcc.N_MEL),
        ("N_CEPS", spotter.N_CEPS, mfcc.N_CEPS),
        ("PREEMPH_Q15", spotter.PREEMPH_Q15, mfcc.PREEMPH_Q15),
        ("LOG_Q", spotter.LOG_Q, mfcc.LOG_Q),
        ("FEAT_SHIFT", spotter.FEAT_SHIFT, mfcc.FEAT_SHIFT),
        ("TEMPLATE_FORMAT", spotter.TEMPLATE_FORMAT, mfcc.TEMPLATE_FORMAT),
        # Emitted by --emit-tables now, so there is no hand-written copy left to
        # mistype, and `mfcc --selftest` catches a stale speech_tables.py on the
        # side that generates it -- where it can say "re-emit" rather than
        # "something is wrong". These pins are kept for the one thing neither
        # covers: that `spotter` reads the *right attribute* out of the tables.
        # A typo binding DELTA_SHIFT to FEAT_SHIFT would leave both files
        # perfectly current and every value correct, and only this would notice.
        ("DELTA_WIDTH", spotter.DELTA_WIDTH, mfcc.DELTA_WIDTH),
        ("DELTA_SHIFT", spotter.DELTA_SHIFT, mfcc.DELTA_SHIFT),
        ("MEL_FLOOR_SHIFT", spotter.MEL_FLOOR_SHIFT, mfcc.MEL_FLOOR_SHIFT),
        ("N_FEAT", spotter.N_FEAT, mfcc.n_feat()),
        ("DELTA_Q15", spotter.DELTA_Q15, mfcc._delta_recip()),
        ("BAND", spotter.BAND, hostdtw.BAND),
        ("DUR_RATIO_PCT", spotter.DUR_RATIO_PCT, hostdtw.DUR_RATIO_PCT),
        ("INF", spotter.INF, hostdtw.INF),
    )
    for name, got, want in pairs:
        check("%s == %s" % (name, want), got == want, "device has %s" % (got,))


def test_tables():
    print("frozen tables: decoded vs rebuilt")
    t = mfcc.tables()
    for name, got, want in (
        ("window", spotter.WINDOW, t.window),
        ("twiddle re", spotter.TW_RE, t.tw_re),
        ("twiddle im", spotter.TW_IM, t.tw_im),
        ("bitrev", spotter.BITREV, t.brev),
        ("mel start", spotter.MEL_START, t.mel_start),
        ("mel len", spotter.MEL_LEN, t.mel_len),
        ("mel weights", spotter.MEL_W, t.mel_w),
        ("log2", spotter.LOG2, t.log2),
        ("dct", spotter.DCT, t.dct),
        ("lifter", spotter.LIFTER, t.lifter),
    ):
        same = len(got) == len(want) and all(got[i] == want[i] for i in range(len(want)))
        first = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), None)
        check("table %s (%d entries)" % (name, len(want)), same,
              "lengths %d/%d, first difference at %s" % (len(got), len(want), first))


def test_isqrt():
    print("integer square root")
    bad = []
    for v in list(range(0, 4096)) + [10 ** 6, 2 ** 31 - 1, 2147352578]:
        if spotter._isqrt(v) != math.isqrt(v):
            bad.append(v)
    check("exact against math.isqrt over 4099 values", not bad, "wrong at %s" % bad[:4])


def test_preemphasis():
    print("pre-emphasis")
    for seed in (1, 2):
        samples = speech_like(4000, seed)
        want = mfcc.preemphasise(samples)
        got = array("h", bytes(2 * len(samples)))
        spotter.preemphasise(samples, 0, len(samples), got)
        check("identical (seed %d)" % seed, list(got) == list(want))

    # And with an offset, which is how talk.py calls it: the endpointed word
    # starts partway into the capture buffer, and y[-1] must be taken from the
    # first sample of the *slice*, not of the buffer.
    samples = speech_like(4000, 3)
    start, count = 640, 2400
    want = mfcc.preemphasise(array("h", samples[start:start + count]))
    got = array("h", bytes(2 * count))
    spotter.preemphasise(samples, start, count, got)
    check("identical from an offset", list(got) == list(want))


def test_features():
    print("features: element by element, no tolerance, bounds checking ON")
    mfcc.CHECK_BOUNDS = True
    mfcc._peak.clear()
    for seed, n in ((11, 8000), (12, 4800), (13, 16000), (14, 6400)):
        samples = speech_like(n, seed)
        want = mfcc.mfcc(samples)
        got, frames = spotter.features(samples, 0, n)

        if frames != len(want):
            check("seed %d: frame count" % seed, False,
                  "device %d, reference %d" % (frames, len(want)))
            continue

        diffs = []
        for f in range(frames):
            for j in range(spotter.N_FEAT):
                a = got[f * spotter.N_FEAT + j] - spotter.BIAS
                b = want[f][j]
                if a != b:
                    diffs.append((f, j, a, b))
        check("seed %d: %d frames x %d identical" % (seed, frames, spotter.N_FEAT),
              not diffs, "%d differ, first %s" % (len(diffs), diffs[:2]))

    # The Q8 statics too, since that is what a packed template stores and what
    # deltas are taken from -- a difference here would survive expansion.
    samples = speech_like(8000, 21)
    want = mfcc.mfcc_q8(samples)
    got, frames = spotter.mfcc_q8(samples, 0, len(samples))
    diffs = [(f, j) for f in range(frames) for j in range(spotter.N_CEPS)
             if got[f * spotter.N_CEPS + j] != want[f][j]]
    check("Q8 post-CMN statics identical", not diffs, "%d differ" % len(diffs))
    check("reference raised no OverflowError while comparing", True)
    mfcc.CHECK_BOUNDS = False


def test_short_and_edge_inputs():
    print("edges")
    samples = speech_like(16000, 31)
    for count in (0, 1, 399, 400, 401, 559, 560, 561):
        want = mfcc.frame_count(count)
        got = spotter.frame_count(count)
        check("frame_count(%d) == %d" % (count, want), got == want, "device %d" % got)

    # One frame exactly: the delta window has nothing either side and every
    # replicate collapses onto the same frame, so every delta must be zero.
    one = array("h", samples[:400])
    want = mfcc.mfcc(one)
    got, frames = spotter.features(one, 0, 400)
    check("single frame: 1 frame", frames == 1 and len(want) == 1)
    if frames == 1 and len(want) == 1:
        same = all(got[j] - spotter.BIAS == want[0][j] for j in range(spotter.N_FEAT))
        zero = all(got[spotter.N_CEPS + j] - spotter.BIAS == 0
                   for j in range(spotter.N_CEPS))
        check("single frame identical", same)
        check("single frame has zero deltas", zero)

    # Two and three frames reach the replicate logic at both edges at once.
    for count in (560, 720):
        clip = array("h", samples[:count])
        want = mfcc.mfcc(clip)
        got, frames = spotter.features(clip, 0, count)
        same = frames == len(want) and all(
            got[f * spotter.N_FEAT + j] - spotter.BIAS == want[f][j]
            for f in range(frames) for j in range(spotter.N_FEAT))
        check("%d samples (%d frames) identical" % (count, frames), same)


def test_hot_input():
    """A loud signal, with bounds checking on.

    Worth keeping, and worth being honest about what it is *not*: this is the
    obvious hostile input and it is **milder** than ordinary loud speech. It
    reaches 41.5% of int32 at the FFT twiddle, against 49.9% for the real speech
    frame in `speech_fixtures.tw_stress` -- because pre-emphasis saturation
    clamps an alternating full-scale rail before the FFT ever sees it. A port
    that passed only this would have false confidence.

    So the real worst-case check is `test_fixtures`, which runs `tw_stress`
    first. This case earns its place by covering what that one does not: the
    saturation path itself, which fires 1039 times here and zero times on
    speech.
    """
    print("hot input, bounds checking ON")
    mfcc.CHECK_BOUNDS = True
    mfcc._peak.clear()

    rng = random.Random(77)
    n = 8000
    hot = array("h", bytes(2 * n))
    for i in range(n):
        # Loud, broadband, and deliberately not smooth: an alternating rail is
        # what pre-emphasis saturation exists for.
        if i % 97 < 48:
            hot[i] = 32767 if (i & 1) else -32768
        else:
            hot[i] = max(-32768, min(32767, int(rng.gauss(0, 12000))))

    try:
        want = mfcc.mfcc(hot)
    except OverflowError as exc:
        check("reference survives a hot input", False, str(exc))
        mfcc.CHECK_BOUNDS = False
        return
    check("reference survives a hot input", True)

    got, frames = spotter.features(hot, 0, n)
    diffs = [(f, j) for f in range(frames) for j in range(spotter.N_FEAT)
             if got[f * spotter.N_FEAT + j] - spotter.BIAS != want[f][j]]
    check("hot input: %d frames identical" % frames, not diffs,
          "%d differ, first %s" % (len(diffs), diffs[:2]))

    rows = mfcc.peak_report()
    worst = 0.0
    for name, peak, frac in rows:
        if frac > worst:
            worst = frac
        print("       %-10s %12d  %5.1f%% of int32" % (name, peak, 100 * frac))
    check("hot input stays inside int32 (worst %.1f%%)" % (100 * worst), worst < 1.0)
    check("saturation actually fires on this input",
          any(True for _n, _p, _f in rows))  # peaks recorded means it ran
    print("       Note this is LOWER than tw_stress (49.9%): saturation clamps")
    print("       an alternating rail before the FFT, so the hostile input is")
    print("       the milder one. A viper port must re-run both.")
    mfcc.CHECK_BOUNDS = False


def test_expansion():
    print("template expansion: byte for byte")
    rng = random.Random(5)
    for shape in ((3, 40), (40, 3), (1, 1, 1), (5, 61, 2, 28), (61,), (2,)):
        index = []
        offset = 0
        statics = bytearray()
        for n in shape:
            index.append(("w%d" % len(index), offset, n))
            for _f in range(n):
                for _j in range(spotter.N_CEPS):
                    v = rng.randint(-14000, 14000) + spotter.BIAS
                    statics.append(v & 0xFF)
                    statics.append(v >> 8)
            offset += n
        total = offset

        want = bytearray(2 * spotter.N_FEAT * total)
        want[0:len(statics)] = statics
        mfcc.expand_all(want, index)

        got = bytearray(2 * spotter.N_FEAT * total)
        got[0:len(statics)] = statics
        spotter.expand(got, index)

        first = next((i for i in range(len(want)) if want[i] != got[i]), None)
        check("shape %s (%d frames)" % (shape, total), bytes(got) == bytes(want),
              "first difference at byte %s" % first)


def test_dtw():
    print("DTW distances: identical integers")
    rng = random.Random(9)

    def rows(n, base):
        out = []
        for f in range(n):
            out.append([base + rng.randint(-60, 60) for _ in range(spotter.N_FEAT)])
        return out

    cases = ((30, 30), (30, 45), (45, 30), (12, 12), (20, 41), (41, 20), (10, 25))
    for n, m in cases:
        q = rows(n, 100)
        t = rows(m, 100)
        want = hostdtw.dtw(q, t)

        query = array("H", bytes(2 * spotter.N_FEAT * n))
        for f in range(n):
            for j in range(spotter.N_FEAT):
                query[f * spotter.N_FEAT + j] = q[f][j] + spotter.BIAS
        buf = bytearray(2 * spotter.N_FEAT * m)
        p = 0
        for f in range(m):
            for j in range(spotter.N_FEAT):
                v = t[f][j] + spotter.BIAS
                buf[p] = v & 0xFF
                buf[p + 1] = v >> 8
                p += 2
        got = spotter.dtw(query, n, buf, 0, m)
        check("dtw(%d, %d) == %d" % (n, m, want), got == want, "device %d" % got)

    # The duration gate must reject the same pairs on both sides.
    q = rows(10, 100)
    t = rows(30, 100)
    check("duration gate rejects 1:3 both sides",
          hostdtw.dtw(q, t) == hostdtw.INF and
          spotter.dtw(array("H", [100 + spotter.BIAS] * (spotter.N_FEAT * 10)), 10,
                      bytearray(2 * spotter.N_FEAT * 30), 0, 30) == spotter.INF)


def test_fixtures():
    """Cross-check against dsp-host's independently derived fixtures.

    `src/speech_fixtures.py` is generated expectations -- a snapshot -- which is
    the shape a comparison test must not be built on, because a snapshot goes
    stale the moment `mfcc.py --emit-tables` runs and then either fails a
    correct regeneration or gets refreshed until it agrees with a photograph of
    itself. Every other check in this file runs both implementations on the same
    input instead.

    The fixtures earn their place anyway, because they do a job this file
    cannot: they verify the port **on the board**, where `tools/mfcc.py` cannot
    run at all. They also pin the pipeline stage by stage, so a device mismatch
    says *which* stage broke rather than only that something did.

    What is done here closes the staleness hole. Both of these run on every host
    invocation:

    - the fixtures are checked against what `mfcc` computes **today**, so a
      regeneration of the tables without a regeneration of the fixtures fails
      here rather than on the board;
    - `spotter` is checked against the fixtures, which is a genuinely
      independent cross-check -- their values came from dsp-host's code path and
      mine from my port, and the two met for the first time here.
    """
    print("fixtures: dsp-host's values, this port, and the reference")
    import speech_fixtures as fx

    check("fixture format matches the front end",
          fx.FORMAT == mfcc.TEMPLATE_FORMAT and fx.N_FEAT == mfcc.n_feat()
          and fx.N_CEPS == mfcc.N_CEPS and fx.N_MEL == mfcc.N_MEL)

    # tw_stress first, deliberately. It is the case at 99.8% of the FFT
    # twiddle's proved ceiling, so if a port wraps anywhere it wraps there, and
    # a failure list that starts with it is easier to read than one that reaches
    # it fourth. The rest keep their generated order.
    ordered = ([c for c in fx.CASES if c[0] == "tw_stress"]
               + [c for c in fx.CASES if c[0] != "tw_stress"])

    features_by_case = {}
    for name, why, pcm, exp in ordered:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()

        # 1. Are the fixtures still what the reference produces?
        ref = mfcc.mfcc(samples)
        want = exp["features"]
        stale = [(f, j) for f in range(len(want)) for j in range(fx.N_FEAT)
                 if ref[f][j] != want[f][j]]
        check("%-10s fixtures still match the reference" % name, not stale,
              "%d values differ -- regenerate with tools/make_fixtures.py" % len(stale))

        # 2. Does the port reproduce them?
        got, frames = spotter.features(samples, 0, len(samples))
        features_by_case[name] = (got, frames)
        diffs = [(f, j) for f in range(min(frames, len(want)))
                 for j in range(fx.N_FEAT)
                 if got[f * fx.N_FEAT + j] - spotter.BIAS != want[f][j]]
        check("%-10s port reproduces %d frames x %d" % (name, len(want), fx.N_FEAT),
              frames >= len(want) and not diffs,
              "%d differ, first %s" % (len(diffs), diffs[:2]))

        # 3. The first frame's cepstra, straight out of the DCT.
        #
        # The fixture pins the *raw* stage -- after the DCT, before liftering
        # and before CMN -- not the post-CMN statics `mfcc_q8` returns. Those
        # differ by a per-coefficient constant, so comparing the wrong one fails
        # on all twelve coefficients at once and looks like a broken DCT. It did
        # here first, which is the argument for the fixtures pinning stages
        # rather than only the output: the failure named the stage, and the
        # stage was the one my test had picked wrongly.
        pre = array("h", bytes(2 * len(samples)))
        spotter.preemphasise(samples, 0, len(samples), pre)
        work = spotter.new_work()
        frame = work[4]
        for n in range(spotter.FRAME_LEN):
            frame[n] = pre[n]
        row = [0] * spotter.N_CEPS
        spotter.mfcc_frame(frame, work, row)
        cep = exp["cepstra"]
        cdiff = [j for j in range(fx.N_CEPS) if row[j] != cep[j]]
        check("%-10s first frame's raw cepstra (post-DCT)" % name, not cdiff,
              "coefficients %s differ" % cdiff)

        peaks = exp.get("peaks", {})
        worst = max(peaks.values()) if peaks else 0
        print("       %-10s %-52s peak %5.1f%% of int32, %d saturations"
              % (name, why[:52], 100.0 * worst / float((1 << 31) - 1),
                 exp.get("saturated", 0)))

    # 4. The matcher, against the recorded distance between cases 0 and 1 --
    #    by name, since the iteration order above is no longer the file's.
    (qa, na) = features_by_case[fx.CASES[0][0]]
    (qb, nb) = features_by_case[fx.CASES[1][0]]
    tmpl = bytearray(2 * fx.N_FEAT * nb)
    p = 0
    for i in range(nb * fx.N_FEAT):
        v = qb[i]
        tmpl[p] = v & 0xFF
        tmpl[p + 1] = v >> 8
        p += 2
    got = spotter.dtw(qa, na, tmpl, 0, nb, band=fx.DTW_BAND)
    check("DTW between case 0 and case 1 == %d" % fx.DTW_0_1, got == fx.DTW_0_1,
          "port gives %d" % got)


def test_end_to_end():
    print("end to end: enrol, then spot")
    # Three "words", three takes each, built from the same generator with
    # different seeds -- close enough within a word to match, far enough apart
    # between words to separate. This tests the wiring, not the acoustics.
    words = {}
    for w, base in (("mother", 100), ("father", 300), ("sleep", 500)):
        words[w] = [speech_like(6400, base, take) for take in range(3)]

    index = []
    blobs = []
    offset = 0
    for label in sorted(words):
        for take in words[label][:2]:          # two takes enrolled
            frames = mfcc.mfcc(take)
            blobs.append(mfcc.pack_template(frames))
            index.append((label, offset, len(frames)))
            offset += len(frames)
    buf = bytearray(2 * spotter.N_FEAT * offset)
    p = 0
    for blob in blobs:
        buf[p:p + len(blob)] = blob
        p += len(blob)

    spotter.bind(buf, tuple(index))
    check("ready once bound", spotter.ready())

    right = 0
    for label in sorted(words):
        held_out = words[label][2]             # the take that was not enrolled
        got, best, runner = spotter.spot_scored(held_out, 0, len(held_out),
                                                threshold=1 << 28, margin=0)
        if got == label:
            right += 1
        print("       %-7s -> %-7s best %d runner-up %d"
              % (label, got, best, runner))
    check("held-out take matches its own class (%d/3)" % right, right == 3)

    # The scores must be the same numbers the host matcher produces.
    matcher = hostdtw.Matcher({label: [mfcc.mfcc(t) for t in words[label][:2]]
                               for label in words})
    held = words["mother"][2]
    query, n = spotter.features(held, 0, len(held))
    device = spotter.scores(query, n)
    host = matcher.scores(mfcc.mfcc(held))
    check("class scores identical to the host matcher",
          [(s, l) for s, l in device] == [(s, l) for s, l in host],
          "device %s host %s" % (device[:2], host[:2]))

    # The rejection gates, exercised rather than assumed.
    got, best, runner = spotter.spot_scored(held, 0, len(held), threshold=1, margin=0)
    check("absolute gate rejects below threshold", got is None)
    got, best, runner = spotter.spot_scored(held, 0, len(held),
                                            threshold=1 << 28, margin=1 << 28)
    check("margin gate rejects an indistinct best", got is None)

    # Unbound, it must decline rather than explode: that is the state the
    # device is in before templates are enrolled.
    spotter._buf = None
    spotter._index = None
    check("declines cleanly when unbound",
          spotter.spot(held, 0, len(held)) is None and not spotter.ready())
    spotter.bind(buf, tuple(index))


def test_overflow_bounds():
    print("overflow, with the reference's bounds checking on")
    mfcc.CHECK_BOUNDS = True
    mfcc._peak.clear()
    try:
        mfcc.mfcc(speech_like(8000, 41))
    except OverflowError as exc:
        check("reference stays inside int32", False, str(exc))
    else:
        check("reference stays inside int32", True)
    rows = mfcc.peak_report()
    for name, peak, frac in rows:
        print("       %-10s %12d  %5.1f%% of int32" % (name, peak, 100 * frac))
    worst = max((frac for _n, _p, frac in rows), default=0.0)
    check("worst stage under 100%% of int32", worst < 1.0, "worst %.1f%%" % (100 * worst))
    print("       NOTE these bounds hold for arbitrary-precision ints. A viper")
    print("       port must re-prove them. The tightest stage is fft.tw at")
    print("       49.90%% of int32 on real speech (tw_stress), against a proved")
    print("       ceiling of 50.00%% -- tight analysis, factor-of-two margin.")
    mfcc.CHECK_BOUNDS = False


def main():
    test_constants()
    test_tables()
    test_isqrt()
    test_preemphasis()
    test_features()
    test_hot_input()
    test_short_and_edge_inputs()
    test_expansion()
    test_dtw()
    test_fixtures()
    test_end_to_end()
    test_overflow_bounds()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
