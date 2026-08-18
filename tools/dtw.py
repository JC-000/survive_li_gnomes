#!/usr/bin/env python3
"""Banded DTW matcher, and the harness that decides where to set the threshold.

The matcher is small. The harness is the point of the file.

What matters for this spotter is **precision at low recall**, not accuracy. A
miss costs nothing -- ELIZA says "PLEASE GO ON" and stays in character -- while
a false fire produces "DO YOU OFTEN THINK OF MOTHER" when the user said
"morning", and that is the moment the illusion dies. So the harness never
reports a single accuracy figure. It reports precision and recall separately
over a sweep of thresholds, against a test set that deliberately includes
utterances the spotter must stay *silent* on, and recommends the threshold
where the false fires stop.

    python3 tools/say_corpus.py corpus/
    python3 tools/dtw.py --eval corpus/          # the sweep and the matrix
    python3 tools/dtw.py --tune corpus/          # compare front-end variants

Features are cached under corpus/_feat, keyed by the front end's parameters, so
only the first run pays for the MFCCs.

No third-party packages.
"""

import array
import hashlib
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# src first, then tools, so tools wins: src/vad.py is the *device* twin of
# tools/vad.py and cannot import `wave`. The harness wants the host one.
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import mfcc            # noqa: E402
import vad             # noqa: E402
import say_corpus      # noqa: E402
import vocab           # noqa: E402

# `vad` MUST be the host copy, and the import above must stay ahead of
# say_corpus, which prepends src/ to sys.path for its own `import vocab`. It
# cost an afternoon once: with a warm feature cache the harness never calls
# read_wav, so picking up the device twin by mistake ran to completion and only
# failed later, on the first run that had to decode a WAV.
if os.path.dirname(os.path.abspath(vad.__file__)) != HERE:
    raise ImportError("vad resolved to %s, wanted the copy in %s"
                      % (vad.__file__, HERE))

# --- Matcher parameters ----------------------------------------------------

BAND = 10            # Sakoe-Chiba radius in frames, about the diagonal ratio
                     # line. 100 ms of *local* deviation; global rate
                     # difference is already absorbed by the ratio line, so
                     # this does not need to be large.
DUR_RATIO_PCT = 200  # reject outright if the two lengths differ by more than
                     # 2:1. Cheap, and it removes the collisions where a long
                     # word warps onto a short one.
INF = 1 << 29        # far above any real path cost, far below int32


def dtw(query, tmpl, band=BAND):
    """Normalised distance between two frame lists, or INF if rejected.

    Symmetric Sakoe-Chiba: steps (1,0) and (0,1) cost d, the diagonal (1,1)
    costs 2d, and the result is divided by N+M. That weighting is the reason
    no path length has to be tracked -- every monotone path from (0,0) to
    (N-1,M-1) accumulates exactly N+M units of step weight, so the normaliser
    is a constant and the comparison between a long template and a short one
    stays honest.

    L1 over the feature vector, not L2: no multiply, and squaring only
    sharpens the influence of the one coefficient that happened to be worst.
    """
    n = len(query)
    m = len(tmpl)
    if n == 0 or m == 0:
        return INF
    if n * 100 > m * DUR_RATIO_PCT or m * 100 > n * DUR_RATIO_PCT:
        return INF

    width = len(query[0])
    prev = array.array("i", bytes(4 * m))
    cur = array.array("i", bytes(4 * m))
    for j in range(m):
        prev[j] = INF

    for i in range(n):
        c = (i * m) // n
        lo = c - band
        hi = c + band
        if lo < 0:
            lo = 0
        if hi > m - 1:
            hi = m - 1
        q = query[i]
        for j in range(m):
            cur[j] = INF
        for j in range(lo, hi + 1):
            t = tmpl[j]
            d = 0
            for k in range(width):
                v = q[k] - t[k]
                d += v if v >= 0 else -v
            if i == 0 and j == 0:
                cur[0] = 2 * d
                continue
            best = INF
            if i > 0:
                a = prev[j] + d                       # (1,0)
                if a < best:
                    best = a
                if j > 0:
                    a = prev[j - 1] + 2 * d           # (1,1)
                    if a < best:
                        best = a
            if j > 0:
                a = cur[j - 1] + d                    # (0,1)
                if a < best:
                    best = a
            cur[j] = best
        prev, cur = cur, prev

    total = prev[m - 1]
    if total >= INF:
        return INF
    return total // (n + m)


class Matcher(object):
    """Templates grouped by class label.

    Three templates per spoken form, and a class may own more than one form
    (SAD owns SAD and SICK), so a class can hold six. The class score is the
    minimum over everything it owns -- nothing in the decision cares which
    template matched, only which class.
    """

    def __init__(self, templates):
        self.templates = templates   # {label: [frames, ...]}
        self.labels = sorted(templates)

    def scores(self, query):
        out = []
        for label in self.labels:
            best = INF
            for tmpl in self.templates[label]:
                s = dtw(query, tmpl)
                if s < best:
                    best = s
            out.append((best, label))
        out.sort()
        return out

    def decide(self, query, threshold, margin=0):
        """(label, score) if the spotter should fire, else (None, score).

        Two gates. The absolute one asks whether this is a good enough match
        to be worth acting on at all. The margin asks whether it is
        *distinctly* the best -- if MOTHER and BROTHER both score 190, the
        right answer is to say nothing, whatever the absolute score is. The
        margin gate turns out to be the cheaper half of the precision, because
        the near-miss confusions cluster tightly while the genuine matches
        stand clear.
        """
        ranked = self.scores(query)
        if not ranked:
            return None, INF
        best, label = ranked[0]
        if best > threshold:
            return None, best
        if margin and len(ranked) > 1 and ranked[1][0] - best < margin:
            return None, best
        return label, best


# --- Feature cache ---------------------------------------------------------

def _cache_key():
    """Fingerprint of every front-end constant, derived rather than listed.

    A stale feature cache is a silent wrong answer: the harness recomputes
    nothing, reports the same precision and recall as before, and the change
    under test simply does not appear. So this must cover every constant that
    can alter a feature.

    It used to be a hand-written format string with a docstring saying "keep
    this up to date". That is not a mechanism, and it had already needed
    editing twice in a day -- once for the mel band, once for the deltas. Now
    it walks `mfcc`'s module namespace and hashes every upper-case int or
    float, so a new constant is covered the moment it exists and nobody has to
    remember. `type(v) in (int, float)` rather than `isinstance`, deliberately:
    `isinstance(True, int)` is True, and folding a debug flag like
    CHECK_BOUNDS into the key would wipe the cache every time it was toggled.

    TEMPLATE_FORMAT stays in the clear so a cache directory can be identified
    at a glance.
    """
    parts = []
    for name in sorted(dir(mfcc)):
        if not name.isupper():
            continue
        value = getattr(mfcc, name)
        if type(value) in (int, float):
            parts.append("%s=%r" % (name, value))
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return "v%d-%s" % (mfcc.TEMPLATE_FORMAT, digest[:16])


def features_for(path, cache_dir):
    """Endpoint, then MFCC, then cache. Returns frames or None."""
    stamp = os.path.join(cache_dir, "_key")
    key = _cache_key()
    if os.path.isdir(cache_dir):
        old = ""
        if os.path.exists(stamp):
            with open(stamp) as fh:
                old = fh.read().strip()
        if old != key:
            for name in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, name))
    else:
        os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(stamp):
        with open(stamp, "w") as fh:
            fh.write(key)

    name = os.path.basename(path).replace(".wav", ".f16")
    cached = os.path.join(cache_dir, name)
    if os.path.exists(cached):
        with open(cached, "rb") as fh:
            blob = fh.read()
        if not blob:
            return None
        return mfcc.unpack_template(blob)

    samples = vad.read_wav(path)
    trimmed = vad.trim(samples)
    frames = mfcc.mfcc(trimmed) if trimmed is not None else []
    blob = mfcc.pack_template(frames)
    with open(cached, "wb") as fh:
        fh.write(blob)
    return frames if frames else None


def load_set(root, sub, quiet=False):
    """{name: [frames, ...]} for one corpus subdirectory."""
    d = os.path.join(root, sub)
    cache = os.path.join(root, "_feat", sub)
    out = {}
    dropped = 0
    files = sorted(os.listdir(d))
    t0 = time.time()
    for i, fn in enumerate(files):
        if not fn.endswith(".wav"):
            continue
        name = fn.rsplit(".", 2)[0]
        frames = features_for(os.path.join(d, fn), cache)
        if frames is None:
            dropped += 1
            continue
        out.setdefault(name, []).append(frames)
        if not quiet and (i % 25 == 0):
            sys.stderr.write("\r  %s %d/%d" % (sub, i, len(files)))
            sys.stderr.flush()
    if not quiet:
        sys.stderr.write("\r  %s %d files, %d endpointed away, %.1f s\n"
                         % (sub, len(files), dropped, time.time() - t0))
    return out, dropped


def build_matcher(enrol, drop=()):
    """Spoken forms -> class templates, minus any dropped class.

    `drop` exists so that "what does the vocabulary cost us?" is a measurement
    rather than an argument. A word that attracts false fires can be removed
    and the whole sweep re-run against the same audio.
    """
    by_label = {}
    for form, takes in enrol.items():
        label = vocab.label_of(form)
        if label is None or label in drop:
            continue
        by_label.setdefault(label, []).extend(takes)
    return Matcher(by_label)


# --- Evaluation ------------------------------------------------------------

def score_everything(matcher, test, oov):
    """All (truth, ranked scores) pairs, computed once.

    Truth is one of three things, and the third is the one that stops the
    numbers lying:

      "mother"                 in vocabulary; must fire, with this label
      ("variant", "mother")    a morphological relative such as MOTHER'S;
                               firing with that label is *correct* behaviour
                               for DOCTOR, so it counts as neither a hit nor a
                               miss. Firing with any other label is a mistake.
      None                     must stay silent

    Every threshold in the sweep then reads the same numbers, which is both
    fast and the only way the sweep is self-consistent.
    """
    rows = []
    total = sum(len(v) for v in test.values()) + sum(len(v) for v in oov.values())
    done = 0
    t0 = time.time()
    for form, takes in sorted(test.items()):
        label = vocab.label_of(form)
        if label not in matcher.templates:
            # A dropped class is not something the spotter is expected to
            # catch, but it is very much something it must not mis-fire on,
            # so it moves to the silent set rather than leaving the corpus.
            label = None
        for frames in takes:
            rows.append((label, matcher.scores(frames), form))
            done += 1
            sys.stderr.write("\r  matching %d/%d" % (done, total))
    for name, takes in sorted(oov.items()):
        variant = say_corpus.OOV_VARIANTS.get(name)
        truth = ("variant", variant) if variant else None
        for frames in takes:
            rows.append((truth, matcher.scores(frames), name))
            done += 1
            sys.stderr.write("\r  matching %d/%d" % (done, total))
    sys.stderr.write("\r  matched %d utterances in %.1f s\n"
                     % (total, time.time() - t0))
    return rows


def measure(rows, threshold, margin):
    """(precision, recall, tp, fp_wrong, fp_oov, n_in_vocab, benign)."""
    tp = fp_wrong = fp_oov = benign = 0
    n_in = 0
    for truth, ranked, _name in rows:
        if isinstance(truth, str):
            n_in += 1
        if not ranked:
            continue
        best, label = ranked[0]
        if best > threshold:
            continue
        if margin and len(ranked) > 1 and ranked[1][0] - best < margin:
            continue
        if truth is None:
            fp_oov += 1
        elif isinstance(truth, tuple):
            if label == truth[1]:
                benign += 1
            else:
                fp_oov += 1
        elif label == truth:
            tp += 1
        else:
            fp_wrong += 1
    fires = tp + fp_wrong + fp_oov
    precision = tp / fires if fires else 1.0
    recall = tp / n_in if n_in else 0.0
    return precision, recall, tp, fp_wrong, fp_oov, n_in, benign


def sweep(rows, margins=(0, 40, 60, 80, 100)):
    """Every threshold at which the decision actually changes.

    Candidates are the observed best-scores themselves rather than a linear
    grid: a grid straddles the one score that matters and reports a boundary
    that does not exist.
    """
    scores = sorted(set(r[1][0][0] for r in rows if r[1] and r[1][0][0] < INF))
    if not scores:
        return []
    out = []
    for margin in margins:
        for th in scores:
            out.append((margin, th) + measure(rows, th, margin))
    return out


def recommend(rows, margins=(0, 40, 60, 80, 100)):
    """Highest-recall setting that fires zero false positives, and the
    highest-recall setting that stays at or above 95% precision.

    The margin candidates matter as much as the thresholds. Measured, margin 80
    is worth 9 points of recall over margin 0 at the same precision, because
    the near-miss confusions cluster tightly (OTHER sits 51 units from FATHER)
    while genuine matches stand clear.

    Two answers because they are different questions. "Never fires wrongly" is
    the one this project actually wants; "95%" is there to show what the extra
    strictness costs, so the choice is visible rather than assumed.
    """
    clean = None
    p95 = None
    for row in sweep(rows, margins):
        # row is (margin, th, precision, recall, tp, fp_wrong, fp_oov,
        #         n_in_vocab, benign)
        r, tp, fw, fo = row[3], row[4], row[5], row[6]
        if fw + fo == 0 and tp > 0:
            if clean is None or r > clean[3]:
                clean = row
        if row[2] >= 0.95:
            if p95 is None or r > p95[3]:
                p95 = row
    return clean, p95


def confusion(rows, threshold, margin, labels):
    """{truth: {fired_label_or_None: count}}."""
    table = {}
    for truth, ranked, _name in rows:
        if isinstance(truth, tuple):
            key = "(variant)"
        elif truth is None:
            key = "(oov)"
        else:
            key = truth
        row = table.setdefault(key, {})
        fired = None
        if ranked:
            best, label = ranked[0]
            if best <= threshold and not (
                    margin and len(ranked) > 1 and ranked[1][0] - best < margin):
                fired = label
        row[fired] = row.get(fired, 0) + 1
    return table


def pair_separation(matcher, test):
    """For each true class, the mean margin between it and its nearest rival.

    This is the number that says whether MOTHER and FATHER are separable. A
    small or negative margin means the pair is a coin toss, and the right fix
    is to drop one of them, not to fiddle with the threshold.
    """
    out = []
    for form, takes in sorted(test.items()):
        truth = vocab.label_of(form)
        if truth is None:
            continue
        gaps = []
        rivals = {}
        for frames in takes:
            ranked = matcher.scores(frames)
            own = None
            other = None
            for s, label in ranked:
                if label == truth and own is None:
                    own = s
                elif label != truth and other is None:
                    other = (s, label)
                if own is not None and other is not None:
                    break
            if own is None or other is None:
                continue
            gaps.append(other[0] - own)
            rivals[other[1]] = rivals.get(other[1], 0) + 1
        if gaps:
            worst = min(rivals, key=lambda k: -rivals[k])
            out.append((sum(gaps) // len(gaps), truth, form, worst))
    out.sort()
    return out


def cmd_eval(root, drop=()):
    print("front end: %s" % _cache_key())
    if drop:
        print("dropped classes: %s" % ", ".join(sorted(drop)))
    enrol, _ = load_set(root, "enrol")
    test, test_dropped = load_set(root, "test")
    oov, oov_dropped = load_set(root, "oov")
    matcher = build_matcher(enrol, drop)

    n_tmpl = sum(len(v) for v in matcher.templates.values())
    tmpl_bytes = sum(len(f) * 2 * mfcc.n_feat()
                     for v in matcher.templates.values() for f in v)
    lens = [len(f) for v in matcher.templates.values() for f in v]
    print("\n%d classes, %d templates, %d frames mean (%d..%d), %d bytes total"
          % (len(matcher.labels), n_tmpl, sum(lens) // len(lens),
             min(lens), max(lens), tmpl_bytes))
    print("endpointing rejected %d test and %d oov utterances outright"
          % (test_dropped, oov_dropped))

    rows = score_everything(matcher, test, oov)
    n_oov = sum(1 for t, _r, _n in rows if not isinstance(t, str))
    n_in = sum(1 for t, _r, _n in rows if isinstance(t, str))
    retired = {}
    for truth, _r, name in rows:
        if truth is None and name in say_corpus.RETIRED:
            retired[name] = retired.get(name, 0) + 1
    print("\n%d in-vocabulary and %d out-of-vocabulary utterances" % (n_in, n_oov))
    if retired:
        # Words cut from the vocabulary are generated into the silent set on
        # purpose (say_corpus.RETIRED). Printed because a corpus built by
        # enumerating vocab.FORMS alone would simply not contain them, and the
        # sweep would look identical while no longer testing the cut.
        print("  including %d utterances of RETIRED words (%s) that must now"
              " stay silent"
              % (sum(retired.values()),
                 ", ".join("%s x%d" % (k, v) for k, v in sorted(retired.items()))))
    print()

    print("threshold sweep (score is Q4 log2 units per unit of path weight)")
    print("  margin  thresh   prec   recall    tp  wrong   oov  benign")
    seen = set()
    for margin, th, p, r, tp, fw, fo, _n, ben in sweep(rows):
        key = (margin, tp, fw, fo, ben)
        if key in seen:
            continue
        seen.add(key)
        print("  %6d  %6d  %5.3f   %5.3f  %4d  %5d  %4d  %6d"
              % (margin, th, p, r, tp, fw, fo, ben))

    clean, p95 = recommend(rows)
    print()
    if clean:
        margin, th, p, r, tp, fw, fo, n, ben = clean
        print("RECOMMENDED (zero false fires): threshold %d, margin %d"
              % (th, margin))
        print("  precision 1.000, recall %.3f (%d of %d in-vocabulary),"
              " %d benign variant fires" % (r, tp, n, ben))
    else:
        print("no setting fires without a false positive")
    if p95:
        margin, th, p, r, tp, fw, fo, n, ben = p95
        print("at 95%% precision: threshold %d, margin %d -> recall %.3f"
              % (th, margin, r))

    print("\nwhat the out-of-vocabulary set does, closest matches first")
    print("  score  utterance        fires as   margin  verdict")
    oov_rows = []
    for truth, ranked, name in rows:
        if isinstance(truth, str) or not ranked:
            continue
        best, label = ranked[0]
        gap = ranked[1][0] - best if len(ranked) > 1 else INF
        ok = isinstance(truth, tuple) and label == truth[1]
        oov_rows.append((best, name, label, gap, ok))
    oov_rows.sort()
    for best, name, label, gap, ok in oov_rows[:18]:
        print("  %5d  %-16s %-10s %6d  %s"
              % (best, name, label, gap,
                 "benign variant" if ok else "MUST NOT FIRE"))

    if clean:
        print("\nconfusion at the recommended setting"
              " (rows: truth, columns: what fired)")
        table = confusion(rows, clean[1], clean[0], matcher.labels)
        for truth in sorted(table, key=str):
            row = table[truth]
            fired = sorted((k for k in row if k is not None))
            silent = row.get(None, 0)
            bits = ", ".join("%s x%d" % (k, row[k]) for k in fired)
            print("  %-10s silent x%-3d %s" % (truth, silent, bits or ""))

    print("\nseparation from nearest rival class (mean over test takes)")
    print("  gap  class      spoken     nearest rival")
    for gap, truth, form, rival in pair_separation(matcher, test):
        flag = "  <-- too close" if gap < 20 else ""
        print("  %4d  %-10s %-10s %s%s" % (gap, truth, form, rival, flag))
    return 0


def cmd_tune(root, drop=()):
    """Compare front-end variants on the same corpus.

    Each variant re-runs the whole front end, so this is slow. It exists
    because "keep c0 or not" and "lifter or not" are the two questions everyone
    argues about from first principles and nobody measures.
    """
    variants = (
        ("baseline 100-7600", {}),
        ("deltas D=2", {"DELTA_WIDTH": 2}),
        ("deltas D=2 unweighted", {"DELTA_WIDTH": 2, "DELTA_SHIFT": 0}),
        ("mel 300-3400 x20", {"MEL_LOW_HZ": 300.0, "MEL_HIGH_HZ": 3400.0,
                              "N_MEL": 20}),
        ("mel 300-3400 +delta", {"MEL_LOW_HZ": 300.0, "MEL_HIGH_HZ": 3400.0,
                                 "N_MEL": 20, "DELTA_WIDTH": 2}),
        ("mel 200-5000 x24", {"MEL_LOW_HZ": 200.0, "MEL_HIGH_HZ": 5000.0,
                              "N_MEL": 24}),
        ("lifter L=22", {"LIFTER_L": 22}),
        ("mel floor 36 dB", {"MEL_FLOOR_SHIFT": 6}),
        ("band 5", {}),
        ("band 20", {}),
    )
    bands = {"band 5": 5, "band 20": 20}
    saved = dict((k, getattr(mfcc, k)) for k in
                 ("LIFTER_L", "MEL_FLOOR_SHIFT", "DELTA_WIDTH", "DELTA_SHIFT",
                  "MEL_LOW_HZ", "MEL_HIGH_HZ", "N_MEL"))
    global BAND
    saved_band = BAND
    print("%-22s %6s %7s %7s %7s %7s" % ("variant", "prec", "recall",
                                          "thresh", "margin", "bytes"))
    import io
    import contextlib
    for name, over in variants:
        for k, v in saved.items():
            setattr(mfcc, k, v)
        for k, v in over.items():
            setattr(mfcc, k, v)
        mfcc._TABLES = None
        BAND = bands.get(name, saved_band)
        with contextlib.redirect_stderr(io.StringIO()):
            enrol, _ = load_set(root, "enrol", quiet=True)
            test, _ = load_set(root, "test", quiet=True)
            oov, _ = load_set(root, "oov", quiet=True)
        matcher = build_matcher(enrol, drop)
        with contextlib.redirect_stderr(io.StringIO()):
            rows = score_everything(matcher, test, oov)
        rows = [(None if isinstance(t, str) and t not in matcher.templates
                 else t, r, n) for t, r, n in rows]
        clean, _p95 = recommend(rows, margins=(0, 40, 80, 120))
        nbytes = sum(len(f) * 2 * mfcc.n_feat()
                     for v in matcher.templates.values() for f in v)
        if clean:
            margin, th, p, r, tp, fw, fo, n, ben = clean
            print("%-22s %6.3f %7.3f %7d %7d %7d"
                  % (name, p, r, th, margin, nbytes))
        else:
            print("%-22s %6s %7s" % (name, "-", "none clean"))
    for k, v in saved.items():
        setattr(mfcc, k, v)
    mfcc._TABLES = None
    BAND = saved_band
    return 0


def main(argv):
    drop = ()
    if "--drop" in argv:
        drop = tuple(argv[argv.index("--drop") + 1].split(","))
    if "--eval" in argv:
        return cmd_eval(argv[argv.index("--eval") + 1], drop)
    if "--tune" in argv:
        return cmd_tune(argv[argv.index("--tune") + 1], drop)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
