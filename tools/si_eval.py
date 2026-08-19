#!/usr/bin/env python3
"""Score a speaker-independent CNN the way `tools/dtw.py` scores the incumbent.

Three numbers, reported separately and never averaged together, because they
answer different questions and only one of them decides anything:

1. **Synthetic validation** -- held-out `say` *voices*, not held-out utterances.
   Held-out utterances of a voice already in training measures nothing: two
   `say` renderings of the same word in the same voice are nearly the same
   waveform, and a model can memorise a voice as easily as a word.
2. **Real speaker** -- the enrolment takes in `takes/`, a person speaking
   through the board's own microphone. This is the number the experiment is
   for. It is expected to be much worse than (1), and **the gap is the
   result**, not a disappointment to be tuned away.
3. **Precision at a confidence threshold** -- swept, in the same shape
   `tools/dtw.py` prints, so the two approaches can be read against each other.

    tools/si_eval.py build/si.tflite corpus/manifest.jsonl --takes takes/

## Why precision and not accuracy

From `docs/speech-design.md`: a miss costs nothing, because ELIZA's deflections
("PLEASE GO ON") are real DOCTOR responses and nobody can tell a missed word
from a sentence that contained nothing interesting. A wrong fire answers
"morning" with "DO YOU OFTEN THINK OF MONEY" and the illusion is over. So a
model that fires on 40% of words and is never wrong beats one that fires on 85%
and is wrong a fifth of the time, and any single accuracy figure ranks those
two the wrong way round.

## The two gates, and how they map onto a softmax

`tools/dtw.py` gates on an absolute distance (THRESHOLD) and on the gap to the
second-best class (MARGIN). Both survive the change of matcher, with the sign
flipped because a probability is better when larger:

| DTW | here |
| --- | --- |
| best distance <= THRESHOLD | top-1 probability >= `thresh` |
| second-best - best >= MARGIN | top-1 - top-2 probability >= `margin` |
| (no equivalent) | top-1 class is not `unknown` |

The third row is the one the CNN adds and DTW cannot have: `unknown` is a
trained class here, with its own gradient, rather than a region of distance
space that no template happens to occupy. The margin is taken against the next
class whatever it is, `unknown` included -- an utterance the model thinks is
60% MOTHER and 35% nothing is exactly the case the gate exists to reject.

Thresholds are swept over the observed top-1 probabilities rather than a linear
grid, for the reason `dtw.sweep` gives: a grid straddles the one value that
matters and reports a boundary that does not exist.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import say_corpus                              # noqa: E402
import si_features                             # noqa: E402
import vocab                                   # noqa: E402

CLASSES = si_features.CLASSES
UNKNOWN_INDEX = CLASSES.index(si_features.UNKNOWN)


# --- scoring ---------------------------------------------------------------

def rows_from_probs(probs, truths, names):
    """-> [(truth, [(p, label), ...] best first, name)], `dtw.py`'s shape.

    `truth` follows `dtw.score_everything` exactly, so the sweep below and the
    one in `tools/dtw.py` are the same function with different inputs:

      "mother"                in vocabulary; firing with this label is a hit
      ("variant", "mother")   a morphological relative (MOTHER'S, COMPUTERS);
                              firing with that label is correct DOCTOR
                              behaviour and counts as neither hit nor miss
      None                    must stay silent
    """
    out = []
    for p, truth, name in zip(probs, truths, names):
        order = sorted(range(len(CLASSES)), key=lambda i: -p[i])
        ranked = [(float(p[i]), CLASSES[i]) for i in order]
        out.append((truth, ranked, name))
    return out


def measure(rows, thresh, margin):
    """(precision, recall, tp, fp_wrong, fp_oov, n_in_vocab, benign).

    Deliberately the same signature and the same counting rules as
    `dtw.measure`. Two figures that are computed differently cannot be compared
    however carefully they are labelled, so they are computed the same.
    """
    tp = fp_wrong = fp_oov = benign = 0
    n_in = 0
    for truth, ranked, _name in rows:
        if isinstance(truth, str):
            n_in += 1
        if not ranked:
            continue
        best, label = ranked[0]
        if label == si_features.UNKNOWN:
            continue
        if best < thresh:
            continue
        if margin and len(ranked) > 1 and best - ranked[1][0] < margin:
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


MARGINS = (0.0, 0.2, 0.4, 0.6, 0.8)


def sweep(rows, margins=MARGINS):
    cands = sorted(set(round(r[1][0][0], 4) for r in rows if r[1]))
    out = []
    for margin in margins:
        for th in cands:
            out.append((margin, th) + measure(rows, th, margin))
    return out


def recommend(rows, margins=MARGINS):
    """Highest-recall setting with zero false fires, and the best at >=95%.

    Same two questions `dtw.recommend` asks, for the same reason: "never fires
    wrongly" is what this project wants, and the 95% row shows what the extra
    strictness costs so the choice is visible rather than assumed.
    """
    clean = None
    p95 = None
    for row in sweep(rows, margins):
        r, tp, fw, fo = row[3], row[4], row[5], row[6]
        if fw + fo == 0 and tp > 0:
            if clean is None or r > clean[3]:
                clean = row
        if row[2] >= 0.95:
            if p95 is None or r > p95[3]:
                p95 = row
    return clean, p95


# Precision milestones the table is printed at. `dtw.py` can print one row per
# distinct threshold because its scores are integers with long plateaus, so its
# sweep collapses to a few dozen rows. Softmax probabilities are all distinct,
# every threshold changes the counts by one, and the same code prints a
# thousand rows of noise. So the axis is turned around: for each margin, the
# highest recall reachable at each precision. Same columns, same counting, same
# `measure` -- one row per decision a reader might actually make.
TARGETS = (0.80, 0.90, 0.95, 0.99, 1.00)


def print_sweep(rows, title, margins=MARGINS):
    print("\n%s" % title)
    print("  for each margin, the highest recall reachable at each precision")
    print("  margin  thresh   prec   recall    tp  wrong   oov  benign")
    seen = set()
    all_rows = sweep(rows, margins)
    for margin in margins:
        for target in TARGETS:
            best = None
            for row in all_rows:
                if row[0] != margin or row[2] < target or row[4] == 0:
                    continue
                if best is None or row[3] > best[3]:
                    best = row
            if best is None:
                continue
            _m, th, prec, rec, tp, fw, fo, _n, benign = best
            key = (margin, tp, fw, fo, benign)
            if key in seen:
                continue
            seen.add(key)
            print("  %6.2f  %6.3f  %5.3f   %5.3f  %4d  %5d  %4d  %6d"
                  % (margin, th, prec, rec, tp, fw, fo, benign))
    clean, p95 = recommend(rows, margins)
    if clean:
        print("  RECOMMENDED (zero false fires): thresh %.3f, margin %.2f"
              % (clean[1], clean[0]))
        print("    precision 1.000, recall %.3f (%d of %d in-vocabulary),"
              " %d benign variant fires"
              % (clean[3], clean[4], clean[7], clean[8]))
    else:
        print("  no setting fires without a false positive")
    if p95:
        print("  at 95%% precision: thresh %.3f, margin %.2f -> recall %.3f"
              % (p95[1], p95[0], p95[3]))


def top1_accuracy(rows):
    """Plain argmax accuracy over in-vocabulary utterances, no gates.

    Reported because it is what everybody else reports and the comparison to
    Hello Edge's 94.4% is otherwise impossible to make. It is not the number
    this project tunes on, and the sweep above is why.
    """
    hit = n = 0
    for truth, ranked, _name in rows:
        if not isinstance(truth, str) or not ranked:
            continue
        n += 1
        if ranked[0][1] == truth:
            hit += 1
    return (hit / n if n else 0.0), hit, n


def unknown_accuracy(rows):
    """How often a must-stay-silent utterance is classified `unknown` outright.

    Before any threshold. A model whose argmax is already right on the negative
    set has real rejection; one that depends entirely on the threshold has
    borrowed it, and will lose it on a speaker the threshold was not tuned for.
    """
    hit = n = 0
    for truth, ranked, _name in rows:
        if truth is not None or not ranked:
            continue
        n += 1
        if ranked[0][1] == si_features.UNKNOWN:
            hit += 1
    return (hit / n if n else 0.0), hit, n


def per_class(rows, thresh, margin):
    """[(recall, label, fires, wrong)] at one operating point, worst first.

    A single recall figure hides the shape of the failure, and the shape is
    what decides what to do next. Twenty classes at 0.8 and one at 0.0 is a
    vocabulary problem -- drop the word, as NO was dropped. Every class at 0.4
    is a model or corpus problem. The two look identical in the headline.
    """
    stats = {}
    for truth, ranked, _name in rows:
        if not isinstance(truth, str) or not ranked:
            continue
        s = stats.setdefault(truth, [0, 0, 0])
        s[0] += 1
        best, label = ranked[0]
        gap = best - ranked[1][0] if len(ranked) > 1 else 1.0
        if label == si_features.UNKNOWN or best < thresh or gap < margin:
            continue
        s[1] += 1
        if label == truth:
            s[2] += 1
    out = []
    for label, (n, fires, hits) in stats.items():
        out.append((hits / n if n else 0.0, label, fires, fires - hits))
    out.sort()
    return out


def print_steals(rows, limit=12):
    """Which classes take utterances from which, by plain argmax.

    **This exists because its absence let three live false fires through.** The
    evaluation used to print per-class *recall* and nothing else, and recall is
    computed along the rows of a confusion matrix while false fires live in its
    columns. `money` was firing as `wife` in 16% of synthetic utterances the
    whole time; nobody saw it until a user said "money" to the device and was
    asked about his wife.

    So: a per-class recall table shows what each class **misses** and hides what
    each class **steals**. Both are printed now.

    Reported before any gate, because a confusion that the threshold currently
    suppresses is still a confusion the model has, and the threshold moves.
    """
    steals = {}
    totals = {}
    for truth, ranked, _name in rows:
        if not isinstance(truth, str) or not ranked:
            continue
        totals[truth] = totals.get(truth, 0) + 1
        got = ranked[0][1]
        if got != truth:
            steals.setdefault(got, {})
            steals[got][truth] = steals[got].get(truth, 0) + 1
    pairs = []
    for thief, victims in steals.items():
        for victim, n in victims.items():
            pairs.append((n, n / max(totals.get(victim, 1), 1), victim, thief))
    if not pairs:
        print("\n  no class takes an utterance from another")
        return
    pairs.sort(reverse=True)
    print("\n  what each class steals, worst first (argmax, before any gate)")
    print("     n    rate  true class  fires as")
    for n, rate, victim, thief in pairs[:limit]:
        flag = "   <-- " if rate >= 0.10 and thief != si_features.UNKNOWN else ""
        print("  %4d  %6.1f%%  %-10s  %s%s"
              % (n, 100 * rate, victim, thief, flag))
    if len(pairs) > limit:
        print("  (%d more pairs)" % (len(pairs) - limit))


def print_per_class(rows, thresh, margin):
    print("\n  per-class recall at thresh %.3f, margin %.2f (worst first)"
          % (thresh, margin))
    print("  recall  class       fires  wrong")
    for recall, label, fires, wrong in per_class(rows, thresh, margin):
        print("   %5.3f  %-10s %6d %6d" % (recall, label, fires, wrong))


def confusion(rows, thresh, margin):
    table = {}
    for truth, ranked, _name in rows:
        if isinstance(truth, tuple):
            key = "(variant)"
        elif truth is None:
            key = "(oov)"
        else:
            key = truth
        fired = None
        if ranked:
            best, label = ranked[0]
            gap = best - ranked[1][0] if len(ranked) > 1 else 1.0
            if (label != si_features.UNKNOWN and best >= thresh
                    and gap >= margin):
                fired = label
        table.setdefault(key, {})
        table[key][fired] = table[key].get(fired, 0) + 1
    return table


# --- inputs ----------------------------------------------------------------

def truth_for(row):
    """A corpus row -> the three-way truth `rows_from_probs` documents.

    `category` is used when the manifest carries it, because the corpus knows
    which utterances are benign variants and this file should not have to
    re-derive it from a filename. The `say_corpus.OOV_VARIANTS` lookup stays as
    the fallback for manifests that predate the field -- but note it is a
    fallback, not a cross-check: a variant the corpus declares and this table
    does not know about must still score as benign.
    """
    if row.get("category") == "variant" and row.get("variant_of"):
        return ("variant", row["variant_of"])
    if row["label"] != si_features.UNKNOWN:
        return row["label"]
    if row.get("category") == "word":
        return row["label"]
    variant = say_corpus.OOV_VARIANTS.get(str(row["form"]).replace("'", "_")
                                          .replace(" ", "_"))
    if variant:
        return ("variant", variant)
    return None


def load_takes(takes_dir, cache=None, jobs=None):
    """The user's enrolment recordings -> (patches, truths, names).

    `tools/enrol.py` writes `<word>_<nn>.wav` plus a `manifest.json` listing
    them. The manifest is preferred when present because it also records which
    takes were rejected at capture time; a bare directory scan is the fallback.

    These went through the board's microphone, its codec and its gain, which is
    the whole point: everything else in this experiment is `say` output that
    never touched an analogue path.
    """
    import numpy as np
    manifest = os.path.join(takes_dir, "manifest.json")
    entries = []
    if os.path.exists(manifest):
        with open(manifest) as fh:
            doc = json.load(fh)
        for entry in doc.get("entries", []):
            path = os.path.join(takes_dir, entry["file"])
            if os.path.exists(path):
                entries.append((path, str(entry["label"]).lower()))
    else:
        for name in sorted(os.listdir(takes_dir)):
            if name.endswith(".wav"):
                entries.append((os.path.join(takes_dir, name),
                                name.rsplit("_", 1)[0].lower()))
    rows = []
    for path, form in entries:
        label = vocab.label_of(form) or (form if form in vocab.LABELS
                                         else si_features.UNKNOWN)
        rows.append({"wav": path, "form": form, "label": label,
                     "voice": "user", "split": "real"})
    if not rows:
        return None, None, None, []
    si_features.extract_all(rows, cache, jobs, progress=False)
    kept = [r for r in rows if r["patch"]]
    dropped = len(rows) - len(kept)
    if not kept:
        return None, None, None, rows
    x = np.zeros((len(kept), si_features.N_FRAMES, si_features.N_BANDS),
                 dtype=np.int8)
    for i, row in enumerate(kept):
        x[i] = np.frombuffer(row["patch"], dtype=np.int8).reshape(
            si_features.N_FRAMES, si_features.N_BANDS)
    truths = [truth_for(r) for r in kept]
    names = [os.path.basename(r["wav"]) for r in kept]
    print("  takes: %d utterances, %d endpointed away" % (len(kept), dropped))
    return x, truths, names, rows


# --- device score dumps ---------------------------------------------------

SCOREDUMP_TAG = "SCORE"


def read_score_dump(path, classes=CLASSES, only_set=None, only_runtime=None):
    """A device's own printed scores -> [(name, probs)], in class order.

    **Format, and it is deliberately dull.** One tagged line per utterance,
    anywhere in the stream:

        SCORE name=<file> frames=<n> clipped=<n> p=<c0>,<c1>,...,<c21>

    - Lines that do not begin with `SCORE` are **ignored**, so a whole serial
      capture can be pasted in unedited. That is the point: a format that needs
      a human to cut the log out of the transcript is a format that gets cut
      wrong at 08:30 with somebody waiting.
    - `p=` is exactly `len(classes)` floats, comma-separated, in the class order
      of the model's `.json`. Six decimals is plenty -- the outputs are
      multiples of 1/256 -- but any parseable float is accepted.
    - `frames` and `clipped` are optional and carried through for diagnosis.
    - **`set=` must be filtered on when a dump mixes populations, and it is not
      optional to think about.** A dump may carry `set=takes` (real utterances,
      ground truth recoverable from the filename) alongside `set=bitexact` (the
      `kw_unknown_*` patches, whose names encode *an earlier model's
      predictions* and are not ground truth at all). Scored together, those 8
      are silently treated as must-stay-silent negatives, and the sweep then
      reports a number that is right for a partly fictional reason. Pass
      `only_set="takes"`.
    - `runtime=` likewise, when one dump carries two runtimes for the same
      inputs -- which is the paired form worth asking for.
    - An optional `q=` field may carry the raw quantised output bytes as
      integers instead of, or as well as, `p=`. If both are present `q=` wins,
      because it is exact and `p=` has been through a printf.

    Returns rows in file order. Duplicated names are kept, not merged: a device
    scoring the same take twice is data about the device.
    """
    rows = []
    n_fields = len(classes)
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line.startswith(SCOREDUMP_TAG):
                continue
            fields = {}
            for token in line[len(SCOREDUMP_TAG):].strip().split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    fields[k] = v
            name = fields.get("name")
            if name is None:
                raise ValueError("%s:%d: SCORE line has no name=" % (path, lineno))
            if "q" in fields:
                raw = [int(v) for v in fields["q"].split(",") if v != ""]
                probs = [v / 256.0 for v in raw]
            elif "p" in fields:
                probs = [float(v) for v in fields["p"].split(",") if v != ""]
            else:
                raise ValueError("%s:%d: SCORE line has neither p= nor q="
                                 % (path, lineno))
            if len(probs) != n_fields:
                raise ValueError(
                    "%s:%d: %d scores for %d classes -- the device and this "
                    "harness disagree about the model, which is exactly the "
                    "mismatch this check exists to stop"
                    % (path, lineno, len(probs), n_fields))
            if only_set is not None and fields.get("set") != only_set:
                continue
            if only_runtime is not None and fields.get("runtime") != only_runtime:
                continue
            rows.append({"name": name, "probs": probs,
                         "set": fields.get("set"),
                         "runtime": fields.get("runtime"),
                         "frames": int(fields.get("frames", 0)),
                         "clipped": int(fields.get("clipped", 0))})
    if not rows:
        raise ValueError("%s: no `%s` lines found%s"
                         % (path, SCOREDUMP_TAG,
                            "" if only_set is None and only_runtime is None
                            else " matching that set/runtime"))
    return rows


def rows_from_dump(dump, truth_for_name):
    """Device dump -> the (truth, ranked, name) shape the sweep already takes.

    So a device's scores go through **the same** `measure`, `sweep`,
    `recommend`, `print_steals` and `print_per_class` as everything else. The
    retune is then not a second implementation that has to be trusted -- it is
    the existing one with a different input.
    """
    import numpy as np
    probs = np.array([d["probs"] for d in dump], dtype="float64")
    names = [d["name"] for d in dump]
    return rows_from_probs(probs, [truth_for_name(n) for n in names], names)


def truth_for_take_name(name):
    """`mother_01.wav` -> "mother"; anything not a keyword -> must stay silent.

    The same convention `load_takes` uses, factored out so a device dump can be
    scored without the WAVs being present.
    """
    form = os.path.basename(name)
    if form.endswith(".wav"):
        form = form[:-4]
    form = form.rsplit("_", 1)[0].lower()
    label = vocab.label_of(form)
    return label if label else None


def predict(model_path, x):
    """int8 patches -> probabilities, from a .tflite or a .keras file."""
    if model_path.endswith(".tflite"):
        import si_train
        with open(model_path, "rb") as fh:
            blob = fh.read()
        return si_train.tflite_predict(blob, x)
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    return model.predict(x.astype("float32")[..., None], verbose=0)


# --- CLI -------------------------------------------------------------------

def cmd_device_scores(path, only_set=None, only_runtime=None):
    """Retune the gates against scores the device printed.

    This is the fallback path for the morning: if the device's arithmetic is
    verified bit-identical to the host's, the operating point transfers by
    construction and this is unnecessary. If it is not, every threshold in
    `docs/speaker-independent.md` was tuned on a model the board does not run,
    and this re-derives them from what it does.
    """
    dump = read_score_dump(path, only_set=only_set, only_runtime=only_runtime)
    rows = rows_from_dump(dump, truth_for_take_name)
    n_kw = sum(1 for r in rows if isinstance(r[0], str))
    print("device scores: %s%s%s"
          % (path,
             "" if only_set is None else "  set=%s" % only_set,
             "" if only_runtime is None else "  runtime=%s" % only_runtime))
    sets = sorted(set(d["set"] for d in dump if d["set"]))
    runtimes = sorted(set(d["runtime"] for d in dump if d["runtime"]))
    if only_set is None and len(sets) > 1:
        print("WARNING: this dump mixes sets %s and none was selected. If any "
              "of them lack ground truth, the numbers below are partly "
              "fiction. Pass --set." % ", ".join(sets))
    if only_runtime is None and len(runtimes) > 1:
        print("WARNING: this dump mixes runtimes %s and none was selected; "
              "they are being scored as one population. Pass --runtime."
              % ", ".join(runtimes))
    print("%d utterances (%d in-vocabulary, %d must stay silent)"
          % (len(rows), n_kw, len(rows) - n_kw))
    dropped = [d["name"] for d in dump if d["frames"] == 0]
    if dropped:
        print("%d reported 0 frames: %s" % (len(dropped), ", ".join(dropped[:6])))
    acc, hit, n = top1_accuracy(rows)
    print("top-1 over in-vocabulary utterances: %.3f (%d of %d)" % (acc, hit, n))
    uacc, uhit, un = unknown_accuracy(rows)
    print("must-stay-silent classified `unknown` outright: %.3f (%d of %d)"
          % (uacc, uhit, un))
    print_sweep(rows, "threshold sweep, DEVICE scores")
    print_steals(rows)
    clean, _ = recommend(rows)
    if clean:
        print_per_class(rows, clean[1], clean[0])
        print("\n  the gates to ship, from the device's own numbers:")
        print("    THRESHOLD = %.3f, MARGIN = %.4f" % (clean[1], clean[0]))
    print("\nFor comparison, the host `.tflite` operating point currently in "
          "`si_spot.py`:\n  THRESHOLD 0.35, MARGIN 2/256, TIE_FLOOR 0.49 "
          "-- precision 1.000, recall 0.500 on these 22 takes.")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model",
                    help=".tflite (int8, what the device runs) or .keras "
                         "(float, for the comparison)")
    ap.add_argument("manifest", help="the synthetic corpus manifest")
    ap.add_argument("--takes", default=None,
                    help="directory of real enrolment recordings. Point this "
                         "at keywords AND negatives together -- a "
                         "keywords-only directory cannot see a fire on an "
                         "ordinary word, which is the failure this project "
                         "cares about, and reports a flatteringly low "
                         "threshold as a result")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device-scores", default=None,
                    help="a device score dump (see read_score_dump). Sweeps "
                         "the gates against what the board actually produced "
                         "instead of against the host model, and skips the "
                         "synthetic evaluation entirely.")
    ap.add_argument("--set", default=None,
                    help="only score SCORE lines with this set= tag. Required "
                         "when a dump mixes populations that do not all carry "
                         "ground truth.")
    ap.add_argument("--runtime", default=None,
                    help="only score SCORE lines with this runtime= tag")
    ap.add_argument("--split", default="val",
                    help="which manifest split is the held-out voices")
    args = ap.parse_args(argv[1:])

    if args.device_scores:
        return cmd_device_scores(args.device_scores, args.set, args.runtime)

    print("model: %s" % args.model)
    print("feature key: %s" % si_features.feature_key())

    rows = si_features.read_manifest(args.manifest)
    si_features.check_split(rows)
    si_features.check_distinct_voices(rows)
    si_features.extract_all(rows, args.cache, args.jobs)
    val = [r for r in rows if r["split"] == args.split and r["patch"]]
    if not val:
        print("no utterances in split %r" % args.split)
        return 1

    import numpy as np
    x = np.zeros((len(val), si_features.N_FRAMES, si_features.N_BANDS),
                 dtype=np.int8)
    for i, row in enumerate(val):
        x[i] = np.frombuffer(row["patch"], dtype=np.int8).reshape(
            si_features.N_FRAMES, si_features.N_BANDS)
    probs = predict(args.model, x)
    syn = rows_from_probs(probs, [truth_for(r) for r in val],
                          [os.path.basename(r["wav"]) for r in val])

    print("\n=== 1. SYNTHETIC, held-out voices (%d voices, %d utterances) ==="
          % (len(set(r["voice"] for r in val)), len(val)))
    acc, hit, n = top1_accuracy(syn)
    print("top-1 over in-vocabulary utterances: %.3f (%d of %d)"
          % (acc, hit, n))
    uacc, uhit, un = unknown_accuracy(syn)
    print("must-stay-silent classified `unknown` outright: %.3f (%d of %d)"
          % (uacc, uhit, un))
    print_sweep(syn, "threshold sweep (top-1 softmax probability)")
    clean_syn, _ = recommend(syn)
    if clean_syn:
        print_per_class(syn, clean_syn[1], clean_syn[0])
    print_steals(syn)

    if args.takes and os.path.isdir(args.takes):
        xr, truths, names, _ = load_takes(args.takes, args.cache, args.jobs)
        if xr is None:
            print("\n=== 2. REAL SPEAKER: no usable recordings in %s ==="
                  % args.takes)
        else:
            probs = predict(args.model, xr)
            real = rows_from_probs(probs, truths, names)
            print("\n=== 2. REAL SPEAKER, %s (%d utterances) ==="
                  % (args.takes, len(xr)))
            acc_r, hit_r, n_r = top1_accuracy(real)
            print("top-1 over in-vocabulary utterances: %.3f (%d of %d)"
                  % (acc_r, hit_r, n_r))
            # Signed, and named in the direction it actually went. An earlier
            # version said "points lost" unconditionally and printed
            # "-7.4 points lost" when real speech had *out*performed the
            # held-out synthetic voices -- which states the project's central
            # result backwards to anyone reading the output rather than the
            # doc.
            delta = 100 * (acc_r - acc)
            print("the gap that is the result: %.3f synthetic -> %.3f real, "
                  "%+.1f points (%s)"
                  % (acc, acc_r, delta,
                     "real speech did better" if delta > 0 else
                     "real speech did worse" if delta < 0 else "no change"))
            print_sweep(real, "threshold sweep, real speaker")
            clean, _ = recommend(real)
            print_steals(real)
            if clean:
                print_per_class(real, clean[1], clean[0])
                print("\nconfusion at the recommended real-speaker setting")
                for truth, fires in sorted(confusion(real, clean[1],
                                                     clean[0]).items()):
                    bits = ", ".join(
                        "%s x%d" % (k or "(silent)", v)
                        for k, v in sorted(fires.items(),
                                           key=lambda kv: (kv[0] or "")))
                    print("  %-12s %s" % (truth, bits))
    else:
        print("\n=== 2. REAL SPEAKER: not run (no --takes directory) ===")
        print("Until this runs, nothing here answers the question the "
              "experiment was set up to ask.")

    print("\n=== 3. THE INCUMBENT ===")
    print("DTW, synthetic corpus, threshold 750 margin 120: "
          "precision 1.000, recall 0.966 (85 of 88).")
    print("Its real-speaker number does not exist yet either -- it needs "
          "enrolment templates. Compare like with like when it does.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
