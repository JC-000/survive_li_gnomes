#!/usr/bin/env python3
"""Run the *incumbent* speaker-independently, as a control for the CNN.

Without this, a bad result from `tools/si_eval.py` has two explanations that
look identical and lead to opposite decisions:

- **the model is not good enough** -- try a bigger one, more augmentation,
  quantisation-aware training; or
- **synthetic voices do not transfer** -- in which case no model trained on
  them will work, and the whole approach should be abandoned rather than tuned.

So this runs DTW, unchanged, in the one configuration it is never used in:
templates enrolled from `say` voices in the **training** split, queries from
the held-out voices. Same corpus, same split, same endpointing, same front end,
same `dtw.measure` -- the only difference from `tools/dtw.py --eval` is that
template and query now come from different speakers.

    tools/si_dtw_control.py corpus/manifest.jsonl --voices 3 --limit 400

Reading it:

| CNN | DTW control | what it means |
| --- | --- | --- |
| good | bad | the CNN is genuinely learning across voices. Pursue it. |
| bad | bad | synthetic voices do not carry across speakers at all. |
| bad | good | the corpus is fine and the model is the problem. |
| good | good | speaker-independence is easy here and probably too
  easy -- suspect a split leak before believing it. |

The DTW control is **not** a deployable alternative: `--voices 3` already means
66 templates, which is the entire device budget, and it buys three synthetic
speakers rather than three takes of the real user. It is a measuring
instrument, not a candidate.

Slow on purpose -- it is the real matcher, in Python. `--limit` subsamples the
query set; the count actually scored is printed, because a control run on 40
utterances and one run on 400 are not the same evidence.
"""

import argparse
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import dtw as dtwmod                           # noqa: E402
import mfcc                                    # noqa: E402
import si_eval                                 # noqa: E402
import si_features                             # noqa: E402
import vad as hostvad                          # noqa: E402
import vocab                                   # noqa: E402


def features(path):
    """Endpoint and MFCC one WAV -> feature rows, or None.

    The same two calls `dtw.features_for` makes, without its cache: this tool
    runs once and the cache would outlive the corpus it was built from.
    """
    samples = hostvad.trim(hostvad.read_wav(path))
    if samples is None:
        return None
    frames = mfcc.mfcc(samples)
    return frames or None


def build_templates(rows, voices, takes):
    """{label: [frames, ...]} from the chosen training voices.

    `takes` caps the templates per spoken form so the control stays inside the
    device's real budget. Three voices x one take is 66 templates, which is
    exactly what `docs/speech.md` measures the incumbent at -- so the two
    numbers are comparable in size as well as in method.
    """
    chosen = {}
    for row in rows:
        if row["voice"] not in voices:
            continue
        if row["label"] == si_features.UNKNOWN:
            continue
        key = (row["voice"], row["form"])
        chosen.setdefault(key, []).append(row)
    templates = {}
    n = 0
    for (_voice, form), group in sorted(chosen.items()):
        label = vocab.label_of(form)
        if label is None:
            continue
        for row in sorted(group, key=lambda r: r["wav"])[:takes]:
            frames = features(row["wav"])
            if frames:
                templates.setdefault(label, []).append(frames)
                n += 1
    return templates, n


def _real_rows(takes, takes_oov):
    """Enrolment WAVs -> corpus-shaped rows, keywords and negatives together.

    Both directories are read into one query set because precision is only
    meaningful over both: keywords alone can only count "fired as the wrong
    keyword", and the failure this project fears is firing at all on something
    ordinary.
    """
    rows = []
    for directory, is_oov in ((takes, False), (takes_oov, True)):
        if not directory or not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".wav"):
                continue
            form = name.rsplit("_", 1)[0].lower()
            label = si_features.UNKNOWN if is_oov else (
                vocab.label_of(form) or si_features.UNKNOWN)
            rows.append({"wav": os.path.join(directory, name), "form": form,
                         "label": label, "voice": "human", "split": "real",
                         "category": "unknown" if is_oov else "word",
                         "variant_of": None})
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest")
    ap.add_argument("--voices", type=int, default=3,
                    help="how many training voices supply templates")
    ap.add_argument("--template-takes", type=int, default=1,
                    help="templates per spoken form per voice")
    ap.add_argument("--split", default="val", help="which split to query with")
    ap.add_argument("--limit", type=int, default=400,
                    help="cap on query utterances; 0 for all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--takes", default=None,
                    help="query with a directory of real enrolment recordings "
                         "instead of the held-out synthetic split")
    ap.add_argument("--takes-oov", default=None,
                    help="directory of real out-of-vocabulary recordings; "
                         "without these, real-speaker precision cannot count "
                         "the failure that matters")
    args = ap.parse_args(argv[1:])

    rows = si_features.read_manifest(args.manifest)
    si_features.check_split(rows)
    si_features.check_distinct_voices(rows)
    train = [r for r in rows if r["split"] == "train"]
    query = [r for r in rows if r["split"] == args.split]
    if not train or not query:
        print("need both a `train` split and a %r split" % args.split)
        return 1

    # Voices are picked by a seeded shuffle rather than by taking the first
    # few, because manifests come out sorted and the first few voices are
    # usually one family.
    rng = random.Random(args.seed)
    train_voices = sorted(set(r["voice"] for r in train))
    rng.shuffle(train_voices)
    picked = set(train_voices[:args.voices])
    print("front end: %s" % dtwmod._cache_key())
    print("templates from %d training voice(s): %s"
          % (len(picked), ", ".join(sorted(picked))))

    t0 = time.time()
    templates, n_templates = build_templates(train, picked, args.template_takes)
    print("%d classes, %d templates, built in %.1f s"
          % (len(templates), n_templates, time.time() - t0))
    matcher = dtwmod.Matcher(templates)

    if args.takes:
        # Real speech replaces the synthetic query set. Templates still come
        # from `say` voices -- that is the whole point: it is the incumbent,
        # enrolled synthetically, meeting a person.
        query = _real_rows(args.takes, args.takes_oov)
        print("querying with %d REAL utterances (%d in-vocabulary, %d "
              "must-stay-silent)"
              % (len(query),
                 sum(1 for r in query if r["label"] != si_features.UNKNOWN),
                 sum(1 for r in query if r["label"] == si_features.UNKNOWN)))
        args.limit = 0

    if args.limit and len(query) > args.limit:
        rng.shuffle(query)
        query = query[:args.limit]
    print("querying with %d utterances from %d held-out voice(s)"
          % (len(query), len(set(r["voice"] for r in query))))

    scored = []
    t0 = time.time()
    for i, row in enumerate(query):
        frames = features(row["wav"])
        if not frames:
            continue
        scored.append((si_eval.truth_for(row), matcher.scores(frames),
                       os.path.basename(row["wav"])))
        if i % 20 == 0:
            sys.stderr.write("\r  matching %d/%d" % (i, len(query)))
            sys.stderr.flush()
    sys.stderr.write("\r  matched %d utterances in %.1f s\n"
                     % (len(scored), time.time() - t0))

    hit = n_in = 0
    for truth, ranked, _name in scored:
        if isinstance(truth, str) and ranked:
            n_in += 1
            if ranked[0][1] == truth:
                hit += 1
    print("\nDTW top-1 over in-vocabulary utterances: %.3f (%d of %d)"
          % (hit / n_in if n_in else 0.0, hit, n_in))

    print("\nthreshold sweep (score is Q4 log2 units per unit of path weight)")
    print("  margin  thresh   prec   recall    tp  wrong   oov  benign")
    seen = set()
    for margin, th, p, r, tp, fw, fo, _n, ben in dtwmod.sweep(scored):
        key = (margin, tp, fw, fo, ben)
        if key in seen:
            continue
        seen.add(key)
        print("  %6d  %6d  %5.3f   %5.3f  %4d  %5d  %4d  %6d"
              % (margin, th, p, r, tp, fw, fo, ben))
    clean, p95 = dtwmod.recommend(scored)
    if clean:
        margin, th, p, r, tp, fw, fo, n, ben = clean
        print("\nRECOMMENDED (zero false fires): threshold %d, margin %d"
              % (th, margin))
        print("  precision 1.000, recall %.3f (%d of %d in-vocabulary),"
              " %d benign variant fires" % (r, tp, n, ben))
    else:
        print("\nno setting fires without a false positive")
    if p95:
        margin, th, p, r, tp, fw, fo, n, ben = p95
        print("at 95%% precision: threshold %d, margin %d -> recall %.3f"
              % (th, margin, r))

    print("\nFor reference, the same matcher speaker-*dependently* on the "
          "single-voice corpus:\n  threshold 750, margin 120 -> precision "
          "1.000, recall 0.966 (docs/speech.md).")
    print("Any shortfall here is what changing speaker costs DTW, measured "
          "rather than assumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
