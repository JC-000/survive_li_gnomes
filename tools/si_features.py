#!/usr/bin/env python3
"""Turn a labelled corpus into fixed-size int8 log-mel patches for a CNN.

This is the input half of the speaker-independent experiment; `tools/si_train.py`
is the model and `tools/si_eval.py` is the measurement. It exists to answer one
question: can a classifier
trained on macOS `say` voices recognise a real person? Everything here is
arranged so that the answer is not flattered.

**It is not a second front end.** Endpointing is `src/vad.py` through
`tools/vad.py`, exactly what the device runs, and the features are
`mfcc.logmel_q8` -- the Q8 log2 filterbank the existing front end already
computes on its way to every cepstrum. `python3 tools/mfcc.py --selftest`
checks it against the `mel` entry of all five cases in
`src/speech_fixtures.py`, which is the array the viper port is already required
to reproduce bit for bit, and it matches 5 of 5. So the device has to add a tap
on an array it is already filling, plus the integer normalisation below, and
nothing else needs verifying on hardware.

The normalisation is the one design decision in this file, and it is written to
be reproducible on the board in integer arithmetic:

    m[i]    = trunc_toward_zero(sum over t of x[t][i], n_frames)   per band
    y[t][i] = clamp((x[t][i] - m[i] + (1 << (S-1))) >> S, -128, 127)

Per-*band* mean subtraction rather than a global one, because that is the
log-mel analogue of the cepstral mean normalisation the DTW path already uses,
and it is there for the same reason: a fixed channel colouration -- this
microphone, this room, this distance from the mouth -- is additive in the log
domain, so subtracting each band's own mean removes it exactly. A global mean
would only remove the level. Since the whole experiment is about surviving a
channel and speaker change, the stage that removes channel is not optional.

Truncation toward zero, not floor, for the same reason `mfcc_q8` does it: that
is what a hardware divide gives and the device has no cheap floor-divide.

`--stats` reports the clipping rate, because a shift that clips is a silently
lossy feature and the loss would show up only as a slightly worse model.
"""

import argparse
import array
import json
import os
import struct
import sys
import time

# Order matters and it is the order `tools/dtw.py` uses: `src/` goes on first
# and `tools/` on top of it, so a name present in both -- `vad` is -- resolves
# to the host wrapper. Reversing these two lines gets `src/vad.py`, which has
# no `read_wav` because it must stay importable on a board with no `wave`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import mfcc                                    # noqa: E402
import vad as hostvad                          # noqa: E402
import vocab                                   # noqa: E402


# Frames of input the model sees. Measured over 880 `say` utterances of all 22
# spoken forms in 20 English voices at 150 and 205 wpm, endpointed by
# `src/vad.py`: min 35, median 55, p95 79, max 130 frames.
#
#     N_FRAMES  utterances centre-cropped
#           64  23.3%
#           80   4.3%
#           96   1.7%
#          112   0.8%
#
# 80 is where the curve flattens. Going on to 96 buys 2.6 points of cropping
# for another 20% of multiply-accumulates, and the utterances still cropped at
# 80 are outliers of a specific kind -- the old formant voices, on which the
# endpointer runs long -- rather than genuinely long words. Note the docs'
# "28..61 frames" figure is for one voice (Samantha) at 172-178 wpm; twenty
# voices across a wider rate range are half again as long, which is worth
# knowing before quoting that range for anything else.
N_FRAMES = 80
N_BANDS = mfcc.N_MEL

# Q8 log2 -> the model's input scale. 4 gives Q4 log2, 1 LSB = 1/16 octave,
# the same resolution the DTW features carry. Measured over 150 of those
# utterances, the shift is a straight trade of resolution against clipping:
#
#     shift  1 LSB      clipped
#         3  0.188 dB   20.2%
#         4  0.376 dB    2.3%
#         5  0.753 dB    0.0%
#
# 4, because what clips at 4 is a band more than 48 dB below its own mean over
# the utterance -- the noise floor inside a silent frame, not any part of the
# word -- so the clip acts as a spectral floor. `docs/speech.md` measured such
# a floor neutral for DTW, and here it comes free with the wider resolution.
# Shift 5 spends half the int8 range to preserve a value nothing depends on.
INPUT_SHIFT = 4

UNKNOWN = "unknown"
CLASSES = tuple(vocab.LABELS) + (UNKNOWN,)


def feature_key():
    """Identifies the feature contract, so a stale cache cannot be reused.

    Includes `mfcc._cache_key`-style front-end constants plus this file's own,
    because a change to either invalidates every cached patch.
    """
    import hashlib
    parts = []
    for name in sorted(dir(mfcc)):
        if name.isupper() and type(getattr(mfcc, name)) in (int, float):
            parts.append("%s=%r" % (name, getattr(mfcc, name)))
    parts.append("N_FRAMES=%d" % N_FRAMES)
    parts.append("INPUT_SHIFT=%d" % INPUT_SHIFT)
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return "si%d-%s" % (mfcc.TEMPLATE_FORMAT, digest[:16])


# --- the feature itself ----------------------------------------------------

def normalise(rows, shift=INPUT_SHIFT):
    """Per-band mean subtraction and rescale to int8. Returns (patch, clipped).

    `rows` is what `mfcc.logmel_q8` returns: a list of N_BANDS Q8 log2 values
    per frame. Integer throughout, and every operation has a viper equivalent.
    """
    n = len(rows)
    if n == 0:
        return [], 0
    out = []
    clipped = 0
    means = [0] * N_BANDS
    for i in range(N_BANDS):
        total = 0
        for t in range(n):
            total += rows[t][i]
        means[i] = total // n if total >= 0 else -((-total) // n)
    half = 1 << (shift - 1)
    for t in range(n):
        row = [0] * N_BANDS
        for i in range(N_BANDS):
            v = (rows[t][i] - means[i] + half) >> shift
            if v > 127:
                v = 127
                clipped += 1
            elif v < -128:
                v = -128
                clipped += 1
            row[i] = v
        out.append(row)
    return out, clipped


def fit(patch, n_frames=N_FRAMES):
    """Centre-pad or centre-crop to exactly `n_frames` rows.

    Padding with zero is padding with the band mean, which after the
    subtraction above is what silence normalises to -- so a short word is
    surrounded by something the model can read as "nothing", not by a spurious
    edge. Cropping takes the middle, because the endpointer has already put the
    word in the middle and the parts a long utterance loses are its margins.
    """
    n = len(patch)
    if n == n_frames:
        return list(patch)
    if n > n_frames:
        start = (n - n_frames) // 2
        return list(patch[start:start + n_frames])
    pad = n_frames - n
    before = pad // 2
    zero = [0] * N_BANDS
    return [list(zero) for _ in range(before)] + list(patch) + \
           [list(zero) for _ in range(pad - before)]


def patch_for_samples(samples, t=None):
    """int16 samples (already endpointed) -> (patch, n_raw_frames, clipped)."""
    rows = mfcc.logmel_q8(samples, t)
    if not rows:
        return None, 0, 0
    norm, clipped = normalise(rows)
    return fit(norm), len(rows), clipped


def patch_for_wav(path, t=None, trim=True):
    """WAV -> (patch, n_raw_frames, clipped). None if endpointing rejects it.

    Endpointing is the device's own `src/vad.py`. An utterance the VAD rejects
    is one the device would never present to the model either, so dropping it
    here keeps the training distribution and the run-time distribution the
    same. The count of drops is reported rather than swallowed.
    """
    samples = hostvad.read_wav(path)
    if trim:
        samples = hostvad.trim(samples)
        if samples is None:
            return None, 0, 0
    return patch_for_samples(samples, t)


# --- corpus plumbing -------------------------------------------------------

def read_manifest(path):
    """One row per utterance. JSONL or CSV, with a header naming the columns.

    Required per row: a path to the WAV, a label, and a **voice**. The voice is
    not optional -- the entire point of the split is that no voice appears in
    both halves, and a manifest without it cannot be checked for that.
    """
    rows = []
    root = os.path.dirname(os.path.abspath(path))
    with open(path) as fh:
        text = fh.read()
    stripped = text.lstrip()
    if stripped.startswith("{") and '"entries"' in stripped[:4096]:
        # `corpus-tts/manifest.json`: one JSON object with an `entries` list.
        # It carries a `category` of word / variant / unknown, which is a
        # better source of truth than inferring benign variants from a name --
        # see `si_eval.truth_for`.
        rows = json.loads(text)["entries"]
    elif stripped.startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        import csv
        import io
        for row in csv.DictReader(io.StringIO(text)):
            rows.append(dict(row))
    out = []
    for row in rows:
        wav = _pick(row, ("wav", "path", "file", "filename"))
        label = _pick(row, ("label", "class", "word", "truth"))
        voice = _pick(row, ("voice", "speaker", "spk"))
        split = _pick(row, ("split", "set", "fold")) or ""
        category = _pick(row, ("category",))
        # A null label with `category: unknown` is a deliberate negative, not a
        # malformed row. Only the combination is unambiguous, so it is checked
        # as a combination.
        if label is None and category == "unknown":
            label = UNKNOWN
        if wav is None or label is None:
            raise ValueError("manifest row lacks a wav or a label: %r" % (row,))
        if voice is None:
            raise ValueError(
                "manifest row has no voice column: %r\n"
                "A voice-disjoint split cannot be verified without it, and an "
                "unverified split is the one way this experiment can report a "
                "number that means nothing." % (row,))
        if not os.path.isabs(wav):
            wav = os.path.normpath(os.path.join(root, wav))
        # A spoken form (SICK) maps to its class (sad); anything not in the
        # vocabulary is the unknown class, which is most of the corpus by
        # design and is the class precision depends on.
        cls = vocab.label_of(label) or (label if label in vocab.LABELS else UNKNOWN)
        # A benign morphological variant (MOTHER'S, COMPUTERS) is labelled with
        # the class it relates to, but it is *not* that class for training:
        # calling COMPUTERS a COMPUTER teaches the model a word it will never
        # be asked for, while calling it unknown teaches the rejection the
        # threshold depends on. Firing on it at run time is still correct
        # DOCTOR behaviour, which is why `si_eval` scores it as neither a hit
        # nor a miss -- the training label and the scoring rule differ on
        # purpose, and `category` is what keeps them straight.
        if category == "variant":
            cls = UNKNOWN
        out.append({"wav": wav, "form": row.get("text") or label,
                    "label": cls, "voice": voice, "split": split,
                    "category": category,
                    "variant_of": vocab.label_of(label) or label
                                  if category == "variant" else None})
    return out


def check_distinct_voices(rows, sample=None):
    """Raise if two voice names produced byte-identical audio.

    `say -v NotInstalled` does not fail. It **silently renders in the system
    default voice**, so a corpus built from a hand-written voice list quietly
    contains N copies of one speaker under N different names -- and if those
    names land on both sides of the split, the speaker-independent validation
    number is measuring a voice against itself and will read near-perfect.

    Found the hard way: in a throwaway 6-voice corpus, `Samantha`, `Fiona`,
    `Tom` and `Alex` produced the same MD5, because only Samantha was
    installed. The DTW control then scored those utterances at distance
    **exactly 0**, which is the only reason it was noticed at all.

    Hashing whole files is the cheap version of the check and catches the
    fallback exactly, because the fallback is byte-identical rather than
    merely similar. It does not catch two genuinely different voices that
    happen to sound alike -- that is a corpus-design question, and
    `corpus-tts/roster.json` handles it with fingerprints and families.
    """
    import hashlib
    seen = {}
    clashes = {}
    for row in rows if sample is None else rows[:sample]:
        try:
            with open(row["wav"], "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
        except OSError:
            continue
        prev = seen.get(digest)
        if prev is None:
            seen[digest] = row["voice"]
        elif prev != row["voice"]:
            clashes.setdefault(tuple(sorted((prev, row["voice"]))), 0)
            clashes[tuple(sorted((prev, row["voice"])))] += 1
    if clashes:
        lines = ["%s and %s share %d identical file(s)" % (a, b, n)
                 for (a, b), n in sorted(clashes.items(),
                                         key=lambda kv: -kv[1])[:8]]
        raise ValueError(
            "voice names that are not distinct voices:\n  %s\n"
            "`say -v` falls back to the default voice for a name it does not "
            "have, without failing. Check the roster against `say -v '?'`."
            % "\n  ".join(lines))
    return len(seen)


def _pick(row, names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def check_split(rows):
    """Raise if any voice appears in more than one split. Returns a summary.

    This is a hard failure and not a warning. A leaked voice inflates the
    synthetic validation number, which is the number the whole experiment is
    measured against, and it does so silently -- the model still trains, the
    curves still look right, and the result is simply not about
    speaker-independence any more.
    """
    where = {}
    for row in rows:
        where.setdefault(row["voice"], set()).add(row["split"])
    leaked = sorted(v for v, s in where.items() if len(s) > 1)
    if leaked:
        raise ValueError("%d voice(s) appear in more than one split: %s"
                         % (len(leaked), ", ".join(leaked[:8])))
    counts = {}
    for row in rows:
        counts.setdefault(row["split"], [0, set()])
        counts[row["split"]][0] += 1
        counts[row["split"]][1].add(row["voice"])
    return dict((k, (v[0], len(v[1]))) for k, v in counts.items())


# --- cached extraction -----------------------------------------------------

_MAGIC = b"SIP1"


def _cache_path(cache_dir, wav):
    import hashlib
    h = hashlib.sha1(os.path.abspath(wav).encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, h + ".sip")


def _cache_open(cache_dir):
    stamp = os.path.join(cache_dir, "_key")
    key = feature_key()
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
    with open(stamp, "w") as fh:
        fh.write(key)


def extract_one(args):
    """(wav, cache_dir) -> (bytes_or_None, n_raw_frames, clipped). Picklable."""
    wav, cache_dir = args
    cached = _cache_path(cache_dir, wav) if cache_dir else None
    if cached and os.path.exists(cached):
        with open(cached, "rb") as fh:
            blob = fh.read()
        if len(blob) < 12:
            return None, 0, 0
        n_raw, clipped = struct.unpack("<II", blob[4:12])
        return blob[12:], n_raw, clipped
    patch, n_raw, clipped = patch_for_wav(wav)
    if patch is None:
        body = b""
    else:
        flat = array.array("b")
        for row in patch:
            flat.extend(row)
        body = flat.tobytes()
    if cached:
        with open(cached, "wb") as fh:
            fh.write(_MAGIC + struct.pack("<II", n_raw, clipped) + body)
    return (body or None), n_raw, clipped


def extract_all(rows, cache_dir=None, jobs=None, progress=True):
    """Adds `patch` (bytes, N_FRAMES*N_BANDS int8) to every row it can.

    Rows the endpointer rejects are returned with `patch=None`; the caller
    decides, and the count is printed rather than hidden, because a corpus that
    quietly loses a fifth of its utterances to endpointing is a finding.
    """
    if cache_dir:
        _cache_open(cache_dir)
    jobs = jobs or (os.cpu_count() or 1)
    # macOS spawns rather than forks, so every worker re-imports `__main__`.
    # From a REPL or a `python - <<EOF` heredoc there is no file to re-import
    # and every worker dies in `runpy` with a `FileNotFoundError` naming
    # `<stdin>` -- a page of traceback that says nothing about this function.
    # Falling back to serial is slower and correct.
    if getattr(sys.modules.get("__main__"), "__file__", None) is None:
        jobs = 1
    work = [(row["wav"], cache_dir) for row in rows]
    t0 = time.time()
    results = []
    if jobs > 1 and len(work) > 8:
        import multiprocessing
        with multiprocessing.Pool(jobs) as pool:
            for i, res in enumerate(pool.imap(extract_one, work, chunksize=16)):
                results.append(res)
                if progress and i % 200 == 0:
                    sys.stderr.write("\r  features %d/%d" % (i, len(work)))
                    sys.stderr.flush()
    else:
        for i, item in enumerate(work):
            results.append(extract_one(item))
            if progress and i % 200 == 0:
                sys.stderr.write("\r  features %d/%d" % (i, len(work)))
                sys.stderr.flush()
    dropped = 0
    total_clip = 0
    for row, (body, n_raw, clipped) in zip(rows, results):
        row["patch"] = body
        row["n_raw"] = n_raw
        row["clipped"] = clipped
        total_clip += clipped
        if body is None:
            dropped += 1
    if progress:
        sys.stderr.write("\r  %d utterances in %.1f s, %d endpointed away, "
                         "%d clipped samples\n"
                         % (len(work), time.time() - t0, dropped, total_clip))
    return rows


def as_arrays(rows, classes=CLASSES):
    """Rows with patches -> (X int8 [n, N_FRAMES, N_BANDS], y int, voices)."""
    import numpy as np
    keep = [r for r in rows if r.get("patch")]
    x = np.zeros((len(keep), N_FRAMES, N_BANDS), dtype=np.int8)
    y = np.zeros((len(keep),), dtype=np.int32)
    index = dict((c, i) for i, c in enumerate(classes))
    voices = []
    for i, row in enumerate(keep):
        x[i] = np.frombuffer(row["patch"], dtype=np.int8).reshape(
            N_FRAMES, N_BANDS)
        y[i] = index[row["label"]]
        voices.append(row["voice"])
    return x, y, voices


# --- CLI -------------------------------------------------------------------

def cmd_stats(manifest, cache_dir, jobs):
    rows = read_manifest(manifest)
    print("feature key: %s" % feature_key())
    print("%d utterances, %d classes + unknown, %d frames x %d bands int8"
          % (len(rows), len(vocab.LABELS), N_FRAMES, N_BANDS))
    summary = check_split(rows)
    for split in sorted(summary):
        n, nv = summary[split]
        print("  split %-8s %6d utterances  %4d voices" % (split or "(none)", n, nv))
    print("%d distinct audio files across %d voice names"
          % (check_distinct_voices(rows), len(set(r["voice"] for r in rows))))
    extract_all(rows, cache_dir, jobs)

    kept = [r for r in rows if r["patch"]]
    if not kept:
        print("nothing survived endpointing")
        return 1
    lens = sorted(r["n_raw"] for r in kept)
    print("\nendpointed length, frames: min %d  p05 %d  median %d  p95 %d  max %d"
          % (lens[0], lens[len(lens) // 20], lens[len(lens) // 2],
             lens[len(lens) * 19 // 20], lens[-1]))
    over = sum(1 for n in lens if n > N_FRAMES)
    print("%d of %d (%.1f%%) exceed N_FRAMES=%d and are centre-cropped"
          % (over, len(lens), 100.0 * over / len(lens), N_FRAMES))
    total_clip = sum(r["clipped"] for r in kept)
    cells = len(kept) * N_FRAMES * N_BANDS
    print("clipping at INPUT_SHIFT=%d: %d of %d cells (%.4f%%)"
          % (INPUT_SHIFT, total_clip, cells, 100.0 * total_clip / cells))

    per = {}
    for row in kept:
        per.setdefault(row["label"], 0)
        per[row["label"]] += 1
    print("\nutterances per class")
    for label in sorted(per, key=lambda k: -per[k]):
        print("  %-10s %6d" % (label, per[label]))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest")
    ap.add_argument("--cache", default=None,
                    help="directory for cached patches; keyed on the feature "
                         "contract, so it self-invalidates")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv[1:])
    return cmd_stats(args.manifest, args.cache, args.jobs)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
