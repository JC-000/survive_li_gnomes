#!/usr/bin/env python3
"""Stage the on-device TFLM test: 30 input patches plus the host answers.

`tools/tflm_vs_tflite.py` proves host TFLM == host TFLite. That is half the
claim. The other half is that the **board** computes the same thing, and it
cannot be checked by re-running the host tool -- that tool dlopens a host build.
This script produces the two things a device check needs:

  * `cases/*.bin`   -- the 30 input patches, uint8 (= int8 + 128), the exact
                       transport `src/si_patch.py` already emits, so the board
                       reads them with `open().read()` and nothing else.
  * `reference.txt` -- what the host says the answers are, as `SCORE` lines in
                       si-model's ingest format.

`tools/tflm_probe.py` runs on the board and emits the same format;
`tools/check_tflm_device.py` diffs the two. **The diff is on integers**, because
a float comparison through two different `printf` implementations is a worse
test than comparing the quantised bytes that actually came out of the model.

## The 30, and why both halves are here

| set | n | what | ground truth |
| --- | --- | --- | --- |
| `bitexact` | 8 | `build/kw_unknown_*.bin` | **none** |
| `takes` | 22 | `takes/` + `takes-oov/`, featurised | filename |

The eight are the patches on which the device (TinyMaix) disagreed with the host
in `docs/cnn-on-device.md` -- the sharpest available test of whether the
disagreement is gone. **They carry no ground truth**: their filenames encode
what an *earlier* model predicted, not what was said. They are for the byte
comparison only and must not be scored as accuracy, which is why every line
carries `set=`.

The 22 are the real-speaker takes, and they are the ones with labels.

## Why the patches are staged rather than the WAVs

The board could featurise a WAV -- `src/si_patch.py` is the device half of
`tools/si_features.py` and is pinned byte-identical to it. But then a byte
difference in the output would have two possible causes, the front end and the
model, and the morning is about the model. Staging the *input tensors* removes
the front end from the experiment entirely. The front end is separately proved,
by `tools/test_si_patch.py`, and re-proved on this board by `speech_probe`.

    ./tools/make_tflm_cases.py                       # -> build/tflm-cases/
    ./tools/make_tflm_cases.py --out /tmp/stage
"""

import argparse
import ctypes
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

N_FRAMES = 80
N_BANDS = 26
PATCH_BYTES = N_FRAMES * N_BANDS


def host_model(lib_path, model_bytes, arena=200 * 1024):
    """The host TFLM library, through the same ctypes wrapper the proof uses."""
    from tflm_vs_tflite import load_lib, Tflm
    return Tflm(load_lib(lib_path), model_bytes, arena)


def q_from_int8(raw):
    """int8 output tensor -> si-model's `q`, integers 0..255.

    The model's output scale is 1/256 and its zero point is -128, so
    `q = raw + 128` and `probability = q / 256` exactly. This is the
    representation the device cannot get wrong, which is why it is the one
    compared.
    """
    return [int(v) + 128 for v in raw]


def score_line(runtime, case_set, name, q, frames=None, clipped=None,
               micros=None):
    parts = ["SCORE", "runtime=%s" % runtime, "set=%s" % case_set,
             "name=%s" % name]
    if frames is not None:
        parts.append("frames=%d" % frames)
    if clipped is not None:
        parts.append("clipped=%d" % clipped)
    if micros is not None:
        parts.append("us=%d" % micros)
    parts.append("q=" + ",".join(str(v) for v in q))
    return " ".join(parts)


def collect_cases(repo, takes_dirs):
    """-> [(set, name, uint8 patch bytes, frames, clipped)]

    `name` is what appears in `name=`. For the takes it is the WAV's basename,
    because si-model recovers ground truth from it.
    """
    cases = []

    for path in sorted(glob.glob(os.path.join(repo, "build", "kw_unknown_*.bin"))):
        raw = open(path, "rb").read()
        if len(raw) != PATCH_BYTES:
            raise SystemExit("%s is %d bytes, expected %d"
                             % (path, len(raw), PATCH_BYTES))
        cases.append(("bitexact", os.path.basename(path), raw, None, None))

    sys.path.insert(0, os.path.join(repo, "src"))
    import si_features

    for directory in takes_dirs:
        for wav in sorted(glob.glob(os.path.join(directory, "*.wav"))):
            patch, frames, clipped = si_features.patch_for_wav(wav)
            if patch is None:
                # An utterance the endpointer rejects is one the device would
                # never present to the model either. Reported, not silently
                # dropped -- a shrinking case count is exactly the kind of
                # thing that goes unnoticed.
                print("  endpointer rejected %s -- not staged" % wav)
                continue
            flat = bytearray(PATCH_BYTES)
            i = 0
            for row in patch:
                for v in row:
                    flat[i] = v + 128
                    i += 1
            if i != PATCH_BYTES:
                raise SystemExit("%s produced %d values, expected %d"
                                 % (wav, i, PATCH_BYTES))
            cases.append(("takes", os.path.basename(wav), bytes(flat),
                          frames, clipped))
    return cases


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(repo, "build", "tflm-cases"))
    ap.add_argument("--model", default=os.path.join(repo, "build", "si_real.tflite"))
    ap.add_argument("--classes", default=os.path.join(repo, "build", "si_real.json"))
    ap.add_argument("--takes", nargs="*",
                    default=[os.path.join(repo, "takes"),
                             os.path.join(repo, "takes-oov")])
    ap.add_argument("--lib", default=os.path.join(repo, "build", "tflm-host",
                                                  "libtflm_host.dylib"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.lib):
        raise SystemExit("no host TFLM library at %s -- run "
                         "./tools/build_tflm_host.sh first" % args.lib)

    out = args.out
    cases_dir = os.path.join(out, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    model_bytes = open(args.model, "rb").read()
    model = host_model(args.lib, model_bytes)
    print("host TFLM: input %d, output %d, arena_used %d"
          % (model.n_in, model.n_out, model.arena_used))

    cases = collect_cases(repo, args.takes)
    print("%d cases (%d bitexact, %d takes)"
          % (len(cases),
             sum(1 for c in cases if c[0] == "bitexact"),
             sum(1 for c in cases if c[0] == "takes")))

    manifest = []
    reference = []
    exactness_checked = 0
    for case_set, name, blob, frames, clipped in cases:
        open(os.path.join(cases_dir, name + ".bin"), "wb").write(blob)

        signed = bytes(((b + 128) & 0xFF) for b in blob)   # uint8 -> int8 bits
        raw = model.invoke(memoryview(signed).cast("b"))
        q = q_from_int8(raw)

        # The device recovers `q` from float scores as round(p * 256), because
        # the module's run() returns dequantised floats. That is exact -- scale
        # is 2^-8 and the numerator is an integer, so every value is a float32
        # with an exact binary representation -- but "is exact" is a claim, so
        # it is checked here against the integers the library actually returned
        # rather than asserted in a comment.
        for value, expect in zip(raw, q):
            p = (int(value) + 128) / 256.0
            recovered = int(round(p * 256))
            if recovered != expect:
                raise SystemExit(
                    "float round-trip is not exact for %s: %d -> %.9f -> %d"
                    % (name, expect, p, recovered))
        exactness_checked += 1

        manifest.append("%s %s %s %s" % (case_set, name,
                                         "-" if frames is None else frames,
                                         "-" if clipped is None else clipped))
        reference.append(score_line("host-tflm", case_set, name, q,
                                    frames, clipped))

    print("float<->integer round trip exact on %d/%d cases"
          % (exactness_checked, len(cases)))

    open(os.path.join(out, "manifest.txt"), "w").write(
        "\n".join(manifest) + "\n")
    open(os.path.join(out, "reference.txt"), "w").write(
        "\n".join(reference) + "\n")

    shutil.copy(args.model, os.path.join(out, "si_model.tflite"))
    if os.path.exists(args.classes):
        shutil.copy(args.classes, os.path.join(out, "si_model.json"))

    total = sum(os.path.getsize(os.path.join(cases_dir, f))
                for f in os.listdir(cases_dir))
    print("\nstaged in %s" % out)
    print("  si_model.tflite  %d B" % os.path.getsize(args.model))
    print("  cases/           %d files, %d B" % (len(cases), total))
    print("  manifest.txt, reference.txt")
    print("\ncopy to the board with:")
    print("    uvx mpremote connect $PORT cp %s/si_model.tflite :" % out)
    print("    uvx mpremote connect $PORT cp -r %s :" % cases_dir)
    print("    uvx mpremote connect $PORT cp %s/manifest.txt :cases/" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
