#!/usr/bin/env python3
"""Is host TFLM bit-identical to host TFLite on this model?

The premise of the whole TFLM proposal is that TFLM's int8 reference kernels
compute what TFLite's int8 reference kernels compute -- same fixed-point
multipliers, same rounding, same saturation -- so that a model evaluated on the
host is the model that runs on the board. `docs/cnn-on-device.md` records that
TinyMaix does *not* have that property (3 of 8 patches disagree on top-1). This
script is the control that says whether TFLM does.

The comparison is on the **raw int8 output tensor**, not on dequantised floats.
Comparing floats would hide a one-count difference behind print precision, and
one count of 1/256 is exactly the size of the margins this project's gates run
on.

What "the same" means here is the **raw int8 output tensor**, not the
dequantised floats. Comparing floats would hide a one-count difference behind
print precision, and one count of 1/256 is exactly the size of the margins
`src/si_spot.py`'s gates run on.

**Which TFLite matters.** `tf.lite.Interpreter` is three different runtimes
depending on how it is constructed, and they do not agree with each other:

    --resolver ref          TFLite's own reference kernels     <- the comparison
    --resolver nodelegate   TFLite's optimised CPU kernels
    --resolver default      the same, plus the XNNPACK delegate

TFLM matches the first exactly and neither of the other two. That is not a
defect in any of them -- optimised int8 kernels are allowed to reassociate --
but it does mean **`--resolver ref` is the only setting under which "host equals
device" is true**, and it is the setting every evaluation of a model destined
for this board has to use.

    ./tools/fetch_tflm.sh
    ./tools/build_tflm_host.sh
    .venv/bin/python tools/tflm_vs_tflite.py --model build/si_real.tflite \\
        --classes build/si_real.json --bins 'build/kw_unknown_*.bin' \\
        --takes takes takes-oov

Exit status is 0 when every case is bit-identical, 2 when any is not.
"""

import argparse
import ctypes
import glob
import json
import os
import sys

import numpy as np


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.tflm_new_model.restype = ctypes.c_void_p
    lib.tflm_new_model.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_int)]
    lib.tflm_new_model_recording.restype = ctypes.c_void_p
    lib.tflm_new_model_recording.argtypes = lib.tflm_new_model.argtypes
    lib.tflm_print_allocations.argtypes = [ctypes.c_void_p]
    lib.tflm_input_len.restype = ctypes.c_int
    lib.tflm_input_len.argtypes = [ctypes.c_void_p]
    lib.tflm_output_len.restype = ctypes.c_int
    lib.tflm_output_len.argtypes = [ctypes.c_void_p]
    lib.tflm_arena_used.restype = ctypes.c_size_t
    lib.tflm_arena_used.argtypes = [ctypes.c_void_p]
    lib.tflm_input_scale.restype = ctypes.c_float
    lib.tflm_input_scale.argtypes = [ctypes.c_void_p]
    lib.tflm_input_zero_point.restype = ctypes.c_int32
    lib.tflm_input_zero_point.argtypes = [ctypes.c_void_p]
    lib.tflm_output_scale.restype = ctypes.c_float
    lib.tflm_output_scale.argtypes = [ctypes.c_void_p]
    lib.tflm_output_zero_point.restype = ctypes.c_int32
    lib.tflm_output_zero_point.argtypes = [ctypes.c_void_p]
    lib.tflm_invoke.restype = ctypes.c_int
    lib.tflm_invoke.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int]
    return lib


class Tflm:
    def __init__(self, lib, model_bytes, arena_bytes, recording=False):
        self.lib = lib
        self.model = ctypes.create_string_buffer(model_bytes, len(model_bytes))
        self.arena = ctypes.create_string_buffer(arena_bytes)
        err = ctypes.c_int(0)
        new = (lib.tflm_new_model_recording if recording else lib.tflm_new_model)
        self.h = new(self.model, len(model_bytes), self.arena, arena_bytes,
                     ctypes.byref(err))
        if not self.h:
            raise RuntimeError("tflm_new_model failed, err=%d" % err.value)
        self.n_in = lib.tflm_input_len(self.h)
        self.n_out = lib.tflm_output_len(self.h)
        self.out_scale = lib.tflm_output_scale(self.h)
        self.out_zp = lib.tflm_output_zero_point(self.h)
        self.in_scale = lib.tflm_input_scale(self.h)
        self.in_zp = lib.tflm_input_zero_point(self.h)

    @property
    def arena_used(self):
        return self.lib.tflm_arena_used(self.h)

    def print_allocations(self):
        self.lib.tflm_print_allocations(self.h)

    def invoke(self, patch_i8):
        buf = np.ascontiguousarray(patch_i8, dtype=np.int8)
        assert buf.size == self.n_in, (buf.size, self.n_in)
        out = ctypes.create_string_buffer(self.n_out)
        rc = self.lib.tflm_invoke(self.h, buf.ctypes.data_as(ctypes.c_char_p),
                                  buf.size, out, None, self.n_out)
        if rc != 0:
            raise RuntimeError("tflm_invoke rc=%d" % rc)
        return np.frombuffer(out.raw, dtype=np.int8, count=self.n_out).copy()


def tflite_runner(model_path, resolver="ref"):
    """`resolver` picks which TFLite kernels to compare against.

    This matters more than it looks. The default interpreter installs the
    XNNPACK delegate, which is an *optimised* int8 path -- comparing TFLM's
    reference kernels against that measures the delegate as much as anything
    else. "ref" selects TFLite's own reference kernels, which is the like-for-
    like comparison, and "nodelegate" is the in-between control.
    """
    import tensorflow as tf
    kinds = {
        "default": None,
        "nodelegate": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        "ref": tf.lite.experimental.OpResolverType.BUILTIN_REF,
    }
    kind = kinds[resolver]
    if kind is None:
        interp = tf.lite.Interpreter(model_path=model_path)
    else:
        interp = tf.lite.Interpreter(model_path=model_path,
                                     experimental_op_resolver_type=kind)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]

    def run(patch_i8):
        x = np.ascontiguousarray(patch_i8, dtype=np.int8).reshape(inp["shape"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(outp["index"]).reshape(-1).copy()

    return run, inp, outp


def patches_from_bins(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8)
        out.append((os.path.basename(p), raw.astype(np.int16) - 128))
    return out


def patches_from_takes(dirs, repo_root):
    sys.path.insert(0, os.path.join(repo_root, "src"))
    sys.path.insert(0, os.path.join(repo_root, "tools"))
    import si_features
    out = []
    for d in dirs:
        for wav in sorted(glob.glob(os.path.join(d, "*.wav"))):
            patch, n_frames, clipped = si_features.patch_for_wav(wav)
            if patch is None:
                print("  (endpointer rejected %s)" % wav)
                continue
            out.append((os.path.basename(wav),
                        np.array(patch, dtype=np.int16).reshape(-1)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    default_lib = os.path.join(repo, "build", "tflm-host", "libtflm_host.dylib")
    if not os.path.exists(default_lib):
        default_lib = default_lib[:-6] + ".so"
    ap.add_argument("--lib", default=default_lib)
    ap.add_argument("--model", required=True)
    ap.add_argument("--classes", default=None,
                    help="the model's .json, for class names")
    ap.add_argument("--bins", default=None, help="glob of uint8 patch files")
    ap.add_argument("--takes", nargs="*", default=[],
                    help="directories of WAVs to featurise")
    ap.add_argument("--repo", default=repo)
    ap.add_argument("--arena", type=int, default=200 * 1024)
    ap.add_argument("--resolver", default="ref",
                    choices=["default", "nodelegate", "ref"])
    args = ap.parse_args(argv)

    model_bytes = open(args.model, "rb").read()
    lib = load_lib(args.lib)

    classes = None
    if args.classes and os.path.exists(args.classes):
        classes = json.load(open(args.classes)).get("classes")

    print("== arena sizing ==")
    rec = Tflm(lib, model_bytes, args.arena, recording=True)
    print("recording interpreter, arena_used = %d bytes" % rec.arena_used)
    rec.print_allocations()
    sys.stdout.flush()

    m = Tflm(lib, model_bytes, args.arena)
    print("plain interpreter,     arena_used = %d bytes" % m.arena_used)
    print("model file            = %d bytes" % len(model_bytes))
    print("input  len=%d scale=%.9g zp=%d" % (m.n_in, m.in_scale, m.in_zp))
    print("output len=%d scale=%.9g zp=%d" % (m.n_out, m.out_scale, m.out_zp))

    # Tightest arena that still allocates: bisect, because the shipping build
    # allocates this once and any slack is heap the capture buffer wants.
    lo, hi = 1024, args.arena
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            Tflm(lib, model_bytes, mid)
            hi = mid
        except RuntimeError:
            lo = mid + 1
    print("minimum working arena = %d bytes (bisected)" % lo)

    run_tflite, inp, outp = tflite_runner(args.model, args.resolver)
    print("tflite kernels        = %s" % args.resolver)
    print("tflite input %s %s, output %s %s" %
          (inp["shape"], inp["dtype"].__name__, outp["shape"],
           outp["dtype"].__name__))

    cases = []
    if args.bins:
        cases += patches_from_bins(args.bins)
    if args.takes:
        cases += patches_from_takes(args.takes, args.repo)
    if not cases:
        print("no cases")
        return 1

    print("\n== %d cases: TFLM int8 output vs TFLite int8 output ==" % len(cases))
    n_exact = 0
    n_top1 = 0
    worst = 0
    disagreements = []
    for name, patch in cases:
        a = m.invoke(patch)
        b = run_tflite(patch)
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        exact = bool((d == 0).all())
        n_exact += exact
        ta, tb = int(np.argmax(a)), int(np.argmax(b))
        n_top1 += (ta == tb)
        worst = max(worst, int(d.max()))
        if not exact or ta != tb:
            disagreements.append((name, ta, tb, int(d.max())))
        label = (lambda i: classes[i] if classes and i < len(classes) else str(i))
        print("  %-28s tflm=%-10s tflite=%-10s  max|d|=%d  %s"
              % (name, label(ta), label(tb), int(d.max()),
                 "exact" if exact else "DIFFERS"))

    print("\n== verdict ==")
    print("bit-identical int8 output tensors : %d / %d" % (n_exact, len(cases)))
    print("top-1 agreement                   : %d / %d" % (n_top1, len(cases)))
    print("worst single-count difference     : %d" % worst)
    if n_exact == len(cases):
        print("RESULT: host TFLM == host TFLite, bit for bit, on every case.")
        return 0
    print("RESULT: host TFLM DIFFERS from host TFLite. The premise fails.")
    for row in disagreements:
        print("   ", row)
    return 2


if __name__ == "__main__":
    sys.exit(main())
