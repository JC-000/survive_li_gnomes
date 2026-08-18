#!/usr/bin/env python3
"""Train a speaker-independent keyword CNN on synthetic voices, and quantise it.

The hypothesis under test: macOS `say` has ~184 voices, so a classifier trained
on all of them plus augmentation might generalise to a real human without
anybody recording a keyword campaign. `docs/speech-design.md` costed a trained
CNN against DTW and chose DTW on **recordings per word**, not on accuracy --
50-100 per keyword against DTW's 3. Synthetic voices are an attempt to make
that row free. If it works, the spotter stops being speaker-dependent; if it
does not, DTW stays and this file is the evidence for why.

    uv run --python .venv/bin/python tools/si_train.py corpus/manifest.jsonl

Requires TensorFlow, which is a host-only training dependency and is not part
of the device path or of anything else in `tools/`. Install it into `.venv`
(gitignored) rather than globally:

    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python "tensorflow>=2.16,<2.21"

## What the model sees

`tools/si_features.py`: 80 frames x 26 Q8 log2 mel bands, per-band mean
subtracted, shifted to int8. That is the existing front end with the DCT
removed, not a new one -- see that file's docstring for why the filterbank and
not the cepstrum.

## The two architectures, and why the choice is not mine alone

`dscnn` follows Zhang et al., "Hello Edge" (arXiv:1711.07128), whose DS-CNN-S
is 38.6 KB / 5.4 MOps / 94.4% on 12 classes on a Cortex-M7. Depthwise-separable
convolution is what makes that size buy that accuracy, and it needs a runtime
that implements depthwise convolution -- TinyMaix does.

`plain` exists because that runtime might not load. Without it the fallback is
a hand-rolled int8 convolution in `@micropython.viper`, and viper is fastest on
one long contiguous multiply-accumulate loop; a depthwise layer is a short loop
run many times, which is the shape it is worst at. So `plain` drops depthwise
entirely and spends the operations on fewer, denser layers.

Both are sized to the same ceiling -- comfortably under 60 KB of weights and
100 KB of activations -- because the binding constraint is neither. It is
multiply-accumulates: DTW's matcher costs ~616 ms for a 44-frame query against
66 templates, so anything much past ~2 MMAC stops being the cheaper option and
the whole "fixed inference cost is a feature" argument goes with it.

## The converter decides the shape, not taste

Both architectures are built from what `TinyMaix/tools/tflite2tmdl.py` can
actually translate, which is a shorter list than Keras offers: CONV_2D,
DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX, RESHAPE, ADD. **There is no
MAX_POOL_2D and no REDUCE_MAX.** So every downsample here is a strided
convolution rather than a pooling layer, and the head is a global *average*
pool. Neither is a preference; both are the only thing that converts. Checked
by conversion, not by reading:

    uv venv --python 3.10 .venv-tm
    uv pip install --python .venv-tm/bin/python "tensorflow-macos==2.13.0" \
        "numpy<2" pillow
    git clone --depth 1 https://github.com/sipeed/TinyMaix.git
    touch TinyMaix/tools/__init__.py     # its converter uses a relative import
    cd TinyMaix && python -m tools.tflite2tmdl in.tflite out.tmdl int8 1 \
        80,26,1 22

TensorFlow 2.13 on Python 3.10 is a second venv on purpose: emlearn's own
TinyMaix example records that the converter needs TensorFlow < 2.14, which does
not install on Python 3.12 or later. Everything in this file runs unchanged
under both that venv and a current one, and the numbers below came out of 2.13.

**Measured, by converting each one (sizes are set by the architecture, so a
one-epoch model measures them exactly):**

| model | .tmdl flash | TinyMaix RAM buffer | single-layer mode |
| --- | --- | --- | --- |
| `dscnn` width 1.0 | 15.4 KB | 20.6 KB | 3.6 KB |
| `dscnn` width 2.0 | 45.2 KB | 41.2 KB | 13.1 KB |
| `plain` width 1.0 | 41.5 KB | 12.5 KB | 20.8 KB |
| any `--pool max` | **does not convert** -- REDUCE_MAX | | |

`dscnn` width 1.0 is 36 KB of flash and RAM together, against a ~490 KB heap
that also holds the ELIZA rules, the capture buffer and the framebuffer. Size
is not what will decide this experiment.

## Training choices that are about precision, not accuracy

- **The unknown class is trained, not thresholded into existence.** The
  corpus's out-of-vocabulary utterances are a real class with real gradient.
  A softmax over keywords alone has no way to represent "none of these" and
  produces its confident nonsense at exactly the moment it matters.
- **SpecAugment** (Park et al., arXiv:1904.08779) masks bands and frames during
  training, and a +/-6 frame shift models the endpointer landing differently on
  a real voice. Both are done here rather than in the corpus because they are
  feature-domain and re-running the front end for them would cost hours for
  nothing.
- **Gain and tilt augmentation would be wasted effort, and that is measured.**
  The per-band mean subtraction in `si_features.normalise` removes any fixed
  linear filter exactly -- a filter is a constant per band in the log domain,
  and gain is the case where the constant is the same in every band. Measured
  on one utterance, patch values spanning -128..97: a -6 dB gain moves the
  features by a mean of 0.68 LSB and a +/-0.15 first-order tilt by 0.35, which
  is a fifth of a decibel and is quantisation, not signal. What survives the
  normalisation and therefore matters is **additive noise**, **reverberation**,
  **speaking rate** and, far above all of them, **more voices**.
- **Label smoothing is off.** It improves accuracy and flattens the softmax,
  and a flattened softmax is precisely what a confidence threshold cannot work
  with. This is the one place the usual advice points the wrong way here.
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import si_features                             # noqa: E402


N_CLASSES = len(si_features.CLASSES)


# --- architectures ---------------------------------------------------------

def build(arch="dscnn", width=1.0, n_classes=N_CLASSES, pool="avg"):
    """Keras model over a (N_FRAMES, N_BANDS, 1) float input in int8 range.

    The input is fed as float in [-128, 127] rather than scaled to [-1, 1]:
    the full-integer TFLite converter puts its own affine quantisation on the
    input anyway, and keeping the training-time range equal to the device-time
    range removes one place where host and device can silently disagree about
    what a sample means.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    def c(n):
        return max(8, int(round(n * width / 4)) * 4)

    inp = keras.Input(shape=(si_features.N_FRAMES, si_features.N_BANDS, 1),
                      name="logmel")
    x = inp

    if arch == "dscnn":
        # Hello Edge's shape: one full convolution to get the time and
        # frequency axes down cheaply, then depthwise-separable blocks, then a
        # global average pool instead of a flatten.
        x = _conv(layers, x, c(32), (5, 5), (2, 2))
        for filters, stride in ((c(32), 2), (c(48), 1),
                                (c(48), 2), (c(64), 1)):
            x = _dsconv(layers, x, filters, stride)
    elif arch == "plain":
        x = _conv(layers, x, c(16), (5, 5), (2, 2))
        x = _conv(layers, x, c(32), (3, 3), (2, 2))
        x = _conv(layers, x, c(48), (3, 3), (2, 2))
        x = _conv(layers, x, c(48), (3, 3), (1, 1))
    else:
        raise ValueError("unknown architecture %r" % arch)

    # Global pooling rather than a flatten: it keeps the classifier head at
    # ~1.4 KB instead of the ~45 KB a flatten would need, and it makes the
    # model tolerant of the word sitting a few frames off centre.
    #
    # Average against max is a real question here and not the usual one.
    # Utterances are padded to N_FRAMES and the median one is 55 frames, so
    # about a third of a typical input is padding. Average pooling divides the
    # evidence by a denominator that is the same for every utterance however
    # much of it was real, which dilutes a short word; max pooling does not
    # notice the padding at all.
    #
    # **Average wins by not being a choice.** TFLite lowers a global average
    # pool to MEAN, which TinyMaix's converter maps to TML_GAP; it lowers a
    # global max pool to REDUCE_MAX, which the converter has no entry for and
    # dies on. Verified by running `tools/tflite2tmdl.py` over both. `--pool
    # max` therefore trains and evaluates but cannot be deployed, and it is
    # kept only so the accuracy cost of the constraint can be measured.
    if pool == "max":
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="keyword")(x)
    return keras.Model(inp, out, name="si_%s_%s" % (arch, pool))


# Batch-norm moving-average momentum. Keras defaults to 0.99, which assumes
# thousands of steps; this corpus gives a few tens of steps per epoch, so at
# 0.99 the inference-time statistics are still near their initial values after
# the whole run. The symptom is brutal and points nowhere near batch norm:
# training accuracy climbs normally while validation output is a *constant*
# near-uniform softmax, because validation runs in inference mode and the
# statistics it uses were never learned. Measured on the dry-run corpus, the
# model reported 0.70 training accuracy and a flat 0.055 top probability --
# 1/18 -- on every held-out utterance.
BN_MOMENTUM = 0.9


def _conv(layers, x, filters, kernel, strides):
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    return layers.ReLU()(x)


def _dsconv(layers, x, filters, stride=1):
    x = layers.DepthwiseConv2D((3, 3), strides=(stride, stride),
                               padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    return layers.ReLU()(x)


def cost(model):
    """(multiply-accumulates, weight bytes int8, peak activation bytes).

    Counted here rather than taken from a converter report, because the number
    that decides whether this is worth doing has to be checkable by reading
    it. MACs count convolution and dense only; batch-norm folds into the
    preceding convolution at quantisation time and costs nothing at inference.
    """
    macs = 0
    weights = 0
    peak_act = 0
    for layer in model.layers:
        shape = layer.output.shape
        size = 1
        for dim in shape[1:]:
            size *= int(dim)
        peak_act = max(peak_act, size)
        name = type(layer).__name__
        if name == "Conv2D":
            oh, ow, oc = [int(d) for d in shape[1:]]
            kh, kw = layer.kernel_size
            ic = int(layer.input.shape[-1])
            macs += oh * ow * oc * kh * kw * ic
            weights += kh * kw * ic * oc
        elif name == "DepthwiseConv2D":
            oh, ow, oc = [int(d) for d in shape[1:]]
            kh, kw = layer.kernel_size
            macs += oh * ow * oc * kh * kw
            weights += kh * kw * oc
        elif name == "Dense":
            n_in = int(layer.input.shape[-1])
            n_out = int(shape[-1])
            macs += n_in * n_out
            weights += n_in * n_out + n_out * 4
    return macs, weights, peak_act


# --- augmentation ----------------------------------------------------------

def spec_augment(x, rng, n_freq=2, freq_width=4, n_time=2, time_width=8,
                 shift=6):
    """SpecAugment plus a time shift, in place on a copy. int8 domain.

    The time shift is here and not in the corpus because the device's
    endpointer decides where a word starts, and it will not decide identically
    on a real voice. Training the model to tolerate a few frames of slop is
    cheaper than making the endpointer exact, and it is the same argument the
    DTW path makes for its Sakoe-Chiba band.

    Masks are filled with 0, which after the per-band mean subtraction is the
    band's own average -- so a mask reads as "no information here" rather than
    as a hole of silence, which would be a different and misleading signal.
    """
    import numpy as np
    out = x.copy()
    n = out.shape[0]
    for i in range(n):
        if shift:
            k = int(rng.integers(-shift, shift + 1))
            if k:
                out[i] = np.roll(out[i], k, axis=0)
                if k > 0:
                    out[i, :k] = 0
                else:
                    out[i, k:] = 0
        for _ in range(n_freq):
            w = int(rng.integers(0, freq_width + 1))
            if w:
                f0 = int(rng.integers(0, out.shape[2] - w + 1))
                out[i, :, f0:f0 + w] = 0
        for _ in range(n_time):
            w = int(rng.integers(0, time_width + 1))
            if w:
                t0 = int(rng.integers(0, out.shape[1] - w + 1))
                out[i, t0:t0 + w, :] = 0
    return out


def generator(x, y, batch, rng, weights=None, augment=True):
    """Endless shuffled batches, augmented, with per-sample weights.

    The weights are yielded rather than passed to `fit(class_weight=...)`
    because Keras 3 rejects `class_weight` alongside a Python generator. Same
    arithmetic, different door.
    """
    import numpy as np
    n = len(x)
    lut = None
    if weights:
        lut = np.array([weights[i] for i in range(len(weights))],
                       dtype="float32")
    while True:
        order = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            idx = order[start:start + batch]
            xb = x[idx]
            if augment:
                xb = spec_augment(xb, rng)
            xb = xb.astype("float32")[..., None]
            if lut is None:
                yield xb, y[idx]
            else:
                yield xb, y[idx], lut[y[idx]]


# --- training --------------------------------------------------------------

def class_weights(y, n_classes=N_CLASSES, unknown_weight=1.0):
    """Inverse-frequency weights, square-rooted, with `unknown` adjustable.

    The unknown class is deliberately many times larger than any keyword --
    that is what makes precision measurable -- and left unweighted the model
    would find "say unknown" a good local minimum. Full inverse frequency
    overcorrects and makes it fire too readily, which is the failure this
    project cares about most. The square root is the usual compromise and it is
    a judgement, not a measurement.

    `unknown_weight` multiplies the unknown class afterwards and is the one
    knob here that maps directly onto the design goal. Above 1 the model is
    pushed towards silence, which costs recall and buys precision; below 1 the
    reverse. Per `docs/speech-design.md` a miss is free and a false fire is
    not, so if a sweep shows the curve is flat, the higher setting is the right
    default -- but sweep it rather than assuming, because pushing too hard
    produces a model that says `unknown` to everything and scores a precision
    of 1.000 on zero fires, which is not the same as being right.
    """
    import numpy as np
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n_classes * counts)) ** 0.5
    w[si_features.CLASSES.index(si_features.UNKNOWN)] *= unknown_weight
    return dict(enumerate(w))


def train(x_tr, y_tr, x_va, y_va, arch="dscnn", width=1.0, epochs=60,
          batch=64, seed=1, quiet=False, pool="avg", unknown_weight=1.0):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(arch, width, pool=pool)
    macs, wbytes, act = cost(model)
    if not quiet:
        print("architecture %s width %.2f: %d params, %.2f MMAC, "
              "%.1f KB weights (int8, est), %.1f KB peak activation"
              % (arch, width, model.count_params(), macs / 1e6,
                 wbytes / 1024.0, act / 1024.0))

    steps = max(1, len(x_tr) // batch)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        # No label smoothing: see the module docstring. A flattened softmax
        # cannot be thresholded, and thresholding is the whole design.
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=6, min_lr=1e-5, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True, verbose=0),
    ]
    # The validation set is weighted with the same class weights as training,
    # which makes `val_accuracy` a *balanced* accuracy. Unweighted it is not a
    # useful early-stopping signal here: `unknown` is several times larger than
    # every keyword put together, so a model that quietly gave up on keywords
    # would still climb, and `restore_best_weights` would then restore it.
    weights = class_weights(y_tr, unknown_weight=unknown_weight)
    val_w = np.array([weights[int(c)] for c in y_va], dtype="float32")
    model.fit(generator(x_tr, y_tr, batch, rng, weights),
              steps_per_epoch=steps, epochs=epochs,
              validation_data=(x_va.astype("float32")[..., None], y_va, val_w),
              callbacks=callbacks, verbose=0 if quiet else 2)
    return model


# --- quantisation ----------------------------------------------------------

def to_int8_tflite(model, x_rep, path):
    """Full-integer post-training quantisation. int8 in, int8 out.

    Post-training rather than quantisation-aware, and that is a choice worth
    defending: QAT costs a training run and a converter path, and buys most of
    its ground back on models whose activations have long tails. Log-mel input
    is already bounded and mean-subtracted, and every activation here follows a
    batch-norm and a ReLU, so the tails are short. The training run prints both
    accuracies at the end; if the int8 model measurably trails the float one,
    QAT is the next thing to try, and only then.

    The representative set is drawn from **training** voices only. Drawing it
    from validation would leak the held-out voices into the quantisation
    ranges, which is a small leak but exactly the kind this experiment must not
    have.
    """
    import tensorflow as tf

    def rep():
        for i in range(0, min(len(x_rep), 500)):
            yield [x_rep[i:i + 1].astype("float32")[..., None]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    # The dense layer must be quantised **per tensor**, not per channel.
    #
    # TinyMaix's `tml_fc` reads `ws[0]` -- the first output channel's scale --
    # and applies it to every logit, while the converter dutifully writes all
    # 22 scales into the file. Per-channel therefore converts cleanly and then
    # returns wrong numbers for 21 of 22 classes, with nothing raised. It looks
    # exactly like a model that trained badly. (Convolutions are fine: those
    # really do use `ws[c]`.)
    #
    # The flag is private and its name has moved between TensorFlow versions,
    # so it is set only if it exists -- and its absence is not treated as
    # success. `tools/tmdl_info.py` is what actually decides: it reads the
    # scale array out of the `.tmdl` and exits non-zero if there is more than
    # one. Run it before any model goes near the board.
    flag = "_experimental_disable_per_channel_quantization_for_dense_layers"
    if hasattr(conv, flag):
        setattr(conv, flag, True)
    else:
        print("note: %s not available on TensorFlow %s -- the dense layer may "
              "quantise per channel. tools/tmdl_info.py is the check that "
              "matters." % (flag, tf.__version__))

    blob = conv.convert()

    # The input quantisation must be exactly scale 1.0, zero point 0, and this
    # is checked rather than hoped for.
    #
    # The device computes int8 log-mel patches and hands them to TinyMaix as
    # raw bytes. The model interprets each byte as `scale * (q - zero)`, so
    # unless scale is 1 and zero is 0, "the value the device supplies" and
    # "the value the model was trained on" are different numbers -- and
    # nothing anywhere would say so. The model would load, run, and be quietly
    # a little wrong, which is the failure mode `docs/speech.md` is arranged
    # against from end to end.
    #
    # It comes out right because the patches span the full int8 range: the
    # clipping measured in `si_features.INPUT_SHIFT` guarantees -128 appears,
    # and +127 turns up across a corpus, so TFLite's min/max gives scale
    # (127 - -128)/255 = 1.0 and zero 0. That is a property of the *data*, so
    # a future corpus that never clips would silently break it. Hence the
    # assertion rather than a comment.
    interp = _interpreter(blob)
    idet = interp.get_input_details()[0]
    scale, zero = idet["quantization"]
    if abs(scale - 1.0) > 1e-6 or zero != 0:
        raise ValueError(
            "input quantisation is scale %.6f, zero point %d, not 1.0 / 0.\n"
            "The device feeds raw int8 log-mel and cannot rescale it, so the "
            "model would be reading different numbers from the ones it was "
            "trained on. Most likely the representative set no longer reaches "
            "both ends of the int8 range." % (scale, zero))
    with open(path, "wb") as fh:
        fh.write(blob)
    return blob


def _interpreter(blob):
    """One place that knows how to make a TFLite interpreter.

    TensorFlow 2.20 deprecates `tf.lite.Interpreter` in favour of
    `ai_edge_litert`, while the 2.13 environment the TinyMaix converter needs
    has only the old one. Both venvs have to work, so the import is tried in
    that order.
    """
    import tensorflow as tf
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_content=blob)


def _out_quant(blob):
    interp = _interpreter(blob)
    interp.allocate_tensors()
    return interp.get_output_details()[0]["quantization"]


def tflite_predict(blob, x, batch_note=True):
    """Run the quantised model over int8 patches -> float probabilities.

    Dequantised back to probabilities so the threshold sweep in
    `tools/si_eval.py` reads the same units as the float model. The
    dequantisation is exact arithmetic on the recorded scale and zero point;
    it does not hide any quantisation error, it only renames the axis.
    """
    import numpy as np

    interp = _interpreter(blob)
    interp.allocate_tensors()
    idet = interp.get_input_details()[0]
    odet = interp.get_output_details()[0]
    iscale, izero = idet["quantization"]
    oscale, ozero = odet["quantization"]
    out = np.zeros((len(x), N_CLASSES), dtype="float32")
    for i in range(len(x)):
        q = np.round(x[i].astype("float32") / iscale + izero)
        q = np.clip(q, -128, 127).astype("int8")[None, ..., None]
        interp.set_tensor(idet["index"], q)
        interp.invoke()
        raw = interp.get_tensor(odet["index"])[0].astype("float32")
        out[i] = (raw - ozero) * oscale
    return out


# --- CLI -------------------------------------------------------------------

def load_split(manifest, cache, jobs, quiet=False):
    """-> (x_train, y_train, x_val, y_val, val_voices, rows)."""
    rows = si_features.read_manifest(manifest)
    summary = si_features.check_split(rows)
    # Raises rather than warns, and it is checked here rather than left to
    # `--stats`, because a corpus whose voices are not distinct trains and
    # validates perfectly happily -- see `check_distinct_voices`.
    si_features.check_distinct_voices(rows)
    if not quiet:
        for split in sorted(summary):
            n, nv = summary[split]
            print("  %-8s %6d utterances  %4d voices" % (split, n, nv))
    si_features.extract_all(rows, cache, jobs, progress=not quiet)
    # Named splits, not "train and everything else". The corpus carries a third
    # split, `test`, and it has to stay out of both training and the epoch-wise
    # validation that early stopping reads -- otherwise it is a second
    # validation set by the time anyone looks at it, and there is no held-out
    # set left to check the tuning against.
    tr = [r for r in rows if r["split"] == "train" and r["patch"]]
    va = [r for r in rows if r["split"] == "val" and r["patch"]]
    if not va:
        raise ValueError("no utterances in the `val` split; the manifest has "
                         "%s" % sorted(set(r["split"] for r in rows)))
    x_tr, y_tr, _ = si_features.as_arrays(tr)
    x_va, y_va, v_va = si_features.as_arrays(va)
    return x_tr, y_tr, x_va, y_va, v_va, rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest")
    ap.add_argument("--arch", default="dscnn", choices=("dscnn", "plain"))
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--pool", default="avg", choices=("avg", "max"))
    ap.add_argument("--unknown-weight", type=float, default=1.0,
                    help="multiplier on the unknown class's training weight; "
                         "above 1 trades recall for precision")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default="build/si",
                    help="prefix for the .keras, .tflite and .json artefacts")
    ap.add_argument("--cost-only", action="store_true",
                    help="print the size and MAC budget and stop")
    args = ap.parse_args(argv[1:])

    if args.cost_only:
        model = build(args.arch, args.width, pool=args.pool)
        macs, wbytes, act = cost(model)
        print("%s width %.2f: %d params, %.3f MMAC, %.1f KB weights, "
              "%.1f KB peak activation"
              % (args.arch, args.width, model.count_params(), macs / 1e6,
                 wbytes / 1024.0, act / 1024.0))
        return 0

    x_tr, y_tr, x_va, y_va, v_va, rows = load_split(
        args.manifest, args.cache, args.jobs)
    print("train %d, val %d (%d held-out voices)"
          % (len(x_tr), len(x_va), len(set(v_va))))

    t0 = time.time()
    model = train(x_tr, y_tr, x_va, y_va, args.arch, args.width,
                  args.epochs, args.batch, args.seed, pool=args.pool,
                  unknown_weight=args.unknown_weight)
    print("trained in %.1f s" % (time.time() - t0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out + ".keras")
    blob = to_int8_tflite(model, x_tr, args.out + ".tflite")
    macs, wbytes, act = cost(model)
    meta = {"arch": args.arch, "width": args.width, "pool": args.pool,
            "unknown_weight": args.unknown_weight,
            "classes": list(si_features.CLASSES),
            "n_frames": si_features.N_FRAMES,
            "n_bands": si_features.N_BANDS,
            "input_shift": si_features.INPUT_SHIFT,
            "feature_key": si_features.feature_key(),
            "params": int(model.count_params()), "macs": int(macs),
            "weight_bytes_est": int(wbytes), "peak_activation": int(act),
            "tflite_bytes": len(blob),
            # The device side needs these to read the output. Probability is
            # `(q - output_zero_point) * output_scale`; at scale 1/256 and
            # zero -128 that is `(q + 128) / 256`, exact in integers.
            "input_scale": 1.0, "input_zero_point": 0,
            "output_scale": float(_out_quant(blob)[0]),
            "output_zero_point": int(_out_quant(blob)[1])}
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    # What quantisation cost, on the held-out voices. Printed here rather than
    # left to the evaluator because it is the one question whose answer decides
    # whether to reach for quantisation-aware training, and it is cheap.
    val_float = model.predict(x_va.astype("float32")[..., None], verbose=0)
    val_int8 = tflite_predict(blob, x_va)
    keyword = y_va != si_features.CLASSES.index(si_features.UNKNOWN)
    for name, probs in (("float32", val_float), ("int8   ", val_int8)):
        pred = probs.argmax(axis=1)
        print("%s  val top-1 all %.3f   keywords only %.3f"
              % (name, float((pred == y_va).mean()),
                 float((pred[keyword] == y_va[keyword]).mean())
                 if keyword.any() else float("nan")))
    agree = float((val_float.argmax(axis=1) == val_int8.argmax(axis=1)).mean())
    print("float and int8 agree on %.3f of held-out utterances" % agree)
    meta["val_top1_float"] = float((val_float.argmax(axis=1) == y_va).mean())
    meta["val_top1_int8"] = float((val_int8.argmax(axis=1) == y_va).mean())
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("wrote %s.keras, %s.tflite (%d bytes), %s.json"
          % (args.out, args.out, len(blob), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
