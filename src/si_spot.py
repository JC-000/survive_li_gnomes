"""The speaker-independent keyword spotter: log-mel patch -> int8 CNN -> label.

Replaces the DTW matcher as the recogniser that ships. `src/spotter.py` stays
deployed and is not deleted -- it is the front end this file taps, it is the
specification the port is proved against, and it is the fallback if the native
module misbehaves. What changed is which one `talk.py` asks first.

## Why this one and not DTW

DTW needs templates enrolled through this board by the person who will use it.
Measured, cross-microphone enrolment collapses top-1 from 69/70 to 36/70
(`docs/speech.md`), and the failure mode is not silence but *confident errors*.
The build this is going into is a workshop piece for many speakers with no
enrolment step, so speaker dependence was the wrong shape regardless of accuracy.

It is also cheaper. Measured on this board: **66.6 ms** of inference against
DTW's **616-672 ms** of matching, and flat in class count where DTW is linear in
template count. See `docs/cnn-on-device.md`.

## What the native module requires, and why each line below exists

`emlearn_cnn_int8` is a thin wrapper over TinyMaix and several of its
assumptions are the caller's problem. Four of them bite silently and are guarded
on the host by `tools/tmdl_info.py`, which every model must pass before it is
deployed -- `tools/deploy.sh` refuses otherwise:

- the activation scratch is sized to the **model file length**, unchecked, so a
  model needing more writes past its allocation. Confirmed on this board: 1164
  bytes overwritten, nothing raised. The shipped `.tmdl` is `--pad`ded.
- a per-channel-quantised dense layer silently uses only the first scale;
- `out_deq` must be 1 or the scores come back as reinterpreted bytes;
- `run()` takes `array('B')` and returns into `array('f')`. A `bytearray` is
  rejected outright.

## The class order is derived, never copied

`CLASSES` is built from `vocab.LABELS`, which is what `tools/si_features.py`
does to build the training labels. Both sides derive from the one source, so
they cannot drift apart the way a transcribed list would -- and `bind()` checks
the model's output width against it, so a model trained on a different
vocabulary fails loudly at load instead of relabelling every reply.

## Rejection is the whole game

`docs/speech.md` argues it at length and none of it changed: `argmax` always
returns a word, so without a gate a cough becomes "DO YOU OFTEN THINK OF MONEY"
and the illusion dies. A miss costs nothing, because ELIZA deflects in character.

There are three gates here rather than DTW's two. The extra one is the trained
`unknown` class, which is the honest way to say "none of these" -- a softmax over
keywords alone cannot represent it and has to fake it by being unconfident.
"""

from array import array

import si_patch

try:
    import vocab
except ImportError:
    vocab = None

# The native modules. Both absent on the host, and either can be absent on a
# board; every combination must degrade to "no keyword" rather than fail to
# start, exactly as the DTW spotter does.
#
# `emlearn_cnn_int8` is TinyMaix, loaded from a `.mpy`. `tflm` is TensorFlow
# Lite Micro, compiled into the firmware -- see docs/tflm-usermod.md. They are
# not equivalent: TinyMaix approximates TFLite's int8 arithmetic and TFLM
# reproduces it exactly, which is the entire reason the second one exists.
try:
    import emlearn_cnn_int8 as _cnn
except ImportError:
    _cnn = None

try:
    import tflm as _tflm
except ImportError:
    _tflm = None

MODEL_PATH = "si_model.tmdl"
MODEL_PATH_TFLM = "si_model.tflite"

# TFLM plans its whole working set into one caller-owned buffer -- the model
# copy and every activation. Measured at 58,752 B on a 64-bit host and expected
# lower on the board, where the metadata pointers halve; `arena_used()` reports
# the truth and `tools/tflm_probe.py` prints it. 64 KB is generous on purpose:
# too small raises at bind() rather than corrupting the heap, which is the one
# behaviour TinyMaix does not have.
TFLM_ARENA_BYTES = 64 * 1024

# Which runtime `bind()` reaches for first.
#
# **TFLM, since 2026-08-19, when the numbers this line used to wait for
# arrived.** On this board, all 30 comparison cases came back byte-identical to
# host TFLM, worst count 0, confirmed by two independently written chains
# (`tools/check_tflm_device.py` exit 0, `tools/tflm_compare_cases.py` exit 0).
# TinyMaix, run in the same session on the same inputs, disagreed with the host
# on top-1 in 5 of the 30 -- the 3 `kw_unknown` patches already on record and,
# newly, two real recordings (`problem_01`, `wonder_01`), worst deviation 190
# counts against a MARGIN measured in single counts. So the argument for TFLM
# stopped being an argument.
#
# It costs inference time: 245.6 ms against TinyMaix's 66.6 ms, measured in that
# same session, which takes a turn from ~940 ms to ~1120 ms. Bought deliberately
# -- it is what makes a host-measured operating point mean anything here.
#
# **THRESHOLD/MARGIN/TIE_FLOOR below are still the TinyMaix-tuned values**, and
# that is a real loose end rather than an oversight; see the note above them.
#
# `bind(backend="tinymaix")` forces the incumbent, which is how the two get
# compared. A board carrying only one of them falls through to it rather than
# going silent.
BACKEND_DEFAULT = "tflm"

UNKNOWN = "unknown"
CLASSES = (tuple(vocab.LABELS) + (UNKNOWN,)) if vocab is not None else ()
UNKNOWN_INDEX = len(CLASSES) - 1

# --- The operating point ---------------------------------------------------
#
# **Measured on this board**, 2026-08-18, against 10 real takes of 9 keyword
# classes and 12 real out-of-vocabulary takes, all endpointed by `src/vad.py`
# and run through this exact path -- `si_patch` plus TinyMaix, not the `.tflite`.
#
#     precision 1.000, recall 0.600
#
# Re-measuring was not optional. `si-model` measured precision 1.000 at recall
# 0.500 against the `.tflite`, and that number does not transfer: TinyMaix is an
# independent reimplementation and computes different values from the same
# weights. The placeholder these constants replaced -- 0.90 and 0.50, picked to
# be conservative -- would have fired on **nothing at all**, recall 0.000. A
# plausible-looking guess at a gate is a spotter that never speaks.
#
# ## The gates are very nearly inert, and that is the finding
#
# Every one of the 12 negatives came back `unknown` by argmax, and no positive
# was ever labelled as the *wrong* keyword. So precision is 1.000 at any
# setting, and the threshold and margin only ever cost recall. **The trained
# `unknown` class is doing all of the rejection work**, which is the argument
# for having trained one rather than thresholding a 21-way softmax.
#
# They are set to something rather than to zero anyway, for one measured reason
# and one principled one:
#
# - MARGIN excludes **ties**. The `father` take came back correct with a margin
#   of exactly 0.0000 -- top-1 and top-2 equal -- so it was a coin flip that
#   happened to land right. `docs/speech.md` is explicit that an utterance
#   resembling two things equally should produce silence whatever its absolute
#   score, and that argument does not depend on which way this particular coin
#   came down. The softmax output is dequantised int8 with `out_s = 1/256`, so
#   2/256 is "at least two quantisation steps clear" and is the smallest gate
#   that means anything. It costs exactly one take: recall 0.700 -> 0.600.
# - THRESHOLD is a floor below every correct fire in the set (the lowest is
#   `dream` at 0.402), so on this data it costs **nothing measurable**. It is
#   there against a low-confidence wrong-keyword fire, of which this set
#   contains zero examples -- so it is prudence, not evidence.
#
# ## What this operating point is not
#
# Ten positives and twelve negatives. Recall moves in steps of 0.1 and precision
# 1.000 rests on twelve negatives never firing. It is enough to ship a workshop
# toy whose failure mode is a deflection; it is not enough to quote as an
# accuracy figure, and the honest next step is more negatives rather than more
# tuning. `docs/cnn-on-device.md` records the full table.
# ## These are TinyMaix's numbers, and the runtime under them changed
#
# **Open, as of 2026-08-19.** Everything above was measured through TinyMaix.
# The default runtime is now TFLM, which computes genuinely different
# probabilities from the same weights -- that is the whole finding, not a
# rounding difference.
#
# What the switch actually does to this gate is *measured*, from the 22 real
# takes in the device capture that proved the byte-exactness:
#
#     THRESHOLD 0.35   ->  8 of 22 takes fire
#     THRESHOLD 0.637  ->  2 of 22 takes fire
#
# 0.637 is si-model's host sweep under the reference kernels (precision 1.000,
# recall 0.300, docs/speaker-independent.md), and it now transfers to the device
# by construction. It is not installed here yet, because installing it would be
# the third distinct operating point in this file's history and the first one
# nobody has run a turn against. Note also that `father_01`, a correct fire,
# lands at 163/256 = 0.6367 -- one quantisation step *below* a literal 0.637,
# so at that setting the rounding of the constant decides the answer.
#
# Resolved 2026-08-19, on the quantisation grid rather than in decimal. The
# sweep's 0.637 *meant* "one count above the worst negative": the worst
# negative under reference kernels is `problem` scoring brother at exactly
# 160/256 = 0.625, and the blessed decimal, taken literally, also excludes
# `father_01` at 163/256 = 0.63672 -- a correct fire deleted by rounding. So
# the TFLM floor is written as the count it is: 161/256, strictly above every
# stay-silent score in the reference capture, at-or-below every correct fire
# the sweep kept. Same operating point the sweep blessed (precision 1.000,
# recall 0.300 on the 22 stored takes), minus the rounding bug. Scores here
# ARE grid points -- both runtimes dequantise int8 at 1/256 -- which is why a
# threshold expressed off-grid was a latent coin-flip.
#
# Each backend carries its own floor and bind() selects it; TinyMaix's live-
# tuned 0.35 is untouched for the fallback path.
THRESHOLD = 0.35        # top-1 probability floor  -- costs nothing on this set
# 2026-08-19, bench verdict: at 161/256 the toy fired on roughly one clear
# "mother" in three and the user called it "not great about responding". The
# floor drops to the live-tuned 0.35: chattier, at the documented price that
# `problem` can fire "Your brother?" (0.625 under these kernels). The strict
# point remains one edit away; both are measured, and which toy this is is the
# user's call -- made, at the desk, in favour of answering.
THRESHOLD_BY_BACKEND = {"tinymaix": THRESHOLD, "tflm": THRESHOLD}
MARGIN = 2.0 / 256.0    # top-1 minus top-2, two output quantisation steps

# An exact tie is forgiven, but only above this probability. Third tuned
# constant, and the only one measured on data this device did not produce.
#
# `MARGIN` above exists to reject ties, on the argument -- which still holds --
# that an utterance resembling two things equally should produce silence. But
# ties are *common* here in a way they are not in a float model: the softmax
# output is dequantised int8 at 1/256, so a genuine two-way split lands on an
# exact tie rather than near one, and three of the labelled live correct answers
# were tie-gated (father 0.500, husband 0.492, dream 0.492).
#
# The distinction that rescues them: at 1/256 resolution a tie at p ~ 0.5 means
# two classes took ~128 counts each and everything else took ~0 -- a confident
# two-way split. A tie at p ~ 0.28 means the mass is spread and nothing stands
# out. So the tie is forgiven only when the top-1 probability says the split was
# confident.
#
# Measured by si-model over 4106 must-stay-silent and 1858 in-vocabulary
# synthetic utterances, against forgiving ties unconditionally:
#
#     tie allowed when          new correct   new false fires   new wrong keyword
#     argmax != unknown              +19            +79                +9
#     that, and top-1 >= 0.49         +3             +1                 0
#
# **The unconditional version is 4.6 wrong fires for every right one** and must
# not be shipped. It looked free on 19 live utterances because 4106 adversarial
# negatives were not in the room.
#
# Corroborated independently on this board's own 22 takes: it admits the one tie
# there (father, correct, p=0.4922) and no negative can reach it, since every
# negative came back argmax `unknown`. Precision 1.000 held, recall 0.600 ->
# 0.700.
#
# **The band this covers is empty today, and that is measured.** Over 6426
# synthetic utterances the margin distribution is bimodal: 113 at exactly 0/256,
# **nothing at all between 1/256 and 6/256**, 2 at 7/256, and 6311 at 8/256 or
# more. That looks like a property of the arithmetic rather than of one corpus
# -- a 22-class softmax quantised at 1/256 either has one class dominating or
# has two collapsing onto the same integer, and an intermediate gap needs the
# underlying float difference to land in one narrow window.
#
# So covering the band rather than only the tie point is chosen on **reasoning,
# not measurement**: all three candidate rules score identically on that corpus
# (tp 1418, wrong-keyword 94, false-oov 1184 -- not close, identical). The
# reasoning is that a flatter model would populate the band, and the floor is
# what the measurement actually means -- a near-tie at p~0.5 is a confident
# two-way split, a near-tie at p=0.28 is nothing standing out. **If that band
# ever fills, re-measure rather than trusting this note.**
#
# **It is not free.** On the synthetic set it costs one false fire in 4106, so
# precision is no longer exactly 1.000 there. That is a real departure from what
# docs/speech.md asks for, taken because three of ten live keywords is a large
# fraction of a demo and one in 4106 is not. The distributions overlap -- tying
# negatives run median 0.441 / max 0.496 against tying correct answers at median
# 0.453 / max 0.492 -- so 0.49 is where the trade is best, not a clean cut.
TIE_FLOOR = 0.49


class _TinyMaix:
    """The incumbent: `emlearn_cnn_int8` over a `.tmdl`.

    Every line of this is a requirement of the wrapper rather than a choice --
    see the module docstring above, and `tools/tmdl_info.py` for the four
    failure modes it does not report.
    """

    name = "tinymaix"
    default_path = MODEL_PATH

    def __init__(self, path):
        handle = open(path, "rb")
        try:
            blob = handle.read()
        finally:
            handle.close()

        # array('B') and not bytearray: the wrapper checks the buffer's
        # typecode and raises ValueError("model should be bytes") for a
        # bytearray. Verified on the board.
        data = array("B", blob)
        del blob
        self.model = _cnn.new(data)
        del data
        self.n_classes = self.model.output_dimensions()[0]

    def run(self, patch, scores):
        self.model.run(patch, scores)


class _Tflm:
    """TensorFlow Lite Micro over the `.tflite` itself.

    Cheaper to set up than the above and with fewer ways to be silently wrong:
    the arena is ours, an undersized one raises, and `bytes` straight off the
    filesystem is an acceptable input rather than something to be copied into
    an `array('B')` at 2.06x the file's size.
    """

    name = "tflm"
    default_path = MODEL_PATH_TFLM

    def __init__(self, path, arena_bytes=TFLM_ARENA_BYTES, arena=None, blob=None):
        if blob is None:
            # On the device the blob should arrive pre-read: this read is a
            # ~30 KB allocation, and post-import the heap refused exactly that
            # (MemoryError: 30208) once the arena had been rescued the same way.
            handle = open(path, "rb")
            try:
                blob = handle.read()
            finally:
                handle.close()
        # Held for the object's life: TFLM plans into it and keeps pointers
        # into it, so it must outlive the model exactly as the arena does in
        # every other TFLM integration.
        #
        # `arena` may be handed in pre-allocated. On the device it MUST be:
        # allocating it here means allocating after every heavy import, and
        # the 64 KB contiguous block reliably no longer exists by then -- the
        # voice binding's import weight was the third thing to starve it.
        # talk.py reserves it at module top with the other early buffers.
        self.arena = arena if arena is not None else bytearray(arena_bytes)
        self.model = _tflm.new(blob, self.arena)
        self.n_classes = self.model.output_dimensions()[0]

    @property
    def arena_used(self):
        return self.model.arena_used()

    def run(self, patch, scores):
        self.model.run(patch, scores)


def _backend_for(name):
    """-> (class, why-not). Exactly one of the two is None."""
    if name == "tflm":
        if _tflm is None:
            return None, "tflm not in this firmware"
        return _Tflm, None
    if name == "tinymaix":
        if _cnn is None:
            return None, "emlearn_cnn_int8 not deployed"
        return _TinyMaix, None
    return None, "unknown backend %r" % (name,)


class Spotter:
    """Holds the model, the front-end scratch and the patch buffer.

    One object for the life of the program. `bind()` is separate from `__init__`
    so the caller controls *when* the 43 KB of model lands on the heap --
    allocation order is a whole-program concern and `docs/speech.md` records
    what learning that cost.
    """

    def __init__(self):
        self.model = None
        self.buffers = None
        self.scores = None
        self.n_classes = 0
        self.error = None
        self.backend = None
        self.threshold = THRESHOLD

    @property
    def available(self):
        return self.model is not None

    def bind(self, path=None, buffers=None, backend=None, arena=None, blob=None):
        """Load the model. Returns True on success; never raises.

        A board that cannot load the model must still hold a conversation --
        every turn simply takes DOCTOR's no-keyword path, which is a real ELIZA
        behaviour. `self.error` carries why, because a spotter that is silently
        absent looks exactly like one that is merely strict.

        `backend` is "tinymaix", "tflm", or None for BACKEND_DEFAULT with a
        fall-through to the other if the default's module is not in this
        firmware. `path` defaults to whichever model file that backend wants,
        so the ordinary caller passes neither and the A/B passes only
        `backend=`. **Everything below the model -- the features, the three
        gates, the labels -- is shared code**, so a difference between the two
        runs is a difference between the runtimes and nothing else.
        """
        if not CLASSES:
            self.error = "vocab not deployed, so no class order"
            return False

        wanted = backend or BACKEND_DEFAULT
        factory, why = _backend_for(wanted)
        if factory is None and backend is None:
            # Import-fallback, and only when the caller did not ask for a
            # specific runtime. An explicit request that cannot be honoured is
            # an error, not an invitation to run the other one and report
            # numbers under the wrong name.
            other = "tflm" if wanted == "tinymaix" else "tinymaix"
            fallback, why_other = _backend_for(other)
            if fallback is not None:
                factory, why = fallback, None
            else:
                why = "%s; %s" % (why, why_other)
        if factory is None:
            self.error = why
            return False

        try:
            kwargs = {}
            if factory.name == "tflm":
                if arena is not None:
                    kwargs["arena"] = arena
                if blob is not None:
                    kwargs["blob"] = blob
            self.model = factory(path if path is not None
                                 else factory.default_path, **kwargs)
            self.backend = factory.name
            self.threshold = THRESHOLD_BY_BACKEND.get(factory.name, THRESHOLD)
            self.n_classes = self.model.n_classes
            if self.n_classes != len(CLASSES):
                # Loud, and at load rather than at the first reply. A model with
                # a different output width is a model trained on a different
                # vocabulary, and quietly using it would relabel every answer.
                self.model = None
                self.backend = None
                self.error = ("model has %d outputs, vocabulary has %d"
                              % (self.n_classes, len(CLASSES)))
                return False

            self.scores = array("f", (0.0 for _ in range(self.n_classes)))
            self.buffers = buffers if buffers is not None else si_patch.allocate()
            self.error = None
            return True
        except Exception as exc:      # noqa: BLE001 -- degrading is the point
            self.model = None
            self.backend = None
            self.error = "%s: %s" % (type(exc).__name__, exc)
            return False

    def scored(self, samples, start, count, pre=None):
        """-> (label or None, top1 probability, margin, class index,
              frames, clipped).

        `frames` and `clipped` are diagnostics, not results, and they are
        returned rather than dropped because between them they distinguish the
        two ways a live utterance goes wrong without anything raising. `frames`
        says what the endpointer handed over -- a word clipped at its onset
        shows up as a short span. `clipped` counts feature values that hit the
        int8 rail: a handful is the spectral floor doing its job, and a large
        count means the input is railing or the shift is wrong, which is
        invisible in the probabilities but obvious here.

        The scores are returned whether or not the gates fire, because the gates
        are numbers that still have to be tuned on this board and a device that
        prints only its verdict gives nobody anything to tune with.
        """
        if self.model is None:
            return None, 0.0, 0.0, -1, 0, 0

        patch, n_frames, clipped = si_patch.patch_for(
            samples, start, count, self.buffers, pre)
        if patch is None:
            return None, 0.0, 0.0, -1, 0, 0

        self.model.run(patch, self.scores)

        scores = self.scores
        best = 0
        for i in range(1, self.n_classes):
            if scores[i] > scores[best]:
                best = i
        second = -1.0
        for i in range(self.n_classes):
            if i != best and scores[i] > second:
                second = scores[i]

        top = scores[best]
        margin = top - second

        label = None
        if best != UNKNOWN_INDEX and top >= self.threshold:
            # Monotone in `margin` by construction: the floor covers the whole
            # sub-MARGIN band rather than only the exact-tie point at its
            # bottom. An earlier version tested `margin == 0.0` here and left a
            # one-step hole -- a margin of 1/256 was rejected while 0/256 was
            # accepted -- which is not a rule anyone can hold in their head and
            # would have come back if `MARGIN` were ever moved.
            enough_separation = margin >= MARGIN
            confident_split = top >= TIE_FLOOR   # rescues the sub-MARGIN band
            if enough_separation or confident_split:
                label = CLASSES[best]
        return label, top, margin, best, n_frames, clipped

    def spot(self, samples, start, count, pre=None):
        return self.scored(samples, start, count, pre)[0]
