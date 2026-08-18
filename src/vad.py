"""Endpointing: where in a push-to-talk buffer the word actually is.

Rabiner & Sambur, "An Algorithm for Determining the Endpoints of Isolated
Utterances" (1975) -- short-time energy plus zero-crossing rate, three
thresholds. See docs/speech-design.md#endpointing for why that paper, and
`tools/vad.py` for the host-side twin.

## This file and tools/vad.py must agree, exactly

Templates are enrolled on the host and matched on the device. If the two sides
trim a word differently -- 30 ms of margin here, none there -- then every
template is compared against a differently-shaped runtime segment, and DTW
distances drift for a reason that never shows up in either file's own tests.

So `frame_stats`, `thresholds` and `endpoints` below are the same algorithm and
the same constants as `tools/vad.py`, and `tools/test_vad.py` asserts they
return identical answers on the same audio. If you change one, change both and
run that test. (One file imported by both would be better; the device copy
exists because `tools/vad.py` also carries WAV loading and a CLI, and a module
that imports `wave` will not load on the board.)

## What this file adds on top

The paper assumes the whole utterance is already in hand -- its ITL is a
fraction of the buffer's peak energy, which is not knowable while still
recording. The `Endpointer` class adds a live layer that decides *when to stop
recording* from the noise floor alone, and then hands the final boundaries to
`endpoints()`, so what the recogniser sees always comes from the same code as
the host.

The live layer's parameters (20 ms decisions, an EMA noise floor over the first
300 ms, 4x/2x hysteresis, a 3-unit start trigger, a 400 ms hangover) are a
separate set from the paper's and are tuned separately, because **they cannot
invalidate a template**: they choose the moment recording stops, not where the
word is. Nothing below the constants block is shared with the host.

One item from that parameter set is deliberately absent: a **pre-roll ring
buffer**. It has no job here. A design that only starts writing once the VAD
triggers has to keep the preceding 200-300 ms somewhere or lose the /y/ of YES
before the energy crosses threshold -- but this records continuously from the
moment of the press, so the onset was never at risk: it is already in the
buffer, and `endpoints()` walks *backwards* to it over the whole recording.
Adding a ring buffer in front of a recording that already contains the audio
would be storing it twice.

Two things that fail silently if you get them wrong:

- **Gate this during playback.** The speaker and the microphone are the same
  ES8311. A VAD left running while the board makes a noise triggers on the
  board. `set_playing(True)` across playback and a little past it -- frames
  handed over while gated are dropped, not analysed. `shake.py` swallows its own
  exceptions by design, so a self-trigger produces no traceback at all: the
  symptom is ELIZA answering a question nobody asked.
- **Energy is `sum(abs(x))`, never squared.** Squared energy over a 160-sample
  frame of full-scale int16 reaches 1.7e11 and needs 64-bit accumulation; |x|
  reaches 5.2e6 and fits an int32 nine times over. It also keeps the inner loop
  in integer arithmetic, which is the only kind `@micropython.viper` speeds up
  -- roughly 16x on integer and bit work, and nothing whatsoever on floats.

## On the viper override

The live pass alone does not need it: framing 20 ms of audio at a time is a few
percent of one core in plain bytecode. The override is here because the *offline*
pass walks the whole recording -- up to 3 s, 48 000 samples -- and because the
portable version stays the specification either way. It is optional in the
strictest sense: if the decorator does not compile, the reference implementation
runs instead and nothing changes but the timing. `tools/speech_probe.py` times
both on real audio, asserts they return identical results, and prints a warning
if the speedup is under 2x, which is what "viper did not take" looks like.
"""

from array import array

# --- shared with tools/vad.py: change both together ------------------------
VAD_FRAME = 160          # 10 ms at 16 kHz, matching the paper's frame -- and,
                         # deliberately, the MFCC hop exactly
                         # (speech_tables.FRAME_STRIDE, also 160). Because they
                         # are equal, every endpoint lands on an MFCC frame
                         # boundary and the segment the recogniser sees is the
                         # segment the VAD chose, with no rounding either way.
                         # If one moves, move both.
SILENCE_FRAMES = 10      # first 100 ms is assumed to be background
ZCR_LOOKBACK = 25        # frames the ZCR pass may reach back over (250 ms)
ZCR_MIN_HITS = 3         # this many high-ZCR frames before the endpoint moves
ZCR_FLOOR = 25           # crossings per 10 ms; the paper's fixed threshold,
                         # kept as a lower bound rather than an upper one
ZCR_SIGMA = 4            # a ZCR hit must clear background by this many sigma
ZCR_GATE_MIN_FRAC = 3    # ...and by at least imn >> this, if sigma is tiny
MARGIN_FRAMES = 3        # 30 ms kept either side, so nothing is shaved
MIN_SPEECH_FRAMES = 15   # 150 ms; shorter than this is a click, not a word
MAX_SPEECH_FRAMES = 180  # 1.8 s; longer is a sentence, and not our problem
# ---------------------------------------------------------------------------

# --- the live layer: deciding when to STOP recording -----------------------
#
# None of this touches the boundaries handed to the recogniser -- those come
# from endpoints() above, which the host shares. These parameters only decide
# when the talker has finished, so they can be tuned freely without
# invalidating a single enrolled template.
#
# Starting points, not measurements. Every one of them wants a look at real
# room audio through tools/speech_probe.py before it is believed.

# A live decision is made over 20 ms, i.e. two of the shared 10 ms frames.
# 10 ms alone is spike-sensitive -- a knuckle on the panel is one frame -- and
# 30 ms smears the boundary of a short word. The *statistics* stay at 10 ms
# because that frame length is shared with tools/vad.py; only the decision is
# coarser.
LIVE_FRAMES = 2

# The noise floor is an EMA over roughly the first 300 ms, then frozen:
# floor += (energy - floor) >> 5. Frozen because an EMA that kept adapting
# through the utterance would climb towards the speech it is meant to detect and
# end the recording early.
#
# The offline pass still uses the first 100 ms (SILENCE_FRAMES) for its own IMN,
# because that constant is shared with the host. The two windows serve different
# jobs and do not have to agree.
LIVE_CALIBRATION_UNITS = 15
LIVE_EMA_SHIFT = 5

# Hysteresis, as multiples of the floor. Starting and stopping at the same level
# makes the boundary flicker on every breath.
LIVE_START_MULTIPLE = 4
LIVE_END_MULTIPLE = 2

# 60 ms above the start threshold before it counts as speech. Rejects clicks,
# knocks and the screen being tapped rather than held.
LIVE_START_UNITS = 3

# 400 ms below the end threshold ends the utterance. Below about 300 ms a
# trailing fricative gets clipped, and so does the gap in the middle of "what...
# do you mean".
LIVE_HANGOVER_UNITS = 20

# Guards on the calibration. Without the floor, a silent room drives the
# thresholds to zero and every frame reads as speech; without the ceiling, a
# user who starts talking before calibration is done calibrates against their
# own voice and nothing is ever loud enough afterwards. In sum(abs(x)) units per
# 20 ms unit: a mean sample magnitude of 32 (about -60 dBFS) and of 2000 (about
# -24 dBFS).
NOISE_FLOOR_MIN = VAD_FRAME * LIVE_FRAMES * 32
NOISE_FLOOR_MAX = VAD_FRAME * LIVE_FRAMES * 2000

# Live states
IDLE = 0
SPEAKING = 1
DONE = 2


def _frame_stats(samples, start, count):
    """(energy << 8) | zero_crossings for one frame.

    Portable reference implementation, and the specification: the viper build
    below must agree with it exactly, including the convention that a sample of
    zero counts as positive (`(v < 0) != (prev < 0)`, as in tools/vad.py).

    Packed into one machine int rather than returned as a tuple because viper
    cannot build a tuple without allocating, and this runs once per 10 ms of
    audio. The packing is why VAD_FRAME must stay at or below 256 samples:
    256 * 32768 << 8 is the last value that fits in 31 bits.

    Indexing an array("h") yields signed values here; viper's ptr16 does not,
    and sign-extends by hand.
    """
    energy = 0
    crossings = 0
    prev = 0
    for i in range(count):
        value = samples[start + i]
        if value < 0:
            energy -= value
            sign = -1
        else:
            energy += value
            sign = 1
        if i and sign != prev:
            crossings += 1
        prev = sign
    return (energy << 8) | crossings


try:  # pragma: no cover -- device only
    # On the device this replaces the reference implementation above. Under host
    # CPython the import fails before either the ptr16 annotations or the
    # decorator are evaluated, and the portable version stands.
    import micropython

    @micropython.viper
    def _frame_stats(samples: ptr16, start: int, count: int) -> int:  # noqa: F811
        energy = 0
        crossings = 0
        prev = 0
        i = 0
        while i < count:
            value = int(samples[start + i])
            if value > 32767:  # ptr16 reads unsigned; sign-extend
                value = value - 65536
            if value < 0:
                energy = energy - value
                sign = -1
            else:
                energy = energy + value
                sign = 1
            if i:
                if sign != prev:
                    crossings = crossings + 1
            prev = sign
            i += 1
        return (energy << 8) | crossings
except ImportError:
    pass


def _isqrt(value):
    """Integer square root, Newton.

    `math.isqrt` is not in every MicroPython build, and importing `math` for one
    call in an otherwise integer-only module is a poor trade for six lines.
    """
    if value <= 0:
        return 0
    guess = value
    step = (value + 1) // 2
    while step < guess:
        guess = step
        step = (guess + value // guess) // 2
    return guess


# --- the paper's algorithm, shared with tools/vad.py -----------------------

def frame_stats(samples, n_frames, energy=None, zcr=None):
    """Per-frame (sum |x|, zero crossings) for `n_frames` frames.

    Fills and returns the caller's arrays if given: the device path reuses the
    ones the live layer already filled, so a turn costs no allocation at all.
    """
    if energy is None:
        energy = array("i", bytearray(4 * n_frames))
    if zcr is None:
        zcr = array("i", bytearray(4 * n_frames))
    for f in range(n_frames):
        packed = _frame_stats(samples, f * VAD_FRAME, VAD_FRAME)
        energy[f] = packed >> 8
        zcr[f] = packed & 0xFF
    return energy, zcr


def _mean(values, first, count):
    total = 0
    for i in range(first, first + count):
        total += values[i]
    return total // count


def _std(values, first, count, mean):
    total = 0
    for i in range(first, first + count):
        delta = values[i] - mean
        total += delta * delta
    return _isqrt(total // count)


def thresholds(energy, zcr, n_frames=None):
    """The paper's three thresholds -- ITL, ITU, IZCT -- plus the noise mean.

    ITL/ITU are derived from the quietest and loudest of the buffer, so they
    track the recording level rather than assuming one.

    IZCT departs from the paper, deliberately. R&S take `min(25, mean +
    2*sigma)`, which assumes a background quiet enough to have a *low* crossing
    rate -- true of a 1975 analog lead-in, false of any digitised one. A
    broadband noise floor crosses zero on nearly every other sample: measured on
    the host corpus, silence runs at 75-90 crossings per 10 ms frame while a
    voiced vowel runs at 3-14. Taking the minimum sets the threshold far *below*
    the background, every silent frame reads as frication, and the endpoint
    walks out to the full lookback every time -- measured before the fix, every
    utterance endpointed to 930-1090 ms regardless of the word.

    So the fixed 25 becomes a lower bound and the adaptive term may rise above
    it. Frication still stands out -- the /s/ of SORRY measures 105-126 against
    a background of 73-92 -- but only just, which is why _zcr_backoff also gates
    on energy.

    That energy gate -- `egate` -- is four standard deviations above the
    background, not a multiple of it, and the difference is not cosmetic. A
    multiple of the mean has to clear ITL to be useful, because a frame above
    ITL is already inside the energy run and needs no help; but ITL is
    `min(3% of (imx - imn) + imn, 4 * imn)`, so a gate at `2 * imn` sits above
    ITL whenever `imx < ~34 * imn` and the whole zero-crossing pass switches
    itself off. Measured on the host corpus: that was 37 of 175 utterances, all
    of them the quieter ones, which is exactly where the pass is wanted. Four
    sigma lands at 1.12-1.34x the mean instead of a flat 2.00, no utterance in
    the corpus has an empty band any more, and the pass moves an endpoint 20
    times out of 175 rather than 8.

    `imn >> ZCR_GATE_MIN_FRAC` is a floor under that, for a pathological
    zero-variance background where sigma alone would admit every frame.
    """
    if n_frames is None:
        n_frames = len(energy)
    quiet = SILENCE_FRAMES if n_frames >= SILENCE_FRAMES else n_frames
    imn = _mean(energy, 0, quiet)
    imx = 0
    for f in range(n_frames):
        if energy[f] > imx:
            imx = energy[f]

    i1 = (3 * (imx - imn)) // 100 + imn   # 3% of the dynamic range above quiet
    i2 = 4 * imn
    itl = i1 if i1 < i2 else i2
    if itl < 1:
        itl = 1
    itu = 5 * itl

    zmean = _mean(zcr, 0, quiet)
    zstd = _std(zcr, 0, quiet, zmean)
    izct = zmean + 2 * zstd
    if izct < ZCR_FLOOR:
        izct = ZCR_FLOOR

    estd = _std(energy, 0, quiet, imn)
    egate = imn + ZCR_SIGMA * estd
    floor = imn + (imn >> ZCR_GATE_MIN_FRAC)
    if egate < floor:
        egate = floor
    return itl, itu, izct, imn, egate


def _energy_onset(energy, n_frames, itl, itu, start, step):
    """First frame of the run that reaches ITU without falling back below ITL.

    Walking in `step` direction. Returns -1 if the energy never reaches ITU,
    which is how "there is no word in here" gets reported.
    """
    f = start
    while 0 <= f < n_frames:
        if energy[f] > itl:
            candidate = f
            g = f
            while 0 <= g < n_frames and energy[g] > itl:
                if energy[g] > itu:
                    return candidate
                g += step
            f = g            # the run died below ITU; resume past it
        f += step
    return -1


def _zcr_backoff(zcr, energy, n_frames, izct, egate, endpoint, step, limit):
    """Extend an endpoint outwards over unvoiced frication.

    Looks up to ZCR_LOOKBACK frames outwards from `endpoint`; if at least
    ZCR_MIN_HITS of them cross more often than IZCT *and* carry more energy than
    `egate`, the endpoint moves to the outermost of those frames.

    This is the part that keeps the /s/ on the front of SORRY and the /f/ on the
    front of FATHER. Both are 100-150 ms of low-energy, high-frequency noise
    that energy alone discards -- and FATHER stripped of its /f/ is MOTHER
    stripped of its /m/, which is the collision the vocabulary can least afford.

    The energy gate is the second half of the IZCT fix in thresholds(). Crossing
    rate alone cannot tell frication from a noise floor, because both are
    broadband; what separates them is that frication is audible. See
    `thresholds()` for why the gate is four sigma above the background rather
    than a multiple of it -- a multiple put the gate above ITL and made this
    whole function inert on 21% of the corpus.
    """
    hits = 0
    outermost = endpoint
    f = endpoint + step
    for _ in range(ZCR_LOOKBACK):
        if f < 0 or f >= n_frames or (step < 0 and f < limit) or (step > 0 and f > limit):
            break
        if zcr[f] > izct and energy[f] > egate:
            hits += 1
            outermost = f
        f += step
    if hits >= ZCR_MIN_HITS:
        return outermost
    return endpoint


def endpoints(samples, count=None, energy=None, zcr=None, n_frames=None):
    """(start_sample, end_sample) of the word, or None if there isn't one.

    `end` is exclusive and both are multiples of VAD_FRAME.

    `samples` may be None if `energy` and `zcr` are already filled -- that is
    the device path, reusing the frame statistics the live pass computed rather
    than walking 96 KB of audio a second time.
    """
    if n_frames is None:
        n_frames = (count if count is not None else len(samples)) // VAD_FRAME
    if n_frames < SILENCE_FRAMES + MIN_SPEECH_FRAMES:
        return None
    if energy is None or zcr is None:
        energy, zcr = frame_stats(samples, n_frames)
    itl, itu, izct, imn, egate = thresholds(energy, zcr, n_frames)

    n1 = _energy_onset(energy, n_frames, itl, itu, 0, 1)
    if n1 < 0:
        return None
    n2 = _energy_onset(energy, n_frames, itl, itu, n_frames - 1, -1)
    if n2 < 0 or n2 <= n1:
        return None

    n1 = _zcr_backoff(zcr, energy, n_frames, izct, egate, n1, -1, 0)
    n2 = _zcr_backoff(zcr, energy, n_frames, izct, egate, n2, 1, n_frames - 1)

    n1 -= MARGIN_FRAMES
    n2 += MARGIN_FRAMES
    if n1 < 0:
        n1 = 0
    if n2 > n_frames - 1:
        n2 = n_frames - 1

    length = n2 - n1 + 1
    if length < MIN_SPEECH_FRAMES:
        return None
    if length > MAX_SPEECH_FRAMES:
        # Keep the loudest MAX_SPEECH_FRAMES rather than failing: an over-long
        # segment is usually a word plus a door slam, and the word is louder.
        best_at, best = n1, -1
        for f in range(n1, n2 - MAX_SPEECH_FRAMES + 2):
            total = 0
            for g in range(f, f + MAX_SPEECH_FRAMES):
                total += energy[g]
            if total > best:
                best, best_at = total, f
        n1 = best_at
        n2 = n1 + MAX_SPEECH_FRAMES - 1

    return n1 * VAD_FRAME, (n2 + 1) * VAD_FRAME


def trim(samples, count=None):
    """Endpointed slice, or None. Convenience wrapper, matching tools/vad.py."""
    span = endpoints(samples, count)
    if span is None:
        return None
    return samples[span[0]:span[1]]


# --- the live layer, device only ------------------------------------------

class Endpointer:
    """Frames in while recording; says when to stop, then says where the word was.

    Deliberately one object doing both jobs, because they share the frame
    statistics: the live pass computes energy and ZCR once, and `bounds()`
    re-reads those same arrays. A turn allocates nothing.

    The live thresholds and the reported boundaries come from different rules,
    which is not an oversight:

    - **When to stop** is decided from the noise floor alone, because the peak
      of an utterance that has not finished is not knowable.
    - **Where the word is** comes from `endpoints()` -- the paper's algorithm,
      byte-identical to the host's -- because those boundaries are compared
      against templates that the host trimmed.
    """

    def __init__(self, max_frames=210, frame=VAD_FRAME):
        if frame != VAD_FRAME:
            # The frame length is baked into the shared constants and into the
            # packed _frame_stats result. Changing it here alone would silently
            # de-synchronise the device from the host.
            raise ValueError("frame length is shared with tools/vad.py; change both")
        self.frame = frame
        self.max_frames = max_frames
        self._energy = array("i", bytearray(4 * max_frames))
        self._zcr = array("i", bytearray(4 * max_frames))
        self.reset()

    def reset(self):
        self.frames = 0
        self.consumed = 0  # samples already turned into frames
        self.state = IDLE
        self.playing = False

        # Live state. `noise_floor` is per 20 ms unit, not per frame.
        self.noise_floor = 0
        self.units = 0
        self._unit_energy = 0
        self._unit_frames = 0
        self._above = 0
        self._quiet = 0
        self._speech_units = 0

    @property
    def live_start(self):
        """Energy per 20 ms unit above which speech is being heard."""
        return LIVE_START_MULTIPLE * self.noise_floor

    @property
    def live_end(self):
        """Energy per 20 ms unit below which the talker is counted as quiet."""
        return LIVE_END_MULTIPLE * self.noise_floor

    def set_playing(self, playing):
        """Mute the detector while the board itself is making a noise."""
        self.playing = bool(playing)

    def feed(self, samples, available):
        """Analyse whole frames up to `available` samples. Returns how many.

        Safe to call repeatedly during a capture; `available` is how much of the
        buffer the DMA has actually written, and anything past it is untouched
        memory that would read as convincing silence.
        """
        analysed = 0
        while (
            self.consumed + self.frame <= available
            and self.frames < self.max_frames
            and self.state != DONE
        ):
            if self.playing:
                self.consumed += self.frame  # gated: skipped, not analysed
                continue
            packed = _frame_stats(samples, self.consumed, self.frame)
            self._push(packed >> 8, packed & 0xFF)
            self.consumed += self.frame
            analysed += 1
        return analysed

    def _push(self, energy, crossings):
        """Record one 10 ms frame, and run the live decision every second one."""
        index = self.frames
        self._energy[index] = energy
        self._zcr[index] = crossings
        self.frames = index + 1

        self._unit_energy += energy
        self._unit_frames += 1
        if self._unit_frames < LIVE_FRAMES:
            return
        unit = self._unit_energy
        self._unit_energy = 0
        self._unit_frames = 0
        self.units += 1

        if self.units <= LIVE_CALIBRATION_UNITS:
            self._calibrate(unit)
            return

        if self.state == IDLE:
            if unit > self.live_start:
                self._above += 1
                if self._above >= LIVE_START_UNITS:
                    self.state = SPEAKING
                    self._speech_units = self._above
                    self._quiet = 0
            else:
                # Consecutive, not cumulative: three scattered loud units over a
                # second are a keyboard, not a word.
                self._above = 0
        elif self.state == SPEAKING:
            if unit < self.live_end:
                self._quiet += 1
                if self._quiet >= LIVE_HANGOVER_UNITS:
                    if self._speech_units >= MIN_SPEECH_FRAMES // LIVE_FRAMES:
                        self.state = DONE
                    else:
                        # Too short to be a word after all. Re-arm rather than
                        # ending the turn on a click that got through.
                        self.state = IDLE
                        self._above = 0
                        self._quiet = 0
            else:
                self._speech_units += 1
                self._quiet = 0

    def _calibrate(self, unit):
        """Track the room, for the first LIVE_CALIBRATION_UNITS only.

        The touch press is a wake event at exactly the right moment: the user
        has pressed but has not yet spoken, so the head of the recording is the
        room. Seeded from the first unit rather than from zero, so the EMA is
        already in the right place instead of climbing towards it.

        Nothing can trigger during calibration, which means a user who starts
        talking within 300 ms of pressing simply does not get an early stop --
        the recording runs to release or the cap, and the offline pass still
        finds the word. That is the safe direction to fail in.
        """
        if self.units == 1:
            self.noise_floor = unit
        else:
            self.noise_floor += (unit - self.noise_floor) >> LIVE_EMA_SHIFT
        if self.noise_floor < NOISE_FLOOR_MIN:
            self.noise_floor = NOISE_FLOOR_MIN
        elif self.noise_floor > NOISE_FLOOR_MAX:
            self.noise_floor = NOISE_FLOOR_MAX

    @property
    def finished(self):
        """True once the talker has been quiet for END_SILENCE_FRAMES."""
        return self.state == DONE

    @property
    def heard_speech(self):
        return self.state != IDLE or self._speech_units > 0

    def bounds(self):
        """(start, end) sample indices, from the shared paper algorithm."""
        return endpoints(None, energy=self._energy, zcr=self._zcr, n_frames=self.frames)
