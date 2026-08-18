#!/usr/bin/env python3
"""Checks the endpointer, and checks that both copies of it agree.

    python3 tools/test_vad.py

`src/vad.py` runs on the device and `tools/vad.py` runs on the host, and they
trim the same words: the host trims enrolment templates, the device trims the
utterance those templates are matched against. If the two ever disagree, every
DTW distance shifts for a reason neither file's own behaviour would reveal. So
the first test here is simply that they return the same answer on the same
audio, over signals designed to land on the awkward cases -- an onset in the
calibration window, a word running off the end of the buffer, pure silence.

The rest are behavioural: does it find a word, does it reject a click, does the
zero-crossing pass actually recover an unvoiced onset, does the playback gate
suppress everything.

Audio here is synthetic. It exercises the arithmetic, not the acoustics -- real
speech through the real microphone is `tools/speech_probe.py` and a person.
"""

import math
import os
import random
import sys
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vad as host_vad          # tools/vad.py  -- first on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# Import the device copy under its own name, without the host one shadowing it.
import importlib.util

_device_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "vad.py")
_spec = importlib.util.spec_from_file_location("device_vad", _device_path)
device_vad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(device_vad)

# Taken from the DSP tables rather than typed, so the synthetic fixtures below
# are always generated at the rate the rest of the system is built for. Written
# out, this would keep generating 16 kHz fixtures after a move to another rate
# and quietly test the wrong thing -- a test holding its own copy of a constant
# does not detect drift, it hides it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import speech_tables

RATE = speech_tables.SAMPLE_RATE
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def clamp(value):
    """int16 saturation. Real audio comes out of an ADC that cannot exceed it."""
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def noise(count, amplitude, rng):
    return [clamp(int(rng.gauss(0, amplitude))) for _ in range(count)]


def room(count, amplitude, rng):
    """Background rumble: white noise through a one-pole low pass.

    Crossing rate is the point. Flat white noise crosses zero on about half of
    all samples whatever its level, so a louder patch of it is not distinguishable
    from a quieter one by ZCR -- which makes it useless for exercising the
    zero-crossing pass. Real room noise is weighted towards low frequencies and
    real frication is not; this is that difference, minimally.
    """
    out = []
    low = 0.0
    for _ in range(count):
        low += (rng.gauss(0, amplitude) - low) * 0.15
        out.append(clamp(int(low * 3)))
    return out


def vowel(count, amplitude, rng, f0=180):
    return [
        clamp(
            int(amplitude * math.sin(2 * math.pi * f0 * i / RATE)
                * (0.6 + 0.4 * math.sin(2 * math.pi * 4 * i / RATE)))
            + int(rng.gauss(0, amplitude * 0.2))
        )
        for i in range(count)
    ]


def ms(count):
    return int(RATE * count / 1000)


def utterances(rng):
    """(name, samples) covering the cases the two implementations could differ on."""
    cases = []
    cases.append(("plain word",
                  noise(ms(200), 150, rng) + vowel(ms(500), 6000, rng) + noise(ms(800), 150, rng)))
    cases.append(("fricative onset",
                  noise(ms(200), 150, rng) + noise(ms(120), 400, rng)
                  + vowel(ms(400), 6000, rng) + noise(ms(600), 150, rng)))
    cases.append(("silence only", noise(ms(1500), 150, rng)))
    cases.append(("click", noise(ms(200), 150, rng) + vowel(ms(20), 20000, rng)
                  + noise(ms(800), 150, rng)))
    cases.append(("speech in the calibration window",
                  vowel(ms(400), 6000, rng) + noise(ms(900), 150, rng)))
    cases.append(("word running off the end",
                  noise(ms(200), 150, rng) + vowel(ms(1000), 6000, rng)))
    cases.append(("two words",
                  noise(ms(150), 150, rng) + vowel(ms(300), 6000, rng)
                  + noise(ms(250), 150, rng) + vowel(ms(300), 6000, rng)
                  + noise(ms(600), 150, rng)))
    cases.append(("very loud",
                  noise(ms(200), 150, rng) + vowel(ms(400), 30000, rng) + noise(ms(700), 150, rng)))
    cases.append(("dead channel", [0] * ms(1500)))
    return [(name, array("h", data)) for name, data in cases]


def test_agreement():
    print("host tools/vad.py vs device src/vad.py")
    rng = random.Random(20260817)
    for name, samples in utterances(rng):
        host = host_vad.endpoints(samples)
        device = device_vad.endpoints(samples)
        check("agree: %s" % name, host == device, "host=%s device=%s" % (host, device))

    # And on the frame statistics themselves, which is where a viper port would
    # go wrong first.
    for name, samples in utterances(rng):
        frames = len(samples) // host_vad.VAD_FRAME
        h_energy, h_zcr = host_vad.frame_stats(samples, frames)
        d_energy, d_zcr = device_vad.frame_stats(samples, frames)
        same = all(h_energy[f] == d_energy[f] and h_zcr[f] == d_zcr[f] for f in range(frames))
        check("frame stats: %s" % name, same)


def test_frame_pinning():
    """The VAD frame and the MFCC hop are equal, and that is not a coincidence.

    Endpoints are multiples of VAD_FRAME and MFCC frames advance by
    FRAME_STRIDE. While the two are equal, every boundary the endpointer picks
    lands exactly on an MFCC frame boundary and the segment the recogniser sees
    is the segment the VAD chose.

    If they diverge, nothing breaks loudly. Endpoints still compute, MFCCs still
    compute, templates still build -- but every template and every query picks
    up a different fractional offset into its first frame, and DTW distances
    degrade by a few percent with nothing to point at. It is indistinguishable
    from "the recogniser is just a bit poor", which is the single worst symptom
    to have to debug, so it is pinned here rather than left to the coincidence
    of three files having been written by people who happened to agree.
    """
    print("frame / hop pinning")
    check("device VAD frame == MFCC hop",
          device_vad.VAD_FRAME == speech_tables.FRAME_STRIDE,
          "VAD_FRAME=%d FRAME_STRIDE=%d" % (device_vad.VAD_FRAME, speech_tables.FRAME_STRIDE))
    check("host VAD frame == MFCC hop",
          host_vad.VAD_FRAME == speech_tables.FRAME_STRIDE,
          "VAD_FRAME=%d FRAME_STRIDE=%d" % (host_vad.VAD_FRAME, speech_tables.FRAME_STRIDE))
    check("VAD frame is 10 ms at the tables' own rate",
          device_vad.VAD_FRAME * 100 == speech_tables.SAMPLE_RATE,
          "%d samples at %d Hz" % (device_vad.VAD_FRAME, speech_tables.SAMPLE_RATE))
    # Not an equality, but worth knowing if it ever stops being true: the
    # analysis window is longer than the hop, so frames overlap.
    check("MFCC window is longer than the hop (frames overlap)",
          speech_tables.FRAME_LEN > speech_tables.FRAME_STRIDE,
          "len=%d stride=%d" % (speech_tables.FRAME_LEN, speech_tables.FRAME_STRIDE))


def test_behaviour():
    print("device endpointer behaviour")
    rng = random.Random(7)

    samples = array("h", noise(ms(200), 150, rng) + vowel(ms(500), 6000, rng) + noise(ms(800), 150, rng))
    span = device_vad.endpoints(samples)
    check("finds the word", span is not None)
    if span:
        start_ms = 1000 * span[0] // RATE
        end_ms = 1000 * span[1] // RATE
        check("start near 200 ms", 140 <= start_ms <= 230, "got %d ms" % start_ms)
        check("end near 700 ms", 670 <= end_ms <= 780, "got %d ms" % end_ms)

    click = array("h", noise(ms(200), 150, rng) + vowel(ms(20), 20000, rng) + noise(ms(800), 150, rng))
    check("rejects a click", device_vad.endpoints(click) is None)
    check("rejects silence",
          device_vad.endpoints(array("h", noise(ms(1500), 150, rng))) is None)

    # The zero-crossing pass, isolated: a quiet broadband onset in front of a
    # vowel, with the rule disabled for comparison.
    #
    # The pass can only fire in the energy band between its own gate and ITL: a
    # frame above ITL is already part of the energy run and needs no help. With
    # the gate at a flat 2x background that band was empty below about 34x
    # background -- 37 of 175 corpus utterances, all of them the quiet ones. The
    # gate is now four sigma above background instead, which lands near 1.2x and
    # keeps the band open. This checks the band is open where it should be, and
    # still asserts sane behaviour if it is not.
    for vowel_amplitude, onset_amplitude in ((30000, 450), (6000, 400)):
        # Low-passed background, white frication: the crossing-rate contrast
        # the pass exists to detect.
        audio = array("h", room(ms(200), 150, rng) + noise(ms(120), onset_amplitude, rng)
                      + vowel(ms(400), vowel_amplitude, rng) + room(ms(600), 150, rng))
        frames = len(audio) // device_vad.VAD_FRAME
        energy, zcr = device_vad.frame_stats(audio, frames)
        itl, itu, izct, imn, egate = device_vad.thresholds(energy, zcr, frames)
        live = itl > egate

        with_zcr = device_vad.endpoints(audio)
        saved = device_vad.ZCR_MIN_HITS
        device_vad.ZCR_MIN_HITS = 10 ** 6  # can never be satisfied
        without = device_vad.endpoints(audio)
        device_vad.ZCR_MIN_HITS = saved

        label = "vowel %d, onset %d (imx/imn=%d)" % (
            vowel_amplitude, onset_amplitude, max(energy) // max(1, imn))
        onset_at = ms(200)
        missed_by_energy = without is not None and without[0] > onset_at

        if live and missed_by_energy:
            check("zcr pass recovers the onset energy alone missed: %s" % label,
                  with_zcr is not None and with_zcr[0] < without[0],
                  "with=%s without=%s" % (with_zcr, without))
        elif live:
            # The energy run already reached the onset, so there is nothing for
            # the pass to add. It must not move the boundary anyway.
            check("zcr pass leaves an already-correct boundary: %s" % label,
                  with_zcr == without, "with=%s without=%s" % (with_zcr, without))
        else:
            check("zcr pass inert as expected: %s" % label, with_zcr == without,
                  "with=%s without=%s" % (with_zcr, without))
            print("       (gate %d is above ITL %d, so the energy run already"
                  " covers everything the pass could add)" % (egate, itl))


def test_live():
    print("live layer (device only)")
    rng = random.Random(11)
    samples = array("h", noise(ms(200), 150, rng) + vowel(ms(500), 6000, rng) + noise(ms(900), 150, rng))
    total = len(samples)

    detector = device_vad.Endpointer(max_frames=total // device_vad.VAD_FRAME + 1)
    available = 0
    finished_at = None
    while available < total:
        available = min(total, available + 320)  # 20 ms of DMA at a time
        detector.feed(samples, available)
        if detector.finished and finished_at is None:
            finished_at = available
    check("stops after the talker does", finished_at is not None)
    if finished_at:
        stopped_ms = 1000 * finished_at // RATE
        # Speech ends at 700 ms; the hangover is LIVE_HANGOVER_UNITS of 20 ms,
        # and the start needs LIVE_START_UNITS above threshold before the
        # hangover can even begin counting.
        want = 700 + 20 * device_vad.LIVE_HANGOVER_UNITS
        check("stops ~%d ms in" % want, abs(stopped_ms - want) <= 80, "got %d ms" % stopped_ms)

    check("live bounds match the offline algorithm",
          detector.bounds() == device_vad.endpoints(samples, detector.consumed),
          "%s vs %s" % (detector.bounds(), device_vad.endpoints(samples, detector.consumed)))

    gated = device_vad.Endpointer(max_frames=total // device_vad.VAD_FRAME + 1)
    gated.set_playing(True)
    gated.feed(samples, total)
    check("playback gate drops every frame", gated.frames == 0 and gated.bounds() is None)
    check("playback gate leaves the live layer untriggered",
          gated.state == device_vad.IDLE and gated.units == 0)

    # The start trigger wants LIVE_START_UNITS consecutive 20 ms units above
    # threshold. One loud unit -- a knuckle on the panel -- must not arm it.
    clicky = array("h", room(ms(400), 150, rng) + vowel(ms(20), 30000, rng)
                   + room(ms(900), 150, rng))
    clicker = device_vad.Endpointer(max_frames=len(clicky) // device_vad.VAD_FRAME + 1)
    clicker.feed(clicky, len(clicky))
    check("a single loud unit does not arm the live layer",
          clicker.state == device_vad.IDLE and not clicker.finished,
          "state=%d" % clicker.state)

    # The EMA floor should land near the actual energy of the room it heard,
    # and the thresholds should straddle it.
    quiet = array("h", room(ms(1200), 150, rng))
    calibrator = device_vad.Endpointer(max_frames=len(quiet) // device_vad.VAD_FRAME + 1)
    calibrator.feed(quiet, len(quiet))
    energy, _zcr = device_vad.frame_stats(quiet, len(quiet) // device_vad.VAD_FRAME)
    actual = (sum(energy) // len(energy)) * device_vad.LIVE_FRAMES
    check("noise floor tracks the room within 2x",
          actual // 2 <= calibrator.noise_floor <= actual * 2,
          "floor=%d room=%d" % (calibrator.noise_floor, actual))
    check("quiet room never triggers", not calibrator.finished
          and calibrator.state == device_vad.IDLE)

    ungated = device_vad.Endpointer(max_frames=total // device_vad.VAD_FRAME + 1)
    ungated.set_playing(True)
    ungated.feed(samples, ms(300))
    ungated.set_playing(False)
    ungated.feed(samples, total)
    check("gate releases cleanly", ungated.frames > 0)


def main():
    test_agreement()
    test_frame_pinning()
    test_behaviour()
    test_live()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
