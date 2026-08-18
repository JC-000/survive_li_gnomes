"""Push-to-talk ELIZA. The alternate program to the Magic 8-Ball.

Hold the screen (or the POWER key), say something, let go. The board endpoints
what it heard, spots a keyword from a small vocabulary, runs it through a real
DOCTOR rule set and puts the reply on the e-paper.

Deployed *as* `main.py` -- `./tools/deploy.sh eliza` -- because main.py is what
autoruns at power-on. The Magic 8-Ball's `src/main.py` is untouched and the two
programs share `board`, `epaper`, `magic8` (for its word wrapper), `es8311` and
`audio_pio_mpy`.

It inherits the 8-Ball's power rules, which are not optional here either:

- The panel is never touched at boot. E-paper holds the last reply through a
  power cut, and redrawing it would cost a flashing full refresh for an
  identical image.
- Nothing is ever written to flash. No transcript, no last reply, no learned
  templates. The display is the storage, and a flash write interrupted by a
  power cut is the one thing that could corrupt the filesystem.

And one that is new: **the codec is configured for 16 kHz capture**, so the
8-Ball's clips (24 kHz packed stereo) are not played here. They would come out
half an octave low and, more to the point, the microphone would hear them --
see the playback gate in `vad`.

The keyword spotter itself is not here yet: `_spot_keyword` calls into a
`spotter` module that the MFCC/DTW work will provide (docs/speech-design.md).
Until it exists every turn takes DOCTOR's no-keyword path, which is a real ELIZA
behaviour rather than an error state, so the rest of the loop -- capture,
endpointing, the engine, the panel -- runs and can be judged.
"""

import time

import board
import epaper
import listen
import screen
import vad

# Both of these are other people's modules and both are imported defensively:
# the program has to be runnable on the board while the rest is still being
# written, and a missing module must degrade rather than crash.
try:
    import eliza
    import vocab
except ImportError as _exc:
    print("eliza/vocab not deployed (%s); replies will be placeholders" % _exc)
    eliza = None
    vocab = None

# The on-device keyword spotter: MFCC front end, banded DTW, rejection gate.
# Still imported defensively, because a board without it deployed should degrade
# to DOCTOR's no-keyword path rather than fail to start.
try:
    import spotter
except ImportError:
    spotter = None

# The enrolled word templates. A separate module from the spotter because it is
# a 137 KB buffer with a loader, not code -- see reserve_templates() for why
# main() allocates it rather than letting the import do it.
try:
    import templates
except ImportError:
    templates = None

POLL_MS = 50

# While recording, poll fast enough to hand the endpointer whole 10 ms frames
# soon after the DMA writes them. This is the only tight loop in the program.
RECORD_POLL_MS = 10

# Inherited from main.py at 8, and almost certainly wrong here. The 8-Ball swaps
# one short line of large text; an ELIZA reply repaints the entire panel with
# small text every turn, and ghosting scales with how much of the screen
# changed. Expect this to need lowering -- but pick the number by looking at the
# glass after a run of partials, not by guessing one here.
PARTIALS_BEFORE_FULL = 8

# Partial refresh diffs against a base image in the controller's RAM, and deep
# sleep can only be left via a hardware reset, which drops it. So the panel
# stays powered between turns and sleeps only after this long idle; the next
# turn pays for a full refresh, which also scrubs ghosting.
PANEL_IDLE_SLEEP_MS = 60_000

# A stuck touch line would otherwise re-record every 3 s forever.
STUCK_MS = 30_000

# Shown when the endpointer found nothing worth calling speech. Deliberately not
# routed through ELIZA: DOCTOR has no rule for "you did not say anything", and
# inventing one would make a hardware problem look like a conversation.
NOTHING_HEARD = "I did not hear anything. Hold the screen and speak."
NO_MICROPHONE = "The microphone is not responding."


class Inputs:
    """Press-and-hold, over the same inputs the 8-Ball uses.

    Deliberately a copy of `main.Inputs` rather than an import: this file is
    deployed *as* main.py, so importing `main` would import itself. It differs
    anyway -- the 8-Ball wants one edge per press, this wants to know how long
    you are holding.
    """

    def __init__(self, i2c):
        self.i2c = i2c
        self.power = board.Pin(board.POWER_KEY_PIN, board.Pin.IN, board.Pin.PULL_UP)

        # Nothing on the board holds the touch controller's reset line high --
        # an internal pull-down wins on GP16 -- so drive it explicitly.
        self.touch_rst = board.Pin(board.TOUCH_RST_PIN, board.Pin.OUT, value=1)

        try:
            import rp2

            self._bootsel = rp2.bootsel_button
        except (ImportError, AttributeError):
            self._bootsel = None

    def _touched(self):
        """Read the FT6336U's touch-count register.

        Deliberately *not* the INT pin. This board ships with reg 0xA4 = 0x01,
        FocalTech's "trigger" mode, where INT emits a brief pulse per touch
        frame rather than sitting low while held -- which is exactly the wrong
        shape for press-and-hold and is why an earlier version of the 8-Ball
        "worked once and then died". TD_STATUS is a level, and reading it also
        clears the controller's pending interrupt.

        Swallows I2C errors: a bus glitch must not kill an unattended loop.
        """
        try:
            return bool(self.i2c.readfrom_mem(board.ADDR_TOUCH, 0x02, 1)[0] & 0x0F)
        except OSError:
            return False

    def down(self):
        if not self.power.value():  # active low
            return True
        if self._touched():
            return True
        # Checked last: rp2.bootsel_button() momentarily disables interrupts and
        # takes over the QSPI CS line, so it is far more expensive than a GPIO
        # read. Never poll it alongside _thread.
        if self._bootsel and self._bootsel():
            return True
        return False

    def wait_for_release(self, timeout_ms=5000):
        """Block until released. Returns False if it timed out still held."""
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        while self.down():
            if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                return False
            time.sleep_ms(POLL_MS)
        return True


class Panel:
    """The e-paper, plus the partial-vs-full refresh policy.

    Same policy as main.Panel and for the same reasons (see docs/design.md);
    what differs is what gets drawn and how often a full refresh is needed --
    see PARTIALS_BEFORE_FULL. Nothing here touches the panel until the first
    show(), which is what keeps a power blip free.
    """

    def __init__(self):
        self.epd = None
        self._base_valid = False
        self._partials = 0
        self._last_use = 0

    def show(self, reply, footer=None):
        """Draw a reply. Returns "full" or "partial" for logging."""
        first = self.epd is None
        if first:
            self.epd = epaper.EPD_1in54()  # constructor full-inits and powers on

        screen.render(self.epd, reply, footer=footer)

        if self._base_valid and self._partials < PARTIALS_BEFORE_FULL:
            self.epd.displayPartial(self.epd.buffer)
            self._partials += 1
            mode = "partial"
        else:
            # Reload the full waveform before writing a base image: after a run
            # of partials the partial LUT is loaded and it will not scrub
            # ghosting. init() also restores power if we were asleep.
            if not first:
                self.epd.init(self.epd.full_update)
            self.epd.displayPartBaseImage(self.epd.buffer)
            self.epd.init(self.epd.part_update)
            self._base_valid = True
            self._partials = 0
            mode = "full"

        self._last_use = time.ticks_ms()
        return mode

    def maybe_sleep(self):
        """Deep-sleep the panel once it has been idle a while.

        Costs the next turn a full refresh, since sleeping drops the base image.
        """
        if not self._base_valid:
            return False
        if time.ticks_diff(time.ticks_ms(), self._last_use) < PANEL_IDLE_SLEEP_MS:
            return False
        self.epd.sleep()
        self._base_valid = False
        return True


def _spot_keyword(samples, start, end):
    """Which vocabulary word was said, or None.

    `samples` is the raw int16 capture buffer -- untrimmed, because the spotter
    wants the endpoints, not a copy -- and `start`/`end` are sample indices from
    `vad.endpoints`, the same boundaries the host used when it trimmed the
    enrolment templates.

    The label it returns must be an exact `vocab` form. Matching in the rule set
    is case-sensitive against literal word tuples, so an inflection the spotter
    invents ("loves", "mothers") matches nothing and produces a deflection that
    looks like a recognition failure rather than a vocabulary one. Normalise
    here or add the form to its class in `vocab.VOCAB`, which costs nothing
    because the class already exists.

    Returns None while the spotter is absent or has no templates bound, which
    makes the turn take DOCTOR's no-keyword path. That is a real ELIZA
    behaviour, not an error state -- and it is also what a deliberate rejection
    looks like, since the spotter is tuned for precision over recall.
    """
    if spotter is None:
        return None
    try:
        # spot_scored returns (label, best, runner_up) and spot is the same
        # thing with the scores dropped. Preferred here purely so the scores
        # reach the log: the two rejection gates are tuned numbers that have to
        # be re-measured against templates enrolled through this board's own
        # ES8311, and a device that only ever prints the verdict gives nobody
        # anything to tune with.
        scored = getattr(spotter, "spot_scored", None)
        if scored is not None:
            label, best, runner_up = scored(samples, start, end)
            print("   spot: %s (best %s, runner-up %s)"
                  % (label or "-", best, runner_up))
            return label
        return spotter.spot(samples, start, end)
    except Exception as exc:  # noqa: BLE001
        print("spotter failed (%s: %s)" % (type(exc).__name__, exc))
        return None


class Conversation:
    """One ELIZA session, and the only place that knows the engine's shape.

    DOCTOR is stateful -- it rotates through each rule's replies and queues
    things to bring up later -- so the session outlives a turn and is built once.

    The spotter feeds `respond_to_keywords`, not `respond`: what comes back from
    a keyword spotter is an unordered bag of recognised words, with everything
    outside the vocabulary simply absent. eliza.py has a whole degraded path for
    exactly that, and handing it a fake sentence instead would be worse.
    """

    def __init__(self):
        self.doctor = None
        self.nouns = None
        self.turns = 0

    def _ensure(self):
        if self.doctor is None and eliza is not None:
            # Both of these come from vocab rather than from the engine's
            # defaults, and both matter more than they look.
            #
            # NOUNS is the list of words worth echoing back. The script's own
            # fallback list is the 1966 family set: it knows MOTHER but not
            # WORK, MONEY, SLEEP, DEATH or LOVE, so five of our twelve nouns
            # would silently never be echoed -- the toy would just quietly go
            # bland. It is upper case because respond_to_keywords upper-cases
            # the bag and compares against it; passing labels matches nothing,
            # equally silently.
            #
            # PRIORITY ranks the keywords the *engine* supplies rather than the
            # ones we spotted -- a bag of {MOTHER} only reaches the family rules
            # because the engine adds the "MY" no recogniser can catch. MY above
            # I is load-bearing: rank the feeling rules first and "my brother is
            # sick" and "my children are sick" both come back as "I am sorry to
            # hear you are sick", which tells the user they are ill when they
            # said their brother was.
            priority = getattr(vocab, "PRIORITY", None) if vocab else None
            self.doctor = eliza.Doctor(priority=priority)
            self.nouns = getattr(vocab, "NOUNS", None) if vocab else None
        return self.doctor is not None

    def _present(self, text):
        """Script text as it should appear on glass.

        The rule set is all upper case because DOCTOR was written for a
        teletype, and eliza.py deliberately leaves that as presentation for the
        caller to undo. A 200x200 panel is the caller: nine lines of shouting is
        markedly harder to read than a sentence, and the echoed noun still lands
        in the middle of it either way.
        """
        try:
            return eliza.sentence_case(text)
        except Exception:  # noqa: BLE001 -- cosmetic; never worth losing a reply
            return text

    def greeting(self):
        if not self._ensure():
            return "STUB: no eliza module deployed."
        try:
            return self._present(self.doctor.greet())
        except Exception as exc:  # noqa: BLE001
            print("eliza greet failed (%s: %s)" % (type(exc).__name__, exc))
            return "Please go on."

    def reply(self, label):
        """Ask DOCTOR. Never raises -- a crash here is a blank screen."""
        if not self._ensure():
            # The engine is not on the device. Say so plainly rather than faking
            # a therapist: a placeholder that reads like a real reply is how a
            # missing module ships to a user.
            return "STUB: no eliza module deployed. Heard: %s" % (label or "nothing")
        spotted = []
        if label:
            spotted.append(vocab.ECHO.get(label, label.upper()) if vocab else label.upper())
        try:
            self.turns += 1
            return self._present(self.doctor.respond_to_keywords(spotted, nouns=self.nouns))
        except Exception as exc:  # noqa: BLE001
            print("eliza failed (%s: %s)" % (type(exc).__name__, exc))
            return "Please go on."


def listen_once(inputs, recorder, detector, i2c):
    """Record for as long as the input is held. Returns (start, end) or None.

    Stops early when the endpointer decides the talker has finished, so a short
    answer is not padded out to the full 3 s cap, and stops immediately on
    release.
    """
    detector.reset()
    if not recorder.start(i2c):
        return None

    while inputs.down() and not recorder.full() and not detector.finished:
        detector.feed(recorder.buf, recorder.captured())
        time.sleep_ms(RECORD_POLL_MS)

    count = recorder.stop()
    # Anything the DMA wrote after the last poll is still unanalysed.
    detector.feed(recorder.buf, count)
    return detector.bounds()


def reserve_templates():
    """Allocate, load and expand the template buffer. Second, after the capture.

    MicroPython's heap never compacts, so the free *total* can be comfortable
    while no single block that large remains. `sounds.allocate_bytes` records
    what taught this project that: 419 KB free, largest block 174 KB, and a
    140 KB clip that still would not allocate.

    **The capture buffer is reserved first, and there are two reasons.** The
    durable one first, because it survives the numbers changing:

    Capture is what the program cannot run without. Templates degrade to
    DOCTOR's no-keyword path, which is a real ELIZA behaviour rather than a
    failure. So the allocation most likely to succeed should be the one that has
    to succeed, whatever the two sizes happen to be.

    The arithmetic agrees today, via a rule worth knowing: **order by transient
    peak, not by final size.** "Largest first" is a shorthand and this is where
    the shorthand and the thing it stands for come apart. Templates are the
    larger *resident* block (137 KB against 94 KB) but have no transient at all
    -- `bytearray(n)` is one allocation and the expansion works inside it through
    1.4 KB of scratch. The capture buffer is `array("h", bytearray(2 * n))`,
    which holds the bytearray *and* the array at once, so it briefly needs
    188 KB to end up with 94:

        templates first:  peak 332 064,  resident 236 064
        capture first:    peak 237 528,  resident 236 064

    Capture-first is 92.3 KB cheaper at the peak, because its 188 KB transient
    happens while nothing else is held and so costs nothing.

    That transient is structural, not a choice. `dma_record_into` sets
    `count=len(buf)` with 16-bit transfers, so `len()` must be a count of
    samples: a bare `bytearray` would set a count of bytes and run the DMA past
    the end of the buffer, and MicroPython has no `memoryview.cast` to bridge
    it. It could be removed by giving `dma_record_into` an explicit sample count
    -- but that edits `audio_pio_mpy.py`, which the working Magic 8-Ball shares,
    for no benefit, because capture-first already makes the spike free. Possible,
    not worth it, leave it alone.

    Returns the buffer, or None when there is nothing to load or nothing to load
    it with -- in which case every turn takes DOCTOR's no-keyword path, which is
    a real ELIZA behaviour rather than an error state.
    """
    if templates is None:
        return None

    # Held by this module for the life of the program, so it must be handed the
    # expansion the packing calls for. getattr rather than a direct reference:
    # the spotter is a parallel piece of work and may not be on the device.
    packed = getattr(templates, "PACKED", "full")
    expand = getattr(spotter, "expand", None) if spotter is not None else None
    if packed == "statics" and expand is None:
        print("templates are packed as statics but no spotter.expand is here; "
              "not loading them (build with --pack full to load without it)")
        return None

    try:
        buf = bytearray(templates.BUFFER_BYTES)
        templates.load(buf, expand)
        # The spotter does not import `templates` and load them itself: this is
        # the largest allocation in the program and the order it is made in is
        # load-bearing, so main() owns it and hands it over.
        spotter.bind(buf, templates.INDEX)
        print("templates: %d bytes, %d frames, packed as %s"
              % (templates.BUFFER_BYTES, templates.TOTAL_FRAMES, packed))
        return buf
    except MemoryError:
        # Worth its own branch: this is the failure the ordering exists to
        # prevent, and if it happens here it happened at the best possible
        # moment -- before the capture buffer, with the most heap available.
        print("no room for %d bytes of templates; running without a spotter"
              % templates.BUFFER_BYTES)
        return None
    except Exception as exc:  # noqa: BLE001
        print("templates unavailable (%s: %s)" % (type(exc).__name__, exc))
        return None


def reserve():
    """Take the two big blocks, in the order that matters. Returns both.

    Split out of main() so the order can be *asserted* rather than described --
    `tools/test_talk.py` instruments both allocations and checks which happens
    first. It is 92 KB of peak heap (see reserve_templates) and it is precisely
    the kind of thing a later tidy-up reorders innocently, so a comment saying
    "not stylistic" is weaker than a test that fails.

    Both return values must be held for the life of the program. `template_buf`
    especially: it is the only strong reference to 137 KB that the spotter reads
    through, `templates.load()` keeps none, and main() never returning is what
    keeps it alive. Dropping the name frees the templates at some unrelated
    later collection, which is a slow and baffling failure.
    """
    recorder = listen.Recorder()                # 94 KB, but 188 KB to build
    template_buf = reserve_templates()          # 137 KB, no transient
    return recorder, template_buf


def main():
    i2c = board.bus()

    recorder, template_buf = reserve()          # noqa: F841 -- see reserve()
    detector = vad.Endpointer(max_frames=listen.MAX_SAMPLES // vad.VAD_FRAME + 1)

    inputs = Inputs(i2c)
    battery = board.Battery()
    panel = Panel()
    session = Conversation()

    reported_no_mic = False
    print("eliza ready -- hold the screen or POWER, speak, let go")

    while True:
        if not inputs.down():
            panel.maybe_sleep()
            time.sleep_ms(POLL_MS)
            continue

        started = time.ticks_ms()
        bounds = listen_once(inputs, recorder, detector, i2c)

        if recorder.available is False:
            if not reported_no_mic:
                panel.show(NO_MICROPHONE, footer="%.2fV" % battery.volts())
                reported_no_mic = True
            # Nothing to listen with; do not spin re-trying it every 50 ms.
            inputs.wait_for_release()
            time.sleep_ms(1000)
            continue

        if bounds is None:
            heard = None
            # First turn with nothing heard: greet, rather than complain. DOCTOR
            # opens the conversation, and a device that says "I did not hear
            # anything" before it has ever said hello reads as broken.
            reply = session.greeting() if session.turns == 0 else NOTHING_HEARD
            print("-> nothing heard (%d frames, noise floor %d, start %d)"
                  % (detector.frames, detector.noise_floor, detector.live_start))
        else:
            start, end = bounds
            heard = _spot_keyword(recorder.buf, start, end)
            reply = session.reply(heard)
            print(
                "-> speech %d..%d (%d ms), heard %s"
                % (start, end, (end - start) * 1000 // listen.SAMPLE_RATE, heard or "-")
            )

        footer = "%s  %.2fV" % (heard or "?", battery.volts())
        mode = panel.show(reply, footer=footer)
        print("   %s [%s, %d ms]" % (reply, mode, time.ticks_diff(time.ticks_ms(), started)))

        if not inputs.wait_for_release(STUCK_MS):
            # Held for 30 s. A human cannot meaningfully do that, so it is a
            # stuck line; back off rather than recording 3 s clips forever.
            print("warning: input still down after %d s, backing off" % (STUCK_MS // 1000))
            time.sleep_ms(1000)


if __name__ == "__main__":
    main()
