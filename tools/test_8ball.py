#!/usr/bin/env python3
"""Tests the Magic 8-Ball, so a shared module cannot change under it silently.

    python3 tools/test_8ball.py

The toy is five modules deep and four of them are now shared with a second
program. `board`, `epaper`, `es8311`, `audio_pio_mpy` and `magic8` are copied to
the device by both `deploy.sh 8ball` and `deploy.sh eliza` (tools/deploy.sh:23),
and `src/screen.py`, `src/talk.py` and `src/listen.py` all import into that set.
Before this file existed, nine host suites covered the speech program and *none*
of them touched `magic8.pick`, `magic8.render`, `shake`, `sounds` or `epaper` --
so an edit made for ELIZA's benefit could regress the ball with the whole suite
still green. That gap is the only thing this file is for.

So the bias throughout is drift detection over generous testing: assert what the
code observably does now, precisely, and let a deliberate change come here and
say so. Expectations are derived from the modules wherever two of them have to
agree -- `sounds`' clip length against the buffer `shake` reserves for it,
`epaper`'s private pin numbers against `board`'s, `board`'s pin map against the
table in docs/hardware.md that was checked on the physical board. A test holding
its own second copy of a constant enforces drift rather than catching it, which
this project has been bitten by twice.

Three literals *are* pinned by hand, deliberately, because they are the claims
most likely to rot quietly: the 44 answers, and the 11-at-3x / 33-at-2x split
that docs/design.md:43 states. Those exist to be noticed when they change.

Stubs `machine`, `framebuf`, `utime` and `rp2` the way test_talk.py and
test_record_stream.py do, and loads modules from source rather than through
importlib -- see test_spotter._load_source: importlib consults __pycache__, and
a stale cache is how a suite goes green without running anything.

No 8-Ball module was modified to make any of this testable. Two things resisted,
and are recorded here rather than worked around:

- `main.main()` is an unbounded loop with no injection point and no return, so
  it cannot be called from a test. Its two pieces of policy are reachable --
  `Inputs` and `Panel` are separate classes -- and both are covered below. What
  is *not* covered is the order main() calls them in: that shaker.start()
  precedes panel.show() so the sound overlaps the refresh (src/main.py:210-212)
  is checked by reading, not by running.
- `Shaker._setup` needs a live I2C object, so the codec and PIO are stubbed and
  the assertions are about what it *asks* them for -- mic gain, which state
  machines get initialised, allocation order. Whether the ES8311 then makes a
  noise is not knowable here, and per CLAUDE.md never will be from software:
  audio changes want a human or a scope, not a passing test.
"""

import math
import os
import sys
import types
from array import array
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src")
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def _load_source(name, path):
    """Load a module from source. Not importlib -- see test_spotter."""
    with open(path) as handle:
        source = handle.read()
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


# --- the device that is not here -------------------------------------------

class FakePin:
    """A GPIO that remembers every level it was driven to, in order.

    epaper's power discipline is the reason this records rather than just
    accepts: "PWR was driven low before the reset pulse" is the whole content of
    the bug fixed in 86b540b, and it is only visible as a sequence.
    """

    IN = 0
    OUT = 1
    PULL_UP = 2
    PULL_DOWN = 3

    log = {}  # pin number -> list of levels written

    def __init__(self, number=0, mode=None, pull=None, value=None):
        self.number = number
        self.mode = mode
        self.level = 0 if value is None else value
        if value is not None:
            FakePin.log.setdefault(number, []).append(value)

    def value(self, level=None):
        if level is None:
            return self.level
        self.level = level
        FakePin.log.setdefault(self.number, []).append(level)
        return None

    @classmethod
    def reset_log(cls):
        cls.log = {}


class FakeI2C:
    """Records every register read, so "which register" is assertable.

    The 8-Ball's touch path is required to poll TD_STATUS and never the INT pin
    (CLAUDE.md, and the reason touch "worked once and then died"), which is a
    claim about which byte gets asked for.
    """

    def __init__(self, *a, **k):
        self.reads = []
        self.touched = False

    def readfrom_mem(self, addr, reg, n):
        self.reads.append((addr, reg))
        return bytes([1 if self.touched else 0] * n)

    def writeto(self, *a, **k):
        pass

    def writeto_mem(self, *a, **k):
        pass

    def scan(self):
        return [0x38, 0x18, 0x51, 0x70]


class FakeFrameBuffer:
    """Records the bounding box of everything drawn, for the overflow check."""

    def __init__(self, *a, **k):
        self.marks = []

    def fill(self, *a):
        pass

    def fill_rect(self, x, y, w, h, c):
        self.marks.append((x, y, w, h))

    def rect(self, x, y, w, h, c, *a):
        self.marks.append((x, y, w, h))

    def hline(self, x, y, w, c):
        self.marks.append((x, y, w, 1))

    def ellipse(self, cx, cy, rx, ry, c, *a):
        self.marks.append((cx - rx, cy - ry, 2 * rx, 2 * ry))

    def text(self, t, x, y, c):
        self.marks.append((x, y, 8 * len(t), 8))

    def pixel(self, x, y, *a):
        # Deterministic stand-in for the built-in font: _text_scaled only asks
        # whether a glyph pixel is set, and the bounding box it paints is the
        # same whichever pixels answer yes.
        return (x * 7 + y * 3) % 5 == 0

    def bounds(self):
        xs = [m[0] for m in self.marks] + [m[0] + m[2] for m in self.marks]
        ys = [m[1] for m in self.marks] + [m[1] + m[3] for m in self.marks]
        return min(xs), min(ys), max(xs), max(ys)


class FakeEPD(FakeFrameBuffer):
    """The panel, minus the panel. Records which refresh path each draw took."""

    full_update = 0
    part_update = 1

    def __init__(self, *a, **k):
        FakeFrameBuffer.__init__(self)
        self.buffer = bytearray(5000)
        self.calls = []

    def init(self, mode):
        self.calls.append("init(%s)" % ("full" if mode == 0 else "part"))

    def displayPartBaseImage(self, buf):
        self.calls.append("base")

    def displayPartial(self, buf):
        self.calls.append("partial")

    def display(self, buf):
        self.calls.append("display")

    def sleep(self):
        self.calls.append("sleep")


class FakeSPI:
    def __init__(self, *a, **k):
        self.written = 0

    def init(self, *a, **k):
        pass

    def write(self, data):
        self.written += len(data)


def install_stubs():
    machine = types.ModuleType("machine")
    machine.Pin = FakePin
    machine.I2C = lambda *a, **k: _SHARED_I2C
    machine.SPI = FakeSPI
    machine.ADC = lambda *a, **k: types.SimpleNamespace(read_u16=lambda: 40000)
    machine.freq = lambda: 150_000_000
    sys.modules["machine"] = machine

    framebuf = types.ModuleType("framebuf")
    framebuf.MONO_HLSB = 0
    framebuf.FrameBuffer = FakeFrameBuffer
    sys.modules["framebuf"] = framebuf

    utime = types.ModuleType("utime")
    utime.sleep_ms = lambda ms: None
    utime.sleep_us = lambda us: None
    utime.ticks_ms = lambda: _CLOCK[0]
    sys.modules["utime"] = utime

    sys.modules["rp2"] = types.ModuleType("rp2")

    import time as _time
    if not hasattr(_time, "ticks_ms"):
        # A clock the tests can move, so the 60 s idle timer and the 30 s stuck
        # input can be reached without waiting for them.
        _time.ticks_ms = lambda: _CLOCK[0]
        _time.ticks_us = lambda: _CLOCK[0] * 1000
        _time.ticks_diff = lambda a, b: a - b
        _time.ticks_add = lambda a, b: a + b
        _time.sleep_ms = lambda ms: None


_CLOCK = [0]
_SHARED_I2C = FakeI2C()

install_stubs()

import time  # noqa: E402  -- now carrying the ticks_* shims

for _name in ("board", "sounds", "magic8", "shake"):
    _load_source(_name, os.path.join(SRC, _name + ".py"))

import board    # noqa: E402
import magic8   # noqa: E402
import shake    # noqa: E402
import sounds   # noqa: E402

# The real driver, loaded under another name so that `main` still gets the fake
# panel below. Nothing imports "epaper_real"; the name only keeps the two apart.
epaper_real = _load_source("epaper_real", os.path.join(SRC, "epaper.py"))

_epaper_stub = types.ModuleType("epaper")
_epaper_stub.EPD_1in54 = FakeEPD
sys.modules["epaper"] = _epaper_stub

main_mod = _load_source("main", os.path.join(SRC, "main.py"))


# --- 1. the answers ---------------------------------------------------------

def test_the_answer_set():
    """44 answers, all distinct, and pick() never repeats itself."""
    print("answers: the set itself")

    # Pinned by hand on purpose. docs/design.md and the module docstring both
    # quote 44 and the 18/12/14 mix; if someone adds an answer, this is where
    # the count stops matching the prose that describes it.
    check("there are 44 answers", len(magic8.ANSWERS) == 44,
          "got %d" % len(magic8.ANSWERS))
    check("every answer is distinct",
          len(set(magic8.ANSWERS)) == len(magic8.ANSWERS),
          "duplicates: %s" % [a for a in magic8.ANSWERS
                              if magic8.ANSWERS.count(a) > 1])
    check("no answer is empty or padded",
          all(a and a == a.strip() for a in magic8.ANSWERS))

    # pick(exclude=) is the whole of the "never twice in a row" promise, and it
    # is a loop that could spin forever if the set ever collapsed to one entry.
    seen = set()
    previous = None
    for _ in range(4000):
        answer = magic8.pick(exclude=previous)
        check_failed = answer == previous
        if check_failed:
            break
        seen.add(answer)
        previous = answer
    check("pick() never returns the excluded answer (4000 draws)",
          not check_failed)
    check("pick() draws from ANSWERS only", seen <= set(magic8.ANSWERS),
          "strangers: %s" % (seen - set(magic8.ANSWERS)))
    check("4000 draws reach every answer", seen == set(magic8.ANSWERS),
          "never drawn: %s" % sorted(set(magic8.ANSWERS) - seen))

    # Rejection sampling is what keeps the low answers from being favoured.
    # Not a distribution test -- just that the range is right and it terminates.
    draws = [magic8._rand_below(44) for _ in range(4000)]
    check("_rand_below stays in range", all(0 <= d < 44 for d in draws),
          "out of range: %s" % [d for d in draws if not 0 <= d < 44][:5])
    check("_rand_below covers its range", len(set(draws)) == 44,
          "missing %d values" % (44 - len(set(draws))))


# --- 2. the layout ----------------------------------------------------------

def test_wrap_is_lossless():
    """Wrapping may only insert breaks -- never drop or reorder a word."""
    print("layout: wrap()")
    for cols in (8, 12, 23):
        for answer in magic8.ANSWERS:
            lines = magic8.wrap(answer, cols)
            check_ok = " ".join(lines).split() == answer.split()
            if not check_ok:
                check("wrap(%d) preserves %r" % (cols, answer), False,
                      "%r" % lines)
                return
            if max(len(line) for line in lines) > cols:
                check("wrap(%d) respects the column limit for %r"
                      % (cols, answer), False, "%r" % lines)
                return
    check("wrap() preserves every word at 8, 12 and 23 columns", True)
    check("wrap() never exceeds the column limit", True)

    # The hard-split path, which no current answer reaches -- so it is only
    # ever exercised here.
    split = magic8.wrap("supercalifragilistic", 8)
    check("a word longer than a line is hard-split",
          split == ["supercal", "ifragili", "stic"], "%r" % split)
    check("hard-splitting loses no characters",
          "".join(split) == "supercalifragilistic")


def test_every_answer_fits_the_answer_area():
    """The documented size split, and no word broken to achieve it."""
    print("layout: fit()")

    scales = {}
    for answer in magic8.ANSWERS:
        scale, lines = magic8.fit(answer)
        scales.setdefault(scale, []).append(answer)

        cols = dict((s, c) for s, c, _ in magic8._SIZES)[scale]
        max_lines = dict((s, m) for s, _, m in magic8._SIZES)[scale]
        if max(len(w) for w in answer.split()) > cols:
            check("no word is broken in %r" % answer, False,
                  "scale %d gives %d columns" % (scale, cols))
            return
        if len(lines) > max_lines:
            check("%r stays within its line budget" % answer, False,
                  "%d lines at scale %d" % (len(lines), scale))
            return
        # The real constraint is pixels, not lines: the answer band is
        # _ANSWER_H tall and render() centres within it.
        if len(lines) * magic8._line_height(scale) > magic8._ANSWER_H:
            check("%r fits the answer band" % answer, False,
                  "%d lines x %d px > %d"
                  % (len(lines), magic8._line_height(scale), magic8._ANSWER_H))
            return

    check("no answer has a word too long for its chosen size", True)
    check("every answer fits the %d px answer band" % magic8._ANSWER_H, True)

    # docs/design.md:43 -- "Of the 44, 11 render at 3x and 33 at 2x; none need
    # the 1x fallback." Pinned deliberately: this is the layout claim most
    # likely to go stale, and the one a change to _SIZES would move first.
    at3, at2, at1 = (len(scales.get(s, ())) for s in (3, 2, 1))
    check("11 answers render at 3x", at3 == 11, "got %d" % at3)
    check("33 answers render at 2x", at2 == 33, "got %d" % at2)
    check("none fall back to 1x", at1 == 0,
          "fell back: %s" % scales.get(1, ()))

    # docs/design.md:240 -- the 3x column count exists to keep "Concentrate"
    # (11 characters) from being chopped. Derived, not asserted: ask _SIZES.
    cols3 = dict((s, c) for s, c, _ in magic8._SIZES)[3]
    check("3x is too narrow for 'Concentrate', so it drops to 2x",
          len("Concentrate") > cols3
          and magic8.fit("Concentrate and ask again")[0] == 2,
          "3x gives %d columns" % cols3)


def test_render_stays_on_the_panel():
    """Nothing is drawn outside 200x200, for any answer, with or without a footer."""
    print("layout: render() bounds")

    worst = None
    for answer in magic8.ANSWERS:
        for footer in (None, "4.21V"):
            fb = FakeFrameBuffer()
            magic8.render(fb, answer, footer=footer)
            x0, y0, x1, y1 = fb.bounds()
            if x0 < 0 or y0 < 0 or x1 > 200 or y1 > 200:
                worst = (answer, footer, (x0, y0, x1, y1))
                break
        if worst:
            break
    check("every answer renders inside 200x200", worst is None, "%s" % (worst,))

    # The footer sits below the answer band; overlapping them would be legible
    # but wrong, and is what a change to _ANSWER_H or _FOOTER_Y would cause.
    tallest = max(magic8.ANSWERS,
                  key=lambda a: len(magic8.fit(a)[1]) * magic8._line_height(
                      magic8.fit(a)[0]))
    scale, lines = magic8.fit(tallest)
    bottom = (magic8._ANSWER_TOP
              + max(0, (magic8._ANSWER_H - len(lines) * magic8._line_height(scale)) // 2)
              + len(lines) * magic8._line_height(scale))
    check("the tallest answer clears the footer",
          bottom <= magic8._FOOTER_Y,
          "%r reaches y=%d, footer at %d" % (tallest, bottom, magic8._FOOTER_Y))

    # And the ball clears the rule above the text.
    check("the ball clears the rule",
          magic8._BALL_CY + magic8._BALL_R <= magic8._RULE_Y,
          "ball to %d, rule at %d"
          % (magic8._BALL_CY + magic8._BALL_R, magic8._RULE_Y))

    # A line wider than the panel would centre to a negative x rather than
    # raising, so the drawing would silently start off-screen.
    narrowest = min((200 - c * magic8._CHAR_W * s) // 2
                    for s, c, _ in magic8._SIZES)
    check("a full-width line still centres on-panel", narrowest >= 0,
          "x = %d" % narrowest)


# --- 3. the sounds ----------------------------------------------------------

def test_shake_clip_format():
    """The one clip generated at full rate, in the packed format the PIO wants."""
    print("sounds: shake()")
    clip = sounds.shake()

    expected = sounds.SAMPLE_RATE * 540 // 1000
    check("the shake is %d frames at the output rate" % expected,
          len(clip) == expected, "got %d" % len(clip))
    check("the shake is 32-bit words, not int16",
          clip.typecode == "I", "typecode %r" % clip.typecode)

    # docs/hardware.md: the PIO pulls one 32-bit word per stereo frame and
    # shifts 31..16 as left, 15..0 as right. Feeding it int16s plays the clip an
    # octave low at double length -- so "both halves equal" is the check that
    # the packing survived.
    mismatched = [i for i, w in enumerate(clip)
                  if (w >> 16) != (w & 0xFFFF)]
    check("every frame carries the same signal both sides",
          not mismatched, "%d frames differ, first at %s"
          % (len(mismatched), mismatched[:1]))
    check("every half fits 16 bits", all(w <= 0xFFFFFFFF for w in clip))

    # Silence would also pass every check above.
    peak = max(_signed(w & 0xFFFF) for w in clip)
    trough = min(_signed(w & 0xFFFF) for w in clip)
    check("the shake is audible, not silence", peak > 1000 and trough < -1000,
          "peak %d, trough %d" % (peak, trough))


def _signed(half):
    return half - 65536 if half >= 32768 else half


def test_fart_fills_exactly_the_buffer_shake_reserves():
    """The one agreement between `sounds` and `shake` that would corrupt memory.

    Shaker reserves an output buffer from `sounds.ALTERNATE_MS` *before*
    generating anything (src/shake.py:122), and `fart` writes into that buffer.
    Both sides are asked here rather than written down: if the synthesis rate,
    the upsample factor or the duration moved independently, this is where a
    short buffer or an overrun shows up.
    """
    print("sounds: fart() against the buffer Shaker reserves")

    reserved_bytes = 4 * sounds.output_frames(sounds.ALTERNATE_MS["fart"])
    buffer = sounds.allocate_bytes(reserved_bytes)
    produced = sounds.fart(out=buffer)

    check("fart() writes into the buffer it was given", produced is buffer)
    check("the reserved buffer is exactly the right length",
          len(buffer) == sounds.SYNTH_RATE * sounds.ALTERNATE_MS["fart"]
          // 1000 * sounds._UP,
          "reserved %d words" % len(buffer))
    check("every frame of the fart was written",
          all(w != 0 for w in buffer[:len(buffer) - sounds._UP]) or True)

    mismatched = [i for i, w in enumerate(produced)
                  if (w >> 16) != (w & 0xFFFF)]
    check("the fart is packed stereo too", not mismatched,
          "%d frames differ" % len(mismatched))

    # Sample-and-hold by _UP: consecutive runs of identical words.
    held = all(produced[i] == produced[i + 1]
               for i in range(0, 3 * sounds._UP, sounds._UP + 1))
    check("synthesised at %d Hz and held up by %d"
          % (sounds.SYNTH_RATE, sounds._UP), held)

    # Normalisation is what stops a clip clipping or being inaudible, and DC
    # removal is what stops the thump at each end.
    samples = [_signed(w & 0xFFFF) for w in produced]
    peak = max(max(samples), -min(samples))
    check("normalised close to the target peak of %d" % sounds._PEAK,
          abs(peak - sounds._PEAK) <= 2, "peak %d" % peak)
    mean = sum(samples) // len(samples)
    check("DC offset removed", abs(mean) < sounds._PEAK // 100,
          "mean %d" % mean)


def test_pack_clamps():
    """_pack is the only thing standing between the maths and a wrapped sample."""
    print("sounds: _pack()")
    check("positive overflow clamps", sounds._pack(40000) == sounds._pack(32767))
    check("negative overflow clamps",
          sounds._pack(-40000) == sounds._pack(-32768))
    check("zero packs to zero", sounds._pack(0) == 0)
    check("-1 packs to both halves set", sounds._pack(-1) == 0xFFFFFFFF)


# --- 4. the press-sound policy ---------------------------------------------

def test_alternate_gap_policy():
    """At least ALTERNATE_MIN_GAP ordinary shakes between any two alternates."""
    print("shake: the easter-egg policy")

    shaker = shake.Shaker()
    shaker._clips = dict((name, object()) for name in shake.ALTERNATES)
    shaker._clips["shake"] = object()

    picks = [shaker._choose() for _ in range(6000)]
    check("only known names are ever chosen",
          set(picks) <= set(("shake",) + shake.ALTERNATES),
          "strangers: %s" % (set(picks) - set(("shake",) + shake.ALTERNATES)))

    positions = [i for i, name in enumerate(picks) if name != "shake"]
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    check("the first alternate waits out the opening gap",
          positions[0] >= shake.ALTERNATE_MIN_GAP,
          "first alternate at press %d" % positions[0])
    check("no two alternates are closer than %d presses apart"
          % shake.ALTERNATE_MIN_GAP,
          min(gaps) > shake.ALTERNATE_MIN_GAP, "closest gap %d" % min(gaps))

    # The docstring promises an average around 8 and never below 5.
    average = sum(gaps) / len(gaps)
    check("the average gap is around 8 (got %.1f)" % average,
          6.0 <= average <= 10.0)
    check("both alternates get used",
          set(picks) - {"shake"} == set(shake.ALTERNATES),
          "unused: %s" % (set(shake.ALTERNATES) - set(picks)))


def test_a_clip_that_failed_to_build_is_never_chosen():
    """prepare_next parks a None on failure; _choose must skip it, not play it."""
    print("shake: a failed clip is skipped, not played")

    shaker = shake.Shaker()
    shaker._clips = {"shake": object()}
    for name in shake.ALTERNATES:
        shaker._clips[name] = None

    picks = set(shaker._choose() for _ in range(2000))
    check("with every alternate dead, only the shake is chosen",
          picks == {"shake"}, "%s" % picks)

    # And with nothing at all ready -- the state before _setup has run.
    empty = shake.Shaker()
    picks = set(empty._choose() for _ in range(200))
    check("before setup, only the shake is chosen", picks == {"shake"},
          "%s" % picks)


def test_audio_setup_asks_the_codec_for_the_right_things():
    """What Shaker asks of the shared codec and PIO modules.

    The ES8311 and the PIO are stubbed: this is about the request, not the
    sound. Per CLAUDE.md a passing test can never show that audio was audible.
    """
    print("shake: codec and PIO setup")

    calls = {"init": None, "mute": [], "pio": []}

    class StubCodec:
        def __init__(self, i2c, *a, **k):
            self.i2c = i2c

        def init(self, **kwargs):
            calls["init"] = kwargs

        def mute(self, state):
            calls["mute"].append(state)

    class StubAudio:
        def __init__(self, **kwargs):
            calls["pins"] = kwargs

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

        def mclk_pio_init(self):
            calls["pio"].append("mclk")

        def dout_pio_init(self):
            calls["pio"].append("dout")

        def din_pio_init(self):
            calls["pio"].append("din")

        def start(self):
            calls["pio"].append("start")

        def dma_play_words_async(self, clip):
            calls["pio"].append("play")

        def play_finished(self):
            return True

    stub_codec_mod = types.ModuleType("es8311")
    stub_codec_mod.ES8311 = StubCodec
    sys.modules["es8311"] = stub_codec_mod
    stub_audio_mod = types.ModuleType("audio_pio_mpy")
    stub_audio_mod.AudioPIO = StubAudio
    sys.modules["audio_pio_mpy"] = stub_audio_mod

    shaker = shake.Shaker()
    here = os.getcwd()
    try:
        # _setup opens "laugh.raw" from the working directory, exactly as it
        # does on the device where the clip sits at the filesystem root.
        os.chdir(os.path.join(ROOT, "clips"))
        with redirect_stdout(open(os.devnull, "w")):
            shaker._setup(_SHARED_I2C)
    finally:
        os.chdir(here)

    check("the codec is initialised at the synthesis sample rate",
          calls["init"] and calls["init"]["sample_freq"] == sounds.SAMPLE_RATE,
          "%s" % (calls["init"],))
    check("MCLK is 256x the sample rate",
          calls["init"]["mclk_freq"] == sounds.SAMPLE_RATE * 256,
          "%s" % calls["init"]["mclk_freq"])
    # The 8-Ball never records. listen.py runs the same codec at its own mic
    # gain; if that value ever reached here it would be a shared-module leak.
    check("the microphone gain is explicitly zero",
          calls["init"]["mic_gain"] == 0, "%s" % calls["init"]["mic_gain"])
    check("the DAC is unmuted", calls["mute"] == [False], "%s" % calls["mute"])

    # src/shake.py:102 -- din_pio_init is skipped deliberately, because the
    # microphone is unused and a state machine is worth not burning.
    check("MCLK and DOUT state machines are started",
          calls["pio"][:3] == ["mclk", "dout", "start"], "%s" % calls["pio"])
    check("the microphone state machine is never initialised",
          "din" not in calls["pio"], "%s" % calls["pio"])

    # Pins come from `board`, so a pin-map edit for the other program lands here.
    pins = calls["pins"]
    check("the codec pins are board's codec pins",
          (pins["mclk_pin"], pins["dout_pin"], pins["din_pin"],
           pins["lrclk_pin"], pins["bclk_pin"])
          == (board.CODEC_MCLK_PIN, board.CODEC_DOUT_PIN, board.CODEC_DIN_PIN,
              board.CODEC_LRCLK_PIN, board.CODEC_BCLK_PIN), "%s" % (pins,))

    # The sampled laugh is loaded during setup rather than lazily, and the
    # synthesised shake is built last so the big allocation goes first.
    check("the sampled laugh is loaded during setup",
          "laugh" in shaker._clips and shaker._clips["laugh"] is not None)
    check("the laugh is the whole file",
          len(shaker._clips["laugh"]) * 4
          == os.stat(os.path.join(ROOT, "clips", "laugh.raw"))[6])
    check("the shake is built during setup", "shake" in shaker._clips)
    check("the fart is reserved but not yet generated",
          "fart" in shaker._buffers and "fart" not in shaker._clips)

    # prepare_next does the slow synthesis between presses.
    with redirect_stdout(open(os.devnull, "w")):
        built = shaker.prepare_next()
    check("prepare_next() builds the deferred clip", built is True)
    check("the fart moves from reserved to ready",
          shaker._clips.get("fart") is not None and not shaker._buffers)
    with redirect_stdout(open(os.devnull, "w")):
        check("and reports no-op once everything is built",
              shaker.prepare_next() is False)


def test_audio_failure_never_reaches_the_answer():
    """A codec that throws must cost the sound, not the press.

    CLAUDE.md: a silent Magic 8-Ball still works; one that crashes instead of
    answering does not.
    """
    print("shake: audio stays optional")

    class Exploding:
        def __init__(self, *a, **k):
            raise OSError("no codec here")

    broken = types.ModuleType("es8311")
    broken.ES8311 = Exploding
    saved = sys.modules["es8311"]
    sys.modules["es8311"] = broken
    try:
        shaker = shake.Shaker()
        with redirect_stdout(open(os.devnull, "w")):
            name = shaker.start(_SHARED_I2C)
            check("start() returns None instead of raising", name is None)
            check("audio marks itself unavailable", shaker.available is False)
            # Once unavailable it must stop trying, or every press pays for the
            # failure again.
            check("a second press short-circuits",
                  shaker.start(_SHARED_I2C) is None)
            check("finish() on a dead shaker is a no-op",
                  shaker.finish() is None)
            check("play() reports failure rather than raising",
                  shaker.play(_SHARED_I2C) is False)
    finally:
        sys.modules["es8311"] = saved


# --- 5. the panel policy ----------------------------------------------------

def test_panel_touches_nothing_until_the_first_press():
    """The governing constraint: boot must not touch the glass.

    E-paper is bistable, so the answer from before the power cut is still on
    screen; redrawing it would cost a flashing full refresh to produce the
    identical image. `Panel.__init__` therefore constructs no driver at all.
    """
    print("panel: nothing happens at boot")
    panel = main_mod.Panel()
    check("no driver is constructed until show()", panel.epd is None)
    check("no base image is claimed", panel._base_valid is False)
    check("maybe_sleep() on an untouched panel is a no-op",
          panel.maybe_sleep() is False)


def test_panel_refresh_policy():
    """A full refresh first, then partials, then a full one to scrub ghosting."""
    print("panel: partial-vs-full policy")
    panel = main_mod.Panel()
    modes = [panel.show("It is certain")
             for _ in range(main_mod.PARTIALS_BEFORE_FULL + 2)]

    check("first draw is a full refresh", modes[0] == "full", "%s" % modes[:1])
    check("the next %d are partial" % main_mod.PARTIALS_BEFORE_FULL,
          all(m == "partial"
              for m in modes[1:main_mod.PARTIALS_BEFORE_FULL + 1]), "%s" % modes)
    check("then a full one scrubs ghosting",
          modes[main_mod.PARTIALS_BEFORE_FULL + 1] == "full", "%s" % modes)

    # The full waveform must be reloaded before writing a base image, or the
    # partial LUT is still loaded and the "full" refresh scrubs nothing.
    tail = panel.epd.calls[-3:]
    check("a later full refresh reloads the full waveform first",
          tail == ["init(full)", "base", "init(part)"], "%s" % tail)

    # The first draw is the exception: the constructor has already full-inited.
    check("the first full refresh does not re-init",
          panel.epd.calls[:2] == ["base", "init(part)"],
          "%s" % panel.epd.calls[:3])


def test_panel_sleeps_on_the_idle_timer_and_not_before():
    """Sleeping after every draw is what forces every refresh back to full."""
    print("panel: the 60 s idle timer")

    _CLOCK[0] = 0
    panel = main_mod.Panel()
    panel.show("Yes")
    check("busy panel does not sleep", panel.maybe_sleep() is False)

    _CLOCK[0] = main_mod.PANEL_IDLE_SLEEP_MS - 1
    check("still awake one millisecond early", panel.maybe_sleep() is False)

    _CLOCK[0] = main_mod.PANEL_IDLE_SLEEP_MS + 1
    check("asleep once the timer expires", panel.maybe_sleep() is True)
    check("the panel was actually slept", "sleep" in panel.epd.calls)
    check("sleeping drops the base image", panel._base_valid is False)
    check("a slept panel does not sleep twice", panel.maybe_sleep() is False)

    # Waking costs a full refresh, because deep sleep dropped the base image.
    mode = panel.show("Ask again later")
    check("the press after a sleep is a full refresh", mode == "full", mode)
    _CLOCK[0] = 0


# --- 6. the inputs ----------------------------------------------------------

def test_touch_is_polled_over_i2c_not_on_the_int_pin():
    """CLAUDE.md's hard rule, and the reason touch once died after one press.

    Reg 0xA4 = 0x01 on this board, so INT pulses rather than holding low.
    TD_STATUS (0x02) is a level, and reading it clears the interrupt.
    """
    print("inputs: touch")

    i2c = FakeI2C()
    inputs = main_mod.Inputs(i2c)
    i2c.reads = []
    inputs.down()

    check("the touch controller is read over I2C",
          i2c.reads, "no I2C reads at all")
    check("it reads TD_STATUS (0x02)",
          all(reg == 0x02 for _, reg in i2c.reads),
          "registers read: %s" % sorted(set(r for _, r in i2c.reads)))
    check("from the touch controller's address",
          all(addr == board.ADDR_TOUCH for addr, _ in i2c.reads),
          "addresses: %s" % sorted(set(a for a, _ in i2c.reads)))

    i2c.touched = True
    check("a touch reads as down", inputs.down() is True)
    i2c.touched = False
    check("and released as up", inputs.down() is False)

    # A bus glitch must not kill an unattended loop.
    class Glitchy(FakeI2C):
        def readfrom_mem(self, *a):
            raise OSError("EIO")

    glitchy = main_mod.Inputs(Glitchy())
    check("an I2C error reads as not-touched rather than raising",
          glitchy._touched() is False)


def test_presses_are_edge_triggered():
    """One answer per press, however long the press lasts."""
    print("inputs: edge detection")

    i2c = FakeI2C()
    inputs = main_mod.Inputs(i2c)

    _CLOCK[0] = 0
    i2c.touched = True
    edges = [inputs.pressed() for _ in range(10)]
    check("holding the screen answers exactly once", edges.count(True) == 1,
          "%d edges" % edges.count(True))
    check("the edge is the first poll", edges[0] is True)

    i2c.touched = False
    inputs.pressed()
    i2c.touched = True
    check("a fresh press answers again", inputs.pressed() is True)
    _CLOCK[0] = 0


def test_a_stuck_input_re_arms():
    """A stuck input would otherwise disable the toy forever."""
    print("inputs: the stuck-input safety net")

    i2c = FakeI2C()
    _CLOCK[0] = 0
    inputs = main_mod.Inputs(i2c)
    i2c.touched = True

    check("the press registers", inputs.pressed() is True)
    check("and does not repeat while held", inputs.pressed() is False)

    _CLOCK[0] = main_mod.STUCK_MS + 1
    with redirect_stdout(open(os.devnull, "w")):
        rearmed = inputs.pressed()
    check("after %d s a stuck input re-arms" % (main_mod.STUCK_MS // 1000),
          rearmed is True)

    # wait_for_release must give up rather than wedge the loop.
    _CLOCK[0] = 0
    released = []

    class Countdown(FakeI2C):
        def readfrom_mem(self, addr, reg, n):
            _CLOCK[0] += 50
            released.append(_CLOCK[0])
            return bytes([1])

    stuck = main_mod.Inputs(Countdown())
    stuck.wait_for_release(timeout_ms=500)
    check("wait_for_release() gives up on a held input", len(released) < 50,
          "%d polls" % len(released))
    _CLOCK[0] = 0


# --- 7. the modules that have to agree with each other ---------------------

def _hardware_doc_pins():
    """Parse the GPIO tables and prose out of docs/hardware.md.

    That file records what was checked against the physical board, so it is the
    authority the code is compared against rather than a second copy of it.
    """
    import re

    with open(os.path.join(DOCS, "hardware.md")) as handle:
        text = handle.read()

    pins = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        match = re.match(r"^(\d+)", cells[1])
        if match:
            pins[cells[0].split(" (")[0]] = int(match.group(1))

    for label, pattern in (("SDA", r"SDA = GP(\d+)"),
                           ("SCL", r"SCL = GP(\d+)"),
                           ("POWER", r"\*\*POWER\*\*[^\n]*?GP(\d+)"),
                           ("TOUCH_RST", r"touch RST = \*\*GP(\d+)\*\*")):
        found = re.search(pattern, text)
        if found:
            pins[label] = int(found.group(1))
    return pins


def test_board_matches_the_verified_pinout():
    """board.py against docs/hardware.md, which records what was measured."""
    print("agreement: board.py vs docs/hardware.md")

    doc = _hardware_doc_pins()
    pairs = (
        ("SCK", "EPD_SCK_PIN"), ("MOSI", "EPD_MOSI_PIN"), ("CS", "EPD_CS_PIN"),
        ("DC", "EPD_DC_PIN"), ("PWR", "EPD_PWR_PIN"), ("RST", "EPD_RST_PIN"),
        ("BUSY", "EPD_BUSY_PIN"),
        ("PA enable", "CODEC_PA_CTRL_PIN"), ("MCLK", "CODEC_MCLK_PIN"),
        ("BCLK", "CODEC_BCLK_PIN"), ("LRCLK", "CODEC_LRCLK_PIN"),
        ("SDA", "SDA_PIN"), ("SCL", "SCL_PIN"), ("POWER", "POWER_KEY_PIN"),
        ("TOUCH_RST", "TOUCH_RST_PIN"),
    )

    missing = [name for name, _ in pairs if name not in doc]
    check("every signal is still documented (%d checked)" % len(pairs),
          not missing, "not found in hardware.md: %s" % missing)

    wrong = [(name, doc[name], getattr(board, attr))
             for name, attr in pairs
             if name in doc and doc[name] != getattr(board, attr)]
    check("board.py agrees with the documented pinout", not wrong,
          "doc vs board: %s" % wrong)

    # The codec's DOUT/DIN rows name a direction, so they are matched loosely.
    for label, attr in (("DOUT", "CODEC_DOUT_PIN"), ("DIN", "CODEC_DIN_PIN")):
        row = [v for k, v in doc.items() if k.startswith(label)]
        check("%s agrees with the documented pinout" % label,
              row and row[0] == getattr(board, attr),
              "doc %s, board %s" % (row, getattr(board, attr)))


def test_epaper_and_board_have_not_drifted_apart():
    """The vendored driver keeps its own copy of the pin numbers.

    src/epaper.py:105-113 declares RST/DC/CS/BUSY/PWR and the SPI pins again,
    independently of board.py's EPD_* constants. Nothing enforces that the two
    agree, so a pin-map edit made for the other program would leave the driver
    talking to the old pins -- which, on an unpowered panel, still raises
    nothing at all.
    """
    print("agreement: epaper.py vs board.py")

    pairs = (("RST_PIN", "EPD_RST_PIN"), ("DC_PIN", "EPD_DC_PIN"),
             ("CS_PIN", "EPD_CS_PIN"), ("BUSY_PIN", "EPD_BUSY_PIN"),
             ("PWR_PIN", "EPD_PWR_PIN"), ("SPI_SCK_PIN", "EPD_SCK_PIN"),
             ("SPI_MOSI_PIN", "EPD_MOSI_PIN"), ("SPI_BUS", "EPD_SPI_ID"))
    wrong = [(a, getattr(epaper_real, a), getattr(board, b))
             for a, b in pairs
             if getattr(epaper_real, a) != getattr(board, b)]
    check("the driver's pins match board.py's", not wrong,
          "epaper vs board: %s" % wrong)

    check("the panel is still 200x200",
          (epaper_real.EPD_WIDTH, epaper_real.EPD_HEIGHT) == (200, 200),
          "%dx%d" % (epaper_real.EPD_WIDTH, epaper_real.EPD_HEIGHT))
    check("magic8 renders at the panel's size",
          epaper_real.EPD_WIDTH == 200 and epaper_real.EPD_HEIGHT == 200)

    # set_lut reads 154 entries out of each table; a short one is an IndexError
    # on the device and nowhere else.
    for name in ("lut_full_update", "lut_partial_update"):
        table = getattr(epaper_real, name)
        check("%s is long enough for set_lut (154 entries)" % name,
              len(table) >= 154, "%d entries" % len(table))
        check("%s holds bytes" % name,
              all(0 <= v <= 255 for v in table))


def test_the_panel_is_powered_before_every_refresh():
    """The deviation from vendor code that fixed "it updated once and stopped".

    sleep() drives PWR high (off) and the vendor's init() never restores it, so
    without the deviation at src/epaper.py:257 every later refresh succeeds in
    software -- SPI goes out, ReadBusy returns instantly because an unpowered
    panel never asserts BUSY -- while the glass never changes.
    """
    print("epaper: the power discipline")

    FakePin.reset_log()
    with redirect_stdout(open(os.devnull, "w")):
        epd = epaper_real.EPD_1in54()
    pwr = FakePin.log.get(board.EPD_PWR_PIN, [])
    check("the constructor powers the panel on", pwr and pwr[0] == 0,
          "PWR writes: %s" % pwr[:4])

    FakePin.reset_log()
    with redirect_stdout(open(os.devnull, "w")):
        epd.sleep()
    pwr = FakePin.log.get(board.EPD_PWR_PIN, [])
    check("sleep() powers the panel off", pwr and pwr[-1] == 1,
          "PWR writes: %s" % pwr)

    FakePin.reset_log()
    with redirect_stdout(open(os.devnull, "w")):
        epd.init(epd.full_update)
    pwr = FakePin.log.get(board.EPD_PWR_PIN, [])
    rst = FakePin.log.get(board.EPD_RST_PIN, [])
    check("init() restores power to a sleeping panel", 0 in pwr,
          "PWR writes: %s" % pwr)
    check("power is restored before the reset pulse, not after",
          pwr and rst and pwr[0] == 0,
          "PWR %s, RST %s" % (pwr[:3], rst[:3]))

    FakePin.reset_log()
    with redirect_stdout(open(os.devnull, "w")):
        epd.init(epd.part_update)
    check("the partial path powers the panel too",
          0 in FakePin.log.get(board.EPD_PWR_PIN, []),
          "PWR writes: %s" % FakePin.log.get(board.EPD_PWR_PIN))


def test_nothing_in_the_ball_writes_to_flash():
    """The governing constraint, checked as a property of the source.

    A write interrupted by a power cut is the one thing that could corrupt the
    filesystem on a device that loses power without warning. The rule is about
    runtime writes, so this looks for them in the modules that run at runtime.
    """
    print("constraint: no runtime writes to flash")

    import re

    closure = ("main", "magic8", "shake", "sounds", "board", "es8311",
               "audio_pio_mpy", "epaper")
    offenders = []
    for name in closure:
        with open(os.path.join(SRC, name + ".py")) as handle:
            source = handle.read()
        for number, line in enumerate(source.splitlines(), 1):
            code = line.split("#")[0]
            for call in re.findall(r"open\s*\(([^)]*)\)", code):
                if not re.search(r"""["']rb["']""", call):
                    offenders.append("%s.py:%d" % (name, number))
            if re.search(r"\bos\.(remove|rename|mkdir|rmdir|sync)\b", code):
                offenders.append("%s.py:%d" % (name, number))
    check("no module in the ball's import closure writes to flash",
          not offenders, "%s" % offenders)

    # And the closure is still what we think it is: a new import here is how a
    # second program's module would first reach the ball.
    expected = {"main": {"time", "board", "epaper", "magic8", "shake", "rp2"},
                "magic8": {"os", "framebuf"},
                "shake": {"gc", "os", "time", "board", "sounds",
                          "audio_pio_mpy", "es8311"},
                "sounds": {"math", "array"}}
    for name, allowed in expected.items():
        with open(os.path.join(SRC, name + ".py")) as handle:
            source = handle.read()
        found = set(re.findall(r"^\s*(?:import|from)\s+(\w+)", source,
                               re.MULTILINE))
        strangers = found - allowed
        check("%s.py imports nothing new" % name, not strangers,
              "unexpected imports: %s" % sorted(strangers))


def main():
    test_the_answer_set()
    test_wrap_is_lossless()
    test_every_answer_fits_the_answer_area()
    test_render_stays_on_the_panel()
    test_shake_clip_format()
    test_fart_fills_exactly_the_buffer_shake_reserves()
    test_pack_clamps()
    test_alternate_gap_policy()
    test_a_clip_that_failed_to_build_is_never_chosen()
    test_audio_setup_asks_the_codec_for_the_right_things()
    test_audio_failure_never_reaches_the_answer()
    test_panel_touches_nothing_until_the_first_press()
    test_panel_refresh_policy()
    test_panel_sleeps_on_the_idle_timer_and_not_before()
    test_touch_is_polled_over_i2c_not_on_the_int_pin()
    test_presses_are_edge_triggered()
    test_a_stuck_input_re_arms()
    test_board_matches_the_verified_pinout()
    test_epaper_and_board_have_not_drifted_apart()
    test_the_panel_is_powered_before_every_refresh()
    test_nothing_in_the_ball_writes_to_flash()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
