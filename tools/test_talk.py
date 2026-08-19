#!/usr/bin/env python3
"""Tests the seam: audio in, a sentence on the panel.

    python3 tools/test_talk.py

Every piece `src/talk.py` joins is covered elsewhere and covered well --
`test_spotter` pins the recogniser bit for bit, `test_eliza` has 71 tests on the
rule engine, `test_vad` and `test_record_stream` cover capture and transport.
None of them covers the *join*, and the join is where the expensive bug lives:
it is only exercised on the device, it crosses three modules written by three
people, and its likely failure is a silent mismatch rather than a crash. A label
the spotter returns that `vocab` maps to nothing, or a form the engine has never
heard of, produces a deflection -- which is also what a correct rejection
produces. It looks like the recogniser being poor.

So the checks here are mostly about *agreement between modules*, and both sides
of every agreement are derived from their own module. Nothing here writes out a
list of words; a hand-written list is the duplicated-constant trap, which has
bitten this project twice already.

Stubs `machine`, `framebuf`, `epaper` and `rp2` the way `test_record_stream.py`
stubs its dependencies, and runs the real `talk.py` under CPython. Loads every
module from source rather than through importlib, for the reason in
`test_spotter._load_source`: importlib consults `__pycache__`, and a stale cache
is how a suite goes green without running anything.
"""

import math
import os
import random
import sys
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src")
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
    IN = 0
    OUT = 1
    PULL_UP = 2
    PULL_DOWN = 3

    def __init__(self, *a, **k):
        self.level = 1

    def value(self, *a):
        return self.level


class FakeI2C:
    def __init__(self, *a, **k):
        self.touched = False

    def readfrom_mem(self, addr, reg, n):
        return bytes([1 if self.touched else 0])

    def writeto(self, *a, **k):
        pass

    def scan(self):
        return [0x38, 0x18]


class FakeFrameBuffer:
    """Records what was drawn, so the renderer can be checked for overflow."""

    def __init__(self, *a, **k):
        self.marks = []

    def fill(self, *a):
        pass

    def fill_rect(self, x, y, w, h, c):
        self.marks.append((x, y, w, h))

    def rect(self, *a):
        pass

    def hline(self, x, y, w, c):
        self.marks.append((x, y, w, 1))

    def text(self, t, x, y, c):
        self.marks.append((x, y, 8 * len(t), 8))

    def pixel(self, x, y):
        return (x + y) % 3 == 0

    def extent(self):
        if not self.marks:
            return 0, 0
        return (max(m[0] + m[2] for m in self.marks),
                max(m[1] + m[3] for m in self.marks))


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

    def sleep(self):
        self.calls.append("sleep")


def install_stubs():
    machine = types.ModuleType("machine")
    machine.Pin = FakePin
    machine.I2C = FakeI2C
    machine.ADC = lambda *a, **k: types.SimpleNamespace(read_u16=lambda: 40000)
    machine.SPI = lambda *a, **k: types.SimpleNamespace(write=lambda *x: None)
    machine.freq = lambda: 150_000_000
    sys.modules["machine"] = machine

    framebuf = types.ModuleType("framebuf")
    framebuf.MONO_HLSB = 0
    framebuf.FrameBuffer = FakeFrameBuffer
    sys.modules["framebuf"] = framebuf

    epaper = types.ModuleType("epaper")
    epaper.EPD_1in54 = FakeEPD
    sys.modules["epaper"] = epaper

    sys.modules["rp2"] = types.ModuleType("rp2")

    if not hasattr(__import__("time"), "ticks_ms"):
        import time as _time
        _time.ticks_ms = lambda: int(_time.monotonic() * 1000)
        _time.ticks_us = lambda: int(_time.monotonic() * 1e6)
        _time.ticks_diff = lambda a, b: a - b
        _time.ticks_add = lambda a, b: a + b
        _time.sleep_ms = lambda ms: None


install_stubs()

for _name in ("speech_tables", "vocab", "eliza_rules", "eliza", "magic8",
              "screen", "listen", "vad", "spotter", "board"):
    _load_source(_name, os.path.join(SRC, _name + ".py"))

import board            # noqa: E402
import eliza            # noqa: E402
import eliza_rules      # noqa: E402
import screen           # noqa: E402
import spotter          # noqa: E402
import vocab            # noqa: E402

talk = _load_source("talk", os.path.join(SRC, "talk.py"))


# --- 1. the vocabulary agreement -------------------------------------------

def test_every_spotted_word_is_known_to_the_engine():
    """Whatever the spotter can return must be a word DOCTOR has heard of.

    Both sides derived: the emittable set from `vocab`, and the accepted set
    from the rule data itself -- keywords, word-class members, and tagged words.
    A word in none of those produces a deflection, which is indistinguishable
    from a correct rejection, so it would look like the recogniser being poor
    rather than like a vocabulary that does not line up.
    """
    print("vocabulary: what the spotter emits vs what the engine accepts")
    doctor = eliza.Doctor(priority=vocab.PRIORITY)
    known = set(eliza_rules.RULES) | set(doctor.word_classes()) | set(eliza_rules.TAGS)

    # There are *two* routes by which a spotted word can reach a reply, and the
    # test found this by getting it wrong: five of the twelve nouns -- WORK,
    # MONEY, SLEEP, DEATH, LOVE -- are not keywords, not word-class members and
    # not tagged. They reach a reply only as an echo, filled into a possessive
    # template from the `nouns=` argument. So they work, and they work *solely*
    # because talk.Conversation passes vocab.NOUNS; drop that argument and these
    # five go silent while the other sixteen classes carry on, which is the
    # quietest possible way for a vocabulary to half-fail.
    direct = [vocab.ECHO[l] for l in vocab.LABELS if vocab.ECHO[l] in known]
    echo_only = [vocab.ECHO[l] for l in vocab.LABELS if vocab.ECHO[l] not in known]

    print("       %d reach the engine directly, %d only as an echo: %s"
          % (len(direct), len(echo_only), " ".join(echo_only)))
    check("every echo-only word is in vocab.NOUNS, i.e. actually passed",
          all(word in vocab.NOUNS for word in echo_only),
          "not passed as nouns: %s"
          % [w for w in echo_only if w not in vocab.NOUNS])
    check("the two routes together cover the vocabulary (%d classes)"
          % len(vocab.LABELS), len(direct) + len(echo_only) == len(vocab.LABELS))

    # Verified rather than asserted: run the same vocabulary through the engine
    # with and without our noun list and see which classes fall silent. This is
    # the dependency the comment above describes, so it is measured here rather
    # than believed.
    deflections = set(eliza_rules.NONE)
    silent = {}
    for nouns, tag in ((vocab.NOUNS, "ours"), (None, "engine default")):
        doc = eliza.Doctor(priority=vocab.PRIORITY)
        silent[tag] = [vocab.ECHO[l] for l in vocab.LABELS
                       if doc.respond_to_keywords([vocab.ECHO[l]], nouns=nouns)
                       in deflections]
    check("with vocab.NOUNS, no class falls silent", not silent["ours"],
          "silent: %s" % silent["ours"])
    check("without it, exactly the echo-only words fall silent",
          sorted(silent["engine default"]) == sorted(echo_only),
          "silent %s, expected %s" % (sorted(silent["engine default"]),
                                      sorted(echo_only)))
    print("       dropping nouns=vocab.NOUNS would silence %d of %d classes"
          % (len(silent["engine default"]), len(vocab.LABELS)))

    # And the other direction of the same join: every spoken form maps back to a
    # class, or enrolling it produces a template nothing can ever return.
    unmapped = [form for form in vocab.FORMS if vocab.label_of(form) is None]
    check("every spoken form maps to a class (%d forms)" % len(vocab.FORMS),
          not unmapped, "unmapped: %s" % unmapped)

    # Substitution runs before matching, so a word the engine rewrites must be
    # known under its rewritten form too.
    #
    # **This check is vacuous today, and that is stated rather than left for
    # the green tick to imply.** `swapped` is empty: `eliza_rules.SUBS` keys are
    # the informal forms a person types -- DAD, MOM, DREAMS, I'M, CANT -- and
    # the vocabulary is spelled in the canonical forms those rewrite *to*, so
    # nothing the spotter can emit is ever substituted. It is a guard against a
    # change, not a measurement of today: add DAD or MOM as a spoken form, both
    # of which are natural candidates, and this starts doing work.
    #
    # It was also broken until now -- the comparison named `accepted`, which is
    # defined nowhere in this file, and only escaped because a comprehension
    # over an empty list never evaluates its condition. A vacuous check and an
    # undefined name hide each other perfectly: either alone is visible, and
    # together they are a passing test. `tools/test_eliza.py`'s TestVocabContract
    # guards its own inputs against exactly this for the same reason.
    swapped = [(w, eliza_rules.SUBS[w]) for w in
               (vocab.ECHO[l] for l in vocab.LABELS) if w in eliza_rules.SUBS]
    bad = [(a, b) for a, b in swapped if b not in known]
    check("substituted forms are also known (%d substituted%s)"
          % (len(swapped), ", so this asserts nothing yet" if not swapped else ""),
          not bad, "lost in substitution: %s" % bad)


# --- 2. a spotted word, end to end -----------------------------------------

def test_a_spotted_word_produces_a_real_reply():
    """Through the real _spot_keyword -> Conversation -> screen path."""
    print("end to end: a spotted word becomes a sentence on the panel")
    session = talk.Conversation()
    deflections = set(eliza_rules.NONE)
    saved_spotter = talk.spotter

    spoke = 0
    for label in vocab.LABELS:
        talk.spotter = types.SimpleNamespace(spot=lambda s, a, b, _l=label: _l)
        heard = talk._spot_keyword(None, 0, 0)
        check_label = heard == label
        reply = session.reply(heard)
        if not check_label or not reply:
            check("label %s survives _spot_keyword" % label, False)
            continue
        # sentence_case has been applied by now, so compare case-insensitively
        # against the script's own deflections.
        if reply.upper() not in deflections and not reply.startswith("STUB"):
            spoke += 1
    check("every class produces a reply that is not a deflection (%d/%d)"
          % (spoke, len(vocab.LABELS)), spoke == len(vocab.LABELS))

    # The echoable nouns must actually appear in the reply some of the time --
    # that is the entire argument for spending the vocabulary on them.
    session = talk.Conversation()
    echoed = 0
    for label in vocab.LABELS:
        if vocab.ECHO[label] not in vocab.NOUNS:
            continue
        for _try in range(6):
            reply = session.reply(label)
            if vocab.ECHO[label].lower() in reply.lower():
                echoed += 1
                break
    check("nouns are echoed back by name (%d/%d)" % (echoed, len(vocab.NOUNS)),
          echoed >= len(vocab.NOUNS) - 2,
          "only %d of %d nouns ever echoed" % (echoed, len(vocab.NOUNS)))
    talk.spotter = saved_spotter


def test_a_rejection_reaches_the_no_keyword_path():
    """Silence, and a below-threshold match, must both say something real."""
    print("rejection: nothing heard, and heard-but-rejected")
    session = talk.Conversation()
    saved_spotter = talk.spotter

    talk.spotter = None
    check("no spotter -> _spot_keyword returns None",
          talk._spot_keyword(None, 0, 0) is None)

    # A spotter that rejects, which is what precision-over-recall looks like.
    talk.spotter = types.SimpleNamespace(spot=lambda s, a, b: None)
    check("a rejecting spotter returns None", talk._spot_keyword(None, 0, 0) is None)

    replies = [session.reply(None) for _ in range(4)]
    check("a rejection produces a real deflection, not an empty string",
          all(r and r.strip() and not r.startswith("STUB") for r in replies),
          "got %s" % replies)
    check("deflections rotate rather than repeating",
          len(set(replies)) > 1, "all identical: %s" % replies[0])

    # The greeting path: nothing heard on the very first turn.
    fresh = talk.Conversation()
    check("first turn with nothing heard greets rather than complaining",
          fresh.turns == 0 and fresh.greeting().upper() == eliza_rules.GREETING)

    # ...and only the first. main() picks between greeting() and NOTHING_HEARD
    # on `turns == 0`, so a greeting that does not count as a turn greets on
    # every failed endpoint instead of once -- and NOTHING_HEARD becomes
    # unreachable until a press succeeds. Endpointing failures are common
    # enough to make that the normal experience rather than an edge case:
    # 49% of utterances at 8 dB SNR, 83% at 6 dB.
    check("the greeting counts as a turn, so main() says it once", fresh.turns == 1,
          "turns is %d after greeting; main() will greet again" % fresh.turns)
    check("a greeting on the stub path counts too",
          _greeting_turns_without_eliza() == 1)
    talk.spotter = saved_spotter


def _greeting_turns_without_eliza():
    """`turns` after one greeting with no engine deployed. See above."""
    saved = talk.eliza
    try:
        talk.eliza = None
        stub = talk.Conversation()
        stub.greeting()
        return stub.turns
    finally:
        talk.eliza = saved


# --- 3. the degraded paths, which are currently only claimed by comments ----

def test_degraded_paths():
    print("degradation: each missing module, as the comments claim")
    saved = (talk.eliza, talk.vocab, talk.spotter, talk.templates)
    try:
        talk.eliza = None
        talk.vocab = None
        session = talk.Conversation()
        reply = session.reply("mother")
        check("no eliza -> a reply that says so, rather than a crash",
              isinstance(reply, str) and reply.startswith("STUB"), reply)
        check("no eliza -> greeting says so too",
              session.greeting().startswith("STUB"))

        talk.eliza, talk.vocab = saved[0], saved[1]
        talk.spotter = None
        check("no spotter -> None, and the loop takes the no-keyword path",
              talk._spot_keyword(None, 0, 0) is None)

        # A spotter that raises must not take the program down with it.
        def _boom(*a):
            raise ValueError("simulated spotter failure")

        talk.spotter = types.SimpleNamespace(spot=_boom)
        check("a spotter that raises is caught", talk._spot_keyword(None, 0, 0) is None)

        talk.templates = None
        check("no templates -> reserve_templates returns None",
              talk.reserve_templates() is None)

        # Packed as statics with no expansion available: must refuse rather than
        # load a buffer the matcher would read as confident nonsense.
        talk.templates = types.SimpleNamespace(
            PACKED="statics", BUFFER_BYTES=1024, TOTAL_FRAMES=8, INDEX=(),
            load=lambda buf=None, expand=None: (_ for _ in ()).throw(
                AssertionError("load must not be called without expand")))
        talk.spotter = types.SimpleNamespace(spot=lambda *a: None)  # no .expand
        check("statics without spotter.expand -> refuses, load never called",
              talk.reserve_templates() is None)
    finally:
        talk.eliza, talk.vocab, talk.spotter, talk.templates = saved


# --- 4. the allocation order, asserted rather than described ---------------

def test_allocation_order():
    """Capture buffer before templates. 92 KB of peak heap rides on it.

    Instrumented rather than read: the two allocations record the order they
    happen in. A refactor that reorders them innocently -- and "largest first"
    is a plausible reason to -- fails here.
    """
    print("allocation order")
    order = []
    saved_recorder = talk.listen.Recorder
    saved_templates = talk.templates
    try:
        class Recorder:
            def __init__(self, *a, **k):
                order.append("capture")

        talk.listen.Recorder = Recorder
        talk.templates = types.SimpleNamespace(
            PACKED="full", BUFFER_BYTES=64, TOTAL_FRAMES=2, INDEX=(),
            load=lambda buf=None, expand=None: order.append("templates") or buf)

        talk.reserve()
        check("capture buffer is reserved before templates",
              order == ["capture", "templates"], "order was %s" % order)
    finally:
        talk.listen.Recorder = saved_recorder
        talk.templates = saved_templates


# --- 5. the reply reaches the glass ----------------------------------------

# The one place in this file that knows the shape of a rule template. It has
# changed twice: `(kind, payload)` gained a `kind` field in VOCAB's sibling
# change, and then became `(kind, mood, payload)` when replies gained mood tags
# and terminal punctuation. Both times it broke consumers that unpacked inline.
#
# `eliza` exposes no template-enumeration accessor, so a test that measures the
# panel budget has to reach into the data. Reaching in from one function means
# the next shape change fails here, once, with a message that says what changed
# -- rather than three times with a tuple-unpacking error that names neither the
# field nor the reason.
TEMPLATE_ARITY = 3


def _reply_templates():
    """Yield (reachable_on_device, text) for every template that can be shown.

    `reachable` is the distinction that matters for the panel budget. In bag
    mode -- the only mode the device uses -- PHRASE templates are filtered out
    and MEMORY only fills on the ordered path `talk.py` never calls. So a
    template that overflows is only a problem if the device can reach it.
    """
    for _rank, _sub, decomps in eliza_rules.RULES.values():
        for _pattern, templates in decomps:
            for entry in templates:
                if len(entry) != TEMPLATE_ARITY:
                    raise AssertionError(
                        "eliza_rules template tuples are %d-wide, expected %d: %r"
                        " -- the rule data changed shape; update _reply_templates"
                        % (len(entry), TEMPLATE_ARITY, entry))
                kind, _mood, payload = entry
                if kind in eliza_rules.SPOTTABLE:
                    yield True, payload
                elif kind == eliza_rules.PHRASE:
                    yield False, payload
    for text in eliza_rules.NONE:
        yield True, text
    for entry in eliza_rules.MEMORY:
        if len(entry) != TEMPLATE_ARITY:
            raise AssertionError(
                "eliza_rules.MEMORY entries are %d-wide, expected %d: %r"
                % (len(entry), TEMPLATE_ARITY, entry))
        yield False, entry[2]


def _rendered(text, echo):
    import re

    return eliza.sentence_case(re.sub(r"\b\d\b", echo, text))


def test_the_reply_renders():
    """The panel budget, at the echo width the device actually produces.

    Measured at three widths rather than one, because the device and the stress
    case answer different questions: one asks whether this works, the other how
    much room is left before it stops working.
    """
    print("rendering: the panel budget")
    widths = (("1-word (what the device produces)", "mother"),
              ("3-word", "your mother and your father"),
              ("6-word (stress)", "your mother and your father again"))

    for name, echo in widths:
        worst_reachable = 0
        worst_any = 0
        overflow_reachable = []
        overflow_unreachable = 0
        for reachable, text in _reply_templates():
            scale, lines = _fit(text, echo)
            worst_any = max(worst_any, len(lines))
            bad = scale != 2 or len(lines) > screen._SIZES[0][2]
            if reachable:
                worst_reachable = max(worst_reachable, len(lines))
                if bad:
                    overflow_reachable.append(_rendered(text, echo))
            elif bad:
                overflow_unreachable += 1

        print("       %-34s worst %d lines reachable, %d overall, %d unreachable"
              " overflow" % (name, worst_reachable, worst_any, overflow_unreachable))
        check("%s: nothing the device can reach overflows" % name,
              not overflow_reachable,
              "%d overflow, first: %s" % (len(overflow_reachable),
                                          overflow_reachable[:1]))

    # The stress case is where it gets tight, and where the only overflow lives
    # in templates bag mode filters out. Pinned so that ceasing to be true --
    # a PHRASE template becoming reachable, say -- fails here.
    stress = "your mother and your father again"
    reachable_lines = max(len(_fit(t, stress)[1])
                          for r, t in _reply_templates() if r)
    check("at a six-word echo the reachable worst case is 9 lines",
          reachable_lines == 9, "got %d" % reachable_lines)

    longest = max((_rendered(t, stress) for _r, t in _reply_templates()), key=len)
    scale, lines = screen.fit(longest)
    reach = "device-reachable" if any(
        r and _rendered(t, stress) == longest for r, t in _reply_templates()
    ) else "NOT device-reachable"
    print("       longest rendered overall (%s): %d chars, scale %d, %d lines"
          % (reach, len(longest), scale, len(lines)))
    print("       -- falling to scale 1 is the graceful path, not truncation")
    print("       %s" % longest)

    fb = FakeFrameBuffer()
    screen.render(fb, longest, footer="MOTHER  4.10V")
    right, bottom = fb.extent()
    check("nothing is drawn outside the 200x200 panel",
          right <= 197 and bottom <= 197, "extends to %d x %d" % (right, bottom))

    # Terminal punctuation is new, and it is what made the budget non-obvious:
    # a character per reply is not free when the worst case had no spare lines.
    punctuated = sum(1 for _r, t in _reply_templates() if t and t[-1] in "?.!")
    total = sum(1 for _ in _reply_templates())
    check("replies carry terminal punctuation (%d/%d)" % (punctuated, total),
          punctuated > total * 0.8, "only %d of %d" % (punctuated, total))


def _fit(text, echo):
    return screen.fit(_rendered(text, echo))


# --- 6. the panel policy ---------------------------------------------------

def test_panel_refresh_policy():
    """A full refresh first, then partials, then a full one to scrub ghosting."""
    print("panel: partial-vs-full policy")
    panel = talk.Panel()
    modes = [panel.show("Hello there") for _ in range(talk.PARTIALS_BEFORE_FULL + 2)]
    check("first draw is a full refresh", modes[0] == "full", modes[:1])
    check("the next %d are partial" % talk.PARTIALS_BEFORE_FULL,
          all(m == "partial" for m in modes[1:talk.PARTIALS_BEFORE_FULL + 1]),
          modes)
    check("then a full one scrubs ghosting",
          modes[talk.PARTIALS_BEFORE_FULL + 1] == "full", modes)
    # The full waveform must be reloaded before writing a base image, or the
    # partial LUT is still loaded and the "full" refresh scrubs nothing.
    tail = panel.epd.calls[-3:]
    check("a later full refresh reloads the full waveform first",
          tail == ["init(full)", "base", "init(part)"], tail)
    check("the panel sleeps when idle", panel.maybe_sleep() is False
          or "sleep" in panel.epd.calls)


def main():
    test_every_spotted_word_is_known_to_the_engine()
    test_a_spotted_word_produces_a_real_reply()
    test_a_rejection_reaches_the_no_keyword_path()
    test_degraded_paths()
    test_allocation_order()
    test_the_reply_renders()
    test_panel_refresh_policy()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
