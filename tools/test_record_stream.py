#!/usr/bin/env python3
"""Round-trips the enrolment transport with no board attached.

    python3 tools/test_record_stream.py

`src/record_stream.py` runs on the device and `tools/pull_recording.py` runs on
the host, and neither has ever met the other. This runs the real device function
against a fake stdout, feeds the bytes it produced into the real host reader
through a fake serial port, and checks the samples came back.

That makes the two failure modes we cannot otherwise see testable:

- **Framing.** The host has to sync on the magic because the REPL echoes the
  command it was sent and prints its own prompts into the same stream. Anything
  that mishandles the bytes either side of the magic -- discarding what was
  already read past it, most obviously -- shows up as an intermittent short read
  on real hardware, depending on nothing more than where USB packet boundaries
  happened to fall. So the fake port is deliberately hostile: it prepends REPL
  echo, and it hands over the stream in awkward pieces, splitting mid-magic,
  mid-header and mid-trailer.
- **Contract drift.** If either side changes MAGIC, the header layout, the
  meaning of `count` (samples, not bytes) or the trailer, this fails. That is
  the entire reason it exists.

The device side is imported with `machine`, `board` and the codec stubbed, and
`listen.Recorder` replaced by one that returns a known waveform. It is the real
`record_stream.stream()` otherwise -- its framing, its chunking, its error path.
"""

import importlib.util
import math
import os
import struct
import sys
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# --- the device side, with the hardware stubbed ----------------------------

def known_waveform(count):
    """Something with structure, so a truncation or a byte swap is visible."""
    return [int(20000 * math.sin(2 * math.pi * 7 * i / count)) for i in range(count)]


class FakeStdoutBuffer:
    def __init__(self):
        self.chunks = []

    def write(self, data):
        self.chunks.append(bytes(data))
        return len(data)

    def value(self):
        return b"".join(self.chunks)


def load_record_stream(fail_at=None):
    """Import src/record_stream.py with its hardware dependencies replaced.

    `fail_at` = "start" makes the fake recorder refuse to start, which is how
    the error frame gets exercised.
    """
    machine = types.ModuleType("machine")

    class Pin:
        IN = 0
        OUT = 1
        PULL_UP = 2

        def __init__(self, *a, **k):
            pass

        def value(self, *a):
            return 1

    machine.Pin = Pin
    machine.I2C = lambda *a, **k: object()
    machine.ADC = lambda *a, **k: object()
    sys.modules["machine"] = machine

    board = types.ModuleType("board")
    board.bus = lambda *a, **k: object()
    board.Pin = Pin
    sys.modules["board"] = board

    es8311 = types.ModuleType("es8311")

    class ES8311:
        inits = []

        def __init__(self, *a, **k):
            pass

        def init(self, **kwargs):
            ES8311.inits.append(kwargs)

    es8311.ES8311 = ES8311
    sys.modules["es8311"] = es8311

    listen = types.ModuleType("listen")

    class Recorder:
        def __init__(self, rate=16000, max_samples=None, playing=None):
            self.rate = rate
            self.max_samples = max_samples
            self.buf = array("h", known_waveform(max_samples))
            self.available = None

        def start(self, i2c):
            return fail_at != "start"

        def full(self):
            return True

        def stop(self):
            # A capture can legitimately return fewer samples than asked for --
            # an early stop, or a DMA aborted on release. The header carries
            # what was captured, not what was requested, which is what keeps a
            # short capture from desyncing the stream.
            if fail_at == "short":
                # The 3 here and the 3 in test_short_capture's expectation are
                # two copies of one number, deliberately. This stub is the
                # behaviour under test and that is what the behaviour should
                # produce; deriving one from the other would leave the test
                # asserting only that the code agrees with itself. They are in
                # one file, and divergence fails loudly with "header says X,
                # captured Y" rather than quietly checking the wrong thing --
                # which is the distinction that matters. A duplicate is
                # dangerous when divergence is silent, not merely when it is a
                # duplicate.
                return self.max_samples // 3
            return self.max_samples

        def close(self):
            pass

    listen.Recorder = Recorder
    listen.MAX_RECORD_MS = 3000
    listen.SAMPLE_RATE = FIXTURE_RATE
    sys.modules["listen"] = listen

    spec = importlib.util.spec_from_file_location(
        "device_record_stream", os.path.join(ROOT, "src", "record_stream.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ES8311 = ES8311
    return module


# --- the host side, with pyserial stubbed ----------------------------------

class FakePort:
    """A serial port that hands out a canned stream in awkward pieces.

    `splits` are byte offsets at which a read is forced to stop short, so the
    host reader sees the boundaries it would see if USB packets landed badly --
    including in the middle of the magic and the middle of the trailer.
    """

    def __init__(self, data, splits=()):
        self.data = data
        self.pos = 0
        self.splits = sorted(s for s in splits if s > 0)
        self.written = []
        self.timeout = 1.0

    @property
    def in_waiting(self):
        return len(self.data) - self.pos

    def read(self, size=1):
        if self.pos >= len(self.data):
            return b""
        end = min(len(self.data), self.pos + size)
        for split in self.splits:
            if self.pos < split < end:
                end = split
                break
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def write(self, data):
        self.written.append(bytes(data))
        return len(data)

    def reset_input_buffer(self):
        pass


def load_pull_recording():
    if "serial" not in sys.modules:
        serial = types.ModuleType("serial")
        serial.Serial = lambda *a, **k: None
        serial.SerialException = Exception
        sys.modules["serial"] = serial
    spec = importlib.util.spec_from_file_location(
        "host_pull_recording", os.path.join(ROOT, "tools", "pull_recording.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the tests -------------------------------------------------------------

# What the REPL puts in front of the data: the echo of the command the host
# sent, and a prompt. This is why there is a magic header at all.
# Deliberately a literal transcript rather than a formatted one: it stands for
# whatever text happens to precede the magic, and the host discards all of it.
# The numbers in it do not have to track the fixture below and should not be
# "fixed" to.
REPL_NOISE = (
    b"import record_stream\r\n>>> record_stream.stream(seconds=0.500, rate=16000)\r\n"
)

# The fixture, in one place. `FIXTURE_COUNT` is derived rather than written out,
# because a test that holds its own copy of a number does not detect drift, it
# enforces it: when the code moves, the test is what fails, and a green suite
# ends up on the wrong side of the argument. The same rule the constant pins
# elsewhere in this file exist to apply.
FIXTURE_SECONDS = 0.5
FIXTURE_RATE = 16000
FIXTURE_COUNT = int(FIXTURE_SECONDS * FIXTURE_RATE)


def device_bytes(seconds, rate, fail_at=None):
    module = load_record_stream(fail_at=fail_at)
    fake = FakeStdoutBuffer()
    real = sys.stdout
    sys.stdout = types.SimpleNamespace(buffer=fake, write=real.write, flush=real.flush)
    try:
        ok = module.stream(seconds=seconds, rate=rate)
    finally:
        sys.stdout = real
    return ok, fake.value(), module, fake.chunks


def test_framing():
    print("device framing")
    ok, data, module, chunks = device_bytes(FIXTURE_SECONDS, FIXTURE_RATE)
    count = FIXTURE_COUNT

    check("stream() reported success", ok is True)
    check("starts with the magic", data.startswith(module.MAGIC),
          "got %r" % data[:8])
    header = data[len(module.MAGIC):len(module.MAGIC) + 8]
    got_rate, got_count = struct.unpack("<II", header)
    check("header is <II rate, count",
          got_rate == FIXTURE_RATE and got_count == count,
          "rate=%d count=%d" % (got_rate, got_count))
    check("count is samples, not bytes", got_count * 2 == len(data)
          - len(module.MAGIC) - 8 - len(module.TRAILER),
          "count=%d payload=%d bytes" % (got_count, len(data) - len(module.MAGIC) - 8 - len(module.TRAILER)))
    check("ends with the trailer", data.endswith(module.TRAILER),
          "got %r" % data[-8:])

    payload = data[len(module.MAGIC) + 8:-len(module.TRAILER)]
    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    expected = known_waveform(count)
    check("samples survive the wire", list(samples) == expected,
          "first differs at %s" % next((i for i, (a, b) in enumerate(zip(samples, expected)) if a != b), None))

    # Chunking, checked against what was actually written rather than against
    # the constant: no single write may carry the whole 16 KB payload, because
    # that is both a stall and a copy of a buffer that exists precisely so it
    # does not have to be copied.
    biggest = max(len(c) for c in chunks)
    data_writes = [c for c in chunks if len(c) > 8]
    check("no write exceeds CHUNK_BYTES", biggest <= module.CHUNK_BYTES,
          "biggest write %d, limit %d" % (biggest, module.CHUNK_BYTES))
    check("payload went out in several writes", len(data_writes) >= 4,
          "%d writes over 8 bytes" % len(data_writes))

    # The codec must be put back, or the next clip plays 1.5x fast.
    restored = [i for i in module._ES8311.inits if i.get("sample_freq") == module.PLAYBACK_RATE]
    check("codec restored to 24 kHz afterwards", len(restored) == 1,
          "inits=%s" % [i.get("sample_freq") for i in module._ES8311.inits])


def test_round_trip():
    print("device -> host round trip, through the real reader")
    ok, data, module, _chunks = device_bytes(FIXTURE_SECONDS, FIXTURE_RATE)
    host = load_pull_recording()
    count = FIXTURE_COUNT

    # Straightforward case, and then the same stream cut at every boundary the
    # reader could plausibly get wrong.
    magic_at = len(REPL_NOISE)
    cases = {
        "clean": (),
        "split mid-magic": (magic_at + 3,),
        "split mid-header": (magic_at + len(module.MAGIC) + 3,),
        "split mid-data": (magic_at + len(module.MAGIC) + 8 + 777,),
        "split mid-trailer": (len(REPL_NOISE) + len(data) - 3,),
        "split everywhere": tuple(range(magic_at - 2, magic_at + 40, 5)),
    }
    for name, splits in cases.items():
        port = FakePort(REPL_NOISE + data, splits=splits)
        try:
            got_rate, pcm, _elapsed = host.capture(
                port, seconds=FIXTURE_SECONDS, rate=FIXTURE_RATE, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            check("round trip (%s)" % name, False, "%s: %s" % (type(exc).__name__, exc))
            continue
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        check("round trip (%s)" % name,
              got_rate == FIXTURE_RATE and len(samples) == count
              and list(samples) == known_waveform(count),
              "rate=%d samples=%d" % (got_rate, len(samples)))


def test_short_capture():
    """A capture that returns less than asked for must still be a clean stream.

    This is the property that makes `got` rather than `count` the right thing to
    put in the header: the host reads exactly what was captured and still finds
    a valid trailer where it expects one, so a short capture is a short capture
    rather than a desynchronised stream that eats the next command's output.
    """
    print("short capture")
    ok, data, module, _chunks = device_bytes(FIXTURE_SECONDS, FIXTURE_RATE, fail_at="short")
    host = load_pull_recording()
    # A third, matching what the fake Recorder returns. The other 3 is in
    # load_record_stream's ShortRecorder.stop(); see the comment there for why
    # these two stay independent rather than one deriving from the other.
    expected = FIXTURE_COUNT // 3

    got_rate, got_count = struct.unpack("<II", data[len(module.MAGIC):len(module.MAGIC) + 8])
    check("header reports what was captured, not what was asked for",
          got_count == expected, "header says %d, captured %d" % (got_count, expected))

    port = FakePort(REPL_NOISE + data)
    try:
        _rate, pcm, _elapsed = host.capture(
            port, seconds=FIXTURE_SECONDS, rate=FIXTURE_RATE, timeout=5.0)
        check("short capture reads cleanly", len(pcm) == expected * 2,
              "%d bytes for %d samples" % (len(pcm), expected))
    except Exception as exc:  # noqa: BLE001
        check("short capture reads cleanly", False, "%s: %s" % (type(exc).__name__, exc))


def test_contract_constants():
    """The wire markers, and the one invariant that otherwise lives only in prose."""
    print("contract constants")
    module = load_record_stream()
    host = load_pull_recording()

    check("MAGIC identical", module.MAGIC == host.MAGIC,
          "%r vs %r" % (module.MAGIC, host.MAGIC))
    check("TRAILER identical", module.TRAILER == host.TRAILER,
          "%r vs %r" % (module.TRAILER, host.TRAILER))
    check("ERROR identical", module.ERROR == host.ERROR,
          "%r vs %r" % (module.ERROR, host.ERROR))

    # The runtime capture cap must be at least the enrolment window: a word that
    # fitted when it was enrolled has to still fit when it is spoken, or the
    # runtime segment is truncated against a template that is not. Documented in
    # src/listen.py and tools/enrol.py, and enforced nowhere -- record_stream
    # passes an explicit max_samples, so nothing on the device stops the two
    # drifting apart. This is the enforcement.
    real_listen = _load("real_listen", os.path.join(ROOT, "src", "listen.py"))
    enrol = _load("host_enrol", os.path.join(ROOT, "tools", "enrol.py"))
    runtime_s = real_listen.MAX_RECORD_MS / 1000.0
    check("runtime cap >= enrolment window",
          runtime_s >= enrol.DEFAULT_SECONDS,
          "listen.MAX_RECORD_MS=%.1fs, enrol.DEFAULT_SECONDS=%.1fs"
          % (runtime_s, enrol.DEFAULT_SECONDS))
    check("enrolment window within record_stream's ceiling",
          enrol.DEFAULT_SECONDS <= module.MAX_SECONDS,
          "%.1f > %.1f" % (enrol.DEFAULT_SECONDS, module.MAX_SECONDS))

    # One sample rate, named in four places, agreeing by authorship rather than
    # by construction. Capture at one rate and analyse at another and nothing
    # raises: the templates are simply built from audio at a different speed
    # from the queries, every MFCC lands in the wrong mel bin by a constant
    # ratio, and the recogniser is quietly worse for no visible reason.
    #
    # (The device's own SAMPLE_RATE is a default, not a lock -- Recorder takes
    # the rate as a parameter so enrolment can ask for the 24 kHz control. What
    # must agree is the default the runtime uses and the rate the tables were
    # generated for.)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    tables = _load("speech_tables", os.path.join(ROOT, "src", "speech_tables.py"))
    check("runtime capture rate == the rate the DSP tables were built for",
          real_listen.SAMPLE_RATE == tables.SAMPLE_RATE,
          "listen=%d tables=%d" % (real_listen.SAMPLE_RATE, tables.SAMPLE_RATE))
    check("host pull default rate agrees too",
          host.DEFAULT_RATE == tables.SAMPLE_RATE,
          "pull_recording=%d tables=%d" % (host.DEFAULT_RATE, tables.SAMPLE_RATE))

    # What a person is asked to say into the microphone must be what the device
    # can recognise. enrol.py held a hand-written copy of the vocabulary until
    # this check was written, and it had already drifted -- it still listed NO,
    # which vocab.py no longer has, so an enrolment run would have spent five
    # takes on a word nothing can return. It now derives from vocab.FORMS; this
    # pins that it stays derived.
    #
    # FORMS rather than LABELS because enrolment records spoken words: SAD and
    # SICK are two things to say, even though the engine treats them as one
    # class.
    vocab = _load("device_vocab", os.path.join(ROOT, "src", "vocab.py"))
    spoken = [form.upper() for form in vocab.FORMS]
    check("enrolment vocabulary == the device's spoken forms",
          sorted(enrol.VOCABULARY) == sorted(spoken),
          "only in enrol: %s; only in vocab: %s"
          % (sorted(set(enrol.VOCABULARY) - set(spoken)),
             sorted(set(spoken) - set(enrol.VOCABULARY))))
    check("every enrolment word maps back to a vocab label",
          all(vocab.label_of(word.lower()) is not None for word in enrol.VOCABULARY),
          "unmapped: %s" % [w for w in enrol.VOCABULARY
                            if vocab.label_of(w.lower()) is None])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_paths():
    print("error paths")
    module = load_record_stream()
    host = load_pull_recording()

    # Too long: refused before anything is allocated.
    ok, data, module, _chunks = device_bytes(99.0, FIXTURE_RATE)
    check("over-long request refused", ok is False and data.startswith(module.ERROR))

    # Capture that will not start.
    ok, data, module, _chunks = device_bytes(FIXTURE_SECONDS, FIXTURE_RATE, fail_at="start")
    check("failed capture emits ERR, not a traceback",
          ok is False and data.startswith(module.ERROR) and data.endswith(b"\n"))

    port = FakePort(REPL_NOISE + data)
    try:
        host.capture(port, seconds=FIXTURE_SECONDS, rate=FIXTURE_RATE, timeout=5.0)
        check("host surfaces the ERR frame", False, "no exception raised")
    except Exception as exc:  # noqa: BLE001
        check("host surfaces the ERR frame", "device reported" in str(exc),
              "got %r" % str(exc))

    # Truncated data: the trailer is what makes this detectable at all.
    ok, data, module, _chunks = device_bytes(FIXTURE_SECONDS, FIXTURE_RATE)
    truncated = data[:-(len(module.TRAILER) + 200)]
    port = FakePort(REPL_NOISE + truncated)
    try:
        host.capture(port, seconds=FIXTURE_SECONDS, rate=FIXTURE_RATE, timeout=1.0)
        check("truncated stream is rejected", False, "accepted a short read")
    except Exception as exc:  # noqa: BLE001
        check("truncated stream is rejected", True)


def main():
    test_framing()
    test_round_trip()
    test_short_capture()
    test_contract_constants()
    test_error_paths()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
