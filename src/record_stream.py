"""Record on the board, send the samples to the host over the USB REPL.

The device half of `tools/pull_recording.py`, which is how enrolment recordings
(`tools/enrol.py`) get made: the user speaks to the board, in the board's own
acoustic path, and the audio comes back up the serial line for the host to turn
into templates.

    uvx --from pyserial python tools/pull_recording.py --seconds 3

Why the board records and not the Mac: templates matched on this microphone must
be *enrolled* on this microphone. A word recorded on a laptop and matched
against the ES8311's front end differs in bandwidth, gain and noise before a
single MFCC is computed.

Why the host stores them and not the board: `CLAUDE.md` forbids writing to
flash, and enrolment on-device would mean exactly that. See
docs/speech-design.md#templates-are-recorded-on-the-host-not-on-the-device --
the "enrol mode that writes once" is specifically the thing not to build.

Nothing here writes to the filesystem. The samples go to stdout and are gone.

## It records through listen.Recorder on purpose

Not a second capture path. `listen.Recorder` owns the 16 kHz bring-up, the
pre-allocated buffer and the playback gate, and enrolment goes through all of
it, so a template and a runtime query differ only in who is speaking -- not in
codec configuration, gain, DMA path or rate. Two capture paths would drift, and
the one that drifted would be this one, which is precisely the channel mismatch
the host-side enrolment exists to avoid. **If a later refactor gives this module
its own capture code, that is the bug.**

## Three things that will waste an afternoon

- **Never put a capture in `boot.py`.** Recent MicroPython delays USB
  enumeration until `boot.py` returns, so a blocking capture there means the
  serial port never appears at all and the board looks bricked.
- **`micropython.kbd_intr(-1)` is not needed here** and is not called. Ctrl-C
  interception applies to the *incoming* stream; nothing about writing bytes out
  is affected by it. It is widely cargo-culted around code like this.
- **A clean transfer proves nothing about the microphone.** A codec that never
  received BCLK returns a constant, and the DMA delivers it perfectly happily.
  `tools/pull_recording.py` computes min/max/RMS on every capture and exits
  non-zero on a flat or near-silent signal, which is why that check is not
  duplicated here.

## Wire format

Agreed with tools/pull_recording.py; both sides must change together.

    MAGIC   b"\\xa5\\x5aREC1"
    header  <II    rate, sample count (SAMPLES, not bytes)
    data    count * int16, little-endian, exactly as captured
    TRAILER b"\\xa5\\x5aEND1"

On failure, `b"\\xa5\\x5aERR1"` followed by one line of text and no data. The
trailer is what makes a truncated transfer distinguishable from a device that
promised more than it sent, so it is written even though the count is known.
"""

import sys

import board
import listen

MAGIC = b"\xa5\x5aREC1"
TRAILER = b"\xa5\x5aEND1"
ERROR = b"\xa5\x5aERR1"

# Bounded by RAM, not by patience: int16 at 16 kHz is 32 KB/s against a ~490 KB
# heap, and the array has to be allocated in one contiguous block. 5 s is 160 KB,
# which is already more than any single enrolment word needs.
MAX_SECONDS = 5.0

# USB CDC throughput, measured on RP2040/RP2350 and worth knowing before
# reaching for a "faster" way to do this:
#
#   print()                     30-50 KB/s
#   sys.stdout.write()          ~200 KB/s
#   sys.stdout.buffer.write()   ~600 KB/s   <- this, and nothing else
#
# Budget 500 KB/s, so 3 s of 16 kHz mono (96 KB) is about 0.2 s on the wire.
#
# **A concurrent high-frequency timer or ISR costs about 5x.** The reported
# "RP2350 USB is only 100 KB/s" regression (micropython #17398) turned out to be
# a stray Timer in the reporter's own main.py. Nothing here runs a timer, but
# the capture DMA and the host's own load are real, so tools/pull_recording.py
# prints achieved KB/s on every capture and speech_probe measures it under load.
#
# Chunked rather than one giant write so a stall is a stalled chunk, and so the
# buffer is never copied whole.
CHUNK_BYTES = 4096

# What the Magic 8-Ball leaves the codec configured for. Restored afterwards so
# that a program which was already running does not find its clips playing at
# the wrong speed. See sounds.SAMPLE_RATE, which is where the number lives.
PLAYBACK_RATE = 24000


def _write(data):
    sys.stdout.buffer.write(data)


def _fail(message):
    _write(ERROR)
    _write(message.encode() if isinstance(message, str) else message)
    _write(b"\n")


def stream(seconds=3.0, rate=16000, restore=True):
    """Record `seconds` at `rate` and write the result to stdout.

    Blocking and single-shot. Prints nothing else: anything on stdout between
    the MAGIC and the trailer would land in the middle of the host's samples.
    """
    if seconds <= 0 or seconds > MAX_SECONDS:
        _fail("seconds must be in (0, %.1f]" % MAX_SECONDS)
        return False

    count = int(rate * seconds)
    try:
        recorder = listen.Recorder(rate=rate, max_samples=count)
    except MemoryError:
        _fail("no room for %d samples (%d KB); ask for less" % (count, count * 2 // 1024))
        return False

    i2c = None
    try:
        i2c = board.bus()

        # The same "listening" tone the ELIZA program uses. Enrolment is a human
        # sitting there saying words on cue, and without an audible start they
        # are guessing when the window opened -- which is precisely how the
        # first speech measurements on this board came back as silence. It
        # blocks until the speaker is quiet, so it never lands in the recording.
        recorder.chirp(i2c)

        if not recorder.start(i2c):
            _fail("capture did not start; see listen.Recorder")
            return False

        # A plain blocking wait. There is no VAD here on purpose -- the host
        # endpoints the recording with tools/vad.py, and a device-side trim
        # would mean the template was cut by different code from the runtime
        # utterance.
        while not recorder.full():
            pass
        got = recorder.stop()
    except Exception as exc:  # noqa: BLE001
        _fail("%s: %s" % (type(exc).__name__, exc))
        return False

    # int16 little-endian on both sides: this is a little-endian MCU talking to
    # a little-endian host, and the format is defined as the raw capture.
    #
    # **The count is `got`, what was captured -- never `count`, what was asked
    # for.** They are usually equal and it looks like a simplification to use the
    # requested figure. It is not: a capture that came up short would then
    # promise more than it sends, the host would read past the end of the data
    # looking for the rest, and the trailer would land in the middle of what it
    # thought was audio. Every subsequent command's output would arrive
    # desynchronised, intermittently, depending on how short the capture was.
    # With `got`, a short capture is simply a short capture and the trailer is
    # exactly where the host expects it. tools/test_record_stream.py has a case
    # for this.
    _write(MAGIC)
    _write(bytes([
        rate & 0xFF, (rate >> 8) & 0xFF, (rate >> 16) & 0xFF, (rate >> 24) & 0xFF,
        got & 0xFF, (got >> 8) & 0xFF, (got >> 16) & 0xFF, (got >> 24) & 0xFF,
    ]))

    # A memoryview, so a chunk is a window onto the capture buffer rather than a
    # copy of it -- 96 KB of copies would defeat the point of pre-allocating.
    data = memoryview(recorder.buf)
    sent = 0
    while sent < got:
        chunk = CHUNK_BYTES // 2  # samples, not bytes
        if sent + chunk > got:
            chunk = got - sent
        _write(data[sent:sent + chunk])
        sent += chunk
    _write(TRAILER)

    if restore:
        # Leave the codec as the 8-Ball expects to find it. Cheap -- a handful
        # of I2C writes -- and it stops a later playback running at two-thirds
        # speed for a reason nothing in shake.py could explain. Best effort: the
        # host already has its audio by this point, so a failure here is not
        # worth reporting up a stream that has already ended.
        try:
            recorder.close()
            import es8311

            es8311.ES8311(i2c).init(
                mclk_freq=PLAYBACK_RATE * 256,
                sample_freq=PLAYBACK_RATE,
                res_in=16,
                res_out=16,
                volume=0,
                mic_gain=0,
            )
        except Exception:  # noqa: BLE001
            pass
    return True
