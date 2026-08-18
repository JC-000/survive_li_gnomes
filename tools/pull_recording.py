#!/usr/bin/env python3
"""Pull one captured utterance off the board over the USB CDC serial link.

Runs on the host; the device end is `src/record_stream.py`. The board records
into a RAM buffer and streams it straight out the wire, so nothing is ever
written to the board's filesystem. That is the entire reason this tool exists
instead of `mpremote cp :rec.wav .` -- see the enrolment note in CLAUDE.md.

    uvx --from pyserial python tools/pull_recording.py out.wav
    PORT=/dev/cu.usbmodem1101 uvx --from pyserial python tools/pull_recording.py out.wav

Not `mpremote`. mpremote's cp sends 256-byte chunks as `repr()`-escaped Python
source over the raw REPL -- roughly 4 characters per byte plus a round trip per
chunk -- and its raw REPL treats 0x04 as the response terminator, so raw PCM
containing that byte truncates the transfer silently. This talks to the friendly
REPL directly with pyserial instead.

The REPL is a shared channel: it echoes the command text back and prints its own
prompts into the same stream. So the reader syncs on a magic header rather than
assuming it knows where the audio starts, and keeps everything it discarded --
a device-side traceback lands there, and it is the first thing worth seeing when
a capture fails.

A clean transfer proves nothing about the microphone. `dma_record_into()` returns
a buffer of constant values when the codec never received BCLK, and it arrives
looking perfect. Every capture is therefore checked for a flat or near-silent
signal, and the tool exits non-zero when it finds one.
"""

import argparse
import math
import os
import struct
import sys
import time
import wave
from array import array

try:
    import serial
except ImportError:  # pragma: no cover - the uvx invocation supplies this
    sys.exit(
        "pyserial not found. Run this as:\n"
        "    uvx --from pyserial python tools/pull_recording.py ..."
    )

# Wire format, agreed with the device side. Count is a number of int16 SAMPLES,
# not bytes. The trailer is what makes a short read distinguishable from a
# device that simply promised more than it sent.
MAGIC = b"\xa5\x5aREC1"
TRAILER = b"\xa5\x5aEND1"
ERROR = b"\xa5\x5aERR1"
HEADER = struct.Struct("<II")  # rate, sample count

DEFAULT_PORT = os.environ.get("PORT", "/dev/cu.usbmodem101")
DEFAULT_RATE = 16000  # whisper/MFCC native rate; the ES8311 is switched to it for capture
DEFAULT_SECONDS = 3.0

# Device-side entry point. src/record_stream.py owns codec init, the 16 kHz mode
# switch and the restore to 24 kHz; the host deliberately knows none of that.
# If that signature changes, this is the only line here that needs to follow.
STREAM_COMMAND = "import record_stream\r\nrecord_stream.stream(seconds=%.3f, rate=%d)\r\n"

# Signal sanity thresholds, against a full scale of 32768.
#
# UNVERIFIED: these are reasoned starting points, not measured against the real
# ES8311 front end -- no board was connected when this was written. SILENCE_RMS
# in particular wants one look at a real "quiet room" capture and a real spoken
# word before it is trusted. Everything here is advisory except FLAT, which is
# unambiguous: a constant buffer is a dead channel, not quiet audio.
SILENCE_RMS = 200
CLIP_FRACTION = 0.001  # 0.1% of samples pinned to the rail is worth mentioning
DC_OFFSET = 1000

# ~500 KB/s is the confirmed ceiling for sys.stdout.buffer.write() on RP2350
# (micropython issue #17398, where a "100 kB/s regression" turned out to be a
# stray high-frequency Timer in the reporter's main.py). A concurrent timer or
# ISR costs about 5x, so a low number here is a real signal, not noise.
SLOW_TRANSFER_KBPS = 150


class CaptureError(Exception):
    """Transport failed. Carries whatever text the device emitted, if any."""

    def __init__(self, message, preamble=b""):
        super().__init__(message)
        self.preamble = preamble


def open_port(port, timeout):
    try:
        return serial.Serial(port, timeout=timeout)
    except serial.SerialException as exc:
        raise CaptureError(
            "cannot open %s: %s\n"
            "Is the board plugged in? After a reset or bootloader the device "
            "disappears and comes back -- wait for it to reappear." % (port, exc)
        )


class _Reader:
    """Buffered reader over the port.

    The pending buffer is the point: scanning for the magic header inevitably
    reads past it, and those bytes are the header and the first of the audio.
    Everything goes through one buffer so they are not dropped on the floor.
    """

    def __init__(self, port):
        self.port = port
        self.pending = bytearray()

    def _fill(self):
        chunk = self.port.read(max(1, getattr(self.port, "in_waiting", 0)))
        if chunk:
            self.pending += chunk
        return bool(chunk)

    def until(self, markers, deadline):
        """Consume through the first of `markers`. Returns (marker, skipped)."""
        while True:
            for marker in markers:
                at = self.pending.find(marker)
                if at >= 0:
                    skipped = bytes(self.pending[:at])
                    del self.pending[: at + len(marker)]
                    return marker, skipped
            if time.monotonic() > deadline:
                raise CaptureError(
                    "timed out waiting for the device to start streaming",
                    bytes(self.pending),
                )
            self._fill()

    def exactly(self, count, deadline):
        while len(self.pending) < count:
            if time.monotonic() > deadline:
                raise CaptureError(
                    "short read: %d of %d bytes before the timeout"
                    % (len(self.pending), count)
                )
            self._fill()
        out = bytes(self.pending[:count])
        del self.pending[:count]
        return out

    def line(self, deadline):
        while b"\n" not in self.pending:
            if time.monotonic() > deadline:
                break
            self._fill()
        at = self.pending.find(b"\n")
        if at < 0:
            at = len(self.pending) - 1
        out = bytes(self.pending[: at + 1])
        del self.pending[: at + 1]
        return out


def capture(port, seconds=DEFAULT_SECONDS, rate=DEFAULT_RATE, timeout=30.0):
    """Trigger one recording and read it back. Returns (rate, pcm, elapsed)."""
    # main.py autoruns at power-on, so there is a running program to interrupt
    # before the REPL will accept anything.
    port.reset_input_buffer()
    port.write(b"\x03\x03")
    time.sleep(0.2)
    port.reset_input_buffer()
    port.write((STREAM_COMMAND % (seconds, rate)).encode())

    # The device holds the line for the length of the recording before the
    # header appears, so the sync deadline has to cover that too.
    deadline = time.monotonic() + timeout + seconds
    reader = _Reader(port)
    marker, preamble = reader.until((MAGIC, ERROR), deadline)

    if marker == ERROR:
        line = reader.line(deadline).decode("utf-8", "replace").strip()
        raise CaptureError("device reported: %s" % (line or "(no detail)"), preamble)

    got_rate, count = HEADER.unpack(reader.exactly(HEADER.size, deadline))
    if count == 0:
        raise CaptureError("device sent a zero-length recording", preamble)
    # Guard against a desynchronised header claiming a nonsense length, which
    # would otherwise park us in _read_exactly until the timeout.
    if count > got_rate * 60:
        raise CaptureError(
            "implausible header: %d samples at %d Hz. Stream is probably "
            "out of sync." % (count, got_rate),
            preamble,
        )

    started = time.monotonic()
    pcm = reader.exactly(count * 2, deadline)
    elapsed = time.monotonic() - started

    trailer = reader.exactly(len(TRAILER), deadline)
    if trailer != TRAILER:
        raise CaptureError(
            "missing end marker -- got %r. The data may be truncated or the "
            "device may have sent more than it promised." % trailer,
            preamble,
        )
    return got_rate, pcm, elapsed


def stats(pcm):
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":  # the wire is little-endian; array() is native
        samples.byteswap()

    n = len(samples)
    lo, hi = min(samples), max(samples)
    total = sum(samples)
    energy = sum(v * v for v in samples)
    clipped = sum(1 for v in samples if v >= 32767 or v <= -32768)
    return {
        "samples": n,
        "min": lo,
        "max": hi,
        "mean": total / n,
        "rms": math.sqrt(energy / n),
        "peak": max(abs(lo), abs(hi)),
        "clipped": clipped,
        "flat": lo == hi,
    }


def problems(st):
    """Return (fatal, advisory) message lists. Fatal means the capture is junk."""
    fatal, advisory = [], []

    if st["flat"]:
        fatal.append(
            "signal is a constant %d -- the codec is not delivering audio. "
            "Check that BCLK/LRCLK reach the PIO and that din_pio_init() ran; "
            "dma_record_into() returns a clean buffer of nothing when it did not."
            % st["min"]
        )
    elif st["rms"] < SILENCE_RMS:
        fatal.append(
            "RMS %.0f is below the %d floor -- this is a noise floor, not speech. "
            "Check mic gain (ES8311.microphone_gain_set) and that you spoke."
            % (st["rms"], SILENCE_RMS)
        )

    if st["clipped"] > st["samples"] * CLIP_FRACTION:
        advisory.append(
            "%d samples (%.2f%%) are pinned to the rail -- reduce mic gain"
            % (st["clipped"], 100.0 * st["clipped"] / st["samples"])
        )
    if abs(st["mean"]) > DC_OFFSET:
        advisory.append(
            "DC offset %.0f is large; MFCC framing will not care, but it "
            "suggests the codec's high-pass filter is off" % st["mean"]
        )
    return fatal, advisory


def write_wav(path, rate, pcm):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def report(path, rate, pcm, elapsed, st, fatal, advisory):
    kbps = len(pcm) / 1024.0 / elapsed if elapsed > 0 else float("inf")
    print("  %s" % path)
    print(
        "  %d samples, %.2f s at %d Hz"
        % (st["samples"], st["samples"] / float(rate), rate)
    )
    print("  %d bytes in %.2f s (%.0f KB/s)" % (len(pcm), elapsed, kbps))
    if kbps < SLOW_TRANSFER_KBPS:
        print(
            "    well under the ~500 KB/s this link can do. A concurrent timer "
            "or ISR costs about 5x -- check what else is running."
        )
    print(
        "  min %d  max %d  peak %d  rms %.0f  dc %.0f"
        % (st["min"], st["max"], st["peak"], st["rms"], st["mean"])
    )
    for note in advisory:
        print("  note: %s" % note)
    for note in fatal:
        print("  FAILED: %s" % note)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest", help="output .wav path")
    ap.add_argument("--port", default=DEFAULT_PORT, help="default $PORT or %s" % DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--rate", type=int, default=DEFAULT_RATE)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--keep-silent",
        action="store_true",
        help="write the file and exit 0 even if the signal looks dead. For "
             "deliberately measuring the noise floor.",
    )
    args = ap.parse_args()

    try:
        port = open_port(args.port, timeout=0.5)
        with port:
            rate, pcm, elapsed = capture(port, args.seconds, args.rate, args.timeout)
    except CaptureError as exc:
        print("capture failed: %s" % exc, file=sys.stderr)
        if exc.preamble:
            text = exc.preamble.decode("utf-8", "replace").strip()
            if text:
                print("\ndevice said:\n%s" % text, file=sys.stderr)
        return 1

    st = stats(pcm)
    fatal, advisory = problems(st)

    # Written before the verdict so a bad capture can still be listened to --
    # hearing what actually arrived is usually how you find out why.
    write_wav(args.dest, rate, pcm)
    report(args.dest, rate, pcm, elapsed, st, fatal, advisory)

    if fatal and not args.keep_silent:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
