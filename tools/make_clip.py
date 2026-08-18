#!/usr/bin/env python3
"""Convert an audio file into a raw clip the board can DMA straight to the codec.

Runs on the host, not the device: MicroPython has no MP3 decoder, and decoding
one on a 150 MHz M33 would be far slower than real time.

    uvx --from miniaudio python tools/make_clip.py in.mp3 clips/laugh.raw --seconds 1.5

Output format is what the audio PIO consumes directly: one little-endian 32-bit
word per stereo frame, left in bits 31..16 and right in 15..0, at 24 kHz. The
device reads it into a pre-reserved buffer with readinto() -- no decoding, no
parsing, no extra allocation.

RAM is the binding constraint: 24 kHz packed stereo costs 96 KB per second
against a ~490 KB heap that also holds the other clips. Hence --seconds, and
hence the automatic search for the most energetic window rather than just taking
the start of the file.
"""

import argparse
import math
import struct
import sys

SAMPLE_RATE = 24000
# Default peak, matching sounds._PEAK. Override with --peak: normalising a quiet
# source up to near full scale can overdrive the amp and small speaker even
# though nothing clips digitally. The laughter sample needed -6 dB.
PEAK = 30000
FADE_MS = 8  # cutting mid-waveform clicks; a short fade hides the splice


def best_window(samples, want, bucket=2400):
    """Offset of the highest-energy window of `want` samples."""
    if len(samples) <= want:
        return 0
    energy = []
    for i in range(0, len(samples), bucket):
        chunk = samples[i:i + bucket]
        energy.append(sum(v * v for v in chunk) // max(1, len(chunk)))
    span = max(1, want // bucket)
    best_at, best_sum = 0, -1
    for i in range(0, max(1, len(energy) - span + 1)):
        total = sum(energy[i:i + span])
        if total > best_sum:
            best_sum, best_at = total, i
    return min(best_at * bucket, len(samples) - want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--start", type=float, default=None,
                    help="start offset in seconds; default picks the loudest window")
    ap.add_argument("--peak", type=int, default=PEAK,
                    help="normalise to this peak (max 32767). Lower it if playback "
                         "sounds overdriven; nothing clips digitally below 32767, "
                         "but the amp and speaker can still be driven too hard.")
    args = ap.parse_args()

    import miniaudio

    decoded = miniaudio.decode_file(
        args.source,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=SAMPLE_RATE,
    )
    samples = list(decoded.samples)
    want = int(args.seconds * SAMPLE_RATE)

    start = int(args.start * SAMPLE_RATE) if args.start is not None \
        else best_window(samples, want)
    clip = samples[start:start + want]
    if len(clip) < want:
        clip += [0] * (want - len(clip))

    peak = max(abs(v) for v in clip) or 1
    gain = args.peak / peak

    fade = int(FADE_MS * SAMPLE_RATE / 1000)
    out = bytearray()
    for i, value in enumerate(clip):
        scaled = int(value * gain)
        if i < fade:
            scaled = scaled * i // fade
        elif i >= len(clip) - fade:
            scaled = scaled * (len(clip) - i) // fade
        scaled = max(-32768, min(32767, scaled))
        half = scaled & 0xFFFF
        out += struct.pack("<I", (half << 16) | half)

    with open(args.dest, "wb") as handle:
        handle.write(out)

    rms = int(math.sqrt(sum(v * v for v in clip) / len(clip)))
    print("%s -> %s" % (args.source, args.dest))
    print("  source %.2f s, took %.2f s from %.2f s"
          % (len(samples) / SAMPLE_RATE, len(clip) / SAMPLE_RATE, start / SAMPLE_RATE))
    print("  peak %d -> %d (gain x%.2f), rms %d" % (peak, args.peak, gain, rms))
    print("  %d frames, %d bytes (%d KB) of RAM on device"
          % (len(clip), len(out), len(out) // 1024))


if __name__ == "__main__":
    sys.exit(main())
