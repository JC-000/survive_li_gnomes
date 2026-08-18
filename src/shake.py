"""The sound of a Magic 8-Ball being shaken, played through the ES8311.

Synthesised on the device rather than shipped as a WAV: three decaying bursts of
low-passed noise, which reads as a die tumbling in liquid. A 0.5 s clip costs
~48 KB as int16 stereo, versus a file on a 3 MB filesystem plus the read.

Audio is strictly optional. Every entry point here swallows its own errors --
this is a Magic 8-Ball, and a silent one still works. Nothing in this module may
be allowed to stop the answer appearing.
"""

from array import array
import time

import board

SAMPLE_RATE = 24000
MCLK_FREQ = SAMPLE_RATE * 256
CHANNELS = 2
DAC_VOLUME = 90  # ES8311 volume is dB-based; 80-100 is the sane range

_BURSTS = 3
_DURATION_MS = 540


def _generate():
    """Three decaying noise bursts as packed 32-bit stereo words.

    The PIO program pulls **one 32-bit word per stereo frame** and shifts out
    bits 31..16 as left, then 15..0 as right. So the buffer is one word per
    frame, not two int16s -- feeding it int16s makes each frame consume half a
    frame's worth of audio and the clip plays an octave low at double length.

    Integer-only: MicroPython floats here would roughly triple generation time
    for ~13k frames, and none of this needs the precision.
    """
    frames = SAMPLE_RATE * _DURATION_MS // 1000
    buf = array("I", bytearray(4 * frames))

    burst = frames // _BURSTS
    seed = 0x1234567
    low = 0

    for i in range(frames):
        # xorshift32 -- os.urandom would be far too slow per-sample here.
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= seed >> 17
        seed ^= (seed << 5) & 0xFFFFFFFF
        white = (seed & 0xFFFF) - 32768

        # One-pole low-pass: bright white noise sounds like static, not liquid.
        low += (white - low) >> 3

        # Envelope, 0..256: fast attack, exponential-ish decay, per burst.
        pos = i % burst
        if pos < burst >> 4:
            env = (pos << 8) // max(1, burst >> 4)
        else:
            remaining = burst - pos
            env = (remaining << 8) // burst
            env = (env * env) >> 8  # square it for a snappier tail

        # Later bursts quieter, as if the ball is settling.
        env = (env * (256 - (i // burst) * 60)) >> 8

        # >>7 puts the peak near 31k of 32767. The low-pass above costs a lot of
        # amplitude, so a gentler shift here leaves the clip inaudibly quiet.
        sample = (low * env) >> 7
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768

        half = sample & 0xFFFF  # two's complement int16 in the low 16 bits
        buf[i] = (half << 16) | half  # same signal to left and right

    return buf


class Shaker:
    """Lazily brings up the codec, then plays the shake clip on demand."""

    def __init__(self):
        self._audio = None
        self._codec = None
        self._clip = None
        self._pa = None
        self.available = None  # None = untried, True/False once known

    def _setup(self, i2c):
        import audio_pio_mpy
        import es8311

        # Power amp enable. Held low except while playing -- leaving it on is a
        # constant drain on a battery device.
        self._pa = board.Pin(board.CODEC_PA_CTRL_PIN, board.Pin.OUT, value=0)

        self._codec = es8311.ES8311(i2c)
        self._codec.init(
            mclk_freq=MCLK_FREQ,
            sample_freq=SAMPLE_RATE,
            res_in=16,
            res_out=16,
            volume=DAC_VOLUME,
            mic_gain=0,
        )
        self._codec.mute(False)

        self._audio = audio_pio_mpy.AudioPIO(
            mclk_pin=board.CODEC_MCLK_PIN,
            dout_pin=board.CODEC_DOUT_PIN,
            din_pin=board.CODEC_DIN_PIN,
            lrclk_pin=board.CODEC_LRCLK_PIN,
            bclk_pin=board.CODEC_BCLK_PIN,
            sm_dout_id=0,
            sm_din_id=5,
            sm_mclk_id=2,
        )
        self._audio.mclk_freq = MCLK_FREQ
        self._audio.sample_freq = SAMPLE_RATE
        self._audio.channel_count = CHANNELS
        self._audio.rx_channel = 0

        # AudioPIO's constructor only stores config -- the state machines stay
        # None until these are called, and start() silently does nothing without
        # them. din_pio_init() is skipped deliberately: the microphone is unused,
        # so there is no reason to burn a state machine on it.
        self._audio.mclk_pio_init()
        self._audio.dout_pio_init()
        self._audio.start()

        self._clip = _generate()

    def start(self, i2c):
        """Begin playback and return immediately. Pair with finish().

        Split from a blocking play() so the shake overlaps the e-paper refresh
        instead of preceding it -- the DMA engine feeds the codec on its own
        while the CPU drives SPI.
        """
        if self.available is False:
            return False
        try:
            if self._audio is None:
                self._setup(i2c)
                self.available = True
            self._pa.value(1)
            self._audio.dma_play_words_async(self._clip)
            return True
        except Exception as exc:  # noqa: BLE001 -- audio must never break the ball
            print("audio unavailable (%s: %s)" % (type(exc).__name__, exc))
            self.available = False
            self._drop_pa()
            return False

    def finish(self):
        """Wait out any remaining audio, then drop the power amp.

        Normally a no-op: the clip is ~0.54 s and the refresh it overlaps is
        ~2.6 s, so playback has long finished by the time this is called.
        """
        try:
            if self._audio is not None:
                while not self._audio.play_finished():
                    time.sleep_ms(5)
        except Exception as exc:  # noqa: BLE001
            print("audio finish failed (%s: %s)" % (type(exc).__name__, exc))
        finally:
            self._drop_pa()

    def _drop_pa(self):
        if self._pa is not None:
            self._pa.value(0)

    def play(self, i2c):
        """Blocking play. Kept for probes and tests; main.py uses start/finish."""
        if not self.start(i2c):
            return False
        self.finish()
        return True
