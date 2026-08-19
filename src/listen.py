"""Microphone capture for the push-to-talk ELIZA program.

16 kHz mono int16, off the ES8311's ADC over PIO + DMA. Nothing here decides
*what* was said -- it only fills a buffer and says how much of it is real.

Why 16 kHz and not the 24 kHz the microphone was first verified at: every speech
model and dataset in the field is 16 kHz, so matching it avoids a resampler
entirely, and the buffer drops by a third (1 s = 32 KB instead of 48 KB) on a
board with a ~490 KB heap. The ES8311 supports 8-48 kHz natively and
`es8311.COEFF_DIV` already carries the row this needs:

    [4096000, 16000, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0xff, 0x04, 0x10, 0x10]

i.e. MCLK = 256 * fs = 4 096 000, BCLK = MCLK / 4 = 1.024 MHz = 64 fs.

**Verified on the hardware**: 15990 Hz measured against wall clock, -0.1 %.
Worth measuring rather than assuming, because the register write can succeed
while the rate does not change and the RX path returns samples either way -- "no
exception raised" proves nothing here, exactly as it proves nothing for the
panel. Re-measure with `tools/speech_probe.py` section (a), which counts samples
against wall-clock time, if MCLK or the `COEFF_DIV` row ever changes.

Two things that will bite silently if changed:

- **Never record while the board is playing.** Playback and capture share one
  codec, and the microphone hears the speaker. `Recorder.start` refuses while a
  `playing` callback says audio is running; `vad` has the matching gate.
- **The recording buffer is allocated once, in the constructor.** 96 KB is a big
  contiguous block, MicroPython's heap never compacts, and `array("h",
  bytearray(n))` holds the bytearray *and* the array at once, so the transient
  peak is twice the final size. `sounds.allocate_bytes` documents the same trap
  the hard way. Build the Recorder before anything else large.
"""

import time
from array import array

import board

SAMPLE_RATE = 16000

# MCLK is always 256 * the sample rate -- 4 096 000 at 16 kHz -- which is what
# the COEFF_DIV row above is indexed by. Derived at bring-up rather than kept as
# a constant, because Recorder takes the rate as a parameter (see __init__).

# A cough, a sleeve across the screen or a stuck touch line must not be able to
# eat the heap, so capture is hard-capped: 48 000 samples, 96 KB.
#
# This number is not free to choose. It must stay **at or above**
# `tools/enrol.py`'s DEFAULT_SECONDS (2.0 s), which is the window each template
# is recorded in -- a word that fitted when it was enrolled has to still fit
# when it is spoken, or the runtime segment is truncated against a template that
# is not. If you lower this, raise that, and vice versa.
#
# The *utterance* is capped separately and much lower, at
# `vad.MAX_SPEECH_FRAMES` (1.8 s), which is what the recogniser actually sees.
# This cap is only about how much RAM one press may consume.
MAX_RECORD_MS = 3000
MAX_SAMPLES = SAMPLE_RATE * MAX_RECORD_MS // 1000

# Written to ES8311 REG16. Not decibels: docs/hardware.md records the noise floor
# at around -44 dBFS with gain 0, and gain 6 railing the input outright, so the
# units are whatever the vendor's table means.
#
# 3 is the vendor default and was the setting here first. Measured at the bench
# it is wrong for this board: at gain 3 every real utterance pinned full scale --
# 849 clipped runs across ten words, mean run 2-3 samples, which is a waveform
# touching the rail at its peaks rather than the isolated glitches seen in
# silence. The noise floor scales cleanly with this setting (IMN 323/137/75/36
# at gains 3/2/1/0), so dropping two steps buys ~4x of headroom and costs no
# signal-to-noise: the room scales with the speech. It also costs the endpointer
# nothing, because IMX/IMN is gain-invariant -- scaling every sample cancels.
MIC_GAIN = 1

# Playback is not used by the ELIZA program (see talk.py), but the codec still
# has a DAC and a power amp attached, so the amp is explicitly dropped.
DAC_VOLUME = 0

# Activation chirp. The board has no vibration motor and no indicator LED, and
# e-paper cannot acknowledge anything in under ~583 ms, so a short tone through
# the speaker is the only instant "I am listening" this hardware can give. It
# plays *before* capture starts and is waited out, because the microphone and
# the speaker share this codec and would otherwise record it.
#
# 300 ms at 500 Hz, both chosen at the bench against this speaker. The first
# attempt was 90 ms at 1200 Hz and was inaudible to the person holding the
# board -- not muted, not quiet, simply too brief to register as a sound. The
# duration is dead time on every press, so do not lengthen it casually; 300 ms
# sits against a panel that already takes ~600 ms to redraw.
CHIRP_MS = 300
CHIRP_HZ = 500
CHIRP_PEAK = 24000    # of 32767. 9000 was measured audible-but-easy-to-miss on
                      # the bench; the sampled clips run at 30000 for comparison.
# How long to hold *after* the tone before opening the microphone.
# `play_finished()` reports that the DMA has drained into the PIO FIFO, not that
# the speaker has stopped moving -- so returning on it alone let up to ~300 ms of
# tone and decay land at the head of every capture. That poisons the endpointer
# specifically: it calibrates its noise floor from the first frames, so a loud
# head raises the floor until the real word cannot clear the start threshold.
# Measured on this board: the tone rings ~180 ms and decays to the noise floor
# over a further ~140 ms. 14 of 22 real recordings were rejected before this.
CHIRP_SETTLE_MS = 140

CHIRP_VOLUME = 90     # ES8311 volume is dB-ish; matches shake.py's DAC_VOLUME.
                      # Restored to DAC_VOLUME, and re-muted, after the tone.


def allocate_samples(count):
    """Reserve a capture buffer of `count` int16 samples.

    Split out so the transient double-size allocation happens somewhere with a
    name, and early. See the module docstring.
    """
    return array("h", bytearray(2 * count))


def _build_chirp(rate):
    """The activation tone, as the packed 32-bit stereo words the PIO wants.

    A square wave because it is integer-only and carries further through a small
    speaker than a sine of the same peak. The ends are faded over an eighth of
    the tone: a square starting at full amplitude clicks, and a click is exactly
    what this is meant not to sound like.
    """
    frames = rate * CHIRP_MS // 1000
    half = max(1, rate // (2 * CHIRP_HZ))
    fade = max(1, frames // 8)
    buf = array("I", bytearray(4 * frames))
    for i in range(frames):
        v = CHIRP_PEAK if (i // half) & 1 else -CHIRP_PEAK
        if i < fade:
            v = v * i // fade
        elif i >= frames - fade:
            v = v * (frames - 1 - i) // fade
        v &= 0xFFFF
        buf[i] = (v << 16) | v
    return buf


class Recorder:
    """The microphone: codec bring-up, one long DMA capture, early stop.

    Capture is *one* DMA transfer into the whole buffer, started and then left
    running, rather than a sequence of short blocking `dma_record_into` calls.
    Each of those restarts the RX state machine and discards the FIFO
    (`AudioPIO._restart_rx`), which would punch a gap into the audio at every
    chunk boundary -- roughly a click every 100 ms through the middle of the
    speech we are trying to recognise.
    """

    def __init__(self, rate=SAMPLE_RATE, max_samples=None, playing=None):
        # `rate` is a parameter only because enrolment recordings go through
        # this same class (see record_stream.py) and the host may ask for the
        # 24 kHz the microphone was originally verified at, as a control. The
        # ELIZA program always uses 16 kHz.
        self.rate = rate
        if max_samples is None:
            max_samples = rate * MAX_RECORD_MS // 1000
        # Allocated up front and reused for every utterance: see the module
        # docstring on why this must not be a per-press allocation.
        self.buf = allocate_samples(max_samples)
        self.max_samples = max_samples

        # Callable returning True while the board is making noise. Capture is
        # refused then -- the microphone would record the speaker.
        self._playing = playing

        self._audio = None
        self._codec = None
        self._pa = None
        self._dma = None
        self._chirp = None
        self._started_us = 0
        self._recording = False
        self._final_count = 0
        self.available = None  # None = untried, True/False once known

    # --- setup -------------------------------------------------------------

    def _setup(self, i2c):
        import audio_pio_mpy
        import es8311

        # Power amp enable, held low. Nothing plays during the ELIZA program,
        # and an amp left on is a constant drain plus an acoustic path back into
        # the microphone.
        self._pa = board.Pin(board.CODEC_PA_CTRL_PIN, board.Pin.OUT, value=0)

        self._codec = es8311.ES8311(i2c)
        self._codec.init(
            mclk_freq=self.rate * 256,
            sample_freq=self.rate,
            res_in=16,
            res_out=16,
            volume=DAC_VOLUME,
            mic_gain=MIC_GAIN,
        )

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
        self._audio.mclk_freq = self.rate * 256
        self._audio.sample_freq = self.rate
        self._audio.channel_count = 1
        self._audio.rx_channel = 0

        # AudioPIO's constructor only stores configuration -- the state machines
        # stay None until these are called, and start() silently does nothing
        # without them. The mirror image of shake.py, which brings up MCLK and
        # DOUT and deliberately skips DIN: here DOUT is the one not worth a state
        # machine, because nothing is played. The state machine IDs match
        # shake.py's so the two can coexist if that ever changes.
        self._audio.mclk_pio_init()
        self._audio.din_pio_init()
        self._audio.start()

    def _ensure(self, i2c):
        if self.available is False:
            return False
        if self._audio is not None:
            return True
        try:
            self._setup(i2c)
            self.available = True
            return True
        except Exception as exc:  # noqa: BLE001
            print("microphone unavailable (%s: %s)" % (type(exc).__name__, exc))
            self.available = False
            return False

    # --- capture -----------------------------------------------------------
    #
    # The DMA plumbing below is `AudioPIO.dma_record_into` with its terminal
    # `while self.dma_rx.active(): pass` removed, the same deviation
    # `dma_play_words_async` already makes on the playback side. It lives here
    # rather than in audio_pio_mpy.py because that file is vendored and shipped
    # to the Magic 8-Ball; this program does not get to change what the 8-Ball
    # runs. If it ever earns its place there, move it and delete this.

    def _configure_dma(self, count):
        import rp2

        audio = self._audio
        if audio.dma_rx is None:
            audio.dma_rx = rp2.DMA()
        self._dma = audio.dma_rx

        audio._restart_rx()
        ctrl = audio._dma_pack_ctrl(
            self._dma,
            size=1,  # 16-bit transfers: the RX path is int16 mono
            inc_read=False,
            inc_write=True,
            treq_sel=audio._pio_dreq(audio.sm_din_id, False),
            high_pri=True,
            bswap=False,
        )
        self._dma.config(
            read=audio.sm_din,
            write=self.buf,
            count=count,
            ctrl=ctrl,
            trigger=True,
        )

    def chirp(self, i2c):
        """Play the activation tone and wait it out. Returns True if it sounded.

        Blocking on purpose. Capture must not begin until the speaker is silent
        again -- the microphone hears it, and a 70 ms tone at the head of every
        utterance would be enrolled into every template as if it were speech.

        Brings up the DOUT state machine on first use. `_setup` deliberately
        does not, because the ELIZA program otherwise plays nothing; the state
        machine IDs were chosen to match shake.py so this could be added without
        disturbing anything.

        Never raises: a silent chirp is a worse toy, not a broken one, so a
        failure here must not stop the board listening.
        """
        if not self._ensure(i2c):
            return False
        try:
            if self._chirp is None:
                self._audio.dout_pio_init()
                self._chirp = _build_chirp(self.rate)
            # The DAC comes out of reset muted and `es8311.init()` does not
            # clear it -- shake.py:84 unmutes and this module never did, because
            # until now it only ever recorded. Without this the chirp plays into
            # a muted output and reports success: no exception, no sound. That
            # is the failure mode CLAUDE.md exists for, and it cost a bench
            # session here.
            self._codec.mute(False)
            self._codec.volume_set(CHIRP_VOLUME)
            self._pa.value(1)
            started = time.ticks_ms()
            self._audio.dma_play_words_async(self._chirp)
            while not self._audio.play_finished():
                time.sleep_ms(2)
            # Wait out the whole tone and its decay, timed from the start of
            # playback rather than from the DMA finishing -- the DMA drains
            # early and is not a proxy for silence.
            while time.ticks_diff(time.ticks_ms(), started) < CHIRP_MS + CHIRP_SETTLE_MS:
                time.sleep_ms(5)
            return True
        except Exception as exc:  # noqa: BLE001
            print("chirp failed (%s: %s)" % (type(exc).__name__, exc))
            return False
        finally:
            # Drop the amp and the volume whatever happened: an amp left on is
            # both a battery drain and an open acoustic path into the capture.
            try:
                self._pa.value(0)
                self._codec.volume_set(DAC_VOLUME)
                # Re-mute for the same reason the amp is dropped: the speaker is
                # an open acoustic path back into the microphone we are about to
                # record with.
                self._codec.mute(True)
            except Exception:  # noqa: BLE001
                pass

    def start(self, i2c):
        """Begin capturing into the buffer. Returns True if recording started.

        Refuses while `playing()` is true: the microphone and the speaker are on
        the same codec, so recording during playback records the board itself.
        """
        if self._playing is not None and self._playing():
            print("refusing to record while audio is playing")
            return False
        if not self._ensure(i2c):
            return False
        try:
            self._configure_dma(self.max_samples)
            self._started_us = time.ticks_us()
            self._recording = True
            return True
        except Exception as exc:  # noqa: BLE001
            print("could not start capture (%s: %s)" % (type(exc).__name__, exc))
            self.available = False
            self._recording = False
            return False

    def captured(self):
        """How many samples are safely in the buffer right now.

        Prefers the DMA's own remaining-transfer count, which is exact. If that
        readback is not supported by this firmware it falls back to elapsed time
        at the *nominal* rate, deliberately one 10 ms frame behind -- a caller
        must never read ahead of the DMA and mistake untouched zeros for silence
        it can act on. The fallback is only as good as the assumption that the
        codec really is at 16 kHz, which is what speech_probe measures.
        """
        if not self._recording:
            return self._final_count
        try:
            remaining = int(self._dma.count)
            captured = self.max_samples - remaining
        except Exception:  # noqa: BLE001 -- count readback is firmware-dependent
            elapsed_us = time.ticks_diff(time.ticks_us(), self._started_us)
            captured = elapsed_us * self.rate // 1_000_000 - self.rate // 100
        if captured < 0:
            return 0
        if captured > self.max_samples:
            return self.max_samples
        return captured

    def full(self):
        return self.captured() >= self.max_samples

    def stop(self):
        """Stop capturing and return the number of valid samples.

        The samples themselves stay in `self.buf`; nothing is copied, because a
        copy of up to 96 KB is exactly the allocation this module exists to
        avoid.
        """
        if not self._recording:
            return self._final_count
        count = self.captured()
        try:
            self._dma.active(0)
        except Exception:  # noqa: BLE001
            # Some firmware will not stop a channel this way. Closing it is
            # heavier -- the next capture builds a fresh one -- but it is certain.
            try:
                self._dma.close()
            except Exception:  # noqa: BLE001
                pass
            self._audio.dma_rx = None
            self._dma = None
        self._recording = False
        self._final_count = count
        return count

    def elapsed_ms(self):
        return time.ticks_diff(time.ticks_us(), self._started_us) // 1000

    def close(self):
        """Release the state machines and DMA channel. Not used by talk.py.

        The main loop keeps the codec up between utterances -- bring-up costs
        tens of milliseconds of I2C writes and there is no press-time budget for
        it -- so this exists for probes and for anything that needs the PIO back.
        """
        try:
            if self._audio is not None:
                self._audio.stop()
        except Exception as exc:  # noqa: BLE001
            print("audio stop failed (%s: %s)" % (type(exc).__name__, exc))
        finally:
            self._audio = None
            self._dma = None
            self._recording = False
