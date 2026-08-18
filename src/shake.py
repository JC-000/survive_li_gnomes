"""Press sounds, played through the ES8311.

Normally the shake. Occasionally -- no more often than once every
`ALTERNATE_MIN_GAP` presses -- an alternate instead: a synthesised fart, or a
sampled laugh read off the filesystem.

Synthesis lives in `sounds`; this module owns the codec, the clip cache and the
choice of what to play.

Audio is strictly optional. Every entry point here swallows its own errors --
this is a Magic 8-Ball, and a silent one still works. Nothing in this module may
be allowed to stop the answer appearing.
"""

import gc
import os
import time

import board
import sounds

MCLK_FREQ = sounds.SAMPLE_RATE * 256
CHANNELS = 2
DAC_VOLUME = 90  # ES8311 volume is dB-based; 80-100 is the sane range

# Easter-egg policy. At least ALTERNATE_MIN_GAP ordinary shakes must pass
# before an alternate can fire again, and then it is a 1-in-ALTERNATE_ONE_IN
# roll each press -- so the average gap is around 8, never below 5.
# For "guaranteed at least once every N", set ALTERNATE_ONE_IN = 1 instead.
ALTERNATE_MIN_GAP = 5
ALTERNATE_ONE_IN = 3
ALTERNATES = ("fart", "laugh")

# Alternates backed by a sampled file rather than synthesis. Built on the host by
# tools/make_clip.py, which emits exactly the packed format the PIO consumes, so
# the device only has to read bytes -- no decoding, no parsing.
SAMPLE_CLIPS = {"laugh": "laugh.raw"}


def _rand_below(n):
    """Uniform random int in [0, n), via the RP2350 hardware RNG.

    Duplicated from magic8 rather than imported: keeps the audio module
    independent of the answer module, and it is six lines.
    """
    limit = 256 - (256 % n)
    while True:
        value = os.urandom(1)[0]
        if value < limit:
            return value % n


class Shaker:
    """Owns the codec and the clip cache; picks and plays a press sound."""

    def __init__(self):
        self._audio = None
        self._codec = None
        self._pa = None
        self._clips = {}      # name -> filled, playable clip
        self._buffers = {}    # name -> reserved but not yet generated
        self._since_alternate = 0
        self.available = None  # None = untried, True/False once known

    # --- setup -------------------------------------------------------------

    def _setup(self, i2c):
        import audio_pio_mpy
        import es8311

        # Power amp enable. Held low except while playing -- leaving it on is a
        # constant drain on a battery device.
        self._pa = board.Pin(board.CODEC_PA_CTRL_PIN, board.Pin.OUT, value=0)

        self._codec = es8311.ES8311(i2c)
        self._codec.init(
            mclk_freq=MCLK_FREQ,
            sample_freq=sounds.SAMPLE_RATE,
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
        self._audio.sample_freq = sounds.SAMPLE_RATE
        self._audio.channel_count = CHANNELS
        self._audio.rx_channel = 0

        # AudioPIO's constructor only stores config -- the state machines stay
        # None until these are called, and start() silently does nothing without
        # them. din_pio_init() is skipped deliberately: the microphone is unused,
        # so there is no reason to burn a state machine on it.
        self._audio.mclk_pio_init()
        self._audio.dout_pio_init()
        self._audio.start()

        # Reserve every alternate's buffer before generating anything -- largest
        # first, and *before* the shake clip. MicroPython's heap never compacts,
        # and allocating an array needs a transient block twice its final size,
        # so this ordering is what makes the largest clip fit at all.
        wanted = {}
        for name in ALTERNATES:
            if name in SAMPLE_CLIPS:
                try:
                    wanted[name] = os.stat(SAMPLE_CLIPS[name])[6]
                except OSError:
                    print("sample '%s' missing (%s); skipping"
                          % (name, SAMPLE_CLIPS[name]))
            else:
                wanted[name] = 4 * sounds.output_frames(sounds.ALTERNATE_MS[name])

        for name in sorted(wanted, key=lambda n: -wanted[n]):
            gc.collect()
            try:
                self._buffers[name] = sounds.allocate_bytes(wanted[name])
            except MemoryError:
                print("no room to reserve '%s' (%d bytes); it will be skipped"
                      % (name, wanted[name]))

        # Sampled clips load now rather than from prepare_next: reading ~140 KB
        # off the filesystem takes tens of milliseconds, unlike synthesis.
        for name, path in SAMPLE_CLIPS.items():
            if name not in self._buffers:
                continue
            try:
                with open(path, "rb") as handle:
                    handle.readinto(self._buffers[name])
                self._clips[name] = self._buffers.pop(name)
            except OSError as exc:
                print("could not read '%s' (%s)" % (path, exc))
                del self._buffers[name]

        gc.collect()
        self._clips["shake"] = sounds.shake()

    def prepare_next(self):
        """Generate one not-yet-built alternate. Call while idle.

        Synthesis is slow enough to be felt (~1.0 s for the fart), so it is done
        between presses rather than during one. Sampled clips are already loaded
        by _setup and never appear here. ALTERNATE_MIN_GAP guarantees everything
        is ready long before the first alternate can fire.

        Returns True if it generated something, so a caller can tell idle work
        from a no-op.
        """
        if self._audio is None:
            return False
        for name in ALTERNATES:
            if name in self._clips or name not in self._buffers:
                continue
            try:
                started = time.ticks_ms()
                self._clips[name] = getattr(sounds, name)(out=self._buffers[name])
                print("prepared '%s' (%d ms)"
                      % (name, time.ticks_diff(time.ticks_ms(), started)))
            except Exception as exc:  # noqa: BLE001
                print("could not build '%s' (%s: %s)"
                      % (name, type(exc).__name__, exc))
                self._clips[name] = None  # don't retry forever
            finally:
                del self._buffers[name]
            return True
        return False

    # --- choosing ----------------------------------------------------------

    def _choose(self):
        ready = [n for n in ALTERNATES if self._clips.get(n) is not None]
        if ready and self._since_alternate >= ALTERNATE_MIN_GAP:
            if _rand_below(ALTERNATE_ONE_IN) == 0:
                self._since_alternate = 0
                return ready[_rand_below(len(ready))]
        self._since_alternate += 1
        return "shake"

    # --- playing -----------------------------------------------------------

    def start(self, i2c):
        """Begin playback, return the clip name (or None if audio is dead).

        Non-blocking, so the sound overlaps the e-paper refresh instead of
        preceding it -- the DMA engine feeds the codec while the CPU drives SPI.
        Pair with finish().
        """
        if self.available is False:
            return None
        try:
            if self._audio is None:
                self._setup(i2c)
                self.available = True
            name = self._choose()
            self._pa.value(1)
            self._audio.dma_play_words_async(self._clips[name])
            return name
        except Exception as exc:  # noqa: BLE001 -- audio must never break the ball
            print("audio unavailable (%s: %s)" % (type(exc).__name__, exc))
            self.available = False
            self._drop_pa()
            return None

    def finish(self):
        """Wait out any remaining audio, then drop the power amp.

        Roughly a wash for the shake (0.54 s against a ~600 ms partial refresh),
        but the laugh is 1.5 s and will still be running, so this genuinely waits
        -- a press that draws the laugh takes about as long as the laugh.
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
        name = self.start(i2c)
        if name is None:
            return False
        self.finish()
        return True
