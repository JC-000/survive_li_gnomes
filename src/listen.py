"""Microphone capture, and the voice, for the push-to-talk ELIZA program.

16 kHz mono int16, off the ES8311's ADC over PIO + DMA. Nothing here decides
*what* was said -- it only fills a buffer and says how much of it is real.

It also plays the replies back. `speak()` streams a 16 kHz IMA-ADPCM clip out
of `voice.pak` through the same buffer, decoding one half while the DMA drains
the other (`src/adpcm.py` is the codec, `tools/voice_pak.py` the encoder).
Capture and playback share one buffer because they can never overlap, and one
sample rate because the clips are rendered at the rate the codec is already
running -- which is not a coincidence but the point: it deletes the re-clocking
dance the 8 kHz stopgap needed around every reply.

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

import adpcm
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

# Speech, NOT the chirp's 90: 90 overdrives this amp and speaker at the clips'
# peak of 15000. Bench-measured twice, at the desk, by ear -- which is the only
# instrument that has ever settled this. `tools/voice_pak.py` renders to the
# matching peak; changing either is a bench measurement, not an edit.
SPEECH_VOLUME = 82

# Smallest play half `bind_voice` will stream through, in 32-bit words. The real
# one is 12000 (750 ms); this only rules out a capture buffer so small that the
# chunk arithmetic cannot round an odd nibble count down. See `_chunk_nibs`.
MIN_PLAY_HALF_WORDS = 64


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
        # The activation tone goes FIRST, on the pristine heap -- 19 KB before
        # anything fragments. Three orderings have now been tried live:
        # press-time (starved once the arena was resident), after-capture
        # (starved the arena), and last-of-the-reservations (nothing left).
        # First is the one that cannot be squeezed out, and at 19 KB it costs
        # the later, larger allocations nothing they cannot spare. Empirical,
        # not modelled: the models lost three times.
        self._chirp = _build_chirp(rate)
        self.buf = allocate_samples(max_samples)
        self.max_samples = max_samples

        # Callable returning True while the board is making noise. Capture is
        # refused then -- the microphone would record the speaker.
        self._playing = playing

        self._audio = None
        self._codec = None
        self._pa = None
        self._dma = None
        # NOT built here, and not lazily at the first press either -- both
        # shipped and both failed live, one press-time (no 19,200-byte block
        # left once the arena and model were resident) and one boot-time (the
        # constructor runs before the TFLM arena, and building the chirp first
        # starved the 64 KB arena instead). The chirp is the SMALLEST of the
        # boot-time reservations, so it goes last: talk.reserve() calls
        # prepare_chirp() after the capture buffer, the spotter arena and the
        # templates. Order by size, largest first; the smallest thing yields.
        self._started_us = 0
        self._recording = False
        self._final_count = 0
        self._dout_ready = False
        self.available = None  # None = untried, True/False once known

        # --- streaming playback geometry ---------------------------------
        #
        # The play buffer IS the capture buffer: 96 KB, idle between turns,
        # already reserved at the heap floor, and a reply and a recording never
        # coexist (`start()` refuses while `playing()`). Split into two halves
        # that ping-pong -- one feeds the DMA while the next chunk of nibbles is
        # decoded into the other -- which is what lifts the old whole-clip
        # limit: a clip is no longer capped at 96 KB, only a chunk is.
        #
        # Counted in WORDS, because that is what the DMA transfers and what
        # `adpcm.decode_into` writes; MicroPython counts the same buffer in
        # halfwords. Mixing those two units is exactly how the "burst of static"
        # bug happened, so the conversion happens here and nowhere else.
        #
        # 12000 words a half = 750 ms of 16 kHz audio, so a clip crosses a
        # boundary about once a second. Decoding a half costs ~4 ms measured
        # (see `play_gap_us`), against 750 ms to drain it: the refill window is
        # not the risk here and never was. The re-arm gap between DMA runs is.
        self._play_half_words = (len(self.buf) // 2) // 2
        # The DMA needs an address, not an index. Built once: a memoryview per
        # chunk would be two heap objects per second on a heap that never
        # compacts. Both start on a word boundary because a half is a whole
        # number of words -- which the viper decoder's ptr32 store requires and
        # `tools/test_adpcm.py` asserts.
        half_h = 2 * self._play_half_words
        self._play_halves = (memoryview(self.buf)[0:half_h],
                             memoryview(self.buf)[half_h:2 * half_h])

        self._pak = None
        self._nibbles = None      # decode scratch; see bind_voice()
        self._last_lookup = (None, None)
        # Upper bound on the widest DMA re-arm gap seen, in microseconds, and
        # how many boundaries were crossed. Kept rather than probed once: it is
        # the number that decides whether streaming clicks, the PIO's TX FIFO
        # covers only ~500 us of it at 16 kHz, and a board that starts missing
        # the window has no other symptom than a tick nobody can reproduce.
        self.play_gap_us = 0
        self.play_boundaries = 0

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

    def prepare_chirp(self):
        """Build the activation tone buffer. Call at boot, AFTER the large
        reservations (capture, arena, templates) -- see the constructor note.
        Safe to call twice; returns False (and stays silent-capable) if even
        ~19 KB cannot be found, in which case chirp() degrades as designed."""
        if self._chirp is not None:
            return True
        try:
            self._chirp = _build_chirp(self.rate)
            return True
        except MemoryError:
            print("chirp buffer did not fit; the toy will be silent")
            return False

    # --- voice output ------------------------------------------------------

    def bind_voice(self, path="voice.pak"):
        """Open `voice.pak` and reserve the decode scratch. Returns True if the
        device can speak.

        Call from `talk.reserve()`, at boot, in a stated position -- **not**
        lazily at the first reply. The scratch is only ~6 KB, but "allocate it
        when it is first needed" is the shape that has silently starved the
        chirp twice on the TFLM image, and a voice that goes missing on the
        build with the tighter heap is the same bug wearing a different hat.
        It is the smallest of the boot reservations, so it goes last.

        Never raises. No pak means the device answers on the panel and says
        nothing, which `talk` already treats as the normal case.
        """
        try:
            pak = adpcm.Pak(path)
            if not pak.open():
                return False
            if pak.rate != self.rate:
                # Same rate end to end is the whole point (see `_speak_pak`);
                # a pak at another rate would play at the wrong pitch rather
                # than fail, so it is refused rather than re-clocked.
                print("voice.pak is %d Hz, capture is %d Hz -- refusing"
                      % (pak.rate, self.rate))
                pak.close()
                return False
            if self._play_half_words < MIN_PLAY_HALF_WORDS:
                # Not a tuning knob: below this the chunk arithmetic in
                # `_chunk_nibs` has no room to round an odd count down and the
                # stream would stall rather than glitch. Reaching here means
                # the capture buffer was built at some size nothing else in
                # this program asks for.
                print("play buffer is %d words a half, too small to stream"
                      % self._play_half_words)
                pak.close()
                return False
            # One chunk of nibbles: two samples a byte, so half a chunk's
            # words, plus one for an odd tail.
            self._nibbles = bytearray(self._play_half_words // 2 + 1)
            self._pak = pak
            print("voice.pak bound (%d clips, %d Hz)" % (pak.count, pak.rate))
            return True
        except Exception as exc:  # noqa: BLE001 -- silence beats a broken boot
            print("voice unavailable (%s: %s)" % (type(exc).__name__, exc))
            self._pak = None
            self._nibbles = None
            return False

    def has_clip(self, clip_id):
        """Is there a clip for this id? `talk._clip_for` asks before speaking.

        The answer is cached for one id, because the caller asks and then
        immediately plays, and each ask is a binary search over the on-flash
        index.
        """
        if self._pak is None:
            return False
        entry = self._pak.lookup(clip_id)
        self._last_lookup = (clip_id, entry)
        return entry is not None

    def speak(self, i2c, clip):
        """Say one reply. Returns True if something was played. Never raises.

        `clip` is whatever `talk.Conversation._clip_for` returned:

        - an 8-character `voice.pak` id -- the path that ships, streamed off
          flash at 16 kHz;
        - a `say_<id>.pcmw` filename -- the 8 kHz stopgap clips, kept working
          until they are deleted from the boards that still carry them.

        The dispatch is on the shape of the argument rather than on a flag,
        because the two differ in more than a rate: only the stopgap needs the
        codec re-clocked, and that difference is the point (see `_speak_pak`).
        """
        if not self._ensure(i2c):
            return False
        if isinstance(clip, str) and clip.endswith(".pcmw"):
            return self._speak_pcmw(clip)
        return self._speak_pak(clip)

    def _open_output(self, volume=SPEECH_VOLUME):
        """Wake the DAC, the amp and the DOUT state machine for playback.

        The unmute is not optional and not obvious: the DAC comes out of reset
        muted, `es8311.init()` does not clear it, and a muted DAC plays a clip
        with no exception and no sound -- the failure shape CLAUDE.md exists
        for, which cost a bench session in this file already.
        """
        if not self._dout_ready:
            self._audio.dout_pio_init()
            self._dout_ready = True
        self._codec.mute(False)
        self._codec.volume_set(volume)
        self._pa.value(1)

    def _close_output(self):
        """Drop the amp, the volume and the mute. An amp left on is both a
        battery drain and an open acoustic path into the capture that follows."""
        try:
            self._pa.value(0)
            self._codec.volume_set(DAC_VOLUME)
            self._codec.mute(True)
        except Exception:  # noqa: BLE001
            pass

    def _speak_pak(self, clip_id):
        """Stream one `voice.pak` clip. The path that ships.

        **The clock does not move.** The clips are 16 kHz, capture is 16 kHz,
        so this touches neither `es8311.init()` nor MCLK -- it unmutes, sets the
        volume, plays, and puts both back. The 8 kHz stopgap had to re-clock the
        codec and the MCLK state machine around every reply and restore them
        afterwards, and every one of those four register paths was a way to
        leave the microphone running at the wrong rate for the next press.
        Same rate end to end deletes that class of bug rather than fixing it.
        """
        if self._pak is None or self._nibbles is None:
            return False
        try:
            cached_id, entry = self._last_lookup
            if cached_id != clip_id:
                entry = self._pak.lookup(clip_id)
            if entry is None:
                return False
            self._open_output()
            try:
                return self._stream(entry)
            finally:
                self._close_output()
        except Exception as exc:  # noqa: BLE001 -- voice must never break a turn
            print("speak failed (%s: %s)" % (type(exc).__name__, exc))
            self._close_output()
            return False

    def _stream(self, entry):
        """Ping-pong the two halves of the capture buffer through the DMA.

        Decode the next chunk into the idle half while the DMA drains the live
        one, swap when it finishes. Returns True once the clip has been played
        out; the caller drops the amp.
        """
        offset, _length, n_samples = entry
        if n_samples < 1:
            return False
        state, header_samples = self._pak.start_clip(offset)
        if header_samples != n_samples:
            # The index and the blob disagree about the length. Refuse rather
            # than play the shorter of the two: this means the pak was written
            # by something that did not agree with `write_pak`, and the next
            # thing it disagrees about might be an offset.
            raise ValueError("clip length %d in index, %d in blob"
                             % (n_samples, header_samples))

        half = self._play_half_words
        buf = self.buf

        # Sample 0 is the header's predictor and is not encoded, so the first
        # chunk carries one fewer nibble than the rest.
        adpcm.emit_sample(buf, 0, state[0])
        remaining = n_samples - 1
        nibs = self._chunk_nibs(remaining, half - 1)
        self._decode(1, nibs, state)
        remaining -= nibs
        slot = 0
        self._arm(slot, 1 + nibs, True)

        while remaining:
            other = slot ^ 1
            nibs = self._chunk_nibs(remaining, half)
            self._decode(other * half, nibs, state)
            remaining -= nibs
            # Tight, deliberately: `time.sleep_ms(1)` here would be twenty times
            # the whole gap budget. The PIO's TX FIFO holds 8 words -- 500 us at
            # 16 kHz -- and that is the entire margin between "seamless" and a
            # tick at every boundary.
            t0 = self._await_dma()
            self._arm(other, nibs, False)
            gap = time.ticks_diff(time.ticks_us(), t0)
            if gap > self.play_gap_us:
                self.play_gap_us = gap
            self.play_boundaries += 1
            slot = other

        self._await_dma()
        # `play_finished()` is the DMA draining into the FIFO, not the speaker
        # stopping. Dropping the amp on it alone clips the tail of every reply.
        time.sleep_ms(CHIRP_SETTLE_MS)
        return True

    def _chunk_nibs(self, remaining, room):
        """How many nibbles the next chunk takes: at most `room`, and even
        unless it is the last.

        Nibble 2k and 2k+1 share a byte, so a chunk that stopped on an odd count
        would leave half a byte behind and the next chunk would have to re-enter
        that byte at its high half -- which `adpcm.decode_into` cannot do, since
        it always starts at a low nibble. Rounding down by one is free (one
        sample earlier in a 12000-sample chunk) and keeps every chunk starting
        at a byte boundary with `src_off=0`.

        Requires `room >= 2`, which `bind_voice` enforces: at room 1 the
        rounding has nowhere to go and this would return 0 and hang the stream
        rather than glitch it. Found by `tools/test_adpcm.py` sweeping the
        edges, not by anything at the desk -- a hang there would have looked
        like the board dying mid-reply.
        """
        n = room if remaining > room else remaining
        if n < remaining and (n & 1):
            n -= 1
        return n

    def _decode(self, out_off, nibs, state):
        """Read one chunk of nibbles off flash and decode it into `out_off`."""
        n_bytes = (nibs + 1) // 2
        got = self._pak.readinto(self._nibbles, n_bytes)
        if got != n_bytes:
            raise ValueError("clip body truncated: %d of %d" % (got, n_bytes))
        adpcm.decode_into(self._nibbles, 0, nibs, self.buf, out_off, state)

    def _await_dma(self):
        """Spin until the DMA has handed the last word to the PIO. Returns the
        tick at which that was noticed, which is the start of the re-arm gap."""
        audio = self._audio
        while not audio.play_finished():
            pass
        return time.ticks_us()

    def _arm(self, slot, count, restart):
        """Point the TX DMA at one half of the buffer and start it.

        `restart=True` on the first chunk only. It runs `_restart_tx()`, which
        jumps the state machine back to the top of `audio_pio_out` and resyncs
        it to an LRCLK frame -- necessary once, so left and right land in the
        right halves of the frame, and fatal every time after that: the program
        opens with a `pull()` whose word is discarded before the `start` label,
        so a restart per chunk would drop a sample at every boundary AND throw
        away the FIFO contents that are covering the re-arm gap.

        Mid-clip it therefore writes two registers and triggers. Same deviation
        from the vendor code as `_configure_dma` above and for the same reason:
        `audio_pio_mpy.py` is vendored and shipped to the Magic 8-Ball, and this
        program does not get to change what the 8-Ball runs.
        """
        import rp2

        audio = self._audio
        if audio.dma_tx is None:
            audio.dma_tx = rp2.DMA()
        dma = audio.dma_tx
        mv = self._play_halves[slot]
        if restart:
            audio._restart_tx()
            ctrl = audio._dma_pack_ctrl(
                dma,
                size=2,  # 32-bit transfers: the TX path is packed stereo words
                inc_read=True,
                inc_write=False,
                treq_sel=audio._pio_dreq(audio.sm_dout_id, True),
                high_pri=True,
                bswap=False,
            )
            dma.config(read=mv, write=audio.sm_dout, count=count, ctrl=ctrl,
                       trigger=True)
        else:
            # CTRL and the write address are still the ones set above; only the
            # source and the length change. `config(trigger=True)` with nothing
            # else is `dma_channel_start`.
            dma.read = mv
            dma.count = count
            dma.config(trigger=True)

    def _speak_pcmw(self, path, rate=8000):
        """Play one whole 8 kHz `.pcmw` clip. The stopgap, kept alive.

        Loads the file into the play buffer entire -- which is why these clips
        were 8 kHz, to fit -- and re-clocks the codec and MCLK down for the
        duration and back afterwards. Both of those constraints are gone on the
        pak path; this exists only until the six stopgap clips are off the
        boards that still hold them. Do not extend it.
        """
        try:
            with open(path, "rb") as fh:
                n_bytes = fh.readinto(self.buf)
            n_words = n_bytes // 4
            if not n_words:
                return False
            if not self._dout_ready:
                self._audio.dout_pio_init()
                self._dout_ready = True
            # re-clock down for the clip...
            self._codec.init(mclk_freq=rate * 256, sample_freq=rate,
                             res_in=16, res_out=16, volume=SPEECH_VOLUME,
                             mic_gain=MIC_GAIN)
            self._codec.mute(False)
            self._audio.mclk_freq = rate * 256
            self._audio.mclk_pio_init()
            self._pa.value(1)
            # count=n_words explicitly: the DMA's default count is ELEMENTS
            # of the buffer it is handed, and this buffer is int16 -- so the
            # first version played DOUBLE the clip, the second half being
            # stale capture audio. That was the "burst of static".
            self._audio.dma_play_words_async(self.buf, count=n_words)
            while not self._audio.play_finished():
                time.sleep_ms(10)
            time.sleep_ms(CHIRP_SETTLE_MS)
            return True
        except Exception as exc:  # noqa: BLE001 -- voice must never break a turn
            print("speak failed (%s: %s)" % (type(exc).__name__, exc))
            return False
        finally:
            # ...and restore the capture clock whatever happened.
            try:
                self._pa.value(0)
                self._codec.init(mclk_freq=self.rate * 256,
                                 sample_freq=self.rate, res_in=16, res_out=16,
                                 volume=DAC_VOLUME, mic_gain=MIC_GAIN)
                self._codec.mute(True)
                self._audio.mclk_freq = self.rate * 256
                self._audio.mclk_pio_init()
            except Exception:  # noqa: BLE001
                pass

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
                return False   # prepare_chirp() never ran or did not fit
            if not self._dout_ready:
                self._audio.dout_pio_init()
                self._dout_ready = True
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
