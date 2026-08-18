"""Phase 0 measurements for the push-to-talk ELIZA program.

Answers, with numbers rather than adjectives, the questions the speech front end
is being designed around. **Run this before recording a vocabulary.** Enrolling
twenty-five words at the wrong sample rate is a genuinely annoying way to lose
an evening, and every failure it looks for is one that produces clean-looking
output rather than an error.

  a. Does the ES8311 actually run at 16 kHz, or does the register write merely
     succeed? (Measured against wall-clock time, with 24 kHz -- the rate the
     microphone was verified at -- as a control.) Also prints the two RMS
     readings, quiet room and one spoken word, that calibrate
     tools/pull_recording.py's SILENCE_RMS -- currently a reasoned guess.
  b. Does the VAD's viper inner loop compile here, and does it agree with the
     portable reference the host shares?
  c. Does emlearn_fft's prebuilt native module load on this chip?
  d. How long does a 512-point FFT take here? The estimate to beat is ~3 ms,
     extrapolated from peterhinch's assembler FFT at 6.97 ms for 1024 points on
     a Pico 2. Nobody has published a MicroPython MFCC benchmark for RP2350, so
     this replaces an extrapolation with a measurement.
  e. What does USB CDC actually manage, with a capture running rather than on an
     idle board? That is the enrolment transport's whole budget.
  f. Do the two big allocations -- the 94 KB capture buffer (188 KB to build)
     and ~137 KB of templates -- coexist, in the order talk.py reserves them?
     The free total is not the number that fails; the largest contiguous block
     is, and there is no API for it.
  g. What does the keyword spotter cost per turn, in plain MicroPython with no
     viper anywhere? The budget is the pause after the button, against a panel
     that takes ~583 ms to redraw regardless.
  h. Is the front end bit-identical to the host **on this board**? The host
     suite cannot answer that -- both sides have unbounded integers there. This
     runs src/speech_fixtures.py, which pins the pipeline stage by stage.

Order of operations on real hardware: confirm the rate (a), then one capture to
confirm the transport and read off KB/s and RMS (e), and only then a real
enrolment run with tools/enrol.py.

Run it with the src/ directory mounted, so it can import the board modules
without anything being copied to the device:

    uvx mpremote connect /dev/cu.usbmodem101 mount src run tools/speech_probe.py

(If the ELIZA program is already deployed, plain `run` works too.)

Every section is independently guarded: a board without emlearn installed still
gets its capture numbers, and vice versa. Nothing here writes to flash.

Reading the results: "no exception raised" proves nothing about audio, exactly
as it proves nothing about the panel -- an unconfigured codec still returns
samples. The numbers that matter are the measured sample rate and the signal
statistics. All-zero samples mean the RX path is dead however happily it ran.
"""

import gc
import math
import sys
import time
from array import array

from machine import I2C, Pin

FFT_SIZE = 512
FFT_REPEATS = 20

# Make a noise while it captures -- the numbers are far more useful if there is
# a signal in the room. It prints when to.
CAPTURE_MS = 1000


def rule(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# --- 0. What we are running on ---------------------------------------------
# The arch tag and .mpy version decide which prebuilt native module will even
# load, so this is the first thing to know before trying to install one.

_ARCHS = (
    "none", "x86", "x64", "armv6", "armv6m", "armv7m", "armv7em",
    "armv7emsp", "armv7emdp", "xtensa", "xtensawin", "rv32imc",
)

rule("0. environment")
print("MicroPython %s on %s" % (".".join(str(v) for v in sys.implementation.version), sys.platform))
try:
    import machine

    print("clock:  %d MHz" % (machine.freq() // 1_000_000))
except Exception as exc:  # noqa: BLE001
    print("clock:  unknown (%s)" % exc)

mpy_arch = None
try:
    mpy = sys.implementation._mpy
    version = mpy & 0xFF
    sub = (mpy >> 8) & 3
    mpy_arch = _ARCHS[mpy >> 10] if (mpy >> 10) < len(_ARCHS) else "unknown(%d)" % (mpy >> 10)
    print("mpy:    version %d.%d, native arch %s" % (version, sub, mpy_arch))
    print("        -> prebuilt .mpy modules must be built for %s / mpy %d.%d"
          % (mpy_arch, version, sub))
except AttributeError:
    print("mpy:    sys.implementation._mpy missing; native modules are not supported")

gc.collect()
print("heap:   %d bytes free, %d allocated" % (gc.mem_free(), gc.mem_alloc()))


# --- a. Does the codec really run at 16 kHz? -------------------------------

def bring_up(i2c, rate):
    """Configure the codec and PIO for `rate` and return (codec, audio)."""
    import audio_pio_mpy
    import es8311

    Pin(0, Pin.OUT, value=0)  # power amp off: this is a capture test

    codec = es8311.ES8311(i2c)
    codec.init(
        mclk_freq=rate * 256,
        sample_freq=rate,
        res_in=16,
        res_out=16,
        volume=0,
        mic_gain=3,
    )

    audio = audio_pio_mpy.AudioPIO(
        mclk_pin=3, dout_pin=1, din_pin=2, lrclk_pin=5, bclk_pin=4,
        sm_dout_id=0, sm_din_id=5, sm_mclk_id=2,
    )
    audio.mclk_freq = rate * 256
    audio.sample_freq = rate
    audio.channel_count = 1
    audio.rx_channel = 0
    audio.mclk_pio_init()
    audio.din_pio_init()
    audio.start()
    return codec, audio


def signal_stats(buf, count):
    """(min, max, rms, mean_abs, flat) over the first `count` samples.

    RMS is the number the host cares about: `tools/pull_recording.py` rejects a
    capture below SILENCE_RMS (200 of full scale 32768), and that constant is a
    reasoned guess until this prints a real quiet room and a real spoken word.

    `flat` -- min == max -- is the unambiguous one. A codec that never received
    BCLK returns a constant, and the DMA delivers it perfectly cleanly, so a
    transfer that looks flawless proves nothing at all about the microphone.
    """
    smallest = 32767
    largest = -32768
    total = 0
    squares = 0
    for i in range(count):
        value = buf[i]
        if value < smallest:
            smallest = value
        if value > largest:
            largest = value
        squares += value * value
        total += value if value >= 0 else -value
    n = max(1, count)
    return smallest, largest, _isqrt(squares // n), total // n, smallest == largest


def _isqrt(value):
    """Integer square root, so RMS stays out of the float path."""
    if value <= 0:
        return 0
    guess = value
    step = (value + 1) // 2
    while step < guess:
        guess = step
        step = (guess + value // guess) // 2
    return guess


def dbfs(value):
    if value <= 0:
        return float("-inf")
    return 20 * math.log(value / 32768.0) / math.log(10)


def measure_rate(i2c, rate, buf, prompt="stay quiet"):
    """Capture `CAPTURE_MS` at `rate` and time it. Returns (rate, count, rms)."""
    count = rate * CAPTURE_MS // 1000
    view = memoryview(buf)[:count]

    codec, audio = bring_up(i2c, rate)
    coeff = codec.get_coeff(rate * 256, rate)
    print("COEFF_DIV row for MCLK=%d fs=%d: %s"
          % (rate * 256, rate, "index %d" % coeff if coeff >= 0 else "MISSING -- codec will not be reconfigured"))
    print("REG02..REG08 readback: %s"
          % " ".join("%02x" % codec.read_reg(reg) for reg in range(0x02, 0x09)))
    print("  (a readback proves the write landed, not that the clock changed)")

    print("%s..." % prompt)
    started = time.ticks_us()
    audio.dma_record_into(view)
    elapsed = time.ticks_diff(time.ticks_us(), started)

    measured = count * 1_000_000 // max(1, elapsed)
    smallest, largest, rms, mean, flat = signal_stats(buf, count)
    print("asked for %5d Hz -> %d samples in %d us = %d Hz measured (%+.1f%%)"
          % (rate, count, elapsed, measured, 100.0 * (measured - rate) / rate))
    print("signal: min %6d  max %6d  RMS %5d (%.1f dBFS)  mean |x| %5d"
          % (smallest, largest, rms, dbfs(rms), mean))
    if flat:
        print("  ** min == max: this is a constant, not audio. The codec is")
        print("     almost certainly not receiving BCLK. A clean transfer of")
        print("     nothing looks exactly like a clean transfer of something. **")
    audio.stop()
    return measured, count, rms


rule("a. capture rate, and the two RMS readings the host needs")
capture_buf = None
captured_16k = 0
quiet_rms = 0
loud_rms = 0
try:
    i2c = I2C(1, sda=Pin(6), scl=Pin(7), freq=100_000)
    if 0x18 not in i2c.scan():
        raise OSError("ES8311 did not ACK at 0x18")

    gc.collect()
    # One buffer, big enough for the 24 kHz control run, reused throughout.
    capture_buf = array("h", bytearray(2 * 24000 * CAPTURE_MS // 1000))

    print("-- 24 kHz (the rate docs/hardware.md verified; a control) --")
    measure_rate(i2c, 24000, capture_buf, prompt="stay quiet")

    print()
    print("-- 16 kHz, quiet room (what listen.py wants) --")
    measured, captured_16k, quiet_rms = measure_rate(i2c, 16000, capture_buf,
                                                     prompt="stay quiet")

    print()
    if abs(measured - 16000) < 16000 * 0.02:
        print("VERDICT: 16 kHz confirmed to within 2%.")
    elif abs(measured - 24000) < 24000 * 0.02:
        print("VERDICT: asked for 16 kHz, got 24 kHz. The rate did NOT change --")
        print("         the codec kept its previous configuration.")
    else:
        print("VERDICT: neither 16 nor 24 kHz. Check MCLK and the COEFF_DIV row.")
    print("(The measured figure includes DMA setup and the wait for the first")
    print(" LRCLK edge, so it reads slightly low. A few tenths of a percent is")
    print(" the method, not the codec.)")

    print()
    print("-- 16 kHz, one spoken word --")
    _measured, _count, loud_rms = measure_rate(i2c, 16000, capture_buf,
                                               prompt="SAY ONE WORD NOW")

    print()
    print("RMS quiet %d, RMS spoken %d, ratio %.1fx"
          % (quiet_rms, loud_rms, loud_rms / max(1.0, float(quiet_rms))))
    print("These two numbers are what calibrate tools/pull_recording.py's")
    print("SILENCE_RMS, currently a reasoned guess at 200 of full scale 32768.")
    print("Set it between the two, nearer the quiet figure.")
    if loud_rms <= quiet_rms:
        print("  ** speaking did not raise RMS. Either nobody spoke, or the")
        print("     microphone is not in the signal path at all. **")
    if loud_rms > 30000:
        print("  ** railed. Lower listen.MIC_GAIN (currently a guess at 3). **")
except Exception as exc:  # noqa: BLE001
    print("capture FAILED (%s: %s)" % (type(exc).__name__, exc))


# --- b. The VAD's inner loop, on real room audio ---------------------------

def _frame_stats_portable(samples, start, count):
    """Copy of vad's reference implementation, kept here to time viper against."""
    energy = 0
    crossings = 0
    prev = 0
    for i in range(count):
        value = samples[start + i]
        if value < 0:
            energy -= value
            sign = -1
        else:
            energy += value
            sign = 1
        if i and sign != prev:
            crossings += 1
        prev = sign
    return (energy << 8) | crossings


rule("b. vad._frame_stats on that capture (does viper compile, and help?)")
try:
    import vad

    if capture_buf is None or captured_16k == 0:
        raise RuntimeError("no capture to analyse")

    frames = captured_16k // vad.VAD_FRAME
    started = time.ticks_us()
    for f in range(frames):
        vad._frame_stats(capture_buf, f * vad.VAD_FRAME, vad.VAD_FRAME)
    viper_us = time.ticks_diff(time.ticks_us(), started)

    started = time.ticks_us()
    for f in range(frames):
        _frame_stats_portable(capture_buf, f * vad.VAD_FRAME, vad.VAD_FRAME)
    plain_us = time.ticks_diff(time.ticks_us(), started)

    # They must agree, or the device is running a different algorithm from the
    # host tools.
    mismatches = 0
    for f in range(frames):
        offset = f * vad.VAD_FRAME
        if vad._frame_stats(capture_buf, offset, vad.VAD_FRAME) != _frame_stats_portable(
            capture_buf, offset, vad.VAD_FRAME
        ):
            mismatches += 1

    print("%d frames of %d samples" % (frames, vad.VAD_FRAME))
    print("as compiled: %6d us total, %5.1f us/frame" % (viper_us, viper_us / frames))
    print("portable:    %6d us total, %5.1f us/frame" % (plain_us, plain_us / frames))
    print("speedup:     %.1fx  (viper claims ~16x on integer work)" % (plain_us / max(1, viper_us)))
    print("agreement:   %s" % ("identical" if mismatches == 0 else "** %d FRAMES DIFFER **" % mismatches))
    if plain_us / max(1, viper_us) < 2:
        print("  ** viper did not take: the fast path is not compiled **")

    detector = vad.Endpointer(max_frames=frames + 1)
    detector.feed(capture_buf, captured_16k)
    energy, zcr = vad.frame_stats(capture_buf, frames)
    itl, itu, izct, imn, egate = vad.thresholds(energy, zcr, frames)
    print("live:        floor=%d start=%d end=%d per 20 ms unit (stop-recording only)"
          % (detector.noise_floor, detector.live_start, detector.live_end))
    print("paper:       itl=%d itu=%d izct=%d imn=%d egate=%d%s"
          % (itl, itu, izct, imn, egate,
             "  ** ZCR BAND EMPTY **" if egate >= itl else ""))
    print("             (these are the boundaries the host shares. An empty band")
    print("              means the zero-crossing pass cannot fire at all here.)")
    print("endpoints:   %s" % (detector.bounds(),))
    print("  -- a quiet room with one spoken word should report a span here;")
    print("     None means either nothing was said or the thresholds are wrong.")
except Exception as exc:  # noqa: BLE001
    print("vad check FAILED (%s: %s)" % (type(exc).__name__, exc))


# --- c. Is emlearn_fft available? ------------------------------------------

rule("c. emlearn_fft")
emlearn_fft = None
try:
    import emlearn_fft  # noqa: F811

    print("imported. module contents: %s" % dir(emlearn_fft))
except ImportError as exc:
    print("not installed (%s)" % exc)
    print()
    print("This board has no networking (RP2350A, not a -W), so it cannot")
    print("mip-install anything itself. Fetch on the Mac and copy:")
    print()
    print("    uvx mpremote connect $PORT mip install \\")
    print("      https://emlearn.github.io/emlearn-micropython/builds/master/%s_<mpy>/emlearn_fft.mpy"
          % (mpy_arch or "<arch>"))
    print()
    print("Check that path against the emlearn-micropython README before")
    print("trusting it -- the layout of the build directory is theirs to change,")
    print("and the project only claims testing on x64 and xtensawin. This board")
    print("reports %s, so a clean import here is the whole question." % (mpy_arch or "?"))


def time_emlearn(module, size, repeats):
    """Try the plausible call shapes; return (name, us_per_transform)."""
    real = array("f", [0.0] * size)
    imag = array("f", [0.0] * size)
    for i in range(size):
        real[i] = math.sin(2 * math.pi * 20 * i / size)

    # Shape 1: emlearn_fft.FFT(n).run(real, imag)
    try:
        planner = module.FFT(size)
        started = time.ticks_us()
        for _ in range(repeats):
            planner.run(real, imag)
        return "FFT(n).run(real, imag)", time.ticks_diff(time.ticks_us(), started) / repeats
    except Exception as exc:  # noqa: BLE001
        print("  FFT(n).run(...) did not work: %s: %s" % (type(exc).__name__, exc))

    # Shape 2: module-level function
    try:
        started = time.ticks_us()
        for _ in range(repeats):
            module.fft(real, imag)
        return "fft(real, imag)", time.ticks_diff(time.ticks_us(), started) / repeats
    except Exception as exc:  # noqa: BLE001
        print("  fft(...) did not work: %s: %s" % (type(exc).__name__, exc))
    return None, 0


if emlearn_fft is not None:
    try:
        name, per_call = time_emlearn(emlearn_fft, FFT_SIZE, FFT_REPEATS)
        if name:
            print("%d-point FFT via %s: %.0f us (%.2f ms)"
                  % (FFT_SIZE, name, per_call, per_call / 1000))
        else:
            print("loaded, but none of the expected call shapes worked --")
            print("check dir() above against the emlearn-micropython docs.")
    except Exception as exc:  # noqa: BLE001
        print("emlearn timing FAILED (%s: %s)" % (type(exc).__name__, exc))


# --- d. A hand-rolled fixed-point FFT, for the fallback case ---------------
#
# This is a BENCHMARK, not the shipping front end -- the MFCC pipeline and its
# feature spec belong to docs/speech.md and tools/mfcc.py. It exists to put a
# number on the fallback if emlearn will not load.
#
# Radix-2 decimation-in-time, Q15 twiddles, integers throughout. Every butterfly
# scales its outputs by 1/2, which costs 9 bits of headroom over 9 stages and is
# what keeps the products inside a 32-bit machine word: without it, wr * xr
# reaches 5.5e11 at 512 points and viper silently wraps.

_cos_table = array("i", bytearray(4 * (FFT_SIZE // 2)))
_sin_table = array("i", bytearray(4 * (FFT_SIZE // 2)))
for _k in range(FFT_SIZE // 2):
    _cos_table[_k] = int(round(32767 * math.cos(-2 * math.pi * _k / FFT_SIZE)))
    _sin_table[_k] = int(round(32767 * math.sin(-2 * math.pi * _k / FFT_SIZE)))


def _fft_portable(re, im, n, cos_t, sin_t):
    i = 1
    j = 0
    while i < n:
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            t = re[i]; re[i] = re[j]; re[j] = t
            t = im[i]; im[i] = im[j]; im[j] = t
        i += 1
    length = 2
    step = n >> 1
    while length <= n:
        half = length >> 1
        start = 0
        while start < n:
            k = 0
            i = start
            while i < start + half:
                wr = cos_t[k]; wi = sin_t[k]
                xr = re[i + half]; xi = im[i + half]
                tr = (wr * xr - wi * xi) >> 15
                ti = (wr * xi + wi * xr) >> 15
                ar = re[i]; ai = im[i]
                re[i + half] = (ar - tr) >> 1
                im[i + half] = (ai - ti) >> 1
                re[i] = (ar + tr) >> 1
                im[i] = (ai + ti) >> 1
                k += step
                i += 1
            start += length
        length <<= 1
        step >>= 1


_fft = _fft_portable

try:
    import micropython

    @micropython.viper
    def _fft(re: ptr32, im: ptr32, n: int, cos_t: ptr32, sin_t: ptr32):  # noqa: F811
        i = 1
        j = 0
        while i < n:
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            i += 1
        length = 2
        step = n >> 1
        while length <= n:
            half = length >> 1
            start = 0
            while start < n:
                k = 0
                i = start
                while i < start + half:
                    wr = cos_t[k]
                    wi = sin_t[k]
                    xr = re[i + half]
                    xi = im[i + half]
                    tr = (wr * xr - wi * xi) >> 15
                    ti = (wr * xi + wi * xr) >> 15
                    ar = re[i]
                    ai = im[i]
                    re[i + half] = (ar - tr) >> 1
                    im[i + half] = (ai - ti) >> 1
                    re[i] = (ar + tr) >> 1
                    im[i] = (ai + ti) >> 1
                    k += step
                    i += 1
                start += length
            length <<= 1
            step >>= 1
    _fft_is_viper = True
except (ImportError, NameError):  # pragma: no cover -- host only
    _fft_is_viper = False


def _fill_tone(re, im, n, bin_index):
    for i in range(n):
        re[i] = int(30000 * math.cos(2 * math.pi * bin_index * i / n))
        im[i] = 0


def _peak_bin(re, im, n):
    peak = -1
    at = -1
    for k in range(n // 2):
        magnitude = abs(re[k]) + abs(im[k])
        if magnitude > peak:
            peak = magnitude
            at = k
    return at, peak


rule("d. hand-rolled fixed-point FFT (%d points)" % FFT_SIZE)
try:
    re = array("i", bytearray(4 * FFT_SIZE))
    im = array("i", bytearray(4 * FFT_SIZE))

    # Correctness before speed: a pure tone at bin 77 must come out at bin 77.
    _fill_tone(re, im, FFT_SIZE, 77)
    _fft(re, im, FFT_SIZE, _cos_table, _sin_table)
    at, peak = _peak_bin(re, im, FFT_SIZE)
    print("tone at bin 77 -> peak at bin %d, magnitude %d  %s"
          % (at, peak, "OK" if at == 77 else "** WRONG **"))

    started = time.ticks_us()
    for _ in range(FFT_REPEATS):
        _fill_tone(re, im, FFT_SIZE, 77)
        _fft(re, im, FFT_SIZE, _cos_table, _sin_table)
    with_fill = time.ticks_diff(time.ticks_us(), started) / FFT_REPEATS

    started = time.ticks_us()
    for _ in range(FFT_REPEATS):
        _fill_tone(re, im, FFT_SIZE, 77)
    fill_only = time.ticks_diff(time.ticks_us(), started) / FFT_REPEATS

    per_fft = with_fill - fill_only
    print("%s FFT: %.0f us (%.2f ms) per transform, over %d runs"
          % ("viper" if _fft_is_viper else "plain-python", per_fft, per_fft / 1000, FFT_REPEATS))
    print("  (the %.0f us of float sine used to refill the input is subtracted)" % fill_only)
    print("estimate to beat: ~3000 us. peterhinch's ARM assembler FFT does")
    print("1024 points in 6970 us on a Pico 2, so ~3 ms was the extrapolation.")

    if _fft_is_viper:
        started = time.ticks_us()
        _fill_tone(re, im, FFT_SIZE, 77)
        _fft_portable(re, im, FFT_SIZE, _cos_table, _sin_table)
        plain = time.ticks_diff(time.ticks_us(), started)
        print("plain-python, one run: %.2f ms -> viper speedup %.1fx"
              % (plain / 1000.0, plain / max(1.0, per_fft)))
except Exception as exc:  # noqa: BLE001
    print("FFT benchmark FAILED (%s: %s)" % (type(exc).__name__, exc))

# --- e. USB CDC throughput, and the async capture path ---------------------
#
# The enrolment transport (src/record_stream.py) is only as good as this. The
# published figures are print() 30-50 KB/s, sys.stdout.write() ~200 KB/s and
# sys.stdout.buffer.write() ~600 KB/s, and a concurrent high-frequency timer or
# ISR costs about 5x -- which is what the "RP2350 is only 100 KB/s" report
# (micropython #17398) turned out to be. So this measures idle *and* with the
# capture DMA in flight, because idle is not the condition it runs in.
#
# This section prints THROUGHPUT_KB of filler. The wall of dots is the
# measurement, not a fault.

THROUGHPUT_KB = 64
_FILLER = b"." * 4096


def measure_cdc():
    started = time.ticks_us()
    for _ in range(THROUGHPUT_KB * 1024 // len(_FILLER)):
        sys.stdout.buffer.write(_FILLER)
    elapsed = time.ticks_diff(time.ticks_us(), started)
    print()
    return THROUGHPUT_KB * 1_000_000 // max(1, elapsed)


rule("e. USB CDC throughput (%d KB of dots is the measurement)" % THROUGHPUT_KB)
try:
    idle_kbps = measure_cdc()
    print("idle:        %d KB/s" % idle_kbps)

    import listen

    recorder = listen.Recorder(rate=16000, max_samples=2 * 16000)
    if recorder.start(i2c):
        before = recorder.captured()
        loaded_kbps = measure_cdc()
        during = recorder.captured()
        got = recorder.stop()
        print("under load:  %d KB/s, with a 16 kHz capture running" % loaded_kbps)
        print("             (%d%% of idle)" % (100 * loaded_kbps // max(1, idle_kbps)))
        print("capture progress during the write: %d -> %d samples, %d at stop"
              % (before, during, got))
        if during <= before:
            print("  ** the sample count did not advance. Either DMA.count does")
            print("     not read back on this firmware -- listen.Recorder falls")
            print("     back to a clock estimate, which is fine -- or the")
            print("     capture is not running at all. **")
        if loaded_kbps < 150:
            print("  ** under 150 KB/s. Look for a timer or ISR; that is the")
            print("     known 5x cost, not a USB limit. **")
        print("3 s of 16 kHz mono is 96 KB, so record_stream should take ~%.2f s"
              % (96.0 / max(1, loaded_kbps)))
    else:
        print("could not start a capture; skipping the under-load figure")
except Exception as exc:  # noqa: BLE001
    print("throughput FAILED (%s: %s)" % (type(exc).__name__, exc))

# --- f. Does the whole program fit in the heap? ----------------------------
#
# The binding constraint on this program is not CPU, it is one number: the
# largest *contiguous* block left. MicroPython's heap never compacts, so the
# free total can look comfortable while no single block that large remains --
# `sounds.allocate_bytes` documents the same trap costing this project a clip it
# had plenty of free memory for.
#
# The order matters and is not "largest first". The templates are the larger
# resident block (137 KB against 94 KB) but allocate as one `bytearray` with no
# transient; the capture buffer is `array("h", bytearray(2*n))`, which holds
# both at once and so briefly needs 188 KB to end up with 94. Capture first is
# 92.3 KB cheaper at the peak. This allocates them in the order
# talk.reserve_templates actually uses, so the number means something.

CAPTURE_KB = 96    # listen.MAX_SAMPLES * 2, at 3 s of 16 kHz mono

# Read from the templates module when it is on the device, so this measures what
# will actually be allocated rather than an estimate of it. The fallback is the
# figure for 21 classes x 3 takes at 24 wide, TEMPLATE_FORMAT 2.
try:
    import templates as _templates

    TEMPLATE_KB = (_templates.BUFFER_BYTES + 1023) // 1024
    TEMPLATE_SOURCE = "templates.BUFFER_BYTES"
except (ImportError, AttributeError):
    TEMPLATE_KB = 137
    TEMPLATE_SOURCE = "estimate; templates.py is not on the device"


def largest_block():
    """Largest single bytearray that can still be allocated, by bisection.

    There is no API for this -- gc.mem_free() is a total, and the total is not
    the number that fails. Costs a few allocations of up to the free size, which
    is why it is at the end of the probe.
    """
    gc.collect()
    low = 0
    high = gc.mem_free()
    while low < high:
        mid = (low + high + 1) // 2
        try:
            block = bytearray(mid)
            del block
            low = mid
        except MemoryError:
            high = mid - 1
        gc.collect()
    return low


rule("f. heap budget (largest contiguous block, not the free total)")
try:
    del capture_buf  # the probe's own 48 KB, out of the way
except NameError:
    pass
try:
    gc.collect()
    print("templates sized from %s" % TEMPLATE_SOURCE)
    print("at rest:     %d free, largest block %d" % (gc.mem_free(), largest_block()))

    held = []
    for name, kb in (("capture buffer", CAPTURE_KB), ("templates", TEMPLATE_KB)):
        gc.collect()
        try:
            if name == "capture buffer":
                # Built the way listen.allocate_samples builds it, transient and
                # all -- allocating a bare bytearray here would measure a
                # different program from the one that runs.
                held.append(array("h", bytearray(kb * 1024)))
            else:
                held.append(bytearray(kb * 1024))
            print("+ %-14s %3d KB: ok, %d free, largest block now %d"
                  % (name, kb, gc.mem_free(), largest_block()))
        except MemoryError:
            print("+ %-14s %3d KB: ** WOULD NOT FIT ** (%d free, largest %d)"
                  % (name, kb, gc.mem_free(), largest_block()))
            break
    else:
        spare = largest_block()
        print()
        print("both fit. %d bytes still allocatable in one block." % spare)
        print("Note this is with neither eliza_rules nor the spotter imported,")
        print("so it is an upper bound on the headroom, not the headroom.")
    del held
    gc.collect()
except Exception as exc:  # noqa: BLE001
    print("heap budget FAILED (%s: %s)" % (type(exc).__name__, exc))

# --- g. What the spotter costs ---------------------------------------------
#
# The latency budget is the pause after the user lets go of the screen, against
# a panel that already takes ~583 ms for a partial refresh. So the question is
# not "is it fast" but "is it small against 583 ms".
#
# src/spotter.py is written in plain MicroPython on purpose -- correct first,
# and no viper anywhere yet. This is the measurement that says whether that is
# a problem. Every timing in docs/speech.md is currently extrapolated from an
# assembler FFT benchmark on a different board.

rule("g. keyword spotter: front end and one DTW")
try:
    import spotter

    if capture_buf is None or captured_16k == 0:
        raise RuntimeError("no capture to analyse")

    # A word-length slice of the real capture, so the frame count is realistic.
    words = min(captured_16k, 16000)  # 1 s
    frames = spotter.frame_count(words)

    gc.collect()
    started = time.ticks_us()
    query, n = spotter.features(capture_buf, 0, words)
    feat_us = time.ticks_diff(time.ticks_us(), started)
    print("front end:  %d ms for %d frames (%.1f ms/frame, %d ms of audio)"
          % (feat_us // 1000, n, feat_us / 1000.0 / max(1, n), words // 16))
    print("            real time would be %.1fx"
          % (feat_us / 1000.0 / max(1.0, words / 16.0)))

    # One DTW against a template of the same length, which is the per-template
    # cost; a turn pays it once per enrolled template.
    tmpl = bytearray(2 * spotter.N_FEAT * n)
    for i in range(len(tmpl)):
        tmpl[i] = (i * 37) & 0xFF
    started = time.ticks_us()
    d = spotter.dtw(query, n, tmpl, 0, n)
    dtw_us = time.ticks_diff(time.ticks_us(), started)
    print("one DTW:    %d us (%d frames vs %d), distance %d" % (dtw_us, n, n, d))

    for count in (66, 63):
        total = feat_us + dtw_us * count
        print("a turn with %d templates: %d ms front end + %d ms matching = %d ms"
              % (count, feat_us // 1000, dtw_us * count // 1000, total // 1000))
    print("against a 583 ms partial refresh, which happens anyway.")
    print("If matching dominates, the first lever is the duration gate --")
    print("DUR_RATIO_PCT rejects mismatched lengths before the inner loop.")
except ImportError as exc:
    print("spotter not on the device (%s)" % exc)
except Exception as exc:  # noqa: BLE001
    print("spotter timing FAILED (%s: %s)" % (type(exc).__name__, exc))

# --- h. Bit-exactness, on the board ----------------------------------------
#
# This is the one check that cannot be done on the host. `tools/test_spotter.py`
# proves the port agrees with `tools/mfcc.py` under CPython, where both get
# arbitrary-precision integers -- so it says nothing about int32. MicroPython's
# small ints are arbitrary-precision too, which is why the plain port should
# pass here; a viper port would not necessarily, because viper's int is a
# machine word that wraps silently.
#
# So: run it now to establish the baseline, and run it again after anyone
# reaches for viper. The fixtures pin the pipeline stage by stage, so a failure
# names which stage wrapped instead of only saying the answer changed.

rule("h. front-end bit-exactness against the host fixtures")
try:
    import speech_fixtures as fx
    import spotter

    print("%d cases, %d frames each, format %d" % (len(fx.CASES), fx.FRAMES, fx.FORMAT))
    print("tw_stress runs first: it is real speech at 99.8% of the FFT")
    print("twiddle's proved ceiling, so a port that wraps, wraps there.")
    failures = 0
    ordered = ([c for c in fx.CASES if c[0] == "tw_stress"]
               + [c for c in fx.CASES if c[0] != "tw_stress"])
    for name, why, pcm, exp in ordered:
        samples = array("h")
        samples.frombytes(pcm)

        started = time.ticks_us()
        got, frames = spotter.features(samples, 0, len(samples))
        us = time.ticks_diff(time.ticks_us(), started)

        want = exp["features"]
        bad = None
        for f in range(len(want)):
            for j in range(fx.N_FEAT):
                if got[f * fx.N_FEAT + j] - spotter.BIAS != want[f][j]:
                    bad = (f, j, got[f * fx.N_FEAT + j] - spotter.BIAS, want[f][j])
                    break
            if bad:
                break

        peaks = exp.get("peaks", {})
        worst = 0
        worst_at = "-"
        for stage in peaks:
            if peaks[stage] > worst:
                worst = peaks[stage]
                worst_at = stage
        if bad is None:
            print("  ok   %-10s %d frames in %d us  (host peak %s %.1f%%)"
                  % (name, frames, us, worst_at, 100.0 * worst / 2147483647.0))
        else:
            failures += 1
            print("  FAIL %-10s frame %d coefficient %d: device %d, host %d"
                  % (name, bad[0], bad[1], bad[2], bad[3]))
            print("       host peak was %s at %.1f%% of int32 -- if this is a"
                  % (worst_at, 100.0 * worst / 2147483647.0))
            print("       viper build, that is the stage to look at first.")

    if failures == 0:
        print()
        print("VERDICT: the front end is bit-identical to the host on this board.")
        print("Templates enrolled on the host will be compared against features")
        print("computed the same way, which is the whole contract.")
    else:
        print()
        print("VERDICT: %d of %d cases differ. Do NOT enrol against this build --"
              % (failures, len(fx.CASES)))
        print("every template would be matched against differently-computed")
        print("features, and the symptom would be a recogniser that is merely")
        print("a bit poor, which is indistinguishable from six other faults.")
except ImportError as exc:
    print("fixtures or spotter not on the device (%s)" % exc)
except Exception as exc:  # noqa: BLE001
    print("bit-exactness check FAILED (%s: %s)" % (type(exc).__name__, exc))

gc.collect()
print()
print("heap after: %d bytes free" % gc.mem_free())
