# Hardware

**Board: Waveshare RP2350-Touch-ePaper-1.54.**

Pin numbers come from Waveshare's demo code
([waveshareteam/RP2350-Touch-ePaper-1.54](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54),
Apache-2.0). Everything marked *verified* was then confirmed against this
physical board on 2026-08-17.

## MCU

| | |
| --- | --- |
| Chip | RP2350A (ARM Cortex-M33, Secure image) — *verified* |
| Clock | 150 MHz — *verified* |
| RAM | 520 KB (491 KB free under MicroPython) — *verified* |
| Flash | 16 MB — *verified* |
| USB serial | `/dev/cu.usbmodem101`, VID `2E8A` PID `0009` |
| Board UID | `9717825cf3bcf97c` |

## E-paper display

1.54", 200 × 200, monochrome, SSD1681 controller. Driver: `src/epaper.py`.

| Signal | GPIO |
| --- | --- |
| SPI bus | SPI1 @ 4 MHz, mode 0 |
| SCK | 10 |
| MOSI | 11 |
| CS | 9 |
| DC | 12 |
| PWR | 13 |
| RST | 14 |
| BUSY | 15 (input, pull-up) |

*Verified* — both refresh modes drive the panel and it holds the image with power
removed.

| | BUSY | wall | flashes? |
| --- | --- | --- | --- |
| full refresh | 1397 ms | 1715 ms | yes |
| partial refresh | 583 ms | 612 ms | no |

Partial refresh diffs against a base image in the controller's RAM, so the panel
cannot be deep-slept between updates — deep sleep is only left via a hardware
reset, which drops the base. `main.Panel` owns that policy; see
[design.md](design.md#partial-refresh).

Curiously, `init(PART_UPDATE)` *does* hardware-reset and the base image survives,
so the SSD1681 retains RAM across reset even though registers clear. It looks
like an ordering bug in the vendor code and is not one.

### Waking a sleeping panel needs PWR restored

`sleep()` calls `module_exit()`, which drives **PWR (GP13) high = off**. The
vendor's `init()` never restores it — their demo sleeps once at the very end and
never wakes a sleeping panel. `src/epaper.py` carries a patch for this; see the
DEVIATION comment in `init()`.

This one is nasty because it fails *silently and successfully*: with the panel
unpowered, the SPI data still goes out and `ReadBusy` returns immediately (an
unpowered panel never asserts BUSY), so `display()` completes normally in about
the usual time. Only the glass never changes. The symptom is "it updated once
after boot and then stopped".

**BUSY wait time is the tell.** A genuine full refresh holds BUSY for ~1290 ms.
Zero busy time means the panel is not powered:

| | PWR | BUSY asserted | busy wait |
| --- | --- | --- | --- |
| First draw after boot | 0 | yes | 1286 ms |
| Later draws, before the fix | 1 | **no** | 0 ms |
| Later draws, after the fix | 0 | yes | 1288 ms |

## I2C devices

All four on **I2C1, SDA = GP6, SCL = GP7** (vendor uses 400 kHz; works 10 kHz–400 kHz).

| Addr | Device | Verified |
| --- | --- | --- |
| `0x70` | SHTC3 temp/humidity | yes — ID `0x0887`, reads 26.5 °C / 45.6 % RH |
| `0x51` | PCF85063A RTC | yes — decodes correctly with seconds at reg `0x04` |
| `0x38` | FT6336U capacitive touch | yes — chip ID `0xA3` = `0x64`, vendor `0xA8` = `0x11` |
| `0x18` | ES8311 audio codec | yes — initialises and plays; see Audio below |

Touch INT = **GP8**, touch RST = **GP16**. RTC INT = **GP17**.

### Touch: INT is a pulse, not a level

The board ships with FT6336U register `0xA4` = `0x01` — FocalTech's *trigger*
mode, where INT emits a brief pulse per touch frame rather than sitting low while
held. Polling that pin catches a pulse occasionally and misses it the rest of the
time, which looks exactly like "it worked once and then died".

**Read `TD_STATUS` (register `0x02`) over I2C instead.** It is a level, and the
read also clears the controller's pending interrupt. `main.Inputs` does this.

Register `0x86` reads `0x01` (monitor / low-power mode) — the default, and fine;
the controller still wakes on touch.

**GP16 is not held high by anything on the board.** An internal pull-down wins
on that pin, so the touch controller's reset must be driven explicitly or its
state is whatever leakage decides. Verified by pull-up/pull-down probing.

## Audio — ES8311

Control over I2C1 at `0x18`; data over I2S driven by **PIO**, because
MicroPython's `machine.I2S` does not emit MCLK and this codec needs it.

| Signal | GPIO |
| --- | --- |
| PA enable | 0 |
| DOUT (MCU → codec) | 1 |
| DIN (codec → MCU) | 2 |
| MCLK | 3 |
| BCLK | 4 |
| LRCLK | 5 |

The codec is I2S **master** — it drives BCLK and LRCLK, and the PIO program waits
on them as inputs.

### The PIO wants 32-bit frames

`audio_pio_out` pulls **one 32-bit word per stereo frame** and shifts bits 31..16
out as left, then 15..0 as right. So a playback buffer is one 32-bit word per
frame, not two int16s.

Feeding it int16s via `dma_play_from_i16` makes each frame consume half a frame
of audio: the clip plays an octave low at exactly double length (measured 1084 ms
for a 540 ms clip). Use `dma_play_words` with packed
`(left << 16) | right` words instead.

Also: `AudioPIO.__init__` only stores configuration. The state machines stay
`None` until `mclk_pio_init()` and `dout_pio_init()` are called, and `start()`
silently does nothing without them.

### Microphone

Same codec, `DIN` on **GP2**. Bring up with `mclk_pio_init()` +
`din_pio_init()`, then `dma_record_into(buf)` with an `array("h")` — the RX path
is 16-bit mono, unlike the 32-bit packed playback path.

*Verified*: 1 s captured at 24 kHz, exact timing, noise floor around -44 dBFS at
`mic_gain=0`. `mic_gain=6` rails the input, so those units are not dB. Nothing in
the project uses it yet.

## Battery

200k / 200k divider (÷2) on **GP29** (ADC3), gated by **GP28**.

```
volts = read_u16() / 65535 * 3.390 * 2
```

`3.390` is Waveshare's calibrated reference, not the nominal 3.3 — using 3.3
under-reads by ~2.7 %.

*Verified*: reads **4.21 V**, and the 12-bit equivalent (2542) lands two counts
off the factory firmware's reported `adc_raw: 2540` / `4.20 V`.

**GP28 must be driven as an output** before GP29 means anything. Left as a
floating input, GP29 returned wildly different values on consecutive reads
(2.977 V then 0.042 V) — noise, not measurement. Waveshare drives GP28 high;
both levels read identically here, but follow the vendor.

**Then wait ~100 ms.** The divider node charges through ~100 k, so it ramps
rather than steps. Measured from a cold reset:

| after enable | reading |
| --- | --- |
| 0 ms | 0.79 V |
| 10 ms | 3.56 V |
| ~36 ms | 4.13 V |
| ~86 ms | 4.21 V (settled) |

An immediate read reports a *plausible-looking* low voltage rather than an
obvious error, so this fails silently — it would read as a flat battery.
`board.Battery` sleeps 150 ms before its first sample.

Order matters too: construct the `ADC` object **before** the settling delay.
`ADC(Pin(29))` switches the pin into analog mode, which restarts the charge, so
sleeping and *then* creating the ADC buys nothing and still reads 0.79 V.

## microSD

On the **other** SPI block — SPI0, SCK = GP18, MOSI = GP19, MISO = GP20, CS = GP23.
Not yet tested; no card was inserted.

## Buttons

- **BOOTSEL** — not a GPIO; read via `rp2.bootsel_button()`.
- **POWER** — GP24, active-low, needs a pull-up.

## Gotchas

**Scanning the wrong I2C pins reports 112 devices.** GP0/GP1, GP2/GP3 and
GP28/GP29 all ACK every address `0x08`–`0x77`. That is a floating SDA, not
devices. Only GP6/GP7 is a real bus.

**One I2C block, one pin pair.** Instantiating `I2C(1, ...)` on GP2/GP3 and then
again on GP6/GP7 wedges the block: every transfer fails `EIO` while `scan()`
keeps succeeding. `mpremote reset` clears it.

## Flash size caveat

The stock `RPI_PICO2` MicroPython build assumes 4 MB, so the filesystem is
**3 MB of the 16 MB present** and the other 13 MB is unaddressed. Recovering it
means building MicroPython with the correct flash size.

## Still open

- **microSD** — untested; no card was ever inserted. Pins are SPI0 SCK 18,
  MOSI 19, MISO 20, CS 23.
- **Microphone** — captures cleanly (verified: 1 s at 24 kHz, noise floor around
  -44 dBFS at gain 0, railed at gain 6, so the gain units are not dB), but
  nothing uses it. It is the obvious route to recording a real clip in place of
  a synthesised one.
- **The other 13 MB of flash** — the stock `RPI_PICO2` build assumes a 4 MB part.
  Recovering it needs a custom MicroPython build.

Everything else on the board has been driven: display (full and partial refresh),
touch, SHTC3, RTC, battery sense and the ES8311 codec.
