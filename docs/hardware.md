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

*Verified* — full-refresh draw completed and the panel holds the image.

## I2C devices

All four on **I2C1, SDA = GP6, SCL = GP7** (vendor uses 400 kHz; works 10 kHz–400 kHz).

| Addr | Device | Verified |
| --- | --- | --- |
| `0x70` | SHTC3 temp/humidity | yes — ID `0x0887`, reads 26.5 °C / 45.6 % RH |
| `0x51` | PCF85063A RTC | yes — decodes correctly with seconds at reg `0x04` |
| `0x38` | FT6336U capacitive touch | yes — chip ID `0xA3` = `0x64`, vendor `0xA8` = `0x11` |
| `0x18` | ES8311 audio codec | partially — ACKs, reset reg `0x00` = `0x1F`; not driven yet |

Touch INT = **GP8**, touch RST = **GP16**. RTC INT = **GP17**.
ES8311 power-amp control = **GP0**.

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

- ES8311 audio has not been brought up.
- microSD untested.
- The e-paper partial-update path (`displayPartBaseImage` / `init(part_update)`)
  is vendored but unexercised.
