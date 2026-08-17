# Hardware

Everything here was measured from the board on 2026-08-17, not read off a
datasheet or a product page. The exact board model has not been identified.

## MCU

| | |
| --- | --- |
| Chip | RP2350A (ARM Cortex-M33, Secure image) |
| Clock | 150 MHz |
| RAM | 520 KB (491 KB free under MicroPython) |
| Flash | 16 MB |
| USB serial | `/dev/cu.usbmodem101`, VID `2E8A` PID `0009` |
| Board UID | `9717825cf3bcf97c` |

## I2C devices

All four sit on **I2C1, SDA = GP6, SCL = GP7**. Verified working at 10 kHz
through 400 kHz.

| Addr | Device | Status |
| --- | --- | --- |
| `0x70` | **SHTC3** temp/humidity | Confirmed — ID register `0x0887` matches the SHTC3 mask; reads 26.18 °C / 46.51 % RH |
| `0x51` | **PCF85063A** RTC | Confirmed — decodes to `2026-01-01 12:24:09` with the PCF85063A register layout (time at `0x04`, *not* the PCF8563 layout at `0x02`); oscillator-stopped flag clear |
| `0x38` | **FocalTech FT6236/FT6336** capacitive touch | Confirmed — vendor ID `0xA8` = `0x11` (FocalTech), chip ID `0xA3` = `0x64`. Implies the board has a touchscreen |
| `0x18` | **unidentified** | ACKs and reads, but register `0x00` returns `0x1F` and every other register returns `0xFF`. Not LIS3DH (WHO_AM_I would be `0x33`) and not MCP9808. Ignores register addressing — looks like a device that returns one status byte per read |

### Gotcha: false positives when scanning

Scanning GP0/GP1, GP2/GP3 and GP28/GP29 reports *every* address from `0x08` to
`0x77`. That is the signature of a floating SDA line with no pull-ups, not 112
devices. Only GP6/GP7 is a real bus.

### Gotcha: one I2C block, one pin pair

Instantiating `I2C(1, ...)` on GP2/GP3 and then again on GP6/GP7 wedges the
block — every transaction fails with `EIO` even though `scan()` still succeeds.
A hard reset (`mpremote reset`) clears it. Do not brute-force pin pairs on a bus
you are already using.

## ADC / battery

**Unresolved.** The factory firmware reported `Battery: 4.20 V` from
`adc_raw: 2540` (12-bit), which implies roughly 2.05 V at the pin behind a
÷2 divider. Under MicroPython no ADC pin matches:

| Pin | first read | second read, minutes later |
| --- | --- | --- |
| GP26 | 59118 (2.977 V) | 832 (0.042 V) |
| GP27 | 58158 (2.929 V) | 1888 (0.095 V) |
| GP28 | 432 (0.022 V) | 288 (0.015 V) |
| GP29 | 60526 (3.048 V) | 7313 (0.368 V) |

The two runs disagree by nearly full scale on three of four pins, which settles
it: these inputs are floating and the values are noise, not measurements. The
most likely explanation is that the battery divider sits behind an enable
transistor that the factory firmware drove high and MicroPython leaves low — so
the battery-sense *and* its enable GPIO both still need to be found before
battery reporting will work.

Core temperature via `ADC(4)` works and read 42.5 °C.

## Open questions

1. What board is this? Knowing the model resolves the battery-sense pin, the
   `0x18` device, and the LCD/SPI pin map in one step.
2. Which GPIO enables the battery divider?
3. What is at `0x18`?
4. Where is the display? A FocalTech touch controller implies a panel, but no
   SPI/parallel pins have been probed yet.

## Flash size caveat

The stock `RPI_PICO2` MicroPython build assumes 4 MB of flash, so the filesystem
is **3 MB of the 16 MB present** — the other 13 MB is unaddressed. Fine for now;
recovering it means building MicroPython with the correct flash size.
