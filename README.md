# survive_li_gnomes

MicroPython on a **Waveshare RP2350-Touch-ePaper-1.54** — RP2350A, 16 MB flash,
200 × 200 SSD1681 e-paper, FT6336U touch, SHTC3 temp/humidity, PCF85063A RTC,
ES8311 audio codec, microSD, Li-ion battery.

> **Project goal: not yet defined.** What's here is a working hardware baseline —
> every peripheral below has been driven on the real board.

## Quick start

The host needs nothing but `uv` — `mpremote` runs via `uvx`.

```sh
# copy the board modules onto the device
uvx mpremote connect /dev/cu.usbmodem101 cp src/board.py src/epaper.py :

# identify everything on the bus and read the sensors
uvx mpremote connect /dev/cu.usbmodem101 run tools/probe.py

# draw sensors + battery + clock to the e-paper
uvx mpremote connect /dev/cu.usbmodem101 run examples/display_status.py

# REPL (Ctrl-] to exit)
uvx mpremote connect /dev/cu.usbmodem101
```

## What works

| Peripheral | Status |
| --- | --- |
| E-paper 200 × 200 | Full refresh verified |
| SHTC3 temp/humidity | Verified — 26.5 °C / 45.6 % RH |
| PCF85063A RTC | Verified — reads and sets |
| FT6336U touch | Verified — chip ID `0x64` |
| Battery sense | Verified — 4.21 V, matches factory firmware |
| ES8311 audio | ACKs on I2C only, not driven |
| microSD | Untested, no card inserted |

## Layout

| Path | What |
| --- | --- |
| `src/board.py` | Pin map + SHTC3, PCF85063A, FT6336U, Battery |
| `src/epaper.py` | SSD1681 driver, vendored from Waveshare (MIT) |
| `examples/display_status.py` | Sensor readout on the display |
| `tools/probe.py` | Bus scan / device ID, run from the host |
| `docs/hardware.md` | Full pinout, what's verified, and the gotchas |
| `docs/restore-factory-firmware.md` | Putting the original firmware back |
| `firmware/backup/` | Factory firmware dump, taken before reflashing |

## Board state

Flashed with MicroPython **v1.28.0** (`RPI_PICO2`, ARM). The factory C firmware
was dumped first and is restorable — see
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md).

Note the filesystem is **3 MB**, not 16 — the stock `RPI_PICO2` build assumes a
4 MB part.

## Credits

Pin definitions and the e-paper driver come from Waveshare's
[demo repository](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54)
(Apache-2.0; the e-paper driver file itself is MIT).
