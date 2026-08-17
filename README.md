# survive_li_gnomes

A Magic 8-Ball on a persistent e-paper display. Press a button, get an answer.
The answer stays on screen with the power off, so it survives the intermittent
power this thing lives on.

Hardware: **Waveshare RP2350-Touch-ePaper-1.54** — RP2350A, 16 MB flash,
200 × 200 SSD1681 e-paper, FT6336U touch, SHTC3, PCF85063A RTC, ES8311 codec,
microSD, Li-ion battery.

## Install

The host needs nothing but `uv` — `mpremote` runs via `uvx`.

```sh
./tools/deploy.sh
```

That copies the four device modules and resets the board. `main.py` autoruns at
power-on, so there is nothing else to start.

## Use

Press the **POWER key**, **BOOTSEL**, or **tap the screen** — any of the three
asks the ball. The board has no brightness button; e-paper has no backlight.

A refresh takes a few seconds and flashes the panel black/white. That is normal
for e-paper, not a fault.

## Behaviour under power loss

| | |
| --- | --- |
| Power cut while idle | Answer stays on screen. Boot does not redraw it. |
| Power cut mid-refresh | Panel may be left mid-transition; next press fixes it. |
| Flash writes | None, ever. Nothing to corrupt. |

See [docs/design.md](docs/design.md) for why.

## Layout

| Path | What |
| --- | --- |
| `src/main.py` | Device entry point — input loop, autoruns at power-on |
| `src/magic8.py` | The twenty answers, RNG, and screen rendering |
| `src/board.py` | Pin map + SHTC3, PCF85063A, FT6336U, Battery |
| `src/epaper.py` | SSD1681 driver, vendored from Waveshare (MIT) |
| `examples/display_status.py` | Sensor/battery/clock readout |
| `tools/probe.py` | Bus scan + battery, run from the host |
| `tools/deploy.sh` | Copy modules to the board and reset |
| `docs/design.md` | Why it is built this way |
| `docs/hardware.md` | Full pinout, what's verified, and the gotchas |
| `docs/restore-factory-firmware.md` | Putting the original firmware back |

## Hardware status

| Peripheral | Status |
| --- | --- |
| E-paper 200 × 200 | Working |
| FT6336U touch | Working — chip ID `0x64` |
| Battery sense | Working — 4.21 V, matches factory firmware |
| SHTC3 temp/humidity | Working — used by `examples/display_status.py` |
| PCF85063A RTC | Working |
| ES8311 audio | ACKs on I2C only, not driven |
| microSD | Untested, no card inserted |

Running MicroPython **v1.28.0** (`RPI_PICO2`, ARM). The factory C firmware was
dumped before reflashing and is restorable — see
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md). Note the
filesystem is 3 MB, not 16 — the stock build assumes a 4 MB part.

## Credits

Pin definitions and the e-paper driver come from Waveshare's
[demo repository](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54)
(Apache-2.0; the e-paper driver file itself is MIT).
