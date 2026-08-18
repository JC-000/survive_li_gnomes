# survive_li_gnomes

A Magic 8-Ball on a persistent e-paper display. Press a button, hear it shake,
get an answer. The answer stays on screen with the power off, so it survives the
intermittent power this thing lives on.

Hardware: **Waveshare RP2350-Touch-ePaper-1.54** — RP2350A, 16 MB flash,
200 × 200 SSD1681 e-paper, FT6336U touch, SHTC3, PCF85063A RTC, ES8311 codec,
microSD, Li-ion battery.

## Install

The host needs nothing but `uv` — `mpremote` runs via `uvx`.

```sh
./tools/deploy.sh
```

That copies the device modules and resets the board. `main.py` autoruns at
power-on, so there is nothing else to start.

## Use

Press the **POWER key**, **BOOTSEL**, or **tap the screen** — any of the three
asks the ball. The board has no brightness button; e-paper has no backlight.

The shake sound plays *while* the panel refreshes — about 1.6 s end to end.
Sound needs a speaker on the board's connector; without one everything still
works silently. The first press after power-on takes ~2 s longer while the codec
comes up and the clip is generated.

Answers are drawn at the largest size they fit: short ones like "Yes" at 24 px,
most at 16 px.

Every so often the shake is replaced by a fart or a laugh — never more often than
once every five presses, averaging about one in eight. Tune it via
`ALTERNATE_MIN_GAP` / `ALTERNATE_ONE_IN` in `src/shake.py`.

To change a sampled clip, edit `tools/build_clips.sh`, run it, then
`./tools/deploy.sh`. If a clip sounds overdriven, lower its `--peak`.

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
| `src/magic8.py` | The 44 answers, RNG, and screen rendering |
| `src/sounds.py` | Waveform synthesis — shake and fart. Pure DSP, host-testable |
| `src/shake.py` | Codec, clip cache, and which sound a press gets |
| `src/es8311.py`, `src/audio_pio_mpy.py` | Codec + I2S-over-PIO, vendored from Waveshare |
| `src/board.py` | Pin map + SHTC3, PCF85063A, FT6336U, Battery |
| `src/epaper.py` | SSD1681 driver, vendored from Waveshare (MIT) |
| `examples/display_status.py` | Sensor/battery/clock readout |
| `tools/probe.py` | Bus scan + battery, run from the host |
| `tools/input_monitor.py` | Log input transitions — use if a button misbehaves |
| `tools/deploy.sh` | Copy modules and clips to the board and reset |
| `tools/make_clip.py` | Convert audio to a raw clip the codec can DMA directly |
| `tools/build_clips.sh` | Rebuild `clips/` — holds the per-clip level tuning |
| `docs/design.md` | Why it is built this way |
| `docs/hardware.md` | Full pinout, what's verified, and the gotchas |
| `docs/restore-factory-firmware.md` | Putting the original firmware back |

## Hardware status

| Peripheral | Status |
| --- | --- |
| E-paper 200 × 200 | Working |
| FT6336U touch | Working — polled over I2C, not via the INT pin |
| Battery sense | Working — 4.21 V, matches factory firmware |
| SHTC3 temp/humidity | Working — used by `examples/display_status.py` |
| PCF85063A RTC | Working |
| ES8311 audio | Working — plays the shake clip |
| microSD | Untested, no card inserted |

Running MicroPython **v1.28.0** (`RPI_PICO2`, ARM). The factory C firmware was
dumped before reflashing and is restorable — see
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md). Note the
filesystem is 3 MB, not 16 — the stock build assumes a 4 MB part.

## Credits

Pin definitions and the e-paper driver come from Waveshare's
[demo repository](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54)
(Apache-2.0; the e-paper driver file itself is MIT).
