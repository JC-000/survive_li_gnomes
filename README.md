# survive_li_gnomes

MicroPython on an RP2350A board with an SHTC3 temp/humidity sensor, a PCF85063A
real-time clock, and a FocalTech capacitive touch controller.

> **Project goal: not yet defined.** The repo is scaffolding + a verified hardware
> baseline. See [docs/hardware.md](docs/hardware.md) for what is actually on the
> board and what is still unknown.

## Quick start

The host needs nothing installed beyond `uv` — `mpremote` runs via `uvx`.

```sh
# open a REPL (Ctrl-] to exit)
uvx mpremote connect /dev/cu.usbmodem101

# read the sensors once
uvx mpremote connect /dev/cu.usbmodem101 run tools/probe.py

# copy the board module onto the device
uvx mpremote connect /dev/cu.usbmodem101 cp src/board.py :board.py
```

## Layout

| Path | What |
| --- | --- |
| `src/board.py` | Verified pin map + SHTC3 and PCF85063A drivers |
| `tools/probe.py` | Bus scan / device identification, run from the host |
| `docs/hardware.md` | Measured hardware facts and open questions |
| `docs/restore-factory-firmware.md` | Putting the original firmware back |
| `firmware/backup/` | Factory firmware dump taken before reflashing |
| `.serena/` | Serena project config and onboarding memories |

## Board state

Flashed with MicroPython **v1.28.0** (`RPI_PICO2`, ARM build). The factory C
firmware was dumped first and is fully restorable — see
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md).
