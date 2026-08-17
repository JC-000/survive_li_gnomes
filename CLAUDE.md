# CLAUDE.md

A Magic 8-Ball on persistent e-paper. MicroPython on a Waveshare
**RP2350-Touch-ePaper-1.54**. Host is macOS.

## The governing constraint: intermittent power

Read `docs/design.md` before changing `src/main.py`. In short: boot never touches
the panel (e-paper already holds the last answer), nothing is ever written to
flash, and the panel is always put back to sleep after a refresh.

**Do not add persistence.** Saving the last answer to a file would reintroduce
the one real corruption risk — a flash write interrupted by a power cut — to
duplicate what the display already does for free.

Deploy with `./tools/deploy.sh`; `main.py` autoruns at power-on.

## Before touching hardware code

Read `docs/hardware.md`. It marks which pins are merely from Waveshare's demo
code and which were confirmed on this physical board. Don't guess pins — the
vendor repo is cloneable and authoritative:
https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54

`src/epaper.py` is vendored Waveshare code (MIT). Its LUT tables and SSD1681
init sequence are panel-specific magic numbers — don't tidy them.

## Battery reads need GP28 driven

GP29 is the sense pin but returns pure noise unless GP28 is first driven as an
output. Use `board.Battery()`, which handles it. And use the calibrated 3.390 V
reference, not 3.3.

## MicroPython, not CPython

`machine`, `rp2`, `micropython` and `time.sleep_ms` do not exist on the host. The
Python language server will flag them. That is expected — do not rewrite them to
CPython equivalents to silence it.

## Talking to the board

Everything goes through `uvx mpremote connect /dev/cu.usbmodem101 ...`; nothing
is installed globally. See `.serena/memories/dev_workflow.md` for the command
set. After `reset` or `bootloader`, wait for the serial device to reappear.

## Failure modes that waste time

See `.serena/memories/gotchas_that_cost_time.md` for the full list. The big ones:

- An I2C scan on a pin pair with no pull-ups ACKs **every** address `0x08`-`0x77`.
  That means nothing is there, not that 112 devices are.
- Creating `I2C(1)` on a second pin pair wedges the block: `scan()` keeps working
  while every read fails `EIO`. Fix with `mpremote reset`.
- Never poll `rp2.bootsel_button()` alongside `_thread` — it takes over the QSPI
  bus and produces phantom presses.
- **Test power-related code from a cold reset.** State left by a previous
  command (a charged divider, a configured pin) makes broken code look correct.

## Firmware

The board shipped with vendor C firmware, dumped before reflashing. Restoring it
is documented in `docs/restore-factory-firmware.md`. `firmware/backup/factory-flash-16MB.bin`
is gitignored and exists only on this machine — don't delete it casually.
