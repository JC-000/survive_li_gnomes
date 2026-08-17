# CLAUDE.md

MicroPython on an RP2350A board. Host is macOS.

**The project's goal is not defined yet** — the repo is scaffolding plus a
verified hardware baseline. Ask before building features; don't infer intent
from the directory name.

## Before touching hardware code

Read `docs/hardware.md`. Every constant in `src/board.py` was measured from the
physical board, not taken from a datasheet or product page. If something
contradicts it, re-measure rather than assuming the doc is stale.

The board model is **unidentified**, so there is no reference pinout to fall back
on. Don't guess pin assignments — probe them.

## MicroPython, not CPython

`machine`, `rp2`, `micropython` and `time.sleep_ms` do not exist on the host. The
Python language server will flag them. That is expected — do not rewrite them to
CPython equivalents to silence it.

## Talking to the board

Everything goes through `uvx mpremote connect /dev/cu.usbmodem101 ...`; nothing
is installed globally. See `.serena/memories/dev_workflow.md` for the command
set. After `reset` or `bootloader`, wait for the serial device to reappear.

## Two failure modes that waste time

- An I2C scan on a pin pair with no pull-ups ACKs **every** address `0x08`-`0x77`.
  That means nothing is there, not that 112 devices are.
- Creating `I2C(1)` on a second pin pair wedges the block: `scan()` keeps working
  while every read fails `EIO`. Fix with `mpremote reset`.

## Firmware

The board shipped with vendor C firmware, dumped before reflashing. Restoring it
is documented in `docs/restore-factory-firmware.md`. `firmware/backup/factory-flash-16MB.bin`
is gitignored and exists only on this machine — don't delete it casually.
