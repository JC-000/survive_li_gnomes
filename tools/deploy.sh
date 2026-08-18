#!/bin/sh
# Copy the device-side modules to the board and restart it.
# main.py autoruns at power-on, so this is all that is needed to "install".
#
#   ./tools/deploy.sh              # default port
#   PORT=/dev/cu.usbmodem1101 ./tools/deploy.sh
set -eu

PORT="${PORT:-/dev/cu.usbmodem101}"
cd "$(dirname "$0")/.."

# One file per invocation: mpremote's multi-file cp is fussy about argument form.
for module in board epaper magic8 sounds shake es8311 audio_pio_mpy main; do
    echo "  -> ${module}.py"
    uvx --quiet mpremote connect "$PORT" cp "src/${module}.py" ":${module}.py"
done

# Sampled clips. Only copied when missing or changed would be nicer, but these
# are small and mpremote has no sync.
for clip in clips/*.raw; do
    [ -e "$clip" ] || continue
    echo "  -> $(basename "$clip")"
    uvx --quiet mpremote connect "$PORT" cp "$clip" ":$(basename "$clip")"
done

echo "resetting"
uvx --quiet mpremote connect "$PORT" reset

echo "done. Press POWER, BOOTSEL, or tap the screen."
