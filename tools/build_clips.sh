#!/bin/sh
# Regenerate the sampled clips in clips/ from their sources.
#
# The peak values here are tuning, not defaults -- they were set by listening.
# The laughter source sits at 43.6% of full scale; normalising it to the usual
# 30000 applied +6.4 dB and overdrove the amp, so it is held near its original
# recorded level instead.
#
#   ./tools/build_clips.sh            # sources from ~/Downloads
#   SRC_DIR=/some/where ./tools/build_clips.sh
set -eu

SRC_DIR="${SRC_DIR:-$HOME/Downloads}"
cd "$(dirname "$0")/.."
mkdir -p clips

laughter="$SRC_DIR/katjasavia-female-sensual-laughter-218077.mp3"
if [ ! -f "$laughter" ]; then
    echo "missing source: $laughter" >&2
    echo "set SRC_DIR, or archive the file alongside the repo." >&2
    exit 1
fi

uvx --quiet --from miniaudio python tools/make_clip.py \
    "$laughter" clips/laugh.raw --seconds 1.5 --peak 15000

echo
echo "run ./tools/deploy.sh to push the clips to the board"
