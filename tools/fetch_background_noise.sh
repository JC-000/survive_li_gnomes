#!/bin/sh
# Fetch Speech Commands' `_background_noise_` folder -- and nothing else.
#
# Google Speech Commands v0.02 overlaps our vocabulary by exactly two words
# (YES and NO, and NO was cut as a homophone of "know"), so the 105,829
# utterances are of no use here. The six background recordings are: they are
# the standard noise source for keyword-spotting augmentation, and real room
# noise is a much harder test of a spectral front end than the LCG hiss
# tools/say_corpus.py adds.
#
#   doing_the_dishes  dude_miaowing  exercise_bike  running_tap
#   pink_noise  white_noise            ~60 s each, 16 kHz mono, 12.7 MB total
#
# The archive is 2.3 GB and offers no way to ask for one folder, but the folder
# happens to sit near the front of the tar stream -- measured, it arrives in
# under 30 s on a normal connection -- so this streams, extracts the six files
# and kills the download the moment they are all present. Without the kill it
# would pull the remaining 2.3 GB to add nothing.
#
#   ./tools/fetch_background_noise.sh
#   DEST=/some/where ./tools/fetch_background_noise.sh
#
# CC BY 4.0, Pete Warden, July 2017. See the README the download brings with it.
set -eu

URL=http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
cd "$(dirname "$0")/.."
DEST="${DEST:-corpus_noise}"
FOLDER="$DEST/_background_noise_"
WANT=6          # the six .wav files; the README arrives alongside them

if [ "$(ls "$FOLDER"/*.wav 2>/dev/null | wc -l)" -ge "$WANT" ]; then
    echo "already have $FOLDER"
    exit 0
fi

mkdir -p "$DEST"
echo "streaming $URL, stopping once $WANT files have arrived..."
( cd "$DEST" && curl -sS "$URL" | tar -xzf - --include '*_background_noise_*' ) &
pid=$!

# Poll rather than wait: `tar` cannot tell us it has passed the folder, and
# every second after the sixth file is bandwidth spent on words we cannot use.
waited=0
while kill -0 "$pid" 2>/dev/null; do
    have=$(ls "$FOLDER"/*.wav 2>/dev/null | wc -l | tr -d ' ')
    if [ "$have" -ge "$WANT" ]; then
        # One more second so the last file finishes writing before the pipe
        # closes under it -- the count goes up when a file is created, not
        # when it is complete.
        sleep 1
        pkill -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        break
    fi
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -gt 600 ]; then
        echo "gave up after 10 minutes with $have of $WANT files" >&2
        kill "$pid" 2>/dev/null || true
        exit 1
    fi
done
wait "$pid" 2>/dev/null || true

ls -l "$FOLDER"
echo
echo "tools/train_corpus.py picks these up automatically."
