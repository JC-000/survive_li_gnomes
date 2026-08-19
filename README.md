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

The shake sound plays *while* the panel refreshes — about 0.6 s end to end for
a normal press. Every ninth press does a slower full refresh that flashes the
panel, to scrub the ghosting partial refreshes leave behind; so does the first
press after the panel has been idle a minute.
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

The occasional flashing black/white refresh is normal for e-paper, not a fault —
it is what clears ghosting.

## The other program: push-to-talk ELIZA

A second program shares the board. Hold the screen, wait for the chirp, say
something, let go — Weizenbaum's 1966 DOCTOR answers on the e-paper, **and out
loud** through the speaker.

```sh
./tools/build_firmware.sh                  # MicroPython + 16 MB flash + TFLM, flash it once
uv run tools/voice_pak.py corpus-voice/    # render the voice (~2 min of `say`)
./tools/deploy.sh eliza                    # the 8-ball is still the bare default
```

It cannot transcribe speech — nothing that fits 490 KB of RAM can. Instead it
spots 21 keywords and leans on the fact that DOCTOR was always a keyword
matcher: unrecognised speech falls through to "Please go on", which is a real
DOCTOR response rather than an error message. A shy recogniser and the
program's own personality are the same thing, so it is tuned for precision
over recall.

**It works out of the box — no enrolment.** The spotter that ships is a
speaker-independent CNN (`models/si_real.tflite`, a DS-CNN after ARM's
"Hello Edge" paper) trained entirely on synthetic macOS `say` voices — no
human recording is in the weights; see [models/README.md](models/README.md)
for provenance. Measured on this board against a real speaker held out from
everything: **precision 1.000, recall 0.600**. It runs through a TFLite Micro
usermod compiled into the firmware image, chosen because it is bit-identical
to the host reference kernels — the TinyMaix backend it replaced computed
different probabilities from the same weights
([docs/tflm-usermod.md](docs/tflm-usermod.md)). Training your own model is one
script: [tools/si_train.py](tools/si_train.py) on a corpus from
[tools/say_corpus.py](tools/say_corpus.py); the full story is in
[docs/speaker-independent.md](docs/speaker-independent.md) and
[docs/cnn-on-device.md](docs/cnn-on-device.md).

**The replies are spoken.** The board cannot synthesise speech at acceptable
quality, so every line DOCTOR can reach (113 clips) is rendered on the Mac by
`say`, IMA-ADPCM encoded, and packed into one ~1.9 MB `voice.pak` that streams
from flash while the panel refreshes ([docs/speech-voice.md](docs/speech-voice.md)).
Pick a different voice with `tools/voice_audition.py`. Without the pak the toy
still works, silently — that is the designed degradation, same as the 8-Ball
without a speaker.

### The parked spotter: DTW template matching

The CNN replaced an MFCC + DTW template matcher, which is parked, not deleted:
`talk.py` falls back to it if the CNN is unavailable. It is speaker-*dependent*
— templates are your voice, recorded through the board's own microphone (not
the Mac's, because a recogniser enrolled on one microphone and run through
another loses most of its margin). Only if you want that path:

```sh
# 1. sample rate, FFT and match timings — runs on the board, src/ mounted
uvx mpremote connect /dev/cu.usbmodem101 mount src run tools/speech_probe.py

# 2. transport: achieved KB/s, and the real noise floor
uvx --from pyserial python tools/pull_recording.py t.wav

# 3. does the recogniser have any margin? both mics at once, ~10 min
uvx --from miniaudio --with pyserial python tools/mic_margin.py record run1/

# 4. enrolment proper: ~10 min of saying 21 words
uvx --from pyserial python tools/enrol.py enrol-takes/

# 5. turn the recordings into the template set, and deploy it
python3 tools/record_templates.py --from enrol-takes/ --pack full
./tools/deploy.sh eliza
```

**Do not enrol into `takes/`.** It looks like the obvious name and it is
already taken: `takes/` and `takes-oov/` hold the ten keywords and twelve
negatives recorded from a real person through this board, and they are the
held-out test set that every speaker-independence figure in
[docs/speaker-independent.md](docs/speaker-independent.md) is measured against.
`tools/corpus.py` goes to some trouble to keep them out of training — there is
no flag that mixes them — and re-recording over them would undo that quietly,
since a fresh `manifest.json` in the same schema is exactly what `enrol.py`
writes. They are gitignored, so there is nothing to restore them from.

Try the conversation half without any hardware:

```sh
uv run tools/eliza_repl.py --bag --show
```

`--bag` throws away word order and keeps only vocabulary words, which is exactly
what the spotter delivers. See [docs/speech-design.md](docs/speech-design.md)
for why it is built this way and [docs/speech.md](docs/speech.md) for the
feature spec.

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
| `src/talk.py` | ELIZA entry point — hold to talk, spot, reply |
| `src/eliza.py`, `src/eliza_rules.py` | The 1966 DOCTOR script and its interpreter |
| `src/si_spot.py`, `src/si_patch.py` | The CNN spotter: gates, backends, and the log-mel front end |
| `src/spotter.py`, `src/speech_tables.py` | The parked MFCC + DTW keyword spotter |
| `src/adpcm.py` | IMA ADPCM decoder + `voice.pak` reader — how the replies get spoken |
| `models/` | The shipping CNN weights and their provenance |
| `src/listen.py`, `src/vad.py`, `src/record_stream.py` | Capture, endpointing, streaming to the host |
| `src/vocab.py`, `src/screen.py` | The spotted word list; full-panel text rendering |
| `src/board.py` | Pin map + SHTC3, PCF85063A, FT6336U, Battery |
| `src/epaper.py` | SSD1681 driver, vendored from Waveshare (MIT) |
| `examples/display_status.py` | Sensor/battery/clock readout |
| `tools/probe.py` | Bus scan + battery, run from the host |
| `tools/input_monitor.py` | Log input transitions — use if a button misbehaves |
| `tools/deploy.sh` | Copy modules and clips to the board and reset |
| `tools/build_firmware.sh` | Build MicroPython with all 16 MB of flash addressed |
| `tools/flash_capacity.py` | Write and read back a real file — proves the filesystem size |
| `tools/tflm_cases.py`, `tools/tflm_device_cases.py`, `tools/tflm_compare_cases.py` | The 30-case host-vs-device TFLM comparison, and its gate |
| `tools/make_clip.py` | Convert audio to a raw clip the codec can DMA directly |
| `tools/build_clips.sh` | Rebuild `clips/` — holds the per-clip level tuning |
| `tools/si_train.py`, `tools/say_corpus.py` | Train the speaker-independent CNN on synthetic voices |
| `tools/voice_pak.py` | Render and pack everything DOCTOR can say |
| `tools/check_banned.sh`, `tools/setup_hooks.sh` | The banned-content gate — run `setup_hooks.sh` once per clone |
| `tools/enrol.py`, `tools/pull_recording.py` | Record your voice through the board, over USB |
| `tools/record_templates.py`, `tools/mfcc.py`, `tools/dtw.py` | Host reference for the spotter; builds templates |
| `tools/mic_margin.py` | Does the recogniser have any margin? Run this first |
| `tools/speech_probe.py` | Sample rate, FFT and match timings — needs the board |
| `tools/eliza_repl.py` | Converse with DOCTOR on the host, no hardware |
| `tools/voice_audition.py` | Render DOCTOR's replies through `say`, to pick a speaking voice |
| `tools/test_*.py` | Nineteen suites, all runnable without the board |
| `docs/design.md` | Why it is built this way |
| `docs/speaker-independent.md` | The road from DTW to the CNN, with every measurement |
| `docs/cnn-on-device.md` | Getting the CNN onto the board, and what it scored there |
| `docs/tflm-usermod.md` | The TFLite Micro usermod, and why TinyMaix was replaced |
| `docs/hardware.md` | Full pinout, what's verified, and the gotchas |
| `docs/speech-design.md` | Why speech *input* is done this way, and what was ruled out |
| `docs/speech-voice.md` | Speaking the reply — voice, prosody, and what fits in 3 MB |
| `docs/speech.md` | The feature spec — normative, to the last integer |
| `docs/restore-factory-firmware.md` | Putting the original firmware back, and the two MicroPython images |
| `docs/morning-runbook.md` | Bench steps for flashing and gating the TFLM image |

## Hardware status

| Peripheral | Status |
| --- | --- |
| E-paper 200 × 200 | Working — full and partial refresh |
| FT6336U touch | Working — polled over I2C, not via the INT pin |
| Battery sense | Working — 4.21 V, matches factory firmware |
| SHTC3 temp/humidity | Working — used by `examples/display_status.py` |
| PCF85063A RTC | Working |
| ES8311 audio | Working — plays the 8-Ball's clips and streams ELIZA's spoken replies |
| ES8311 microphone | Working — 23991 Hz at 24 kHz, 15991 Hz at 16 kHz, measured against wall clock |
| microSD | Untested, no card inserted |
| Flash filesystem | Working — 15,728,640 bytes, 10 MB written and read back identically |

Running MicroPython **v1.28.0**, built for this board by
[`tools/build_firmware.sh`](tools/build_firmware.sh) for two reasons: all 16 MB
of flash is addressed — the filesystem is **15,728,640 bytes**, measured,
against the 3 MB the stock `RPI_PICO2` image formats — and the TFLite Micro
usermod that runs the CNN is compiled in ([docs/tflm-usermod.md](docs/tflm-usermod.md)). 10 MB of it has been written and read
back to prove the part answers there and does not wrap.

The stock image remains the rollback, and the factory C firmware was dumped
before any of this and is restorable — both in
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md), along with
the warning that flashing either MicroPython image reformats the filesystem.

## Credits

Pin definitions and the e-paper driver come from Waveshare's
[demo repository](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54)
(Apache-2.0; the e-paper driver file itself is MIT).
