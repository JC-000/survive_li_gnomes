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

A second program shares the board. Hold the screen, say something, let go, and
Weizenbaum's 1966 DOCTOR answers on the e-paper.

```sh
./tools/deploy.sh eliza     # the 8-ball is still the bare default
```

It cannot transcribe speech — nothing that fits 490 KB of RAM can. Instead it
spots about 21 words with DTW template matching and leans on the fact that
DOCTOR was always a keyword matcher: unrecognised speech falls through to "Please
go on", which is a real DOCTOR response rather than an error message. A shy
recogniser and the program's own personality are the same thing, so it is tuned
for precision over recall.

**It needs enrolling before it works.** Templates are your voice, recorded
through the board's own microphone — not the Mac's, because a recogniser
enrolled on one microphone and run through another loses most of its margin.
With the board connected:

```sh
# 1. sample rate, FFT and match timings — runs on the board, src/ mounted
uvx mpremote connect /dev/cu.usbmodem101 mount src run tools/speech_probe.py

# 2. transport: achieved KB/s, and the real noise floor
uvx --from pyserial python tools/pull_recording.py t.wav

# 3. does the recogniser have any margin? both mics at once, ~10 min
uvx --from miniaudio --with pyserial python tools/mic_margin.py record run1/

# 4. enrolment proper: ~10 min of saying 21 words
uvx --from pyserial python tools/enrol.py takes/
```

Steps 1 and 2 calibrate what step 4 depends on, and step 3 says whether the
recogniser has any margin at all before you spend ten minutes on it. Running
them out of order risks a whole vocabulary recorded at the wrong sample rate.

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
| `src/spotter.py`, `src/speech_tables.py` | On-device MFCC + DTW keyword spotter |
| `src/listen.py`, `src/vad.py`, `src/record_stream.py` | Capture, endpointing, streaming to the host |
| `src/vocab.py`, `src/screen.py` | The spotted word list; full-panel text rendering |
| `src/board.py` | Pin map + SHTC3, PCF85063A, FT6336U, Battery |
| `src/epaper.py` | SSD1681 driver, vendored from Waveshare (MIT) |
| `examples/display_status.py` | Sensor/battery/clock readout |
| `tools/probe.py` | Bus scan + battery, run from the host |
| `tools/input_monitor.py` | Log input transitions — use if a button misbehaves |
| `tools/deploy.sh` | Copy modules and clips to the board and reset |
| `tools/make_clip.py` | Convert audio to a raw clip the codec can DMA directly |
| `tools/build_clips.sh` | Rebuild `clips/` — holds the per-clip level tuning |
| `tools/enrol.py`, `tools/pull_recording.py` | Record your voice through the board, over USB |
| `tools/record_templates.py`, `tools/mfcc.py`, `tools/dtw.py` | Host reference for the spotter; builds templates |
| `tools/mic_margin.py` | Does the recogniser have any margin? Run this first |
| `tools/speech_probe.py` | Sample rate, FFT and match timings — needs the board |
| `tools/eliza_repl.py` | Converse with DOCTOR on the host, no hardware |
| `tools/test_*.py` | Six suites, all runnable without the board |
| `docs/design.md` | Why it is built this way |
| `docs/hardware.md` | Full pinout, what's verified, and the gotchas |
| `docs/speech-design.md` | Why speech is done this way, and what was ruled out |
| `docs/speech.md` | The feature spec — normative, to the last integer |
| `docs/restore-factory-firmware.md` | Putting the original firmware back |

## Hardware status

| Peripheral | Status |
| --- | --- |
| E-paper 200 × 200 | Working — full and partial refresh |
| FT6336U touch | Working — polled over I2C, not via the INT pin |
| Battery sense | Working — 4.21 V, matches factory firmware |
| SHTC3 temp/humidity | Working — used by `examples/display_status.py` |
| PCF85063A RTC | Working |
| ES8311 audio | Working — plays the shake, fart and laugh clips |
| ES8311 microphone | Captures cleanly at 24 kHz; the ELIZA program wants 16 kHz, **unverified** |
| microSD | Untested, no card inserted |

Running MicroPython **v1.28.0** (`RPI_PICO2`, ARM). The factory C firmware was
dumped before reflashing and is restorable — see
[docs/restore-factory-firmware.md](docs/restore-factory-firmware.md). Note the
filesystem is 3 MB, not 16 — the stock build assumes a 4 MB part.

## Credits

Pin definitions and the e-paper driver come from Waveshare's
[demo repository](https://github.com/waveshareteam/RP2350-Touch-ePaper-1.54)
(Apache-2.0; the e-paper driver file itself is MIT).
