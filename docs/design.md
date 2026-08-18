# Design notes

A Magic 8-Ball on e-paper, built to tolerate **intermittent power**. That
constraint drives most of the decisions below.

## The display is the storage

E-paper is bistable: it holds its image with the power off. So the answer on
screen survives a power cut for free, and there is nothing to persist.

Three consequences:

1. **Boot never touches the panel.** The last answer is already on screen;
   redrawing it would cost a ~4 s flashing full refresh to produce a byte-identical
   image. `main.py` initialises the panel lazily, on the first press.
2. **Nothing is ever written to flash.** There is no state worth saving, and a
   flash write interrupted by a power cut is the one thing that could corrupt the
   filesystem. The "don't repeat the last answer" state lives in RAM and resetting
   it on power loss is harmless.
3. **The panel is always put back to sleep** after a refresh, so a power cut never
   leaves it holding a bias voltage.

If power drops *during* a refresh the panel may be left mid-transition. The next
press fixes it, and no data is at risk.

## Randomness

`os.urandom` is backed by the RP2350's hardware RNG. `magic8._rand_below` uses
rejection sampling rather than `% 20`, which would bias the first 16 answers
upward. Verified uniform over 20 000 draws (min 944, max 1042, expected 1000).

Consecutive repeats are suppressed — drawing the same answer twice in a row reads
as a broken toy even though it is perfectly random.

## Input

The board has no brightness button (e-paper has no backlight), so *any* of three
inputs asks the ball:

- **POWER key** on GP24, active-low. Verified wired — an internal pull-down loses
  to an external pull-up on that net.
- **Screen tap**, by polling the FT6336U's `TD_STATUS` register over I2C.
- **BOOTSEL**, checked last — `rp2.bootsel_button()` momentarily disables
  interrupts and takes over the QSPI CS line, so it is far more expensive than a
  GPIO read.

### Why touch is polled over I2C rather than watching the INT pin

The first version watched GP8 (touch INT) as a level. It worked exactly once and
then appeared dead. The board ships with FT6336U register `0xA4` = `0x01`,
FocalTech's *trigger* mode, where INT emits a brief pulse per touch frame instead
of sitting low while held — so a 50 ms poll catches a pulse by luck and misses
the rest.

Reading `TD_STATUS` (`0x02`) instead is a level rather than an edge, and the read
clears the controller's pending interrupt as a side effect. The I2C read is
wrapped in `try/except OSError`, because a bus glitch must not kill an
unattended loop.

The controller's reset line (GP16) is now driven high explicitly — nothing on the
board holds it there.

### Stuck-input safety net

If any input reads down continuously for 30 s, the loop logs it and re-arms.
A human cannot meaningfully hold a button that long, so re-arming is safe, and it
beats an unattended device that has silently bricked itself on a stuck line.

## Audio

A shake sound plays **while** the panel refreshes, not before it. It is
synthesised on the device: three decaying bursts of low-passed noise,
integer-only, which reads as a die tumbling in liquid. ~50 KB as packed stereo
frames, versus shipping a WAV on a 3 MB filesystem.

The vendor's `dma_play_words` ends with `while self.dma_tx.active(): pass`, which
forced sound and screen to happen strictly one after the other — 0.54 s of noise,
then a silent refresh. `dma_play_words_async` triggers the transfer and returns:
the DMA engine feeds the codec while the CPU drives SPI, so `shaker.start()`
before the refresh and `shaker.finish()` after it overlaps the two. Press time
dropped from ~3.1 s to ~1.6 s as a side effect.

### Alternate sounds

Most presses shake. Occasionally the clip is swapped for a fart or a laugh.
The policy lives in `shake.py`:

| Constant | Value | Effect |
| --- | --- | --- |
| `ALTERNATE_MIN_GAP` | 5 | at least five ordinary shakes between alternates |
| `ALTERNATE_ONE_IN` | 3 | then a 1-in-3 roll per press |

Simulated over 200 000 presses: minimum gap 6, mean 8, ~12.5% of presses, split
evenly between the two. For "guaranteed at least once every N" instead of "never
more often than every N", set `ALTERNATE_ONE_IN = 1`.

The fart is synthesised (a descending buzz with an irregular sputter). The laugh
is a **sampled clip**, converted on the host by `tools/make_clip.py` and read off
the filesystem — MicroPython has no MP3 decoder, and decoding one on a 150 MHz
M33 would be far slower than real time.

`tools/build_clips.sh` holds the recipe, so the conversion is reproducible rather
than a one-off blob. The output is exactly the packed format the PIO consumes, so
the device only reads bytes: no decoding, no parsing, and `readinto` fills a
pre-reserved buffer with no extra allocation.

### Clip level: do not normalise to full scale

The laughter source peaks at 43.6% of full scale. Normalising it to the usual
30000 applied +6.4 dB and **overdrove the amp and speaker** — audibly distorted,
even though nothing clipped digitally (peak 29916 of 32767). It is now built with
`--peak 15000`, roughly its original recorded level.

Digital headroom is not the same as analogue headroom. If a clip sounds
overdriven, lower `--peak` and rebuild; there is no need to touch the codec
volume or the runtime.

Length is bounded by RAM, not flash: 24 kHz packed stereo costs 96 KB/s against a
~490 KB heap that also holds the other clips, so the 5.04 s source is trimmed to
the highest-energy 1.5 s window (140 KB) with 8 ms fades to hide the splice.

### Why the clips are built while idle

Synthesis is slow enough to feel — ~1.0 s for the fart — so
`Shaker.prepare_next()` builds it *after* the answer is on screen, and `main`
calls it from the idle path. Sampled clips load in `_setup` instead: reading
140 KB off the filesystem takes tens of milliseconds, not seconds.
`ALTERNATE_MIN_GAP` guarantees everything is ready long before it can fire.

### Heap fragmentation: reserve big buffers early, largest first

The three clips are ~213 KB of the ~490 KB heap. Two things bite:

1. MicroPython's heap never compacts, so the free *total* can be comfortable
   while no single block that large remains.
2. `array("I", bytearray(n))` holds the bytearray **and** the array at once, so
   the transient peak is *twice* the final size.

Measured: after generating the shake clip, 419 KB was free and the largest block
was 174 KB — ample for a 140 KB clip, and still not enough to allocate one,
because allocating it briefly needs 280 KB. `Shaker._setup` therefore reserves the
alternates' buffers largest-first and *before* the shake clip, then fills them
later. Reordering those three lines is the whole fix.

**Audio is strictly optional and must stay that way.** Every entry point in
`shake.py` swallows its own exceptions and sets `available = False` on failure.
A silent Magic 8-Ball still works; one that crashes instead of answering does not.

The codec and its ~50 KB clip are brought up lazily on first press, not at boot,
so a power blip costs nothing. That does mean the *first* press after power-on
waits ~2 s before any sound.

**Never combine this loop with `_thread`.** Polling BOOTSEL while another core
executes from flash produces spurious presses — during development that caused
the loop to fire three answers from a single press. Single-threaded it is clean
(200 consecutive samples across a refresh, zero false positives).

## Edge handling

Presses are edge-triggered, so holding a button gives one answer, not a stream.
After drawing, `wait_for_release` re-samples the inputs instead of clearing the
"was down" flag outright — clearing it unconditionally meant a button held past
the 5 s timeout looked like a fresh press on the very next poll. The timeout
itself exists so a touch controller that latches INT low cannot wedge the loop.

## Layout and text size

200 × 200, 1-bit. `framebuf`'s built-in font is fixed at 8 × 8 and there is no
scaled blit, so `magic8._text_scaled` renders glyphs into a scratch mono buffer
and paints each set pixel as a `scale × scale` block. Costs ~150 ms for a screen
of text, which is nothing beside the panel refresh.

`magic8.fit()` picks the **largest size each answer actually fits at**, rather
than one size for everything:

| Scale | Glyph | Columns | Max lines | Answers |
| --- | --- | --- | --- | --- |
| 3× | 24 px | 8 | 2 | 8 |
| 2× | 16 px | 12 | 4 | 12 |
| 1× | 8 px | 23 | 6 | none currently |

A size is rejected if any single word would have to be hard-split — a word
broken mid-way reads far worse than one size smaller. That is what keeps
"Concentrate" (11 characters) from being chopped at 3×.

The ball is deliberately smaller than it could be (r = 28, not 34): legible text
matters more than a big graphic. The "8" is two stacked white rings inside a
filled black disc, because a `text()` "8" would be an 8 px speck inside it.
