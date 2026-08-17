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

- **POWER key** on GP24, active-low.
- **Screen tap** via the FT6336U INT line on GP8.
- **BOOTSEL**, checked last — `rp2.bootsel_button()` momentarily disables
  interrupts and takes over the QSPI CS line, so it is far more expensive than a
  GPIO read.

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

## Layout

200 × 200, 1-bit. `framebuf`'s built-in font is 8 × 8 and cannot be scaled, so:

- The "8" is drawn as two stacked white rings inside a filled black disc. An `8`
  drawn with `text()` would be an 8 px speck inside a 68 px ball.
- Text wraps at 23 characters, not the 25 that technically fit — 25 runs edge to
  edge and collides with the border. Only two answers need a second line.
