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

A shake sound plays before the answer is drawn — the order a real 8-ball works
in. It is synthesised on the device: three decaying bursts of low-passed noise,
integer-only, which reads as a die tumbling in liquid. ~50 KB as packed stereo
frames, versus shipping a WAV on a 3 MB filesystem.

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

## Layout

200 × 200, 1-bit. `framebuf`'s built-in font is 8 × 8 and cannot be scaled, so:

- The "8" is drawn as two stacked white rings inside a filled black disc. An `8`
  drawn with `text()` would be an 8 px speck inside a 68 px ball.
- Text wraps at 23 characters, not the 25 that technically fit — 25 runs edge to
  edge and collides with the border. Only two answers need a second line.
