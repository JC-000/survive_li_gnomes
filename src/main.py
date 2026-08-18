"""Magic 8-Ball. Runs automatically at power-on.

Designed around intermittent power:

- The panel is never touched at boot. E-paper is bistable, so the last answer is
  still on screen after a power cut and redrawing it would only cost a flashing
  full refresh to produce the identical image.
- Nothing is ever written to flash. There is no state worth persisting -- the
  display *is* the storage -- and a write interrupted by a power cut is the one
  thing that could corrupt the filesystem.
- The panel is initialised lazily, on the first press, so a power blip costs
  nothing and disturbs nothing.
- After the first press, refreshes are partial: quick and non-flashing. See the
  Panel class for when that has to fall back to a full refresh.

Any of three inputs asks the ball: the POWER key, BOOTSEL, or a screen tap.
"""

import time

import board
import epaper
import magic8
import shake

POLL_MS = 50

# Partial refreshes are fast and don't flash, but e-paper ghosting accumulates,
# so a full refresh is forced periodically to scrub it.
PARTIALS_BEFORE_FULL = 8

# Partial refresh needs the panel's RAM and registers intact, so the panel has to
# stay powered between presses -- deep sleep can only be left via a hardware
# reset, which drops the base image. Idle current is negligible next to the
# refresh itself, but there is no reason to hold it forever, so it sleeps after
# this long and the next press pays for a full refresh (which also clears any
# accumulated ghosting).
PANEL_IDLE_SLEEP_MS = 60_000


# A stuck input would otherwise disable the device forever. A human cannot
# meaningfully hold a button this long, so re-arming after it is safe and beats
# bricking an unattended device.
STUCK_MS = 30_000


class Inputs:
    """Any button or a screen tap, edge-triggered and debounced."""

    def __init__(self, i2c):
        self.i2c = i2c
        self.power = board.Pin(board.POWER_KEY_PIN, board.Pin.IN, board.Pin.PULL_UP)

        # The touch controller's reset line is not held high by anything on the
        # board -- an internal pull-down wins on GP16 -- so drive it explicitly
        # or the controller's state is whatever leakage decides.
        self.touch_rst = board.Pin(board.TOUCH_RST_PIN, board.Pin.OUT, value=1)

        try:
            import rp2

            self._bootsel = rp2.bootsel_button
        except (ImportError, AttributeError):
            self._bootsel = None

        self._down_since = None
        self._was_down = self.down()

    def _touched(self):
        """Read the FT6336U's touch-count register.

        Deliberately *not* the INT pin. This board ships with reg 0xA4 = 0x01,
        FocalTech's "trigger" mode, where INT emits a brief pulse per touch frame
        rather than sitting low while held -- polling that pin every 50 ms catches
        a pulse occasionally and misses it the rest of the time. TD_STATUS is a
        level, and reading it also clears the controller's pending interrupt.

        Swallows I2C errors: a bus glitch must not kill an unattended loop.
        """
        try:
            return bool(self.i2c.readfrom_mem(board.ADDR_TOUCH, 0x02, 1)[0] & 0x0F)
        except OSError:
            return False

    def down(self):
        if not self.power.value():  # active low
            return True
        if self._touched():
            return True
        # Checked last: rp2.bootsel_button() momentarily disables interrupts and
        # takes over the QSPI CS line, so it is far more expensive than a GPIO
        # read. It is also unsafe to poll while another core is executing from
        # flash -- never combine this loop with _thread.
        if self._bootsel and self._bootsel():
            return True
        return False

    def pressed(self):
        """True once per press, on the leading edge."""
        now_down = self.down()

        # Safety net: never let a stuck input disable the device permanently.
        if now_down:
            if self._down_since is None:
                self._down_since = time.ticks_ms()
            elif time.ticks_diff(time.ticks_ms(), self._down_since) > STUCK_MS:
                print("warning: input stuck down for %d s, re-arming" % (STUCK_MS // 1000))
                self._was_down = False
                self._down_since = time.ticks_ms()
        else:
            self._down_since = None

        edge = now_down and not self._was_down
        self._was_down = now_down
        return edge

    def wait_for_release(self, timeout_ms=5000):
        """Block until released, but give up eventually.

        The timeout exists so a touch controller that latches INT low can't wedge
        the loop forever. Re-sample rather than assuming released: clearing the
        flag unconditionally would make a button held past the timeout look like
        a fresh press on the very next poll, spraying answers while held.
        """
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        while self.down() and time.ticks_diff(deadline, time.ticks_ms()) > 0:
            time.sleep_ms(POLL_MS)
        self._was_down = self.down()


class Panel:
    """The e-paper, plus the policy for when a refresh can be a partial one.

    A full refresh takes ~1.3 s and flashes the whole panel black/white. A
    partial is far quicker and doesn't flash, but it only works while the
    controller still holds the base image written by displayPartBaseImage --
    so this tracks whether that base is still valid.

    Nothing here touches the panel until the first show(), which is what keeps a
    power blip free.
    """

    def __init__(self):
        self.epd = None
        self._base_valid = False
        self._partials = 0
        self._last_use = 0

    def show(self, answer, footer=None):
        """Draw an answer. Returns "full" or "partial" for logging."""
        first = self.epd is None
        if first:
            self.epd = epaper.EPD_1in54()  # constructor full-inits and powers on

        magic8.render(self.epd, answer, footer=footer)

        if self._base_valid and self._partials < PARTIALS_BEFORE_FULL:
            self.epd.displayPartial(self.epd.buffer)
            self._partials += 1
            mode = "partial"
        else:
            # Reload the full waveform before writing a base image: after a
            # partial run the partial LUT is loaded, and it will not scrub
            # ghosting. init() also restores power if we were asleep.
            if not first:
                self.epd.init(self.epd.full_update)
            self.epd.displayPartBaseImage(self.epd.buffer)
            self.epd.init(self.epd.part_update)
            self._base_valid = True
            self._partials = 0
            mode = "full"

        self._last_use = time.ticks_ms()
        return mode

    def maybe_sleep(self):
        """Deep-sleep the panel once it has been idle a while.

        Costs the next press a full refresh, since sleeping drops the base image.
        """
        if not self._base_valid:
            return False
        if time.ticks_diff(time.ticks_ms(), self._last_use) < PANEL_IDLE_SLEEP_MS:
            return False
        self.epd.sleep()
        self._base_valid = False
        return True


def main():
    i2c = board.bus()
    inputs = Inputs(i2c)
    battery = board.Battery()
    shaker = shake.Shaker()
    panel = Panel()

    last_answer = None

    print("magic 8-ball ready -- press POWER, BOOTSEL, or tap the screen")

    while True:
        if inputs.pressed():
            answer = magic8.pick(exclude=last_answer)
            last_answer = answer

            # Start the sound and let it run *through* the panel refresh rather
            # than before it -- the DMA feeds the codec while the CPU drives SPI,
            # so the sound and the screen changing happen together.
            # Never allowed to raise; a silent ball still works.
            sound = shaker.start(i2c)
            mode = panel.show(answer, footer="%.2fV" % battery.volts())
            shaker.finish()  # reap the clip if still playing; drops the power amp
            print("->", answer, "[%s, %s]" % (sound or "silent", mode))

            inputs.wait_for_release()

            # Build the alternate clips while idle. Synthesis is slow enough to
            # be felt, and ALTERNATE_MIN_GAP means they are not needed yet.
            shaker.prepare_next()
        else:
            panel.maybe_sleep()

        time.sleep_ms(POLL_MS)


if __name__ == "__main__":
    main()
