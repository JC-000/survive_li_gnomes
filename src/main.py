"""Magic 8-Ball. Runs automatically at power-on.

Designed around intermittent power:

- The panel is never touched at boot. E-paper is bistable, so the last answer is
  still on screen after a power cut and redrawing it would only cost a 2 s
  flashing refresh to produce the identical image.
- Nothing is ever written to flash. There is no state worth persisting -- the
  display *is* the storage -- and a write interrupted by a power cut is the one
  thing that could corrupt the filesystem.
- The panel is initialised lazily, on the first press, so a power blip costs
  nothing and disturbs nothing.

Any of three inputs asks the ball: the POWER key, BOOTSEL, or a screen tap.
"""

import time

import board
import epaper
import magic8
import shake

POLL_MS = 50


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


def main():
    i2c = board.bus()
    inputs = Inputs(i2c)
    battery = board.Battery()
    shaker = shake.Shaker()

    epd = None
    last_answer = None

    print("magic 8-ball ready -- press POWER, BOOTSEL, or tap the screen")

    while True:
        if inputs.pressed():
            answer = magic8.pick(exclude=last_answer)
            last_answer = answer
            print("->", answer)

            # Start the shake and let it run *through* the panel refresh rather
            # than before it -- the DMA feeds the codec while the CPU drives SPI,
            # so the sound and the screen changing happen together.
            # Never allowed to raise; a silent ball still works.
            shaker.start(i2c)

            if epd is None:
                epd = epaper.EPD_1in54()  # constructor also inits
            else:
                epd.init(epd.full_update)

            magic8.render(epd, answer, footer="%.2fV" % battery.volts())
            epd.display(epd.buffer)
            epd.sleep()  # never leave the panel biased
            shaker.finish()  # audio is long done by here; drops the power amp

            inputs.wait_for_release()

        time.sleep_ms(POLL_MS)


if __name__ == "__main__":
    main()
