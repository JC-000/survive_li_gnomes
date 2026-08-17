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

POLL_MS = 50


class Inputs:
    """Any button or a screen tap, edge-triggered and debounced."""

    def __init__(self, i2c):
        self.power = board.Pin(board.POWER_KEY_PIN, board.Pin.IN, board.Pin.PULL_UP)
        self.touch_int = board.Pin(board.TOUCH_INT_PIN, board.Pin.IN, board.Pin.PULL_UP)
        try:
            import rp2

            self._bootsel = rp2.bootsel_button
        except (ImportError, AttributeError):
            self._bootsel = None
        self._was_down = self.down()

    def down(self):
        if not self.power.value():  # active low
            return True
        if not self.touch_int.value():  # FT6336U pulls INT low while touched
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

    epd = None
    last_answer = None

    print("magic 8-ball ready -- press POWER, BOOTSEL, or tap the screen")

    while True:
        if inputs.pressed():
            answer = magic8.pick(exclude=last_answer)
            last_answer = answer
            print("->", answer)

            if epd is None:
                epd = epaper.EPD_1in54()  # constructor also inits
            else:
                epd.init(epd.full_update)

            magic8.render(epd, answer, footer="%.2fV" % battery.volts())
            epd.display(epd.buffer)
            epd.sleep()  # never leave the panel biased

            inputs.wait_for_release()

        time.sleep_ms(POLL_MS)


if __name__ == "__main__":
    main()
