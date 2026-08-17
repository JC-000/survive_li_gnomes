"""Log every input transition. Run this, then press things.

    uvx mpremote connect /dev/cu.usbmodem101 run tools/input_monitor.py

Prints a line whenever an input changes state, so you can see which button
actually registers. Ctrl-C to stop.

Samples the touch controller two ways on purpose:

  TD_STATUS  - read over I2C, a level. This is what main.py uses.
  INT pin    - GP8. This board sets reg 0xA4 = 0x01 ("trigger" mode), so INT
               emits a brief pulse per touch frame rather than sitting low.
               Expect it to look almost always idle even while you press --
               that is the bug this tool exists to demonstrate.
"""

from machine import I2C, Pin
import time

import rp2

i2c = I2C(1, sda=Pin(6), scl=Pin(7), freq=100_000)
Pin(16, Pin.OUT, value=1)  # touch reset: nothing on the board holds this high
power = Pin(24, Pin.IN, Pin.PULL_UP)
touch_int = Pin(8, Pin.IN, Pin.PULL_UP)

print("reg 0xA4 (INT mode) = 0x%02x  (0x01 = trigger/pulse)" % i2c.readfrom_mem(0x38, 0xA4, 1)[0])
print("watching. press POWER / BOOTSEL / tap the screen. Ctrl-C to stop.\n")
print("%-10s %-8s %-10s %-9s %s" % ("time", "POWER", "TD_STATUS", "INT pin", "BOOTSEL"))


def sample():
    try:
        td = i2c.readfrom_mem(0x38, 0x02, 1)[0] & 0x0F
    except OSError:
        td = -1
    return (not power.value(), td, not touch_int.value(), bool(rp2.bootsel_button()))


last = None
int_pulses = 0
start = time.ticks_ms()

try:
    while True:
        now = sample()
        # Count INT pulses separately -- they are too short to show up as a
        # state change at this sample rate, which is the whole point.
        if now[2]:
            int_pulses += 1
        if now != last:
            print(
                "%-10s %-8s %-10s %-9s %s"
                % (
                    "%.2fs" % (time.ticks_diff(time.ticks_ms(), start) / 1000),
                    "DOWN" if now[0] else "-",
                    now[1] if now[1] >= 0 else "I2C ERR",
                    "LOW" if now[2] else "-",
                    "DOWN" if now[3] else "-",
                )
            )
            last = now
        time.sleep_ms(20)
except KeyboardInterrupt:
    print("\nINT pin seen low %d times" % int_pulses)
