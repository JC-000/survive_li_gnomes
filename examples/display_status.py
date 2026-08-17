"""Draw temperature, humidity, battery and clock to the e-paper.

Needs board.py and epaper.py on the device:

    uvx mpremote connect /dev/cu.usbmodem101 cp src/board.py src/epaper.py :
    uvx mpremote connect /dev/cu.usbmodem101 run examples/display_status.py

A full refresh takes a few seconds and flashes the panel black/white. The image
persists after the board is unplugged -- that is e-paper working, not a bug.
"""

import board
import epaper

epd = epaper.EPD_1in54()
epd.Clear(0xFF)
epd.fill(0xFF)

i2c = board.bus()
temp_c, humidity = board.SHTC3(i2c).read()
rtc = board.PCF85063A(i2c)
year, month, day, _weekday, hour, minute, _second = rtc.datetime()

epd.text("survive_li_gnomes", 4, 10, 0x00)
epd.text("RP2350-ePaper-1.54", 4, 26, 0x00)
epd.hline(4, 42, 192, 0x00)

epd.text("%.1f C  %.0f %%RH" % (temp_c, humidity), 4, 60, 0x00)
epd.text("bat %.2f V" % board.Battery().volts(), 4, 80, 0x00)
epd.text("%04d-%02d-%02d %02d:%02d" % (year, month, day, hour, minute), 4, 100, 0x00)

if rtc.oscillator_stopped():
    epd.text("(clock unset)", 4, 116, 0x00)

epd.display(epd.buffer)

# Always sleep the panel. Leaving it powered wastes battery and, on e-paper,
# holding a bias voltage indefinitely is bad for the display.
epd.sleep()
