"""Identify what is on the board's I2C bus and read the sensors once.

Self-contained on purpose so it works without copying anything to the device:

    uvx mpremote connect /dev/cu.usbmodem101 run tools/probe.py

If devices ACK a scan but every read returns EIO, the I2C block is wedged from a
previous session. Run `uvx mpremote connect /dev/cu.usbmodem101 reset` first.
"""

from machine import ADC, I2C, Pin
import time

KNOWN = {
    0x70: "SHTC3 temp/humidity",
    0x51: "PCF85063A RTC",
    0x38: "FT6236/FT6336 touch",
    0x18: "unidentified",
}

i2c = I2C(1, sda=Pin(6), scl=Pin(7), freq=100_000)

print("I2C1 gp6/gp7:")
for addr in i2c.scan():
    print("  0x%02x  %s" % (addr, KNOWN.get(addr, "NEW - not seen during onboarding")))

# --- SHTC3 -----------------------------------------------------------------
try:
    i2c.writeto(0x70, b"\x35\x17")
    time.sleep_ms(5)
    i2c.writeto(0x70, b"\x78\x66")
    time.sleep_ms(20)
    d = i2c.readfrom(0x70, 6)
    i2c.writeto(0x70, b"\xb0\x98")
    print(
        "SHTC3: %.2f C  %.2f %%RH"
        % (-45 + 175 * ((d[0] << 8 | d[1]) / 65535.0), 100 * ((d[3] << 8 | d[4]) / 65535.0))
    )
except Exception as exc:
    print("SHTC3: FAILED", exc)

# --- RTC -------------------------------------------------------------------
try:
    r = i2c.readfrom_mem(0x51, 0x04, 7)  # PCF85063A: seconds at 0x04
    bcd = lambda v: (v >> 4) * 10 + (v & 0x0F)
    print(
        "RTC:   20%02d-%02d-%02d %02d:%02d:%02d%s"
        % (
            bcd(r[6]),
            bcd(r[5] & 0x1F),
            bcd(r[3] & 0x3F),
            bcd(r[2] & 0x3F),
            bcd(r[1] & 0x7F),
            bcd(r[0] & 0x7F),
            "  (OSCILLATOR STOPPED - time is not trustworthy)" if r[0] & 0x80 else "",
        )
    )
except Exception as exc:
    print("RTC:   FAILED", exc)

# --- ADC -------------------------------------------------------------------
# None of these currently corresponds to the battery; see docs/hardware.md.
print("ADC:")
for gp in (26, 27, 28, 29):
    raw = ADC(Pin(gp)).read_u16()
    print("  gp%-2d  %-6d  %.3f V" % (gp, raw, raw / 65535 * 3.3))
print("  core  %.1f C" % (27 - ((ADC(4).read_u16() / 65535 * 3.3) - 0.706) / 0.001721))
