"""Board support for the RP2350A + SHTC3 + PCF85063A board.

Every constant here was verified against the physical board, not a datasheet.
See docs/hardware.md. Runs on the device, not the host.
"""

from machine import I2C, Pin
import time

# The only real I2C bus on this board. Scans of other pin pairs return every
# address from 0x08-0x77, which means a floating SDA, not devices.
I2C_ID = 1
SDA_PIN = 6
SCL_PIN = 7

ADDR_SHTC3 = 0x70
ADDR_RTC = 0x51  # PCF85063A
ADDR_TOUCH = 0x38  # FocalTech FT6236/FT6336
ADDR_UNKNOWN = 0x18  # see docs/hardware.md


def bus(freq=100_000):
    """The board's I2C bus.

    Only ever build I2C(1) on these pins. Creating a second I2C(1) on a
    different pin pair wedges the block until a hard reset.
    """
    return I2C(I2C_ID, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=freq)


class SHTC3:
    """Sensirion SHTC3 temperature / humidity sensor."""

    _WAKEUP = b"\x35\x17"
    _SLEEP = b"\xb0\x98"
    _READ_ID = b"\xef\xc8"
    # Normal-power measurement, temperature first, clock stretching disabled.
    _MEASURE = b"\x78\x66"

    def __init__(self, i2c, addr=ADDR_SHTC3):
        self.i2c = i2c
        self.addr = addr

    def _wake(self):
        self.i2c.writeto(self.addr, self._WAKEUP)
        time.sleep_ms(5)

    def id(self):
        self._wake()
        self.i2c.writeto(self.addr, self._READ_ID)
        time.sleep_ms(5)
        raw = self.i2c.readfrom(self.addr, 3)
        ident = (raw[0] << 8) | raw[1]
        self.i2c.writeto(self.addr, self._SLEEP)
        return ident

    def present(self):
        # Only bits 5:0 and 11 are fixed for the SHTC3; the rest vary per part.
        return (self.id() & 0x083F) == 0x0807

    def read(self):
        """Return (temperature_c, relative_humidity_pct)."""
        self._wake()
        self.i2c.writeto(self.addr, self._MEASURE)
        time.sleep_ms(20)  # max conversion time is ~12.1 ms
        d = self.i2c.readfrom(self.addr, 6)
        self.i2c.writeto(self.addr, self._SLEEP)
        # Bytes 2 and 5 are CRC-8 checksums, currently unverified.
        temp_raw = (d[0] << 8) | d[1]
        hum_raw = (d[3] << 8) | d[4]
        return (-45 + 175 * temp_raw / 65535.0, 100 * hum_raw / 65535.0)


def _bcd_to_int(value):
    return (value >> 4) * 10 + (value & 0x0F)


def _int_to_bcd(value):
    return ((value // 10) << 4) | (value % 10)


class PCF85063A:
    """NXP PCF85063A real-time clock.

    Note the register layout: seconds live at 0x04. The PCF8563 shares this I2C
    address but puts seconds at 0x02, and decoding one as the other yields
    plausible-looking garbage.
    """

    _REG_TIME = 0x04

    def __init__(self, i2c, addr=ADDR_RTC):
        self.i2c = i2c
        self.addr = addr

    def datetime(self):
        """Return (year, month, day, weekday, hour, minute, second).

        Tuple order matches machine.RTC.datetime() minus the subsecond field.
        """
        r = self.i2c.readfrom_mem(self.addr, self._REG_TIME, 7)
        return (
            2000 + _bcd_to_int(r[6]),
            _bcd_to_int(r[5] & 0x1F),
            _bcd_to_int(r[3] & 0x3F),
            r[4] & 0x07,
            _bcd_to_int(r[2] & 0x3F),
            _bcd_to_int(r[1] & 0x7F),
            _bcd_to_int(r[0] & 0x7F),
        )

    def oscillator_stopped(self):
        """True if the clock lost time — its value is not trustworthy."""
        return bool(self.i2c.readfrom_mem(self.addr, self._REG_TIME, 1)[0] & 0x80)

    def set_datetime(self, year, month, day, weekday, hour, minute, second):
        self.i2c.writeto_mem(
            self.addr,
            self._REG_TIME,
            bytes(
                (
                    _int_to_bcd(second),  # clears the oscillator-stopped flag
                    _int_to_bcd(minute),
                    _int_to_bcd(hour),
                    _int_to_bcd(day),
                    weekday & 0x07,
                    _int_to_bcd(month),
                    _int_to_bcd(year % 100),
                )
            ),
        )
