"""Board support for the Waveshare RP2350-Touch-ePaper-1.54.

RP2350A, 16 MB flash, 200x200 SSD1681 e-paper, FT6336U touch, SHTC3,
PCF85063A RTC, ES8311 audio codec, microSD, Li-ion battery sensing.

Pin numbers come from Waveshare's demo code and were then checked against the
physical board where observable. See docs/hardware.md. Runs on the device.
"""

from machine import ADC, I2C, Pin
import time

# The only real I2C bus on this board. Scans of other pin pairs return every
# address from 0x08-0x77, which means a floating SDA, not devices.
I2C_ID = 1
SDA_PIN = 6
SCL_PIN = 7

ADDR_SHTC3 = 0x70
ADDR_RTC = 0x51  # PCF85063A
ADDR_TOUCH = 0x38  # FocalTech FT6336U
ADDR_CODEC = 0x18  # ES8311 audio codec

# E-paper (SSD1681, 200x200). Driver lives in epaper.py.
EPD_SPI_ID = 1
EPD_SCK_PIN = 10
EPD_MOSI_PIN = 11
EPD_CS_PIN = 9
EPD_DC_PIN = 12
EPD_PWR_PIN = 13
EPD_RST_PIN = 14
EPD_BUSY_PIN = 15

TOUCH_INT_PIN = 8
TOUCH_RST_PIN = 16
RTC_INT_PIN = 17

# microSD, on the *other* SPI block.
SD_SPI_ID = 0
SD_SCK_PIN = 18
SD_MOSI_PIN = 19
SD_MISO_PIN = 20
SD_CS_PIN = 23

CODEC_PA_CTRL_PIN = 0  # ES8311 power-amp enable
POWER_KEY_PIN = 24  # active low, needs a pull-up. BOOTSEL is rp2.bootsel_button()

# Battery sense: 200k/200k divider on GP29 (ADC3), gated by GP28.
BAT_ADC_PIN = 29
BAT_EN_PIN = 28
BAT_DIVIDER = 2.0  # (200k + 200k) / 200k
BAT_VREF = 3.390  # Waveshare's calibrated reference, not the nominal 3.3


# The divider node charges through ~100k, so it ramps rather than steps. Measured
# from a cold reset: 0.79 V immediately, 3.56 V at 10 ms, 4.13 V at ~36 ms,
# settled 4.21 V by ~86 ms. Reading too early silently under-reports the battery.
BAT_SETTLE_MS = 150


class Battery:
    """Li-ion pack voltage.

    GP29 only reads sensibly once GP28 is driven as an output -- left as a
    floating input the divider has no return path and GP29 reads noise. Waveshare
    drives GP28 high; both levels gave identical readings when measured here, but
    high is what the vendor firmware does, so that is what this does.
    """

    def __init__(self):
        self.enable = Pin(BAT_EN_PIN, Pin.OUT, value=1)
        self.adc = ADC(Pin(BAT_ADC_PIN))
        self._settled = False

    def volts(self, samples=5):
        if not self._settled:
            time.sleep_ms(BAT_SETTLE_MS)
            self._settled = True
        raw = sum(self.adc.read_u16() for _ in range(samples)) / samples
        return raw / 65535 * BAT_VREF * BAT_DIVIDER


class FT6336U:
    """Capacitive touch controller."""

    _REG_TD_STATUS = 0x02
    _REG_TOUCH1 = 0x03

    def __init__(self, i2c, addr=ADDR_TOUCH):
        self.i2c = i2c
        self.addr = addr
        self.rst = Pin(TOUCH_RST_PIN, Pin.OUT, value=1)
        self.int = Pin(TOUCH_INT_PIN, Pin.IN, Pin.PULL_UP)

    def reset(self):
        self.rst.value(0)
        time.sleep_ms(10)
        self.rst.value(1)
        time.sleep_ms(100)

    def chip_id(self):
        return self.i2c.readfrom_mem(self.addr, 0xA3, 1)[0]  # 0x64 on this board

    def touch(self):
        """Return (x, y) of the first touch point, or None."""
        if not self.i2c.readfrom_mem(self.addr, self._REG_TD_STATUS, 1)[0] & 0x0F:
            return None
        d = self.i2c.readfrom_mem(self.addr, self._REG_TOUCH1, 4)
        return (((d[0] & 0x0F) << 8) | d[1], ((d[2] & 0x0F) << 8) | d[3])


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
