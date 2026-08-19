# CMake file for the Waveshare RP2350-Touch-ePaper-1.54.
#
# Waveshare's boards have no pico-sdk support of their own, so the SDK is
# pointed at the header next to this file. That header is where the flash size
# lives -- see waveshare_rp2350_touch_epaper_154.h for why it has to be there
# and not in mpconfigboard.h.
#
# The same arrangement is used in-tree by boards/WAVESHARE_RP2350B_CORE and
# boards/NULLBITS_BIT_C_PRO, so it is a supported path rather than a trick.

set(PICO_PLATFORM "rp2350")

list(APPEND PICO_BOARD_HEADER_DIRS ${MICROPY_BOARD_DIR})
set(PICO_BOARD "waveshare_rp2350_touch_epaper_154")
