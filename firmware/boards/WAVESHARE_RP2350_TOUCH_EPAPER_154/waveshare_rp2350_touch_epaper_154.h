/*
 * pico-sdk board header for the Waveshare RP2350-Touch-ePaper-1.54.
 *
 * The board is a plain RP2350A with a 16 MB flash part, so everything except
 * the flash size is inherited from the SDK's own pico2.h.
 *
 * THE FLASH SIZE HAS TO BE DECLARED HERE, TWICE, AND HERE IS WHY:
 *
 *   - The `pico_board_cmake_set(...)` line is not C. pico-sdk's
 *     cmake/generic_board.cmake reads this file *as text* and turns those
 *     lines into CMake variables. ports/rp2/CMakeLists.txt then passes
 *     PICO_FLASH_SIZE_BYTES to the linker as __micropy_flash_size__, which is
 *     the FLASH region length in memmap_mp_rp2350.ld. Without it the linker
 *     falls back to its 4096k default.
 *
 *   - The `#define` is the C half, which is what rp2_flash.c computes the
 *     filesystem base and extent from.
 *
 * The text scraper does not follow #includes, so inheriting pico2.h below does
 * NOT inherit its `pico_board_cmake_set_default(PICO_FLASH_SIZE_BYTES, ...)`.
 * That is the trap: define the C half only and you get a filesystem that
 * believes in 16 MB linked against a firmware image that believes in 4 MB.
 *
 * Defining PICO_FLASH_SIZE_BYTES in mpconfigboard.h instead -- as
 * boards/WAVESHARE_RP2350B_CORE does -- reaches only the translation units
 * that include MicroPython's mpconfigport.h, leaving the SDK's own sources
 * with pico2.h's 4 MB. Here it is one value for everybody.
 *
 * Flash is 16 MB *verified* on this physical board (docs/hardware.md); it is
 * not taken from a datasheet or from Waveshare's page.
 */

#ifndef _BOARDS_WAVESHARE_RP2350_TOUCH_EPAPER_154_H
#define _BOARDS_WAVESHARE_RP2350_TOUCH_EPAPER_154_H

pico_board_cmake_set(PICO_PLATFORM, rp2350)

// For board detection
#define WAVESHARE_RP2350_TOUCH_EPAPER_154

// --- FLASH ---
// Set, not set_default: the size is a property of the part fitted to the
// board, so there is nothing for a command-line -D to usefully override.
pico_board_cmake_set(PICO_FLASH_SIZE_BYTES, (16 * 1024 * 1024))
#define PICO_FLASH_SIZE_BYTES (16 * 1024 * 1024)

// Everything else -- RP2350A variant, boot stage 2, flash clock divider,
// default peripheral pins, A2 stepping support -- is the Pico 2's. Included
// last so that the #ifndef guards in it see the definition above.
//
// The default UART/I2C/SPI pins it brings are the Pico 2's and are wrong for
// this board -- machine_spi.c and machine_i2c.c read PICO_DEFAULT_SPI_* and
// PICO_DEFAULT_I2C to fill in a bus constructed without pins. They are left
// wrong on purpose: the stock RPI_PICO2 build the board runs today inherits
// exactly the same values, so nothing about pin defaults changes underneath
// the code by switching to this board. src/ names every pin explicitly
// (docs/hardware.md) and so never sees them.
#include "boards/pico2.h"

#endif
