// Board and hardware specific configuration for the
// Waveshare RP2350-Touch-ePaper-1.54.
//
// Deliberately minimal. Everything this board needs beyond the stock Pico 2
// configuration is flash size, and that is set in
// waveshare_rp2350_touch_epaper_154.h so that the SDK and the linker agree
// with MicroPython about it.

#define MICROPY_HW_BOARD_NAME "Waveshare RP2350-Touch-ePaper-1.54"

// 1 MB for the firmware, the rest for the filesystem -- 15 MB, against 3 MB
// under the stock RPI_PICO2 build, which reserves the same 1 MB out of a flash
// it believes is 4 MB. The reserve is unchanged rather than tightened: the
// image is ~330 KB today and the headroom is what a native module (emlearn,
// TinyMaix) or a frozen manifest would grow into.
#define MICROPY_HW_FLASH_STORAGE_BYTES (PICO_FLASH_SIZE_BYTES - 1024 * 1024)

// USB VID/PID are left at the rp2 port defaults (0x2E8A/0x0005, mpconfigport.h)
// so the board enumerates exactly as it does under the stock build and
// mpremote's device discovery is unaffected.
