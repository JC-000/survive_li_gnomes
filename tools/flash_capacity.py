"""Prove the filesystem is as big as it says it is. Runs on the board.

    uvx mpremote connect /dev/cu.usbmodem101 run tools/flash_capacity.py

`os.statvfs('/')` reports what littlefs was *told* the block device is, which
is what `MICROPY_HW_FLASH_STORAGE_BYTES` said at compile time. On a board whose
firmware has just been rebuilt for a larger flash, that is the number under
test, not the evidence -- exactly the shape CLAUDE.md warns about for the panel
and the codec. A part that is smaller than the firmware believes reports 15 MB
and then quietly wraps: SPI NOR flash typically ignores the high address bits
it does not have, so writing at 12 MB lands back at 4 MB and destroys what was
written there, with no error anywhere.

So this writes a real file, one that does not fit in the old 3 MB filesystem,
and reads it back. Every 4 KB chunk carries its own offset in its first and
last four bytes, which is what makes wrapping visible: after a wrap, the chunk
read at offset X carries the stamp of some other offset, and the first
mismatch says where the part really ends.

    write 10.0 MB ... 2560/2560 chunks
    verify        ... 2560/2560 chunks, 0 bad
    ok: 10485760 bytes written and read back identically

**Attended bench tool. Never deployed to the board.** Nothing the board runs
writes to flash by design (docs/design.md, and the rule in CLAUDE.md), because
a write interrupted by a power cut is the one thing that can corrupt the
filesystem. This writes 10 MB deliberately, which is exactly why it is run by
hand over the REPL, on a board somebody is watching, and never copied to the
device or imported by anything. It removes its file afterwards, including when
the verify fails -- the evidence is the offset it prints, not the file.

Expect it to be slow: littlefs on this part manages a few hundred kB/s, so
10 MB is a minute or two, and it prints as it goes. Run it on a freshly
formatted filesystem, before redeploying a program, so that a failure leaves
nothing behind worth cleaning up.
"""

import os
import time

PATH = "/_capacity_test.bin"
CHUNK = 4096
MEGABYTE = 1024 * 1024

# 10 MB: comfortably past the 3 MB the stock firmware formats, so a board that
# somehow still had the old filesystem cannot pass, and comfortably inside the
# 15 MB the new one formats, so a pass is not a fluke of running out of room.
# Overridden below if the filesystem genuinely cannot hold it.
WANT = 10 * MEGABYTE

# Keep a margin free. littlefs needs room to manoeuvre and a filesystem filled
# to the last block is a bad state to leave a board in.
MARGIN = 512 * 1024


def statvfs():
    s = os.statvfs("/")
    return s[0] * s[2], s[0] * s[3]  # total, free


def stamp(buf, offset):
    """Write `offset` into the ends of buf, leaving the middle alone.

    Byte-repeating patterns are useless here: a wrapped read of the wrong
    chunk would match them. The offset has to be *in* the bytes. It only needs
    to be in a few of them, though, and that matters -- stamping all 4096
    bytes from Python costs a thousand loop iterations per chunk and turns a
    two-minute test into a twenty-minute one. Wraps are aligned to a power of
    two no smaller than a chunk, so a stamp at the head of each chunk catches
    every one of them.
    """
    buf[0] = offset & 0xFF
    buf[1] = (offset >> 8) & 0xFF
    buf[2] = (offset >> 16) & 0xFF
    buf[3] = (offset >> 24) & 0xFF
    # Repeated at the tail as well, so a chunk that is half from one place and
    # half from another -- a partial write, rather than a wrap -- is not read
    # as intact.
    buf[-4] = buf[0]
    buf[-3] = buf[1]
    buf[-2] = buf[2]
    buf[-1] = buf[3]


def main():
    total, free = statvfs()
    print("filesystem: %d bytes total, %d free" % (total, free))
    print("            %.1f MB total, %.1f MB free" % (total / MEGABYTE, free / MEGABYTE))

    want = WANT
    if free - MARGIN < want:
        want = (free - MARGIN) // CHUNK * CHUNK
        print("            only %.1f MB free, testing that instead" % (want / MEGABYTE))
    if want < CHUNK:
        print("FAIL: no room to test with")
        return

    chunks = want // CHUNK
    # A filler that is not all one byte, so that a chunk of erased flash
    # (0xFF) or of zeros cannot pass for data. Built once.
    buf = bytearray(CHUNK)
    for i in range(CHUNK):
        buf[i] = (i * 7 + (i >> 5)) & 0xFF
    expect = bytearray(buf)
    check = bytearray(CHUNK)

    try:
        t0 = time.ticks_ms()
        with open(PATH, "wb") as f:
            for c in range(chunks):
                stamp(buf, c * CHUNK)
                f.write(buf)
                if c % 256 == 0:
                    print("write %.1f MB ... %d/%d chunks" % (want / MEGABYTE, c, chunks))
        t_write = time.ticks_diff(time.ticks_ms(), t0)

        size = os.stat(PATH)[6]
        print("wrote %d bytes in %d ms (%d kB/s), stat says %d"
              % (want, t_write, want // max(t_write, 1), size))
        if size != want:
            print("FAIL: file is %d bytes, expected %d" % (size, want))
            return

        t0 = time.ticks_ms()
        bad = 0
        first_bad = -1
        with open(PATH, "rb") as f:
            for c in range(chunks):
                n = f.readinto(check)
                stamp(expect, c * CHUNK)
                if n != CHUNK or check != expect:
                    bad += 1
                    if first_bad < 0:
                        first_bad = c * CHUNK
                if c % 256 == 0:
                    print("verify        ... %d/%d chunks, %d bad" % (c, chunks, bad))
        t_read = time.ticks_diff(time.ticks_ms(), t0)

        print("read %d bytes in %d ms (%d kB/s)"
              % (want, t_read, want // max(t_read, 1)))
        if bad:
            print("FAIL: %d of %d chunks differ, first at offset %d (%.2f MB)"
                  % (bad, chunks, first_bad, first_bad / MEGABYTE))
            print("      a first mismatch at a power-of-two boundary is the")
            print("      part wrapping -- that offset is its real size")
        else:
            print("ok: %d bytes written and read back identically" % want)
    finally:
        try:
            os.remove(PATH)
        except OSError:
            pass
        total, free = statvfs()
        print("after: %d bytes free (%.1f MB)" % (free, free / MEGABYTE))


main()
