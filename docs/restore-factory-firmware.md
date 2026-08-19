# Restoring the factory firmware

The original vendor firmware was dumped before MicroPython was flashed. Two
artifacts exist in `firmware/backup/`:

| File | Size | In git? | What it is |
| --- | --- | --- | --- |
| `factory-main.uf2` | 827 KB | yes | Just the application program (`main`, `0x10000000`–`0x10067440`) |
| `factory-flash-16MB.bin` | 16 MB | **no** — gitignored | Byte-for-byte dump of the entire flash, including whatever the vendor stored outside the program |
| `SHA256SUMS` | — | yes | Hashes of both |

`factory-flash-16MB.bin` is deliberately kept out of git because of its size. It
is only on this machine. **If you wipe the working tree, it is gone.** Copy it
somewhere else if the factory image matters to you.

## Entering BOOTSEL

`picotool` could reboot the *factory* firmware into BOOTSEL automatically (it
takes ~10 s, longer than picotool's own timeout, so it prints a failure message
and succeeds anyway — just re-run the command). Under MicroPython use:

```sh
uvx mpremote connect /dev/cu.usbmodem101 bootloader
```

Otherwise hold the BOOTSEL button while plugging the board in.

## Restore the program only

```sh
picotool load firmware/backup/factory-main.uf2 -x
```

## Restore the entire flash

```sh
picotool load firmware/backup/factory-flash-16MB.bin -o 0x10000000 -x
```

The `-o` offset is required for a raw `.bin` — without it picotool has no idea
where the image belongs. Verify first:

```sh
shasum -a 256 -c firmware/backup/SHA256SUMS
```

## Re-flashing MicroPython

The UF2 is gitignored (re-downloadable):

```sh
curl -sSLO https://micropython.org/resources/firmware/RPI_PICO2-20260406-v1.28.0.uf2
picotool load RPI_PICO2-20260406-v1.28.0.uf2 -x
```

Use the **ARM** build. The `-RISCV-` variant targets the RP2350's Hazard3 cores
and will not boot the image this board was configured for.

## The two MicroPython images

There are two, and the difference between them is how much of the flash the
filesystem gets.

| | Filesystem | Built by | Rollback target |
| --- | --- | --- | --- |
| `firmware/RPI_PICO2-20260406-v1.28.0.uf2` | 3 MB | micropython.org | yes — this is the known-good one |
| `firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2` | 15 MB | `tools/build_firmware.sh` | — this is what the board runs, since 2026-08-18 |

Same MicroPython (v1.28.0), same pico-sdk (2.2.0), same frozen modules, same
`rp2350-arm-s` family. The custom build differs from the official one in the
board name, the flash size, and its compiler being Arm's 14.2.rel1
(GCC 14.2.1, newlib 4.4.0) rather than the GCC 14.2.0 / newlib 4.5.0 that built
the official image. Both `.uf2` files are gitignored — the stock one is a
download, the custom one is a build.

Confirm which is which without flashing anything:

```sh
picotool info -a firmware/<image>.uf2 | grep 'embedded drive'
#  stock:   0x10100000-0x10400000  (3072K)
#  16 MB:   0x10100000-0x11000000 (15360K)
```

### Flashing either one wipes the filesystem

Not because the flash is erased — `picotool load` writes only the region the
UF2 covers, and both images stop well short of `0x10100000`. It is that the two
images disagree about how many blocks the filesystem has. littlefs stores the
block count in its superblock and refuses to mount when it does not match
(`lfs2.c`, the `block_count` check in `lfs2_mount`), so `_boot.py` falls into
its `except` branch and reformats:

```python
try:
    fs = vfs.VfsLfs2(bdev, progsize=256)
except:
    vfs.VfsLfs2.mkfs(bdev, progsize=256)
```

That is in **both** directions — going to 15 MB and coming back to 3 MB. There
is no in-place resize to reach for either: littlefs itself has `lfs2_fs_grow`,
but MicroPython v1.28.0 exposes no binding for it.

So before flashing, copy off anything that exists only on the board. Deployed
modules and clips all come back from `tools/deploy.sh`; **enrolled templates do
not** — `src/templates.bin` and `src/templates.py` are gitignored, and if the
host's copies are gone, the ones on the board are the only ones there are.

```sh
uvx mpremote connect $PORT ls
uvx mpremote connect $PORT cp :templates.bin src/templates.bin
```

### Flashing the 16 MB build

```sh
./tools/build_firmware.sh                         # produces the .uf2
uvx mpremote connect $PORT bootloader
picotool load firmware/WAVESHARE_RP2350_TOUCH_EPAPER_154-v1.28.0-16MB.uf2 -x
```

Then measure the result rather than assuming it — the board is the only thing
that can confirm the part answers above 4 MB:

```sh
uvx mpremote connect $PORT exec \
  "import os; s = os.statvfs('/'); print(s[0] * s[2], 'bytes')"
```

*Measured* on 2026-08-18: **15,728,640 bytes total, 15,720,448 free** — littlefs
kept 8,192 bytes for itself. 3,145,728 means the stock image is still running;
that is what the same command returned before the flash.

That figure is only what littlefs was told at compile time, though. To make the
board prove it, write a file bigger than the old filesystem and read it back:

```sh
uvx mpremote connect $PORT run tools/flash_capacity.py
```

A part smaller than the firmware believes does not report an error; it drops
the high address bits and wraps, so a write at 12 MB lands at 4 MB. That tool
stamps every chunk with its own offset precisely so a wrap shows up as a
mismatch, and prints the offset where it starts. *Measured* on 2026-08-18:
10,485,760 bytes written and read back identically, 0 of 2560 chunks bad,
210 kB/s writing and 9,320 kB/s reading.

### Rolling back

Two commands and a redeploy, and nothing about it depends on the custom build
still existing:

```sh
uvx mpremote connect $PORT bootloader
picotool load firmware/RPI_PICO2-20260406-v1.28.0.uf2 -x
PORT=$PORT ./tools/deploy.sh eliza     # the filesystem was reformatted, again
```
