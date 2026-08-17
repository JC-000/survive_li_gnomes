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
