# Vendored binaries

## `emlearn_cnn_int8.mpy`

The int8 CNN runtime the speaker-independent spotter needs: a MicroPython
**native** module wrapping Sipeed's TinyMaix, from
[emlearn-micropython](https://github.com/emlearn/emlearn-micropython) (MIT).

    https://emlearn.github.io/emlearn-micropython/builds/latest/armv7emsp_6.3/emlearn_cnn_int8.mpy

    sha256  60492a0bf6fb618edb2fb7ea5a2d55e49b5ad3a0ed67682cd07e96914a002012
    5470 bytes, byte-identical to the 0.11.1 pin
    header: mpy 6.3, arch armv7emsp, 31-bit small ints

**It is committed rather than fetched because the board cannot fetch it.** This
is an RP2350A, not a -W: no networking, so no `mip install` on the device. It
has to be downloaded on the Mac and copied over, and a build that depends on a
URL being up is a build that breaks when it is not.

### Do not rename it to `tinymaix_cnn`

That was its name in 0.4.0 and 0.5.0, and it is the name most of the writing
about it still uses. It became `emlearn_cnn` at 0.6.0 and split into
`emlearn_cnn_int8` / `emlearn_cnn_fp32` at 0.8.0. The old path 404s under
`builds/latest/`; it survives only under `builds/master/`, which is a directory
that accumulates and is never cleaned, so anything found there is of unknown
vintage. Pin `latest` or a version.

### Verified on this board

Imports, loads and classifies correctly on `armv7emsp` at MicroPython 1.28.0 --
10/10 on emlearn's own MNIST digits. Upstream claims testing on x64 and
xtensawin only, so that was genuinely unknown until measured. 12096 bytes of
heap at import. See `docs/cnn-on-device.md` for the rest, including four ways
it fails silently that `tools/tmdl_info.py` exists to refuse.
