#!/bin/sh
# Build MicroPython for this board with all 16 MB of flash addressed.
#
# The stock RPI_PICO2 image the board runs is built for the reference Pico 2's
# 4 MB part, so it formats a 3 MB filesystem and never touches the remaining
# 13 MB. Nothing is wrong with the flash -- it is simply not in the map. This
# script rebuilds MicroPython against a board definition that says 16 MB, which
# yields a ~15 MB filesystem (1 MB is reserved for the firmware, exactly as the
# stock build reserves 1 MB of its 4).
#
#   ./tools/build_firmware.sh
#   USER_C_MODULES=/path/to/micropython.cmake ./tools/build_firmware.sh
#
# Everything is pinned and everything lands under build/ (gitignored): the
# MicroPython checkout, the Arm toolchain, the CMake build tree. The only
# things this script needs from the host are cmake, make, git and curl:
#
#   brew install cmake
#
# Deliberately NOT `brew install arm-none-eabi-gcc`. Homebrew's build of that
# compiler ships without newlib -- no libc, no nosys.specs -- and pico-sdk's
# boot stage 2 fails to link against it with:
#
#   arm-none-eabi-gcc: fatal error: cannot read spec file 'nosys.specs'
#
# So the toolchain comes from Arm directly, as a tarball unpacked into build/.
# The cask `gcc-arm-embedded` installs the same toolchain from a .pkg, but that
# wants an admin password and puts it in /Applications; a tarball needs neither.
#
# MICROPYTHON VERSION. Pinned to the version already on the board. Building a
# newer MicroPython would change the language runtime underneath a program full
# of measured timings and viper functions at the same moment as it changes the
# flash map -- two variables, one flash. If MicroPython is to be upgraded, do it
# as its own step, from a board that is otherwise known good.
#
# There is a second, harder reason, for whoever is tempted to bump the version
# while they are in here anyway. The CNN spotter loads a **native** .mpy --
# `vendor/emlearn_cnn_int8.mpy`, built for `armv7emsp` at mpy 6.3 -- and a
# native module is rejected unless MPY_VERSION *and* MPY_SUB_VERSION both match
# the firmware (py/persistentcode.h). v1.28.0 is 6.3, so pinning to it keeps
# that file loadable; a version bump that moves either number silently costs
# the spotter its runtime, and the board has no networking to fetch a
# replacement with. See vendor/README.md.
set -eu

MPY_VERSION="${MPY_VERSION:-v1.28.0}"
BOARD="${BOARD:-WAVESHARE_RP2350_TOUCH_EPAPER_154}"

# Arm GNU Toolchain 14.2.rel1, darwin-arm64. Not the newest -- the newest is
# the reason this line has a comment.
#
# The version is chosen to match the compiler that built the official
# v1.28.0 image, which the image itself will tell you:
#
#   $ strings firmware/RPI_PICO2-20260406-v1.28.0.uf2 | grep MinSizeRel
#   v1.28.0 on 2026-04-06 (GNU 14.2.0 MinSizeRel)
#
# With 15.3.rel1 the build fails, and not in the port: mbedtls' constant-time
# XOR helper trips -Werror=array-bounds in aes.c, because MicroPython compiles
# everything with -Werror and GCC 15 reasons harder about that inlining than
# GCC 14 does. The fixes available -- carve -Werror out of the crypto library,
# or switch mbedtls off, which on a board with no networking is tempting -- both
# change the firmware to suit the compiler. Matching the compiler to the
# firmware changes nothing.
#
# The hash is of the tarball fetched here, recorded on 2026-08-18; Arm does not
# publish a checksum next to this artifact. It is checked because an interrupted
# 135 MB download otherwise surfaces much later as an incomprehensible
# compiler error.
TOOLCHAIN_VERSION="${TOOLCHAIN_VERSION:-14.2.rel1}"
TOOLCHAIN_HOST="${TOOLCHAIN_HOST:-darwin-arm64}"
TOOLCHAIN_SHA256="${TOOLCHAIN_SHA256:-c7c78ffab9bebfce91d99d3c24da6bf4b81c01e16cf551eb2ff9f25b9e0a3818}"

cd "$(dirname "$0")/.."
REPO="$(pwd)"

# Every path below is made absolute against the repo root, because a relative
# one does not survive the trip. Both ways it fails were found by someone
# using this script, and neither looks like a path problem:
#
#   USER_C_MODULES=firmware/usermod/tflm/micropython.cmake
#     -> the rp2 Makefile hands it to cmake unchanged, cmake resolves it
#        against ports/rp2, and the error names a path nobody wrote:
#        "USER_C_MODULES doesn't exist: .../ports/rp2/firmware/usermod/..."
#
#   BUILD_DIR=build/mpy-16mb-tflm
#     -> worse, because it fails *late*: `make -C ports/rp2` resolves it
#        against ports/rp2, the whole image builds there quite happily, and
#        then the copy at the end misses:
#        "cp: build/mpy-16mb-tflm/firmware.uf2: No such file or directory"
#        Exit 1 on a successful build, with a good .uf2 buried in the
#        MicroPython checkout that build/ is free to delete.
abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        "") printf '%s\n' "" ;;
        *)  printf '%s\n' "$REPO/$1" ;;
    esac
}

BOARD_DIR="$REPO/firmware/boards/$BOARD"
MPY_DIR="$(abspath "${MPY_DIR:-$REPO/build/micropython}")"
BUILD_DIR="$(abspath "${BUILD_DIR:-$REPO/build/mpy-16mb}")"
TOOLCHAIN_DIR="$(abspath "${TOOLCHAIN_DIR:-$REPO/build/toolchain}")"
OUT="$(abspath "${OUT:-$REPO/firmware/$BOARD-$MPY_VERSION-16MB.uf2}")"

# The hook the TensorFlow Lite Micro work needs. An empty USER_C_MODULES is the
# same build as no USER_C_MODULES, so this costs nothing until something uses
# it, and means a usermod does not arrive as a change to this script.
USER_C_MODULES="$(abspath "${USER_C_MODULES:-}")"

[ -d "$BOARD_DIR" ] || { echo "no board definition at $BOARD_DIR" >&2; exit 2; }

# CMake generates Unix Makefiles here -- ports/rp2 does not ask for Ninja --
# so make is the build tool, not an alternative to it.
for tool in cmake make git curl; do
    command -v "$tool" > /dev/null || { echo "missing $tool (brew install cmake)" >&2; exit 2; }
done

# --- toolchain ---------------------------------------------------------------

TC="arm-gnu-toolchain-$TOOLCHAIN_VERSION-$TOOLCHAIN_HOST-arm-none-eabi"
TC_ROOT="$TOOLCHAIN_DIR/$TC"

if [ ! -x "$TC_ROOT/bin/arm-none-eabi-gcc" ]; then
    mkdir -p "$TOOLCHAIN_DIR"
    TARBALL="$TOOLCHAIN_DIR/$TC.tar.xz"
    if [ ! -f "$TARBALL" ]; then
        echo "fetching $TC (135 MB, once)"
        curl -sSL -o "$TARBALL" \
            "https://gitlab.arm.com/api/v4/projects/tooling%2Fgnu-toolchains-for-arm/packages/generic/gnu-toolchain/$TOOLCHAIN_VERSION/$TC.tar.xz"
    fi
    if [ -n "$TOOLCHAIN_SHA256" ]; then
        echo "$TOOLCHAIN_SHA256  $TARBALL" | shasum -a 256 -c - \
            || { echo "toolchain hash mismatch - refusing to build" >&2; exit 3; }
    fi
    echo "unpacking $TC"
    tar -xf "$TARBALL" -C "$TOOLCHAIN_DIR"
fi

PATH="$TC_ROOT/bin:$PATH"
export PATH

# --- micropython -------------------------------------------------------------

if [ ! -d "$MPY_DIR/.git" ]; then
    echo "cloning micropython $MPY_VERSION"
    git clone --depth 1 --branch "$MPY_VERSION" \
        https://github.com/micropython/micropython.git "$MPY_DIR"
fi

# Whatever is in build/micropython has to be the pinned version, not whatever a
# previous run or a stray `git pull` left there -- the whole point of the pin.
HAVE="$(git -C "$MPY_DIR" describe --tags --exact-match 2>/dev/null || echo unknown)"
if [ "$HAVE" != "$MPY_VERSION" ]; then
    echo "$MPY_DIR is at '$HAVE', not $MPY_VERSION" >&2
    echo "  rm -rf $MPY_DIR   # and re-run, or set MPY_DIR" >&2
    exit 4
fi

# `make submodules` is a no-op once they are present, but it is not free (it
# configures a throwaway CMake tree to work out which ones this board needs),
# so it only runs when pico-sdk is missing.
if [ ! -f "$MPY_DIR/lib/pico-sdk/README.md" ]; then
    echo "fetching submodules"
    make -C "$MPY_DIR/ports/rp2" \
        BOARD_DIR="$BOARD_DIR" BUILD="$BUILD_DIR" submodules
fi

# --- build -------------------------------------------------------------------

# mpy-cross first, on its own, with no -j. It is a *host* build (the frozen
# manifest is cross-compiled with it), and its qstr generation races under
# parallel make: the generated qstrdefs header is read while still being
# written, and every file that uses a qstr then fails with a wall of
# "use of undeclared identifier 'MP_QSTR_sort'". Serially it is a 30 s build
# and the failure cannot happen. Observed here, not theoretical.
#
# USER_C_MODULES is cleared for this one command on purpose. mpy-cross builds
# through py/py.mk, which reads it from the environment and expects the *make*
# convention -- a directory holding micropython.mk. The cmake ports expect a
# .cmake file. Leave it set and a perfectly good module path fails here, before
# the port is even configured, as:
#
#   py/py.mk:37: *** USER_C_MODULES doesn't exist: .../micropython.cmake
#
# which names the file that does exist and is the right one.
USER_C_MODULES= make -C "$MPY_DIR/mpy-cross"

echo "building $BOARD, micropython $MPY_VERSION, $(arm-none-eabi-gcc -dumpversion)"
if [ -n "$USER_C_MODULES" ]; then
    echo "  with USER_C_MODULES=$USER_C_MODULES"
fi

# Extra -D flags for the module being built, e.g.
#
#   EXTRA_CMAKE_ARGS=-DTFLM_STRIP_ERROR_STRINGS=1
#
# They cannot go through `make CMAKE_ARGS=...`: ports/rp2's Makefile builds
# CMAKE_ARGS with `+=`, and a command-line variable overrides the whole thing,
# taking MICROPY_BOARD and MICROPY_BOARD_DIR with it. So the configure step is
# done here instead, exactly as the Makefile would do it -- the Makefile then
# finds $BUILD_DIR/Makefile already present and goes straight to building.
#
# This exists because the alternative is worse than not supporting the flag:
# a -D that is silently dropped produces a successful build of the wrong
# thing, and nothing downstream says so.
if [ -n "${EXTRA_CMAKE_ARGS:-}" ] && [ ! -f "$BUILD_DIR/Makefile" ]; then
    echo "  with $EXTRA_CMAKE_ARGS"
    # shellcheck disable=SC2086
    cmake -S "$MPY_DIR/ports/rp2" -B "$BUILD_DIR" -DPICO_BUILD_DOCS=0 \
        -DMICROPY_BOARD="$BOARD" \
        -DMICROPY_BOARD_DIR="$BOARD_DIR" \
        ${USER_C_MODULES:+-DUSER_C_MODULES="$USER_C_MODULES"} \
        $EXTRA_CMAKE_ARGS
fi

# BOARD_DIR outside the MicroPython tree is a supported path -- ports/rp2's
# Makefile takes the board name from the directory name and hands CMake the
# absolute path -- and it is the reason the board definition can live in this
# repo, under version control, instead of as an untracked edit inside a
# checkout that build/ is free to delete.
make -C "$MPY_DIR/ports/rp2" -j"$(sysctl -n hw.ncpu)" \
    BOARD_DIR="$BOARD_DIR" \
    BUILD="$BUILD_DIR" \
    ${USER_C_MODULES:+USER_C_MODULES="$USER_C_MODULES"}

cp "$BUILD_DIR/firmware.uf2" "$OUT"

# --- what came out -----------------------------------------------------------

# Numbers, not adjectives, and these are the ones worth reading before
# flashing. `embedded drive` is the filesystem MicroPython will format, and the
# image itself carries it: rp2_flash.c declares the block device with bi_decl(),
# so picotool reads the real compiled-in extent out of the binary rather than
# repeating what this script hoped it would be. Against the stock image the
# same line reads 0x10100000-0x10400000 (3072K).
#
# If it says 3072K, the board definition did not take effect and this file is
# the stock image under a different name.
echo
echo "$OUT"
ls -l "$OUT" | awk '{printf "  uf2            %s bytes\n", $5}'
# text + data only: bss is RAM, and what matters here is what the image
# occupies in the 1 MB kept back from the filesystem.
arm-none-eabi-size "$BUILD_DIR/firmware.elf" | \
    awk 'NR==2 {printf "  firmware       %s bytes of the 1 MB reserved for it\n", $1 + $2}'
# An absolute symbol, so nm prints its value: the FLASH region length the
# linker used. 01000000 is the 16 MB this whole exercise is about.
arm-none-eabi-nm "$BUILD_DIR/firmware.elf" | \
    awk '/__micropy_flash_size__/ {printf "  linker FLASH   0x%s\n", $1}'
if command -v picotool > /dev/null; then
    picotool info -a "$OUT" | sed -nE 's/^ (embedded drive|pico_board|sdk version):/  \1:/p'
fi

echo
echo "Not verifiable without the board: that the part answers above 4 MB, and"
echo "that littlefs formats and mounts the full 15 MB. Flash, then measure it"
echo "with os.statvfs('/') -- do not take the numbers above as proof of either."
echo
echo "FLASHING WIPES THE FILESYSTEM. The new firmware finds a littlefs"
echo "superblock whose block count is the old 3 MB one, fails to mount it, and"
echo "reformats (ports/rp2/modules/_boot.py). Copy anything that exists only on"
echo "the board off it first. See docs/restore-factory-firmware.md."
