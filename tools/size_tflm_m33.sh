#!/bin/bash
# Cross-compile TFLM + the shim for the RP2350's Cortex-M33 and measure what it
# actually costs in flash and static RAM.
#
# The number that matters is not the archive size -- 1.2 MB here, 1.3 MB for
# OpenMV's prebuilt libtflm.a -- because most of it is kernels nothing links.
# It is the size of the subset the linker keeps once --gc-sections has run
# against a program that calls exactly what the model needs. That is what this
# measures, and it is measured as a *difference* against an otherwise identical
# link, so newlib's own baseline is not counted as TFLM's.
#
#   CROSS=/path/to/arm-gnu-toolchain/bin ./tools/size_tflm_m33.sh
#   CROSS=... ./tools/size_tflm_m33.sh build/some_other.tflite
#
# Subtract the model from the reported delta: it is linked in as a const array
# so that --gc-sections sees which kernels the graph actually reaches, and a
# stub model would undercount.
set -e

MODEL="${1:-$(cd "$(dirname "$0")/.." && pwd)/build/si_real.tflite}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TFLM="${TFLM_DIR:-$HERE/vendor/tflm}"
MOD="$HERE/firmware/usermod/tflm"
OUT="${TFLM_M33_BUILD:-$HERE/build/tflm-m33}"

# Point CROSS at an arm-none-eabi bin directory. Nothing is installed by this
# script; the toolchain is whatever is already on the machine (fw-16mb has one
# for the firmware build) or an Arm GNU tarball unpacked anywhere.
CROSS="${CROSS:?set CROSS to an arm-none-eabi toolchain bin directory}"
CC="$CROSS/arm-none-eabi-gcc"
CXX="$CROSS/arm-none-eabi-g++"
TC="$CROSS"

if [ ! -f "$TFLM/tensorflow/lite/micro/micro_interpreter.cpp" ]; then
    echo "No TFLM tree at $TFLM -- run ./tools/fetch_tflm.sh first." >&2
    exit 1
fi
mkdir -p "$OUT/obj"

INC="-I$TFLM -I$TFLM/third_party/flatbuffers/include \
     -I$TFLM/third_party/gemmlowp -I$TFLM/third_party/ruy -I$MOD"

# Exactly the rp2 port's own architecture flags for RP2350 (ports/rp2 with
# PICO_PLATFORM=rp2350 builds -mcpu=cortex-m33 with softfp and fpv5-sp-d16).
ARCH="-mcpu=cortex-m33 -mthumb -mfloat-abi=softfp -mfpu=fpv5-sp-d16"
# TFLM_DEFS lets the caller measure the release configuration:
#   TFLM_DEFS=-DTF_LITE_STRIP_ERROR_STRINGS ./tools/size_tflm_m33.sh
# which drops every diagnostic string and the newlib printf machinery they drag
# in behind them. Worth ~47 KB, and worth having *off* during bring-up.
DEFS="-DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG -DNDEBUG ${TFLM_DEFS:-}"
COMMON="$ARCH $DEFS $INC -O2 -ffunction-sections -fdata-sections -w"
CXXFLAGS="-std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics $COMMON"
CFLAGS="-std=c11 $COMMON"

collect() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | sort; }

SRCS=""
for d in \
  "$TFLM/tensorflow/lite/core/api" \
  "$TFLM/tensorflow/compiler/mlir/lite/core/api" \
  "$TFLM/tensorflow/compiler/mlir/lite/schema" \
  "$TFLM/tensorflow/lite/core/c" \
  "$TFLM/tensorflow/lite/kernels" \
  "$TFLM/tensorflow/lite/kernels/internal" \
  "$TFLM/tensorflow/lite/kernels/internal/reference" \
  "$TFLM/tensorflow/lite/schema" \
  "$TFLM/tensorflow/lite/micro" \
  "$TFLM/tensorflow/lite/micro/arena_allocator" \
  "$TFLM/tensorflow/lite/micro/kernels" \
  "$TFLM/tensorflow/lite/micro/memory_planner" \
  "$TFLM/tensorflow/lite/micro/tflite_bridge" ; do
  SRCS="$SRCS $(collect "$d" '*.cpp')"
done
SRCS="$SRCS $MOD/tflm_shim.cpp"
CSRCS="$(collect $TFLM/tensorflow/lite/c '*.c')"

DROP="_test test_helpers kernel_runner mock_micro_graph fake_micro_context
      micro_test conv_test_common flexbuffers_generated_data
      detection_postprocess"

OBJS=""
for s in $SRCS $CSRCS; do
  base=$(basename "$s"); skip=0
  for d in $DROP; do case "$base" in *"$d"*) skip=1;; esac; done
  [ "$skip" = 1 ] && continue
  o="$OUT/obj/$(echo "$s" | sed "s|$TFLM/||; s|$HERE/||; s|/|_|g").o"
  if [ ! -f "$o" ] || [ "$s" -nt "$o" ]; then
    case "$s" in
      *.c)   "$CC"  $CFLAGS   -c "$s" -o "$o" ;;
      *.cpp) "$CXX" $CXXFLAGS -c "$s" -o "$o" ;;
    esac
  fi
  OBJS="$OBJS $o"
done

"$TC/arm-none-eabi-ar" rcs "$OUT/libtflm_m33.a" $OBJS
echo "archive (all objects, pre-link):"
ls -la "$OUT/libtflm_m33.a"

# A program that uses the module the way the board would, so --gc-sections
# keeps the kernels this model reaches and drops the rest. `model` is a real
# si_real.tflite baked in as a const array -- the shape of the model decides
# which kernels survive, so a stub model would undercount.
python3 - "$MODEL" > "$OUT/si_model.c" <<'PYEOF'
import sys
data = open(sys.argv[1], "rb").read()
print("#include <stdint.h>")
print("__attribute__((aligned(16))) const uint8_t si_model[] = {")
for i in range(0, len(data), 16):
    print(",".join(str(b) for b in data[i:i + 16]) + ",")
print("};")
print("const unsigned int si_model_len = %d;" % len(data))
PYEOF

cat > "$OUT/probe.c" <<'EOF'
#include <stdint.h>
#include <string.h>
#include "tflm_shim.h"
extern const uint8_t si_model[];
extern const unsigned int si_model_len;
static uint8_t arena[40 * 1024] __attribute__((aligned(16)));
static int8_t input[80 * 26];
static float scores[22];
int main(void) {
    int err = 0;
    tflm_model *m = tflm_new_model(si_model, si_model_len, arena, sizeof(arena), &err);
    if (!m) return err;
    volatile int rc = tflm_invoke(m, input, sizeof(input), 0, scores, 22);
    volatile size_t used = tflm_arena_used(m);
    (void)rc; (void)used;
    return 0;
}
/* Newlib wants these; the real firmware has MicroPython's. */
void _exit(int c) { (void)c; for (;;) {} }
int _write(int f, char *p, int l) { (void)f; (void)p; return l; }
int _read(int f, char *p, int l) { (void)f; (void)p; (void)l; return 0; }
int _close(int f) { (void)f; return -1; }
int _lseek(int f, int o, int w) { (void)f; (void)o; (void)w; return 0; }
int _fstat(int f, void *s) { (void)f; (void)s; return 0; }
int _isatty(int f) { (void)f; return 1; }
int _getpid(void) { return 1; }
int _kill(int p, int s) { (void)p; (void)s; return -1; }
void *_sbrk(int i) { extern char end; static char *h = &end; char *p = h; h += i; return p; }
EOF

"$CC" $ARCH -O2 -x c "$OUT/si_model.c" -c -o "$OUT/obj/si_model.o"
"$CC" $ARCH -O2 -I"$MOD" -ffunction-sections -fdata-sections \
      -c "$OUT/probe.c" -o "$OUT/obj/probe.o"

# Two links: one with the module, one without, so the difference is the
# module's cost and not newlib's.
"$CXX" $ARCH -specs=nosys.specs -Wl,--gc-sections -Wl,-Map,"$OUT/with.map" \
       "$OUT/obj/probe.o" "$OUT/obj/si_model.o" "$OUT/libtflm_m33.a" \
       -o "$OUT/with.elf" 2>&1 | tail -5

cat > "$OUT/bare.c" <<'EOF'
int main(void) { return 0; }
void _exit(int c) { (void)c; for (;;) {} }
EOF
"$CC" $ARCH -O2 -specs=nosys.specs -Wl,--gc-sections "$OUT/bare.c" -o "$OUT/bare.elf"

echo
echo "== sizes =="
"$TC/arm-none-eabi-size" "$OUT/bare.elf" "$OUT/with.elf"
