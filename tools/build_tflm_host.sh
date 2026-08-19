#!/bin/bash
# Build TFLM plus firmware/usermod/tflm's C shim into a host shared library.
#
# This is the host half of the bit-exactness proof: `tools/tflm_vs_tflite.py`
# drives this library through ctypes and compares it, output tensor by output
# tensor, against `tf.lite.Interpreter`. The point of building *the shim* rather
# than a separate harness is that the code proved here is literally the code the
# firmware runs -- same file, same op resolver, same arena discipline.
#
#   ./tools/fetch_tflm.sh
#   ./tools/build_tflm_host.sh
#   .venv/bin/python tools/tflm_vs_tflite.py --model build/si_real.tflite \
#       --classes build/si_real.json --bins 'build/kw_unknown_*.bin' \
#       --takes takes takes-oov
#
# It does not use TFLM's own Makefile. That Makefile requires GNU make >= 3.82
# and macOS ships 3.81, and globbing the same directories it globs costs
# nothing -- the CMake and Make integrations glob too, so all three build the
# same set and a source appearing upstream is picked up by all three at once.
set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TFLM="${TFLM_DIR:-$HERE/vendor/tflm}"
MOD="$HERE/firmware/usermod/tflm"
OUT="${TFLM_HOST_BUILD:-$HERE/build/tflm-host}"

if [ ! -f "$TFLM/tensorflow/lite/micro/micro_interpreter.cpp" ]; then
    echo "No TFLM tree at $TFLM -- run ./tools/fetch_tflm.sh first." >&2
    exit 1
fi

mkdir -p "$OUT/obj"

INC="-I$TFLM -I$TFLM/third_party/flatbuffers/include \
     -I$TFLM/third_party/gemmlowp -I$TFLM/third_party/ruy -I$MOD"

# TF_LITE_STATIC_MEMORY selects TFLM's no-malloc paths and is what the firmware
# defines. Without it this would be a different library from the one on the
# board, which would make the whole comparison worthless.
DEFS="-DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG -DNDEBUG"
CXXFLAGS="-std=c++17 -O2 -fno-rtti -fno-exceptions -fPIC -w $DEFS $INC"
CFLAGS="-std=c11 -O2 -fPIC -w $DEFS $INC"

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
CSRCS="$(collect "$TFLM/tensorflow/lite/c" '*.c')"

# Keep this list in step with micropython.cmake and micropython.mk. All three
# drop the same test scaffolding, and a mismatch shows up as a link error.
DROP="_test test_helpers kernel_runner mock_micro_graph fake_micro_context
      micro_test conv_test_common flexbuffers_generated_data
      detection_postprocess"

OBJS=""; n=0
for s in $SRCS $CSRCS; do
    base=$(basename "$s"); skip=0
    for d in $DROP; do case "$base" in *"$d"*) skip=1;; esac; done
    [ "$skip" = 1 ] && continue
    o="$OUT/obj/$(echo "$s" | sed "s|$TFLM/||; s|$HERE/||; s|/|_|g").o"
    if [ ! -f "$o" ] || [ "$s" -nt "$o" ]; then
        case "$s" in
            *.c)   clang   $CFLAGS   -c "$s" -o "$o" ;;
            *.cpp) clang++ $CXXFLAGS -c "$s" -o "$o" ;;
        esac
    fi
    OBJS="$OBJS $o"; n=$((n + 1))
done

case "$(uname -s)" in
    Darwin) LIB="$OUT/libtflm_host.dylib" ;;
    *)      LIB="$OUT/libtflm_host.so" ;;
esac
clang++ -shared -o "$LIB" $OBJS
echo "$n objects -> $LIB"
ls -la "$LIB"
