#!/bin/bash
# Fetch and assemble a TensorFlow Lite Micro source tree into vendor/tflm.
#
# Not vendored into git: ~10 MB of upstream sources plus three third-party
# header trees, all reproducible from the pinned commits below. Same reasoning
# as `build/` -- regenerable, binary-ish, and large.
#
# The pins are the point. TFLM's arithmetic is what this project is buying, so
# the tree the firmware is built from has to be the tree the host bit-exactness
# proof (`tools/tflm_vs_tflite.py`) was run against, and "whatever main was
# that day" is not that.
#
#   ./tools/fetch_tflm.sh            # into vendor/tflm
#   TFLM_DIR=/somewhere ./tools/fetch_tflm.sh
set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TFLM_DIR:-$HERE/vendor/tflm}"

# Pinned 2026-08-18. Bumping these means re-running tools/tflm_vs_tflite.py:
# a TFLM release could change a kernel, and the whole case for the module is
# that its output equals TFLite's.
TFLM_COMMIT=f8c117b558b0cf25cd8bff7143f4c361fbeced4f
FLATBUFFERS_TAG=v25.9.23
FLATBUFFERS_DIR=flatbuffers-25.9.23
GEMMLOWP_COMMIT=719139ce755a0f31cbf1c37f7f98adcc7fc9f425
RUY_COMMIT=d37128311b445e758136b8602d1bbd2a755e115d

if [ -f "$DEST/tensorflow/lite/micro/micro_interpreter.cc" ]; then
    echo "$DEST already populated; delete it to re-fetch."
    exit 0
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== tflite-micro @ $TFLM_COMMIT =="
git -C "$TMP" clone --quiet --filter=blob:none https://github.com/tensorflow/tflite-micro.git tflm
git -C "$TMP/tflm" checkout --quiet "$TFLM_COMMIT"
mkdir -p "$DEST/tensorflow"
cp -R "$TMP/tflm/tensorflow/lite" "$DEST/tensorflow/"
mkdir -p "$DEST/tensorflow/compiler/mlir/lite"
cp -R "$TMP/tflm/tensorflow/compiler/mlir/lite/core" "$DEST/tensorflow/compiler/mlir/lite/"
cp -R "$TMP/tflm/tensorflow/compiler/mlir/lite/schema" "$DEST/tensorflow/compiler/mlir/lite/"
# schema_utils.cc includes it, and nothing in the paths above provides it.
cp -R "$TMP/tflm/tensorflow/compiler/mlir/lite/kernels" "$DEST/tensorflow/compiler/mlir/lite/"
# `micro_ops.h` includes signal/micro/kernels/*.h unconditionally, so the tree
# has to carry them even though this project registers no signal operator.
cp -R "$TMP/tflm/signal" "$DEST/signal"
cp "$TMP/tflm/LICENSE" "$DEST/LICENSE"

echo "== third_party =="
mkdir -p "$DEST/third_party"
cd "$TMP"
curl -sSL -o fb.zip "https://github.com/google/flatbuffers/archive/refs/tags/$FLATBUFFERS_TAG.zip"
curl -sSL -o gemmlowp.zip "https://github.com/google/gemmlowp/archive/$GEMMLOWP_COMMIT.zip"
curl -sSL -o ruy.zip "https://github.com/google/ruy/archive/$RUY_COMMIT.zip"
unzip -q fb.zip && unzip -q gemmlowp.zip && unzip -q ruy.zip

# TFLM patches flatbuffers to remove its dynamic allocation. Applying it is not
# cosmetic: without it the interpreter can reach the C heap, which on the board
# is not MicroPython's heap.
(cd "$FLATBUFFERS_DIR" && git init -q . && \
 git apply "$TMP/tflm/tensorflow/lite/micro/tools/make/flatbuffers.patch")
mkdir -p "$DEST/third_party/flatbuffers"
cp -R "$FLATBUFFERS_DIR/include" "$DEST/third_party/flatbuffers/"
cp "$FLATBUFFERS_DIR/LICENSE" "$DEST/third_party/flatbuffers/" 2>/dev/null || true

mkdir -p "$DEST/third_party/gemmlowp"
cp -R "gemmlowp-$GEMMLOWP_COMMIT/fixedpoint" "$DEST/third_party/gemmlowp/"
cp -R "gemmlowp-$GEMMLOWP_COMMIT/internal" "$DEST/third_party/gemmlowp/"
cp "gemmlowp-$GEMMLOWP_COMMIT/LICENSE" "$DEST/third_party/gemmlowp/"

mkdir -p "$DEST/third_party/ruy/ruy/profiler"
cp "ruy-$RUY_COMMIT/ruy/profiler/instrumentation.h" "$DEST/third_party/ruy/ruy/profiler/"
cp "ruy-$RUY_COMMIT/LICENSE" "$DEST/third_party/ruy/"

# Rename .cc -> .cpp, which is what upstream's own create_tflm_tree.py does for
# IDE integrations and what MicroPython's Make-based ports require: py.mk
# pattern-matches SRC_USERMOD_CXX against `%.cpp` only, so a `.cc` file is
# accepted into the variable, never compiled, and never linked. The build then
# succeeds and the module is simply absent -- `ImportError: no module named
# 'tflm'` after a clean fifteen-minute build. Nothing in the tree #includes a
# .cc file, so the rename is safe.
find "$DEST" -name '*.cc' -exec sh -c 'mv "$1" "${1%.cc}.cpp"' _ {} \;

echo
echo "TFLM tree at $DEST"
du -sh "$DEST"
echo "$(find "$DEST" -name '*.cpp' | wc -l | tr -d ' ') C++ sources, $(find "$DEST" -name '*.h' | wc -l | tr -d ' ') headers"
