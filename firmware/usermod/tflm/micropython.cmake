# TensorFlow Lite Micro as a MicroPython USER_C_MODULE, for the rp2 port.
#
#   cmake -S <mp>/ports/rp2 -B build \
#         -DMICROPY_BOARD=WAVESHARE_RP2350_TOUCH_EPAPER_154 \
#         -DMICROPY_BOARD_DIR=<repo>/firmware/boards/WAVESHARE_RP2350_TOUCH_EPAPER_154 \
#         -DUSER_C_MODULES=<repo>/firmware/usermod/tflm/micropython.cmake
#
# TFLM_DIR must point at a generated TFLM tree -- `tools/fetch_tflm.sh` makes
# one. It is not vendored into this repo: it is ~10 MB of upstream sources plus
# three third-party header trees, all reproducible from two pinned commits.
#
# Sources are globbed the way TFLM's own Makefile globs them rather than listed,
# because the file set changes upstream and a stale explicit list fails as a
# link error at the end of a fifteen-minute build. The one thing that must not
# be globbed is the kernel directory: `kernels/*.cpp` is the reference set and
# `kernels/cmsis_nn/*.cpp` would silently replace seven of them with the
# optimised ARM versions. See docs/tflm-usermod.md for why that choice is not
# free -- the reference kernels are the ones proved bit-identical to host
# TFLite, and this build takes correctness over the speed it is not short of.

if(NOT DEFINED TFLM_DIR)
    set(TFLM_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../vendor/tflm)
endif()
if(NOT EXISTS ${TFLM_DIR}/tensorflow/lite/micro/micro_interpreter.cpp)
    message(FATAL_ERROR
        "TFLM_DIR=${TFLM_DIR} has no TFLM tree. Run tools/fetch_tflm.sh first.")
endif()

# TFLM is a real static library rather than more INTERFACE sources, so that
# MicroPython's QSTR scan -- which runs over every source of a usermod target --
# never sees four hundred C++ translation units that contain no QSTRs. Only
# modtflm.c is scanned, because only modtflm.c has any.
add_library(usermod_tflm INTERFACE)
add_library(tflm_lib STATIC)

set(TF_LITE ${TFLM_DIR}/tensorflow/lite)
set(TF_MICRO ${TF_LITE}/micro)

file(GLOB TFLM_SRCS
    ${TF_LITE}/c/*.c
    ${TF_LITE}/core/api/*.cpp
    ${TF_LITE}/core/c/*.cpp
    ${TF_LITE}/kernels/*.cpp
    ${TF_LITE}/kernels/internal/*.cpp
    ${TF_LITE}/kernels/internal/reference/*.cpp
    ${TF_LITE}/schema/*.cpp
    ${TFLM_DIR}/tensorflow/compiler/mlir/lite/core/api/*.cpp
    ${TFLM_DIR}/tensorflow/compiler/mlir/lite/schema/*.cpp
    ${TF_MICRO}/*.cpp
    ${TF_MICRO}/arena_allocator/*.cpp
    ${TF_MICRO}/kernels/*.cpp
    ${TF_MICRO}/memory_planner/*.cpp
    ${TF_MICRO}/tflite_bridge/*.cpp
)

# Test scaffolding that the glob picks up and that pulls in host-only
# dependencies. Dropping these is what the generated tree's own Makefile does.
foreach(pattern
        "_test\\.cpp$"
        "test_helpers"
        "kernel_runner"
        "mock_micro_graph"
        "fake_micro_context"
        "micro_test"
        "conv_test_common"
        "flexbuffers_generated_data"
        "detection_postprocess")
    list(FILTER TFLM_SRCS EXCLUDE REGEX ${pattern})
endforeach()

target_sources(tflm_lib PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/tflm_shim.cpp
    ${TFLM_SRCS}
)
target_sources(usermod_tflm INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/modtflm.c
)

set(TFLM_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}
    ${TFLM_DIR}
    ${TFLM_DIR}/third_party/flatbuffers/include
    ${TFLM_DIR}/third_party/gemmlowp
    ${TFLM_DIR}/third_party/ruy
)
target_include_directories(tflm_lib PUBLIC ${TFLM_INCLUDES})
target_include_directories(usermod_tflm INTERFACE ${TFLM_INCLUDES})

target_compile_definitions(tflm_lib PUBLIC
    # Selects TFLM's no-malloc paths. Not optional: without it the interpreter
    # reaches for operator new, which on this port would be the C heap and not
    # MicroPython's, and the arena discipline this module exists for is gone.
    TF_LITE_STATIC_MEMORY=1
    TF_LITE_MCU_DEBUG_LOG
    NDEBUG
)

# -fno-rtti/-fno-exceptions are what TFLM is written for and what every
# embedded integration of it uses; without them the image pulls in the
# unwinder. -Wno-error because TFLM does not compile warning-clean under
# MicroPython's flags and its warnings are not this project's to fix.
#
# No -ffast-math, and this is load-bearing rather than habit: the int8 kernels
# are integer end to end, but the quantisation multipliers are computed in
# double at AllocateTensors, and that is exactly where a relaxed
# floating-point mode would break the bit-exactness the module is for.
target_compile_options(tflm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    -Wno-error
    -Wno-error=float-conversion
    -Wno-error=double-promotion
    -Wno-error=sign-compare
    -Wno-error=unused-const-variable
    -Wno-error=maybe-uninitialized
    -O2
)

# The static library needs MicroPython's own include path and compile
# definitions to see mpconfigport.h; gathering the target's properties is how
# the port hands those to a usermod.
include(${MICROPY_DIR}/py/py.cmake)
micropy_gather_target_properties(usermod_tflm)

target_link_libraries(usermod_tflm INTERFACE tflm_lib)
target_link_libraries(usermod INTERFACE usermod_tflm)
