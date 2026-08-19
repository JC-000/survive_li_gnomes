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
# Normalise before anything uses it. `-ffile-prefix-map` below matches
# __FILE__ as a literal string prefix, and __FILE__ carries the path the
# compiler was handed via -I -- which cmake normalises. Leave TFLM_DIR as
# `.../usermod/tflm/../../../vendor/tflm` and the flag appears on every compile
# line, matches nothing, and the paths stay in the image. It looks fixed in the
# build log and is not; verify with `strings firmware.elf | grep vendor/tflm`.
get_filename_component(TFLM_DIR "${TFLM_DIR}" ABSOLUTE)
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

# TFLM's diagnostic strings, and the newlib printf machinery they drag in
# behind them, are worth 46,336 bytes of the image -- measured, by building
# this both ways against the real firmware. They are on by default because
# during bring-up "Failed to allocate tail memory. Requested: 816" is the
# difference between a diagnosis and a guess, and off is a one-line change:
#
#   TFLM_STRIP_ERROR_STRINGS=1 USER_C_MODULES=... ./tools/build_firmware.sh
#
# Strip them once the arena is sized and the model is settled.
if(TFLM_STRIP_ERROR_STRINGS)
    target_compile_definitions(tflm_lib PUBLIC TF_LITE_STRIP_ERROR_STRINGS)
    message(STATUS "tflm: error strings stripped")
endif()

# -fno-rtti/-fno-exceptions are what TFLM is written for and what every
# embedded integration of it uses; without them the image pulls in the
# unwinder. -Wno-error because TFLM does not compile warning-clean under
# MicroPython's flags and its warnings are not this project's to fix.
#
# No -ffast-math, and this is load-bearing rather than habit: the int8 kernels
# are integer end to end, but the quantisation multipliers are computed in
# double at AllocateTensors, and that is exactly where a relaxed
# floating-point mode would break the bit-exactness the module is for.
# These are PRIVATE on tflm_lib and not INTERFACE on usermod_tflm, which is the
# distinction that matters: compile options set INTERFACE on the usermod target
# reach `modtflm.c` and nothing else. TFLM compiles here, so the flags have to
# be here. (fw-16mb flagged the reverse mistake as the likely one.)
#
# -fno-use-cxa-atexit and -fno-unwind-tables match what the rp2 port sets for
# its own C++, and are not inherited: `-Wall -Werror` and the port's C++ flags
# are PRIVATE on the `firmware` target, so a separate static library gets
# neither. Without -fno-use-cxa-atexit a static object with a destructor
# registers through __cxa_atexit and wants __dso_handle.
target_compile_options(tflm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-use-cxa-atexit>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables>
    -Wno-error
    -Wno-error=float-conversion
    -Wno-error=double-promotion
    -Wno-error=sign-compare
    -Wno-error=unused-const-variable
    -Wno-error=maybe-uninitialized
    # TFLM's assertion macros bake __FILE__ into the image. Unmapped that is 19
    # absolute host paths and 2,133 bytes of flash -- but the flash is the minor
    # half. The major half is that the same sources from a different checkout
    # directory produce a different binary, which quietly voids any
    # byte-for-byte reproducibility claim about the firmware. (Found by fw-16mb
    # in a string diff of the two images.)
    -ffile-prefix-map=${TFLM_DIR}=tflm
    -ffile-prefix-map=${CMAKE_CURRENT_LIST_DIR}=usermod/tflm
    # -O2 rather than the port's -Os, and it is not free: measured on this
    # target, -O2 costs 15,948 bytes of flash over -Os (62,332 against 46,384,
    # error strings stripped). Taken because inference time on the board is the
    # one figure this module still has no measurement for, and because flash is
    # the resource there is most of -- 710 KB free in the reserve. If the board
    # turns out to be fast enough anyway, -Os is the cheaper default.
    -O2
)

# The static library needs MicroPython's own include path and compile
# definitions to see mpconfigport.h; gathering the target's properties is how
# the port hands those to a usermod.
include(${MICROPY_DIR}/py/py.cmake)
micropy_gather_target_properties(usermod_tflm)

target_link_libraries(usermod_tflm INTERFACE tflm_lib)
target_link_libraries(usermod INTERFACE usermod_tflm)
