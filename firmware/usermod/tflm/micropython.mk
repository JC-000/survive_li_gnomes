# Makefile half of the tflm user C module. The rp2 port uses micropython.cmake;
# this is here for the **unix** port, which is where the module is proved to
# compile and run without holding the board:
#
#   make -C ports/unix \
#        USER_C_MODULES=<repo>/firmware/usermod \
#        TFLM_DIR=<repo>/vendor/tflm
#
# Note the path: the Make mechanism wants the *parent* of the module
# directories and globs `*/micropython.mk` under it, where the CMake mechanism
# wants the .cmake file itself. Pointing Make at `firmware/usermod/tflm`
# instead builds and links cleanly and then fails with `ImportError: no module
# named 'tflm'`, because nothing was ever added.
#
# That build is not a stand-in for the firmware -- different compiler, different
# word size -- but it does answer "does the binding compile and does `tflm.new`
# return something that classifies", which is the whole of what a host can say.

TFLM_MOD_DIR := $(USERMOD_DIR)
TFLM_DIR ?= $(TFLM_MOD_DIR)/../../../vendor/tflm

TF_LITE := $(TFLM_DIR)/tensorflow/lite
TF_MICRO := $(TF_LITE)/micro

TFLM_ALL_CPP := \
    $(wildcard $(TF_LITE)/core/api/*.cpp) \
    $(wildcard $(TF_LITE)/core/c/*.cpp) \
    $(wildcard $(TF_LITE)/kernels/*.cpp) \
    $(wildcard $(TF_LITE)/kernels/internal/*.cpp) \
    $(wildcard $(TF_LITE)/kernels/internal/reference/*.cpp) \
    $(wildcard $(TF_LITE)/schema/*.cpp) \
    $(wildcard $(TFLM_DIR)/tensorflow/compiler/mlir/lite/core/api/*.cpp) \
    $(wildcard $(TFLM_DIR)/tensorflow/compiler/mlir/lite/schema/*.cpp) \
    $(wildcard $(TF_MICRO)/*.cpp) \
    $(wildcard $(TF_MICRO)/arena_allocator/*.cpp) \
    $(wildcard $(TF_MICRO)/kernels/*.cpp) \
    $(wildcard $(TF_MICRO)/memory_planner/*.cpp) \
    $(wildcard $(TF_MICRO)/tflite_bridge/*.cpp)

# Test scaffolding the wildcards pick up. Same list as micropython.cmake's;
# they must stay in step, and a mismatch shows up as an undefined symbol at
# link rather than as anything subtle.
TFLM_DROP := %_test.cpp %test_helpers.cpp %kernel_runner.cpp %mock_micro_graph.cpp \
             %fake_micro_context.cpp %testing_helpers_test.cpp \
             %conv_test_common.cpp %flexbuffers_generated_data.cpp \
             %detection_postprocess.cpp

# TFLM's own sources go in the *LIB* variables, and that is not cosmetic:
# py.mk adds SRC_USERMOD_C/CXX to SRC_QSTR, which preprocesses every listed
# file in one clang invocation to harvest MP_QSTR_ names. Four hundred TFLM
# translation units contain no QSTRs and preprocessing them costs minutes and
# a great deal of memory. The LIB variables compile and link without being
# scanned, which is exactly what a third-party library wants.
SRC_USERMOD_LIB_CXX += $(filter-out $(TFLM_DROP),$(TFLM_ALL_CPP)) \
                       $(TFLM_MOD_DIR)/tflm_shim.cpp
SRC_USERMOD_LIB_C += $(wildcard $(TF_LITE)/c/*.c)

# Only the binding itself is scanned -- it is the only file with QSTRs.
SRC_USERMOD_C += $(TFLM_MOD_DIR)/modtflm.c

TFLM_INC := -I$(TFLM_MOD_DIR) -I$(TFLM_DIR) \
            -I$(TFLM_DIR)/third_party/flatbuffers/include \
            -I$(TFLM_DIR)/third_party/gemmlowp \
            -I$(TFLM_DIR)/third_party/ruy

TFLM_DEFS := -DTF_LITE_STATIC_MEMORY=1 -DTF_LITE_MCU_DEBUG_LOG -DNDEBUG

CFLAGS_USERMOD += $(TFLM_INC) $(TFLM_DEFS)
CXXFLAGS_USERMOD += $(TFLM_INC) $(TFLM_DEFS) \
                    -std=c++17 -fno-rtti -fno-exceptions -w -O2
LDFLAGS_USERMOD += -lstdc++
