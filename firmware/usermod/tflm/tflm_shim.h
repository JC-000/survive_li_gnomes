/* A C facade over TFLM's C++ interpreter.
 *
 * This is the layer the MicroPython usermod would bind to, and the layer the
 * host bit-exactness harness drives through ctypes. Keeping one file for both
 * is deliberate: the thing proved on the host is then literally the thing that
 * runs on the board, rather than its cousin.
 *
 * Everything here is allocate-once. The caller owns the model bytes and the
 * arena for the whole life of the handle; nothing inside allocates.
 */
#ifndef TFLM_SHIM_H
#define TFLM_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. Negative so a caller can test `< 0` on the ints returned by
 * the invoke path without a separate out-parameter. */
#define TFLM_OK              0
#define TFLM_ERR_MODEL      -1   /* schema version or unparseable buffer   */
#define TFLM_ERR_ALLOC      -2   /* arena too small                        */
#define TFLM_ERR_OP         -3   /* model uses an operator not registered  */
#define TFLM_ERR_INVOKE     -4   /* Invoke() returned an error             */
#define TFLM_ERR_SHAPE      -5   /* caller's buffer is the wrong length    */
#define TFLM_ERR_TYPE       -6   /* tensor is not the int8 this build wants*/

typedef struct tflm_model tflm_model;

/* Curated op resolver: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, SOFTMAX,
 * AVERAGE_POOL_2D, MAX_POOL_2D, MEAN, RESHAPE, ADD, QUANTIZE, DEQUANTIZE.
 * `model` and `arena` must outlive the handle. */
tflm_model *tflm_new_model(const uint8_t *model, size_t model_len,
                     uint8_t *arena, size_t arena_len, int *err);

void tflm_free(tflm_model *m);

/* Shape/type of tensor 0 in and out. Lengths are in elements. */
int tflm_input_len(tflm_model *m);
int tflm_output_len(tflm_model *m);
int tflm_input_dims(tflm_model *m, int32_t *out, int max);
int tflm_output_dims(tflm_model *m, int32_t *out, int max);
float tflm_output_scale(tflm_model *m);
int32_t tflm_output_zero_point(tflm_model *m);
float tflm_input_scale(tflm_model *m);
int32_t tflm_input_zero_point(tflm_model *m);

/* How much of the arena the interpreter actually used, in bytes. Valid after
 * tflm_new(). This is what sizes the arena for the shipping build. */
size_t tflm_arena_used(tflm_model *m);

/* The arena_len this handle was constructed with. The MicroPython binding
 * hands TFLM only the part of the caller's buffer below its copy of the model,
 * and needs this to report the whole buffer's requirement rather than TFLM's
 * share of it. */
size_t tflm_arena_capacity(tflm_model *m);

/* As tflm_new_model(), but through TFLM's RecordingMicroInterpreter, which can
 * report where the arena went. Costs ~1 KB of arena; host tooling only. */
tflm_model *tflm_new_model_recording(const uint8_t *model, size_t model_len,
                                     uint8_t *arena, size_t arena_len, int *err);
void tflm_print_allocations(tflm_model *m);

/* One inference. `in` is int8 already quantised by the caller (this project's
 * front end emits int8 directly). `out_i8` receives the raw output tensor,
 * `out_f` the dequantised scores; either may be NULL. */
int tflm_invoke(tflm_model *m, const int8_t *in, int in_len,
                int8_t *out_i8, float *out_f, int out_len);

/* Same, but taking uint8 = int8 + 128, matching the transport
 * `src/si_patch.py` already uses for the TinyMaix wrapper. */
int tflm_invoke_u8(tflm_model *m, const uint8_t *in, int in_len,
                   int8_t *out_i8, float *out_f, int out_len);

#ifdef __cplusplus
}
#endif

#endif /* TFLM_SHIM_H */
