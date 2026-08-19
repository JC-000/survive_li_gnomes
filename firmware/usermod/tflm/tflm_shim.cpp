/* C facade over TFLM. See tflm_shim.h.
 *
 * No dynamic allocation: the handle, the resolver and the interpreter are
 * placement-new'd into the head of the caller's arena, so the whole cost of a
 * loaded model is one buffer the caller decided the size of.
 */
#include "tflm_shim.h"

#include <new>
#include <cstring>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

// Ten builtins. The eleventh slot is spare so adding one op does not silently
// overflow the resolver's fixed table.
constexpr unsigned int kOpCount = 11;
using Resolver = tflite::MicroMutableOpResolver<kOpCount>;

}  // namespace

struct tflm_model {
  Resolver resolver;
  const tflite::Model* model;
  tflite::MicroInterpreter* interp;
  size_t arena_capacity;
  // Storage for the interpreter, placement-new'd. Avoids operator new.
  alignas(8) uint8_t interp_storage[sizeof(tflite::RecordingMicroInterpreter)];
  bool recording;
};

extern "C" {

tflm_model* tflm_new_model(const uint8_t* model_bytes, size_t model_len,
                     uint8_t* arena, size_t arena_len, int* err) {
  (void)model_len;
  if (err) *err = TFLM_OK;

  // Carve the handle off the front of the arena, 8-byte aligned.
  size_t head = (sizeof(tflm_model) + 7u) & ~(size_t)7u;
  if (arena_len <= head) {
    if (err) *err = TFLM_ERR_ALLOC;
    return nullptr;
  }
  tflm_model* m = new (arena) tflm_model();
  m->recording = false;
  m->arena_capacity = arena_len;

  m->model = tflite::GetModel(model_bytes);
  if (m->model->version() != TFLITE_SCHEMA_VERSION) {
    if (err) *err = TFLM_ERR_MODEL;
    return nullptr;
  }

  // The curated set. Anything else in the model fails at AllocateTensors with
  // TFLM_ERR_OP rather than being silently skipped.
  TfLiteStatus s = kTfLiteOk;
  if (m->resolver.AddConv2D() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddDepthwiseConv2D() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddFullyConnected() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddSoftmax() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddAveragePool2D() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddMaxPool2D() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddMean() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddReshape() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddAdd() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddQuantize() != kTfLiteOk) s = kTfLiteError;
  if (m->resolver.AddDequantize() != kTfLiteOk) s = kTfLiteError;
  if (s != kTfLiteOk) {
    if (err) *err = TFLM_ERR_OP;
    return nullptr;
  }

  m->interp = new (m->interp_storage) tflite::MicroInterpreter(
      m->model, m->resolver, arena + head, arena_len - head);

  if (m->interp->AllocateTensors() != kTfLiteOk) {
    if (err) *err = TFLM_ERR_ALLOC;
    return nullptr;
  }
  return m;
}

/* Same, but through RecordingMicroInterpreter, which reports where the arena
 * went. Costs ~1 KB of arena for the recordings themselves. */
tflm_model* tflm_new_model_recording(const uint8_t* model_bytes, size_t model_len,
                               uint8_t* arena, size_t arena_len, int* err) {
  (void)model_len;
  if (err) *err = TFLM_OK;
  size_t head = (sizeof(tflm_model) + 7u) & ~(size_t)7u;
  if (arena_len <= head) {
    if (err) *err = TFLM_ERR_ALLOC;
    return nullptr;
  }
  tflm_model* m = new (arena) tflm_model();
  m->recording = true;
  m->arena_capacity = arena_len;
  m->model = tflite::GetModel(model_bytes);
  if (m->model->version() != TFLITE_SCHEMA_VERSION) {
    if (err) *err = TFLM_ERR_MODEL;
    return nullptr;
  }
  m->resolver.AddConv2D();
  m->resolver.AddDepthwiseConv2D();
  m->resolver.AddFullyConnected();
  m->resolver.AddSoftmax();
  m->resolver.AddAveragePool2D();
  m->resolver.AddMaxPool2D();
  m->resolver.AddMean();
  m->resolver.AddReshape();
  m->resolver.AddAdd();
  m->resolver.AddQuantize();
  m->resolver.AddDequantize();

  auto* r = new (m->interp_storage) tflite::RecordingMicroInterpreter(
      m->model, m->resolver, arena + head, arena_len - head);
  m->interp = r;
  if (r->AllocateTensors() != kTfLiteOk) {
    if (err) *err = TFLM_ERR_ALLOC;
    return nullptr;
  }
  return m;
}

void tflm_print_allocations(tflm_model* m) {
  if (m && m->recording) {
    static_cast<tflite::RecordingMicroInterpreter*>(m->interp)
        ->GetMicroAllocator()
        .PrintAllocations();
  }
}

void tflm_free(tflm_model* m) {
  if (!m) return;
  m->interp->~MicroInterpreter();
  m->~tflm_model();
}

static int elems(const TfLiteTensor* t) {
  int n = 1;
  for (int i = 0; i < t->dims->size; ++i) n *= t->dims->data[i];
  return n;
}

int tflm_input_len(tflm_model* m) { return elems(m->interp->input(0)); }
int tflm_output_len(tflm_model* m) { return elems(m->interp->output(0)); }

int tflm_input_dims(tflm_model* m, int32_t* out, int max) {
  const TfLiteTensor* t = m->interp->input(0);
  int n = t->dims->size < max ? t->dims->size : max;
  for (int i = 0; i < n; ++i) out[i] = t->dims->data[i];
  return t->dims->size;
}

int tflm_output_dims(tflm_model* m, int32_t* out, int max) {
  const TfLiteTensor* t = m->interp->output(0);
  int n = t->dims->size < max ? t->dims->size : max;
  for (int i = 0; i < n; ++i) out[i] = t->dims->data[i];
  return t->dims->size;
}

float tflm_input_scale(tflm_model* m) { return m->interp->input(0)->params.scale; }
int32_t tflm_input_zero_point(tflm_model* m) {
  return m->interp->input(0)->params.zero_point;
}
float tflm_output_scale(tflm_model* m) {
  return m->interp->output(0)->params.scale;
}
int32_t tflm_output_zero_point(tflm_model* m) {
  return m->interp->output(0)->params.zero_point;
}

size_t tflm_arena_capacity(tflm_model* m) { return m->arena_capacity; }

size_t tflm_arena_used(tflm_model* m) {
  // Plus the handle we carved off the front, so the number the caller reads is
  // the whole buffer this needs and not the interpreter's share of it.
  size_t head = (sizeof(tflm_model) + 7u) & ~(size_t)7u;
  return m->interp->arena_used_bytes() + head;
}

int tflm_invoke(tflm_model* m, const int8_t* in, int in_len, int8_t* out_i8,
                float* out_f, int out_len) {
  TfLiteTensor* it = m->interp->input(0);
  TfLiteTensor* ot = m->interp->output(0);
  if (it->type != kTfLiteInt8 || ot->type != kTfLiteInt8) return TFLM_ERR_TYPE;
  if (in_len != elems(it) || out_len != elems(ot)) return TFLM_ERR_SHAPE;

  memcpy(it->data.int8, in, (size_t)in_len);
  if (m->interp->Invoke() != kTfLiteOk) return TFLM_ERR_INVOKE;

  const int8_t* o = ot->data.int8;
  if (out_i8) memcpy(out_i8, o, (size_t)out_len);
  if (out_f) {
    const float scale = ot->params.scale;
    const int32_t zp = ot->params.zero_point;
    for (int i = 0; i < out_len; ++i) out_f[i] = scale * (float)((int32_t)o[i] - zp);
  }
  return TFLM_OK;
}

int tflm_invoke_u8(tflm_model* m, const uint8_t* in, int in_len, int8_t* out_i8,
                   float* out_f, int out_len) {
  TfLiteTensor* it = m->interp->input(0);
  if (it->type != kTfLiteInt8) return TFLM_ERR_TYPE;
  if (in_len != elems(it)) return TFLM_ERR_SHAPE;
  int8_t* dst = it->data.int8;
  for (int i = 0; i < in_len; ++i) dst[i] = (int8_t)((int)in[i] - 128);
  // Re-enter through the int8 path without the memcpy: copy back is harmless
  // but wasteful, so invoke directly here.
  TfLiteTensor* ot = m->interp->output(0);
  if (ot->type != kTfLiteInt8) return TFLM_ERR_TYPE;
  if (out_len != elems(ot)) return TFLM_ERR_SHAPE;
  if (m->interp->Invoke() != kTfLiteOk) return TFLM_ERR_INVOKE;
  const int8_t* o = ot->data.int8;
  if (out_i8) memcpy(out_i8, o, (size_t)out_len);
  if (out_f) {
    const float scale = ot->params.scale;
    const int32_t zp = ot->params.zero_point;
    for (int i = 0; i < out_len; ++i) out_f[i] = scale * (float)((int32_t)o[i] - zp);
  }
  return TFLM_OK;
}

}  // extern "C"
