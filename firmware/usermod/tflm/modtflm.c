/*
 * `tflm` -- TensorFlow Lite Micro as a MicroPython user C module.
 *
 * Why this exists is in docs/tflm-usermod.md: TinyMaix approximates TFLite's
 * int8 arithmetic and the approximation is measurable (3 of 8 patches change
 * top-1), so the operating point has to be tuned against the runtime rather
 * than against the model. TFLM computes what TFLite's reference kernels
 * compute -- 30 of 30 output tensors bit-identical, measured -- so with this
 * module the host and the board agree and the divergence stops existing.
 *
 * The shape of the API is `emlearn_cnn_int8`'s, deliberately, so `si_spot.py`
 * swaps one import and one constructor call rather than being rewritten:
 *
 *     import tflm
 *     arena = bytearray(32768)                      # allocate once, up front
 *     model = tflm.new(open("si_model.tflite", "rb").read(), arena)
 *     model.output_dimensions()                     # -> (22,)
 *     model.run(patch, scores)                      # array('B'), array('f')
 *
 * Three deliberate differences from emlearn's wrapper, each one a defect this
 * project has already paid for (see docs/cnn-on-device.md):
 *
 *  1. **The arena is the caller's.** emlearn sizes its scratch by the model
 *     file's length and never checks it against the model's own requirement,
 *     which overran the heap by a measured 1164 bytes. Here the caller passes
 *     a buffer, TFLM's allocator plans into it, and a buffer that is too small
 *     raises at construction. `arena_used()` reports what was actually needed,
 *     so sizing is a measurement rather than a guess.
 *
 *  2. **The input's signedness is in the method name, not sniffed from the
 *     buffer.** emlearn rejects `bytearray` because `mp_get_buffer` does not
 *     report a typecode for it; sniffing typecodes is where that goes wrong.
 *     `run()` takes uint8 (= int8 + 128, the transport `si_patch.py` already
 *     emits) and `run_int8()` takes int8 directly. Any buffer object works for
 *     either.
 *
 *  3. **The model is copied into the arena.** FlatBuffers reads the model
 *     in place with aligned loads, and a `bytes` object off the heap is not
 *     guaranteed to be 16-byte aligned. Copying costs the model's size once
 *     and removes a class of fault that would present as a wrong answer.
 */

#include <string.h>

#include "py/runtime.h"
#include "py/objtuple.h"
#include "py/binary.h"

#include "tflm_shim.h"

typedef struct _tflm_model_obj_t {
    mp_obj_base_t base;
    tflm_model *handle;
    // Held so the GC cannot collect the arena while the interpreter points
    // into it. Every pointer TFLM owns lives inside this one buffer.
    mp_obj_t arena_obj;
    int n_in;
    int n_out;
} tflm_model_obj_t;

static const mp_obj_type_t tflm_type_Model;

// Each message is its own MP_ERROR_TEXT literal so the build's string
// compression sees it; a `const char *` selected at run time would not be
// compressible and would land in RAM.
static NORETURN void tflm_raise(int err) {
    switch (err) {
        case TFLM_ERR_MODEL:
            mp_raise_ValueError(MP_ERROR_TEXT("not a TFLite model (schema version 3 expected)"));
        case TFLM_ERR_ALLOC:
            mp_raise_ValueError(MP_ERROR_TEXT("arena too small"));
        case TFLM_ERR_OP:
            mp_raise_ValueError(MP_ERROR_TEXT("model uses an unregistered operator"));
        case TFLM_ERR_INVOKE:
            mp_raise_ValueError(MP_ERROR_TEXT("invoke failed"));
        case TFLM_ERR_SHAPE:
            mp_raise_ValueError(MP_ERROR_TEXT("buffer length does not match the tensor"));
        case TFLM_ERR_TYPE:
            mp_raise_ValueError(MP_ERROR_TEXT("model is not int8 in and int8 out"));
        default:
            mp_raise_ValueError(MP_ERROR_TEXT("tflm error"));
    }
}

/* tflm.new(model_bytes, arena) -> Model
 *
 * `model_bytes` may be any read-only buffer; it is copied into `arena` and
 * need not outlive this call. `arena` must be a writable buffer -- a
 * `bytearray` -- and must outlive the model, which is why it is held.
 */
static mp_obj_t tflm_new(mp_obj_t model_in, mp_obj_t arena_in) {
    mp_buffer_info_t model_buf, arena_buf;
    mp_get_buffer_raise(model_in, &model_buf, MP_BUFFER_READ);
    mp_get_buffer_raise(arena_in, &arena_buf, MP_BUFFER_WRITE);

    // Model at the top of the arena, 16-byte aligned down from the end, so the
    // interpreter's planner gets one contiguous region below it.
    size_t model_len = model_buf.len;
    size_t aligned = (model_len + 15u) & ~(size_t)15u;
    if (arena_buf.len <= aligned) {
        tflm_raise(TFLM_ERR_ALLOC);
    }
    uint8_t *base = (uint8_t *)arena_buf.buf;
    uint8_t *model_at = base + arena_buf.len - aligned;
    model_at = (uint8_t *)((uintptr_t)model_at & ~(uintptr_t)15u);
    memcpy(model_at, model_buf.buf, model_len);

    int err = TFLM_OK;
    tflm_model *h = tflm_new_model(model_at, model_len, base,
                                   (size_t)(model_at - base), &err);
    if (h == NULL) {
        tflm_raise(err);
    }

    tflm_model_obj_t *self = mp_obj_malloc(tflm_model_obj_t, &tflm_type_Model);
    self->handle = h;
    self->arena_obj = arena_in;
    self->n_in = tflm_input_len(h);
    self->n_out = tflm_output_len(h);
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_2(tflm_new_obj, tflm_new);

static mp_obj_t dims_tuple(int (*get)(tflm_model *, int32_t *, int),
                           tflm_model *h) {
    int32_t dims[6];
    int n = get(h, dims, 6);
    if (n > 6) {
        n = 6;
    }
    // Leading batch dimension of 1 is dropped: the caller thinks in
    // (frames, bands, channels) and (classes,), not in batches of one.
    int start = (n > 1 && dims[0] == 1) ? 1 : 0;
    mp_obj_t items[6];
    for (int i = start; i < n; ++i) {
        items[i - start] = MP_OBJ_NEW_SMALL_INT(dims[i]);
    }
    return mp_obj_new_tuple(n - start, items);
}

static mp_obj_t model_output_dimensions(mp_obj_t self_in) {
    tflm_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return dims_tuple(tflm_output_dims, self->handle);
}
static MP_DEFINE_CONST_FUN_OBJ_1(model_output_dimensions_obj, model_output_dimensions);

static mp_obj_t model_input_dimensions(mp_obj_t self_in) {
    tflm_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return dims_tuple(tflm_input_dims, self->handle);
}
static MP_DEFINE_CONST_FUN_OBJ_1(model_input_dimensions_obj, model_input_dimensions);

/* Bytes of the arena the planner actually used, including the model copy and
 * this module's own bookkeeping. The number to size the next `bytearray` by. */
static mp_obj_t model_arena_used(mp_obj_t self_in) {
    tflm_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t arena;
    mp_get_buffer_raise(self->arena_obj, &arena, MP_BUFFER_READ);
    size_t model_part = arena.len - tflm_arena_capacity(self->handle);
    return mp_obj_new_int_from_uint(tflm_arena_used(self->handle) + model_part);
}
static MP_DEFINE_CONST_FUN_OBJ_1(model_arena_used_obj, model_arena_used);

static mp_obj_t run_common(mp_obj_t self_in, mp_obj_t in_obj, mp_obj_t out_obj,
                           bool unsigned_input) {
    tflm_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t in, out;
    mp_get_buffer_raise(in_obj, &in, MP_BUFFER_READ);
    mp_get_buffer_raise(out_obj, &out, MP_BUFFER_WRITE);

    if ((int)in.len != self->n_in) {
        tflm_raise(TFLM_ERR_SHAPE);
    }
    // The output buffer is float32 per class -- array('f'), as emlearn's is.
    if ((int)out.len != self->n_out * (int)sizeof(float)) {
        tflm_raise(TFLM_ERR_SHAPE);
    }

    int rc;
    if (unsigned_input) {
        rc = tflm_invoke_u8(self->handle, (const uint8_t *)in.buf, self->n_in,
                            NULL, (float *)out.buf, self->n_out);
    } else {
        rc = tflm_invoke(self->handle, (const int8_t *)in.buf, self->n_in,
                         NULL, (float *)out.buf, self->n_out);
    }
    if (rc != TFLM_OK) {
        tflm_raise(rc);
    }
    return mp_const_none;
}

static mp_obj_t model_run(mp_obj_t self_in, mp_obj_t in_obj, mp_obj_t out_obj) {
    return run_common(self_in, in_obj, out_obj, true);
}
static MP_DEFINE_CONST_FUN_OBJ_3(model_run_obj, model_run);

static mp_obj_t model_run_int8(mp_obj_t self_in, mp_obj_t in_obj, mp_obj_t out_obj) {
    return run_common(self_in, in_obj, out_obj, false);
}
static MP_DEFINE_CONST_FUN_OBJ_3(model_run_int8_obj, model_run_int8);

static const mp_rom_map_elem_t model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_run), MP_ROM_PTR(&model_run_obj) },
    { MP_ROM_QSTR(MP_QSTR_run_int8), MP_ROM_PTR(&model_run_int8_obj) },
    { MP_ROM_QSTR(MP_QSTR_output_dimensions), MP_ROM_PTR(&model_output_dimensions_obj) },
    { MP_ROM_QSTR(MP_QSTR_input_dimensions), MP_ROM_PTR(&model_input_dimensions_obj) },
    { MP_ROM_QSTR(MP_QSTR_arena_used), MP_ROM_PTR(&model_arena_used_obj) },
};
static MP_DEFINE_CONST_DICT(model_locals_dict, model_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    tflm_type_Model,
    MP_QSTR_Model,
    MP_TYPE_FLAG_NONE,
    locals_dict, &model_locals_dict
    );

#if defined(TFLM_BUILTIN_MODEL)
/* A model linked into the firmware image rather than read off the filesystem.
 *
 * Worth having because it is the difference between the model costing 30 KB of
 * heap and costing none: flash is read as memory here, so a `const` array is
 * already where FlatBuffers wants it -- aligned, immovable, and free. The
 * arena then holds activations only.
 *
 * Off by default, and it should stay off until the model stops changing:
 * a frozen model means reflashing the board to retrain, where a file means
 * copying one. Define TFLM_BUILTIN_MODEL and link a translation unit providing
 * these two symbols to turn it on.
 */
extern const uint8_t tflm_builtin_model[];
extern const unsigned int tflm_builtin_model_len;

static mp_obj_t tflm_builtin(mp_obj_t arena_in) {
    mp_buffer_info_t arena;
    mp_get_buffer_raise(arena_in, &arena, MP_BUFFER_WRITE);
    int err = TFLM_OK;
    tflm_model *h = tflm_new_model(tflm_builtin_model, tflm_builtin_model_len,
                                   (uint8_t *)arena.buf, arena.len, &err);
    if (h == NULL) {
        tflm_raise(err);
    }
    tflm_model_obj_t *self = mp_obj_malloc(tflm_model_obj_t, &tflm_type_Model);
    self->handle = h;
    self->arena_obj = arena_in;
    self->n_in = tflm_input_len(h);
    self->n_out = tflm_output_len(h);
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_1(tflm_builtin_obj, tflm_builtin);
#endif

static const mp_rom_map_elem_t tflm_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_tflm) },
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&tflm_new_obj) },
    { MP_ROM_QSTR(MP_QSTR_Model), MP_ROM_PTR(&tflm_type_Model) },
    #if defined(TFLM_BUILTIN_MODEL)
    { MP_ROM_QSTR(MP_QSTR_builtin), MP_ROM_PTR(&tflm_builtin_obj) },
    #endif
};
static MP_DEFINE_CONST_DICT(tflm_module_globals, tflm_module_globals_table);

const mp_obj_module_t tflm_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&tflm_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_tflm, tflm_user_cmodule);
