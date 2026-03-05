#include "math/tensor_ops.h"
#include "math/tensor.h"
#include <math.h>

void add_scalar_u8(Tensor *t, void *scalar, size_t n) {
  uint8_t sc = *(uint8_t *)scalar;
  uint8_t *data = (uint8_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

void add_scalar_u32(Tensor *t, void *scalar, size_t n) {
  uint32_t sc = *(uint32_t *)scalar;
  uint32_t *data = (uint32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

void add_scalar_i32(Tensor *t, void *scalar, size_t n) {
  int32_t sc = *(int32_t *)scalar;
  int32_t *data = (int32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

void add_scalar_f32(Tensor *t, void *scalar, size_t n) {
  float_t sc = *(float_t *)scalar;
  float_t *data = (float_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

void t_add_scalar(Tensor *t, void *scalar) {
  size_t n = t_len(t);

  add_scalar_fn dtable[] = {
      add_scalar_u8,
      add_scalar_u32,
      add_scalar_i32,
      add_scalar_f32,
  };
  dtable[t->dtype](t, scalar, n);
}

void mul_scalar_u8(Tensor *t, void *scalar, size_t n) {
  uint8_t sc = *(uint8_t *)scalar;
  uint8_t *data = (uint8_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

void mul_scalar_u32(Tensor *t, void *scalar, size_t n) {
  uint32_t sc = *(uint32_t *)scalar;
  uint32_t *data = (uint32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

void mul_scalar_i32(Tensor *t, void *scalar, size_t n) {
  int32_t sc = *(int32_t *)scalar;
  int32_t *data = (int32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

void mul_scalar_f32(Tensor *t, void *scalar, size_t n) {
  float_t sc = *(float_t *)scalar;
  float_t *data = (float_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}


void t_multiply(Tensor *t, void *scalar) {
  size_t n = t_len(t);

  mul_scalar_fn dtable[] = {
      mul_scalar_u8,
      mul_scalar_u32,
      mul_scalar_i32,
      mul_scalar_f32,
  };
  dtable[t->dtype](t, scalar, n);
}
