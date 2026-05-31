#include "math/tensor_ops.h"
#include "math/tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void add_scalar_u8(Tensor *t, void *scalar, size_t n) {
  uint8_t sc = *(uint8_t *)scalar;
  uint8_t *data = (uint8_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

static void add_scalar_u32(Tensor *t, void *scalar, size_t n) {
  uint32_t sc = *(uint32_t *)scalar;
  uint32_t *data = (uint32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

static void add_scalar_i32(Tensor *t, void *scalar, size_t n) {
  int32_t sc = *(int32_t *)scalar;
  int32_t *data = (int32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] += sc;
  }
}

static void add_scalar_f32(Tensor *t, void *scalar, size_t n) {
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

static void mul_scalar_u8(Tensor *t, void *scalar, size_t n) {
  uint8_t sc = *(uint8_t *)scalar;
  uint8_t *data = (uint8_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

static void mul_scalar_u32(Tensor *t, void *scalar, size_t n) {
  uint32_t sc = *(uint32_t *)scalar;
  uint32_t *data = (uint32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

static void mul_scalar_i32(Tensor *t, void *scalar, size_t n) {
  int32_t sc = *(int32_t *)scalar;
  int32_t *data = (int32_t *)t->data;
  for (size_t i = 0; i < n; i++) {
    data[i] *= sc;
  }
}

static void mul_scalar_f32(Tensor *t, void *scalar, size_t n) {
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

bool _t_is_same_shape(const Tensor *t1, const Tensor *t2) {
  if (t1->dtype != t2->dtype) {
    return false;
  }

  if (t_len(t1) != t_len(t2)) {
    return false;
  }

  if (t1->dims != t2->dims) {
    return false;
  }

  for (size_t i = 0; i < t1->dims; i++) {
    if (t1->shape[i] != t2->shape[i]) {
      return false;
    }
  }
  return true;
}

static void add_tensor_u8(const Tensor *t1, const Tensor *t2, Tensor *result) {
  uint8_t *data1 = (uint8_t *)t1->data;
  uint8_t *data2 = (uint8_t *)t2->data;
  uint8_t *result_data = (uint8_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] + data2[i];
  }
}

static void add_tensor_u32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  uint32_t *data1 = (uint32_t *)t1->data;
  uint32_t *data2 = (uint32_t *)t2->data;
  uint32_t *result_data = (uint32_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] + data2[i];
  }
}

static void add_tensor_i32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  int32_t *data1 = (int32_t *)t1->data;
  int32_t *data2 = (int32_t *)t2->data;
  int32_t *result_data = (int32_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] + data2[i];
  }
}

static void add_tensor_f32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  float_t *data1 = (float_t *)t1->data;
  float_t *data2 = (float_t *)t2->data;
  float_t *result_data = (float_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] + data2[i];
  }
}

Tensor *t_add_tensor(const Tensor *t1, const Tensor *t2) {

  if (!_t_is_same_shape(t1, t2)) {
    fprintf(
        stderr,
        "error: Tensors must have the same shape and dtype for addition.\n");
    return NULL;
  }

  Tensor *result = t_zeros_like(t1);

  add_tensor_fn dtable[] = {
      add_tensor_u8,
      add_tensor_u32,
      add_tensor_i32,
      add_tensor_f32,
  };
  dtable[t1->dtype](t1, t2, result);
  return result;
}

static void mul_tensor_u8(const Tensor *t1, const Tensor *t2, Tensor *result) {
  uint8_t *data1 = (uint8_t *)t1->data;
  uint8_t *data2 = (uint8_t *)t2->data;
  uint8_t *result_data = (uint8_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] * data2[i];
  }
}

static void mul_tensor_u32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  uint32_t *data1 = (uint32_t *)t1->data;
  uint32_t *data2 = (uint32_t *)t2->data;
  uint32_t *result_data = (uint32_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] * data2[i];
  }
}

static void mul_tensor_i32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  int32_t *data1 = (int32_t *)t1->data;
  int32_t *data2 = (int32_t *)t2->data;
  int32_t *result_data = (int32_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] * data2[i];
  }
}
static void mul_tensor_f32(const Tensor *t1, const Tensor *t2, Tensor *result) {
  float_t *data1 = (float_t *)t1->data;
  float_t *data2 = (float_t *)t2->data;
  float_t *result_data = (float_t *)result->data;
  size_t n = t_len(t1);
  for (size_t i = 0; i < n; i++) {
    result_data[i] = data1[i] * data2[i];
  }
}

Tensor *t_mul_tensor(const Tensor *t1, const Tensor *t2) {
  if (!_t_is_same_shape(t1, t2)) {
    fprintf(stderr, "error: Tensors must have the same shape and dtype for "
                    "multiplication.\n");
    return NULL;
  }

  Tensor *result = t_zeros_like(t1);

  mul_tensor_fn dtable[] = {
      mul_tensor_u8,
      mul_tensor_u32,
      mul_tensor_i32,
      mul_tensor_f32,
  };
  dtable[t1->dtype](t1, t2, result);
  return result;
}

static void matmul_tensor_u8(const Tensor *t1, const Tensor *t2,
                             Tensor *result) {
  size_t n = t1->shape[1];

  for (size_t i = 0; i < result->shape[0]; i++) {
    for (size_t j = 0; j < result->shape[1]; j++) {

      uint8_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (*(uint8_t *)t_get_element(t1, (size_t[]){i, k})) *
               (*(uint8_t *)t_get_element(t2, (size_t[]){k, j}));
      }
      t_set_element(result, &sum, (size_t[]){i, j});
    }
  }
}

static void matmul_tensor_u32(const Tensor *t1, const Tensor *t2,
                              Tensor *result) {
  size_t n = t1->shape[1];

  for (size_t i = 0; i < result->shape[0]; i++) {
    for (size_t j = 0; j < result->shape[1]; j++) {

      uint32_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (*(uint32_t *)t_get_element(t1, (size_t[]){i, k})) *
               (*(uint32_t *)t_get_element(t2, (size_t[]){k, j}));
      }
      t_set_element(result, &sum, (size_t[]){i, j});
    }
  }
}

static void matmul_tensor_i32(const Tensor *t1, const Tensor *t2,
                              Tensor *result) {
  size_t n = t1->shape[1];

  for (size_t i = 0; i < result->shape[0]; i++) {
    for (size_t j = 0; j < result->shape[1]; j++) {

      int32_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (*(int32_t *)t_get_element(t1, (size_t[]){i, k})) *
               (*(int32_t *)t_get_element(t2, (size_t[]){k, j}));
      }
      t_set_element(result, &sum, (size_t[]){i, j});
    }
  }
}
static void matmul_tensor_f32(const Tensor *t1, const Tensor *t2,
                              Tensor *result) {
  size_t n = t1->shape[1];

  for (size_t i = 0; i < result->shape[0]; i++) {
    for (size_t j = 0; j < result->shape[1]; j++) {

      float_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += (*(float_t *)t_get_element(t1, (size_t[]){i, k})) *
               (*(float_t *)t_get_element(t2, (size_t[]){k, j}));
      }
      t_set_element(result, &sum, (size_t[]){i, j});
    }
  }
}
Tensor *t_matmul_tensor(const Tensor *t1, const Tensor *t2) {
  if (t1->dtype != t2->dtype) {
    fprintf(stderr, "error: Tensors have to have the same dtype\n");
    return NULL;
  }

  if (t1->dims != 2 || t2->dims != 2) {
    fprintf(stderr, "error: Both tensors have to be two-dimensional tensors\n");
    return NULL;
  }

  if (t1->shape[1] != t2->shape[0]) {
    fprintf(stderr,
            "error: The number of columns in the first tensor must match the "
            "number of rows in the second tensor for multiplication.\n");
    return NULL;
  }

  size_t new_shape[2] = {t1->shape[0], t2->shape[1]};
  Tensor *result = t_new(t1->dtype, 2, new_shape);

  matmul_tensor_fn dtable[] = {
      matmul_tensor_u8,
      matmul_tensor_u32,
      matmul_tensor_i32,
      matmul_tensor_f32,
  };

  dtable[t1->dtype](t1, t2, result);
  return result;
}
