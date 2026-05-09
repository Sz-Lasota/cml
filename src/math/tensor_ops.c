#include "math/tensor_ops.h"
#include "math/tensor.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
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

void _t_is_same_shape(const Tensor* t1, const Tensor* t2) {
    if (t1->dtype != t2->dtype) {
        fprintf(stderr, "error: cannot operate on tensors of different data types\n");
        exit(EXIT_FAILURE);
    }

    if (t_len(t1) != t_len(t2)) {
        fprintf(stderr, "error: cannot operate on tensors of different sizes\n");
        exit(EXIT_FAILURE);
    }

    if (t1->dims != t2->dims) {
        fprintf(stderr, "error: cannot operate on tensors of different dimensions\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < t1->dims; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "error: cannot operate on tensors of different shapes\n");
            exit(EXIT_FAILURE);
        }
    }
}

static void add_tensor_u8(Tensor* t1, const Tensor* t2) {
    uint8_t *data1 = (uint8_t *)t1->data;
    uint8_t *data2 = (uint8_t *)t2->data;
    size_t n = t_len(t1);
    for (size_t i = 0; i < n; i++) {
        data1[i] += data2[i];
    }
}

static void add_tensor_u32(Tensor* t1, const Tensor* t2) {
    uint32_t *data1 = (uint32_t *)t1->data;
    uint32_t *data2 = (uint32_t *)t2->data;
    size_t n = t_len(t1);
    for (size_t i = 0; i < n; i++) {
        data1[i] += data2[i];
    }
}

static void add_tensor_i32(Tensor* t1, const Tensor* t2) {
    int32_t *data1 = (int32_t *)t1->data;
    int32_t *data2 = (int32_t *)t2->data;
    size_t n = t_len(t1);
    for (size_t i = 0; i < n; i++) {
        data1[i] += data2[i];
    }
}

static void add_tensor_f32(Tensor* t1, const Tensor* t2) {
    float_t *data1 = (float_t *)t1->data;
    float_t *data2 = (float_t *)t2->data;
    size_t n = t_len(t1);
    for (size_t i = 0; i < n; i++) {
        data1[i] += data2[i];
    }
}

void t_add_tensor(Tensor* t1, const Tensor* t2) {
    _t_is_same_shape(t1, t2);

    add_tensor_fn dtable[] = {
        add_tensor_u8,
        add_tensor_u32,
        add_tensor_i32,
        add_tensor_f32,
    };
    dtable[t1->dtype](t1, t2);

}
