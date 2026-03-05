#include "math/tensor_ops.h"
#include "math/tensor.h"
#include <math.h>

void t_add_scalar(Tensor *t, void *scalar) {
  size_t n = t_len(t);

  switch (t->dtype) {
  case DTYPE_UINT8: {
    uint8_t sc = *(uint8_t *)scalar;
    uint8_t *data = (uint8_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] += sc;
    }
    break;
  }
  case DTYPE_UINT32: {
    uint32_t sc = *(uint32_t *)scalar;
    uint32_t *data = (uint32_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] += sc;
    }
    break;
  }
  case DTYPE_INT32: {
    int32_t sc = *(int32_t *)scalar;
    int32_t *data = (int32_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] += sc;
    }
    break;
  }
  case DTYPE_FLOAT32: {
    float_t sc = *(float_t *)scalar;
    float_t *data = (float_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] += sc;
    }
    break;
  }
  }
}
void t_multiply(Tensor *t, void *scalar) {
  size_t n = t_len(t);

  switch (t->dtype) {
  case DTYPE_UINT8: {
    uint8_t sc = *(uint8_t *)scalar;
    uint8_t *data = (uint8_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] *= sc;
    }
    break;
  }
  case DTYPE_UINT32: {
    uint32_t sc = *(uint32_t *)scalar;
    uint32_t *data = (uint32_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] *= sc;
    }
    break;
  }
  case DTYPE_INT32: {
    int32_t sc = *(int32_t *)scalar;
    int32_t *data = (int32_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] *= sc;
    }
    break;
  }
  case DTYPE_FLOAT32: {
    float_t sc = *(float_t *)scalar;
    float_t *data = (float_t *)t->data;
    for (size_t i = 0; i < n; i++) {
      data[i] *= sc;
    }
    break;
  }
  }
}
