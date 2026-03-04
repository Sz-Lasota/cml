#ifndef MATH_TENSOR_H
#define MATH_TENSOR_H

#include <stdint.h>
#include <stdlib.h>

typedef enum {
  DTYPE_UINT8,
  DTYPE_FLOAT32,
} DType;

typedef struct {
  void *data;
  size_t *shape;
  size_t *stride;
  size_t dims;
  DType dtype;
} Tensor;

void *tensor_get(const Tensor *tensor, uint32_t argc, ...);

#endif // MATH_TENSOR_H
