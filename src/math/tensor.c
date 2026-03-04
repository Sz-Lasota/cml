#include "math/tensor.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void *tensor_get(const Tensor *tensor, uint32_t argc, ...) {
  if (argc != tensor->dims) {
    printf("error: invalid dimensions\n");
    exit(EXIT_FAILURE);
  }

  va_list args;
  va_start(args, argc);

  size_t idx = 0;
  for (uint32_t i = 0; i < argc; i++) {
    size_t traverse = va_arg(args, size_t);
    if (traverse >= tensor->shape[i]) {
      printf("error: index out of bounds for tensor\n");
      exit(EXIT_FAILURE);
    }
    idx += traverse * tensor->stride[i];
  }
  va_end(args);

  switch (tensor->dtype) {
  case DTYPE_UINT8: {
    uint8_t *data = (uint8_t *)tensor->data;
    return &data[idx];
  }
  case DTYPE_FLOAT32: {
    float_t *data = (float_t *)tensor->data;
    return &data[idx];
  }
  default:
    return NULL;
  }
}
