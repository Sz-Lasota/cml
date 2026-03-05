#ifndef MATH_TENSOR_H
#define MATH_TENSOR_H

#include <stdint.h>
#include <stdlib.h>

typedef enum {
  DTYPE_UINT8,
  DTYPE_UINT32,
  DTYPE_INT32,
  DTYPE_FLOAT32,
} DType;

typedef struct {
  void *data;
  size_t *shape;
  size_t *stride;
  size_t dims;
  DType dtype;
} Tensor;

uint32_t dtype_size(DType type);

Tensor *t_new(DType type, size_t dims, size_t *shape);
Tensor *t_arange(size_t range, DType type);
void t_free(Tensor *tensor);

void *t_get_element(const Tensor *tensor, size_t *indices);
void t_get_tensor(const Tensor *tensor_in, Tensor *tensor_out,
                  size_t indices_len, size_t *indices);

void t_set_element(Tensor *tensor, void *element, size_t *indices);
void t_reshape(Tensor *tensor, size_t new_shape_len, int32_t *shape);
void t_flatten(Tensor *tensor);

void t_print(const Tensor *tensor);
size_t t_len(const Tensor *tensor);

#endif // MATH_TENSOR_H
