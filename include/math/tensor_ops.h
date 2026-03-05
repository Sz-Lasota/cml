#ifndef MATH_TENSOR_OPS_H
#define MATH_TENSOR_OPS_H
#include "math/tensor.h"

typedef void (*add_scalar_fn)(Tensor*, void*, size_t);
void t_add_scalar(Tensor *t, void *scalar);

typedef void (*mul_scalar_fn)(Tensor*, void*, size_t);
void t_multiply(Tensor *t, void *scalar);
#endif // MATH_TENSOR_OPS_H
