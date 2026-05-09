#ifndef MATH_TENSOR_OPS_H
#define MATH_TENSOR_OPS_H
#include "math/tensor.h"

typedef void (*add_scalar_fn)(Tensor*, void*, size_t);
void t_add_scalar(Tensor *t, void *scalar);

typedef void (*mul_scalar_fn)(Tensor*, void*, size_t);
void t_multiply(Tensor *t, void *scalar);

typedef void (*add_tensor_fn)(Tensor*, const Tensor*);
void t_add_tensor(Tensor* t1, const Tensor* t2);

void _t_is_same_shape(const Tensor* t1, const Tensor* t2);

#endif // MATH_TENSOR_OPS_H
