#ifndef MATH_TENSOR_OPS_H
#define MATH_TENSOR_OPS_H
#include "math/tensor.h"
#include <stdbool.h>

typedef void (*add_scalar_fn)(Tensor *, void *, size_t);
void t_add_scalar(Tensor *t, void *scalar);

typedef void (*mul_scalar_fn)(Tensor *, void *, size_t);
void t_multiply(Tensor *t, void *scalar);

typedef void (*add_tensor_fn)(const Tensor *, const Tensor *, Tensor *);
Tensor *t_add_tensor(const Tensor *t1, const Tensor *t2);

typedef void (*sub_tensor_fn)(const Tensor *, const Tensor *, Tensor *);
Tensor *t_sub_tensor(const Tensor* t1, const Tensor *t2);

typedef void (*mul_tensor_fn)(const Tensor *, const Tensor *, Tensor *);
Tensor *t_mul_tensor(const Tensor *t1, const Tensor *t2);

typedef void (*matmul_tensor_fn)(const Tensor *, const Tensor *, Tensor *);
Tensor *t_matmul_tensor(const Tensor *t1, const Tensor *t2);

bool _t_is_same_shape(const Tensor *t1, const Tensor *t2);

#endif // MATH_TENSOR_OPS_H
