#ifndef MATH_TENSOR_OPS_H
#define MATH_TENSOR_OPS_H
#include "math/tensor.h"

void t_add_scalar(Tensor *t, void *scalar);
void t_multiply(Tensor *t, void *scalar);
#endif // MATH_TENSOR_OPS_H
