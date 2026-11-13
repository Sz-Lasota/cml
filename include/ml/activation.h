#ifndef ML_ACTIVATION_H
#define ML_ACTIVATION_H

#include "math/tensor.h"
typedef void (*activation_fn)(Tensor* output);
void sigmoid_activation(Tensor* output);

#endif // ML_ACTIVATION_H
