#ifndef ML_LOSS_H
#define ML_LOSS_H

#include "math/tensor.h"
typedef void *(loss_fn)(Tensor* pred, Tensor* gt);


float mse_loss(Tensor* pred, Tensor* gt);


#endif  // ML_LOSS_H
