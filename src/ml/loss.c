#include "ml/loss.h"
#include "math/tensor.h"
#include "math/tensor_ops.h"
#include <stdio.h>

float mse_loss(Tensor *pred, Tensor *gt) {
  Tensor *diff = t_sub_tensor(pred, gt);
  Tensor *sq = t_mul_tensor(diff, diff);
  Tensor *sq_sum = t_sum(sq);
  t_free(diff);
  t_free(sq);

  void *nominator = t_get_element(sq_sum, (size_t[]){0});
  float denominator =
      2; // Dividing allows to drop constant when calculating gradient

  switch (pred->dtype) {
  case DTYPE_FLOAT32: {
    float fnom = *(float *)nominator;
    return fnom / denominator;
  }
  case DTYPE_UINT8: {
    uint8_t unom = *(uint8_t *)nominator;
    return unom / denominator;
  }
  case DTYPE_INT32: {
    int32_t unom = *(int32_t *)nominator;
    return unom / denominator;
  }
  case DTYPE_UINT32:

  {
    uint32_t unom = *(uint32_t *)nominator;
    return unom / denominator;
  }
  }
  return 0;
}
