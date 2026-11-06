#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "ml/loss.h"
#include "ml/mlp.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Tensor *pred = t_zeros(DTYPE_FLOAT32, 2, (size_t[]){1, 4});
  t_print(pred);

  Tensor *gt = t_zeros(DTYPE_FLOAT32, 2, (size_t[]){1, 4});
  float el = 0.5;
  t_set_element(gt, &el, (size_t[]){0, 1});
  t_print(gt);

  float mse = mse_loss(pred, gt);
  printf("MSE Loss: %f\n", mse);
}
