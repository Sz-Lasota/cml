#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "ml/activation.h"
#include "ml/loss.h"
#include "ml/mlp.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Tensor *input = t_random(DTYPE_UINT8, 2, (size_t[]){3, 4});
  Tensor *gt = t_random(DTYPE_UINT8, 2, (size_t[]){1, 4});
  t_print(input);
  t_print(gt);

  Tensor *result = t_add_tensor_with_broadcast(input, gt);
  t_print(result);

}

