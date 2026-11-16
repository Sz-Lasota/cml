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

  Tensor *input = t_random(DTYPE_FLOAT32, 2, (size_t[]){1, 4});
  Tensor *gt = t_random(DTYPE_FLOAT32, 2, (size_t[]){1, 2});
  t_print(input);
  MLP *mlp = mlp_new(4);
  mlp_append_layer(mlp, 4, sigmoid_activation);
  mlp_append_layer(mlp, 2, sigmoid_activation);

  printf("Running back prop\n");

 mlp_backward(mlp, input, gt, 0.01);
 t_print(mlp->layers[0]->weights);
}

