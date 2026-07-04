#include "math/tensor.h"
#include "ml/mlp.h"
#include <stdint.h>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Tensor *input = t_random(DTYPE_FLOAT32, 2, (size_t[]){1, 3});
  t_print(input);
  MLP *mlp = mlp_new(3);
  mlp_append_layer(mlp, 3);
  mlp_append_layer(mlp, 2);

  Tensor *output = mlp_forward(mlp, input);
  t_print(output);
}
