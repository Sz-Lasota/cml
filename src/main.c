#include "math/tensor.h"
#include "ml/mlp.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Tensor *input = t_random(DTYPE_FLOAT32, 2, (size_t[]){1, 128});
  t_print(input);
  MLP *mlp = mlp_new(128);
  mlp_append_layer(mlp, 256);
  mlp_append_layer(mlp, 256);
  mlp_append_layer(mlp, 128);
  mlp_append_layer(mlp, 64);
  mlp_append_layer(mlp, 32);
  mlp_append_layer(mlp, 2);

  Tensor *output = mlp_forward(mlp, input);
  if (output == NULL) {
    fprintf(stderr, "error: MLP forward pass failed.\n");
    mlp_free(mlp);
    t_free(input);
    return 1;
  }
  t_print(output);
}
