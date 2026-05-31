#include "math/tensor.h"
#include "math/tensor_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  Tensor *t1 = t_arange(6, DTYPE_FLOAT32);
  t_reshape(t1, 2, (int32_t[]){2, 3});
    printf("Tensor 1 dims: %zu\n", t1->dims);
  Tensor *t2 = t_arange(12, DTYPE_FLOAT32);
  t_reshape(t2, 2, (int32_t[]){3, 4});

    printf("Tensor 1 dims: %zu\n", t2->dims);
  t_print(t1);
  t_print(t2);
  printf("Result of matmul\n");
  Tensor *result = t_matmul_tensor(t1, t2);
  if (result == NULL) {
    fprintf(stderr, "Error: matmul failed\n");
    return 1;
  }
  t_print(result);
  return 0;
}
