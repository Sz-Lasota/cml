#include "math/tensor.h"
#include "math/tensor_ops.h"
#include <stdint.h>

int main(int argc, char *argv[]) {
  int32_t shape[] = {15, 2};
  Tensor *t = t_arange(30, DTYPE_INT32);
  t_reshape(t, 2, shape);
  int32_t sc = -320;
  t_multiply(t, &sc);
  t_print(t);
  return 0;
}
