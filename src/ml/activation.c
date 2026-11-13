#include "ml/activation.h"
#include "math/tensor.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static float sigmoid(float x) { return 1 / (1 + expf(-x)); }

void sigmoid_activation(Tensor *output) {
  for (size_t n = 0; n < output->shape[1]; n++) {
    size_t idx[] = {0, n};
    float cv = *(float *)t_get_element(output, idx);
    float scv = sigmoid(cv);
    t_set_element(output, &scv, idx);
  }
}
