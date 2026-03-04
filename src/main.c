#include "data/mnist.h"
#include "data/utils.h"
#include "math/tensor.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <image_file> <label_file>\n", argv[0]);
    return 1;
  }
  const char *image_filename = argv[1];
  const char *label_filename = argv[2];

  Tensor images;
  load_idx_tensor(image_filename, &images);

  Tensor labels;
  load_idx_tensor(label_filename, &labels);

  for (uint32_t i = 0; i < 10; i++) {
    printf("Label: %d\n", *(uint8_t *)tensor_get(&labels, 1, i));
    print_digit(&images, i);
    printf("\n");
  }
  return 0;
}
