#include "data/mnist.h"
#include "data/utils.h"
#include "math/tensor.h"
#include <stdio.h>

void load_mnist(const char *image_filename, const char *label_filename,
                Dataset *ds) {
  Tensor *images = malloc(sizeof(Tensor));
  load_idx_tensor(image_filename, images);
  Tensor *labels = malloc(sizeof(Tensor));
  load_idx_tensor(label_filename, labels);
  ds->samples = images;
  ds->labels = labels;
}

void print_digit(const Tensor *image) {
  for (uint32_t i = 0; i < MNIST_IMAGE_W; i++) {
    for (uint32_t j = 0; j < MNIST_IMAGE_H; j++) {
            size_t indices[] = {i, j};
      uint8_t pixel_value =
          *(uint32_t *)t_get_element(image, indices);
      char ascii_char;
      choose_ascii_filling(pixel_value, &ascii_char);
      printf("%c", ascii_char);
    }
    printf("\n");
  }
}
