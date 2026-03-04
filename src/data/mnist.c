#include "data/mnist.h"
#include "data/utils.h"
#include <stdio.h>

void load_mnist(const char *image_filename, const char *label_filename,
                Dataset *ds) {}

void print_digit(const Tensor *image, size_t index) {
  for (uint32_t i = 0; i < MNIST_IMAGE_W; i++) {
    for (uint32_t j = 0; j < MNIST_IMAGE_H; j++) {
      uint8_t pixel_value = *(uint32_t *)tensor_get(image, image->dims, index, i, j);
      char ascii_char;
      choose_ascii_filling(pixel_value, &ascii_char);
      printf("%c", ascii_char);
    }
    printf("\n");
  }
}
