#include "data/mnist.h"
#include "data/utils.h"
#include "math/tensor.h"
#include <stdint.h>
#include <stdio.h>

static Tensor *convert_img_to_float(Tensor *image) {
  Tensor *fimg =
      t_zeros(DTYPE_FLOAT32, 1, (size_t[]){image->shape[0] * image->shape[1]});
  float *fdata = (float *)fimg->data;
  uint8_t *idata = (uint8_t *)image->data;
  for (size_t i = 0; i < fimg->shape[0]; i++) {
    fdata[i] = (float)idata[i] / UINT8_MAX;
  }
  return fimg;
}

void load_mnist(const char *image_filename, const char *label_filename,
                Dataset *ds) {
  Tensor *images = malloc(sizeof(Tensor));
  load_idx_tensor(image_filename, images);
  // Transform tensor to float, flatten images
  Tensor *data = t_zeros(
      DTYPE_FLOAT32, 2,
      (size_t[]){images->shape[0], images->shape[1] * images->shape[2]});

  for (size_t img_idx = 0; img_idx < images->shape[0]; img_idx++) {
      Tensor *iimg = malloc(sizeof(Tensor));
      t_get_tensor(images, iimg, 1, (size_t[]){img_idx});
      Tensor *fimg = convert_img_to_float(iimg);

      t_set_tensor(data, fimg, img_idx);

      t_free(iimg);
      t_free(fimg);
  }

  t_free(images);

  Tensor *labels = malloc(sizeof(Tensor));
  load_idx_tensor(label_filename, labels);
  Tensor *encoded_labels = t_zeros(DTYPE_FLOAT32, 2, (size_t[]){labels->shape[0], 10});
  float one = 1.0;
  for (size_t lidx = 0; lidx < labels->shape[0]; lidx++) {
      size_t one_idx = (size_t)*(uint8_t*)t_get_element(labels, (size_t[]){lidx});
      t_set_element(encoded_labels, &one, (size_t[]){lidx, one_idx});
  }

  t_free(labels);
  ds->samples = data;
  ds->labels = encoded_labels;
}

void print_digit(const Tensor *image) {
  for (uint32_t i = 0; i < MNIST_IMAGE_W; i++) {
    for (uint32_t j = 0; j < MNIST_IMAGE_H; j++) {
      size_t indices[] = {i, j};
      uint8_t pixel_value = *(uint32_t *)t_get_element(image, indices);
      char ascii_char;
      choose_ascii_filling(pixel_value, &ascii_char);
      printf("%c", ascii_char);
    }
    printf("\n");
  }
}
