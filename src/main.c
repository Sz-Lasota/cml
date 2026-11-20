#include "data/mnist.h"
#include "data/utils.h"
#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "ml/activation.h"
#include "ml/dataset.h"
#include "ml/loss.h"
#include "ml/mlp.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEST_PICTURES 5

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 5) {
    fprintf(stderr, "Usage: %s <image_file> <label_file>\n", argv[0]);
    return 1;
  }

  const char *image_filename = argv[1];
  const char *label_filename = argv[2];

  const char *image_test_set_path = NULL;
  const char *label_test_set_path = NULL;

  if (argc == 5) {
    image_test_set_path = argv[3];
    label_test_set_path = argv[4];
  }

  srand(time(NULL));

  Tensor images;
  load_idx_tensor(image_filename, &images);

  Tensor labels;
  load_idx_tensor(label_filename, &labels);

  Tensor *image = malloc(sizeof(Tensor));
  for (uint32_t i = 0; i < TEST_PICTURES; i++) {
    printf("Label: %d\n", *(uint8_t *)t_get_element(&labels, (size_t[]){i}));
    t_get_tensor(&images, image, 1, (size_t[]){i});
    print_digit(image);
    printf("\n");
  }

  // Initialize training set
  Dataset *ds = malloc(sizeof(Dataset));
  load_mnist(image_filename, label_filename, ds);
  ds_init_pool(ds);

  // Initialize validation set
  Dataset *vs = NULL;
  if (image_test_set_path != NULL && label_test_set_path != NULL) {
    vs = malloc(sizeof(Dataset));
    load_mnist(image_test_set_path, label_test_set_path, vs);
    ds_init_pool(vs);
  }

  // Create MLP network
  MLP *mlp = mlp_new(28 * 28);
  mlp_append_layer(mlp, 32, sigmoid_activation);
  mlp_append_layer(mlp, 16, sigmoid_activation);
  mlp_append_layer(mlp, 10, sigmoid_activation);

  if (vs != NULL) {
      float accuracy = mlp_accuracy(mlp, vs);
      printf("Accuracy before training on test set: %3.6f\n", accuracy);
  }

  printf("Training start\n");
  mlp_train(mlp, ds, 5000, 256, 1.0, vs);
  printf("Training finished\n");

  if (vs != NULL) {
      float accuracy = mlp_accuracy(mlp, vs);
      printf("Accuracy after training on test set: %3.6f\n", accuracy);
  }
  return 0;
}
