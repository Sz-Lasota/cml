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

    Tensor *image = malloc(sizeof(Tensor));
  for (uint32_t i = 0; i < 1; i++) {
    printf("Label: %d\n", *(uint8_t *)t_get_element(&labels, (size_t[]){i}));
    t_get_tensor(&images, image, 1, (size_t[]){i});
    print_digit(image);
    printf("\n");
  }

  Dataset *ds = malloc(sizeof(Dataset));
  Tensor *fr = malloc(sizeof(Tensor));
  load_mnist(image_filename, label_filename, ds);
  t_get_tensor(ds->labels, fr, 1, (size_t[]){0});
  t_print(fr);

  MLP *mlp = mlp_new(28 * 28);
  mlp_append_layer(mlp, 32, sigmoid_activation);
  mlp_append_layer(mlp, 16, sigmoid_activation);
  mlp_append_layer(mlp, 10, sigmoid_activation);

  Tensor* test_img = malloc(sizeof(Tensor));
  Tensor* test_ds = t_zeros(DTYPE_FLOAT32, 2, (size_t[]){1, 28*28});

  t_get_tensor(ds->samples, test_img, 1, (size_t[]){0});
  t_set_tensor(test_ds, test_img, 0);

  printf("Initial answer:\n");
  Tensor *init_ans = mlp_forward(mlp, test_ds);
  t_print(init_ans);

  printf("Training start\n");
  mlp_train(mlp, ds, 60, 256, 0.1);
  printf("Training finished\n");

  printf("After answer:\n");
  Tensor *after_ans = mlp_forward(mlp, test_ds);
  t_print(after_ans);


  return 0;
}
