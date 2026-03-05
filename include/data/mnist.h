#ifndef DATA_MNIST_H
#define DATA_MNIST_H

#include "common.h"
#define MNIST_IMAGE_W 28
#define MNIST_IMAGE_H 28

void load_mnist(const char *image_filename, const char *label_filename,
                Dataset *ds);

void print_digit(const Tensor *image);

#endif // DATA_MNIST_H
