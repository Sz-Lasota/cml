#include "data/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint32_t read_uint32_msb(const uint8_t *buffer) {
  return ((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16) |
         ((uint32_t)buffer[2] << 8) | (uint32_t)buffer[3];
}

int32_t read_int32_msb(const uint8_t *buffer) {
  return (int32_t)(((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16) |
                   ((uint32_t)buffer[2] << 8) | (uint32_t)buffer[3]);
}

void choose_ascii_filling(uint8_t value, char *output) {
  if (value < 10) {
    *output = '.';
  } else if (value < 50) {
    *output = '\'';
  } else if (value < 100) {
    *output = ';';
  } else if (value < 200) {
    *output = 'o';
  } else {
    *output = '@';
  }
}

void load_idx_tensor(const char *filename, Tensor *tensor) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "error: could not open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  size_t file_size;
  fseek(file, 0, SEEK_END);
  file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  uint8_t *buffer = (uint8_t *)malloc(file_size);
  if (!buffer) {
    fprintf(stderr, "error: could not allocate memory for file buffer\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fread(buffer, 1, file_size, file);
  fclose(file);

  uint8_t dtype = buffer[2];
  if (dtype != 0x08) {
    fprintf(
        stderr,
        "error: unsupported data type in IDX file (expected 0x08 for uint8)\n");
    free(buffer);
    exit(EXIT_FAILURE);
  }

  uint8_t dim_num = buffer[3];
  if (dim_num < 1) {
    fprintf(stderr,
            "error: invalid number of dimensions in IDX file (must be >= 1)\n");
    free(buffer);
    exit(EXIT_FAILURE);
  }

  tensor->dims = dim_num;
  tensor->dtype = DTYPE_UINT8;
  tensor->shape = malloc(dim_num * sizeof(size_t));
  for (uint8_t i = 0; i < dim_num; i++) {
    tensor->shape[i] = read_int32_msb(&buffer[4 + i * 4]);
  }

  tensor->stride = malloc(dim_num * sizeof(size_t));
  tensor->stride[dim_num - 1] = 1;
  for (int32_t i = dim_num - 2; i >= 0; i--) {
    tensor->stride[i] = tensor->shape[i + 1] * tensor->stride[i + 1];
  }

  size_t data_size = file_size - 4 - dim_num * 4;
  tensor->data = malloc(data_size);
  if (!tensor->data) {
    fprintf(stderr, "error: could not allocate memory for tensor data\n");
    free(buffer);
    free(tensor->shape);
    exit(EXIT_FAILURE);
  }
  memcpy(tensor->data, &buffer[4 + dim_num * 4], data_size);
  free(buffer);
}
