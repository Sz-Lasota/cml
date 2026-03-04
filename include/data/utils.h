#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include "math/tensor.h"
#include <stdint.h>

uint32_t read_uint32_msb(const uint8_t *buffer);
int32_t read_int32_msb(const uint8_t *buffer);

void load_idx_tensor(const char *filename, Tensor *tensor);
void choose_ascii_filling(uint8_t value, char *output);


#endif // DATA_UTILS_H
