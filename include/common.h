#ifndef COMMON_H
#define COMMON_H

#include "math/tensor.h"
#include <stdint.h>

typedef struct {
  Tensor *samples;
  Tensor *labels;
} Dataset;


#endif // COMMON_H
