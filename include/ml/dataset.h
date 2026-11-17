#ifndef _ML_DATASET_H
#define _ML_DATASET_H

#include "math/tensor.h"
#include <stdint.h>

typedef struct {
  Tensor *samples;
  Tensor *labels;
} Dataset;
void ds_get_batch(Dataset *ds, uint32_t batch_size, Tensor **data,
                  Tensor **label);
#endif // _ML_DATASET_H
