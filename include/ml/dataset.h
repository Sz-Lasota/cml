#ifndef _ML_DATASET_H
#define _ML_DATASET_H

#include "math/tensor.h"
#include <stdlib.h>
#include <stdint.h>

typedef struct {
  Tensor *samples;
  Tensor *labels;
  size_t *pool;
} Dataset;


void ds_init_pool(Dataset *ds);
void ds_get_batch(Dataset *ds, uint32_t batch_size, Tensor **data,
                  Tensor **label);
#endif // _ML_DATASET_H
