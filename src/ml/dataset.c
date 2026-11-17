#include "ml/dataset.h"
#include <stdint.h>

static inline size_t random_index(size_t min, size_t max) {
  return min + (size_t)rand() % (max - min + 1);
}

void ds_init_pool(Dataset *ds) {
  size_t n = ds->samples->shape[0];
  size_t *pool = malloc(sizeof(size_t) * n);
  for (size_t i = 0; i < n; i++)
    pool[i] = i;

  ds->pool = pool;
}

void ds_get_batch(Dataset *ds, uint32_t batch_size, Tensor **data,
                  Tensor **label) {
  size_t n = ds->samples->shape[0];
  if (batch_size > n)
    batch_size = n;

  *data = t_zeros(
      ds->samples->dtype, 2,
      (size_t[]){batch_size, ds->samples->shape[ds->samples->dims - 1]});
  *label =
      t_zeros(ds->labels->dtype, 2,
              (size_t[]){batch_size, ds->labels->shape[ds->labels->dims - 1]});

  for (uint32_t i = 0; i < batch_size; i++) {
    size_t j = random_index(i, n - 1); // pick from the remaining part
    size_t tmp = ds->pool[i];
    ds->pool[i] = ds->pool[j];
    ds->pool[j] = tmp;
  }

  for (uint32_t i = 0; i < batch_size; i++) {

    Tensor *example = malloc(sizeof(Tensor));
    t_get_tensor(ds->samples, example, 1, (size_t[]){ds->pool[i]});
    t_set_tensor(*data, example, i);
    t_free(example);

    Tensor *answer = malloc(sizeof(Tensor));
    t_get_tensor(ds->labels, answer, 1, (size_t[]){ds->pool[i]});
    t_set_tensor(*label, answer, i);
    t_free(answer);
  }
}
