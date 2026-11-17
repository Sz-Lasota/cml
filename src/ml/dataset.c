#include "ml/dataset.h"
#include "math/tensor.h"
#include <stdint.h>
#include <sys/random.h>

static inline size_t random_index(size_t min, size_t max) {
  return min + (size_t)rand() % (max - min + 1);
}

void ds_get_batch(Dataset *ds, uint32_t batch_size, Tensor **data,
                  Tensor **label) {
  *data = t_zeros(
      ds->samples->dtype, 2,
      (size_t[]){batch_size, ds->samples->shape[ds->samples->dims - 1]});
  *label =
      t_zeros(ds->labels->dtype, 2,
              (size_t[]){batch_size, ds->labels->shape[ds->labels->dims - 1]});

  size_t n = ds->samples->shape[0];
  size_t *pool = malloc(sizeof(size_t) * n);
  for (size_t i = 0; i < n; i++)
    pool[i] = i;

  for (uint32_t i = 0; i < batch_size; i++) {
    size_t j = random_index(i, n - 1); // pick from the remaining part
    size_t tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }

  for (uint32_t i = 0; i < batch_size; i++) {

    Tensor *example = malloc(sizeof(Tensor));
    t_get_tensor(ds->samples, example, 1, (size_t[]){pool[i]});
    t_set_tensor(*data, example, i);
    t_free(example);

    Tensor *answer = malloc(sizeof(Tensor));
    t_get_tensor(ds->labels, answer, 1, (size_t[]){pool[i]});
    t_set_tensor(*label, answer, i);
    t_free(answer);
  }
}
