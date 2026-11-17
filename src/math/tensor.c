#include "math/tensor.h"
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

#define _TPRINT_INDENT 2

uint32_t dtype_size(DType type) {
  switch (type) {
  case DTYPE_UINT8:
    return 1;
  case DTYPE_FLOAT32:
    return 4;
  case DTYPE_UINT32:
    return 4;
  case DTYPE_INT32:
    return 4;
  }
  return 0;
}

static inline void random_bytes(void *buf, size_t len) {
  getrandom(buf, len, 0);
}

static inline float random_float() {
  int32_t rand_int;
  random_bytes(&rand_int, sizeof(rand_int));
  return (float)rand_int / (float)UINT32_MAX;
}

static void t_calc_stride(Tensor *t) {
  t->stride = malloc(t->dims * sizeof(size_t));
  t->stride[t->dims - 1] = 1;
  for (int32_t i = t->dims - 2; i >= 0; i--) {
    t->stride[i] = t->shape[i + 1] * t->stride[i + 1];
  }
}

static size_t t_data_size(const Tensor *t) {
  size_t mem_size = dtype_size(t->dtype);
  for (size_t i = 0; i < t->dims; i++) {

    mem_size *= t->shape[i];
  }
  return mem_size;
}

Tensor *t_new(DType type, size_t dims, size_t *shape) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = type;
  t->dims = dims;

  t->shape = malloc(dims * sizeof(size_t));
  for (size_t i = 0; i < dims; i++) {
    t->shape[i] = shape[i];
  }

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  t_calc_stride(t);
  return t;
}

static void arange_fill_u8(void *tdata, size_t range) {
  uint8_t *data = (uint8_t *)tdata;
  for (size_t i = 0; i < range; i++) {
    *data = i;
    data++;
  }
}
static void arange_fill_u32(void *tdata, size_t range) {
  uint32_t *data = (uint32_t *)tdata;
  for (size_t i = 0; i < range; i++) {
    *data = i;
    data++;
  }
}

static void arange_fill_i32(void *tdata, size_t range) {
  int32_t *data = (int32_t *)tdata;
  for (size_t i = 0; i < range; i++) {
    *data = i;
    data++;
  }
}

static void arange_fill_f32(void *tdata, size_t range) {
  float *data = (float *)tdata;
  for (size_t i = 0; i < range; i++) {
    *data = i;
    data++;
  }
}

static void fill_random_u8(Tensor *tensor) {
  uint8_t *data = (uint8_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    uint8_t random_value;
    random_bytes(&random_value, sizeof(random_value));
    data[i] = random_value;
  }
}

static void fill_random_u32(Tensor *tensor) {
  uint32_t *data = (uint32_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    uint32_t random_value;
    random_bytes(&random_value, sizeof(random_value));
    data[i] = random_value;
  }
}

static void fill_random_i32(Tensor *tensor) {
  int32_t *data = (int32_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    int32_t random_value;
    random_bytes(&random_value, sizeof(random_value));
    data[i] = random_value;
  }
}

static void fill_random_f32(Tensor *tensor) {
  float *data = (float *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    data[i] = random_float();
  }
}

static void fill_ones_u8(Tensor *tensor) {
  uint8_t *data = (uint8_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    data[i] = 1;
  }
}

static void fill_ones_u32(Tensor *tensor) {
  uint32_t *data = (uint32_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    data[i] = 1;
  }
}

static void fill_ones_i32(Tensor *tensor) {
  int32_t *data = (int32_t *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    data[i] = 1;
  }
}

static void fill_ones_f32(Tensor *tensor) {
  float *data = (float *)tensor->data;
  size_t len = t_len(tensor);
  for (size_t i = 0; i < len; i++) {
    data[i] = 1.0;
  }
}

Tensor *t_arange(size_t range, DType type) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dims = 1;
  t->dtype = type;
  t->shape = malloc(sizeof(size_t));
  t->shape[0] = range;

  t_calc_stride(t);

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);

  arange_fill_fn dtable[] = {
      arange_fill_u8,
      arange_fill_u32,
      arange_fill_i32,
      arange_fill_f32,
  };
  dtable[t->dtype](t->data, range);
  return t;
}

Tensor *t_random(DType type, size_t dims, size_t *shape) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = type;
  t->dims = dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->shape, shape, t->dims * sizeof(size_t));
  t_calc_stride(t);
  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);

  fill_tensor_with_random_data dtable[] = {
      fill_random_u8,
      fill_random_u32,
      fill_random_i32,
      fill_random_f32,
  };
  dtable[t->dtype](t);

  return t;
}

Tensor *t_zeros(DType type, size_t dims, size_t *shape) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = type;
  t->dims = dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->shape, shape, t->dims * sizeof(size_t));
  t_calc_stride(t);

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  memset(t->data, 0, mem_size);

  return t;
}

Tensor *t_ones(DType type, size_t dims, size_t *shape) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = type;
  t->dims = dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->shape, shape, t->dims * sizeof(size_t));
  t_calc_stride(t);

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  fill_tensor_with_ones dtable[] = {
      fill_ones_u8,
      fill_ones_u32,
      fill_ones_i32,
      fill_ones_f32,
  };
  dtable[t->dtype](t);

  return t;
}

Tensor *t_random_like(const Tensor *tensor) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = tensor->dtype;
  t->dims = tensor->dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->stride, tensor->stride, t->dims * sizeof(size_t));
  memcpy(t->shape, tensor->shape, t->dims * sizeof(size_t));

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);

  fill_tensor_with_random_data dtable[] = {
      fill_random_u8,
      fill_random_u32,
      fill_random_i32,
      fill_random_f32,
  };
  dtable[t->dtype](t);

  return t;
}

Tensor *t_zeros_like(const Tensor *tensor) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = tensor->dtype;
  t->dims = tensor->dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->stride, tensor->stride, t->dims * sizeof(size_t));
  memcpy(t->shape, tensor->shape, t->dims * sizeof(size_t));

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  memset(t->data, 0, mem_size);

  return t;
}

Tensor *t_ones_like(const Tensor *tensor) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = tensor->dtype;
  t->dims = tensor->dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->stride, tensor->stride, t->dims * sizeof(size_t));
  memcpy(t->shape, tensor->shape, t->dims * sizeof(size_t));

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  fill_tensor_with_ones dtable[] = {
      fill_ones_u8,
      fill_ones_u32,
      fill_ones_i32,
      fill_ones_f32,
  };
  dtable[t->dtype](t);

  return t;
}

Tensor *t_copy(const Tensor *tensor) {
  Tensor *t = malloc(sizeof(Tensor));
  t->dtype = tensor->dtype;
  t->dims = tensor->dims;
  t->shape = malloc(t->dims * sizeof(size_t));
  t->stride = malloc(t->dims * sizeof(size_t));

  memcpy(t->stride, tensor->stride, t->dims * sizeof(size_t));
  memcpy(t->shape, tensor->shape, t->dims * sizeof(size_t));

  size_t mem_size = t_data_size(t);
  t->data = malloc(mem_size);
  memcpy(t->data, tensor->data, mem_size);
  return t;
}

static void *get_element_u8(void *tdata, size_t idx) {
  uint8_t *data = (uint8_t *)tdata;
  return &data[idx];
}

static void *get_element_f32(void *tdata, size_t idx) {
  float *data = (float *)tdata;
  return &data[idx];
}

static void *get_element_u32(void *tdata, size_t idx) {
  uint32_t *data = (uint32_t *)tdata;
  return &data[idx];
}

static void *get_element_i32(void *tdata, size_t idx) {
  int32_t *data = (int32_t *)tdata;
  return &data[idx];
}

void *t_get_element(const Tensor *tensor, size_t *indices) {
  size_t idx = 0;
  for (uint32_t i = 0; i < tensor->dims; i++) {
    size_t traverse = indices[i];
    if (traverse >= tensor->shape[i]) {
      printf("error: index out of bounds for get element\n");
      exit(EXIT_FAILURE);
    }
    idx += traverse * tensor->stride[i];
  }

  get_element_fn dtable[] = {
      get_element_u8,
      get_element_u32,
      get_element_i32,
      get_element_f32,
  };
  return dtable[tensor->dtype](tensor->data, idx);
}

void t_get_tensor(const Tensor *tensor_in, Tensor *tensor_out,
                  size_t indices_len, size_t *indices) {
  if (indices_len > tensor_in->dims) {
    printf("error: invalid dimensions\n");
    exit(EXIT_FAILURE);
  }

  size_t offset = 0;
  for (size_t i = 0; i < indices_len; i++) {
    size_t traverse = indices[i];
    if (traverse >= tensor_in->shape[i]) {
      printf("error: index out of bounds for get tensor\n");
      exit(EXIT_FAILURE);
    }
    offset += traverse * tensor_in->stride[i];
  }

  tensor_out->dtype = tensor_in->dtype;
  tensor_out->dims =
      tensor_in->dims == indices_len ? 1 : tensor_in->dims - indices_len;
  tensor_out->shape = malloc(tensor_out->dims * sizeof(size_t));
  for (size_t i = 0; i < tensor_out->dims; i++) {
    // Only when argc == tensor_in->dims
    if (i + indices_len == tensor_in->dims) {
      tensor_out->shape[i] = 1;
      break;
    }
    tensor_out->shape[i] = tensor_in->shape[i + indices_len];
  }

  t_calc_stride(tensor_out);

  size_t mem_size = t_data_size(tensor_out);

  tensor_out->data = malloc(mem_size);
  memcpy(tensor_out->data,
         tensor_in->data + dtype_size(tensor_out->dtype) * offset, mem_size);
}

static void set_element_u8(void *tdata, size_t idx, void *element) {
  uint8_t *data = (uint8_t *)tdata;
  data[idx] = *(uint8_t *)element;
}

static void set_element_f32(void *tdata, size_t idx, void *element) {
  float *data = (float *)tdata;
  data[idx] = *(float *)element;
}

static void set_element_u32(void *tdata, size_t idx, void *element) {
  uint32_t *data = (uint32_t *)tdata;
  data[idx] = *(uint32_t *)element;
}

static void set_element_i32(void *tdata, size_t idx, void *element) {
  int32_t *data = (int32_t *)tdata;
  data[idx] = *(int32_t *)element;
}

void t_set_element(Tensor *tensor, void *element, size_t *indices) {
  size_t idx = 0;
  for (uint32_t i = 0; i < tensor->dims; i++) {
    size_t traverse = indices[i];
    if (traverse >= tensor->shape[i]) {
      printf("error: index out of bounds for set element\n");
      exit(EXIT_FAILURE);
    }
    idx += traverse * tensor->stride[i];
  }

  set_element_fn dtable[] = {
      set_element_u8,
      set_element_u32,
      set_element_i32,
      set_element_f32,
  };
  dtable[tensor->dtype](tensor->data, idx, element);
}

void t_set_tensor(Tensor *tensor_dst, Tensor *tensor_src, size_t index) {
  if (tensor_dst->dtype != tensor_src->dtype) {
    fprintf(stderr, "error: source tensor have to have the same dtype\n");
    return;
  }

  if (tensor_dst->dims != tensor_src->dims + 1) {
    fprintf(stderr, "error: source tensor have to has one less dimension than "
                    "destination\n");
    return;
  }

  for (size_t i = 0; i < tensor_src->dims; i++) {
    if (tensor_src->shape[i] != tensor_dst->shape[i + 1]) {
      fprintf(stderr,
              "error: shape mismatch, tensor_src->shape[%zu] != "
              "tensor_dst->shape[%zu]",
              i, i + 1);
      return;
    }
  }

  size_t data_size = t_data_size(tensor_src);
  size_t byte_offset = index * data_size;
  memcpy(tensor_dst->data + byte_offset, tensor_src->data, data_size);
}

void t_reshape(Tensor *tensor, size_t new_shape_len, int32_t *shape) {
  size_t *new_shape = malloc(new_shape_len * sizeof(size_t));
  int64_t wildcard_idx = -1;
  size_t new_capacity = 1;
  size_t old_capacity = t_data_size(tensor) / dtype_size(tensor->dtype);

  for (size_t i = 0; i < new_shape_len; i++) {
    int32_t shp = shape[i];
    new_shape[i] = shp;
    if (shp < 0 && wildcard_idx >= 0) {
      printf("error: only one -1 is allowed when reshaping\n");
      exit(EXIT_FAILURE);
    }
    if (shp < 0) {
      wildcard_idx = i;
      continue;
    }
    new_capacity *= shp;
  }

  if (wildcard_idx >= 0) {
    if (old_capacity % new_capacity != 0) {
      printf("error: could not determine wildcard value!\n");
      exit(EXIT_FAILURE);
    }
    new_shape[wildcard_idx] = old_capacity / new_capacity;
    new_capacity *= new_shape[wildcard_idx];
  }

  if (new_capacity != old_capacity) {
    fprintf(stderr,
            "error: cannot reshape tensor! new cap: %ld, old cap: %ld\n",
            new_capacity, old_capacity);
    exit(EXIT_FAILURE);
  }

  tensor->dims = new_shape_len;

  free(tensor->shape);
  tensor->shape = new_shape;

  free(tensor->stride);
  t_calc_stride(tensor);
}

void t_flatten(Tensor *tensor) {
  int32_t shape[] = {-1};
  t_reshape(tensor, 1, shape);
}

void t_free(Tensor *tensor) {
  if (tensor == NULL) {
    return;
  }
  free(tensor->shape);
  free(tensor->stride);
  free(tensor->data);
  free(tensor);
}

static void print_element_u8(const Tensor *tensor, size_t *indices) {
  printf(" %3d", *(uint8_t *)t_get_element(tensor, indices));
}

static void print_element_u32(const Tensor *tensor, size_t *indices) {
  printf(" %3d", *(uint32_t *)t_get_element(tensor, indices));
}

static void print_element_i32(const Tensor *tensor, size_t *indices) {
  printf(" %3d", *(int32_t *)t_get_element(tensor, indices));
}

static void print_element_f32(const Tensor *tensor, size_t *indices) {
  printf(" %3.6f", *(float *)t_get_element(tensor, indices));
}

static void t_print_row(const Tensor *tensor, uint32_t indent) {
  if (tensor->dims != 1) {
    printf("error: print row prints only vectors!");
    exit(EXIT_FAILURE);
  }
  if (indent > 0) {
    printf("%*s[", indent, " ");
  } else {

    printf("[");
  }
  for (size_t i = 0; i < tensor->shape[0]; i++) {
    size_t indices[] = {i};
    print_element_fn dtable[] = {
        print_element_u8,
        print_element_u32,
        print_element_i32,
        print_element_f32,

    };
    dtable[tensor->dtype](tensor, indices);
  }
  printf("]");
}

static void t_print_indent(const Tensor *tensor, uint32_t indent) {
  if (tensor->dims == 1) {
    t_print_row(tensor, indent + _TPRINT_INDENT);
    printf("\n");
    return;
  }

  printf("%*s[\n", indent, " ");
  for (size_t i = 0; i < tensor->shape[0]; i++) {
    Tensor child;
    size_t indices[] = {i};

    t_get_tensor(tensor, &child, 1, indices);
    t_print_indent(&child, indent + _TPRINT_INDENT);
  }
  printf("%*s]", indent, " ");
  printf("\n");
}

void t_print(const Tensor *tensor) {
  if (tensor->dims == 1) {
    t_print_row(tensor, 0);
    printf("\n");
    return;
  }

  printf("[\n");
  for (size_t i = 0; i < tensor->shape[0]; i++) {
    Tensor child;
    size_t indices[] = {i};

    t_get_tensor(tensor, &child, 1, indices);
    t_print_indent(&child, _TPRINT_INDENT);
  }
  printf("]");
  printf("\n");
}

size_t t_len(const Tensor *tensor) {
  return t_data_size(tensor) / dtype_size(tensor->dtype);
}

static void sum_tensor_u8(const Tensor *tensor, Tensor *result, size_t n) {
  uint8_t *data = (uint8_t *)tensor->data;
  uint8_t *acc = (uint8_t *)result->data;
  for (size_t i = 0; i < n; i++) {
    acc[0] += data[i];
  }
}

static void sum_tensor_u32(const Tensor *tensor, Tensor *result, size_t n) {
  uint32_t *data = (uint32_t *)tensor->data;
  uint32_t *acc = (uint32_t *)result->data;
  for (size_t i = 0; i < n; i++) {
    acc[0] += data[i];
  }
}

static void sum_tensor_i32(const Tensor *tensor, Tensor *result, size_t n) {
  int32_t *data = (int32_t *)tensor->data;
  int32_t *acc = (int32_t *)result->data;
  for (size_t i = 0; i < n; i++) {
    acc[0] += data[i];
  }
}

static void sum_tensor_f32(const Tensor *tensor, Tensor *result, size_t n) {
  float *data = (float *)tensor->data;
  float *acc = (float *)result->data;
  for (size_t i = 0; i < n; i++) {
    acc[0] += data[i];
  }
}

Tensor *t_sum(const Tensor *tensor) {
  size_t n = t_len(tensor);
  Tensor *result = t_new(tensor->dtype, 1, (size_t[]){1});

  tensor_sum_fn dtable[] = {
      sum_tensor_u8,
      sum_tensor_u32,
      sum_tensor_i32,
      sum_tensor_f32,
  };
  dtable[tensor->dtype](tensor, result, n);
  return result;
}
