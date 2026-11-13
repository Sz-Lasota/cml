#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "unity.h"

void test_new_int32(void) {
  size_t shape[] = {2, 3};
  Tensor *tensor = t_new(DTYPE_INT32, 2, shape);

  TEST_ASSERT_NOT_NULL(tensor);
  TEST_ASSERT_EQUAL(DTYPE_INT32, tensor->dtype);
  TEST_ASSERT_EQUAL(2, tensor->dims);
  TEST_ASSERT_EQUAL(2, tensor->shape[0]);
  TEST_ASSERT_EQUAL(3, tensor->shape[1]);
  t_free(tensor);
}

void test_new_uint32(void) {
  size_t shape[] = {2, 3};
  Tensor *tensor = t_new(DTYPE_UINT32, 2, shape);

  TEST_ASSERT_NOT_NULL(tensor);
  TEST_ASSERT_EQUAL(DTYPE_UINT32, tensor->dtype);
  TEST_ASSERT_EQUAL(2, tensor->dims);
  TEST_ASSERT_EQUAL(2, tensor->shape[0]);
  TEST_ASSERT_EQUAL(3, tensor->shape[1]);
  t_free(tensor);
}

void test_new_float32(void) {
  size_t shape[] = {2, 3};
  Tensor *tensor = t_new(DTYPE_FLOAT32, 2, shape);

  TEST_ASSERT_NOT_NULL(tensor);
  TEST_ASSERT_EQUAL(DTYPE_FLOAT32, tensor->dtype);
  TEST_ASSERT_EQUAL(2, tensor->dims);
  TEST_ASSERT_EQUAL(2, tensor->shape[0]);
  TEST_ASSERT_EQUAL(3, tensor->shape[1]);
  t_free(tensor);
}

void test_add_float32(void) {
  size_t shape[] = {2, 3};

  Tensor *t1 = t_new(DTYPE_FLOAT32, 2, shape);
  Tensor *t2 = t_new(DTYPE_FLOAT32, 2, shape);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      size_t idx[] = {i, j};
      float t1_value = (float)i;
      float t2_value = (float)j;

      t_set_element(t1, &t1_value, idx);
      t_set_element(t2, &t2_value, idx);
    }
  }
  Tensor *result = t_add_tensor(t1, t2);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      size_t idx[] = {i, j};
      float expected_value = (float)(i + j);
      float actual_value = *(float *)t_get_element(result, idx);
      TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_value, actual_value);
    }
  }

  t_free(t1);
  t_free(t2);
  t_free(result);
}

void test_mul_float32(void) {
  size_t shape[] = {2, 3};

  Tensor *t1 = t_new(DTYPE_FLOAT32, 2, shape);
  Tensor *t2 = t_new(DTYPE_FLOAT32, 2, shape);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      size_t idx[] = {i, j};
      float t1_value = (float)i;
      float t2_value = (float)j;

      t_set_element(t1, &t1_value, idx);
      t_set_element(t2, &t2_value, idx);
    }
  }
  Tensor *result = t_mul_tensor(t1, t2);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      size_t idx[] = {i, j};
      float expected_value = (float)(i * j);
      float actual_value = *(float *)t_get_element(result, idx);
      TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_value, actual_value);
    }
  }

  t_free(t1);
  t_free(t2);
  t_free(result);
}
