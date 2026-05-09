#include "unity.h"
#include "math/tensor.h"

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
