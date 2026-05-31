#include "unity.h"
#include "unity_internals.h"

void setUp(void) {}
void tearDown(void) {}

void test_new_int32(void);
void test_new_uint32(void);
void test_new_float32(void);
void test_add_float32(void);
void test_mul_float32(void);

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_new_int32);
  RUN_TEST(test_new_uint32);
  RUN_TEST(test_new_float32);

  RUN_TEST(test_add_float32);

  RUN_TEST(test_mul_float32);

  return UNITY_END();
}
