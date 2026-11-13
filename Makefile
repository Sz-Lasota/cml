CC      := gcc
CFLAGS  := -Wall -Wextra -Iinclude -Ivendors/unity/src 
LDFLAGS := -lm

SRC_DIR := src
TEST_DIR := tests
UNITY_DIR := vendors/unity/src

OBJ_DIR := build/obj
TEST_OBJ_DIR := build/test_obj
BIN_DIR := build/bin

TARGET  := $(BIN_DIR)/cml
TEST_TARGET := $(BIN_DIR)/test_runner

SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

TEST_SRCS := $(shell find $(TEST_DIR) -name '*.c')
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c,$(TEST_OBJ_DIR)/%.o,$(TEST_SRCS))

UNITY_SRC := $(UNITY_DIR)/unity.c
UNITY_OBJ := build/unity.o

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(UNITY_OBJ): $(UNITY_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS) $(UNITY_OBJ) $(filter-out $(OBJ_DIR)/main.o,$(OBJS))
	@mkdir -p $(BIN_DIR)
	$(CC) $^ -o $@

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all test clean
