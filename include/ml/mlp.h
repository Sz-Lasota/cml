#ifndef ML_MLP_H
#define ML_MLP_H

#include "math/tensor.h"
#include "ml/dataset.h"
#include "ml/activation.h"
#include <stdint.h>

typedef struct {
  uint32_t input_size;
  uint32_t output_size;
  Tensor *weights;
  Tensor *biases;
  activation_fn activation;
} Layer;

typedef struct {
  uint32_t layers_count;
  Layer **layers;
  uint32_t input_size;
} MLP;

Layer *layer_new(uint32_t input_size, uint32_t output_size,
                 activation_fn act_fn);
void layer_free(Layer *layer);
Tensor *layer_forward(Layer *layer, Tensor *input);

MLP *mlp_new(uint32_t input_size);
void mlp_free(MLP *mlp);

void mlp_append_layer(MLP *mlp, uint32_t neurons_count, activation_fn act_fn);
void mlp_insert_layer(MLP *mlp, uint32_t index, uint32_t neurons_count,
                      activation_fn act_fn);
void mlp_remove_layer(MLP *mlp, uint32_t index);
void mlp_pop_layer(MLP *mlp);

Tensor *mlp_forward(MLP *mlp, Tensor *input);
void mlp_backward(MLP *mlp, Tensor *input, Tensor *gt, float lr);

void mlp_train(MLP *mlp, Dataset *ds, uint32_t epochs, uint32_t batch_size, float lr);
#endif // ML_MLP_H
