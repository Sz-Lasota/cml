#include "ml/mlp.h"
#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "ml/activation.h"
#include <stdint.h>
#include <stdio.h>

MLP *mlp_new(uint32_t input_size) {
  MLP *mlp = (MLP *)(malloc(sizeof(MLP)));
  mlp->layers_count = 0;
  mlp->layers = NULL;
  mlp->input_size = input_size;
  return mlp;
}

void mlp_free(MLP *mlp) {

  if (mlp == NULL) {
    return;
  }

  for (uint32_t i = 0; i < mlp->layers_count; i++) {
    Layer *layer = mlp->layers[i];
    if (layer != NULL) {
      t_free(layer->weights);
      t_free(layer->biases);
      free(layer);
    }
  }
  free(mlp->layers);
  free(mlp);
}

Layer *layer_new(uint32_t input_size, uint32_t output_size,
                 activation_fn act_fn) {
  Layer *layer = (Layer *)(malloc(sizeof(Layer)));
  layer->input_size = input_size;
  layer->output_size = output_size;
  if (act_fn == NULL) {
    fprintf(stderr, "error: Activation function cannot be null.\n");
  }
  layer->activation = act_fn;

  size_t weights_shape[2] = {input_size, output_size};
  layer->weights = t_random(DTYPE_FLOAT32, 2, weights_shape);

  size_t biases_shape[2] = {1, output_size};
  layer->biases = t_random(DTYPE_FLOAT32, 2, biases_shape);

  return layer;
}

void layer_free(Layer *layer) {
  if (layer == NULL) {
    return;
  }
  t_free(layer->weights);
  t_free(layer->biases);
  free(layer);
}

Tensor *layer_forward(Layer *layer, Tensor *input) {
  if (input->dims != 2 || input->shape[1] != layer->input_size) {
    fprintf(stderr,
            "error: Input tensor shape does not match layer input size.\n");
    return NULL;
  }

  Tensor *output = t_matmul_tensor(input, layer->weights);
  if (output == NULL) {
    fprintf(stderr, "error: Matrix multiplication failed.\n");
    return NULL;
  }

  Tensor *output_with_bias = t_add_tensor(output, layer->biases);

  t_free(output);
  if (output_with_bias == NULL) {
    fprintf(stderr, "error: Adding bias failed.\n");
    return NULL;
  }
  layer->activation(output_with_bias);

  return output_with_bias;
}

void mlp_append_layer(MLP *mlp, uint32_t neurons_count, activation_fn act_fn) {
  Layer *current_tail =
      mlp->layers_count > 0 ? mlp->layers[mlp->layers_count - 1] : NULL;

  Layer *layer = layer_new(current_tail == NULL ? mlp->input_size
                                                : current_tail->output_size,
                           neurons_count, act_fn);

  mlp->layers_count++;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));
  mlp->layers[mlp->layers_count - 1] = layer;
}

void mlp_insert_layer(MLP *mlp, uint32_t index, uint32_t neurons_count,
                      activation_fn act_fn) {
  Layer *current_tail =
      mlp->layers_count > 0 ? mlp->layers[mlp->layers_count - 1] : NULL;

  Layer *layer = layer_new(current_tail == NULL ? mlp->input_size
                                                : current_tail->output_size,
                           neurons_count, act_fn);

  mlp->layers_count++;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));

  for (uint32_t i = mlp->layers_count - 1; i > index; i--) {
    mlp->layers[i] = mlp->layers[i - 1];
  }
  mlp->layers[index] = layer;
}

void mlp_remove_layer(MLP *mlp, uint32_t index) {
  if (index >= mlp->layers_count) {
    return;
  }

  Layer *layer = mlp->layers[index];
  layer_free(layer);

  for (uint32_t i = index; i < mlp->layers_count - 1; i++) {
    mlp->layers[i] = mlp->layers[i + 1];
  }

  mlp->layers_count--;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));
}

void mlp_pop_layer(MLP *mlp) {
  if (mlp->layers_count == 0) {
    return;
  }

  Layer *layer = mlp->layers[mlp->layers_count - 1];
  layer_free(layer);

  mlp->layers_count--;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));
}

Tensor *mlp_forward(MLP *mlp, Tensor *input) {
  Tensor *current_output = input;

  for (uint32_t i = 0; i < mlp->layers_count; i++) {
    Layer *layer = mlp->layers[i];
    Tensor *next_output = layer_forward(layer, current_output);

    if (next_output == NULL) {
      fprintf(stderr, "error: Forward pass failed at layer %u.\n", i);
      return NULL;
    }

    if (i > 0) {
      t_free(current_output);
    }
    current_output = next_output;
  }

  return current_output;
}
