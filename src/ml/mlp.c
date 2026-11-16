#include "ml/mlp.h"
#include "math/tensor.h"
#include "math/tensor_ops.h"
#include "ml/activation.h"
#include <stdint.h>
#include <stdio.h>

MLP *mlp_new(uint32_t input_size)
{
  MLP *mlp = (MLP *)(malloc(sizeof(MLP)));
  mlp->layers_count = 0;
  mlp->layers = NULL;
  mlp->input_size = input_size;
  return mlp;
}

void mlp_free(MLP *mlp)
{

  if (mlp == NULL)
  {
    return;
  }

  for (uint32_t i = 0; i < mlp->layers_count; i++)
  {
    Layer *layer = mlp->layers[i];
    if (layer != NULL)
    {
      t_free(layer->weights);
      t_free(layer->biases);
      free(layer);
    }
  }
  free(mlp->layers);
  free(mlp);
}

Layer *layer_new(uint32_t input_size, uint32_t output_size,
                 activation_fn act_fn)
{
  Layer *layer = (Layer *)(malloc(sizeof(Layer)));
  layer->input_size = input_size;
  layer->output_size = output_size;
  if (act_fn == NULL)
  {
    fprintf(stderr, "error: Activation function cannot be null.\n");
  }
  layer->activation = act_fn;

  size_t weights_shape[2] = {input_size, output_size};
  layer->weights = t_random(DTYPE_FLOAT32, 2, weights_shape);

  size_t biases_shape[2] = {1, output_size};
  layer->biases = t_random(DTYPE_FLOAT32, 2, biases_shape);

  return layer;
}

void layer_free(Layer *layer)
{
  if (layer == NULL)
  {
    return;
  }
  t_free(layer->weights);
  t_free(layer->biases);
  free(layer);
}

Tensor *layer_forward(Layer *layer, Tensor *input)
{
  if (input->dims != 2 || input->shape[1] != layer->input_size)
  {
    fprintf(stderr,
            "error: Input tensor shape does not match layer input size.\n");
    return NULL;
  }

  Tensor *output = t_matmul_tensor(input, layer->weights);
  if (output == NULL)
  {
    fprintf(stderr, "error: Matrix multiplication failed.\n");
    return NULL;
  }

  Tensor *output_with_bias = t_add_tensor(output, layer->biases);

  t_free(output);
  if (output_with_bias == NULL)
  {
    fprintf(stderr, "error: Adding bias failed.\n");
    return NULL;
  }
  layer->activation(output_with_bias);

  return output_with_bias;
}

void mlp_append_layer(MLP *mlp, uint32_t neurons_count, activation_fn act_fn)
{
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
                      activation_fn act_fn)
{
  Layer *current_tail =
      mlp->layers_count > 0 ? mlp->layers[mlp->layers_count - 1] : NULL;

  Layer *layer = layer_new(current_tail == NULL ? mlp->input_size
                                                : current_tail->output_size,
                           neurons_count, act_fn);

  mlp->layers_count++;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));

  for (uint32_t i = mlp->layers_count - 1; i > index; i--)
  {
    mlp->layers[i] = mlp->layers[i - 1];
  }
  mlp->layers[index] = layer;
}

void mlp_remove_layer(MLP *mlp, uint32_t index)
{
  if (index >= mlp->layers_count)
  {
    return;
  }

  Layer *layer = mlp->layers[index];
  layer_free(layer);

  for (uint32_t i = index; i < mlp->layers_count - 1; i++)
  {
    mlp->layers[i] = mlp->layers[i + 1];
  }

  mlp->layers_count--;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));
}

void mlp_pop_layer(MLP *mlp)
{
  if (mlp->layers_count == 0)
  {
    return;
  }

  Layer *layer = mlp->layers[mlp->layers_count - 1];
  layer_free(layer);

  mlp->layers_count--;
  mlp->layers =
      (Layer **)(realloc(mlp->layers, mlp->layers_count * sizeof(Layer *)));
}

Tensor *mlp_forward(MLP *mlp, Tensor *input)
{
  Tensor *current_output = input;

  for (uint32_t i = 0; i < mlp->layers_count; i++)
  {
    Layer *layer = mlp->layers[i];
    Tensor *next_output = layer_forward(layer, current_output);

    if (next_output == NULL)
    {
      fprintf(stderr, "error: Forward pass failed at layer %u.\n", i);
      return NULL;
    }

    if (i > 0)
    {
      t_free(current_output);
    }
    current_output = next_output;
  }

  return current_output;
}

// NOTE: For now it only supports setup: sigmoid with MSE
void mlp_backward(MLP *mlp, Tensor *input, Tensor *gt, float lr)
{
  // Step 1: Calculate activations and output
  Tensor **activations = malloc(sizeof(Tensor *) * mlp->layers_count);
  Tensor *current_output = input;
  int32_t batch_size = input->shape[0];
  float one_over_bs = 1.0 / batch_size;

  for (uint32_t i = 0; i < mlp->layers_count; i++)
  {
    Layer *layer = mlp->layers[i];
    Tensor *next_output = layer_forward(layer, current_output);
    activations[i] = next_output;

    if (next_output == NULL)
    {
      // Cleanup
      for (uint32_t j = 0; j < i; j++)
      {
        t_free(activations[j]);
      }
      free(activations);

      fprintf(stderr, "error: Forward pass failed at layer %u.\n", i);
      return;
    }

    current_output = next_output;
  }

  // Step 2: Backward propagation and param update
  Tensor *delta;
  Tensor *prev_delta = NULL;
  Tensor *cdw, *cdb;
  Tensor *pdw = NULL, *pdb = NULL;
  float mone = -1;
  float one = 1;
  for (int32_t i = mlp->layers_count - 1; i >= 0; i--)
  {
    Tensor *cactv = activations[i];
    Tensor *pactv = i == 0 ? input : activations[i - 1];
    if (i == mlp->layers_count - 1)
    {
      Tensor *diff = t_sub_tensor(cactv, gt);
      Tensor *mul = t_mul_tensor(cactv, diff);
      Tensor *act_cpy = t_copy(cactv);
      t_multiply(act_cpy, &mone);
      t_add_scalar(act_cpy, &one);
      delta = t_mul_tensor(mul, act_cpy);
      t_free(diff);
      t_free(mul);
      t_free(act_cpy);
    }
    else
    {
      Tensor *next_w_trans = t_transpose(mlp->layers[i + 1]->weights);
      Tensor *first_part = t_matmul_tensor(prev_delta, next_w_trans);
      Tensor *mul = t_mul_tensor(first_part, cactv);
      Tensor *act_cpy = t_copy(cactv);
      t_multiply(act_cpy, &mone);
      t_add_scalar(act_cpy, &one);
      delta = t_mul_tensor(mul, act_cpy);
      t_free(first_part);
      t_free(mul);
      t_free(act_cpy);
      t_free(next_w_trans);
    }

    // Calculate current dW
    Tensor *pact_t = t_transpose(pactv);
    cdw = t_matmul_tensor(pact_t, delta);
    t_free(pact_t);

    t_multiply(cdw, &one_over_bs);

    // Calculate current db
    cdb = t_zeros_like(mlp->layers[i]->biases);
    for (uint32_t j = 0; j < cdb->shape[1]; j++)
    {
      float sum = 0.0f;
      for (uint32_t b = 0; b < delta->shape[0]; b++)
      {
        sum += *(float *)t_get_element(delta, (size_t[]){b, j});
      }
      t_set_element(cdb, &sum, (size_t[]){0, j});
    }
    t_multiply(cdb, &one_over_bs);

    if (prev_delta)
    {
      Tensor *step_w = t_copy(pdw);
      t_multiply(step_w, &lr);
      Tensor *step_b = t_copy(pdb);
      t_multiply(step_b, &lr);

      // TODO: Update weights of next layer
      Tensor *old_weights = mlp->layers[i + 1]->weights;
      mlp->layers[i + 1]->weights = t_sub_tensor(old_weights, step_w);

      Tensor *old_bias = mlp->layers[i + 1]->biases;
      mlp->layers[i + 1]->biases = t_sub_tensor(old_bias, step_b);

      t_free(old_bias);
      t_free(old_weights);
      t_free(step_w);
      t_free(step_b);
    }

    t_free(pdw);
    t_free(pdb);
    pdw = cdw;
    pdb = cdb;
    prev_delta = delta;
  }
  // Update weights of first layer
  Tensor *step_w = t_copy(pdw);
  t_multiply(step_w, &lr);
  Tensor *step_b = t_copy(pdb);
  t_multiply(step_b, &lr);
  Tensor *old_weights = mlp->layers[0]->weights;
  mlp->layers[0]->weights = t_sub_tensor(old_weights, step_w);

  Tensor *old_bias = mlp->layers[0]->biases;
  mlp->layers[0]->biases = t_sub_tensor(old_bias, step_b);

  t_free(old_bias);
  t_free(old_weights);
  t_free(step_w);
  t_free(step_b);

  // Cleanup
  for (uint32_t i = 0; i < mlp->layers_count; i++)
  {
    t_free(activations[i]);
  }
  free(activations);
  t_free(pdw);
  t_free(pdb);
  t_free(prev_delta);
}
