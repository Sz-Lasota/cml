# CML

This project is a simple implementation of MLP in C. Goal of this project is to
learn how to implement a simple neural network from scratch in C language,
including tensor operations, forward and backward propagation, and training
using (stochastic) gradient descent.

## Usage

This project is aimed to run on linux systems (for now). To run the project, clone it with git (initializing git submodules) and run the following commands:

```bash
git clone --recurse-submodules

cd cml
make
```

Run binary with:

```bash
./build/bin/cml <path/to/mnist-images> <path/to/mnist-labesl> [path/to/test-mnist-images] [path/to/test-mnist-labesl] 
```

## Results

Network takes around 3000 epochs to train and achieves around 93% of accuracy. Known limitations:
1. MSE loss function is hard-coded, both in training and in gradient calculation
2. There is no function for splitting datastes and test dataset is used as validation dataset in training
3. Poor exception handling

