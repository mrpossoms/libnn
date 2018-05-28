# libnn

Tests: [![CircleCI](https://circleci.com/gh/mrpossoms/libnn.svg?style=svg)](https://circleci.com/gh/mrpossoms/libnn)

libnn is an intuitive feed-forward neural network library designed with embedded linux systems in mind. It's intended to instantiate a trained model for performing predictions in the field.

## Requirements
* POSIX compliant OS
* Python 2.7+ (for running tests)
* gcc
* gnu make

## Installation

```
$ make install
```
Will build a static library, and copy it and the header file to `/usr/local/lib` and `/usr/local/include` respectively.

## Usage

libnn defines two primary struct types to be aware of, `mat_t` and `nn_layer_t`. `mat_t` serves as a description and container for either and NxM matrix or a tensor. Allocating a zero filled matrix is as easy as the following.

```C
mat_t M = {
	.dims = { 4, 4 }
};

nn_mat_init(&M); // returns 0 on success
```

### _Network Declaration_

A libnn neural network is composed of `nn_layer_t` instances, which in turn contain a handful of matrices. When defining a network architecture there are only a few that you need to be concerned with. `w` the connection weights, `b` the biases, and `A`, a pointer to the vector of activations for that layer. The network should be defined as an array of `nn_layer_t` instances, with the final layer being empty to act as a terminator. The flow of activations follows the order of the layers defined in the array.

```C
mat_t x = {
	.dims = { 1, 768 },
};
nn_mat_init(&x);

nn_layer_t L[] = {
	{ // Layer 0
		.w = { .dims = { 256, 768 } }, // shape of layer 0's weight matrix
		.b = { .dims = { 256, 1 } },   // shape of layer 0's bias matrix
		.activation = nn_act_relu      // pointer to layer 0's activation function
	},
	{ // Layer 1 (output layer)
		.w = { .dims = { 3, 256 } }, // shape of layer 1's weight matrix
		.b = { .dims = { 3, 1 } },   // shape of layer 1's bias matrix
		.activation = nn_act_softmax // pointer to layer 1's activation function
	},
	{} // terminator
};

nn_init(L, &x); // returns 0 on success
```

### _Loading a Trained Model_

In practice, however, you would want to load stored weights and biases from files using `nn_mat_load`.

```c
nn_layer_t L[] = {
	{
		.w = nn_mat_load("data/dense.kernel"),
		.b = nn_mat_load("data/dense.bias"),
		.activation = nn_act_relu
	},
	{
		.w = nn_mat_load("data/dense_1.kernel"),
		.b = nn_mat_load("data/dense_1.bias"),
		.activation = nn_act_softmax
	},
	{}
};
```

Matrices loaded by `nn_mat_load` are stored in a simple binary format. With a header starting with a 1 byte integer stating the number of dimensions, and the equivalent number of 4 byte integers following it.

```
[ ui8 num_dimensions | ui32 dim 0 | ... | ui32 dim num_dimensions - 1 ]

```

After the header, the remainder of the matrix consists of a number of 32 bit floats equivalent to the product of the dimensions in the header. The default matrix indexer assumes they are stored in row major order.

### _Making Predictions_

A prediction can be carried out by the network with a call to `nn_predict` like so. `nn_predict` returns a pointer to the set of activations produced by the final layer, which is the output of the network.

```C
mat_t* y = nn_predict(L, &x);

float* p = y->data.f;
printf("predictions: %f %f %f\n", p[0], p[1], p[2])
```
