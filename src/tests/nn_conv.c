#include "test.h"
#include "../nn.h"
#include "../nn.c"

uint8_t* indexer(mat_t* src, int row, int col, size_t* size)
{
	static const float zeros[256] = {};

	*size = sizeof(float) * src->dims[2];

	// Zero padding for SAME convolutions
	if (row < 0 || col < 0 ||
	    row >= src->dims[0] || col >= src->dims[1])
	{
		return (uint8_t*)zeros;
	}

	int cols = src->dims[1];
	return (uint8_t*)(src->_data.f + (row * cols) + col);
}


int conv_patch(void)
{
	float x0_s[] = {
		0, 1, 1,
		0, 1, 1,
		0, 1, 1
	};
	mat_t X0 = {
		.dims = { 3, 3, 1 },
		._rank = 3,
		._size = 9,
		._data.f = x0_s
	};

	float x1_s[] = {
		1, 1, 1,
		1, 1, 1,
		1, 1, 1
	};
	mat_t X1 = {
		.dims = { 3, 3, 1 },
		._rank = 3,
		._size = 9,
		._data.f = x1_s
	};

	float w_s[] = {
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1,
	};
	float b_s[] = {
		0
	};
	nn_layer_t conv = {
		.w = {
			.dims = { 3, 3, 1, 1 },
			._data.f = w_s
		},
		.filter = {
			.kernel = { 3, 3 },
			.stride = { 1, 1 },
			.pixel_indexer = indexer
		},
		.activation = nn_act_sigmoid
	};
	assert(nn_conv_init(&conv, &X0) == 0);

	// conv_op_t op = {
	// 	.kernel = { 3, 3 },
	// 	.stride = { 1, 1 },
	// 	.pixel_indexer = indexer
// };/

	nn_conv_ff(&conv, &X0);
	Log("A[0] -> %f\n", 1, conv.A->_data.f[0]);
	assert(conv.A->_data.f[0] > 0.5);

	nn_conv_ff(&conv, &X1);
	Log("A[0] -> %f\n", 1, conv.A->_data.f[0]);
	assert(conv.A->_data.f[0] <= 0.5);

	return 0;
}

TEST_BEGIN
	.name = "nn_conv",
	.description = "Runs one convolution.",
	.run = conv_patch,
TEST_END
