/*
 * This file is part of libnn.
 *
 * libnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libnn.  If not, see <https://www.gnu.org/licenses/>.
 */

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
	return (uint8_t*)(src->data.f + (row * cols) + col);
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
		.data.f = x0_s
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
		.data.f = x1_s
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
			.data.f = w_s
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
	Log("A[0] -> %f\n", 1, conv.A->data.f[0]);
	assert(conv.A->data.f[0] > 0.5);

	nn_conv_ff(&conv, &X1);
	Log("A[0] -> %f\n", 1, conv.A->data.f[0]);
	assert(conv.A->data.f[0] <= 0.5);

	return 0;
}

TEST_BEGIN
	.name = "nn_conv",
	.description = "Runs one convolution.",
	.run = conv_patch,
TEST_END
