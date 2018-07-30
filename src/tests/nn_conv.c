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


void mat_copy(mat_t* dst, float* src)
{
	for (int i = dst->dims[0]; i--;)
	for (int j = dst->dims[1]; j--;)
	{
		*nn_mat_e(dst, i, j) = src[i * dst->dims[1] + j];
	}
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
		.is_activation_map = 1,
#ifdef USE_VECTORIZATION
		.row_major = 1,
#endif
	};
	assert(nn_mat_init(&X0) == 0);
	mat_copy(&X0, x0_s);

	float x1_s[] = {
		1, 1, 1,
		1, 1, 1,
		1, 1, 1
	};
	mat_t X1 = {
		.dims = { 3, 3, 1 },
		.is_activation_map = 1,
#ifdef USE_VECTORIZATION
		.row_major = 1,
#endif
	};
	assert(nn_mat_init(&X1) == 0);
	mat_copy(&X1, x1_s);

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
			.dims = { 9, 1 },
		},
		.b = {
			.dims = { 1, 1 },
		},
		.filter = {
			.kernel = { 3, 3 },
			.stride = { 1, 1 },
			.pixel_indexer = nn_default_indexer
		},
		.activation = nn_act_relu
	};
	assert(nn_conv_init(&conv, &X0) == 0);
	mat_copy(&conv.w, w_s);


	nn_conv_ff(&conv, &X0);
	Log("A[0] -> %f\n", 1, conv.A->data.f[0]);
	assert(conv.A->data.f[0] == 3);

	nn_conv_ff(&conv, &X1);
	Log("A[0] -> %f\n", 1, conv.A->data.f[0]);
	assert(conv.A->data.f[0] == 0);

	return 0;
}

TEST_BEGIN
	.name = "nn_conv",
	.description = "Runs one convolution.",
	.run = conv_patch,
TEST_END
