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
	int cols = src->dims[1];
	*size = sizeof(float);
	return (void*)(src->data.f + (row * cols) + col);
}

int conv_pool(void)
{
	float src_buf[] = {
		0, 0, 2, 0,
		0, 1, 1, 1,
		0, 1, 0, 1,
		3, 2, 4, 2,
	};
	mat_t src = {
		.dims = { 4, 4, 1 },
		._rank = 3,
		._size = 16,
		.data = src_buf,
	};

	mat_t pool = {
		.dims = { 2, 2, 1 }
	};
	nn_mat_init(&pool);

	conv_op_t op = {
		.kernel = { 2, 2 },
		.corner = { 0, 0 },
		.stride = { 2, 2 },
		.pixel_indexer = indexer
	};

	nn_conv_max_pool(&pool, &src, op);
	for (int num = 1; num <= 4; ++num)
	{
		assert(pool.data.f[num-1] == num);
	}


	return 0;
}

TEST_BEGIN
	.name = "nn_conv_max_pool",
	.description = "Checks correctness of max pooling.",
	.run = conv_pool,
TEST_END
