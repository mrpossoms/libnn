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


int conv_patch(void)
{
	float src_buf[] = {
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	};
	mat_t src = {
		.dims = { 4, 4, 1 },
	};
	nn_mat_init(&src);
	src.data.ptr = (void*)src_buf;

	mat_t patch = {
		.dims = { 1, 4 }
	};
	nn_mat_init(&patch);

	conv_op_t op = {
		.kernel = { 2, 2 },
		.corner = { 0, 0 },
		.pixel_indexer = nn_default_indexer,
	};

	int corners[][2] = {
		{0, 0},
		{0, 2},
		{2, 0},
		{2, 2},
	};

	for (int num = 1; num <= 4; ++num)
	{
		float exp = num;
		op.corner.row = corners[num-1][0];
		op.corner.col = corners[num-1][1];
		nn_conv_patch(&patch, &src, op);

		for (int i = patch.dims[0]; i--;)
		for (int j = patch.dims[1]; j--;)
		{
			float e = *nn_mat_e(&patch, i, j);
			if (e != exp)
			{
				Log("patch[%d,%d] == %f, expected %f", 0, i, j, e, exp);
				return -num;
			}
		}
	}


	return 0;
}

TEST_BEGIN
	.name = "nn_conv_patch",
	.description = "Checks correctness of patch slicing for convolutions.",
	.run = conv_patch,
TEST_END
