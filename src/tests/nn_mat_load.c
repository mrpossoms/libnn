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

int mat_load(void)
{
	mat_t M = nn_mat_load("data/model0/dense.kernel");

	assert(M.data.ptr);
	assert(M.dims[0] == 768);
	assert(M.dims[1] == 128);
	assert(M._size == 98304);
	assert(M._rank == 2);

	return 0;
}

TEST_BEGIN
	.name = "nn_mat_load",
	.description = "Checks matrix loading.",
	.run = mat_load,
TEST_END
