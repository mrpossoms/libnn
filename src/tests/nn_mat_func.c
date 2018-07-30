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

#include <math.h>

float sigmoid_f(float v)
{
	return 1 - (powf(v, 2) / (1 + powf(v, 2)));
}

int mat_mul(void)
{
	mat_t I = {
		.dims = { 1, 3 },
	};

	mat_t R = {
		.dims = { 1, 3 }
	};

	assert(nn_mat_init(&R) == 0);
	assert(nn_mat_init(&I) == 0);

	for (int c = 3; c--;)
	{
		float* e = nn_mat_e(&I, 0, c);
		*e = 1.f;
	}

	nn_mat_f(&R, &I, sigmoid_f);

	for (int i = 3; i--;)
	{
		float e = *nn_mat_e(&R, 0, i);
		if (e != 0.5) {
			Log("R[%d] -> %f\n", 0, i, e);
			return -1;
		}
	}

	return 0;
}

TEST_BEGIN
	.name = "nn_mat_f",
	.description = "Checks element-wise function application.",
	.run = mat_mul,
TEST_END
