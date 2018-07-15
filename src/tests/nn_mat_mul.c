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


void ident(mat_t* M)
{
	for (int r = M->dims[0]; r--;)
	for (int c = M->dims[1]; c--;)
	{
		if (r == c) *e2f(M, r, c) = 1.f;
		else *e2f(M, r, c) = 0.f;
	}
}


void rand_mat(mat_t* M)
{
	for (int r = M->dims[0]; r--;)
	for (int c = M->dims[1]; c--;)
	{
		*e2f(M, r, c) = random() % 10;
	}
}


int mat_mul(void)
{

	mat_t I = {
		.dims = { 3, 3 },
#ifdef USE_VECTORIZATION
		.row_major = 1,
#endif
	};

	mat_t M = {
		.dims = { 3, 3 },
	};

	mat_t R = {
		.dims = { 3, 3 },
#ifdef USE_VECTORIZATION
		.row_major = 1,
#endif
	};
	nn_mat_init(&I);
	nn_mat_init(&M);
	nn_mat_init(&R);

	ident(&I);
	rand_mat(&M);

	nn_mat_mul(&R, &I, &M);

	for (int r = M.dims[0]; r--;)
	for (int c = M.dims[1]; c--;)
	{
		if (*e2f(&R, r, c) != *e2f(&M, r, c))
			return -1;
	}

	return 0;
}

TEST_BEGIN
	.name = "nn_mat_mul",
	.description = "Checks correctness of matrix multiplication.",
	.run = mat_mul,
TEST_END
