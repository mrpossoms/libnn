#include "test.h"
#include "../nn.h"
#include "../nn.c"

int mat_mul(void)
{
	float i[] = {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	};
	mat_t I = {
		.dims = { 3, 3 },
		._rank = 2,
		._size = 9,
		.data = i,
	};

	float m[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	};
	mat_t M = {
		.dims = { 3, 3 },
		._rank = 2,
		._size = 9,
		.data = m,
	};

	mat_t R = {
		.dims = { 3, 3 }
	};
	nn_mat_init(&R);

	nn_mat_mul(&R, &I, &M);

	for (int i = 9; i--;)
	{
		if (m[i] != R.data.f[i])
			return -1;
	}

	return 0;
}

TEST_BEGIN
	.name = "nn_mat_mul",
	.description = "Checks correctness of matrix multiplication.",
	.run = mat_mul,
TEST_END
