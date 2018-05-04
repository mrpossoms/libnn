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
	float i[] = {
		1, 1, 1,
	};
	mat_t I = {
		.dims = { 1, 3 },
		._rank = 2,
		._size = 3,
		._data = i,
	};

	mat_t R = {
		.dims = { 1, 3 }
	};
	nn_mat_init(&R);

	nn_mat_f(&R, &I, sigmoid_f);

	for (int i = 3; i--;)
	{
		if (R._data.f[i] != 0.5) {
			Log("R[%d] -> %f\n", 0, i, R._data.f[i]);
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
