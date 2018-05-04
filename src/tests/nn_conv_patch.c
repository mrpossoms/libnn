#include "test.h"
#include "../nn.h"
#include "../nn.c"

uint8_t* indexer(mat_t* src, int row, int col, size_t* size)
{
	int cols = src->dims[1];
	*size = sizeof(float);
	return (void*)(src->_data.f + (row * cols) + col);
}

int conv_patch(void)
{
	float src_buf[] = {
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	};
	mat_t src = {
		.dims = { 4, 4 },
		._rank = 2,
		._size = 16,
		._data = src_buf,
	};

	mat_t patch = {
		.dims = { 2, 2 }
	};
	nn_mat_init(&patch);

	conv_op_t op = {
		.kernel = { 2, 2 },
		.corner = { 0, 0 },
		.pixel_indexer = indexer
	};

	int corners[][2] = {
		{0, 0},
		{0, 2},
		{2, 0},
		{2, 2},
	};

	for (int num = 1; num <= 4; ++num)
	{
		op.corner.row = corners[num-1][0];
		op.corner.col = corners[num-1][1];
		nn_conv_patch(&patch, &src, op);

		for (int i = patch._size; i--;)
		{
			if (patch._data.f[i] != num)
			{
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
