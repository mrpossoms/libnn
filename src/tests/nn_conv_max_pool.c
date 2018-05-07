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
