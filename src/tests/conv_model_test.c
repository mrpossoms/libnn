#include "test.h"
#include "../nn.h"
#include "../nn.c"
#include "hay.h"

#define ROOT_DIR "/tmp/"

float hay[] = HAY;


#define MAX_POOL_HALF {          \
	.type = POOLING_MAX,         \
	.op = {                      \
	   .stride = { 2, 2 },       \
	   .kernel = { 2, 2 },       \
	}                            \
}\

int model_test(void)
{
	mat_t x = {
		.dims = { 16, 16, 3 },
		.data.f = hay
	};
	nn_mat_init(&x);

	nn_layer_t L[] = {
		{
			.w = nn_mat_load(ROOT_DIR "model/conv2d.kernel"),
			.b = nn_mat_load(ROOT_DIR "model/conv2d.bias"),
			.activation = nn_act_relu,
			.filter = {
				.kernel = { 3, 3 },
				.stride = { 1, 1 },
				.padding = PADDING_SAME,
			},
			.pool = MAX_POOL_HALF
		},
		{
			.w = nn_mat_load(ROOT_DIR "model/conv2d_1.kernel"),
			.b = nn_mat_load(ROOT_DIR "model/conv2d_1.bias"),
			.activation = nn_act_relu,
			.filter = {
				.kernel = { 3, 3 },
				.stride = { 1, 1 },
				.padding = PADDING_SAME,
			},
			.pool = MAX_POOL_HALF
		},
		{
			.w = nn_mat_load(ROOT_DIR "model/conv2d_2.kernel"),
			.b = nn_mat_load(ROOT_DIR "model/conv2d_2.bias"),
			.activation = nn_act_relu,
			.filter = {
				.kernel = { 3, 3 },
				.stride = { 1, 1 },
				.padding = PADDING_SAME,
			},
			.pool = MAX_POOL_HALF
		},
		{
			.w = nn_mat_load(ROOT_DIR "model/dense.kernel"),
			.b = nn_mat_load(ROOT_DIR "model/dense.bias"),
			.activation = nn_act_relu
		},
		{
			.w = nn_mat_load(ROOT_DIR "model/dense_1.kernel"),
			.b = nn_mat_load(ROOT_DIR "model/dense_1.bias"),
			.activation = nn_act_softmax
		}
	};

	assert(nn_conv_init(L + 0, &x) == 0);
	for (int i = 1; i < 3; i++)
	{
		assert(nn_conv_init(L + i, (L + i - 1)->A) == 0);
	}

	for (int i = 3; i < 5; i++)
	{
		assert(nn_fc_init(L + i, (L+i-1)->A) == 0);
	}

	nn_conv_ff(&x, L + 0);

	for (int i = 1; i < 3; ++i)
	{
		nn_conv_ff(L[i - 1].A, L + i);
	}

	mat_t A_1 = *L[2].A;
	A_1.dims[0] = 1;
	A_1.dims[1] = A_1._size;
	A_1.dims[2] = 0;
	A_1._rank = 2;

	for (int i = 3; i < 5; i++)
	{
		nn_fc_ff(L + i, &A_1);
		A_1 = *L[i].A;
		A_1.dims[1] = A_1.dims[0];
		A_1.dims[0] = 1;
	}

	Log("%f %f %f", 1,
	A_1.data.f[0],
	A_1.data.f[1],
	A_1.data.f[2]);

	// mat_t fcw0 = nn_mat_load("model/dense.kernel");
	// mat_t fcb0 = nn_mat_load("model/dense.bias");
	// mat_t fcw1 = nn_mat_load("model/dense_1.kernel");
	// mat_t fcb1 = nn_mat_load("model/dense_1.bias");

	return 0;
}

TEST_BEGIN
	.name = "conv_model_test",
	.description = "Puts everything together and builds a convolutional model",
	.run = model_test,
TEST_END
