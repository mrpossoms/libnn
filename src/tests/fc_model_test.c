#include "test.h"
#include "../nn.h"
#include "../nn.c"
#include "hay.h"

float hay[] = HAY;

int model_test(void)
{
	mat_t x = {
		.dims = { 1, 768 },
		.data.f = hay
	};

	nn_layer_t L[] = {
		{
			.w = nn_mat_load("data/dense.kernel"),
			.b = nn_mat_load("data/dense.bias"),
			.activation = nn_act_relu
		},
		{
			.w = nn_mat_load("data/dense_1.kernel"),
			.b = nn_mat_load("data/dense_1.bias"),
			.activation = nn_act_softmax
		},
		{}
	};

	// Allocate and setup layers and matrices
	assert(nn_mat_init(&x) == 0);
	assert(nn_init(L, &x) == 0);

	mat_t* y = nn_predict(L, &x);

	Log("%f %f %f", 1,
	y->data.f[0],
	y->data.f[1],
	y->data.f[2]);

	// mat_t fcw0 = nn_mat_load("model/dense.kernel");
	// mat_t fcb0 = nn_mat_load("model/dense.bias");
	// mat_t fcw1 = nn_mat_load("model/dense_1.kernel");
	// mat_t fcb1 = nn_mat_load("model/dense_1.bias");

	return 0;
}

TEST_BEGIN
	.name = "fc_model_test",
	.description = "Loads a one layer fully connected NN and produces a hypothesis on test input",
	.run = model_test,
TEST_END
