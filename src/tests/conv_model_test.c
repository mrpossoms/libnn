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


float classes[3][9 * 9] =  {
	{ 0.4373552677664668,0.4779668005977227,0.1930176971271862,0.15935195050556805,-0.03454942695534857,0.022374068795748436,0.06405680026965888,0.39244537551760794,-0.2735623389910957,0.38858326609909133,0.2389487947151785,0.48374378419428665,-0.2873746681814301,0.15689272126594211,-0.2247380377239223,-0.36435552100792035,-0.113894519160612,-0.433816069482448,-0.047338430355342576,-0.0074328193130708264,0.25158455017447046,-0.1316869118355698,0.32732382729263276,-0.03149328270894902,0.40673487883180215,0.23059401025303827,-0.14761593512965798,-0.489243676698937,0.4205509690622199,0.307691127548238,0.19089984721281394,0.39117070269955867,-0.32715124571459187,0.1411097709774165,0.10011676261464031,0.005543642303233787,0.12302408086427319,-0.42839800593512833,-0.03371917793608259,0.44079954354622175,-0.19602342865999123,0.44025648310799415,0.4663607283882778,0.10492501453134695,-0.34127869123826937,-0.18989831312044736,0.2029763412847888,0.2215775047424121,0.22286667629166013,0.17640028827130871,0.20423948382178803,-0.11416520245672268,0.4351949295559211,0.16930399873131985,-0.3953896478574027,-0.05084905409556473,-0.25921158228950436,-0.14211563779966496,-0.42265289391845084,-0.36933147427508617,-0.49382288624234916,-0.044746257776212994,-0.014855572304796283,0.4080228481416096,-0.3220690518534908,0.1394160420002314,0.09462338821082628,0.2779240661556387,0.26627358437614124,0.1280167941356074,-0.4852490528273933,0.3672766997201402,0.24093681779077813,0.4829553576078329,-0.40584452872711907,-0.2034742481867905,-0.13570583777988487,0.1493635626701023,-0.17886701602813115,0.19386606894576586,-0.36245899007683546, },

	{  0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1,
	   0, 0 ,1 ,1 ,1 ,1 ,1 ,1 ,1,
	   0, 0, 0 ,1 ,1 ,1 ,1 ,1 ,1,
	   0, 0, 0, 0 ,1 ,1 ,1 ,1 ,1,
	   0, 0, 0, 0, 0 ,1 ,1 ,1 ,1,
	   0, 0, 0, 0, 0, 0 ,1 ,1 ,1,
	   0, 0, 0, 0, 0, 0, 0 ,1 ,1,
	   0, 0, 0, 0, 0, 0, 0, 0  ,1,
	   0, 0, 0, 0, 0, 0, 0, 0, 0, },

	{ 1 ,1 ,1, 0, 0, 0, 0, 0, 0,
	  1 ,1 ,1 ,1, 0, 0, 0, 0, 0,
	  1 ,1 ,1 ,1 ,1, 0, 0, 0, 0,
	  1 ,1 ,1 ,1 ,1 ,1, 0, 0, 0,
	  1 ,1 ,1 ,1 ,1 ,1 ,1, 0, 0,
	  1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, 0,
	  1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1,
	  1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1,
	  1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, },
};

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
		.dims = { 9, 9, 1 },
		.data.f = classes[0]
	};
	nn_mat_init(&x);

	nn_layer_t L[] = {
		{
			.w = nn_mat_load("data/model1/c0.kernel"),
			.b = nn_mat_load("data/model1/c0.bias"),
			.activation = nn_act_relu,
			.filter = {
				.kernel = { 3, 3 },
				.stride = { 1, 1 },
				.padding = PADDING_VALID,
			},
			//.pool = MAX_POOL_HALF
		},
		{
			.w = nn_mat_load("data/model1/c1.kernel"),
			.b = nn_mat_load("data/model1/c1.bias"),
			.activation = nn_act_softmax,
			.filter = {
				.kernel = { 7, 7 },
				.stride = { 1, 1 },
				.padding = PADDING_VALID,
			},
			//.pool = MAX_POOL_HALF
		},
		{}
	};

	assert(nn_conv_init(L + 0, &x) == 0);
	for (int i = 1; i < 2; i++)
	{
		assert(nn_conv_init(L + i, (L + i - 1)->A) == 0);
	}

	for (int c = 0; c < 3; c++)
	// for (int c = 3; c--;)
	{
		x.data.f = classes[c];
		nn_conv_ff(L + 0, &x);
		for (int i = 1; i < 2; ++i)
		{
			nn_conv_ff(L + i, L[i - 1].A);
		}

		mat_t A_1 = *L[1].A;

		Log("%f %f %f", 1,
		A_1.data.f[0],
		A_1.data.f[1],
		A_1.data.f[2]);
	}

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
