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

#include <arpa/inet.h>
#include "test.h"
#include "../nn.h"
#include "../nn.c"



#define MAX_POOL_HALF {          \
    .type = POOLING_MAX,         \
    .op = {                      \
       .padding = PADDING_VALID,  \
       .stride = { 2, 2 },       \
       .kernel = { 2, 2 },       \
    }                            \
}\

typedef struct {
	uint8_t pixels[28 * 28];
} idx_image_t;

typedef struct {
	uint32_t magic, items;
} idx1_header_t;

typedef struct {
	idx1_header_t base;
	uint32_t rows, cols;
} idx3_header_t;


float* idx3_next()
{
	static int fd;
	static idx3_header_t hdr;

	if (!fd)
	{
		fd = open("data/model_conv2/ds/test/images-idx3-ubyte", O_RDONLY);
        assert(fd >= 0);

		assert(read(fd, &hdr, sizeof(hdr)) == sizeof(hdr));
        hdr.base.magic = ntohl(hdr.base.magic);
        hdr.base.items = ntohl(hdr.base.items);
        hdr.rows = ntohl(hdr.rows);
        hdr.cols = ntohl(hdr.cols);
	}

	idx_image_t img;

	if (read(fd, &img, sizeof(img)) == sizeof(img))
	{
		static float img_vec[28 * 28];
		for (int i = 28 * 28; i--;) img_vec[i] = img.pixels[i] / 255.f;
		return img_vec;
	}
	else
	{
		close(fd);
		fd = 0;
	}

	return NULL;
}

int idx1_next()
{
	static int fd;
	static idx1_header_t hdr;

	if (!fd)
	{
		fd = open("data/model_conv2/ds/test/labels-idx1-ubyte", O_RDONLY);
		read(fd, &hdr, sizeof(hdr));
        hdr.magic = ntohl(hdr.magic);
        hdr.items = ntohl(hdr.items);
	}

    uint8_t label;
	if (read(fd, &label, sizeof(label)) == sizeof(label))
	{
		return label;
	}
	else
	{
		close(fd);
		fd = 0;
	}

	return -1;
}

void mat_copy(mat_t* dst, float* src)
{
    for (int i = dst->dims[0]; i--;)
    for (int j = dst->dims[1]; j--;)
    {
        *nn_mat_e(dst, i, j) = src[i * dst->dims[1] + j];
    }
}

int max_idx(float* a, int len)
{
    int i = 0;
    for (;len--;)
    {
        if (a[len] > a[i]) i = len;
    }

    return i;
}

int model_test(void)
{
    mat_t x = {
        .dims = { 28, 28, 1 },
        .is_activation_map = 1,
#ifdef USE_VECTORIZATION
        .row_major = 1,
#endif
    };
    nn_mat_init(&x);

    nn_layer_t L[] = {
        {
            .w = nn_mat_load_row_order("data/model_conv2/c0.kernel", 0),
            .b = nn_mat_load_row_order("data/model_conv2/c0.bias", 1),
            .activation = nn_act_relu,
            .filter = {
                .kernel = { 3, 3 },
                .stride = { 1, 1 },
                .padding = PADDING_VALID,

            },
            .pool = MAX_POOL_HALF,
        },
        {
         .w = nn_mat_load_row_order("data/model_conv2/c1.kernel", 0),
         .b = nn_mat_load_row_order("data/model_conv2/c1.bias", 1),
         .activation = nn_act_relu,
         .filter = {
             .kernel = { 5, 5 },
             .stride = { 1, 1 },
             .padding = PADDING_VALID,
         },
         .pool = MAX_POOL_HALF,
        },
        {
         .w = nn_mat_load_row_order("data/model_conv2/c2.kernel", 0),
         .b = nn_mat_load_row_order("data/model_conv2/c2.bias", 1),
         .activation = nn_act_linear,
         .filter = {
             .kernel = { 4, 4 },
             .stride = { 1, 1 },
             .padding = PADDING_VALID,
         },
        },
        {}
    };

    assert(nn_init(L, &x) == 0);
    int right = 0, wrong = 0, total = 0;

    for(int i = 10000; i--;)
    {
        float* f = idx3_next();

        if (!f) break;

        mat_copy(&x, f);
        int label = idx1_next();
        mat_t* A_1 = nn_predict(L, &x);

        if (max_idx(A_1->data.f, 10) == label)
        {
            right++;
        }
        else
        {
            wrong++;
        }

        total++;

        // all_passed &= passed;

        // Log("label: %d", 1, label);
        // Log("%f %f %f %f %f %f %f %f %f %f", passed,
        // A_1->data.f[0],
        // A_1->data.f[1],
        // A_1->data.f[2],
        // A_1->data.f[3],
        // A_1->data.f[4],
        // A_1->data.f[5],
        // A_1->data.f[6],
        // A_1->data.f[7],
        // A_1->data.f[8],
        // A_1->data.f[9]
        // );
    }

    float acc = right / (float)total;

    Log("right/wrong: %d/%d", acc > 0.8, right, wrong);
    Log("accuracy: %f%%", acc > 0.8, acc);

    return 0;
}

TEST_BEGIN
    .name = "conv_model2_test",
    .description = "Puts everything together and builds a convolutional model",
    .run = model_test,
TEST_END
