#include "nn.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#ifdef USE_VECTORIZATION
#define e2f(M, r, c) ((M)->row_major ? \
	(M)->data.f + ((r) * (M)->_p_dims[1]) + (c) : \
	(M)->data.f + ((c) * (M)->_p_dims[0]) + (r))
#else
#define e2f(M, r, c) ((M)->data.f + (M)->dims[1] * r + c)
#endif

float* nn_default_indexer(mat_t* src, int row, int col, size_t* size)
{
	static const float zeros[256] = {};

	int cols = src->dims[1];
	int depth = src->dims[2] ? src->dims[2] : 1;

	*size = depth;

	// Zero padding for SAME convolutions
	if (row < 0 || col < 0 ||
	    row >= src->dims[0] || col >= src->dims[1])
	{
		return (float*)zeros;
	}


	return src->data.f + (row * (cols * depth)) + (col * depth);
}
//------------------------------------------------------------------------------


static float zero_fill(mat_t* M)
{
	return 0;
}
//------------------------------------------------------------------------------


int nn_mat_init(mat_t* M)
{
	if (!M) return -1;
	if (!M->fill)
	{
		M->fill = zero_fill;
	}

	// determine the dimensionality asked for
	M->_size = 1;
	for (M->_rank = 0; M->_rank < NN_MAT_MAX_DIMS && M->dims[M->_rank];)
	{
		int d = M->dims[M->_rank];
#ifdef USE_VECTORIZATION
		if (d == 1 || M->is_activation_map)
		{ // prevent the non-major dimension from being padded.
		  // that would cause NN_MAT_BLOCK_SIZE * (major dim len)
		  // extra operations which are all useless.
			M->_p_dims[M->_rank] = d;
		}
		else
		{
			M->_p_dims[M->_rank] = ceil(d / (float)NN_MAT_BLOCK_SIZE);
			M->_p_dims[M->_rank] *= NN_MAT_BLOCK_SIZE;
		}

		M->_size *= M->_p_dims[M->_rank];
#else
		M->_size *= d;
#endif
		++M->_rank;
	}

	// size less than 2 is no good
	assert(M->_rank >= 2);

	size_t total_elements;

#ifdef USE_VECTORIZATION
	total_elements = M->_p_dims[0];
#else
	total_elements = M->dims[0];
#endif

	for (int i = 1; i < M->_rank; ++i)
	{
#ifdef USE_VECTORIZATION
		total_elements *= M->_p_dims[i];
#else
		total_elements *= M->dims[i];
#endif
	}

	if (M->data.ptr == NULL)
	{
		M->data.ptr = calloc(total_elements, sizeof(float));

		// Check for allocation failure
		if (!M->data.ptr) return -2;

		// perform fill initialization
		for (int i = total_elements; i--;)
		{
			(M->data.f)[i] = M->fill(M);
		}
	}

	return 0;
}
//------------------------------------------------------------------------------


float* nn_mat_e(mat_t* M, int row, int col)
{
		return e2f(M, row, col);
}
//------------------------------------------------------------------------------


void nn_mat_transpose(mat_t* M)
{
	int t;
#ifdef USE_VECTORIZATION
	M->row_major = !M->row_major;
	t = M->_p_dims[0];
	M->_p_dims[0] = M->_p_dims[1];
	M->_p_dims[1] = t;
#endif

	t = M->dims[0];
	M->dims[0] = M->dims[1];
	M->dims[1] = t;

}
//------------------------------------------------------------------------------


static inline void _BATCH4_MUL(float* res, float* row_A, int i, int col_B, mat_t* B)
{
	float* b_col = B->data.f + col_B;
	float _A __attribute__ ((vector_size(16))) = {
		row_A[i + 0],
		row_A[i + 1],
		row_A[i + 2],
		row_A[i + 3],
	};
	float _B __attribute__ ((vector_size(16))) = {
		b_col[B->dims[1] * (i + 0)],
		b_col[B->dims[1] * (i + 1)],
		b_col[B->dims[1] * (i + 2)],
		b_col[B->dims[1] * (i + 3)],
	};

	_A *= _B;

	*res += _A[0] + _A[1] + _A[2] + _A[3];
}
//------------------------------------------------------------------------------


void nn_mat_mul(mat_t* R, mat_t* A, mat_t* B)
{
	// MxN * NxO = MxO

	assert(R->_rank == A->_rank);
	assert(A->_rank == B->_rank);
	assert(R->dims[0] == A->dims[0]);
	assert(R->dims[1] == B->dims[1]);

	if(A->dims[1] != B->dims[0])
	{
		fprintf(stderr,
		        "nn_mat_mul: %dx%d not compatible with %dx%d\n",
		        A->dims[0], A->dims[1],
		        B->dims[0], B->dims[1]);

		exit(-1);
	}

#ifdef USE_VECTORIZATION
	int inner_blocks = A->_p_dims[1] / NN_MAT_BLOCK_SIZE;

	if (A->row_major == B->row_major)
	{
		fprintf(stderr, "%dx%d and %dx%d matrices are of the same majority\n",
		A->dims[0], A->dims[1],
		B->dims[0], B->dims[1]);

		exit(-2);
	}

	for (int ri = 0; ri < A->dims[0]; ++ri)
	for (int ci = 0; ci < B->dims[1]; ++ci)
	{
		v4f acc = {};

		v4f *a, *b;

		if (A->row_major)
			a = A->data.v + (ri * A->_p_dims[1] / NN_MAT_BLOCK_SIZE);
		else
			a = A->data.v + (ci * A->_p_dims[0] / NN_MAT_BLOCK_SIZE);

		if (B->row_major)
			b = B->data.v + (ri * B->_p_dims[1] / NN_MAT_BLOCK_SIZE);
		else
			b = B->data.v + (ci * B->_p_dims[0] / NN_MAT_BLOCK_SIZE);

		for (int bi = 0; bi < inner_blocks; ++bi)
		{
			acc += a[bi] * b[bi];
		}

		*e2f(R, ri, ci) =  acc[0] + acc[1] + acc[2] + acc[3];
	}
#else
	for (int ar = A->dims[0]; ar--;)
	for (int bc = B->dims[1]; bc--;)
	{
		float res = 0;
		float* row = e2f(A, ar, 0);

		for (int i = B->dims[0]; i;)
		{
			if (i > 16)
			{
				i -= 16;
				_BATCH4_MUL(&res, row, i, bc, B);
				_BATCH4_MUL(&res, row, i + 4, bc, B);
				_BATCH4_MUL(&res, row, i + 8, bc, B);
				_BATCH4_MUL(&res, row, i + 12, bc, B);
			}
			else if (i > 8)
			{
				i -= 8;
				_BATCH4_MUL(&res, row, i, bc, B);
				_BATCH4_MUL(&res, row, i + 4, bc, B);
			}
			else if (i >= 4)
			{
				i -= 4;
				_BATCH4_MUL(&res, row, i, bc, B);
			}
			else
			{
				i -= 1;
				res += row[i] * *e2f(B, i, bc);
			}
		}

		*e2f(R, ar, bc) = res;
	}
#endif
}
//------------------------------------------------------------------------------


void nn_mat_mul_e(mat_t* R, mat_t* A, mat_t* B)
{
	assert(R->_rank == A->_rank);
	assert(A->_rank == B->_rank);
	if(!(A->dims[0] == B->dims[0] && A->dims[1] == B->dims[1]))
	{
		fprintf(stderr,
		        "nn_mat_mul_e: %dx%d not compatible with %dx%d\n",
		        A->dims[0], A->dims[1],
		        B->dims[0], B->dims[1]);

		exit(-1);
	}


	for (int r = A->dims[0]; r--;)
	for (int c = A->dims[1]; c--;)
	{
		*e2f(R, r, c) = *e2f(A, r, c) * *e2f(B, r, c);
	}
}
//------------------------------------------------------------------------------


void nn_mat_add_e(mat_t* R, mat_t* A, mat_t* B)
{
	assert(R->_rank == A->_rank);
	assert(A->_rank == B->_rank);
	if(!(A->dims[0] == B->dims[0] && A->dims[1] == B->dims[1]))
	{
		fprintf(stderr,
		        "nn_mat_add_e: %dx%d not compatible with %dx%d\n",
		        A->dims[0], A->dims[1],
		        B->dims[0], B->dims[1]);

		exit(-1);
	}

	for (int r = A->dims[0]; r--;)
	for (int c = A->dims[1]; c--;)
	{
		*e2f(R, r, c) = *e2f(A, r, c) + *e2f(B, r, c);
	}
}
//------------------------------------------------------------------------------


void nn_mat_scl_e(mat_t* R, mat_t* M, float s)
{
	assert(R->_rank == M->_rank);
	assert(M->_rank == R->_rank);
	assert(R->dims[0] == M->dims[0] && R->dims[1] == M->dims[1]);

	for (int r = M->dims[0]; r--;)
	for (int c = M->dims[1]; c--;)
	{
		*e2f(R, r, c) = *e2f(M, r, c) * s;
	}
}
//------------------------------------------------------------------------------


void nn_mat_f(mat_t* R, mat_t* M, float (*func)(float))
{
	assert(R->_size == M->_size);

	for (int r = M->dims[0]; r--;)
	for (int c = M->dims[1]; c--;)
	{
		*e2f(R, r, c) = func(*e2f(M, r, c));
	}
}
//------------------------------------------------------------------------------


int nn_mat_max(mat_t* M)
{
	float max = M->data.f[0];
	int max_i = 0;
	for (int i = M->_size; i--;)
	{
		if (M->data.f[i] > max)
		{
			max = M->data.f[i];
			max_i = i;
		}
	}

	return max_i;
}
//------------------------------------------------------------------------------


static int is_conv_layer(nn_layer_t* l)
{
	if (!l) return -1;
	return l->filter.kernel.w && l->filter.kernel.h;
}
//------------------------------------------------------------------------------


static int is_empty_layer(nn_layer_t* l)
{
	return l == NULL || l->w.data.ptr == NULL;
}
//------------------------------------------------------------------------------


int nn_init(nn_layer_t* li, mat_t* x_in)
{
	if (!x_in) return -1;

	mat_t* A = x_in;

	while (!is_empty_layer(li))
	{
		int res = 0;
		if (is_conv_layer(li))
		{
			res = nn_conv_init(li, A);
		}
		else
		{
			res = nn_fc_init(li, A);
		}

		if (res) return res;

		A = li->A;
		li++;
	}

	return 0;
}
//------------------------------------------------------------------------------


int nn_clone(nn_layer_t* dst, nn_layer_t* src, mat_t* x_in)
{
	if (!src) return -1;
	if (!x_in) return -2;

	mat_t* A = x_in;

	while (!is_empty_layer(src))
	{
		int res = 0;

		*dst = *src;
		dst->A = NULL;
		dst->_CA.data.ptr = NULL;
		dst->_conv_patch.data.ptr = NULL;
		dst->_z.data.ptr = NULL;

		if (is_conv_layer(dst))
		{
			res = nn_conv_init(dst, A);
		}
		else
		{
			res = nn_fc_init(dst, A);
		}

		if (res) return res;

		A = dst->A;
		src++;
		dst++;
	}

	return 0;
}
//------------------------------------------------------------------------------


int nn_fc_init(nn_layer_t* li, mat_t* a_in)
{
	int res = 0;
	assert(li);
	assert(a_in);

	if (li->_z.data.ptr == NULL)
	{
		mat_t z = {
			.dims = { 1, li->w.dims[1] },
	#ifdef USE_VECTORIZATION
			.row_major = 1
	#endif
		};

		res += nn_mat_init(&z) * -10;
		li->_z = z;
	}

	li->_CA = li->_z;
	li->A = &li->_CA;

	return res;
}
//------------------------------------------------------------------------------


void nn_fc_ff(nn_layer_t* li, mat_t* a_in)
{

	nn_mat_mul(&li->_z, a_in, &li->w);
	nn_mat_add_e(li->A, li->A, &li->b);

	li->activation(li->A);
}
//------------------------------------------------------------------------------


int nn_conv_init(nn_layer_t* li, mat_t* a_in)
{
	int res = 0;
	assert(li);
	assert(a_in);

	int a_rows = a_in->dims[0];
	int a_cols = a_in->dims[1];
	// int depth_in = li->w.dims[2];
	int depth_out = li->w.dims[1];

	{ // Setup matrices for weights and biases
		// li->w.dims[0] = li->w.dims[0] * li->w.dims[1] * depth_in;
		// li->w.dims[1] = depth_out;
		// li->w.dims[2] = li->w.dims[3] = 0;
		res += nn_mat_init(&li->w) * -10;

		res += nn_mat_init(&li->b) * -20;

		if (res) return res;
	}

	{ // Setup preactivation vector
		mat_t z = {
			.dims = { 1, depth_out },
#ifdef USE_VECTORIZATION
			.row_major = 1,
#endif
		};
		res += nn_mat_init(&z) * -30;
		li->_z = z;

		if (res) return res;
	}

	{ // Setup patch vector
		mat_t patch = {
			.dims = { 1, li->w.dims[0] },
#ifdef USE_VECTORIZATION
			.row_major = 1,
#endif
		};
		res += nn_mat_init(&patch) * -40;
		li->_conv_patch = patch;

		if (res) return res;
	}

	{ // Setup convolution activation map
		int pad_row = 0;
		int pad_col = 0;

		conv_op_t f = li->filter;
		if (f.padding == PADDING_SAME)
		{
			pad_row = f.kernel.h / 2;
			pad_col = f.kernel.w / 2;
		}

		// if no custom indexing function is selected use the builtin
		if (!f.pixel_indexer) li->filter.pixel_indexer = nn_default_indexer;

		int ca_rows = ((a_rows - f.kernel.h + 2 * pad_row) / f.stride.row) + 1;
		int ca_cols = ((a_cols - f.kernel.w + 2 * pad_col) / f.stride.col) + 1;

		ca_rows = ca_rows ? ca_rows : 1;
		ca_cols = ca_cols ? ca_cols : 1;

		mat_t CA = {
			.dims = { ca_rows, ca_cols, depth_out },
#ifdef USE_VECTORIZATION
			.row_major = 1,
			.is_activation_map = 1,
#endif
		};
		res += nn_mat_init(&CA) * -50;
		li->_CA = CA;
		li->A = &li->_CA;

		if (res) return res;
	}

	// Setup pooling matrix
	switch (li->pool.type)
	{
		case POOLING_MAX:
		{
			mat_t PA = {
				.dims = {
					li->_CA.dims[0] / li->pool.op.kernel.h,
					li->_CA.dims[1] / li->pool.op.kernel.w,
					depth_out
				},
				.is_activation_map = 1,
			};
			res += nn_mat_init(&PA) * -60;
			li->pool._PA = PA;

			// if no custom indexing function is selected use the builtin
			if (!li->pool.op.pixel_indexer)
			{
				li->pool.op.pixel_indexer = nn_default_indexer;
			}

			li->A = &li->pool._PA;
			if (res) return res;
		}
		case POOLING_NONE:;
	}

	return res;
}
//------------------------------------------------------------------------------


void nn_conv_patch(mat_t* patch, mat_t* src, conv_op_t op)
{
	assert(patch->data.ptr);
	assert(src->data.ptr);

	for (int row = op.kernel.h; row--;)
	for (int col = op.kernel.w; col--;)
	{
		int ri = op.corner.row + row;
		int ci = op.corner.col + col;
		int i = row * op.kernel.w + col;
		size_t depth;
		float* pixel_chan = op.pixel_indexer(src, ri, ci, &depth);

		for (int d = depth; d--;)
		{
			// TODO: cleanup
			int i = col * depth;
			int j = row * op.kernel.w * depth;
			float* pile = nn_mat_e(patch, 0, j + i);
			pile[d] = pixel_chan[d];
			// printf("[%d + %d] = %f\n", i, j, pile[d]);
		}
	}
}
//------------------------------------------------------------------------------


void nn_conv_ff(nn_layer_t* li, mat_t* a_in)
{
	assert(a_in);
	assert(li);

	conv_op_t op = li->filter;
	int pad_row = 0;
	int pad_col = 0;
	mat_t* patch = &li->_conv_patch;

	if (op.padding == PADDING_SAME)
	{
		pad_row = op.kernel.h / 2;
		pad_col = op.kernel.w / 2;
	}

	// For each pile of channels in the pool...
	// for (int p_row = li->_CA.dims[0]; p_row--;)
	// for (int p_col = li->_CA.dims[1]; p_col--;)
	// TODO
	for (int p_row = 0; p_row < li->_CA.dims[0]; ++p_row)
	for (int p_col = 0; p_col < li->_CA.dims[1]; ++p_col)
	{
		op.corner.row = p_row * op.stride.row - pad_row;
		op.corner.col = p_col * op.stride.col - pad_col;

		// get the convolution window from the input activation volume
		nn_conv_patch(patch, a_in, op);

		// apply the filter
		nn_mat_mul(&li->_z, patch, &li->w);
		nn_mat_add_e(&li->_z, &li->_z, &li->b);

		// activate
		li->activation(&li->_z);

		size_t depth;
		float* z_pile = op.pixel_indexer(&li->_z, 0, 0, &depth);
		float* ca_pile = op.pixel_indexer(&li->_CA, p_row, p_col, &depth);

		for (int d = depth; d--;)
		{
			ca_pile[d] = z_pile[d];
			// ca_pile[d] = *nn_mat_e(&li->_z, p_row, d);
		}
	}


	// Apply pooling if specified
	switch (li->pool.type)
	{
		case POOLING_MAX:
		{
			nn_conv_max_pool(&li->pool._PA, &li->_CA, li->pool.op);
		}
		case POOLING_NONE:;
	}
}
//------------------------------------------------------------------------------


void nn_conv_max_pool(mat_t* pool, mat_t* src, conv_op_t op)
{
	int exp_size[NN_MAT_MAX_DIMS] = {
		src->dims[0] / op.stride.row,
		src->dims[1] / op.stride.col
	};

	assert(pool->_rank == src->_rank);
	for (int i = 2; i--;) assert(exp_size[i] == pool->dims[i]);

	memset(pool->data.f, 0, sizeof(float) * pool->_size);

	// For each pile of channels in the pool...
	for (int p_row = pool->dims[0]; p_row--;)
	for (int p_col = pool->dims[1]; p_col--;)
	{
		// get the pile address
		size_t size;
		float* pool_pile = op.pixel_indexer(pool, p_row, p_col, &size);

		// For each kernel window position
		for (int k_row = op.kernel.w; k_row--;)
		for (int k_col = op.kernel.h; k_col--;)
		{
			int s_row = p_row * op.stride.row + k_row;
			int s_col = p_col * op.stride.col + k_col;

			float* src_pile = op.pixel_indexer(src, s_row, s_col, &size);

			for (int chan = pool->dims[2]; chan--;)
			{
				if (pool_pile[chan] < src_pile[chan])
				{
					pool_pile[chan] = src_pile[chan];
				}
			}
		}
	}
}
//------------------------------------------------------------------------------


mat_t nn_mat_load(const char* path)
{
	return nn_mat_load_row_order(path, 1);
}
//------------------------------------------------------------------------------


mat_t nn_mat_load_row_order(const char* path, int row_major)
{
	mat_t M = {
#ifdef USE_VECTORIZATION
		.row_major = row_major
#endif
	};
	uint8_t dims = 0;
	int fd = open(path, O_RDONLY);

	// NOTE: bias indexing is effed up

	// open file, read the dimensions
	if (fd < 0) goto abort;
	if (read(fd, &dims, sizeof(uint8_t)) != sizeof(uint8_t)) goto abort;
	for (int i = 0; i < dims; ++i)
	{
		if (read(fd, M.dims + i, sizeof(int)) != sizeof(int)) goto abort;
	}

	if (dims == 1)
	{
		M.dims[1] = 1;
		nn_mat_transpose(&M);
#ifdef USE_VECTORIZATION
		M.row_major = !M.row_major;
#endif
	}

	// TODO: clean up, loading is not the right place
	// to reformat the parameters dimensions
	else if (dims >= 4)
	{
		int out_depth = M.dims[3];
		int in_size = M.dims[0] * M.dims[1] * M.dims[2];
		M.dims[0] = in_size;
		M.dims[1] = out_depth;
		M.dims[2] = M.dims[3] = 0;
		M._rank = 2;
	}

	// allocate space for the matrix
	if (nn_mat_init(&M)) goto abort;

	// read the entire matrix
	for (int ri = 0; ri < M.dims[0]; ++ri)
	for (int ci = 0; ci < M.dims[1]; ++ci)
	{
		read(fd, e2f(&M, ri, ci), sizeof(float));
	}

	close(fd);

	return M;
abort:
	free(M.data.ptr);
	M.data.ptr = NULL;
	close(fd);
	exit(-13);
	return M;
}
//------------------------------------------------------------------------------


static float _sigmoid_e(float v)
{
	return 1 / (1 + powf(M_E, -v));
}
//------------------------------------------------------------------------------


void nn_act_sigmoid(mat_t* z)
{
	nn_mat_f(z, z, _sigmoid_e);
}
//------------------------------------------------------------------------------


static float _relu_e(float v)
{
	return v > 0 ? v : 0;
}
//------------------------------------------------------------------------------


void nn_act_relu(mat_t* z)
{
	nn_mat_f(z, z, _relu_e);
}
//------------------------------------------------------------------------------


static float _softmax_num_f(float v)
{
	return powf(M_E, v);
}
//------------------------------------------------------------------------------


void nn_act_softmax(mat_t* z)
{
	nn_mat_f(z, z, _softmax_num_f);
	float sum = 0;
	for (int i = z->_size; i--;) sum += z->data.f[i];
	float denom = 1.f / sum ;
	nn_mat_scl_e(z, z, denom);
}
//------------------------------------------------------------------------------


void nn_act_linear(mat_t* z)
{
	return;
}
//------------------------------------------------------------------------------


mat_t* nn_predict(nn_layer_t* l, mat_t* x)
{
	mat_t* a_1 = x;

	for (;!is_empty_layer(l); ++l)
	{
		if (is_conv_layer(l))
		{
			nn_conv_ff(l, a_1);
		}
		else
		{
			nn_fc_ff(l, a_1);
		}

		a_1 = l->A;
	}

	return a_1;
}
