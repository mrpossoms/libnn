#ifndef _NN_H
#define _NN_H

#include <sys/types.h>
#include <inttypes.h>

#define NN_MAT_MAX_DIMS 4


struct mat_t {
	/**
	 * @brief Int array specifying the length of each
	 *        orthoganal dimension. Must be null terminated
	 */
	int dims[NN_MAT_MAX_DIMS];

	/**
	 * @brief Optional: Uses this function to initialize each value
	 *                  of the matrix
	 */
	float (*fill)(struct mat_t*);

	/**
	 * @brief number of dimensions, this will be filled in automatically
	 */
	unsigned int _rank;

	/**
	 * @brief total number of elements
	 */
	unsigned int _size;

	/**
	 * @brief Raw pointer to a contiguous array used
	 *        to store the matrix's data
	 */
	union {
		void* ptr;
		float* f;
		double* d;
	} data;
};
typedef struct mat_t mat_t;


typedef enum {
	PADDING_VALID,
	PADDING_SAME
} conv_padding_t;


typedef enum {
	POOLING_NONE = 0,
	POOLING_MAX,
} conv_pooling_t;


typedef struct {
	struct {
		int w, h;
	} kernel;

	struct {
		int row, col;
	} stride;

	conv_padding_t padding;

	/**
	 * Returns a pointer to a pixel and all its consecutive channels.
	 *
	 * @param src  Matrix to retrieve a pixel from
	 * @param row  Row the pixel resides in, if outside the bounds of 'src'
	 *             a pointer to a 0 filled buffer of 'size' must be returned.
	 * @param col  Column the pixel resides in, if outside the bounds of 'src'
	 *             a pointer to a 0 filled buffer of 'size' must be returned.
	 * @param size Will contain the size of the pixel and its channels in bytes
	 * @return Pointer to contigious memory containing pixel
	 */
	uint8_t* (*pixel_indexer)(mat_t* src, int row, int col, size_t* size);

	struct {
		int row, col;
	} corner;

} conv_op_t;


typedef struct {
	mat_t w;
	mat_t b;
	void (*activation)(mat_t* z);

	conv_op_t filter;

	struct {
		conv_op_t op;
		conv_pooling_t type;
		mat_t _PA;
	} pool;

	/**
	 * @brief Final output of activations
	 */
	mat_t* A;

	mat_t _CA;
	mat_t _conv_patch;
	mat_t _z;
} nn_layer_t;


/**
 * @brief Allocates memory for matrix described by 'M'
 * @param M - description of desired matrix
 * @return 0 on success.
 **/
int nn_mat_init(mat_t* M);

// Matrix operations
// ------------------------------------
// All of these functions implicitly succeed, errors or inconsistencies
// will cause program termination

/**
 * @brief Performs matrix multiplication A x B storing the result in R
 * @param R - Resulting matrix of the multiplication. It's dimensions must be valid
 * @param A - Left hand of the muliplication
 * @param B - Right hand of the multiplication
 */
void nn_mat_mul(mat_t* R, mat_t* A, mat_t* B);

void nn_mat_mul_conv(mat_t* R, mat_t* A, mat_t* B);

/**
 * @brief Returns the index of the element with the most positive value.
 * @param  M - Pointer to a matrix
 * @return   Index of the element with the largest value
 */
int nn_mat_max(mat_t* M);

/**
 * @brief Performs element-wise multiplication A x B storing the result in R
 * @param R - Resulting matrix of the multiplication. It's dimensions must be valid
 * @param A - Left hand of the muliplication
 * @param B - Right hand of the multiplication
 */
void nn_mat_mul_e(mat_t* R, mat_t* A, mat_t* B);

/**
 * @brief Performs scaling operation on whole matrix M storing the result in R
 * @param R - Resulting matrix of the multiplication. It's dimensions must be valid
 * @param M - Left hand of the muliplication
 * @param s - Scalar that is multiplied by each element of M
 */
void nn_mat_scl_e(mat_t* R, mat_t* M, float s);

/**
 * @brief Performs element-wise addition A + B storing the result in R
 * @param R - Resulting matrix of the addition. It's dimensions must be valid
 * @param A - Left hand of the addition
 * @param B - Right hand of the addition
 */
void nn_mat_add_e(mat_t* R, mat_t* A, mat_t* B);

/**
 * Applies function element wise to all values in matrix M.
 * @param R -  Result of func on M. R must be the same shape as M.
 * @param M -  Matrix whose values will be passed through func
 * @param func Pointer to a function that takes a numeric value, apply
 *             some transformation and returns the result.
 */
void nn_mat_f(mat_t* R, mat_t* M, float (*func)(float));

/**
 * Loads a matrix from a file. Loading failure terminates the program.
 * @param  path - Path to the matrix file to be loaded
 * @return Matrix instance.
 */
mat_t nn_mat_load(const char* path);

/**
 * Allocates and performs initialization for a feed forward network.
 * @param  layers - Pointer to an array of described layers, with the final
 *                  layer being empty and uninitialized.
 * @param  x      - Pointer to feature vector to use as input.
 * @return          0 on success
 */
int nn_init(nn_layer_t* li, mat_t* x_in);

/**
 * Allocates matrices needed for the fully connected layer, and also
 * computes _size and _rank. 'w' and 'b' members of 'li' must be set
 * before calling this function.
 * @param  li - Pointer to a layer describing the desired layer.
 * @param  a_in - Pointer to an activation vector which will act as the input
 *                for this layer.
 * @return 0 on success
 */
int nn_fc_init(nn_layer_t* li, mat_t* a_in);

/**
 * Computes next set of activations for a fully connected layer.
 * @param li   - Pointer to fully connected layer
 * @param a_in - Activations from the previous layer
 */
void nn_fc_ff(nn_layer_t* li, mat_t* a_in);

/**
 * Allocates matrices needed for the convolutional layer, and also
 * computes _size and _rank. 'filter', 'stride', 'w' and 'b' members
 * of 'li' must be set.
 * @param  li - Pointer to a layer describing the desired layer.
 * @param  a_in - Pointer to an activation vector which will act as the input
 *                for this layer.
 * @return 0 on success
 */
int nn_conv_init(nn_layer_t* li, mat_t* a_in);

/**
 * Slices out a matrix of the dimensions of 'patch', at the location specified
 * by 'op.corner' from the 'src' matrix.
 * @param patch - Pointer to a matrix that will be assigned the sliced out data.
 * @param src   - Pointer to the matrix from which the patch will be sampled.
 * @param op    - Defines the size of the patch, the location and how the data
 *                are sampled.
 */
void nn_conv_patch(mat_t* patch, mat_t* src, conv_op_t op);

/**
 * Applies a max pooling operation to matrix 'src'
 * @param pool - Pointer to the matrix that will contain the pooling result
 * @param src  - Pointer to the matrix that will have the pooling operation applied
 * @param op   - Describes the window and stride of the pooling operation.
 */
void nn_conv_max_pool(mat_t* pool, mat_t* src, conv_op_t op);

/**
 * Computes next set of activations for a convolutional layer.
 * @param li   - Pointer to a convolutional layer
 * @param a_in - Activations from the previous layer
 */
void nn_conv_ff(nn_layer_t* li, mat_t* a_in);

/**
 * https://en.wikipedia.org/wiki/Sigmoid_function
 * @param z Pointer to matrix of preactivation values
 */
void nn_act_sigmoid(mat_t* z);

/**
 * https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 * @param z Pointer to matrix of preactivation values
 */
void nn_act_relu(mat_t* z);

/**
 * https://en.wikipedia.org/wiki/Softmax_function
 * @param z Pointer to matrix of preactivation values
 */
void nn_act_softmax(mat_t* z);

/**
 * Evaluates a feed forward network.
 * @param  layers - Pointer to an array of initialized layers, with the final
 *                  layer being empty and uninitialized.
 * @param  x      - Pointer to feature vector to use as input.
 * @return        Pointer to a vector of predictions.
 */
mat_t* nn_predict(nn_layer_t* layers, mat_t* x);

#endif
