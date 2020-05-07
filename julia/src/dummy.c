#include <starpu.h>

void  *dummy_function_list[] = {
				starpu_matrix_filter_vertical_block,
				starpu_matrix_filter_block,
				starpu_vector_filter_block,
				starpu_init,
};

