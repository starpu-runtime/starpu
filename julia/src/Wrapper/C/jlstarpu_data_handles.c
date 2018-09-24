
#include "jlstarpu.h"




#if 0
void print_vector_interface(struct starpu_vector_interface * i)
{
	printf("Vector interface at %p\n", i);
	printf("\tdev_handle : %p\n", i->dev_handle);
	printf("\telement_size : %u\n", i->elemsize);
	printf("\tnx : %u\n", i->nx);
	printf("\toffset : %u\n", i->offset);
	printf("\tptr : %p\n", i->ptr);
	printf("\tslide_base : %u\n", i->slice_base);
}
#endif


enum jlstarpu_data_filter_func
{
    JLSTARPU_MATRIX_FILTER_VERTICAL_BLOCK = 0,
    JLSTARPU_MATRIX_FILTER_BLOCK
};



struct jlstarpu_data_filter
{
	enum jlstarpu_data_filter_func func;
	unsigned int nchildren;

};


void * jlstarpu_translate_data_filter_func(enum jlstarpu_data_filter_func func)
{

	switch (func){

	case JLSTARPU_MATRIX_FILTER_VERTICAL_BLOCK:
		return starpu_matrix_filter_vertical_block;

	case JLSTARPU_MATRIX_FILTER_BLOCK:
		return starpu_matrix_filter_block;

	default:
		return NULL;

	}

}


void jlstarpu_translate_data_filter
(
		const struct jlstarpu_data_filter * const input,
		struct starpu_data_filter * output
)
{
	memset(output, 0, sizeof(struct starpu_data_filter));

	output->filter_func = jlstarpu_translate_data_filter_func(input->func);
	output->nchildren = input->nchildren;

}









void jlstarpu_data_partition
(
		starpu_data_handle_t handle,
		const struct jlstarpu_data_filter * const jl_filter
)
{
	struct starpu_data_filter filter;
	jlstarpu_translate_data_filter(jl_filter, &filter);

	starpu_data_partition(handle, &filter);

}


void jlstarpu_data_map_filters_1_arg
(
		starpu_data_handle_t handle,
		const struct jlstarpu_data_filter * const jl_filter
)
{
	struct starpu_data_filter filter;
	jlstarpu_translate_data_filter(jl_filter, &filter);

	starpu_data_map_filters(handle, 1, &filter);

}


void jlstarpu_data_map_filters_2_arg
(
		starpu_data_handle_t handle,
		const struct jlstarpu_data_filter * const jl_filter_1,
		const struct jlstarpu_data_filter * const jl_filter_2
)
{
	struct starpu_data_filter filter_1;
	jlstarpu_translate_data_filter(jl_filter_1, &filter_1);

	struct starpu_data_filter filter_2;
	jlstarpu_translate_data_filter(jl_filter_2, &filter_2);


	starpu_data_map_filters(handle, 2, &filter_1, &filter_2);

}




#define JLSTARPU_GET(interface, field, ret_type)\
	\
	ret_type jlstarpu_##interface##_get_##field(const struct starpu_##interface##_interface * const x)\
	{\
		return (ret_type) x->field;\
	}\





JLSTARPU_GET(vector, ptr, void *)
JLSTARPU_GET(vector, nx, uint32_t)
JLSTARPU_GET(vector, elemsize, size_t)



JLSTARPU_GET(matrix, ptr, void *)
JLSTARPU_GET(matrix, ld, uint32_t)
JLSTARPU_GET(matrix, nx, uint32_t)
JLSTARPU_GET(matrix, ny, uint32_t)
JLSTARPU_GET(matrix, elemsize, size_t)















