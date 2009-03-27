#include "vector_filters.h"
#include "vector_interface.h"

unsigned block_filter_func_vector(filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	vector_interface_t *vector_root = &root_data->interface[0].vector;
	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(nx, arg);

	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (nx + nchunks - 1)/nchunks;
		size_t offset = chunk*chunk_size*elemsize;

		uint32_t child_nx = 
			STARPU_MIN(chunk_size, nx - chunk*chunk_size);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			vector_interface_t *local = &root_data->children[chunk].interface[node].vector;

			local->nx = child_nx;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				local->ptr = root_data->interface[node].vector.ptr + offset;
			}
		}
	}

	return nchunks;
}


unsigned divide_in_2_filter_func_vector(filter *f, data_state *root_data)
{
	uint32_t length_first = f->filter_arg;

	vector_interface_t *vector_root = &root_data->interface[0].vector;
	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* first allocate the children data_state */
	root_data->children = calloc(2, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	STARPU_ASSERT(length_first < nx);

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		vector_interface_t *local = &root_data->children[0].interface[node].vector;

		local->nx = length_first;
		local->elemsize = elemsize;

		if (root_data->per_node[node].allocated) {
			local->ptr = root_data->interface[node].vector.ptr;
		}
	}

	for (node = 0; node < MAXNODES; node++)
	{
		vector_interface_t *local = &root_data->children[1].interface[node].vector;

		local->nx = nx - length_first;
		local->elemsize = elemsize;

		if (root_data->per_node[node].allocated) {
			local->ptr = root_data->interface[node].vector.ptr + length_first*elemsize;
		}
	}

	return 2;
}

unsigned list_filter_func_vector(filter *f, data_state *root_data)
{
	uint32_t nchunks = f->filter_arg;
	uint32_t *length_tab = f->filter_arg_ptr;

	vector_interface_t *vector_root = &root_data->interface[0].vector;
	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	unsigned current_pos = 0;

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = length_tab[chunk];

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			vector_interface_t *local = &root_data->children[chunk].interface[node].vector;

			local->nx = chunk_size;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				local->ptr = root_data->interface[node].vector.ptr + current_pos*elemsize;
			}
		}

		current_pos += chunk_size;
	}

	STARPU_ASSERT(current_pos == nx);

	return nchunks;
}
