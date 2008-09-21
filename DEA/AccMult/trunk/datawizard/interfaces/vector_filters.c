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
	nchunks = MIN(nx, arg);

	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (nx + nchunks - 1)/nchunks;
		size_t offset = chunk*chunk_size*elemsize;

		uint32_t child_nx = 
			MIN(chunk_size, nx - chunk*chunk_size);

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
