#include "filters.h"

/*
 * an example of a dummy partition function : blocks ...
 */
unsigned block_filter_func(filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	/* we will have arg chunks */
	nchunks = MIN(root_data->nx, arg);
	
	/* first allocate the children data_state */
	root_data->children = malloc(nchunks*sizeof(data_state));
	ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (root_data->nx + nchunks - 1)/nchunks;
		size_t offset = chunk*chunk_size*root_data->elemsize;

		root_data->children[chunk].nx = 
			MIN(chunk_size, root_data->nx - chunk*chunk_size);
		root_data->children[chunk].ny = root_data->ny;

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			local_data_state *local = &root_data->children[chunk].per_node[node];
			if (root_data->per_node[node].allocated) {
				local->ptr = root_data->per_node[node].ptr + offset;
				local->ld = root_data->per_node[node].ld;
			}
		}
	}

	return nchunks;
}

unsigned vertical_block_filter_func(filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	/* we will have arg chunks */
	nchunks = MIN(root_data->ny, arg);
	
	/* first allocate the children data_state */
	root_data->children = malloc(nchunks*sizeof(data_state));
	ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (root_data->ny + nchunks - 1)/nchunks;


		root_data->children[chunk].nx = root_data->nx; 
		root_data->children[chunk].ny =
			MIN(chunk_size, root_data->ny - chunk*chunk_size);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			local_data_state *local = &root_data->children[chunk].per_node[node];
			if (root_data->per_node[node].allocated) {
				size_t offset = chunk*chunk_size*root_data->per_node[node].ld*root_data->elemsize;
				local->ptr = root_data->per_node[node].ptr + offset;
				local->ld = root_data->per_node[node].ld;
			}
		}
	}

	return nchunks;
}
