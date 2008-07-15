#include "csr_filters.h"
#include "csr_interface.h"

unsigned vertical_block_filter_func_csr(filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	uint32_t nrow = root_data->interface[0].csr.nrow;
	size_t elemsize = root_data->interface[0].csr.elemsize;
	uint32_t firstentry = root_data->interface[0].csr.firstentry;

	/* we will have arg chunks */
	nchunks = MIN(nrow, arg);
	
	/* first allocate the children data_state */
	root_data->children = malloc(nchunks*sizeof(data_state));
	ASSERT(root_data->children);

	/* actually create all the chunks */
	uint32_t chunk_size = (nrow + nchunks - 1)/nchunks;

	/* XXX */
	ASSERT(root_data->per_node[0].allocated);
	uint32_t *rowptr = root_data->interface[0].csr.rowptr;

	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t first_index = chunk*chunk_size - firstentry;
		uint32_t local_firstentry = rowptr[first_index];

		uint32_t child_nrow = 
			MIN(chunk_size, nrow - chunk*chunk_size);

		uint32_t local_nnz = rowptr[first_index + child_nrow] - rowptr[first_index]; 

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			csr_interface_t *local = &root_data->children[chunk].interface[node].csr;

			local->nnz = local_nnz;
			local->nrow = child_nrow;
			local->firstentry = local_firstentry;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				local->rowptr = &root_data->interface[node].csr.rowptr[first_index];
				local->colind = &root_data->interface[node].csr.colind[local_firstentry];
				float *nzval = (float *)(root_data->interface[node].csr.nzval);
				local->nzval = (uintptr_t)&nzval[local_firstentry];
			}
		}
	}

	return nchunks;
}
