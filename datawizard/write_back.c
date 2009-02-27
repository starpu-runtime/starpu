#include <datawizard/write_back.h>
#include <datawizard/coherency.h>

void write_through_data(data_state *state, uint32_t requesting_node, 
					   uint32_t write_through_mask)
{
	if ((write_through_mask & ~(1<<requesting_node)) == 0) {
		/* nothing will be done ... */
		return;
	}

	while (take_mutex_try(&state->header_lock))
		datawizard_progress(requesting_node);

	/* first commit all changes onto the nodes specified by the mask */
	uint32_t node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (write_through_mask & (1<<node)) {
			/* we need to commit the buffer on that node */
			if (node != requesting_node) 
			{
				/* the requesting node already has the data by
				 * definition */
				int ret;
				ret = driver_copy_data_1_to_1(state, 
						requesting_node, node, 0);

				/* there must remain memory on the write-through mask to honor the request */
				if (ret)
					STARPU_ASSERT(0);
			}
				
			/* now the data is shared among the nodes on the
			 * write_through_mask */
			state->per_node[node].state = SHARED;
		}
	}

	/* the requesting node is now one sharer */
	if (write_through_mask & ~(1<<requesting_node))
	{
		state->per_node[requesting_node].state = SHARED;
	}

	release_mutex(&state->header_lock);
}

void data_set_wb_mask(data_state *data, uint32_t wb_mask)
{
	data->wb_mask = wb_mask;

	/* in case the data has some children, set their wb_mask as well */
	if (data->nchildren > 0) 
	{
		int child;
		for (child = 0; child < data->nchildren; child++)
			data_set_wb_mask(&data->children[child], wb_mask);
	}
}
