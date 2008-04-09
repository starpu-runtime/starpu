#include "hierarchy.h"

void monitor_new_data(data_state *state, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize)
{
	ASSERT(state);

	/* initialize the new lock */
	state->lock.taken = FREE;
#ifdef USE_SPU
	state->ea_data_state = (uintptr_t)&state;
	state->lock.ea_taken = (uintptr_t)&state->lock.taken;
#endif

	/* first take care to properly lock the data */
	take_lock(&state->lock);

	/* we assume that all nodes may use that data */
	state->nnodes = MAXNODES;

	/* there is no hierarchy yet */
	state->nchildren = 0;

	/* make sure we do have a valid copy */
	ASSERT(home_node < MAXNODES);

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (node == home_node) {
			/* this is the home node with the only valid copy */
			state->per_node[node].state = OWNER;
			state->per_node[node].ptr = ptr;
			state->per_node[node].ld = ld;
			state->per_node[node].allocated = 1;
			state->per_node[node].automatically_allocated = 0;
		}
		else {
			/* the value is not available here yet */
			state->per_node[node].state = INVALID;
			state->per_node[node].ptr = 0;
			state->per_node[node].allocated = 0;
		}
	}

	state->nx = nx;
	state->ny = ny;
	state->elemsize = elemsize;

	/* now the data is available ! */
	release_lock(&state->lock);
}

/*
 * This function applies a filter on all the elements of a partition
 */
void map_filter(data_state *root_data, filter *f)
{
	/* we need to apply the filter on all leaf of the tree */
	if (root_data->nchildren == 0) 
	{
		/* this is a leaf */
		partition_data(root_data, f);
	}
	else {
		/* try to apply the filter recursively */
		int child;
		for (child = 0; child < root_data->nchildren; child++)
		{
			map_filter(&root_data->children[child], f);
		}
	}
}

/*
 * For now, we assume that partitionned_data is already properly allocated;
 * at least by the filter function !
 */
void partition_data(data_state *initial_data, filter *f)
{
	int nparts;
	int i;

	/* first take care to properly lock the data */
	take_lock(&initial_data->lock);

	/* this should update the pointers and size of the chunk */
	nparts = f->filter_func(f, initial_data);
	ASSERT(nparts > 0);

	initial_data->nchildren = nparts;

	for (i = 0; i < nparts; i++)
	{
		data_state *children = &initial_data->children[i];

		children->elemsize = initial_data->elemsize;
		children->nchildren = 0;

		/* initialize the chunk lock */
		children->lock.taken = FREE;

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			children->per_node[node].state = 
				initial_data->per_node[node].state;
			children->per_node[node].allocated = 
				initial_data->per_node[node].allocated;
			children->per_node[node].automatically_allocated = 0;
		}
	}
}

void unpartition_data(data_state *root_data, uint32_t gathering_node)
{
	int child;
	unsigned node;

	/* note that at that point, the parent lock should be taken ! 
	 * XXX check that */

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_data->nchildren; child++)
	{
		fetch_data_without_lock(&root_data->children[child],
			gathering_node, 1/* read */, 0 /* write */);
	}

	/* the gathering_node should now have a valid copy of all the children.
	 * For all nodes, if the node had all copies and none was locally
	 * allocated then the data is still valid there, else, it's invalidated
	 * for the gathering node, if we have some locally allocated data, we 
	 * copy all the children (XXX this should not happen so we just do not
	 * do anything since this is transparent ?) */
	unsigned still_valid[MAXNODES];

	/* we do 2 passes : the first pass determines wether the data is still
	 * valid or not, the second pass is needed to choose between SHARED and
	 * OWNER */

	unsigned nvalids = 0;

	/* still valid ? */
	for (node = 0; node < MAXNODES; node++)
	{
		/* until an issue is found the data is assumed to be valid */
		unsigned isvalid = 1;

		for (child = 0; child < root_data->nchildren; child++)
		{
			local_data_state *local = &root_data->children[child].per_node[node];

			if (local->state == INVALID) {
				isvalid = 0; 
			}
	
			if (local->allocated && local->automatically_allocated){
				//ASSERT(gathering_node != node);
				/* XXX free the data copy ! */

				isvalid = 0; 
			}
		}

		/* no problem was found so the node still has a valid copy */
		still_valid[node] = isvalid;
		nvalids++;
	}

	/* either shared or owned */
	ASSERT(nvalids > 0);

	cache_state newstate = (nvalids == 1)?OWNER:SHARED;

	for (node = 0; node < MAXNODES; node++)
	{
		root_data->per_node[node].state = 
			still_valid[node]?newstate:INVALID;
	}

	/* there is no child anymore */
	root_data->nchildren = 0;

	/* now the parent may be used again so we release the lock */
	release_lock(&root_data->lock);
}

