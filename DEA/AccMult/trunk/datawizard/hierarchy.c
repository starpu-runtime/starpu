#include "hierarchy.h"

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
				ASSERT(root_data != node);
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

	/* now the parent may be used again so we release the lock */
	release_lock(&root_data->lock);
}
