/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "hierarchy.h"

/* 
 * Stop monitoring a data
 */
/* TODO : move in a more appropriate file */
void starpu_delete_data(data_state *state)
{
	unsigned node;

	STARPU_ASSERT(state);
	for (node = 0; node < MAXNODES; node++)
	{
		local_data_state *local = &state->per_node[node];

		if (local->allocated && local->automatically_allocated){
			/* free the data copy in a lazy fashion */
			request_mem_chunk_removal(state, node);
		}
	}

#ifdef NO_DATA_RW_LOCK
	data_requester_list_delete(state->req_list);
#endif
}

void register_new_data(data_state *state, uint32_t home_node, uint32_t wb_mask)
{
	STARPU_ASSERT(state);

	/* initialize the new lock */
#ifndef NO_DATA_RW_LOCK
	init_rw_lock(&state->data_lock);
#else
	state->req_list = data_requester_list_new();
	state->refcnt = 0;
#endif
	pthread_spin_init(&state->header_lock, 0);

	/* first take care to properly lock the data */
	pthread_spin_lock(&state->header_lock);

	/* we assume that all nodes may use that data */
	state->nnodes = MAXNODES;

	/* there is no hierarchy yet */
	state->nchildren = 0;

	state->is_not_important = 0;

	/* make sure we do have a valid copy */
	STARPU_ASSERT(home_node < MAXNODES);

	state->wb_mask = wb_mask;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (node == home_node) {
			/* this is the home node with the only valid copy */
			state->per_node[node].state = OWNER;
			state->per_node[node].allocated = 1;
			state->per_node[node].automatically_allocated = 0;
			state->per_node[node].refcnt = 0;
		}
		else {
			/* the value is not available here yet */
			state->per_node[node].state = INVALID;
			state->per_node[node].allocated = 0;
			state->per_node[node].refcnt = 0;
		}
	}

	/* now the data is available ! */
	pthread_spin_unlock(&state->header_lock);
}

/*
 * This function applies a starpu_filter on all the elements of a partition
 */
static void map_filter(data_state *root_data, starpu_filter *f)
{
	/* we need to apply the starpu_filter on all leaf of the tree */
	if (root_data->nchildren == 0) 
	{
		/* this is a leaf */
		starpu_partition_data(root_data, f);
	}
	else {
		/* try to apply the starpu_filter recursively */
		int child;
		for (child = 0; child < root_data->nchildren; child++)
		{
			map_filter(&root_data->children[child], f);
		}
	}
}

void starpu_map_filters(data_state *root_data, unsigned nfilters, ...)
{
	unsigned i;
	va_list pa;
	va_start(pa, nfilters);
	for (i = 0; i < nfilters; i++)
	{
		starpu_filter *next_filter;
		next_filter = va_arg(pa, starpu_filter *);

		STARPU_ASSERT(next_filter);

		map_filter(root_data, next_filter);
	}
	va_end(pa);
}

/*
 * example get_sub_data(data_state *root_data, 3, 42, 0, 1);
 */
data_state *get_sub_data(data_state *root_data, unsigned depth, ... )
{
	STARPU_ASSERT(root_data);
	data_state *current_data = root_data;

	/* the variable number of argument must correlate the depth in the tree */
	unsigned i; 
	va_list pa;
	va_start(pa, depth);
	for (i = 0; i < depth; i++)
	{
		unsigned next_child;
		next_child = va_arg(pa, unsigned);

		STARPU_ASSERT((int)next_child < current_data->nchildren);

		current_data = &current_data->children[next_child];
	}
	va_end(pa);

	return current_data;
}

/*
 * For now, we assume that partitionned_data is already properly allocated;
 * at least by the starpu_filter function !
 */
void starpu_partition_data(data_state *initial_data, starpu_filter *f)
{
	int nparts;
	int i;

	/* first take care to properly lock the data header */
	pthread_spin_lock(&initial_data->header_lock);

	/* there should not be mutiple filters applied on the same data */
	STARPU_ASSERT(initial_data->nchildren == 0);

	/* this should update the pointers and size of the chunk */
	nparts = f->filter_func(f, initial_data);
	STARPU_ASSERT(nparts > 0);

	initial_data->nchildren = nparts;

	for (i = 0; i < nparts; i++)
	{
		data_state *children = &initial_data->children[i];

		STARPU_ASSERT(children);

		children->nchildren = 0;

		children->is_not_important = initial_data->is_not_important;

		/* it is possible that the children does not use the same interface as the parent,
		 * in that case, the starpu_filter must set the proper methods */
		if (!children->ops)
			children->ops = initial_data->ops;

		children->wb_mask = initial_data->wb_mask;

		/* initialize the chunk lock */
#ifndef NO_DATA_RW_LOCK
		init_rw_lock(&children->data_lock);
#else
		children->req_list = data_requester_list_new();
		children->refcnt = 0;
#endif
		pthread_spin_init(&children->header_lock, 0);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			children->per_node[node].state = 
				initial_data->per_node[node].state;
			children->per_node[node].allocated = 
				initial_data->per_node[node].allocated;
			children->per_node[node].automatically_allocated = initial_data->per_node[node].automatically_allocated;
			children->per_node[node].refcnt = 0;
		}
	}

	/* now let the header */
	pthread_spin_unlock(&initial_data->header_lock);
}

void starpu_unpartition_data(data_state *root_data, uint32_t gathering_node)
{
	int child;
	unsigned node;

	pthread_spin_lock(&root_data->header_lock);

#ifdef NO_DATA_RW_LOCK
#warning starpu_unpartition_data is not supported with NO_DATA_RW_LOCK yet ...
#endif

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_data->nchildren; child++)
	{
		/* make sure the intermediate children is unpartitionned as well */
		if (root_data->children[child].nchildren > 0)
			starpu_unpartition_data(&root_data->children[child], gathering_node);

		int ret;
		ret = _fetch_data(&root_data->children[child], gathering_node, 1, 0);
		/* for now we pretend that the RAM is almost unlimited and that gathering 
		 * data should be possible from the node that does the unpartionning ... we
		 * don't want to have the programming deal with memory shortage at that time,
		 * really */
		STARPU_ASSERT(ret == 0); 
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
				/* free the data copy in a lazy fashion */
				request_mem_chunk_removal(root_data, node);
				isvalid = 0; 
			}
		}

		/* no problem was found so the node still has a valid copy */
		still_valid[node] = isvalid;
		nvalids++;
	}

	/* either shared or owned */
	STARPU_ASSERT(nvalids > 0);

	cache_state newstate = (nvalids == 1)?OWNER:SHARED;

	for (node = 0; node < MAXNODES; node++)
	{
		root_data->per_node[node].state = 
			still_valid[node]?newstate:INVALID;
	}

	/* there is no child anymore */
	root_data->nchildren = 0;

	/* now the parent may be used again so we release the lock */
	pthread_spin_unlock(&root_data->header_lock);
}

void starpu_advise_if_data_is_important(data_state *state, unsigned is_important)
{

	pthread_spin_lock(&state->header_lock);

	/* first take all the children lock (in order !) */
	int child;
	for (child = 0; child < state->nchildren; child++)
	{
		/* make sure the intermediate children is advised as well */
		if (state->children[child].nchildren > 0)
			starpu_advise_if_data_is_important(&state->children[child], is_important);
	}

	state->is_not_important = !is_important;

	/* now the parent may be used again so we release the lock */
	pthread_spin_unlock(&state->header_lock);

}
