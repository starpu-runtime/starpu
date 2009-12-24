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

static void starpu_data_liberate_interfaces(starpu_data_handle handle)
{
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
		free(handle->interface[node]);
}

/* TODO : move in a more appropriate file */
void starpu_delete_data(starpu_data_handle handle)
{
	unsigned node;

	STARPU_ASSERT(handle);
	for (node = 0; node < MAXNODES; node++)
	{
		local_data_state *local = &handle->per_node[node];

		if (local->allocated && local->automatically_allocated){
			/* free the data copy in a lazy fashion */
			request_mem_chunk_removal(handle, node);
		}
	}

	data_requester_list_delete(handle->req_list);

	starpu_data_liberate_interfaces(handle);

	free(handle);
}

void register_new_data(starpu_data_handle handle, uint32_t home_node, uint32_t wb_mask)
{
	STARPU_ASSERT(handle);

	/* initialize the new lock */
	handle->req_list = data_requester_list_new();
	handle->refcnt = 0;
	starpu_spin_init(&handle->header_lock);

	/* first take care to properly lock the data */
	starpu_spin_lock(&handle->header_lock);

	/* we assume that all nodes may use that data */
	handle->nnodes = MAXNODES;

	/* there is no hierarchy yet */
	handle->nchildren = 0;

	handle->is_not_important = 0;

	handle->wb_mask = wb_mask;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		if (node == home_node) {
			/* this is the home node with the only valid copy */
			handle->per_node[node].state = OWNER;
			handle->per_node[node].allocated = 1;
			handle->per_node[node].automatically_allocated = 0;
			handle->per_node[node].refcnt = 0;
		}
		else {
			/* the value is not available here yet */
			handle->per_node[node].state = INVALID;
			handle->per_node[node].allocated = 0;
			handle->per_node[node].refcnt = 0;
		}
	}

	/* now the data is available ! */
	starpu_spin_unlock(&handle->header_lock);
}

/*
 * This function applies a starpu_filter on all the elements of a partition
 */
static void map_filter(starpu_data_handle root_handle, starpu_filter *f)
{
	/* we need to apply the starpu_filter on all leaf of the tree */
	if (root_handle->nchildren == 0)
	{
		/* this is a leaf */
		starpu_partition_data(root_handle, f);
	}
	else {
		/* try to apply the starpu_filter recursively */
		int child;
		for (child = 0; child < root_handle->nchildren; child++)
		{
			map_filter(&root_handle->children[child], f);
		}
	}
}

void starpu_map_filters(starpu_data_handle root_handle, unsigned nfilters, ...)
{
	unsigned i;
	va_list pa;
	va_start(pa, nfilters);
	for (i = 0; i < nfilters; i++)
	{
		starpu_filter *next_filter;
		next_filter = va_arg(pa, starpu_filter *);

		STARPU_ASSERT(next_filter);

		map_filter(root_handle, next_filter);
	}
	va_end(pa);
}

/*
 * example get_sub_data(starpu_data_handle root_handle, 3, 42, 0, 1);
 */
starpu_data_handle starpu_data_get_child(starpu_data_handle handle, unsigned i)
{
#warning TODO nchildren should not be an int
	STARPU_ASSERT(i < (unsigned)handle->nchildren);

	return &handle->children[i];
}

starpu_data_handle get_sub_data(starpu_data_handle root_handle, unsigned depth, ... )
{
	STARPU_ASSERT(root_handle);
	starpu_data_handle current_handle = root_handle;

	/* the variable number of argument must correlate the depth in the tree */
	unsigned i; 
	va_list pa;
	va_start(pa, depth);
	for (i = 0; i < depth; i++)
	{
		unsigned next_child;
		next_child = va_arg(pa, unsigned);

		STARPU_ASSERT((int)next_child < current_handle->nchildren);

		current_handle = &current_handle->children[next_child];
	}
	va_end(pa);

	return current_handle;
}

/*
 * For now, we assume that partitionned_data is already properly allocated;
 * at least by the starpu_filter function !
 */
void starpu_partition_data(starpu_data_handle initial_handle, starpu_filter *f)
{
	int nparts;
	int i;

	/* first take care to properly lock the data header */
	starpu_spin_lock(&initial_handle->header_lock);

	/* there should not be mutiple filters applied on the same data */
	STARPU_ASSERT(initial_handle->nchildren == 0);

	/* this should update the pointers and size of the chunk */
	nparts = f->filter_func(f, initial_handle);
	STARPU_ASSERT(nparts > 0);

	initial_handle->nchildren = nparts;

	for (i = 0; i < nparts; i++)
	{
		starpu_data_handle children =
			starpu_data_get_child(initial_handle, i);

		STARPU_ASSERT(children);

		children->nchildren = 0;

		children->is_not_important = initial_handle->is_not_important;

		/* it is possible that the children does not use the same interface as the parent,
		 * in that case, the starpu_filter must set the proper methods */
		if (!children->ops)
			children->ops = initial_handle->ops;

		children->wb_mask = initial_handle->wb_mask;

		/* initialize the chunk lock */
		children->req_list = data_requester_list_new();
		children->refcnt = 0;
		starpu_spin_init(&children->header_lock);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			children->per_node[node].state = 
				initial_handle->per_node[node].state;
			children->per_node[node].allocated = 
				initial_handle->per_node[node].allocated;
			children->per_node[node].automatically_allocated = initial_handle->per_node[node].automatically_allocated;
			children->per_node[node].refcnt = 0;
		}
	}

	/* now let the header */
	starpu_spin_unlock(&initial_handle->header_lock);
}

void starpu_unpartition_data(starpu_data_handle root_handle, uint32_t gathering_node)
{
	int child;
	unsigned node;

	starpu_spin_lock(&root_handle->header_lock);

#warning starpu_unpartition_data is not supported with NO_DATA_RW_LOCK yet ...

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_handle->nchildren; child++)
	{
		/* make sure the intermediate children is unpartitionned as well */
		if (root_handle->children[child].nchildren > 0)
			starpu_unpartition_data(&root_handle->children[child], gathering_node);

		int ret;
		ret = fetch_data_on_node(&root_handle->children[child], gathering_node, 1, 0, 0);
		/* for now we pretend that the RAM is almost unlimited and that gathering 
		 * data should be possible from the node that does the unpartionning ... we
		 * don't want to have the programming deal with memory shortage at that time,
		 * really */
		STARPU_ASSERT(ret == 0); 

		starpu_data_liberate_interfaces(&root_handle->children[child]);
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

		for (child = 0; child < root_handle->nchildren; child++)
		{
			local_data_state *local = &root_handle->children[child].per_node[node];

			if (local->state == INVALID) {
				isvalid = 0; 
			}
	
			if (local->allocated && local->automatically_allocated){
				/* free the data copy in a lazy fashion */
				request_mem_chunk_removal(root_handle, node);
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
		root_handle->per_node[node].state = 
			still_valid[node]?newstate:INVALID;
	}

	/* there is no child anymore */
	root_handle->nchildren = 0;

	/* now the parent may be used again so we release the lock */
	starpu_spin_unlock(&root_handle->header_lock);
}

/* TODO move ! */
void starpu_advise_if_data_is_important(starpu_data_handle handle, unsigned is_important)
{
	starpu_spin_lock(&handle->header_lock);

	/* first take all the children lock (in order !) */
	int child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure the intermediate children is advised as well */
		if (handle->children[child].nchildren > 0)
			starpu_advise_if_data_is_important(&handle->children[child], is_important);
	}

	handle->is_not_important = !is_important;

	/* now the parent may be used again so we release the lock */
	starpu_spin_unlock(&handle->header_lock);

}

starpu_data_handle starpu_data_state_create(size_t interfacesize)
{
	struct starpu_data_state_t *state =
		calloc(1, sizeof(struct starpu_data_state_t));

	STARPU_ASSERT(state);

	state->interface_size = interfacesize;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		state->interface[node] = calloc(1, interfacesize);
		STARPU_ASSERT(state->interface[node]);
	}

	return state;
}

void starpu_data_create_children(starpu_data_handle handle,
		unsigned nchildren, size_t interfacesize)
{
	handle->children = calloc(nchildren, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(handle->children);

	unsigned node;
	unsigned child;

	for (child = 0; child < nchildren; child++)
	{
		starpu_data_handle handle_child = &handle->children[child];

		handle_child->interface_size = interfacesize;

		for (node = 0; node < MAXNODES; node++)
		{
			handle_child->interface[node] = calloc(1, interfacesize);
			STARPU_ASSERT(handle->children->interface[node]);
		}
	}

	handle->nchildren = nchildren;
}
