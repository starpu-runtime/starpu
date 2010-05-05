/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <datawizard/filters.h>

/*
 * This function applies a starpu_filter on all the elements of a partition
 */
static void map_filter(starpu_data_handle root_handle, starpu_filter *f)
{
	/* we need to apply the starpu_filter on all leaf of the tree */
	if (root_handle->nchildren == 0)
	{
		/* this is a leaf */
		starpu_data_partition(root_handle, f);
	}
	else {
		/* try to apply the starpu_filter recursively */
		unsigned child;
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
 * example starpu_data_get_sub_data(starpu_data_handle root_handle, 3, 42, 0, 1);
 */
starpu_data_handle starpu_data_get_child(starpu_data_handle handle, unsigned i)
{
	STARPU_ASSERT(i < handle->nchildren);

	return &handle->children[i];
}

starpu_data_handle starpu_data_get_sub_data(starpu_data_handle root_handle, unsigned depth, ... )
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

		STARPU_ASSERT(next_child < current_handle->nchildren);

		current_handle = &current_handle->children[next_child];
	}
	va_end(pa);

	return current_handle;
}

/*
 * For now, we assume that partitionned_data is already properly allocated;
 * at least by the starpu_filter function !
 */
void starpu_data_partition(starpu_data_handle initial_handle, starpu_filter *f)
{
	int nparts;
	int i;

	/* first take care to properly lock the data header */
	_starpu_spin_lock(&initial_handle->header_lock);

	/* there should not be mutiple filters applied on the same data */
	STARPU_ASSERT(initial_handle->nchildren == 0);

	/* this should update the pointers and size of the chunk */
	f->filter_func(f, initial_handle);

	nparts = initial_handle->nchildren;
	STARPU_ASSERT(nparts > 0);

	for (i = 0; i < nparts; i++)
	{
		starpu_data_handle children =
			starpu_data_get_child(initial_handle, i);

		STARPU_ASSERT(children);

		children->nchildren = 0;
		children->root_handle = initial_handle->root_handle;
		children->father_handle = initial_handle;
		children->sibling_index = i;
		children->depth = initial_handle->depth + 1;

		children->is_not_important = initial_handle->is_not_important;

		children->wb_mask = initial_handle->wb_mask;

		/* initialize the chunk lock */
		children->req_list = starpu_data_requester_list_new();
		children->refcnt = 0;
		_starpu_spin_init(&children->header_lock);

		children->sequential_consistency = initial_handle->sequential_consistency;

		unsigned node;
		for (node = 0; node < STARPU_MAXNODES; node++)
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
	_starpu_spin_unlock(&initial_handle->header_lock);
}

void starpu_data_unpartition(starpu_data_handle root_handle, uint32_t gathering_node)
{
	unsigned child;
	unsigned node;

	_starpu_spin_lock(&root_handle->header_lock);

#warning starpu_data_unpartition is not supported with NO_DATA_RW_LOCK yet ...

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_handle->nchildren; child++)
	{
		/* make sure the intermediate children is unpartitionned as well */
		if (root_handle->children[child].nchildren > 0)
			starpu_data_unpartition(&root_handle->children[child], gathering_node);

		int ret;
		ret = _starpu_fetch_data_on_node(&root_handle->children[child], gathering_node, 1, 0, 0);
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
	unsigned still_valid[STARPU_MAXNODES];

	/* we do 2 passes : the first pass determines wether the data is still
	 * valid or not, the second pass is needed to choose between STARPU_SHARED and
	 * STARPU_OWNER */

	unsigned nvalids = 0;

	/* still valid ? */
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		/* until an issue is found the data is assumed to be valid */
		unsigned isvalid = 1;

		for (child = 0; child < root_handle->nchildren; child++)
		{
			starpu_local_data_state *local = &root_handle->children[child].per_node[node];

			if (local->state == STARPU_INVALID) {
				isvalid = 0; 
			}
	
			if (local->allocated && local->automatically_allocated){
				/* free the data copy in a lazy fashion */
				_starpu_request_mem_chunk_removal(root_handle, node);
				isvalid = 0; 
			}
		}

		/* no problem was found so the node still has a valid copy */
		still_valid[node] = isvalid;
		nvalids++;
	}

	/* either shared or owned */
	STARPU_ASSERT(nvalids > 0);

	starpu_cache_state newstate = (nvalids == 1)?STARPU_OWNER:STARPU_SHARED;

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		root_handle->per_node[node].state = 
			still_valid[node]?newstate:STARPU_INVALID;
	}

	/* there is no child anymore */
	root_handle->nchildren = 0;

	/* now the parent may be used again so we release the lock */
	_starpu_spin_unlock(&root_handle->header_lock);
}

/* TODO create an alternative version of that function which takes an array of
 * data interface ops in case each child may have its own interface type */
void starpu_data_create_children(starpu_data_handle handle,
		unsigned nchildren, struct starpu_data_interface_ops_t *children_interface_ops)
{
	handle->children = calloc(nchildren, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(handle->children);

	unsigned node;
	unsigned child;

	for (child = 0; child < nchildren; child++)
	{
		starpu_data_handle handle_child = &handle->children[child];

		handle_child->ops = children_interface_ops;

		size_t interfacesize = children_interface_ops->interface_size;

		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			handle_child->interface[node] = calloc(1, interfacesize);
			STARPU_ASSERT(handle->children->interface[node]);
		}
	}

	handle->nchildren = nchildren;
}
