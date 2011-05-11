/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <datawizard/filters.h>
#include <datawizard/footprint.h>

static void starpu_data_create_children(starpu_data_handle handle, unsigned nchildren, struct starpu_data_filter *f);

/*
 * This function applies a data filter on all the elements of a partition
 */
static void map_filter(starpu_data_handle root_handle, struct starpu_data_filter *f)
{
	/* we need to apply the data filter on all leaf of the tree */
	if (root_handle->nchildren == 0)
	{
		/* this is a leaf */
		starpu_data_partition(root_handle, f);
	}
	else {
		/* try to apply the data filter recursively */
		unsigned child;
		for (child = 0; child < root_handle->nchildren; child++)
		{
			map_filter(&root_handle->children[child], f);
		}
	}
}
void starpu_data_map_filters(starpu_data_handle root_handle, unsigned nfilters, ...)
{
	unsigned i;
	va_list pa;
	va_start(pa, nfilters);
	for (i = 0; i < nfilters; i++)
	{
		struct starpu_data_filter *next_filter;
		next_filter = va_arg(pa, struct starpu_data_filter *);

		STARPU_ASSERT(next_filter);

		map_filter(root_handle, next_filter);
	}
	va_end(pa);
}

int starpu_data_get_nb_children(starpu_data_handle handle)
{
        return handle->nchildren;
}

starpu_data_handle starpu_data_get_child(starpu_data_handle handle, unsigned i)
{
	STARPU_ASSERT(i < handle->nchildren);

	return &handle->children[i];
}

/*
 * example starpu_data_get_sub_data(starpu_data_handle root_handle, 3, 42, 0, 1);
 */
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

void starpu_data_partition(starpu_data_handle initial_handle, struct starpu_data_filter *f)
{
	unsigned nparts;
	unsigned i;

	/* first take care to properly lock the data header */
	_starpu_spin_lock(&initial_handle->header_lock);

	/* there should not be mutiple filters applied on the same data */
	STARPU_ASSERT(initial_handle->nchildren == 0);

	/* how many parts ? */
	if (f->get_nchildren)
	  nparts = f->get_nchildren(f, initial_handle);
	else
	  nparts = f->nchildren;

	STARPU_ASSERT(nparts > 0);

	/* allocate the children */
	starpu_data_create_children(initial_handle, nparts, f);

	unsigned nworkers = starpu_worker_get_count();

	for (i = 0; i < nparts; i++)
	{
		starpu_data_handle child =
			starpu_data_get_child(initial_handle, i);

		STARPU_ASSERT(child);

		child->nchildren = 0;
                child->rank = initial_handle->rank;
		child->root_handle = initial_handle->root_handle;
		child->father_handle = initial_handle;
		child->sibling_index = i;
		child->depth = initial_handle->depth + 1;

		child->is_not_important = initial_handle->is_not_important;
		child->wt_mask = initial_handle->wt_mask;
		child->home_node = initial_handle->home_node;
		child->is_readonly = initial_handle->is_readonly;

		/* initialize the chunk lock */
		child->req_list = starpu_data_requester_list_new();
		child->reduction_req_list = starpu_data_requester_list_new();
		child->refcnt = 0;
		_starpu_spin_init(&child->header_lock);

		child->sequential_consistency = initial_handle->sequential_consistency;

		PTHREAD_MUTEX_INIT(&child->sequential_consistency_mutex, NULL);
		child->last_submitted_mode = STARPU_R;
		child->last_submitted_writer = NULL;
		child->last_submitted_readers = NULL;
		child->post_sync_tasks = NULL;
		child->post_sync_tasks_cnt = 0;

		/* The methods used for reduction are propagated to the
		 * children. */
		child->redux_cl = initial_handle->redux_cl;
		child->init_cl = initial_handle->init_cl;

		child->reduction_refcnt = 0;
		child->reduction_req_list = starpu_data_requester_list_new();

#ifdef STARPU_USE_FXT
		child->last_submitted_ghost_writer_id_is_valid = 0;
		child->last_submitted_ghost_writer_id = 0;
		child->last_submitted_ghost_readers_id = NULL;
#endif

		unsigned node;
		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			struct starpu_data_replicate_s *initial_replicate; 
			struct starpu_data_replicate_s *child_replicate;

			initial_replicate = &initial_handle->per_node[node];
			child_replicate = &child->per_node[node];

			child_replicate->state = initial_replicate->state;
			child_replicate->allocated = initial_replicate->allocated;
			child_replicate->automatically_allocated = initial_replicate->automatically_allocated;
			child_replicate->refcnt = 0;
			child_replicate->memory_node = node;
			child_replicate->relaxed_coherency = 0;
			
			/* update the interface */
			void *initial_interface = starpu_data_get_interface_on_node(initial_handle, node);
			void *child_interface = starpu_data_get_interface_on_node(child, node);

			f->filter_func(initial_interface, child_interface, f, i, nparts);
		}

		unsigned worker;
		for (worker = 0; worker < nworkers; worker++)
		{
			struct starpu_data_replicate_s *child_replicate;
			child_replicate = &child->per_worker[worker];
			
			child_replicate->state = STARPU_INVALID;
			child_replicate->allocated = 0;
			child_replicate->automatically_allocated = 0;
			child_replicate->refcnt = 0;
			child_replicate->memory_node = starpu_worker_get_memory_node(worker);

			for (node = 0; node < STARPU_MAXNODES; node++)
			{
				child_replicate->requested[node] = 0;
				child_replicate->request[node] = NULL;
			}

			child_replicate->relaxed_coherency = 1;
			child_replicate->initialized = 0;

			/* duplicate  the content of the interface on node 0 */
			memcpy(child_replicate->data_interface, child->per_node[0].data_interface, child->ops->interface_size);
		}

		/* We compute the size and the footprint of the child once and
		 * store it in the handle */
		child->data_size = child->ops->get_size(child);
		child->footprint = _starpu_compute_data_footprint(child);

		void *ptr;
		ptr = starpu_handle_to_pointer(child, 0);
		if (ptr != NULL)
		{
			_starpu_data_register_ram_pointer(child, ptr);
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

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_handle->nchildren; child++)
	{
		struct starpu_data_state_t *child_handle = &root_handle->children[child];

		/* make sure the intermediate children is unpartitionned as well */
		if (child_handle->nchildren > 0)
			starpu_data_unpartition(child_handle, gathering_node);

		int ret;
		ret = _starpu_fetch_data_on_node(child_handle, &child_handle->per_node[gathering_node], STARPU_R, 0, NULL, NULL);
		/* for now we pretend that the RAM is almost unlimited and that gathering 
		 * data should be possible from the node that does the unpartionning ... we
		 * don't want to have the programming deal with memory shortage at that time,
		 * really */
		STARPU_ASSERT(ret == 0); 

		_starpu_data_free_interfaces(&root_handle->children[child]);
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
			struct starpu_data_replicate_s *local = &root_handle->children[child].per_node[node];

			if (local->state == STARPU_INVALID) {
				isvalid = 0; 
			}
	
			if (local->allocated && local->automatically_allocated){
				/* free the data copy in a lazy fashion */
				_starpu_request_mem_chunk_removal(root_handle, node);
				isvalid = 0; 
			}
#ifdef STARPU_DEVEL
#warning free the data replicate if needed
#endif

		}

		/* if there was no invalid copy, the node still has a valid copy */
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

/* each child may have his own interface type */
static void starpu_data_create_children(starpu_data_handle handle, unsigned nchildren, struct starpu_data_filter *f)
{
	handle->children = calloc(nchildren, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(handle->children);

	unsigned node;
	unsigned worker;
	unsigned child;

	unsigned nworkers = starpu_worker_get_count();

	for (child = 0; child < nchildren; child++)
	{
		starpu_data_handle handle_child = &handle->children[child];
		
		struct starpu_data_interface_ops_t *ops;
		
		/* what's this child's interface ? */
		if (f->get_child_ops)
		  ops = f->get_child_ops(f, child);
		else
		  ops = handle->ops;
		
		handle_child->ops = ops;

		size_t interfacesize = ops->interface_size;

		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			/* relaxed_coherency = 0 */
			handle_child->per_node[node].handle = handle_child;
			handle_child->per_node[node].data_interface = calloc(1, interfacesize);
			STARPU_ASSERT(handle_child->per_node[node].data_interface);
		}

		for (worker = 0; worker < nworkers; worker++)
		{
			handle_child->per_worker[worker].handle = handle_child;
			handle_child->per_worker[worker].data_interface = calloc(1, interfacesize);
			STARPU_ASSERT(handle_child->per_worker[worker].data_interface);
		}
	}
	
	/* this handle now has children */
	handle->nchildren = nchildren;
}
