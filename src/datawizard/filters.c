/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011                                     Antoine Lucas
 * Copyright (C) 2011-2012,2016                           Inria
 * Copyright (C) 2008-2017,2019                           Université de Bordeaux
 * Copyright (C) 2010                                     Mehdi Juhoor
 * Copyright (C) 2010-2013,2015-2017                      CNRS
 * Copyright (C) 2013                                     Thibaut Lambert
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
#include <datawizard/interfaces/data_interface.h>
#include <core/task.h>

/*
 * This function applies a data filter on all the elements of a partition
 */
static void map_filter(starpu_data_handle_t root_handle, struct starpu_data_filter *f)
{
	/* we need to apply the data filter on all leaf of the tree */
	if (root_handle->nchildren == 0)
	{
		/* this is a leaf */
		starpu_data_partition(root_handle, f);
	}
	else
	{
		/* try to apply the data filter recursively */
		unsigned child;
		for (child = 0; child < root_handle->nchildren; child++)
		{
			starpu_data_handle_t handle_child = starpu_data_get_child(root_handle, child);
			map_filter(handle_child, f);
		}
	}
}
void starpu_data_vmap_filters(starpu_data_handle_t root_handle, unsigned nfilters, va_list pa)
{
	unsigned i;
	for (i = 0; i < nfilters; i++)
	{
		struct starpu_data_filter *next_filter;
		next_filter = va_arg(pa, struct starpu_data_filter *);

		STARPU_ASSERT(next_filter);

		map_filter(root_handle, next_filter);
	}
}

void starpu_data_map_filters(starpu_data_handle_t root_handle, unsigned nfilters, ...)
{
	va_list pa;
	va_start(pa, nfilters);
	starpu_data_vmap_filters(root_handle, nfilters, pa);
	va_end(pa);
}

void fstarpu_data_map_filters(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter **filters)
{
	int i;
	assert(nfilters >= 0);
	for (i = 0; i < nfilters; i++)
	{
		struct starpu_data_filter *next_filter = filters[i];
		STARPU_ASSERT(next_filter);
		map_filter(root_handle, next_filter);
	}
}

int starpu_data_get_nb_children(starpu_data_handle_t handle)
{
        return handle->nchildren;
}

starpu_data_handle_t starpu_data_get_child(starpu_data_handle_t handle, unsigned i)
{
	STARPU_ASSERT_MSG(handle->nchildren != 0, "Data %p has to be partitioned before accessing children", handle);
	STARPU_ASSERT_MSG(i < handle->nchildren, "Invalid child index %u in handle %p, maximum %u", i, handle, handle->nchildren);
	return &handle->children[i];
}

/*
 * example starpu_data_get_sub_data(starpu_data_handle_t root_handle, 3, 42, 0, 1);
 */
starpu_data_handle_t starpu_data_get_sub_data(starpu_data_handle_t root_handle, unsigned depth, ... )
{
	va_list pa;
	va_start(pa, depth);
	starpu_data_handle_t handle = starpu_data_vget_sub_data(root_handle, depth, pa);
	va_end(pa);

	return handle;
}

starpu_data_handle_t starpu_data_vget_sub_data(starpu_data_handle_t root_handle, unsigned depth, va_list pa )
{
	STARPU_ASSERT(root_handle);
	starpu_data_handle_t current_handle = root_handle;

	/* the variable number of argument must correlate the depth in the tree */
	unsigned i;
	for (i = 0; i < depth; i++)
	{
		unsigned next_child;
		next_child = va_arg(pa, unsigned);

		STARPU_ASSERT_MSG(current_handle->nchildren != 0, "Data %p has to be partitioned before accessing children", current_handle);
		STARPU_ASSERT_MSG(next_child < current_handle->nchildren, "Bogus child number %u, data %p only has %u children", next_child, current_handle, current_handle->nchildren);

		current_handle = &current_handle->children[next_child];
	}

	return current_handle;
}

starpu_data_handle_t fstarpu_data_get_sub_data(starpu_data_handle_t root_handle, int depth, int *indices)
{
	STARPU_ASSERT(root_handle);
	starpu_data_handle_t current_handle = root_handle;

	STARPU_ASSERT(depth >= 0);
	/* the variable number of argument must correlate the depth in the tree */
	int i;
	for (i = 0; i < depth; i++)
	{
		int next_child;
		next_child = indices[i];
		STARPU_ASSERT(next_child >= 0);

		STARPU_ASSERT_MSG(current_handle->nchildren != 0, "Data %p has to be partitioned before accessing children", current_handle);
		STARPU_ASSERT_MSG((unsigned) next_child < current_handle->nchildren, "Bogus child number %d, data %p only has %u children", next_child, current_handle, current_handle->nchildren);

		current_handle = &current_handle->children[next_child];
	}

	return current_handle;
}

static unsigned _starpu_data_partition_nparts(starpu_data_handle_t initial_handle, struct starpu_data_filter *f)
{
	/* how many parts ? */
	if (f->get_nchildren)
	  return f->get_nchildren(f, initial_handle);
	else
	  return f->nchildren;

}

static void _starpu_data_partition(starpu_data_handle_t initial_handle, starpu_data_handle_t *childrenp, unsigned nparts, struct starpu_data_filter *f, int inherit_state)
{
	unsigned i;
	unsigned node;

	/* first take care to properly lock the data header */
	_starpu_spin_lock(&initial_handle->header_lock);

	initial_handle->nplans++;

	STARPU_ASSERT_MSG(nparts > 0, "Partitioning data %p in 0 piece does not make sense", initial_handle);

	/* allocate the children */
	if (inherit_state)
	{
		_STARPU_CALLOC(initial_handle->children, nparts, sizeof(struct _starpu_data_state));
		/* this handle now has children */
		initial_handle->nchildren = nparts;
	}

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (initial_handle->per_node[node].state != STARPU_INVALID)
			break;
	}
	if (node == STARPU_MAXNODES)
	{
		/* This is lazy allocation, allocate it now in main RAM, so as
		 * to have somewhere to gather pieces later */
		/* FIXME: mark as unevictable! */
		int ret = _starpu_allocate_memory_on_node(initial_handle, &initial_handle->per_node[STARPU_MAIN_RAM], 0);
#ifdef STARPU_DEVEL
#warning we should reclaim memory if allocation failed
#endif
		STARPU_ASSERT(!ret);
	}

	_starpu_data_unregister_ram_pointer(initial_handle);

	if (nparts && !inherit_state)
	{
		STARPU_ASSERT_MSG(childrenp, "Passing NULL pointer for parameter childrenp while parameter inherit_state is 0");
	}

	for (i = 0; i < nparts; i++)
	{
		starpu_data_handle_t child;

		if (inherit_state)
			child = &initial_handle->children[i];
		else
			child = childrenp[i];
		STARPU_ASSERT(child);

		struct starpu_data_interface_ops *ops;

		/* each child may have his own interface type */
		/* what's this child's interface ? */
		if (f->get_child_ops)
			ops = f->get_child_ops(f, i);
		else
			ops = initial_handle->ops;

		_starpu_data_handle_init(child, ops, initial_handle->mf_node);

		child->nchildren = 0;
		child->nplans = 0;
		child->switch_cl = NULL;
		child->partitioned = 0;
		child->readonly = 0;
                child->mpi_data = initial_handle->mpi_data;
		child->root_handle = initial_handle->root_handle;
		child->father_handle = initial_handle;
		child->sibling_index = i;
		child->depth = initial_handle->depth + 1;

		child->is_not_important = initial_handle->is_not_important;
		child->wt_mask = initial_handle->wt_mask;
		child->home_node = initial_handle->home_node;

		/* initialize the chunk lock */
		_starpu_data_requester_list_init(&child->req_list);
		_starpu_data_requester_list_init(&child->reduction_req_list);
		child->reduction_tmp_handles = NULL;
		child->write_invalidation_req = NULL;
		child->refcnt = 0;
		child->unlocking_reqs = 0;
		child->busy_count = 0;
		child->busy_waiting = 0;
		STARPU_PTHREAD_MUTEX_INIT(&child->busy_mutex, NULL);
		STARPU_PTHREAD_COND_INIT(&child->busy_cond, NULL);
		child->reduction_refcnt = 0;
		_starpu_spin_init(&child->header_lock);

		child->sequential_consistency = initial_handle->sequential_consistency;
		child->initialized = initial_handle->initialized;
		child->ooc = initial_handle->ooc;

		STARPU_PTHREAD_MUTEX_INIT(&child->sequential_consistency_mutex, NULL);
		child->last_submitted_mode = STARPU_R;
		child->last_sync_task = NULL;
		child->last_submitted_accessors.task = NULL;
		child->last_submitted_accessors.next = &child->last_submitted_accessors;
		child->last_submitted_accessors.prev = &child->last_submitted_accessors;
		child->post_sync_tasks = NULL;
		/* Tell helgrind that the race in _starpu_unlock_post_sync_tasks is fine */
		STARPU_HG_DISABLE_CHECKING(child->post_sync_tasks_cnt);
		child->post_sync_tasks_cnt = 0;

		/* The methods used for reduction are propagated to the
		 * children. */
		child->redux_cl = initial_handle->redux_cl;
		child->init_cl = initial_handle->init_cl;

#ifdef STARPU_USE_FXT
		child->last_submitted_ghost_sync_id_is_valid = 0;
		child->last_submitted_ghost_sync_id = 0;
		child->last_submitted_ghost_accessors_id = NULL;
#endif

		if (_starpu_global_arbiter)
			/* Just for testing purpose */
			starpu_data_assign_arbiter(child, _starpu_global_arbiter);
		else
			child->arbiter = NULL;
		_starpu_data_requester_list_init(&child->arbitered_req_list);

		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			struct _starpu_data_replicate *initial_replicate;
			struct _starpu_data_replicate *child_replicate;

			initial_replicate = &initial_handle->per_node[node];
			child_replicate = &child->per_node[node];

			if (inherit_state)
				child_replicate->state = initial_replicate->state;
			else
				child_replicate->state = STARPU_INVALID;
			if (inherit_state || !initial_replicate->automatically_allocated)
				child_replicate->allocated = initial_replicate->allocated;
			else
				child_replicate->allocated = 0;
			/* Do not allow memory reclaiming within the child for parent bits */
			child_replicate->automatically_allocated = 0;
			child_replicate->refcnt = 0;
			child_replicate->memory_node = node;
			child_replicate->relaxed_coherency = 0;
			if (inherit_state)
				child_replicate->initialized = initial_replicate->initialized;
			else
				child_replicate->initialized = 0;

			/* update the interface */
			void *initial_interface = starpu_data_get_interface_on_node(initial_handle, node);
			void *child_interface = starpu_data_get_interface_on_node(child, node);

			STARPU_ASSERT_MSG(!(!inherit_state && child_replicate->automatically_allocated && child_replicate->allocated), "partition planning is currently not supported when handle has some automatically allocated buffers");
			f->filter_func(initial_interface, child_interface, f, i, nparts);
		}

		child->per_worker = NULL;
		child->user_data = NULL;

		/* We compute the size and the footprint of the child once and
		 * store it in the handle */
		child->footprint = _starpu_compute_data_footprint(child);

		void *ptr;
		ptr = starpu_data_handle_to_pointer(child, STARPU_MAIN_RAM);
		if (ptr != NULL)
			_starpu_data_register_ram_pointer(child, ptr);

		_STARPU_TRACE_HANDLE_DATA_REGISTER(child);
	}
	/* now let the header */
	_starpu_spin_unlock(&initial_handle->header_lock);
}

static
void _starpu_empty_codelet_function(void *buffers[], void *args)
{
	(void) buffers; // unused;
	(void) args; // unused;
}

void starpu_data_unpartition(starpu_data_handle_t root_handle, unsigned gathering_node)
{
	unsigned child;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	unsigned node;
	unsigned sizes[root_handle->nchildren];
	void *ptr;

	_STARPU_TRACE_START_UNPARTITION(root_handle, gathering_node);
	_starpu_spin_lock(&root_handle->header_lock);

	STARPU_ASSERT_MSG(root_handle->nchildren != 0, "data %p is not partitioned, can not unpartition it", root_handle);

	/* first take all the children lock (in order !) */
	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);

		/* make sure the intermediate children is unpartitionned as well */
		if (child_handle->nchildren > 0)
			starpu_data_unpartition(child_handle, gathering_node);

		/* If this is a multiformat handle, we must convert the data now */
#ifdef STARPU_DEVEL
#warning TODO: _starpu_fetch_data_on_node should be doing it
#endif
		if (_starpu_data_is_multiformat_handle(child_handle) &&
			starpu_node_get_kind(child_handle->mf_node) != STARPU_CPU_RAM)
		{
			struct starpu_codelet cl =
			{
				.where = STARPU_CPU,
				.cpu_funcs = { _starpu_empty_codelet_function },
				.modes = { STARPU_RW },
				.nbuffers = 1
			};
			struct starpu_task *task = starpu_task_create();
			task->name = "convert_data";

			STARPU_TASK_SET_HANDLE(task, child_handle, 0);
			task->cl = &cl;
			task->synchronous = 1;
			if (_starpu_task_submit_internally(task) != 0)
				_STARPU_ERROR("Could not submit the conversion task while unpartitionning\n");
		}

		int ret;
		/* for now we pretend that the RAM is almost unlimited and that gathering
		 * data should be possible from the node that does the unpartionning ... we
		 * don't want to have the programming deal with memory shortage at that time,
		 * really */
		/* Acquire the child data on the gathering node. This will trigger collapsing any reduction */
		ret = starpu_data_acquire_on_node(child_handle, gathering_node, STARPU_RW);
		STARPU_ASSERT(ret == 0);
		starpu_data_release_on_node(child_handle, gathering_node);

		_starpu_spin_lock(&child_handle->header_lock);
		child_handle->busy_waiting = 1;
		_starpu_spin_unlock(&child_handle->header_lock);

		/* Wait for all requests to finish (notably WT requests) */
		STARPU_PTHREAD_MUTEX_LOCK(&child_handle->busy_mutex);
		while (1)
		{
			/* Here helgrind would shout that this an unprotected access,
			 * but this is actually fine: all threads who do busy_count--
			 * are supposed to call _starpu_data_check_not_busy, which will
			 * wake us up through the busy_mutex/busy_cond. */
			if (!child_handle->busy_count)
				break;
			/* This is woken by _starpu_data_check_not_busy, always called
			 * after decrementing busy_count */
			STARPU_PTHREAD_COND_WAIT(&child_handle->busy_cond, &child_handle->busy_mutex);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&child_handle->busy_mutex);

		_starpu_spin_lock(&child_handle->header_lock);

		sizes[child] = _starpu_data_get_size(child_handle);

		_starpu_data_unregister_ram_pointer(child_handle);

		if (child_handle->per_worker)
		for (worker = 0; worker < nworkers; worker++)
		{
			struct _starpu_data_replicate *local = &child_handle->per_worker[worker];
			STARPU_ASSERT(local->state == STARPU_INVALID);
			if (local->allocated && local->automatically_allocated)
				_starpu_request_mem_chunk_removal(child_handle, local, starpu_worker_get_memory_node(worker), sizes[child]);
		}

		_starpu_memory_stats_free(child_handle);
	}

	ptr = starpu_data_handle_to_pointer(root_handle, STARPU_MAIN_RAM);
	if (ptr != NULL)
		_starpu_data_register_ram_pointer(root_handle, ptr);

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
		struct _starpu_data_replicate *local;
		/* until an issue is found the data is assumed to be valid */
		unsigned isvalid = 1;

		for (child = 0; child < root_handle->nchildren; child++)
		{
			starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
			local = &child_handle->per_node[node];

			if (local->state == STARPU_INVALID || local->automatically_allocated == 1)
			{
				/* One of the bits is missing or is not inside the parent */
				isvalid = 0;
			}

			if (local->mc && local->allocated && local->automatically_allocated)
				/* free the child data copy in a lazy fashion */
				_starpu_request_mem_chunk_removal(child_handle, local, node, sizes[child]);
		}

		local = &root_handle->per_node[node];

		if (!local->allocated)
			/* Even if we have all the bits, if we don't have the
			 * whole data, it's not valid */
			isvalid = 0;

		if (!isvalid && local->mc && local->allocated && local->automatically_allocated)
			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(root_handle, local, node, _starpu_data_get_size(root_handle));

		/* if there was no invalid copy, the node still has a valid copy */
		still_valid[node] = isvalid;
		if (isvalid)
			nvalids++;
	}

	/* either shared or owned */
	STARPU_ASSERT(nvalids > 0);

	enum _starpu_cache_state newstate = (nvalids == 1)?STARPU_OWNER:STARPU_SHARED;

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		root_handle->per_node[node].state =
			still_valid[node]?newstate:STARPU_INVALID;
	}

	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
		_starpu_data_free_interfaces(child_handle);
		_starpu_spin_unlock(&child_handle->header_lock);
		_starpu_spin_destroy(&child_handle->header_lock);
	}

	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
		_starpu_data_clear_implicit(child_handle);
		STARPU_PTHREAD_MUTEX_DESTROY(&child_handle->busy_mutex);
		STARPU_PTHREAD_COND_DESTROY(&child_handle->busy_cond);
		STARPU_PTHREAD_MUTEX_DESTROY(&child_handle->sequential_consistency_mutex);

		_STARPU_TRACE_HANDLE_DATA_UNREGISTER(child_handle);
	}

	/* there is no child anymore */
	starpu_data_handle_t children = root_handle->children;
	root_handle->children = NULL;
	root_handle->nchildren = 0;
	root_handle->nplans--;

	/* now the parent may be used again so we release the lock */
	_starpu_spin_unlock(&root_handle->header_lock);

	free(children);

	_STARPU_TRACE_END_UNPARTITION(root_handle, gathering_node);
}

void starpu_data_partition(starpu_data_handle_t initial_handle, struct starpu_data_filter *f)
{
	unsigned nparts = _starpu_data_partition_nparts(initial_handle, f);
	STARPU_ASSERT_MSG(initial_handle->nchildren == 0, "there should not be mutiple filters applied on the same data %p, futher filtering has to be done on children", initial_handle);
	STARPU_ASSERT_MSG(initial_handle->nplans == 0, "partition planning and synchronous partitioning is not supported");

	initial_handle->children = NULL;

	/* Make sure to wait for previous tasks working on the whole data */
	starpu_data_acquire_on_node(initial_handle, STARPU_ACQUIRE_NO_NODE, initial_handle->initialized?STARPU_RW:STARPU_W);
	starpu_data_release_on_node(initial_handle, STARPU_ACQUIRE_NO_NODE);

	_starpu_data_partition(initial_handle, NULL, nparts, f, 1);
}

void starpu_data_partition_plan(starpu_data_handle_t initial_handle, struct starpu_data_filter *f, starpu_data_handle_t *childrenp)
{
	unsigned i;
	unsigned nparts = _starpu_data_partition_nparts(initial_handle, f);
	STARPU_ASSERT_MSG(initial_handle->nchildren == 0, "partition planning and synchronous partitioning is not supported");
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	struct starpu_codelet *cl = initial_handle->switch_cl;
	int home_node = initial_handle->home_node;
	if (home_node == -1)
		/* Nothing better for now */
		/* TODO: pass -1, and make _starpu_fetch_nowhere_task_input
		 * really call _starpu_fetch_data_on_node, and make that update
		 * the coherency.
		 */
		home_node = STARPU_MAIN_RAM;

	for (i = 0; i < nparts; i++)
		_STARPU_CALLOC(childrenp[i], 1, sizeof(struct _starpu_data_state));
	_starpu_data_partition(initial_handle, childrenp, nparts, f, 0);

	if (!cl)
	{
		/* Create a codelet that will make the coherency on the home node */
		_STARPU_CALLOC(initial_handle->switch_cl, 1, sizeof(*initial_handle->switch_cl));
		cl = initial_handle->switch_cl;
		cl->where = STARPU_NOWHERE;
		cl->nbuffers = STARPU_VARIABLE_NBUFFERS;
		cl->name = "data_partition_switch";
		cl->specific_nodes = 1;
	}
	if (initial_handle->switch_cl_nparts < nparts)
	{
		/* First initialization, or previous initialization was with fewer parts, enlarge it */
		_STARPU_REALLOC(cl->dyn_nodes, (nparts+1) * sizeof(*cl->dyn_nodes));
		for (i = initial_handle->switch_cl_nparts; i < nparts+1; i++)
			cl->dyn_nodes[i] = home_node;
		initial_handle->switch_cl_nparts = nparts;
	}
}

void starpu_data_partition_clean(starpu_data_handle_t root_handle, unsigned nparts, starpu_data_handle_t *children)
{
	unsigned i;

	for (i = 0; i < nparts; i++)
		starpu_data_unregister_submit(children[i]);

	_starpu_spin_lock(&root_handle->header_lock);
	root_handle->nplans--;
	_starpu_spin_unlock(&root_handle->header_lock);
}

void starpu_data_partition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 0, "One can't submit several partition plannings at the same time");
	STARPU_ASSERT_MSG(initial_handle->readonly == 0, "One can't submit a partition planning while a readonly partitioning is active");
	initial_handle->partitioned++;
	_starpu_spin_unlock(&initial_handle->header_lock);

	if (!initial_handle->initialized)
		/* No need for coherency, it is not initialized */
		return;
	unsigned i;
	struct starpu_data_descr descr[nparts];
	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->father_handle == initial_handle, "children parameter of starpu_data_partition_submit must be the children of the parent parameter");
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
	}
	/* TODO: assert nparts too */
	starpu_task_insert(initial_handle->switch_cl, STARPU_RW, initial_handle, STARPU_DATA_MODE_ARRAY, descr, nparts, 0);
	starpu_data_invalidate_submit(initial_handle);
}

void starpu_data_partition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 0 || initial_handle->readonly, "One can't submit a readonly partition planning at the same time as a readwrite partition planning");
	initial_handle->partitioned++;
	initial_handle->readonly = 1;
	_starpu_spin_unlock(&initial_handle->header_lock);

	STARPU_ASSERT_MSG(initial_handle->initialized, "It is odd to read-only-partition a data which does not have a value yet");
	unsigned i;
	struct starpu_data_descr descr[nparts];
	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->father_handle == initial_handle, "children parameter of starpu_data_partition_submit must be the children of the parent parameter");
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
	}
	/* TODO: assert nparts too */
	starpu_task_insert(initial_handle->switch_cl, STARPU_R, initial_handle, STARPU_DATA_MODE_ARRAY, descr, nparts, 0);
}

void starpu_data_partition_readwrite_upgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 1, "One can't upgrade a readonly partition planning to readwrite while other readonly partition plannings are active");
	STARPU_ASSERT_MSG(initial_handle->readonly == 1, "One can only upgrade a readonly partition planning");
	initial_handle->readonly = 0;
	_starpu_spin_unlock(&initial_handle->header_lock);

	unsigned i;
	struct starpu_data_descr descr[nparts];
	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->father_handle == initial_handle, "children parameter of starpu_data_partition_submit must be the children of the parent parameter");
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
	}
	/* TODO: assert nparts too */
	starpu_task_insert(initial_handle->switch_cl, STARPU_RW, initial_handle, STARPU_DATA_MODE_ARRAY, descr, nparts, 0);
	starpu_data_invalidate_submit(initial_handle);
}

void starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	STARPU_ASSERT_MSG(gather_node == initial_handle->home_node || gather_node == -1, "gathering node different from home node is currently not supported");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned >= 1, "No partition planning is active for this handle");
	initial_handle->partitioned--;
	if (!initial_handle->partitioned)
		initial_handle->readonly = 0;
	_starpu_spin_unlock(&initial_handle->header_lock);

	unsigned i, n;
	struct starpu_data_descr descr[nparts];
	for (i = 0, n = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->father_handle == initial_handle, "children parameter of starpu_data_partition_submit must be the children of the parent parameter");
		if (!children[i]->initialized)
			/* Dropped value, do not care about coherency for this one */
			continue;
		descr[n].handle = children[i];
		descr[n].mode = STARPU_RW;
		n++;
	}
	/* TODO: assert nparts too */
	starpu_task_insert(initial_handle->switch_cl, STARPU_W, initial_handle, STARPU_DATA_MODE_ARRAY, descr, n, 0);
	for (i = 0; i < nparts; i++)
		starpu_data_invalidate_submit(children[i]);
}

void starpu_data_unpartition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	STARPU_ASSERT_MSG(gather_node == initial_handle->home_node || gather_node == -1, "gathering node different from home node is currently not supported");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned >= 1, "No partition planning is active for this handle");
	initial_handle->readonly = 1;
	_starpu_spin_unlock(&initial_handle->header_lock);

	unsigned i, n;
	struct starpu_data_descr descr[nparts];
	for (i = 0, n = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->father_handle == initial_handle, "children parameter of starpu_data_partition_submit must be the children of the parent parameter");
		if (!children[i]->initialized)
			/* Dropped value, do not care about coherency for this one */
			continue;
		descr[n].handle = children[i];
		descr[n].mode = STARPU_R;
		n++;
	}
	/* TODO: assert nparts too */
	starpu_task_insert(initial_handle->switch_cl, STARPU_W, initial_handle, STARPU_DATA_MODE_ARRAY, descr, n, 0);
}

/*
 * Given an integer N, NPARTS the number of parts it must be divided in, ID the
 * part currently considered, determines the CHUNK_SIZE and the OFFSET, taking
 * into account the size of the elements stored in the data structure ELEMSIZE
 * and LD, the leading dimension.
 */
void
_starpu_filter_nparts_compute_chunk_size_and_offset(unsigned n, unsigned nparts,
					     size_t elemsize, unsigned id,
					     unsigned ld, unsigned *chunk_size,
					     size_t *offset)
{
	*chunk_size = n/nparts;
	unsigned remainder = n % nparts;
	if (id < remainder)
		(*chunk_size)++;
	/*
	 * Computing the total offset. The formula may not be really clear, but
	 * it really just is:
	 *
	 * total = 0;
	 * for (i = 0; i < id; i++)
	 * {
	 * 	total += n/nparts;
	 * 	if (i < n%nparts)
	 *		total++;
	 * }
	 * offset = total * elemsize * ld;
	 */
	if (offset != NULL)
		*offset = (id *(n/nparts) + STARPU_MIN(remainder, id)) * ld * elemsize;
}
