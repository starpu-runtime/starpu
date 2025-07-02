/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021-2021  Federal University of Rio Grande do Sul (UFRGS)
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2010-2010  Mehdi Juhoor
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

//#define STARPU_VERBOSE

#include <datawizard/filters.h>
#include <datawizard/footprint.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/memory_nodes.h>
#include <core/task.h>

void starpu_data_set_gathering_node(starpu_data_handle_t handle, unsigned node)
{
	STARPU_ASSERT_MSG(!handle->nplans && !handle->nchildren, "starpu_data_set_gathering_node can only be called before partitioning the data");
	handle->gathering_node = node;
}

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

void starpu_data_map_filters_parray(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter **filters)
{
	int i;
	STARPU_ASSERT(nfilters >= 0);
	for (i = 0; i < nfilters; i++)
	{
		struct starpu_data_filter *next_filter = filters[i];
		STARPU_ASSERT(next_filter);
		map_filter(root_handle, next_filter);
	}
}

void starpu_data_map_filters_array(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter *filters)
{
	int i;
	STARPU_ASSERT(nfilters >= 0);
	for (i = 0; i < nfilters; i++)
	{
		map_filter(root_handle, &filters[i]);
	}
}

void fstarpu_data_map_filters(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter **filters)
{
	starpu_data_map_filters_parray(root_handle, nfilters, filters);
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
starpu_data_handle_t starpu_data_get_sub_data(starpu_data_handle_t root_handle, unsigned depth, ...)
{
	va_list pa;
	va_start(pa, depth);
	starpu_data_handle_t handle = starpu_data_vget_sub_data(root_handle, depth, pa);
	va_end(pa);

	return handle;
}

starpu_data_handle_t starpu_data_vget_sub_data(starpu_data_handle_t root_handle, unsigned depth, va_list pa)
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

	for (node = 0; node < STARPU_MAXNODES; node++)
		_starpu_data_unmap(initial_handle, node);

	/* first take care to properly lock the data header */
	_starpu_spin_lock(&initial_handle->header_lock);

	if (initial_handle->gathering_node == -1)
		/* This handle will gather data, decide for good where */
		initial_handle->gathering_node = _starpu_data_get_gathering_node(initial_handle);
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
		STARPU_ASSERT(initial_handle->per_node[node].mapped == STARPU_UNMAPPED);

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

		/* As most of the fields must be initialized at NULL, let's put
		 * 0 everywhere */
		memset(child, 0, sizeof(*child));
		_starpu_data_handle_init(child, ops, initial_handle->mf_node);

		child->root_handle = initial_handle->root_handle;
		child->parent_handle = initial_handle;
		child->filter = f;

		child->nsiblings = nparts;
		if (inherit_state)
		{
			//child->siblings = NULL;
		}
		else
			child->siblings = childrenp;
		child->sibling_index = i;
		child->depth = initial_handle->depth + 1;

		child->active = inherit_state;

		child->home_node = initial_handle->home_node;
		child->gathering_node = initial_handle->gathering_node;
		child->wt_mask = initial_handle->wt_mask;

		child->aliases = initial_handle->aliases;
		//child->readonly_dup = NULL;
		//child->readonly_dup_of = NULL;

		child->is_not_important = initial_handle->is_not_important;

		child->sequential_consistency = initial_handle->sequential_consistency;
		child->initialized = initial_handle->initialized;
		child->readonly = initial_handle->readonly;
		child->ooc = initial_handle->ooc;

		/* The methods used for reduction are propagated to the
		 * children. */
		child->redux_cl = initial_handle->redux_cl;
		child->redux_cl_arg = initial_handle->redux_cl_arg;
		child->init_cl = initial_handle->init_cl;
		child->init_cl_arg = initial_handle->init_cl_arg;

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
			{
				//child_replicate->allocated = 0;
			}
			/* Do not allow memory reclaiming within the child for parent bits */
			//child_replicate->automatically_allocated = 0;
			//child_replicate->refcnt = 0;
			child_replicate->memory_node = node;
			//child_replicate->relaxed_coherency = 0;
			child_replicate->mapped = STARPU_UNMAPPED;
			if (inherit_state)
				child_replicate->initialized = initial_replicate->initialized;
			else
			{
				//child_replicate->initialized = 0;
			}
			//child_replicate->nb_tasks_prefetch = 0;

			/* update the interface */
			void *initial_interface = starpu_data_get_interface_on_node(initial_handle, node);
			void *child_interface = starpu_data_get_interface_on_node(child, node);

			f->filter_func(initial_interface, child_interface, f, i, nparts);
		}

		/* We compute the size and the footprint of the child once and
		 * store it in the handle */
		child->footprint = _starpu_compute_data_footprint(child);

		_starpu_trace_handle_data_register(&child);
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

	_starpu_trace_start_unpartition(&root_handle, gathering_node);

	STARPU_ASSERT_MSG(root_handle->nchildren != 0, "data %p is not partitioned, can not unpartition it", root_handle);

	/* first gather data from all the children */
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

		/* Make sure it is not mapped */
		for (node = 0; node < STARPU_MAXNODES; node++)
			_starpu_data_unmap(child_handle, node);
	}

	/* Wait for all requests to finish (notably WT and UNMAP requests) */
	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);

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
	}

	/* take all the locks (in order !) */
	_starpu_spin_lock(&root_handle->header_lock);
	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);

		_starpu_spin_lock(&child_handle->header_lock);

		sizes[child] = _starpu_data_get_alloc_size(child_handle);

		struct _starpu_unregister_hook_func *a;
		for (a  = _starpu_unregister_hook_func_list_begin(&child_handle->unregister_hook);
		     a != _starpu_unregister_hook_func_list_end(&child_handle->unregister_hook);
		     a  = _starpu_unregister_hook_func_list_next(a))
			a->hook_func(child_handle);

		if (child_handle->per_worker)
		{
			for (worker = 0; worker < nworkers; worker++)
			{
				struct _starpu_data_replicate *local = &child_handle->per_worker[worker];
				STARPU_ASSERT(local->state == STARPU_INVALID);
				if (local->allocated && local->automatically_allocated)
					_starpu_request_mem_chunk_removal(child_handle, local, starpu_worker_get_memory_node(worker), sizes[child]);
			}
		}

		_starpu_memory_stats_free(child_handle);
	}

	/* the gathering_node should now have a valid copy of all the children.
	 * For all nodes, if the node had all copies and none was locally
	 * allocated then the data is still valid there, else, it's invalidated
	 * for the gathering node, if we have some locally allocated data, we
	 * copy all the children (XXX this should not happen so we just do not
	 * do anything since this is transparent ?) */
	unsigned still_valid[STARPU_MAXNODES];

	/* we do 2 passes : the first pass determines whether the data is still
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

		if (!isvalid && local->mc && local->allocated && local->automatically_allocated && !local->refcnt)
			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(root_handle, local, node, _starpu_data_get_alloc_size(root_handle));

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
		root_handle->per_node[node].state = still_valid[node]?newstate:STARPU_INVALID;
	}

	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
		_starpu_data_free_interfaces(child_handle);
		_starpu_spin_unlock(&child_handle->header_lock);
		_starpu_spin_destroy(&child_handle->header_lock);
	}

	/* Set the initialized state */
	starpu_data_handle_t first_child = starpu_data_get_child(root_handle, 0);
	root_handle->initialized = first_child->initialized;
	for (child = 1; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
		STARPU_ASSERT_MSG(child_handle->initialized == root_handle->initialized, "Inconsistent state between children initialization");
	}
	if (root_handle->initialized)
	{
		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			struct _starpu_data_replicate *root_replicate;

			root_replicate = &root_handle->per_node[node];
			root_replicate->initialized = still_valid[node];
		}
	}

	for (child = 0; child < root_handle->nchildren; child++)
	{
		starpu_data_handle_t child_handle = starpu_data_get_child(root_handle, child);
		_starpu_data_clear_implicit(child_handle);
		free(child_handle->active_readonly_children);
		free(child_handle->active_readonly_nchildren);

		STARPU_PTHREAD_MUTEX_DESTROY(&child_handle->busy_mutex);
		STARPU_PTHREAD_COND_DESTROY(&child_handle->busy_cond);
		STARPU_PTHREAD_MUTEX_DESTROY(&child_handle->sequential_consistency_mutex);
#ifdef STARPU_RECURSIVE_TASKS
		STARPU_PTHREAD_MUTEX_DESTROY(&child_handle->unpartition_mutex);
#endif

		STARPU_HG_ENABLE_CHECKING(child_handle->post_sync_tasks_cnt);
		STARPU_HG_ENABLE_CHECKING(child_handle->busy_count);

		_starpu_data_requester_prio_list_deinit(&child_handle->req_list);
		_starpu_data_requester_prio_list_deinit(&child_handle->reduction_req_list);

		if (child_handle->switch_cl)
		{
			free(child_handle->switch_cl->dyn_nodes);
			free(child_handle->switch_cl);
		}

		_starpu_trace_handle_data_unregister(&child_handle);
	}

	/* there is no child anymore */
	starpu_data_handle_t children = root_handle->children;
	root_handle->children = NULL;
	root_handle->nchildren = 0;
	root_handle->nplans--;

	/* now the parent may be used again so we release the lock */
	_starpu_spin_unlock(&root_handle->header_lock);

	free(children);

	_starpu_trace_end_unpartition(&root_handle, gathering_node);
}

void starpu_data_partition(starpu_data_handle_t initial_handle, struct starpu_data_filter *f)
{
	unsigned nparts = _starpu_data_partition_nparts(initial_handle, f);
	STARPU_ASSERT_MSG(initial_handle->nchildren == 0, "there should not be multiple filters applied on the same data %p, further filtering has to be done on children", initial_handle);
	STARPU_ASSERT_MSG(initial_handle->nplans == 0, "partition planning and synchronous partitioning is not supported");

	initial_handle->children = NULL;

	/* Make sure to wait for previous tasks working on the whole data */
	starpu_data_acquire_on_node(initial_handle, STARPU_ACQUIRE_NO_NODE, initial_handle->initialized?STARPU_RW:STARPU_W);
	starpu_data_release_on_node(initial_handle, STARPU_ACQUIRE_NO_NODE);

	_starpu_data_partition(initial_handle, NULL, nparts, f, 1);
}

int starpu_data_partition_get_nplans(starpu_data_handle_t initial_handle)
{
	return initial_handle->nplans;
}

void starpu_data_partition_plan(starpu_data_handle_t initial_handle, struct starpu_data_filter *f, starpu_data_handle_t *childrenp)
{
	unsigned i;
	unsigned nparts = _starpu_data_partition_nparts(initial_handle, f);
	STARPU_ASSERT_MSG(initial_handle->nchildren == 0, "partition planning and synchronous partitioning is not supported");
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	struct starpu_codelet *cl = initial_handle->switch_cl;
	starpu_data_handle_t *children;

	_STARPU_MALLOC(children, nparts * sizeof(*children));
	for (i = 0; i < nparts; i++)
	{
		_STARPU_CALLOC(children[i], 1, sizeof(struct _starpu_data_state));
		childrenp[i] = children[i];
	}
	_starpu_data_partition(initial_handle, children, nparts, f, 0);

	if (!cl)
	{
		/* Create a codelet that will make the coherency on the home node */
		_STARPU_CALLOC(initial_handle->switch_cl, 1, sizeof(*initial_handle->switch_cl));
		cl = initial_handle->switch_cl;
		cl->where = STARPU_NOWHERE;
		cl->nbuffers = STARPU_VARIABLE_NBUFFERS;
		cl->flags = STARPU_CODELET_NOPLANS;
		cl->name = "data_partition_switch";
		cl->specific_nodes = 1;
	}
}

void starpu_data_partition_clean_node(starpu_data_handle_t root_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node)
{
	unsigned i;

	if (children[0]->active)
	{
		starpu_data_unpartition_submit(root_handle, nparts, children, gather_node);
	}

	free(children[0]->siblings);

	for (i = 0; i < nparts; i++)
	{
		children[i]->siblings = NULL;
		starpu_data_unregister_submit(children[i]);
	}

	_starpu_spin_lock(&root_handle->header_lock);
	root_handle->nplans--;
	_starpu_spin_unlock(&root_handle->header_lock);
}

void starpu_data_partition_clean(starpu_data_handle_t root_handle, unsigned nparts, starpu_data_handle_t *children)
{
#ifdef STARPU_DEVEL
#warning FIXME: better choose gathering node
#endif
	starpu_data_partition_clean_node(root_handle, nparts, children, root_handle->home_node);
}

static
void _starpu_data_partition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, unsigned char *handles_sequential_consistency)
{
	unsigned i;
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 0, "One can't submit several partition plannings at the same time");
	STARPU_ASSERT_MSG(initial_handle->part_readonly == 0, "One can't submit a partition planning while a readonly partitioning is active");
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	initial_handle->partitioned++;
	initial_handle->active_nchildren = children[0]->nsiblings;
	initial_handle->active_children = children[0]->siblings;
	_starpu_spin_unlock(&initial_handle->header_lock);

	for (i = 0; i < nparts; i++)
	{
		_starpu_spin_lock(&children[i]->header_lock);
		children[i]->active = 1;
		_starpu_spin_unlock(&children[i]->header_lock);
	}

	if (!initial_handle->initialized)
		/* No need for coherency, it is not initialized */
		return;

	struct starpu_data_mode_node_descr descr[nparts];
	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
		descr[i].node = gathering_node;
	}
	/* TODO: assert nparts too */
	int ret;
	if (handles_sequential_consistency)
		ret = starpu_task_insert(initial_handle->switch_cl, STARPU_RW, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, nparts,
					 STARPU_NAME, "partition",
					 STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, handles_sequential_consistency,
					 0);
	else
		ret = starpu_task_insert(initial_handle->switch_cl, STARPU_RW, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, nparts,
					 STARPU_NAME, "partition",
					 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	if (!handles_sequential_consistency || handles_sequential_consistency[0])
		_starpu_data_invalidate_submit_noplan(initial_handle);
}

void starpu_data_partition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int sequential_consistency)
{
	unsigned i;
	unsigned char handles_sequential_consistency[nparts+1];
	handles_sequential_consistency[0] = sequential_consistency;
	for(i=1 ; i<nparts+1 ; i++) handles_sequential_consistency[i] = children[i-1]->sequential_consistency;

	_starpu_data_partition_submit(initial_handle, nparts, children, handles_sequential_consistency);
}

void starpu_data_partition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	_starpu_data_partition_submit(initial_handle, nparts, children, NULL);
}

void starpu_data_partition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	starpu_data_partition_readonly_submit_sequential_consistency(initial_handle, nparts, children, initial_handle->sequential_consistency);
}

void starpu_data_partition_readonly_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int sequential_consistency)
{
	unsigned i;
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 0 || initial_handle->part_readonly, "One can't submit a readonly partition planning at the same time as a readwrite partition planning");
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	initial_handle->partitioned++;
	initial_handle->part_readonly = 1;
	if (initial_handle->nactive_readonly_children < initial_handle->partitioned)
	{
		_STARPU_REALLOC(initial_handle->active_readonly_children, initial_handle->partitioned * sizeof(initial_handle->active_readonly_children[0]));
		_STARPU_REALLOC(initial_handle->active_readonly_nchildren, initial_handle->partitioned * sizeof(initial_handle->active_readonly_nchildren[0]));
		initial_handle->nactive_readonly_children = initial_handle->partitioned;
	}
	initial_handle->active_readonly_children[initial_handle->partitioned-1] = children[0]->siblings;
	initial_handle->active_readonly_nchildren[initial_handle->partitioned-1] = children[0]->nsiblings;
	_starpu_spin_unlock(&initial_handle->header_lock);

	for (i = 0; i < nparts; i++)
	{
		_starpu_spin_lock(&children[i]->header_lock);
		children[i]->active = 1;
		children[i]->active_ro = 1;
		_starpu_spin_unlock(&children[i]->header_lock);
	}

	STARPU_ASSERT_MSG(initial_handle->initialized || initial_handle->init_cl, "It is odd to read-only-partition a data which does not have a value yet");
	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	struct starpu_data_mode_node_descr descr[nparts];
	char handles_sequential_consistency[nparts+1];
	handles_sequential_consistency[0] = sequential_consistency;

	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
		descr[i].node = gathering_node;
		handles_sequential_consistency[i+1] = (char) children[i]->sequential_consistency;
	}
	/* TODO: assert nparts too */
	int ret = starpu_task_insert(initial_handle->switch_cl, STARPU_R, initial_handle,
				     STARPU_DATA_MODE_NODE_ARRAY, descr, nparts,
				     STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, handles_sequential_consistency,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

void starpu_data_partition_readwrite_upgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 1, "One can't upgrade a readonly partition planning to readwrite while other readonly partition plannings are active");
	STARPU_ASSERT_MSG(initial_handle->part_readonly == 1, "One can only upgrade a readonly partition planning");
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	initial_handle->part_readonly = 0;
	initial_handle->active_nchildren = initial_handle->active_readonly_nchildren[0];
	initial_handle->active_children = initial_handle->active_readonly_children[0];
	initial_handle->active_readonly_children[0] = NULL;
	initial_handle->active_readonly_nchildren[0] = 0;
	_starpu_spin_unlock(&initial_handle->header_lock);

	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	unsigned i;
	struct starpu_data_mode_node_descr descr[nparts];
	for (i = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		children[i]->active_ro = 0;
		descr[i].handle = children[i];
		descr[i].mode = STARPU_W;
		descr[i].node = gathering_node;
	}
	/* TODO: assert nparts too */
	int ret = starpu_task_insert(initial_handle->switch_cl, STARPU_RW, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, nparts, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	_starpu_data_invalidate_submit_noplan(initial_handle);
}

void starpu_data_partition_readonly_downgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children)
{
	unsigned i;
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned == 1, "One can't downgrade a read-write partition planning to read-only while other partition plannings are active");
	STARPU_ASSERT_MSG(initial_handle->part_readonly == 0, "Partition is already read-only");
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	initial_handle->part_readonly = 1;
	if (initial_handle->nactive_readonly_children < initial_handle->partitioned)
	{
		_STARPU_REALLOC(initial_handle->active_readonly_children, initial_handle->partitioned * sizeof(initial_handle->active_readonly_children[0]));
		_STARPU_REALLOC(initial_handle->active_readonly_nchildren, initial_handle->partitioned * sizeof(initial_handle->active_readonly_nchildren[0]));
		initial_handle->nactive_readonly_children = initial_handle->partitioned;
	}
	initial_handle->active_readonly_children[initial_handle->partitioned-1] = children[0]->siblings;
	initial_handle->active_readonly_nchildren[initial_handle->partitioned-1] = children[0]->nsiblings;
	initial_handle->active_children = NULL;
	initial_handle->active_nchildren = 0;
	_starpu_spin_unlock(&initial_handle->header_lock);

	for (i = 0; i < nparts; i++)
	{
		_starpu_spin_lock(&children[i]->header_lock);
		children[i]->active = 1;
		children[i]->active_ro = 1;
		_starpu_spin_unlock(&children[i]->header_lock);
	}

	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	struct starpu_data_mode_node_descr descr[nparts];
	unsigned n;
	for (i = 0, n = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		if (!children[i]->initialized)
			/* Dropped value, do not care about coherency for this one */
			continue;
		descr[n].handle = children[i];
		descr[n].mode = STARPU_R;
		descr[n].node = gathering_node;
		n++;
	}
	/* TODO: assert nparts too */
	int ret = starpu_task_insert(initial_handle->switch_cl, initial_handle->initialized?STARPU_RW:STARPU_W, initial_handle,
				     STARPU_DATA_MODE_NODE_ARRAY, descr, n,
				     ///STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, handles_sequential_consistency,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

void _starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node, unsigned char *handles_sequential_consistency, void (*callback_func)(void *), void *callback_arg)
{
	unsigned i;
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	STARPU_ASSERT_MSG(gather_node == initial_handle->home_node || gather_node == -1, "gathering node different from home node is currently not supported");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned >= 1, "No partition planning is active for handle %p", initial_handle);
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	if (initial_handle->part_readonly)
	{
		/* Replace this children set with the last set in the list of readonly children sets */
		for (i = 0; i < initial_handle->partitioned-1; i++)
		{
			if (initial_handle->active_readonly_children[i] == children[0]->siblings)
			{
				initial_handle->active_readonly_children[i] = initial_handle->active_readonly_children[initial_handle->partitioned-1];
				initial_handle->active_readonly_nchildren[i] = initial_handle->active_readonly_nchildren[initial_handle->partitioned-1];
				initial_handle->active_readonly_children[initial_handle->partitioned-1] = NULL;
				initial_handle->active_readonly_nchildren[initial_handle->partitioned-1] = 0;
				break;
			}
		}
	}
	else
	{
		initial_handle->active_nchildren = 0;
		initial_handle->active_children = NULL;
	}
	initial_handle->partitioned--;
	if (!initial_handle->partitioned)
		initial_handle->part_readonly = 0;
	initial_handle->active_nchildren = 0;
	initial_handle->active_children = NULL;
	_starpu_spin_unlock(&initial_handle->header_lock);

	for (i = 0; i < nparts; i++)
	{
		_starpu_spin_lock(&children[i]->header_lock);
		children[i]->active = 0;
		children[i]->active_ro = 0;
		_starpu_spin_unlock(&children[i]->header_lock);
	}

	unsigned n;
	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	struct starpu_data_mode_node_descr descr[nparts];
	for (i = 0, n = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		if (!children[i]->initialized)
			/* Dropped value, do not care about coherency for this one */
			continue;
		descr[n].handle = children[i];
		descr[n].mode = STARPU_RW;
		descr[n].node = gathering_node;
		n++;
	}
	/* TODO: assert nparts too */
	int ret;
	if (handles_sequential_consistency)
		ret = starpu_task_insert(initial_handle->switch_cl, STARPU_W, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, n,
					 STARPU_NAME, "unpartition",
					 STARPU_HANDLES_SEQUENTIAL_CONSISTENCY, handles_sequential_consistency,
					 STARPU_CALLBACK_WITH_ARG_NFREE, callback_func, callback_arg,
					 0);
	else
		ret = starpu_task_insert(initial_handle->switch_cl, STARPU_W, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, n,
					 STARPU_NAME, "unpartition",
					 STARPU_CALLBACK_WITH_ARG_NFREE, callback_func, callback_arg,
					 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	for (i = 0; i < nparts; i++)
	{
#ifdef STARPU_DEVEL
#warning that s costly, perhaps we could add a STARPU_INVALIDATE mode that does the invalidation after the task?
#endif
		if (!handles_sequential_consistency || handles_sequential_consistency[i+1])
			_starpu_data_invalidate_submit_noplan(children[i]);
	}
}

void starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node)
{
	_starpu_data_unpartition_submit(initial_handle, nparts, children, gather_node, NULL, NULL, NULL);
}

void starpu_data_unpartition_submit_sequential_consistency_cb(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node, int sequential_consistency, void (*callback_func)(void *), void *callback_arg)
{
	unsigned i;
	unsigned char handles_sequential_consistency[nparts+1];
	handles_sequential_consistency[0] = sequential_consistency;
	for(i=1 ; i<nparts+1 ; i++) handles_sequential_consistency[i] = children[i-1]->sequential_consistency;
	_starpu_data_unpartition_submit(initial_handle, nparts, children, gather_node, handles_sequential_consistency, callback_func, callback_arg);
}

void starpu_data_unpartition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node, int sequential_consistency)
{
	unsigned i;
	unsigned char handles_sequential_consistency[nparts+1];
	handles_sequential_consistency[0] = sequential_consistency;
	for(i=1 ; i<nparts+1 ; i++) handles_sequential_consistency[i] = children[i-1]->sequential_consistency;
	_starpu_data_unpartition_submit(initial_handle, nparts, children, gather_node, handles_sequential_consistency, NULL, NULL);
}

void starpu_data_unpartition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node)
{
	STARPU_ASSERT_MSG(initial_handle->sequential_consistency, "partition planning is currently only supported for data with sequential consistency");
	STARPU_ASSERT_MSG(gather_node == initial_handle->home_node || gather_node == -1, "gathering node different from home node is currently not supported");
	_starpu_spin_lock(&initial_handle->header_lock);
	STARPU_ASSERT_MSG(initial_handle->partitioned >= 1, "No partition planning is active for handle %p", initial_handle);
	STARPU_ASSERT_MSG(nparts > 0, "One can't partition into 0 parts");
	initial_handle->part_readonly = 1;
	_starpu_spin_unlock(&initial_handle->header_lock);

	unsigned i, n;
	struct starpu_data_mode_node_descr descr[nparts];
	int gathering_node = _starpu_data_get_gathering_node(initial_handle);
	for (i = 0, n = 0; i < nparts; i++)
	{
		STARPU_ASSERT_MSG(children[i]->parent_handle == initial_handle, "child(%d) %p is partitioned from %p and not from the given parameter %p", i, children[i], children[i]->parent_handle, initial_handle);
		if (!children[i]->initialized)
			/* Dropped value, do not care about coherency for this one */
			continue;
		descr[n].handle = children[i];
		descr[n].mode = STARPU_R;
		descr[n].node = gathering_node;
		n++;
	}
	/* TODO: assert nparts too */
	int ret = starpu_task_insert(initial_handle->switch_cl, STARPU_W, initial_handle, STARPU_DATA_MODE_NODE_ARRAY, descr, n, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert")
}

/* Unpartition everything below ancestor */
static void starpu_data_unpartition_submit_r(starpu_data_handle_t ancestor, int gathering_node)
{
	unsigned i, j, nsiblings;
	if (!ancestor->partitioned)
		/* It's already unpartitioned */
		return;
	_STARPU_DEBUG("ancestor %p needs unpartitioning\n", ancestor);
	if (ancestor->part_readonly)
	{
		unsigned n = ancestor->partitioned;
		/* Uh, has to go through all read-only partitions */
		for (i = 0; i < n; i++)
		{
			/* Note: active_readonly_children is emptied by starpu_data_unpartition_submit_r calls below, that's why we always refer to [0] here */
			starpu_data_handle_t *children = ancestor->active_readonly_children[0];
			_STARPU_DEBUG("unpartition readonly children %p etc.\n", children[0]);
			nsiblings = children[0]->nsiblings;
			for (j = 0; j < nsiblings; j++)
			{
				/* Make sure our children are unpartitioned */
				starpu_data_unpartition_submit_r(children[j], gathering_node);
			}
			/* And unpartition them */
			starpu_data_unpartition_submit(ancestor, nsiblings, children, gathering_node);
		}
	}
	else
	{
		_STARPU_DEBUG("unpartition children %p\n", ancestor->active_children);
		/* Only one partition */
		nsiblings = ancestor->active_children[0]->nsiblings;
		for (i = 0; i < nsiblings; i++)
			starpu_data_unpartition_submit_r(ancestor->active_children[i], gathering_node);
		/* And unpartition ourself */
		starpu_data_unpartition_submit(ancestor, nsiblings, ancestor->active_children, gathering_node);
	}
}

/* Make ancestor partition itself properly for target */
static void _starpu_data_partition_access_look_up(starpu_data_handle_t ancestor, starpu_data_handle_t target, int write)
{
	/* First make sure ancestor has proper state, if not, ask parent */
	if (!ancestor->active || (write && ancestor->active_ro))
	{
		/* (The root is always active-rw) */
		STARPU_ASSERT(ancestor->parent_handle);
		_STARPU_DEBUG("ancestor %p is not ready: %s, asking parent %p\n", ancestor, ancestor->active ? ancestor->active_ro ? "RO" : "RW" : "NONE", ancestor->parent_handle);
		_starpu_data_partition_access_look_up(ancestor->parent_handle, ancestor, write);
		_STARPU_DEBUG("ancestor %p is now ready\n", ancestor);
	}
	else
		_STARPU_DEBUG("ancestor %p was ready\n", ancestor);

	/* We shouldn't be called for nothing */
	STARPU_ASSERT(!ancestor->partitioned || !target || ancestor->active_children != target->siblings || (ancestor->part_readonly && write));

	/* Then unpartition ancestor if needed */
	if (ancestor->partitioned &&
			/* Not the right children, unpartition ourself */
			((target && write && ancestor->active_children != target->siblings) ||
			 (target && !write && !ancestor->part_readonly) ||
			/* We are partitioned and we want to write or some child
			 * is writing and we want to read, unpartition ourself*/
			(!target && (write || !ancestor->part_readonly))))
	{
#ifdef STARPU_DEVEL
#warning FIXME: better choose gathering node
#endif
		starpu_data_unpartition_submit_r(ancestor, ancestor->home_node);
	}

	if (!target)
	{
		_STARPU_DEBUG("ancestor %p is done\n", ancestor);
		/* No child target, nothing more to do actually.  */
		return;
	}

	/* Then partition ancestor towards target, if needed */
	if (ancestor->partitioned)
	{
		/* That must be readonly, otherwise we would have unpartitioned it */
		STARPU_ASSERT(ancestor->part_readonly);
		if (write)
		{
			_STARPU_DEBUG("ancestor %p is already partitioned RO, turn RW\n", ancestor);
			/* Already partitioned, normally it's already for the target */
			STARPU_ASSERT(ancestor->active_children == target->siblings);
			/* And we are here just because we haven't partitioned rw */
			STARPU_ASSERT(ancestor->part_readonly && write);
			/* So we just need to upgrade ro to rw */
			starpu_data_partition_readwrite_upgrade_submit(ancestor, target->nsiblings, target->siblings);
		}
		else
		{
			_STARPU_DEBUG("ancestor %p is already partitioned RO, but not to target, partition towards target too\n", ancestor);
			/* So we just need to upgrade ro to rw */
			starpu_data_partition_readonly_submit(ancestor, target->nsiblings, target->siblings);
		}
	}
	else
	{
		/* Just need to partition properly for the child */
		if (write)
		{
			_STARPU_DEBUG("partition ancestor %p RW\n", ancestor);
			starpu_data_partition_submit(ancestor, target->nsiblings, target->siblings);
		}
		else
		{
			_STARPU_DEBUG("partition ancestor %p RO\n", ancestor);
			starpu_data_partition_readonly_submit(ancestor, target->nsiblings, target->siblings);
		}
	}
}

void _starpu_data_partition_access_submit(starpu_data_handle_t target, int write)
{
	_STARPU_DEBUG("accessing %p %s\n", target, write ? "RW" : "RO");
	_starpu_data_partition_access_look_up(target, NULL, write);
}

void starpu_filter_nparts_compute_chunk_size_and_offset(unsigned n, unsigned nparts,
							size_t elemsize, unsigned id,
							size_t blocksize, size_t *chunk_size,
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
	 * offset = total * elemsize * blocksize;
	 */
	if (offset != NULL)
		*offset = (id *(n/nparts) + STARPU_MIN(remainder, id)) * blocksize * elemsize;
}
