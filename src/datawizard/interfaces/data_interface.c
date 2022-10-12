/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdint.h>
#include <stdarg.h>

#include <datawizard/datawizard.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/memstats.h>
#include <datawizard/malloc.h>
#include <core/dependencies/data_concurrency.h>
#include <common/knobs.h>
#include <common/starpu_spinlock.h>
#include <core/task.h>
#include <core/workers.h>
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>
#endif

static struct starpu_data_interface_ops **_id_to_ops_array;
static unsigned _id_to_ops_array_size;

/* Hash table mapping host pointers to data handles.  */
static int32_t nregistered, maxnregistered;
static int _data_interface_number = STARPU_MAX_INTERFACE_ID;
starpu_arbiter_t _starpu_global_arbiter;
static int max_memory_use;

static void _starpu_data_unregister(starpu_data_handle_t handle, unsigned coherent, unsigned nowait);

void _starpu_data_interface_fini(void);

void _starpu_data_interface_init(void)
{
	max_memory_use = starpu_getenv_number_default("STARPU_MAX_MEMORY_USE", 0);

	/* Just for testing purpose */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		_starpu_global_arbiter = starpu_arbiter_create();

	_starpu_crash_add_hook(&_starpu_data_interface_fini);
}

void _starpu_data_interface_fini(void)
{
	if (max_memory_use)
		_STARPU_DISP("Memory used for %d data handles: %lu MiB\n", maxnregistered, (unsigned long) (maxnregistered * sizeof(struct _starpu_data_state)) >> 20);
}

void _starpu_data_interface_shutdown()
{
	free(_id_to_ops_array);
	_id_to_ops_array = NULL;
	_id_to_ops_array_size = 0;

	_starpu_data_interface_fini();
}

struct starpu_data_interface_ops *_starpu_data_interface_get_ops(unsigned interface_id)
{
	switch (interface_id)
	{
		case STARPU_MATRIX_INTERFACE_ID:
			return &starpu_interface_matrix_ops;

		case STARPU_BLOCK_INTERFACE_ID:
			return &starpu_interface_block_ops;

		case STARPU_VECTOR_INTERFACE_ID:
			return &starpu_interface_vector_ops;

		case STARPU_CSR_INTERFACE_ID:
			return &starpu_interface_csr_ops;

		case STARPU_BCSR_INTERFACE_ID:
			return &starpu_interface_bcsr_ops;

		case STARPU_VARIABLE_INTERFACE_ID:
			return &starpu_interface_variable_ops;

		case STARPU_VOID_INTERFACE_ID:
			return &starpu_interface_void_ops;

		case STARPU_MULTIFORMAT_INTERFACE_ID:
			return &starpu_interface_multiformat_ops;

		case STARPU_COO_INTERFACE_ID:
			return &starpu_interface_coo_ops;

		case STARPU_TENSOR_INTERFACE_ID:
			return &starpu_interface_tensor_ops;

		case STARPU_NDIM_INTERFACE_ID:
			return &starpu_interface_ndim_ops;

		default:
		{
			if (interface_id-STARPU_MAX_INTERFACE_ID > _id_to_ops_array_size || _id_to_ops_array[interface_id-STARPU_MAX_INTERFACE_ID]==NULL)
			{
				_STARPU_MSG("There is no 'struct starpu_data_interface_ops' registered for interface %d\n", interface_id);
				STARPU_ABORT();
				return NULL;
			}
			else
				return _id_to_ops_array[interface_id-STARPU_MAX_INTERFACE_ID];
		}
	}
}

/*
 * Start monitoring a piece of data
 */

static void _starpu_register_new_data(starpu_data_handle_t handle,
					int home_node, uint32_t wt_mask)
{
	STARPU_ASSERT(handle);

	/* first take care to properly lock the data */
	_starpu_spin_lock(&handle->header_lock);

	handle->root_handle = handle;
	//handle->father_handle = NULL;
	//handle->nsiblings = 0;
	//handle->siblings = NULL;
	//handle->sibling_index = 0; /* could be anything for the root */
	handle->depth = 1; /* the tree is just a node yet */

	handle->active = 1;

	/* Store some values directly in the handle not to recompute them all
	 * the time. */
	handle->footprint = _starpu_compute_data_footprint(handle);

	handle->home_node = home_node;

	handle->wt_mask = wt_mask;

	//handle->aliases = 0;
	//handle->readonly_dup = NULL;
	//handle->readonly_dup_of = NULL;

	//handle->is_not_important = 0;

	handle->sequential_consistency =
		starpu_data_get_default_sequential_consistency_flag();
	handle->initialized = home_node != -1;
	//handle->readonly = 0;
	handle->ooc = 1;

	/* By default, there are no methods available to perform a reduction */
	//handle->redux_cl = NULL;
	//handle->init_cl = NULL;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_node[node];

		replicate->memory_node = node;
		//replicate->relaxed_coherency = 0;
		//replicate->refcnt = 0;
		//replicate->nb_tasks_prefetch = 0;

		if ((int) node == home_node)
		{
			/* this is the home node with the only valid copy */
			replicate->state = STARPU_OWNER;
			replicate->allocated = 1;
			//replicate->automatically_allocated = 0;
			replicate->initialized = 1;
		}
		else
		{
			/* the value is not available here yet */
			replicate->state = STARPU_INVALID;
			//replicate->allocated = 0;
			//replicate->initialized = 0;
		}

		replicate->mapped = STARPU_UNMAPPED;
	}

	/* now the data is available ! */
	_starpu_spin_unlock(&handle->header_lock);
	(void)STARPU_ATOMIC_ADD(&nregistered, 1);
	_starpu_perf_counter_update_max_int32(&maxnregistered, nregistered);
}

void
_starpu_data_initialize_per_worker(starpu_data_handle_t handle)
{
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();

	_starpu_spin_checklocked(&handle->header_lock);

	_STARPU_CALLOC(handle->per_worker, nworkers, sizeof(*handle->per_worker));

	size_t interfacesize = handle->ops->interface_size;

	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_data_replicate *replicate;
		//unsigned node;
		replicate = &handle->per_worker[worker];
		//replicate->allocated = 0;
		//replicate->automatically_allocated = 0;
		replicate->state = STARPU_INVALID;
		//replicate->refcnt = 0;
		replicate->handle = handle;
		//replicate->nb_tasks_prefetch = 0;

		//for (node = 0; node < STARPU_MAXNODES; node++)
		//{
		//	replicate->request[node] = NULL;
		//	replicate->last_request[node] = NULL;
		//}
		//replicate->load_request = NULL;

		/* Assuming being used for SCRATCH for now, patched when entering REDUX mode */
		replicate->relaxed_coherency = 1;
		//replicate->initialized = 0;
		replicate->memory_node = starpu_worker_get_memory_node(worker);
		replicate->mapped = STARPU_UNMAPPED;

		_STARPU_CALLOC(replicate->data_interface, 1, interfacesize);
		/* duplicate  the content of the interface on node 0 */
		memcpy(replicate->data_interface, handle->per_node[STARPU_MAIN_RAM].data_interface, interfacesize);
	}
}

void starpu_data_ptr_register(starpu_data_handle_t handle, unsigned node)
{
	struct _starpu_data_replicate *replicate = &handle->per_node[node];

	_starpu_spin_lock(&handle->header_lock);
	STARPU_ASSERT_MSG(replicate->allocated == 0, "starpu_data_ptr_register must be called right after starpu_data_register");
	replicate->allocated = 1;
	replicate->automatically_allocated = 0;
	_starpu_spin_unlock(&handle->header_lock);
}

int _starpu_data_handle_init(starpu_data_handle_t handle, struct starpu_data_interface_ops *interface_ops, unsigned int mf_node)
{
	unsigned node;

	/* Tell helgrind that our access to busy_count in
	 * starpu_data_unregister is actually safe */
	STARPU_HG_DISABLE_CHECKING(handle->busy_count);

	handle->magic = 42;

	/* When not specified, the fields are initialized in _starpu_register_new_data and _starpu_data_partition */

	_starpu_data_requester_prio_list_init0(&handle->req_list);
	//handle->refcnt = 0;
	//handle->unlocking_reqs = 0;
	//handle->current_mode = STARPU_NONE;
	_starpu_spin_init(&handle->header_lock);

	//handle->busy_count = 0;
	//handle->busy_waiting = 0;
	STARPU_PTHREAD_MUTEX_INIT0(&handle->busy_mutex, NULL);
	STARPU_PTHREAD_COND_INIT0(&handle->busy_cond, NULL);
#ifdef STARPU_BUBBLE
	STARPU_PTHREAD_MUTEX_INIT0(&handle->unpartition_mutex, NULL);
#endif

	//handle->root_handle
	//handle->father_handle
	//handle->active_children = NULL;
	//handle->active_nchildren = 0;
	//handle->active_readonly_children = NULL;
	//handle->active_readonly_nchildren = NULL;
	//handle->nactive_readonly_children = 0;
	//handle->nsiblings
	//handle->siblings
	//handle->sibling_index
	//handle->depth

	/* there is no hierarchy yet */
	//handle->children = NULL;
	//handle->nchildren = 0;
	//handle->nplans = 0;
	//handle->switch_cl = NULL;
	//handle->switch_cl_nparts = 0;
	//handle->partitioned = 0;
	//handle->part_readonly = 0;

	//handle->active
	//handle->active_ro = 0;

	//handle->per_node below

	handle->ops = interface_ops;
	size_t interfacesize = interface_ops->interface_size;

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		_starpu_memory_stats_init_per_node(handle, node);

		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_node[node];
		/* relaxed_coherency = 0 */

		replicate->handle = handle;

		_STARPU_CALLOC(replicate->data_interface, 1, interfacesize);
		if (handle->ops->init) handle->ops->init(replicate->data_interface);
	}

	//handle->per_worker = NULL;
	//handle->ops above

	//handle->footprint

	//handle->home_node
	//handle->wt_mask
	//handle->aliases = 0;
	//handle->is_not_important
	//handle->sequential_consistency
	//handle->initialized
	//handle->readonly
	//handle->ooc
	//handle->lazy_unregister = 0;
	//handle->removed_from_context_hash = 0;

	STARPU_PTHREAD_MUTEX_INIT0(&handle->sequential_consistency_mutex, NULL);

	handle->last_submitted_mode = STARPU_R;
	//handle->last_sync_task = NULL;
	//handle->last_submitted_accessors.task = NULL;
	handle->last_submitted_accessors.next = &handle->last_submitted_accessors;
	handle->last_submitted_accessors.prev = &handle->last_submitted_accessors;

#ifdef STARPU_USE_FXT
	//handle->last_submitted_ghost_sync_id_is_valid = 0;
	//handle->last_submitted_ghost_sync_id = 0;
	//handle->last_submitted_ghost_accessors_id = NULL;
#endif

	//handle->post_sync_tasks = NULL;
	/* Tell helgrind that the race in _starpu_unlock_post_sync_tasks is fine */
	STARPU_HG_DISABLE_CHECKING(handle->post_sync_tasks_cnt);
	//handle->post_sync_tasks_cnt = 0;

	//handle->redux_cl
	//handle->init_cl

	//handle->reduction_refcnt = 0;

	_starpu_data_requester_prio_list_init0(&handle->reduction_req_list);

	//handle->reduction_tmp_handles = NULL;

	//handle->write_invalidation_req = NULL;

	//handle->mpi_data = NULL; /* invalid until set */

	_starpu_memory_stats_init(handle);

	handle->mf_node = mf_node;

	//handle->unregister_hook = NULL;

	if (_starpu_global_arbiter)
		/* Just for testing purpose */
		starpu_data_assign_arbiter(handle, _starpu_global_arbiter);
	else
	{
		//handle->arbiter = NULL;
	}
	_starpu_data_requester_prio_list_init0(&handle->arbitered_req_list);

	handle->last_locality = -1;

	//handle->dimensions = 0;
	//handle->coordinates = {};

	//handle->user_data = NULL;
	//handle->sched_data = NULL;

	return 0;
}

static
starpu_data_handle_t _starpu_data_handle_allocate(struct starpu_data_interface_ops *interface_ops, unsigned int mf_node)
{
	starpu_data_handle_t handle;
	_STARPU_CALLOC(handle, 1, sizeof(struct _starpu_data_state));
	_starpu_data_handle_init(handle, interface_ops, mf_node);
	return handle;
}

void starpu_data_register(starpu_data_handle_t *handleptr, int home_node,
			  void *data_interface,
			  struct starpu_data_interface_ops *ops)
{
	starpu_data_handle_t handle = _starpu_data_handle_allocate(ops, home_node);

	STARPU_ASSERT(handleptr);
	*handleptr = handle;

	/* check the interfaceid is set */
	STARPU_ASSERT(ops->interfaceid != STARPU_UNKNOWN_INTERFACE_ID);

	/* fill the interface fields with the appropriate method */
	STARPU_ASSERT(ops->register_data_handle);
	ops->register_data_handle(handle, home_node, data_interface);

	if ((unsigned)ops->interfaceid >= STARPU_MAX_INTERFACE_ID)
	{
		if ((unsigned)ops->interfaceid > _id_to_ops_array_size)
		{
			if (!_id_to_ops_array_size)
			{
				_id_to_ops_array_size = 16;
			}
			else
			{
				_id_to_ops_array_size *= 2;
			}
			_STARPU_REALLOC(_id_to_ops_array, _id_to_ops_array_size * sizeof(struct starpu_data_interface_ops *));
		}
		_id_to_ops_array[ops->interfaceid-STARPU_MAX_INTERFACE_ID] = ops;
	}

	_starpu_register_new_data(handle, home_node, 0);
	_STARPU_TRACE_HANDLE_DATA_REGISTER(handle);
}

void starpu_data_register_same(starpu_data_handle_t *handledst, starpu_data_handle_t handlesrc)
{
	void *local_interface = starpu_data_get_interface_on_node(handlesrc, STARPU_MAIN_RAM);
	starpu_data_register(handledst, -1, local_interface, handlesrc->ops);
}

void *starpu_data_handle_to_pointer(starpu_data_handle_t handle, unsigned node)
{
	/* Check whether the operation is supported and the node has actually
	 * been allocated.  */
	if (!starpu_data_test_if_allocated_on_node(handle, node))
		return NULL;
	if (handle->ops->to_pointer)
	{
		return handle->ops->to_pointer(starpu_data_get_interface_on_node(handle, node), node);
	}

	/* Deprecated */
	if (handle->ops->handle_to_pointer)
	{
		return handle->ops->handle_to_pointer(handle, node);
	}

	return NULL;
}

int starpu_data_pointer_is_inside(starpu_data_handle_t handle, unsigned node, void *ptr)
{
	/* Check whether the operation is supported and the node has actually
	 * been allocated.  */
	if (!starpu_data_test_if_allocated_on_node(handle, node))
		return 0;
	if (handle->ops->pointer_is_inside)
	{
		return handle->ops->pointer_is_inside(starpu_data_get_interface_on_node(handle, node), node, ptr);
	}
	/* Don't know :/ */
	return -1;
}

void *starpu_data_get_local_ptr(starpu_data_handle_t handle)
{
	return starpu_data_handle_to_pointer(handle, starpu_worker_get_local_memory_node());
}

struct starpu_data_interface_ops* starpu_data_get_interface_ops(starpu_data_handle_t handle)
{
	return handle->ops;
}

void _starpu_data_free_interfaces(starpu_data_handle_t handle)
{
	unsigned node;
	unsigned nworkers = starpu_worker_get_count();

	if (handle->ops->unregister_data_handle)
		handle->ops->unregister_data_handle(handle);

	for (node = 0; node < STARPU_MAXNODES; node++)
		free(handle->per_node[node].data_interface);

	if (handle->per_worker)
	{
		unsigned worker;
		for (worker = 0; worker < nworkers; worker++)
			free(handle->per_worker[worker].data_interface);
		free(handle->per_worker);
	}
}

struct _starpu_unregister_callback_arg
{
	unsigned memory_node;
	starpu_data_handle_t handle;
	unsigned terminated;
	starpu_pthread_mutex_t mutex;
	starpu_pthread_cond_t cond;
};

/* Check whether we should tell starpu_data_unregister that the data handle is
 * not busy any more.
 * The header is supposed to be locked.
 * This may free the handle, if it was lazily unregistered (1 is returned in
 * that case).  The handle pointer thus becomes invalid for the caller.
 *
 * Note: we inline some of the tests in the _starpu_data_check_not_busy macro.
 */
int __starpu_data_check_not_busy(starpu_data_handle_t handle)
{
	if (STARPU_LIKELY(handle->busy_count))
		return 0;

	/* Not busy any more, perhaps have to unregister etc.  */
	if (STARPU_UNLIKELY(handle->busy_waiting))
	{
		STARPU_PTHREAD_MUTEX_LOCK(&handle->busy_mutex);
		STARPU_PTHREAD_COND_BROADCAST(&handle->busy_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->busy_mutex);
	}

	/* The handle has been destroyed in between (eg. this was a temporary
	 * handle created for a reduction.) */
	if (STARPU_UNLIKELY(handle->lazy_unregister))
	{
		handle->lazy_unregister = 0;
		_starpu_spin_unlock(&handle->header_lock);
		_starpu_data_unregister(handle, 0, 1);
		/* Warning: in case we unregister the handle, we must be sure
		 * that the caller will not try to unlock the header after
		 * !*/
		return 1;
	}

	return 0;
}

static
void _starpu_check_if_valid_and_fetch_data_on_node(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, const char *origin)
{
	unsigned node;
	unsigned nnodes = starpu_memory_nodes_get_count();
	int valid = 0;

	for (node = 0; node < nnodes; node++)
	{
		if (handle->per_node[node].state != STARPU_INVALID)
		{
			/* we found a copy ! */
			valid = 1;
		}
	}
	if (valid)
	{
		int ret = _starpu_fetch_data_on_node(handle, handle->home_node, replicate, STARPU_R, 0, NULL, STARPU_FETCH, 0, NULL, NULL, 0, origin);
		STARPU_ASSERT(!ret);
		_starpu_release_data_on_node(handle, 0, STARPU_NONE, replicate);
	}
	else
	{
		_starpu_spin_lock(&handle->header_lock);
		if (!_starpu_notify_data_dependencies(handle, STARPU_NONE))
			_starpu_spin_unlock(&handle->header_lock);
	}
}

static void _starpu_data_unregister_fetch_data_callback(void *_arg)
{
	struct _starpu_unregister_callback_arg *arg = (struct _starpu_unregister_callback_arg *) _arg;

	starpu_data_handle_t handle = arg->handle;

	STARPU_ASSERT(handle);

	struct _starpu_data_replicate *replicate = &handle->per_node[arg->memory_node];

	_starpu_check_if_valid_and_fetch_data_on_node(handle, replicate, "_starpu_data_unregister_fetch_data_callback");

	/* unlock the caller */
	STARPU_PTHREAD_MUTEX_LOCK(&arg->mutex);
	arg->terminated = 1;
	STARPU_PTHREAD_COND_SIGNAL(&arg->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&arg->mutex);
}

void _starpu_data_set_unregister_hook(starpu_data_handle_t handle, _starpu_data_handle_unregister_hook func)
{
	STARPU_ASSERT(handle->unregister_hook == NULL);
	handle->unregister_hook = func;
}

/*
 * We are about to unregister this R/O data. There might be still other aliases,
 * in which case this returns 0. If not, users are not supposed to see it
 * any more, so detach it from their sight and return 1 to let unregistration happen.
 */
static int _starpu_ro_data_detach(starpu_data_handle_t handle)
{
	_starpu_spin_lock(&handle->header_lock);
	if (handle->aliases)
	{
		handle->aliases--;
		_starpu_spin_unlock(&handle->header_lock);
		return 0;
	}
	if (handle->readonly_dup)
	{
		STARPU_ASSERT(handle->readonly_dup->readonly_dup_of == handle);
		handle->readonly_dup->readonly_dup_of = NULL;
		handle->readonly_dup = NULL;
	}
	if (handle->readonly_dup_of)
	{
		STARPU_ASSERT(handle->readonly_dup_of->readonly_dup == handle);
		handle->readonly_dup_of->readonly_dup = NULL;
		handle->readonly_dup_of = NULL;
	}
	/* So that unregistration can use write dependencies to wait for
	 * anything to finish */
	handle->readonly = 0;
	_starpu_spin_unlock(&handle->header_lock);
	return 1;
}

/* Unregister the data handle, perhaps we don't need to update the home_node
 * (in that case coherent is set to 0)
 * nowait is for internal use when we already know for sure that we won't have to wait.
 */
static void _starpu_data_unregister(starpu_data_handle_t handle, unsigned coherent, unsigned nowait)
{
	STARPU_ASSERT(handle);
	STARPU_ASSERT_MSG(handle->nchildren == 0, "data %p needs to be unpartitioned before unregistration", handle);
	STARPU_ASSERT_MSG(handle->nplans == 0, "data %p needs its partition plans to be cleaned before unregistration", handle);
	STARPU_ASSERT_MSG(handle->partitioned == 0, "data %p needs its partitioned plans to be unpartitioned before unregistration", handle);
	/* TODO: also check that it has the latest coherency */
	STARPU_ASSERT(!(nowait && handle->busy_count != 0));

	if (!_starpu_ro_data_detach(handle))
		return;

	int sequential_consistency = handle->sequential_consistency;
	if (sequential_consistency && !nowait)
	{
		/* We will acquire it in write mode to catch all dependencies,
		 * but possibly it's not actually initialized. Fake it to avoid
		 getting caught doing it */
		handle->initialized = 1;
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_data_unregister must not be called from a task or callback, perhaps you can use starpu_data_unregister_submit instead");

		/* If sequential consistency is enabled, wait until data is available */
		if ((handle->nplans && !handle->nchildren) || handle->siblings)
			_starpu_data_partition_access_submit(handle, !handle->readonly);
		_starpu_data_wait_until_available(handle, handle->readonly?STARPU_R:STARPU_RW, "starpu_data_unregister");
	}

	if (coherent && !nowait)
	{
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_data_unregister must not be called from a task or callback, perhaps you can use starpu_data_unregister_submit instead");

		/* Fetch data in the home of the data to ensure we have a valid copy
		 * where we registered it */
		int home_node = handle->home_node;
		if (home_node >= 0)
		{
			struct _starpu_unregister_callback_arg arg = { 0 };
			arg.handle = handle;
			arg.memory_node = (unsigned)home_node;
			arg.terminated = 0;
			STARPU_PTHREAD_MUTEX_INIT0(&arg.mutex, NULL);
			STARPU_PTHREAD_COND_INIT0(&arg.cond, NULL);

			if (!_starpu_attempt_to_submit_data_request_from_apps(handle, STARPU_R,
					_starpu_data_unregister_fetch_data_callback, &arg))
			{
				/* no one has locked this data yet, so we proceed immediately */
				struct _starpu_data_replicate *home_replicate = &handle->per_node[home_node];
				_starpu_check_if_valid_and_fetch_data_on_node(handle, home_replicate, "_starpu_data_unregister");
			}
			else
			{
				STARPU_PTHREAD_MUTEX_LOCK(&arg.mutex);
				while (!arg.terminated)
					STARPU_PTHREAD_COND_WAIT(&arg.cond, &arg.mutex);
				STARPU_PTHREAD_MUTEX_UNLOCK(&arg.mutex);
			}
			STARPU_PTHREAD_MUTEX_DESTROY(&arg.mutex);
			STARPU_PTHREAD_COND_DESTROY(&arg.cond);
		}

		/* Driver porters: adding your driver here is optional, only needed for the support of multiple formats.  */

		/* If this handle uses a multiformat interface, we may have to convert
		 * this piece of data back into the CPU format.
		 * XXX : This is quite hacky, could we submit a task instead ?
		 */
		if (_starpu_data_is_multiformat_handle(handle) && (starpu_node_get_kind(handle->mf_node) != STARPU_CPU_RAM))
		{
			_STARPU_DEBUG("Conversion needed\n");
			void *buffers[1];
			struct starpu_multiformat_interface *format_interface;
			home_node = handle->home_node;
			if (home_node < 0 || (starpu_node_get_kind(home_node) != STARPU_CPU_RAM))
				home_node = STARPU_MAIN_RAM;
			format_interface = (struct starpu_multiformat_interface *) starpu_data_get_interface_on_node(handle, home_node);
			struct starpu_codelet *cl = NULL;
			enum starpu_node_kind node_kind = starpu_node_get_kind(handle->mf_node);

			switch (node_kind)
			{
#ifdef STARPU_USE_CUDA
				case STARPU_CUDA_RAM:
				{
					struct starpu_multiformat_data_interface_ops *mf_ops;
					mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
					cl = mf_ops->cuda_to_cpu_cl;
					break;
				}
#endif
#ifdef STARPU_USE_OPENCL
				case STARPU_OPENCL_RAM:
				{
					struct starpu_multiformat_data_interface_ops *mf_ops;
					mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
					cl = mf_ops->opencl_to_cpu_cl;
					break;
				}
#endif
				case STARPU_CPU_RAM:      /* Impossible ! */
				default:
					STARPU_ABORT();
			}
			buffers[0] = format_interface;

			_starpu_cl_func_t func = _starpu_task_get_cpu_nth_implementation(cl, 0);
			STARPU_ASSERT(func);
			func(buffers, NULL);
		}
	}

	/* Prevent any further unregistration */
	handle->magic = 0;

	_starpu_spin_lock(&handle->header_lock);
	if (!coherent)
	{
		/* Should we postpone the unregister operation ? */
		if (handle->lazy_unregister)
		{
			if (handle->busy_count > 0)
			{
				_starpu_spin_unlock(&handle->header_lock);
				return;
			}
			handle->lazy_unregister = 0;
		}
	}

	/* Tell holders of references that we're starting waiting */
	handle->busy_waiting = 1;
	_starpu_spin_unlock(&handle->header_lock);

	/* Request unmapping of any mapped data */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
		_starpu_data_unmap(handle, node);

retry_busy:
	/* Wait for all requests to finish (notably WT requests) */
	STARPU_PTHREAD_MUTEX_LOCK(&handle->busy_mutex);
	while (1)
	{
		/* Here helgrind would shout that this an unprotected access,
		 * but this is actually fine: all threads who do busy_count--
		 * are supposed to call _starpu_data_check_not_busy, which will
		 * wake us up through the busy_mutex/busy_cond. */
		if (!handle->busy_count)
			break;
		/* This is woken by _starpu_data_check_not_busy, always called
		 * after decrementing busy_count */
		STARPU_PTHREAD_COND_WAIT(&handle->busy_cond, &handle->busy_mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&handle->busy_mutex);

	/* Unregister MPI things after having waiting for MPI reqs etc. to settle down */
	if (handle->unregister_hook)
	{
		handle->unregister_hook(handle);
		handle->unregister_hook = NULL;
	}

	/* Wait for finished requests to release the handle */
	_starpu_spin_lock(&handle->header_lock);
	if (handle->busy_count)
	{
		/* Bad luck: some request went in in between, wait again... */
		_starpu_spin_unlock(&handle->header_lock);
		goto retry_busy;
	}

	size_t size = _starpu_data_get_alloc_size(handle);

	/* Destroy the data now */
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *local = &handle->per_node[node];
		STARPU_ASSERT(!local->refcnt);
		if (local->allocated)
		{
		/* free the data copy in a lazy fashion */
			if (local->automatically_allocated)
				_starpu_request_mem_chunk_removal(handle, local, node, size);
		}
	}
	if (handle->per_worker)
	{
		unsigned worker;
		unsigned nworkers = starpu_worker_get_count();
		for (worker = 0; worker < nworkers; worker++)
		{
			struct _starpu_data_replicate *local = &handle->per_worker[worker];
			STARPU_ASSERT(!local->refcnt);
			/* free the data copy in a lazy fashion */
			if (local->allocated && local->automatically_allocated)
				_starpu_request_mem_chunk_removal(handle, local, starpu_worker_get_memory_node(worker), size);
		}
	}
	_starpu_data_free_interfaces(handle);

	_starpu_memory_stats_free(handle);

	_starpu_spin_unlock(&handle->header_lock);
	_starpu_spin_destroy(&handle->header_lock);

	_starpu_data_clear_implicit(handle);
	free(handle->active_readonly_children);
	free(handle->active_readonly_nchildren);

	STARPU_PTHREAD_MUTEX_DESTROY(&handle->busy_mutex);
	STARPU_PTHREAD_COND_DESTROY(&handle->busy_cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&handle->sequential_consistency_mutex);
#ifdef STARPU_BUBBLE
	STARPU_PTHREAD_MUTEX_DESTROY(&handle->unpartition_mutex);
#endif

	STARPU_HG_ENABLE_CHECKING(handle->post_sync_tasks_cnt);
	STARPU_HG_ENABLE_CHECKING(handle->busy_count);

	_starpu_data_requester_prio_list_deinit(&handle->req_list);
	_starpu_data_requester_prio_list_deinit(&handle->reduction_req_list);

	if (handle->switch_cl)
	{
		free(handle->switch_cl->dyn_nodes);
		free(handle->switch_cl);
	}
	_STARPU_TRACE_HANDLE_DATA_UNREGISTER(handle);
	free(handle);
	(void)STARPU_ATOMIC_ADD(&nregistered, -1);
}

void starpu_data_unregister(starpu_data_handle_t handle)
{
	STARPU_ASSERT_MSG(handle->magic == 42, "data %p is invalid (was it already registered?)", handle);
	STARPU_ASSERT_MSG(!handle->lazy_unregister, "data %p can not be unregistered twice", handle);

	_starpu_data_unregister(handle, 1, 0);
}

void starpu_data_unregister_no_coherency(starpu_data_handle_t handle)
{
	STARPU_ASSERT_MSG(handle->magic == 42, "data %p is invalid (was it already registered?)", handle);

	_starpu_data_unregister(handle, 0, 0);
}

static void _starpu_data_unregister_submit_cb(void *arg)
{
	starpu_data_handle_t handle = arg;

	_starpu_spin_lock(&handle->header_lock);
	handle->lazy_unregister = 1;
	/* The handle should be busy since we are working on it.
	 * when we releases the handle below, it will be destroyed by
	 * _starpu_data_check_not_busy */
	STARPU_ASSERT(handle->busy_count);
	_starpu_spin_unlock(&handle->header_lock);

	starpu_data_release_on_node(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL);
}

void starpu_data_unregister_submit(starpu_data_handle_t handle)
{
	STARPU_ASSERT_MSG(handle->magic == 42, "data %p is invalid (was it already registered?)", handle);
	STARPU_ASSERT_MSG(!handle->lazy_unregister, "data %p can not be unregistered twice", handle);

	if (!_starpu_ro_data_detach(handle))
		return;

	/* Wait for all task dependencies on this handle before putting it for free */
	starpu_data_acquire_on_node_cb(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL, handle->initialized?STARPU_RW:STARPU_W, _starpu_data_unregister_submit_cb, handle);
}

static void _starpu_data_invalidate(void *data)
{
	starpu_data_handle_t handle = data;
	size_t size = _starpu_data_get_alloc_size(handle);
	_starpu_spin_lock(&handle->header_lock);

	//_STARPU_DEBUG("Really invalidating data %p\n", data);

#ifdef STARPU_DEBUG
	{
		/* There shouldn't be any pending request since we acquired the data in W mode */
		unsigned i, j, nnodes = starpu_memory_nodes_get_count();
		for (i = 0; i < nnodes; i++)
			for (j = 0; j < nnodes; j++)
				STARPU_ASSERT_MSG(!handle->per_node[i].request[j], "request for handle %p pending from %u to %u while invalidating data!", handle, j, i);
	}
#endif

	unsigned node;

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *local = &handle->per_node[node];

		if (local->mc && local->allocated && local->automatically_allocated)
		{
			unsigned mapping;
			for (mapping = 0; mapping < STARPU_MAXNODES; mapping++)
				if (handle->per_node[mapping].mapped == (int) node)
					break;

			if (mapping == STARPU_MAXNODES)
			{
				/* free the data copy in a lazy fashion */
				_starpu_request_mem_chunk_removal(handle, local, node, size);
			}
		}

		if (local->state != STARPU_INVALID)
			_STARPU_TRACE_DATA_STATE_INVALID(handle, node);
		local->state = STARPU_INVALID;
		local->initialized = 0;
	}

	if (handle->per_worker)
	{
		unsigned worker;
		unsigned nworkers = starpu_worker_get_count();
		for (worker = 0; worker < nworkers; worker++)
		{
			struct _starpu_data_replicate *local = &handle->per_worker[worker];

			if (local->mc && local->allocated && local->automatically_allocated)
				/* free the data copy in a lazy fashion */
				_starpu_request_mem_chunk_removal(handle, local, starpu_worker_get_memory_node(worker), size);

			local->state = STARPU_INVALID;
		}
	}

	_starpu_spin_unlock(&handle->header_lock);

	starpu_data_release_on_node(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL);
}

void starpu_data_invalidate(starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle);

	starpu_data_acquire_on_node(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL, STARPU_W);

	_starpu_data_invalidate(handle);

	handle->initialized = 0;
}

void starpu_data_invalidate_submit(starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle);

	starpu_data_acquire_on_node_cb(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL, STARPU_W, _starpu_data_invalidate, handle);

	handle->initialized = 0;
}

void _starpu_data_invalidate_submit_noplan(starpu_data_handle_t handle)
{
	STARPU_ASSERT(handle);

	starpu_data_acquire_on_node_cb(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL, STARPU_W | STARPU_NOPLAN, _starpu_data_invalidate, handle);

	handle->initialized = 0;
}

enum starpu_data_interface_id starpu_data_get_interface_id(starpu_data_handle_t handle)
{
	return handle->ops->interfaceid;
}

void *starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned memory_node)
{
	return handle->per_node[memory_node].data_interface;
}

int starpu_data_interface_get_next_id(void)
{
	_data_interface_number += 1;
	return _data_interface_number-1;
}

int starpu_data_pack_node(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT_MSG(handle->ops->pack_data, "The datatype interface %s (%d) does not have a pack operation", handle->ops->name, handle->ops->interfaceid);
	return handle->ops->pack_data(handle, node, ptr, count);
}

int starpu_data_pack(starpu_data_handle_t handle, void **ptr, starpu_ssize_t *count)
{
	return starpu_data_pack_node(handle, starpu_worker_get_local_memory_node(), ptr, count);
}

int starpu_data_peek_node(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT_MSG(handle->ops->peek_data, "The datatype interface %s (%d) does not have a peek operation", handle->ops->name, handle->ops->interfaceid);
	int ret;
	ret = handle->ops->peek_data(handle, node, ptr, count);
	return ret;
}

int starpu_data_peek(starpu_data_handle_t handle, void *ptr, size_t count)
{
	return starpu_data_peek_node(handle, starpu_worker_get_local_memory_node(), ptr, count);
}

int starpu_data_unpack_node(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT_MSG(handle->ops->unpack_data, "The datatype interface %s (%d) does not have an unpack operation", handle->ops->name, handle->ops->interfaceid);
	int ret;
	ret = handle->ops->unpack_data(handle, node, ptr, count);
	return ret;
}

int starpu_data_unpack(starpu_data_handle_t handle, void *ptr, size_t count)
{
	return starpu_data_unpack_node(handle, starpu_worker_get_local_memory_node(), ptr, count);
}

size_t starpu_data_get_size(starpu_data_handle_t handle)
{
	return handle->ops->get_size(handle);
}

size_t starpu_data_get_alloc_size(starpu_data_handle_t handle)
{
	if (handle->ops->get_alloc_size)
		return handle->ops->get_alloc_size(handle);
	else
		return handle->ops->get_size(handle);
}

void starpu_data_set_name(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, const char *name STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_TRACE_DATA_NAME(handle, name);
}

int starpu_data_get_home_node(starpu_data_handle_t handle)
{
	return handle->home_node;
}

void starpu_data_set_coordinates_array(starpu_data_handle_t handle, unsigned dimensions, int dims[])
{
	unsigned i;
	unsigned max_dimensions = sizeof(handle->coordinates)/sizeof(handle->coordinates[0]);

	if (dimensions > max_dimensions)
		dimensions = max_dimensions;

	handle->dimensions = dimensions;
	for (i = 0; i < dimensions; i++)
		handle->coordinates[i] = dims[i];

	_STARPU_TRACE_DATA_COORDINATES(handle, dimensions, dims);
}

void starpu_data_set_coordinates(starpu_data_handle_t handle, unsigned dimensions, ...)
{
	int dims[dimensions];
	unsigned i;
	va_list varg_list;

	va_start(varg_list, dimensions);
	for (i = 0; i < dimensions; i++)
		dims[i] = va_arg(varg_list, int);
	va_end(varg_list);

	starpu_data_set_coordinates_array(handle, dimensions, dims);
}

unsigned starpu_data_get_coordinates_array(starpu_data_handle_t handle, unsigned dimensions, int dims[])
{
	unsigned i;

	if (dimensions > handle->dimensions)
		dimensions = handle->dimensions;

	for (i = 0; i < dimensions; i++)
		dims[i] = handle->coordinates[i];

	return dimensions;
}

void starpu_data_print(starpu_data_handle_t handle, unsigned node, FILE *stream)
{
	if (handle->ops == NULL)
		fprintf(stream, "Undefined");
	else
	{
		switch (handle->ops->interfaceid)
		{
		case(STARPU_MATRIX_INTERFACE_ID):
			fprintf(stream, "Matrix");
			break;
		case(STARPU_BLOCK_INTERFACE_ID):
			fprintf(stream, "Block");
			break;
		case(STARPU_VECTOR_INTERFACE_ID):
			fprintf(stream, "Vector");
			break;
		case(STARPU_CSR_INTERFACE_ID):
			fprintf(stream, "CSR");
			break;
		case(STARPU_BCSR_INTERFACE_ID):
			fprintf(stream, "BCSR");
			break;
		case(STARPU_VARIABLE_INTERFACE_ID):
			fprintf(stream, "Variable");
			break;
		case(STARPU_VOID_INTERFACE_ID):
			fprintf(stream, "Void");
			break;
		case(STARPU_MULTIFORMAT_INTERFACE_ID):
			fprintf(stream, "Multfiformat");
			break;
		case(STARPU_COO_INTERFACE_ID):
			fprintf(stream, "COO");
			break;
		case(STARPU_TENSOR_INTERFACE_ID):
			fprintf(stream, "Tensor");
			break;
		case(STARPU_UNKNOWN_INTERFACE_ID):
			fprintf(stream, "UNKNOWN");
			break;
		default:
			fprintf(stream, "User interface with id %d", handle->ops->interfaceid);
			break;
		}
	}
	void *data_interface = NULL;
	if (starpu_data_test_if_allocated_on_node(handle, node))
		data_interface = starpu_data_get_interface_on_node(handle, node);
	if (starpu_data_test_if_allocated_on_node(handle, handle->home_node))
		data_interface = starpu_data_get_interface_on_node(handle, handle->home_node);
	if (handle->ops && handle->ops->describe && data_interface)
	{
		char buffer[1024];
		handle->ops->describe(data_interface, buffer, sizeof(buffer));
		fprintf(stream, " %s\n", buffer);
	}
	else
		fprintf(stream, "\n");

}
