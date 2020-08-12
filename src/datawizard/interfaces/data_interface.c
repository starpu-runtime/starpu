/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/uthash.h>
#include <common/starpu_spinlock.h>
#include <core/task.h>
#include <core/workers.h>
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>
#endif

/* Entry in the `registered_handles' hash table.  */
struct handle_entry
{
	UT_hash_handle hh;
	void *pointer;
	starpu_data_handle_t handle;
};

/* Hash table mapping host pointers to data handles.  */
static int nregistered, maxnregistered;
static struct handle_entry *registered_handles;
static struct _starpu_spinlock    registered_handles_lock;
static int _data_interface_number = STARPU_MAX_INTERFACE_ID;
starpu_arbiter_t _starpu_global_arbiter;

static void _starpu_data_unregister(starpu_data_handle_t handle, unsigned coherent, unsigned nowait);

void _starpu_data_interface_init(void)
{
	_starpu_spin_init(&registered_handles_lock);

	/* Just for testing purpose */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		_starpu_global_arbiter = starpu_arbiter_create();
}

void _starpu_data_interface_shutdown()
{
	struct handle_entry *entry=NULL, *tmp=NULL;

	if (registered_handles)
	{
		_STARPU_DISP("[warning] The application has not unregistered all data handles.\n");
	}

	_starpu_spin_destroy(&registered_handles_lock);

	HASH_ITER(hh, registered_handles, entry, tmp)
	{
		HASH_DEL(registered_handles, entry);
		nregistered--;
		free(entry);
	}

	if (starpu_get_env_number_default("STARPU_MAX_MEMORY_USE", 0))
		_STARPU_DISP("Memory used for %d data handles: %lu MiB\n", maxnregistered, (unsigned long) (maxnregistered * sizeof(struct _starpu_data_state)) >> 20);

	STARPU_ASSERT(nregistered == 0);
	registered_handles = NULL;
}

#ifdef STARPU_OPENMP
void _starpu_omp_unregister_region_handles(struct starpu_omp_region *region)
{
	_starpu_spin_lock(&region->registered_handles_lock);
	struct handle_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, (region->registered_handles), entry, tmp)
	{
		entry->handle->removed_from_context_hash = 1;
		HASH_DEL(region->registered_handles, entry);
		starpu_data_unregister(entry->handle);
		free(entry);
	}
	_starpu_spin_unlock(&region->registered_handles_lock);
}

void _starpu_omp_unregister_task_handles(struct starpu_omp_task *task)
{
	struct handle_entry *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, task->registered_handles, entry, tmp)
	{
		entry->handle->removed_from_context_hash = 1;
		HASH_DEL(task->registered_handles, entry);
		starpu_data_unregister(entry->handle);
		free(entry);
	}
}
#endif

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

		default:
			STARPU_ABORT();
			return NULL;
	}
}

/* Register the mapping from PTR to HANDLE.  If PTR is already mapped to
 * some handle, the new mapping shadows the previous one.   */
void _starpu_data_register_ram_pointer(starpu_data_handle_t handle, void *ptr)
{
	struct handle_entry *entry;

	_STARPU_MALLOC(entry, sizeof(*entry));

	entry->pointer = ptr;
	entry->handle = handle;

#ifdef STARPU_OPENMP
	struct starpu_omp_task *task = _starpu_omp_get_task();
	if (task)
	{
		if (task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT)
		{
			struct starpu_omp_region *parallel_region = task->owner_region;
			_starpu_spin_lock(&parallel_region->registered_handles_lock);
			HASH_ADD_PTR(parallel_region->registered_handles, pointer, entry);
			_starpu_spin_unlock(&parallel_region->registered_handles_lock);
		}
		else
		{
			HASH_ADD_PTR(task->registered_handles, pointer, entry);
		}
	}
	else
#endif
	{
		struct handle_entry *old_entry;

		_starpu_spin_lock(&registered_handles_lock);
		HASH_FIND_PTR(registered_handles, &ptr, old_entry);
		if (old_entry)
		{
			/* Already registered this pointer, avoid undefined
			 * behavior of duplicate in hash table */
			_starpu_spin_unlock(&registered_handles_lock);
			free(entry);
		}
		else
		{
			nregistered++;
			if (nregistered > maxnregistered)
				maxnregistered = nregistered;
			HASH_ADD_PTR(registered_handles, pointer, entry);
			_starpu_spin_unlock(&registered_handles_lock);
		}
	}
}

starpu_data_handle_t starpu_data_lookup(const void *ptr)
{
	starpu_data_handle_t result;

#ifdef STARPU_OPENMP
	struct starpu_omp_task *task = _starpu_omp_get_task();
	if (task)
	{
		if (task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT)
		{
			struct starpu_omp_region *parallel_region = task->owner_region;
			_starpu_spin_lock(&parallel_region->registered_handles_lock);
			{
				struct handle_entry *entry;

				HASH_FIND_PTR(parallel_region->registered_handles, &ptr, entry);
				if(STARPU_UNLIKELY(entry == NULL))
					result = NULL;
				else
					result = entry->handle;
			}
			_starpu_spin_unlock(&parallel_region->registered_handles_lock);
		}
		else
		{
			struct handle_entry *entry;

			HASH_FIND_PTR(task->registered_handles, &ptr, entry);
			if(STARPU_UNLIKELY(entry == NULL))
				result = NULL;
			else
				result = entry->handle;
		}
	}
	else
#endif
	{
		_starpu_spin_lock(&registered_handles_lock);
		{
			struct handle_entry *entry;

			HASH_FIND_PTR(registered_handles, &ptr, entry);
			if(STARPU_UNLIKELY(entry == NULL))
				result = NULL;
			else
				result = entry->handle;
		}
		_starpu_spin_unlock(&registered_handles_lock);
	}

	return result;
}

/*
 * Start monitoring a piece of data
 */

static void _starpu_register_new_data(starpu_data_handle_t handle,
					int home_node, uint32_t wt_mask)
{
	void *ptr;

	STARPU_ASSERT(handle);

	/* initialize the new lock */
	_starpu_data_requester_prio_list_init(&handle->req_list);
	handle->refcnt = 0;
	handle->unlocking_reqs = 0;
	handle->busy_count = 0;
	handle->busy_waiting = 0;
	STARPU_PTHREAD_MUTEX_INIT(&handle->busy_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&handle->busy_cond, NULL);
	_starpu_spin_init(&handle->header_lock);

	/* first take care to properly lock the data */
	_starpu_spin_lock(&handle->header_lock);

	/* there is no hierarchy yet */
	handle->nchildren = 0;
	handle->nplans = 0;
	handle->switch_cl = NULL;
	handle->partitioned = 0;
	handle->readonly = 0;
	handle->active = 1;
	handle->active_ro = 0;
	handle->root_handle = handle;
	handle->father_handle = NULL;
	handle->active_children = NULL;
	handle->active_readonly_children = NULL;
	handle->nactive_readonly_children = 0;
	handle->nsiblings = 0;
	handle->siblings = NULL;
	handle->sibling_index = 0; /* could be anything for the root */
	handle->depth = 1; /* the tree is just a node yet */
        handle->mpi_data = NULL; /* invalid until set */

	handle->is_not_important = 0;

	handle->sequential_consistency =
		starpu_data_get_default_sequential_consistency_flag();
	handle->initialized = home_node != -1;
	handle->ooc = 1;

	STARPU_PTHREAD_MUTEX_INIT(&handle->sequential_consistency_mutex, NULL);
	handle->last_submitted_mode = STARPU_R;
	handle->last_sync_task = NULL;
	handle->last_submitted_accessors.task = NULL;
	handle->last_submitted_accessors.next = &handle->last_submitted_accessors;
	handle->last_submitted_accessors.prev = &handle->last_submitted_accessors;
	handle->post_sync_tasks = NULL;

	/* Tell helgrind that the race in _starpu_unlock_post_sync_tasks is fine */
	STARPU_HG_DISABLE_CHECKING(handle->post_sync_tasks_cnt);
	handle->post_sync_tasks_cnt = 0;

	/* By default, there are no methods available to perform a reduction */
	handle->redux_cl = NULL;
	handle->init_cl = NULL;

	handle->reduction_refcnt = 0;
	_starpu_data_requester_prio_list_init(&handle->reduction_req_list);
	handle->reduction_tmp_handles = NULL;
	handle->write_invalidation_req = NULL;

#ifdef STARPU_USE_FXT
	handle->last_submitted_ghost_sync_id_is_valid = 0;
	handle->last_submitted_ghost_sync_id = 0;
	handle->last_submitted_ghost_accessors_id = NULL;
#endif

	handle->wt_mask = wt_mask;

	/* Store some values directly in the handle not to recompute them all
	 * the time. */
	handle->footprint = _starpu_compute_data_footprint(handle);

	handle->home_node = home_node;

	if (_starpu_global_arbiter)
		/* Just for testing purpose */
		starpu_data_assign_arbiter(handle, _starpu_global_arbiter);
	else
		handle->arbiter = NULL;
	_starpu_data_requester_prio_list_init(&handle->arbitered_req_list);
	handle->last_locality = -1;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *replicate;
		replicate = &handle->per_node[node];

		replicate->memory_node = node;
		replicate->relaxed_coherency = 0;
		replicate->refcnt = 0;

		if ((int) node == home_node)
		{
			/* this is the home node with the only valid copy */
			replicate->state = STARPU_OWNER;
			replicate->allocated = 1;
			replicate->automatically_allocated = 0;
			replicate->initialized = 1;
		}
		else
		{
			/* the value is not available here yet */
			replicate->state = STARPU_INVALID;
			replicate->allocated = 0;
			replicate->initialized = 0;
		}
	}

	handle->per_worker = NULL;
	handle->user_data = NULL;

	/* now the data is available ! */
	_starpu_spin_unlock(&handle->header_lock);

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (starpu_node_get_kind(node) != STARPU_CPU_RAM)
			continue;

		ptr = starpu_data_handle_to_pointer(handle, node);
		if (ptr != NULL)
			_starpu_data_register_ram_pointer(handle, ptr);
	}
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
		unsigned node;
		replicate = &handle->per_worker[worker];
		replicate->allocated = 0;
		replicate->automatically_allocated = 0;
		replicate->state = STARPU_INVALID;
		replicate->refcnt = 0;
		replicate->handle = handle;
		replicate->requested = 0;

		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			replicate->request[node] = NULL;
		}

		/* Assuming being used for SCRATCH for now, patched when entering REDUX mode */
		replicate->relaxed_coherency = 1;
		replicate->initialized = 0;
		replicate->memory_node = starpu_worker_get_memory_node(worker);

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
	handle->ops = interface_ops;
	handle->mf_node = mf_node;
	handle->mpi_data = NULL;
	handle->partition_automatic_disabled = 0;

	size_t interfacesize = interface_ops->interface_size;

	_starpu_memory_stats_init(handle);
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

	/* fill the interface fields with the appropriate method */
	STARPU_ASSERT(ops->register_data_handle);
	ops->register_data_handle(handle, home_node, data_interface);

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

/*
 * Stop monitoring a piece of data
 */
void _starpu_data_unregister_ram_pointer(starpu_data_handle_t handle, unsigned node)
{
	if (starpu_node_get_kind(node) != STARPU_CPU_RAM)
		return;

#ifdef STARPU_OPENMP
	if (handle->removed_from_context_hash)
		return;
#endif
	const void *ram_ptr = starpu_data_handle_to_pointer(handle, node);

	if (ram_ptr != NULL)
	{
		/* Remove the PTR -> HANDLE mapping.  If a mapping from PTR
		 * to another handle existed before (e.g., when using
		 * filters), it becomes visible again.  */
		struct handle_entry *entry;
#ifdef STARPU_OPENMP
		struct starpu_omp_task *task = _starpu_omp_get_task();
		if (task)
		{
			if (task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT)
			{
				struct starpu_omp_region *parallel_region = task->owner_region;
				_starpu_spin_lock(&parallel_region->registered_handles_lock);
				HASH_FIND_PTR(parallel_region->registered_handles, &ram_ptr, entry);
				STARPU_ASSERT(entry != NULL);
				HASH_DEL(registered_handles, entry);
				_starpu_spin_unlock(&parallel_region->registered_handles_lock);
			}
			else
			{
				HASH_FIND_PTR(task->registered_handles, &ram_ptr, entry);
				STARPU_ASSERT(entry != NULL);
				HASH_DEL(task->registered_handles, entry);
			}
		}
		else
#endif
		{

			_starpu_spin_lock(&registered_handles_lock);
			HASH_FIND_PTR(registered_handles, &ram_ptr, entry);
			if (entry)
			{
				if (entry->handle == handle)
				{
					nregistered--;
					HASH_DEL(registered_handles, entry);
				}
				else
					/* don't free it, it's not ours */
					entry = NULL;
			}
			_starpu_spin_unlock(&registered_handles_lock);
		}
		free(entry);
	}
}

void _starpu_data_free_interfaces(starpu_data_handle_t handle)
{
	unsigned node;
	unsigned nworkers = starpu_worker_get_count();

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
		int ret = _starpu_fetch_data_on_node(handle, handle->home_node, replicate, STARPU_R, 0, STARPU_FETCH, 0, NULL, NULL, 0, origin);
		STARPU_ASSERT(!ret);
		_starpu_release_data_on_node(handle, 0, replicate);
	}
	else
	{
		_starpu_spin_lock(&handle->header_lock);
		if (!_starpu_notify_data_dependencies(handle))
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

	int sequential_consistency = handle->sequential_consistency;
	if (sequential_consistency && !nowait)
	{
		/* We will acquire it in write mode to catch all dependencies,
		 * but possibly it's not actually initialized. Fake it to avoid
		 getting caught doing it */
		handle->initialized = 1;
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_data_unregister must not be called from a task or callback, perhaps you can use starpu_data_unregister_submit instead");

		/* If sequential consistency is enabled, wait until data is available */
		_starpu_data_wait_until_available(handle, STARPU_RW, "starpu_data_unregister");
	}

	if (coherent && !nowait)
	{
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_data_unregister must not be called from a task or callback, perhaps you can use starpu_data_unregister_submit instead");

		/* Fetch data in the home of the data to ensure we have a valid copy
		 * where we registered it */
		int home_node = handle->home_node;
		if (home_node >= 0)
		{
			struct _starpu_unregister_callback_arg arg;
			arg.handle = handle;
			arg.memory_node = (unsigned)home_node;
			arg.terminated = 0;
			STARPU_PTHREAD_MUTEX_INIT(&arg.mutex, NULL);
			STARPU_PTHREAD_COND_INIT(&arg.cond, NULL);

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
#ifdef STARPU_USE_MIC
				case STARPU_MIC_RAM:
				{
					struct starpu_multiformat_data_interface_ops *mf_ops;
					mf_ops = (struct starpu_multiformat_data_interface_ops *) handle->ops->get_mf_ops(format_interface);
					cl = mf_ops->mic_to_cpu_cl;
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
		if ((handle->busy_count > 0) && handle->lazy_unregister)
		{
			_starpu_spin_unlock(&handle->header_lock);
			return;
		}
	}

	/* Tell holders of references that we're starting waiting */
	handle->busy_waiting = 1;
	_starpu_spin_unlock(&handle->header_lock);

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
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *local = &handle->per_node[node];
		if (local->allocated)
		{
			_starpu_data_unregister_ram_pointer(handle, node);

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

	STARPU_PTHREAD_MUTEX_DESTROY(&handle->busy_mutex);
	STARPU_PTHREAD_COND_DESTROY(&handle->busy_cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&handle->sequential_consistency_mutex);

	STARPU_HG_ENABLE_CHECKING(handle->post_sync_tasks_cnt);
	STARPU_HG_ENABLE_CHECKING(handle->busy_count);

	if (handle->switch_cl)
	{
		free(handle->switch_cl->dyn_nodes);
		free(handle->switch_cl);
	}
	_STARPU_TRACE_HANDLE_DATA_UNREGISTER(handle);
	free(handle);
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
			_starpu_data_unregister_ram_pointer(handle, node);

			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(handle, local, node, size);
		}

		if (local->state != STARPU_INVALID)
			_STARPU_TRACE_DATA_STATE_INVALID(handle, node);
		local->state = STARPU_INVALID;
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

int starpu_data_pack(starpu_data_handle_t handle, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT_MSG(handle->ops->pack_data, "The datatype interface %s (%d) does not have a pack operation", handle->ops->name, handle->ops->interfaceid);
	return handle->ops->pack_data(handle, starpu_worker_get_local_memory_node(), ptr, count);
}

int starpu_data_unpack(starpu_data_handle_t handle, void *ptr, size_t count)
{
	STARPU_ASSERT_MSG(handle->ops->unpack_data, "The datatype interface %s (%d) does not have an unpack operation", handle->ops->name, handle->ops->interfaceid);
	int ret;
	ret = handle->ops->unpack_data(handle, starpu_worker_get_local_memory_node(), ptr, count);
	return ret;
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
