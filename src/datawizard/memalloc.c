/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <datawizard/memory_manager.h>
#include <datawizard/memalloc.h>
#include <datawizard/footprint.h>
#include <core/disk.h>
#include <starpu.h>

/* This per-node spinlock protect lru_list */
static struct _starpu_spinlock lru_rwlock[STARPU_MAXNODES];

/* Last Recently used memory chunkgs */
static struct _starpu_mem_chunk_lru_list *starpu_lru_list[STARPU_MAXNODES];


/* This per-node RW-locks protect mc_list and memchunk_cache entries */
/* Note: handle header lock is always taken before this */
static starpu_pthread_rwlock_t mc_rwlock[STARPU_MAXNODES];

/* Potentially in use memory chunks */
static struct _starpu_mem_chunk_list *mc_list[STARPU_MAXNODES];

/* Explicitly caches memory chunks that can be reused */
static struct _starpu_mem_chunk_list *memchunk_cache[STARPU_MAXNODES];


/* When reclaiming memory to allocate, we reclaim MAX(what_is_to_reclaim_on_device, data_size_coefficient*data_size) */
const unsigned starpu_memstrategy_data_size_coefficient=2;

static void starpu_lru(unsigned node);
static int get_better_disk_can_accept_size(starpu_data_handle_t handle, unsigned node);
static unsigned choose_target(starpu_data_handle_t handle, unsigned node);

void _starpu_init_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		STARPU_PTHREAD_RWLOCK_INIT(&mc_rwlock[i], NULL);
		_starpu_spin_init(&lru_rwlock[i]);
		mc_list[i] = _starpu_mem_chunk_list_new();
		starpu_lru_list[i] = _starpu_mem_chunk_lru_list_new();
		memchunk_cache[i] = _starpu_mem_chunk_list_new();
	}
}

void _starpu_deinit_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		_starpu_mem_chunk_list_delete(mc_list[i]);
		_starpu_mem_chunk_list_delete(memchunk_cache[i]);
		_starpu_mem_chunk_lru_list_delete(starpu_lru_list[i]);
		_starpu_spin_destroy(&lru_rwlock[i]);
		STARPU_PTHREAD_RWLOCK_DESTROY(&mc_rwlock[i]);
	}
}

/*
 *	Manipulate subtrees
 */

static void unlock_all_subtree(starpu_data_handle_t handle)
{
	/* lock all sub-subtrees children
	 * Note that this is done in the reverse order of the
	 * lock_all_subtree so that we avoid deadlock */
	unsigned i;
	for (i =0; i < handle->nchildren; i++)
	{
		unsigned child = handle->nchildren - 1 - i;
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		unlock_all_subtree(child_handle);
	}

	_starpu_spin_unlock(&handle->header_lock);
}

static int lock_all_subtree(starpu_data_handle_t handle)
{
	int child;

	/* lock parent */
	if (_starpu_spin_trylock(&handle->header_lock))
		/* the handle is busy, abort */
		return 0;

	/* lock all sub-subtrees children */
	for (child = 0; child < (int) handle->nchildren; child++)
	{
		if (!lock_all_subtree(starpu_data_get_child(handle, child))) {
			/* Some child is busy, abort */
			while (--child >= 0)
				/* Unlock what we have already uselessly locked */
				unlock_all_subtree(starpu_data_get_child(handle, child));
			return 0;
		}
	}

	return 1;
}

static unsigned may_free_subtree(starpu_data_handle_t handle, unsigned node)
{
	/* we only free if no one refers to the leaf */
	uint32_t refcnt = _starpu_get_data_refcnt(handle, node);
	if (refcnt)
		return 0;

	if (!handle->nchildren)
		return 1;

	/* look into all sub-subtrees children */
	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		unsigned res;
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		res = may_free_subtree(child_handle, node);
		if (!res) return 0;
	}

	/* no problem was found */
	return 1;
}

static void transfer_subtree_to_node(starpu_data_handle_t handle, unsigned src_node,
				     unsigned dst_node)
{
	unsigned i;
	unsigned last = 0;
	unsigned cnt;
	int ret;

	STARPU_ASSERT(dst_node != src_node);

	if (handle->nchildren == 0)
	{
		struct _starpu_data_replicate *src_replicate = &handle->per_node[src_node];
		struct _starpu_data_replicate *dst_replicate = &handle->per_node[dst_node];

		/* this is a leaf */
		switch(src_replicate->state)
		{
		case STARPU_OWNER:
			/* the local node has the only copy */
			/* the owner is now the destination_node */
			src_replicate->state = STARPU_INVALID;
			dst_replicate->state = STARPU_OWNER;

#ifdef STARPU_DEVEL
#warning we should use requests during memory reclaim
#endif
			/* TODO use request !! */
			/* Take temporary references on the replicates */
			_starpu_spin_checklocked(&handle->header_lock);
			src_replicate->refcnt++;
			dst_replicate->refcnt++;
			handle->busy_count+=2;

			ret = _starpu_driver_copy_data_1_to_1(handle, src_replicate, dst_replicate, 0, NULL, 1);
			STARPU_ASSERT(ret == 0);

			src_replicate->refcnt--;
			dst_replicate->refcnt--;
			STARPU_ASSERT(handle->busy_count >= 2);
			handle->busy_count -= 2;
			ret = _starpu_data_check_not_busy(handle);
			STARPU_ASSERT(ret == 0);

			break;
		case STARPU_SHARED:
			/* some other node may have the copy */
			src_replicate->state = STARPU_INVALID;

			/* count the number of copies */
			cnt = 0;
			for (i = 0; i < STARPU_MAXNODES; i++)
			{
				if (handle->per_node[i].state == STARPU_SHARED)
				{
					cnt++;
					last = i;
				}
			}
			STARPU_ASSERT(cnt > 0);

			if (cnt == 1)
				handle->per_node[last].state = STARPU_OWNER;

			break;
		case STARPU_INVALID:
			/* nothing to be done */
			break;
		default:
			STARPU_ABORT();
			break;
		}
	}
	else
	{
		/* lock all sub-subtrees children */
		unsigned child;
		for (child = 0; child < handle->nchildren; child++)
		{
			starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
			transfer_subtree_to_node(child_handle, src_node, dst_node);
		}
	}
}

static void notify_handle_children(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned node)
{
	unsigned child;

	replicate->allocated = 0;

	/* XXX why do we need that ? */
	replicate->automatically_allocated = 0;

	for (child = 0; child < handle->nchildren; child++)
	{
		/* Notify children that their buffer has been deallocated too */
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		notify_handle_children(child_handle, &child_handle->per_node[node], node);
	}
}

static size_t free_memory_on_node(struct _starpu_mem_chunk *mc, unsigned node)
{
	size_t freed = 0;

	STARPU_ASSERT(mc->ops);
	STARPU_ASSERT(mc->ops->free_data_on_node);

	starpu_data_handle_t handle = mc->data;

	struct _starpu_data_replicate *replicate = mc->replicate;

	if (handle)
		_starpu_spin_checklocked(&handle->header_lock);

	if (mc->automatically_allocated &&
		(!handle || replicate->refcnt == 0))
	{
		if (handle)
			STARPU_ASSERT(replicate->allocated);

#if defined(STARPU_USE_CUDA) && defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
		if (starpu_node_get_kind(node) == STARPU_CUDA_RAM)
		{
			/* To facilitate the design of interface, we set the
			 * proper CUDA device in case it is needed. This avoids
			 * having to set it again in the free method of each
			 * interface. */
			starpu_cuda_set_device(_starpu_memory_node_get_devid(node));
		}
#endif

		mc->ops->free_data_on_node(mc->chunk_interface, node);

		if (handle)
			notify_handle_children(handle, replicate, node);

		freed = mc->size;

		if (handle)
			STARPU_ASSERT(replicate->refcnt == 0);
	}

	return freed;
}



static size_t do_free_mem_chunk(struct _starpu_mem_chunk *mc, unsigned node)
{
	size_t size;
	starpu_data_handle_t handle = mc->data;

	if (handle) {
		_starpu_spin_checklocked(&handle->header_lock);
		mc->size = _starpu_data_get_size(handle);
	}

	mc->replicate->mc=NULL;

	/* free the actual buffer */
	size = free_memory_on_node(mc, node);

	/* remove the mem_chunk from the list */
	_starpu_mem_chunk_list_erase(mc_list[node], mc);

	free(mc->chunk_interface);
	_starpu_mem_chunk_delete(mc);

	return size;
}

/* This function is called for memory chunks that are possibly in used (ie. not
 * in the cache). They should therefore still be associated to a handle. */
static size_t try_to_free_mem_chunk(struct _starpu_mem_chunk *mc, unsigned node)
{
	size_t freed = 0;

	starpu_data_handle_t handle;
	handle = mc->data;
	STARPU_ASSERT(handle);

	/* This data should be written through to this node, avoid dropping it! */
	if (handle->wt_mask & (1<<node))
		return 0;

	/* This data was registered from this node, we will not be able to drop it anyway */
	if ((int) node == handle->home_node)
		return 0;

	/* REDUX memchunk */
	if (mc->relaxed_coherency == 2)
	{
		/* TODO: reduce it back to e.g. main memory */
	}
	else
	/* Either it's a "relaxed coherency" memchunk (SCRATCH), or it's a
	 * memchunk that could be used with filters. */
	if (mc->relaxed_coherency == 1)
	{
		STARPU_ASSERT(mc->replicate);

		if (_starpu_spin_trylock(&handle->header_lock))
			/* Handle is busy, abort */
			return 0;

		if (mc->replicate->refcnt == 0)
		{
			/* Note taht there is no need to transfer any data or
			 * to update the status in terms of MSI protocol
			 * because this memchunk is associated to a replicate
			 * in "relaxed coherency" mode. */
			freed = do_free_mem_chunk(mc, node);
		}

		_starpu_spin_unlock(&handle->header_lock);
	}
	/* try to lock all the subtree */
	else if (lock_all_subtree(handle))
	{
		/* check if they are all "free" */
		if (may_free_subtree(handle, node))
		{
			int target = -1;

			/* XXX Considering only owner to invalidate */

			STARPU_ASSERT(handle->per_node[node].refcnt == 0);

			/* in case there was nobody using that buffer, throw it
			 * away after writing it back to main memory */
			
			/* choose the best target */
			target = choose_target(handle, node);

			if (target != -1) {
#ifdef STARPU_MEMORY_STATS
				if (handle->per_node[node].state == STARPU_OWNER)
					_starpu_memory_handle_stats_invalidated(handle, node);
#endif
				transfer_subtree_to_node(handle, node, target);
#ifdef STARPU_MEMORY_STATS
				_starpu_memory_handle_stats_loaded_owner(handle, target);
#endif

				STARPU_ASSERT(handle->per_node[node].refcnt == 0);

				/* now the actual buffer may be freed */
				freed = do_free_mem_chunk(mc, node);
			}
		}

		/* unlock the tree */
		unlock_all_subtree(handle);
	}
	return freed;
}

#ifdef STARPU_USE_ALLOCATION_CACHE
/* We assume that mc_rwlock[node] is taken. is_already_in_mc_list indicates
 * that the mc is already in the list of buffers that are possibly used, and
 * therefore not in the cache. */
static void reuse_mem_chunk(unsigned node, struct _starpu_data_replicate *new_replicate, struct _starpu_mem_chunk *mc, unsigned is_already_in_mc_list)
{
	/* we found an appropriate mem chunk: so we get it out
	 * of the "to free" list, and reassign it to the new
	 * piece of data */

	if (!is_already_in_mc_list)
	{
		_starpu_mem_chunk_list_erase(memchunk_cache[node], mc);
	}

	struct _starpu_data_replicate *old_replicate = mc->replicate;
	old_replicate->allocated = 0;
	old_replicate->automatically_allocated = 0;
	old_replicate->initialized = 0;

	new_replicate->allocated = 1;
	new_replicate->automatically_allocated = 1;
	new_replicate->initialized = 0;

	STARPU_ASSERT(new_replicate->data_interface);
	STARPU_ASSERT(mc->chunk_interface);
	memcpy(new_replicate->data_interface, mc->chunk_interface, old_replicate->handle->ops->interface_size);

	mc->data = new_replicate->handle;
	/* mc->ops, mc->footprint and mc->interface should be
 	 * unchanged ! */

	/* reinsert the mem chunk in the list of active memory chunks */
	if (!is_already_in_mc_list)
	{
		_starpu_mem_chunk_list_push_front(mc_list[node], mc);
	}
}

static unsigned try_to_reuse_mem_chunk(struct _starpu_mem_chunk *mc, unsigned node, struct _starpu_data_replicate *replicate, unsigned is_already_in_mc_list)
{
	unsigned success = 0;

	starpu_data_handle_t old_data;

	old_data = mc->data;

	STARPU_ASSERT(old_data);

	/* try to lock all the subtree */
	/* and check if they are all "free" */
	if (lock_all_subtree(old_data))
	{
		if (may_free_subtree(old_data, node))
		{
			success = 1;

			/* in case there was nobody using that buffer, throw it
			 * away after writing it back to main memory */
			transfer_subtree_to_node(old_data, node, 0);

			/* now replace the previous data */
			reuse_mem_chunk(node, replicate, mc, is_already_in_mc_list);
		}

		/* unlock the tree */
		unlock_all_subtree(old_data);
	}

	return success;
}

static int _starpu_data_interface_compare(void *data_interface_a, struct starpu_data_interface_ops *ops_a,
                                          void *data_interface_b, struct starpu_data_interface_ops *ops_b)
{
	if (ops_a->interfaceid != ops_b->interfaceid)
		return -1;

	int ret = ops_a->compare(data_interface_a, data_interface_b);

	return ret;
}

/* This function must be called with mc_rwlock[node] taken in write mode */
static struct _starpu_mem_chunk *_starpu_memchunk_cache_lookup_locked(unsigned node, starpu_data_handle_t handle)
{
	uint32_t footprint = _starpu_compute_data_footprint(handle);

	/* go through all buffers in the cache */
	struct _starpu_mem_chunk *mc;
	for (mc = _starpu_mem_chunk_list_begin(memchunk_cache[node]);
	     mc != _starpu_mem_chunk_list_end(memchunk_cache[node]);
	     mc = _starpu_mem_chunk_list_next(mc))
	{
		if (mc->footprint == footprint)
		{
			/* Is that a false hit ? (this is _very_ unlikely) */
			if (_starpu_data_interface_compare(handle->per_node[node].data_interface, handle->ops, mc->chunk_interface, mc->ops))
				continue;

			/* Cache hit */

			/* Remove from the cache */
			_starpu_mem_chunk_list_erase(memchunk_cache[node], mc);
			return mc;
		}
	}

	/* This is a cache miss */
	return NULL;
}

/* this function looks for a memory chunk that matches a given footprint in the
 * list of mem chunk that need to be freed. This function must be called with
 * mc_rwlock[node] taken in write mode. */
static unsigned try_to_find_reusable_mem_chunk(unsigned node, starpu_data_handle_t data, struct _starpu_data_replicate *replicate, uint32_t footprint)
{
	struct _starpu_mem_chunk *mc, *next_mc;

	/* go through all buffers in the cache */
	mc = _starpu_memchunk_cache_lookup_locked(node, data);
	if (mc)
	{
		/* We found an entry in the cache so we can reuse it */
		reuse_mem_chunk(node, replicate, mc, 0);
		return 1;
	}

	/* now look for some non essential data in the active list */
	for (mc = _starpu_mem_chunk_list_begin(mc_list[node]);
	     mc != _starpu_mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* there is a risk that the memory chunk is freed before next
		 * iteration starts: so we compute the next element of the list
		 * now */
		next_mc = _starpu_mem_chunk_list_next(mc);

		if (mc->data->is_not_important && (mc->footprint == footprint))
		{
//			fprintf(stderr, "found a candidate ...\n");
			if (try_to_reuse_mem_chunk(mc, node, replicate, 1))
				return 1;
		}
	}

	return 0;
}
#endif

/*
 * Free the memory chuncks that are explicitely tagged to be freed. The
 * mc_rwlock[node] rw-lock should be taken prior to calling this function.
 */
static size_t flush_memchunk_cache(unsigned node, size_t reclaim)
{
	struct _starpu_mem_chunk *mc;
	struct _starpu_mem_chunk_list *busy_memchunk_cache;

	size_t freed = 0;

	if (_starpu_mem_chunk_list_empty(memchunk_cache[node]))
		return 0;

	busy_memchunk_cache = _starpu_mem_chunk_list_new();

	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);
	while (!_starpu_mem_chunk_list_empty(memchunk_cache[node])) {
		mc = _starpu_mem_chunk_list_pop_front(memchunk_cache[node]);
		starpu_data_handle_t handle = mc->data;

		if (handle)
			if (_starpu_spin_trylock(&handle->header_lock)) {
				/* The handle is still busy, leave this chunk for later */
				_starpu_mem_chunk_list_push_front(busy_memchunk_cache, mc);
				continue;
			}

		freed += free_memory_on_node(mc, node);
		if (handle)
			_starpu_spin_unlock(&handle->header_lock);

		free(mc->chunk_interface);
		_starpu_mem_chunk_delete(mc);

		if (reclaim && freed >= reclaim)
			break;
	}
	_starpu_mem_chunk_list_push_list_front(busy_memchunk_cache, memchunk_cache[node]);
	_starpu_mem_chunk_list_delete(busy_memchunk_cache);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
	return freed;
}

/*
 * Try to free the buffers currently in use on the memory node. If the force
 * flag is set, the memory is freed regardless of coherency concerns (this
 * should only be used at the termination of StarPU for instance). The
 * mc_rwlock[node] rw-lock should be taken prior to calling this function.
 */
static size_t free_potentially_in_use_mc(unsigned node, unsigned force, size_t reclaim)
{
	size_t freed = 0;

	struct _starpu_mem_chunk *mc, *next_mc;

	/*
	 * We have to unlock mc_rwlock before locking header_lock, so we have
	 * to be careful with the list.  We try to do just one pass, by
	 * remembering the next mc to be tried. If it gets dropped, we restart
	 * from zero. So we continue until we go through the whole list without
	 * finding anything to free.
	 */

restart:
	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	for (mc = _starpu_mem_chunk_list_begin(mc_list[node]);
	     mc != _starpu_mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* mc hopefully gets out of the list, we thus need to prefetch
		 * the next element */
		next_mc = _starpu_mem_chunk_list_next(mc);

		if (!force)
		{
			freed += try_to_free_mem_chunk(mc, node);

			if (reclaim && freed >= reclaim)
				break;
		}
		else
		{
			starpu_data_handle_t handle = mc->data;

			if (_starpu_spin_trylock(&handle->header_lock))
			{
				/* Ergl. We are shutting down, but somebody is
				 * still locking the handle. That's not
				 * supposed to happen, but better be safe by
				 * letting it go through. */
				STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
				goto restart;
			}

			/* We must free the memory now, because we are
			 * terminating the drivers: note that data coherency is
			 * not maintained in that case ! */
			freed += do_free_mem_chunk(mc, node);

			_starpu_spin_unlock(&handle->header_lock);
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);

	return freed;
}

size_t _starpu_memory_reclaim_generic(unsigned node, unsigned force, size_t reclaim)
{
	size_t freed = 0;

	starpu_lru(node);

	/* remove all buffers for which there was a removal request */
	freed += flush_memchunk_cache(node, reclaim);

	/* try to free all allocated data potentially in use */
	if (reclaim && freed<reclaim)
		freed += free_potentially_in_use_mc(node, force, reclaim);

	return freed;

}

/*
 * This function frees all the memory that was implicitely allocated by StarPU
 * (for the data replicates). This is not ensuring data coherency, and should
 * only be called while StarPU is getting shut down.
 */
size_t _starpu_free_all_automatically_allocated_buffers(unsigned node)
{
	return _starpu_memory_reclaim_generic(node, 1, 0);
}

static struct _starpu_mem_chunk *_starpu_memchunk_init(struct _starpu_data_replicate *replicate, size_t interface_size, unsigned automatically_allocated)
{
	struct _starpu_mem_chunk *mc = _starpu_mem_chunk_new();
	starpu_data_handle_t handle = replicate->handle;

	STARPU_ASSERT(handle);
	STARPU_ASSERT(handle->ops);

	mc->data = handle;
	mc->footprint = _starpu_compute_data_footprint(handle);
	mc->ops = handle->ops;
	mc->automatically_allocated = automatically_allocated;
	mc->relaxed_coherency = replicate->relaxed_coherency;
	mc->replicate = replicate;
	mc->replicate->mc = mc;

	/* Save a copy of the interface */
	mc->chunk_interface = malloc(interface_size);
	STARPU_ASSERT(mc->chunk_interface);
	memcpy(mc->chunk_interface, replicate->data_interface, interface_size);

	return mc;
}

static void register_mem_chunk(struct _starpu_data_replicate *replicate, unsigned automatically_allocated)
{
	unsigned dst_node = replicate->memory_node;

	struct _starpu_mem_chunk *mc;

	/* the interface was already filled by ops->allocate_data_on_node */
	size_t interface_size = replicate->handle->ops->interface_size;

	/* Put this memchunk in the list of memchunk in use */
	mc = _starpu_memchunk_init(replicate, interface_size, automatically_allocated);

	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[dst_node]);

	_starpu_mem_chunk_list_push_back(mc_list[dst_node], mc);

	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[dst_node]);
}

/* This function is called when the handle is destroyed (eg. when calling
 * unregister or unpartition). It puts all the memchunks that refer to the
 * specified handle into the cache.
 */
void _starpu_request_mem_chunk_removal(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned node, size_t size)
{
	struct _starpu_mem_chunk *mc = replicate->mc;

	STARPU_ASSERT(mc->data == handle);

	/* Record the allocated size, so that later in memory
	 * reclaiming we can estimate how much memory we free
	 * by freeing this.  */
	mc->size = size;

	/* This memchunk doesn't have to do with the data any more. */
	replicate->mc = NULL;
	replicate->allocated = 0;
	replicate->automatically_allocated = 0;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	mc->data = NULL;
	/* remove it from the main list */
	_starpu_mem_chunk_list_erase(mc_list[node], mc);

	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);

	/* We would only flush the RAM nodes cache if memory gets tight, either
	 * because StarPU automatically knows the total memory size of the
	 * machine, or because the user has provided a limitation.
	 *
	 * We don't really want the former scenario to be eating a lot of
	 * memory just for caching allocations. Allocating main memory is cheap
	 * anyway.
	 */
	/* This is particularly important when
	 * STARPU_USE_ALLOCATION_CACHE is not enabled, as we
	 * wouldn't even re-use these allocations! */
	if (starpu_node_get_kind(node) == STARPU_CPU_RAM)
	{
		/* Free data immediately */
		free_memory_on_node(mc, node);

		free(mc->chunk_interface);
		_starpu_mem_chunk_delete(mc);
	}
	else
	{
		/* put it in the list of buffers to be removed */
		STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);
		_starpu_mem_chunk_list_push_front(memchunk_cache[node], mc);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
	}
}

/*
 * In order to allocate a piece of data, we try to reuse existing buffers if
 * its possible.
 *	1 - we try to reuse a memchunk that is explicitely unused.
 *	2 - we go through the list of memory chunks and find one that is not
 *	referenced and that has the same footprint to reuse it.
 *	3 - we call the usual driver's alloc method
 *	4 - we go through the list of memory chunks and release those that are
 *	not referenced (or part of those).
 *
 */

static starpu_ssize_t _starpu_allocate_interface(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned dst_node, unsigned is_prefetch)
{
	unsigned attempts = 0;
	starpu_ssize_t allocated_memory;
	int ret;

	_starpu_spin_checklocked(&handle->header_lock);

	_starpu_data_allocation_inc_stats(dst_node);

#ifdef STARPU_USE_ALLOCATION_CACHE
	/* perhaps we can directly reuse a buffer in the free-list */
	uint32_t footprint = _starpu_compute_data_footprint(handle);

	_STARPU_TRACE_START_ALLOC_REUSE(dst_node);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[dst_node]);

	if (try_to_find_reusable_mem_chunk(dst_node, handle, replicate, footprint))
	{
		STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[dst_node]);
		_starpu_allocation_cache_hit(dst_node);
		starpu_ssize_t data_size = _starpu_data_get_size(handle);
		return data_size;
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[dst_node]);
	_STARPU_TRACE_END_ALLOC_REUSE(dst_node);
#endif

	do
	{
		STARPU_ASSERT(handle->ops);
		STARPU_ASSERT(handle->ops->allocate_data_on_node);

		_STARPU_TRACE_START_ALLOC(dst_node);
		STARPU_ASSERT(replicate->data_interface);

#if defined(STARPU_USE_CUDA) && defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
		if (starpu_node_get_kind(dst_node) == STARPU_CUDA_RAM)
		{
			/* To facilitate the design of interface, we set the
			 * proper CUDA device in case it is needed. This avoids
			 * having to set it again in the malloc method of each
			 * interface. */
			starpu_cuda_set_device(_starpu_memory_node_get_devid(dst_node));
		}
#endif

		allocated_memory = handle->ops->allocate_data_on_node(replicate->data_interface, dst_node);
		_STARPU_TRACE_END_ALLOC(dst_node);

		if (allocated_memory == -ENOMEM)
		{
			size_t reclaim = 0.25*_starpu_memory_manager_get_global_memory_size(dst_node);
			size_t handle_size = handle->ops->get_size(handle);
			if (starpu_memstrategy_data_size_coefficient*handle_size > reclaim)
				reclaim = starpu_memstrategy_data_size_coefficient*handle_size;

			/* Take temporary reference on the replicate */
			replicate->refcnt++;
			handle->busy_count++;
			_starpu_spin_unlock(&handle->header_lock);

			_STARPU_TRACE_START_MEMRECLAIM(dst_node,is_prefetch);
			if (is_prefetch)
			{
				flush_memchunk_cache(dst_node, reclaim);
			}
			else
				_starpu_memory_reclaim_generic(dst_node, 0, reclaim);
			_STARPU_TRACE_END_MEMRECLAIM(dst_node,is_prefetch);

			int cpt = 0;
			while (cpt < STARPU_SPIN_MAXTRY && _starpu_spin_trylock(&handle->header_lock))
			{
				cpt++;
				_starpu_datawizard_progress(_starpu_memory_node_get_local_key(), 0);
			}
			if (cpt == STARPU_SPIN_MAXTRY)
				_starpu_spin_lock(&handle->header_lock);

			replicate->refcnt--;
			STARPU_ASSERT(replicate->refcnt >= 0);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
			ret = _starpu_data_check_not_busy(handle);
			STARPU_ASSERT(ret == 0);
		}

	}
	while((allocated_memory == -ENOMEM) && attempts++ < 2);

	return allocated_memory;
}

int _starpu_allocate_memory_on_node(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned is_prefetch)
{
	starpu_ssize_t allocated_memory;

	unsigned dst_node = replicate->memory_node;

	STARPU_ASSERT(handle);

	/* A buffer is already allocated on the node */
	if (replicate->allocated)
		return 0;

	STARPU_ASSERT(replicate->data_interface);
	allocated_memory = _starpu_allocate_interface(handle, replicate, dst_node, is_prefetch);

	/* perhaps we could really not handle that capacity misses */
	if (allocated_memory == -ENOMEM)
		return -ENOMEM;

	register_mem_chunk(replicate, 1);

	replicate->allocated = 1;
	replicate->automatically_allocated = 1;

	if (dst_node == 0)
	{
		void *ptr = starpu_data_handle_to_pointer(handle, 0);
		if (ptr != NULL)
		{
			_starpu_data_register_ram_pointer(handle, ptr);
		}
	}

	return 0;
}

unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle_t handle, unsigned memory_node)
{
	return handle->per_node[memory_node].allocated;
}

/* Record that this memchunk has been recently used */
void _starpu_memchunk_recently_used(struct _starpu_mem_chunk *mc, unsigned node)
{
	_starpu_spin_lock(&lru_rwlock[node]);
	struct _starpu_mem_chunk_lru *mc_lru=_starpu_mem_chunk_lru_new();
	mc_lru->mc=mc;
	_starpu_mem_chunk_lru_list_push_front(starpu_lru_list[node],mc_lru);
	_starpu_spin_unlock(&lru_rwlock[node]);
}

/* Push the given memchunk, recently used, at the end of the chunks to be evicted */
/* The mc_rwlock[node] rw-lock should be taken prior to calling this function.*/
static void _starpu_memchunk_recently_used_move(struct _starpu_mem_chunk *mc, unsigned node)
{
	/* Note: Sometimes the memchunk is not in the list... */
	struct _starpu_mem_chunk *mc_iter;
	for (mc_iter = _starpu_mem_chunk_list_begin(mc_list[node]);
	     mc_iter != _starpu_mem_chunk_list_end(mc_list[node]);
	     mc_iter = _starpu_mem_chunk_list_next(mc_iter) )
	{
		if (mc_iter==mc)
		{
			_starpu_mem_chunk_list_erase(mc_list[node], mc);
			_starpu_mem_chunk_list_push_back(mc_list[node], mc);
			return;
		}

	}
}

/* Put the recently used memchunks at the end of the mc_list, in the same order
 * as the LRU list, so that the most recently used memchunk eventually comes
 * last in the mc_list */
static void starpu_lru(unsigned node)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	_starpu_spin_lock(&lru_rwlock[node]);
	while (!_starpu_mem_chunk_lru_list_empty(starpu_lru_list[node]))
	{
		struct _starpu_mem_chunk_lru *mc_lru=_starpu_mem_chunk_lru_list_front(starpu_lru_list[node]);
		_starpu_memchunk_recently_used_move(mc_lru->mc, node);
		_starpu_mem_chunk_lru_list_erase(starpu_lru_list[node], mc_lru);
		_starpu_mem_chunk_lru_delete(mc_lru);
	}
	_starpu_spin_unlock(&lru_rwlock[node]);

	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
}

#ifdef STARPU_MEMORY_STATS
void _starpu_memory_display_stats_by_node(int node)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	if (!_starpu_mem_chunk_list_empty(mc_list[node]))
	{
		struct _starpu_mem_chunk *mc;

		fprintf(stderr, "#-------\n");
		fprintf(stderr, "Data on Node #%d\n",node);

		for (mc = _starpu_mem_chunk_list_begin(mc_list[node]);
		     mc != _starpu_mem_chunk_list_end(mc_list[node]);
		     mc = _starpu_mem_chunk_list_next(mc))
		{
			if (mc->automatically_allocated == 0)
				_starpu_memory_display_handle_stats(mc->data);
		}

	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
}
#endif

void starpu_data_display_memory_stats(void)
{
#ifdef STARPU_MEMORY_STATS
	unsigned node;

	fprintf(stderr, "\n#---------------------\n");
	fprintf(stderr, "Memory stats :\n");
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
	     _starpu_memory_display_stats_by_node(node);
	}
	fprintf(stderr, "\n#---------------------\n");
#endif
}


static int
get_better_disk_can_accept_size(starpu_data_handle_t handle, unsigned node)
{
	int target = -1;
	unsigned nnodes = starpu_memory_nodes_get_count();
	unsigned int i;
	double time_disk = 0;
				
	for (i = 0; i < nnodes; i++)
	{
		if (starpu_node_get_kind(i) == STARPU_DISK_RAM && i != node &&
		    (_starpu_memory_manager_test_allocate_size_(_starpu_data_get_size(handle), i) == 1 ||
		     handle->per_node[i].allocated))
		{
			/* if we can write on the disk */
			if (_starpu_get_disk_flag(i) != STARPU_DISK_NO_RECLAIM)
			{
				/* only time can change between disk <-> main_ram 
				 * and not between main_ram <-> worker if we compare diks*/
				double time_tmp = starpu_transfer_predict(i, STARPU_MAIN_RAM, _starpu_data_get_size(handle));
				if (target == -1 || time_disk > time_tmp)
				{
					target = i;
					time_disk = time_tmp;
				}
			}	
		}
	}
	return target;
}


static unsigned
choose_target(starpu_data_handle_t handle, unsigned node)
{
	unsigned target = -1;
	size_t size_handle = _starpu_data_get_size(handle);
	if (handle->home_node != -1)
		/* try to push on RAM if we can before to push on disk */
		if(starpu_node_get_kind(handle->home_node) == STARPU_DISK_RAM && node != STARPU_MAIN_RAM)
		{
			if (handle->per_node[STARPU_MAIN_RAM].allocated || 
			    _starpu_memory_manager_test_allocate_size_(size_handle, STARPU_MAIN_RAM) == 1)
			{
				target = STARPU_MAIN_RAM;
			}
			else
			{
				target = get_better_disk_can_accept_size(handle, node);
			}

		}
          	/* others memory nodes */
		else 
		{
			target = handle->home_node;
		}
	else
	{
		/* handle->home_node == -1 */
		/* no place for datas in RAM, we push on disk */
		if (node == STARPU_MAIN_RAM)
		{
			target = get_better_disk_can_accept_size(handle, node);
		}
		/* node != 0 */
		/* try to push data to RAM if we can before to push on disk*/
		else if (handle->per_node[STARPU_MAIN_RAM].allocated || 
			 _starpu_memory_manager_test_allocate_size_(size_handle, STARPU_MAIN_RAM) == 1)
		{
			target = STARPU_MAIN_RAM;
		}
		/* no place in RAM */
		else
		{
			target = get_better_disk_can_accept_size(handle, node);
		}
	}
	/* we haven't the right to write on the disk */
	if (target != -1 && starpu_node_get_kind(target) == STARPU_DISK_RAM && _starpu_get_disk_flag(target) == STARPU_DISK_NO_RECLAIM)
		target = -1;

	return target;
}
