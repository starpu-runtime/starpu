/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

#include <datawizard/memalloc.h>
#include <datawizard/footprint.h>

/* This per-node RW-locks protect mc_list and memchunk_cache entries */
static pthread_rwlock_t mc_rwlock[STARPU_MAXNODES]; 

/* Potentially in use memory chunks */
static starpu_mem_chunk_list_t mc_list[STARPU_MAXNODES];

/* Explicitly caches memory chunks that can be reused */
static starpu_mem_chunk_list_t memchunk_cache[STARPU_MAXNODES];

void _starpu_init_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		PTHREAD_RWLOCK_INIT(&mc_rwlock[i], NULL);
		mc_list[i] = starpu_mem_chunk_list_new();
		memchunk_cache[i] = starpu_mem_chunk_list_new();
	}
}

void _starpu_deinit_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		starpu_mem_chunk_list_delete(mc_list[i]);
		starpu_mem_chunk_list_delete(memchunk_cache[i]);
	}
}

/*
 *	Manipulate subtrees
 */

static void lock_all_subtree(starpu_data_handle handle)
{
	if (handle->nchildren == 0)
	{
		/* this is a leaf */
		while (_starpu_spin_trylock(&handle->header_lock))
			_starpu_datawizard_progress(_starpu_get_local_memory_node(), 0);
	}
	else {
		/* lock all sub-subtrees children */
		unsigned child;
		for (child = 0; child < handle->nchildren; child++)
		{
			lock_all_subtree(&handle->children[child]);
		}
	}
}

static void unlock_all_subtree(starpu_data_handle handle)
{
	if (handle->nchildren == 0)
	{
		/* this is a leaf */	
		_starpu_spin_unlock(&handle->header_lock);
	}
	else {
		/* lock all sub-subtrees children 
		 * Note that this is done in the reverse order of the
		 * lock_all_subtree so that we avoid deadlock */
		unsigned i;
		for (i =0; i < handle->nchildren; i++)
		{
			unsigned child = handle->nchildren - 1 - i;
			unlock_all_subtree(&handle->children[child]);
		}
	}
}

static unsigned may_free_subtree(starpu_data_handle handle, unsigned node)
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
		res = may_free_subtree(&handle->children[child], node);
		if (!res) return 0;
	}

	/* no problem was found */
	return 1;
}

static void transfer_subtree_to_node(starpu_data_handle handle, unsigned src_node, 
						unsigned dst_node)
{
	unsigned i;
	unsigned last = 0;
	unsigned cnt;
	int ret;

	if (handle->nchildren == 0)
	{
		struct starpu_data_replicate_s *src_replicate = &handle->per_node[src_node];
		struct starpu_data_replicate_s *dst_replicate = &handle->per_node[dst_node];

		/* this is a leaf */
		switch(src_replicate->state) {
		case STARPU_OWNER:
			/* the local node has the only copy */
			/* the owner is now the destination_node */
			src_replicate->state = STARPU_INVALID;
			dst_replicate->state = STARPU_OWNER;

#warning we should use requests during memory reclaim
			/* TODO use request !! */
			src_replicate->refcnt++;
			dst_replicate->refcnt++;

			ret = _starpu_driver_copy_data_1_to_1(handle, src_replicate, dst_replicate, 0, NULL, 1);
			STARPU_ASSERT(ret == 0);

			src_replicate->refcnt--;
			dst_replicate->refcnt--;

			break;
		case STARPU_SHARED:
			/* some other node may have the copy */
			src_replicate->state = STARPU_INVALID;

			/* count the number of copies */
			cnt = 0;
			for (i = 0; i < STARPU_MAXNODES; i++)
			{
				if (handle->per_node[i].state == STARPU_SHARED) {
					cnt++; 
					last = i;
				}
			}

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
	else {
		/* lock all sub-subtrees children */
		unsigned child;
		for (child = 0; child < handle->nchildren; child++)
		{
			transfer_subtree_to_node(&handle->children[child],
							src_node, dst_node);
		}
	}
}

static size_t free_memory_on_node(starpu_mem_chunk_t mc, uint32_t node)
{
	size_t freed = 0;

	STARPU_ASSERT(mc->ops);
	STARPU_ASSERT(mc->ops->free_data_on_node);

	starpu_data_handle handle = mc->data;

	/* Does this memory chunk refers to a handle that does not exist
	 * anymore ? */
	unsigned data_was_deleted = mc->data_was_deleted;

	struct starpu_data_replicate_s *replicate = mc->replicate;

//	while (_starpu_spin_trylock(&handle->header_lock))
//		_starpu_datawizard_progress(_starpu_get_local_memory_node());

#warning can we block here ?
//	_starpu_spin_lock(&handle->header_lock);

	if (mc->automatically_allocated && 
		(!handle || data_was_deleted || replicate->refcnt == 0))
	{
		if (handle && !data_was_deleted)
			STARPU_ASSERT(replicate->allocated);

		mc->ops->free_data_on_node(mc->interface, node);

		if (handle && !data_was_deleted)
		{
			replicate->allocated = 0;

			/* XXX why do we need that ? */
			replicate->automatically_allocated = 0;
		}

		freed = mc->size;

		if (handle && !data_was_deleted)
			STARPU_ASSERT(replicate->refcnt == 0);
	}

//	_starpu_spin_unlock(&handle->header_lock);

	return freed;
}



static size_t do_free_mem_chunk(starpu_mem_chunk_t mc, unsigned node)
{
	size_t size;

	/* free the actual buffer */
	size = free_memory_on_node(mc, node);

	/* remove the mem_chunk from the list */
	starpu_mem_chunk_list_erase(mc_list[node], mc);

	free(mc->interface);
	starpu_mem_chunk_delete(mc);

	return size; 
}

/* This function is called for memory chunks that are possibly in used (ie. not
 * in the cache). They should therefore still be associated to a handle. */
static size_t try_to_free_mem_chunk(starpu_mem_chunk_t mc, unsigned node)
{
	size_t freed = 0;

	starpu_data_handle handle;
	handle = mc->data;
	STARPU_ASSERT(handle);

	/* Either it's a "relaxed coherency" memchunk, or it's a memchunk that
	 * could be used with filters. */
	if (mc->relaxed_coherency)
	{
		STARPU_ASSERT(mc->replicate);

		while (_starpu_spin_trylock(&handle->header_lock))
			_starpu_datawizard_progress(_starpu_get_local_memory_node(), 0);

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
	else {
		/* try to lock all the leafs of the subtree */
		lock_all_subtree(handle);
	
		/* check if they are all "free" */
		if (may_free_subtree(handle, node))
		{
			STARPU_ASSERT(handle->per_node[node].refcnt == 0);
	
			/* in case there was nobody using that buffer, throw it 
			 * away after writing it back to main memory */
			transfer_subtree_to_node(handle, node, 0);
	
			STARPU_ASSERT(handle->per_node[node].refcnt == 0);
	
			/* now the actual buffer may be freed */
			freed = do_free_mem_chunk(mc, node);
		}
	
		/* unlock the leafs */
		unlock_all_subtree(handle);
	}
	return freed;
}

#ifdef STARPU_USE_ALLOCATION_CACHE
/* We assume that mc_rwlock[node] is taken. is_already_in_mc_list indicates
 * that the mc is already in the list of buffers that are possibly used, and
 * therefore not in the cache. */
static void reuse_mem_chunk(unsigned node, struct starpu_data_replicate_s *new_replicate, starpu_mem_chunk_t mc, unsigned is_already_in_mc_list)
{
	starpu_data_handle old_data;
	old_data = mc->data;

	/* we found an appropriate mem chunk: so we get it out
	 * of the "to free" list, and reassign it to the new
	 * piece of data */

	if (!is_already_in_mc_list)
	{
		starpu_mem_chunk_list_erase(memchunk_cache[node], mc);
	}

	struct starpu_data_replicate_s *old_replicate = mc->replicate;
	old_replicate->allocated = 0;
	old_replicate->automatically_allocated = 0;
	old_replicate->initialized = 0;

	new_replicate->allocated = 1;
	new_replicate->automatically_allocated = 1;
	new_replicate->initialized = 0;

	STARPU_ASSERT(new_replicate->interface);
	STARPU_ASSERT(mc->interface);
	memcpy(new_replicate->interface, mc->interface, old_replicate->ops->interface_size);

	mc->data = new_replicate->handle;
	mc->data_was_deleted = 0;
	/* mc->ops, mc->size, mc->footprint and mc->interface should be
 	 * unchanged ! */
	
	/* reinsert the mem chunk in the list of active memory chunks */
	if (!is_already_in_mc_list)
	{
		starpu_mem_chunk_list_push_front(mc_list[node], mc);
	}
}

static unsigned try_to_reuse_mem_chunk(starpu_mem_chunk_t mc, unsigned node, starpu_data_handle new_data, unsigned is_already_in_mc_list)
{
	unsigned success = 0;

	starpu_data_handle old_data;

	old_data = mc->data;

	STARPU_ASSERT(old_data);

	/* try to lock all the leafs of the subtree */
	lock_all_subtree(old_data);

	/* check if they are all "free" */
	if (may_free_subtree(old_data, node))
	{
		success = 1;

		/* in case there was nobody using that buffer, throw it 
		 * away after writing it back to main memory */
		transfer_subtree_to_node(old_data, node, 0);

		/* now replace the previous data */
		reuse_mem_chunk(node, new_data, mc, is_already_in_mc_list);
	}

	/* unlock the leafs */
	unlock_all_subtree(old_data);

	return success;
}

static int _starpu_data_interface_compare(void *interface_a, struct starpu_data_interface_ops_t *ops_a,
						void *interface_b, struct starpu_data_interface_ops_t *ops_b)
{
	if (ops_a->interfaceid != ops_b->interfaceid)
		return -1;

	int ret = ops_a->compare(interface_a, interface_b);

	return ret;
}

/* This function must be called with mc_rwlock[node] taken in write mode */
static starpu_mem_chunk_t _starpu_memchunk_cache_lookup_locked(uint32_t node, starpu_data_handle handle)
{
	uint32_t footprint = _starpu_compute_data_footprint(handle);

	/* go through all buffers in the cache */
	starpu_mem_chunk_t mc;
	for (mc = starpu_mem_chunk_list_begin(memchunk_cache[node]);
	     mc != starpu_mem_chunk_list_end(memchunk_cache[node]);
	     mc = starpu_mem_chunk_list_next(mc))
	{
		if (mc->footprint == footprint)
		{
			/* Is that a false hit ? (this is _very_ unlikely) */
			if (_starpu_data_interface_compare(handle->per_node[node].interface, handle->ops, mc->interface, mc->ops))
				continue;

			/* Cache hit */

			/* Remove from the cache */
			starpu_mem_chunk_list_erase(memchunk_cache[node], mc);
			return mc;
		}
	}

	/* This is a cache miss */
	return NULL;
}

/* this function looks for a memory chunk that matches a given footprint in the
 * list of mem chunk that need to be freed. This function must be called with
 * mc_rwlock[node] taken in write mode. */
static unsigned try_to_find_reusable_mem_chunk(unsigned node, starpu_data_handle data, uint32_t footprint)
{
	starpu_mem_chunk_t mc, next_mc;

	/* go through all buffers in the cache */
	mc = _starpu_memchunk_cache_lookup_locked(node, handle);
	if (mc)
	{
		/* We found an entry in the cache so we can reuse it */
		reuse_mem_chunk(node, data, mc, 0);
		return 1;
	}

	/* now look for some non essential data in the active list */
	for (mc = starpu_mem_chunk_list_begin(mc_list[node]);
	     mc != starpu_mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* there is a risk that the memory chunk is freed before next
		 * iteration starts: so we compute the next element of the list
		 * now */
		next_mc = starpu_mem_chunk_list_next(mc);

		if (mc->data->is_not_important && (mc->footprint == footprint))
		{
//			fprintf(stderr, "found a candidate ...\n");
			if (try_to_reuse_mem_chunk(mc, node, data, 1))
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
static size_t flush_memchunk_cache(uint32_t node)
{
	starpu_mem_chunk_t mc, next_mc;
	
	size_t freed = 0;

	for (mc = starpu_mem_chunk_list_begin(memchunk_cache[node]);
	     mc != starpu_mem_chunk_list_end(memchunk_cache[node]);
	     mc = next_mc)
	{
		next_mc = starpu_mem_chunk_list_next(mc);

		freed += free_memory_on_node(mc, node);

		starpu_mem_chunk_list_erase(memchunk_cache[node], mc);

		free(mc->interface);
		starpu_mem_chunk_delete(mc);
	}

	return freed;
}

/*
 * Try to free the buffers currently in use on the memory node. If the force
 * flag is set, the memory is freed regardless of coherency concerns (this
 * should only be used at the termination of StarPU for instance). The
 * mc_rwlock[node] rw-lock should be taken prior to calling this function.
 */
static size_t free_potentially_in_use_mc(uint32_t node, unsigned force)
{
	size_t freed = 0;

	starpu_mem_chunk_t mc, next_mc;

	for (mc = starpu_mem_chunk_list_begin(mc_list[node]);
	     mc != starpu_mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* there is a risk that the memory chunk is freed 
		   before next iteration starts: so we compute the next
		   element of the list now */
		next_mc = starpu_mem_chunk_list_next(mc);

		if (!force)
		{
			freed += try_to_free_mem_chunk(mc, node);
			#if 0
			if (freed > toreclaim)
				break;
			#endif
		}
		else {
			/* We must free the memory now: note that data
			 * coherency is not maintained in that case ! */
			freed += do_free_mem_chunk(mc, node);
		}
	}

	return freed;
}

static size_t reclaim_memory_generic(uint32_t node, unsigned force)
{
	size_t freed = 0;

	PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	/* remove all buffers for which there was a removal request */
	freed += flush_memchunk_cache(node);

	/* try to free all allocated data potentially in use */
	freed += free_potentially_in_use_mc(node, force);

	PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);

	return freed;

}

/*
 * This function frees all the memory that was implicitely allocated by StarPU
 * (for the data replicates). This is not ensuring data coherency, and should
 * only be called while StarPU is getting shut down.
 */
size_t _starpu_free_all_automatically_allocated_buffers(uint32_t node)
{
	return reclaim_memory_generic(node, 1);
}

static starpu_mem_chunk_t _starpu_memchunk_init(struct starpu_data_replicate_s *replicate, size_t size, size_t interface_size, unsigned automatically_allocated)
{
	starpu_mem_chunk_t mc = starpu_mem_chunk_new();
	starpu_data_handle handle = replicate->handle;

	STARPU_ASSERT(handle);
	STARPU_ASSERT(handle->ops);

	mc->data = handle;
	mc->size = size;
	mc->footprint = _starpu_compute_data_footprint(handle);
	mc->ops = handle->ops;
	mc->data_was_deleted = 0;
	mc->automatically_allocated = automatically_allocated;
	mc->relaxed_coherency = replicate->relaxed_coherency;		
	mc->replicate = replicate;

	/* Save a copy of the interface */
	mc->interface = malloc(interface_size);
	STARPU_ASSERT(mc->interface);
	memcpy(mc->interface, replicate->interface, interface_size);

	return mc;
}

static void register_mem_chunk(struct starpu_data_replicate_s *replicate, size_t size, unsigned automatically_allocated)
{
	unsigned dst_node = replicate->memory_node;

	starpu_mem_chunk_t mc;

	/* the interface was already filled by ops->allocate_data_on_node */
	size_t interface_size = replicate->handle->ops->interface_size;

	/* Put this memchunk in the list of memchunk in use */
	mc = _starpu_memchunk_init(replicate, size, interface_size, automatically_allocated); 

	PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[dst_node]);

	starpu_mem_chunk_list_push_front(mc_list[dst_node], mc);

	PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[dst_node]);
}

/* This function is called when the handle is destroyed (eg. when calling
 * unregister or unpartition). It puts all the memchunks that refer to the
 * specified handle into the cache. */
void _starpu_request_mem_chunk_removal(starpu_data_handle handle, unsigned node)
{
	PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	/* iterate over the list of memory chunks and remove the entry */
	starpu_mem_chunk_t mc, next_mc;
	for (mc = starpu_mem_chunk_list_begin(mc_list[node]);
	     mc != starpu_mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		next_mc = starpu_mem_chunk_list_next(mc);

		if (mc->data == handle) {
			/* we found the data */
			mc->data_was_deleted = 1;

			/* remove it from the main list */
			starpu_mem_chunk_list_erase(mc_list[node], mc);

			/* put it in the list of buffers to be removed */
			starpu_mem_chunk_list_push_front(memchunk_cache[node], mc);

			/* Note that we do not stop here because there can be
			 * multiple replicates associated to the same handle on
			 * the same memory node.  */
		}
	}

	/* there was no corresponding buffer ... */
	PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
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

static ssize_t _starpu_allocate_interface(starpu_data_handle handle, struct starpu_data_replicate_s *replicate, uint32_t dst_node)
{
	unsigned attempts = 0;
	ssize_t allocated_memory;

	_starpu_data_allocation_inc_stats(dst_node);

#ifdef STARPU_USE_ALLOCATION_CACHE
	/* perhaps we can directly reuse a buffer in the free-list */
	uint32_t footprint = _starpu_compute_data_footprint(handle);

	STARPU_TRACE_START_ALLOC_REUSE(dst_node);
	PTHREAD_RWLOCK_WRLOCK(&mc_rwlock[node]);

	if (try_to_find_reusable_mem_chunk(dst_node, handle, footprint))
	{
		PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
		_starpu_allocation_cache_hit(dst_node);
		ssize_t data_size = _starpu_data_get_size(handle);
		return data_size;
	}

	PTHREAD_RWLOCK_UNLOCK(&mc_rwlock[node]);
	STARPU_TRACE_END_ALLOC_REUSE(dst_node);
#endif

	do {
		STARPU_ASSERT(handle->ops);
		STARPU_ASSERT(handle->ops->allocate_data_on_node);

		STARPU_TRACE_START_ALLOC(dst_node);
		STARPU_ASSERT(replicate->interface);
		allocated_memory = handle->ops->allocate_data_on_node(replicate->interface, dst_node);
		STARPU_TRACE_END_ALLOC(dst_node);

		if (allocated_memory == -ENOMEM)
		{
			replicate->refcnt++;
			_starpu_spin_unlock(&handle->header_lock);

			STARPU_TRACE_START_MEMRECLAIM(dst_node);
			reclaim_memory_generic(dst_node, 0);
			STARPU_TRACE_END_MEMRECLAIM(dst_node);

		        while (_starpu_spin_trylock(&handle->header_lock))
		                _starpu_datawizard_progress(_starpu_get_local_memory_node(), 0);
		
			replicate->refcnt--;
		}
		
	} while((allocated_memory == -ENOMEM) && attempts++ < 2);

	return allocated_memory;
}

int _starpu_allocate_memory_on_node(starpu_data_handle handle, struct starpu_data_replicate_s *replicate)
{
	ssize_t allocated_memory;

	unsigned dst_node = replicate->memory_node;

	STARPU_ASSERT(handle);

	/* A buffer is already allocated on the node */
	if (replicate->allocated)
		return 0;

	STARPU_ASSERT(replicate->interface);
	allocated_memory = _starpu_allocate_interface(handle, replicate, dst_node);

	/* perhaps we could really not handle that capacity misses */
	if (allocated_memory == -ENOMEM)
		return -ENOMEM;

	register_mem_chunk(replicate, allocated_memory, 1);

	replicate->allocated = 1;
	replicate->automatically_allocated = 1;

	return 0;
}

unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle handle, uint32_t memory_node)
{
	return handle->per_node[memory_node].allocated;
}
