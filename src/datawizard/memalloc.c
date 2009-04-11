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

#include "memalloc.h"
#include <datawizard/footprint.h>

extern mem_node_descr descr;
static starpu_mutex mc_mutex[MAXNODES]; 
static mem_chunk_list_t mc_list[MAXNODES];
static mem_chunk_list_t mc_list_to_free[MAXNODES];

static size_t liberate_memory_on_node(mem_chunk_t mc, uint32_t node);

void init_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < MAXNODES; i++)
	{
		init_mutex(&mc_mutex[i]);
		mc_list[i] = mem_chunk_list_new();
		mc_list_to_free[i] = mem_chunk_list_new();
	}
}

void deinit_mem_chunk_lists(void)
{
	unsigned i;
	for (i = 0; i < MAXNODES; i++)
	{
		mem_chunk_list_delete(mc_list[i]);
		mem_chunk_list_delete(mc_list_to_free[i]);
	}
}

static void lock_all_subtree(data_state *data)
{
	if (data->nchildren == 0)
	{
		/* this is a leaf */	
		while (take_mutex_try(&data->header_lock))
			datawizard_progress(get_local_memory_node());
	}
	else {
		/* lock all sub-subtrees children */
		int child;
		for (child = 0; child < data->nchildren; child++)
		{
			lock_all_subtree(&data->children[child]);
		}
	}
}

static void unlock_all_subtree(data_state *data)
{
	if (data->nchildren == 0)
	{
		/* this is a leaf */	
		release_mutex(&data->header_lock);
	}
	else {
		/* lock all sub-subtrees children */
		int child;
		for (child = data->nchildren - 1; child >= 0; child--)
		{
			unlock_all_subtree(&data->children[child]);
		}
	}
}

static unsigned may_free_subtree(data_state *data, unsigned node)
{
	if (data->nchildren == 0)
	{
		/* we only free if no one refers to the leaf */
		uint32_t refcnt = get_data_refcnt(data, node);
		return (refcnt == 0);
	}
	else {
		/* lock all sub-subtrees children */
		int child;
		for (child = 0; child < data->nchildren; child++)
		{
			unsigned res;
			res = may_free_subtree(&data->children[child], node);
			if (!res) return 0;
		}

		/* no problem was found */
		return 1;
	}
}

static size_t do_free_mem_chunk(mem_chunk_t mc, unsigned node)
{
	size_t size;

	/* free the actual buffer */
	size = liberate_memory_on_node(mc, node);

	/* remove the mem_chunk from the list */
	mem_chunk_list_erase(mc_list[node], mc);
	mem_chunk_delete(mc);

	return size; 
}

static void transfer_subtree_to_node(data_state *data, unsigned src_node, 
						unsigned dst_node)
{
	unsigned i;
	unsigned last = 0;
	unsigned cnt;
	int ret;

	if (data->nchildren == 0)
	{
		/* this is a leaf */
		switch(data->per_node[src_node].state) {
		case OWNER:
			/* the local node has the only copy */
			/* the owner is now the destination_node */
			data->per_node[src_node].state = INVALID;
			data->per_node[dst_node].state = OWNER;

			ret = driver_copy_data_1_to_1(data, src_node, dst_node, 0);
			STARPU_ASSERT(ret == 0);

			break;
		case SHARED:
			/* some other node may have the copy */
			data->per_node[src_node].state = INVALID;

			/* count the number of copies */
			cnt = 0;
			for (i = 0; i < MAXNODES; i++)
			{
				if (data->per_node[i].state == SHARED) {
					cnt++; 
					last = i;
				}
			}

			if (cnt == 1)
				data->per_node[last].state = OWNER;

			break;
		case INVALID:
			/* nothing to be done */
			break;
		default:
			STARPU_ASSERT(0);
			break;
		}
	}
	else {
		/* lock all sub-subtrees children */
		int child;
		for (child = 0; child < data->nchildren; child++)
		{
			transfer_subtree_to_node(&data->children[child],
							src_node, dst_node);
		}
	}
}


static size_t try_to_free_mem_chunk(mem_chunk_t mc, unsigned node, unsigned attempts)
{
	size_t liberated = 0;

	data_state *data;

	data = mc->data;

	STARPU_ASSERT(data);

	if (attempts == 0)
	{
		/* this is the first attempt to free memory
		   so we avoid to drop requested memory */
		/* TODO */
	}

	/* try to lock all the leafs of the subtree */
	lock_all_subtree(data);

	/* check if they are all "free" */
	if (may_free_subtree(data, node))
	{
		/* in case there was nobody using that buffer, throw it 
		 * away after writing it back to main memory */
		transfer_subtree_to_node(data, node, 0);

		/* now the actual buffer may be liberated */
		liberated = do_free_mem_chunk(mc, node);
	}

	/* unlock the leafs */
	unlock_all_subtree(data);

	return liberated;
}

#ifdef USE_ALLOCATION_CACHE
/* we assume that mc_mutex[node] is taken */
static void reuse_mem_chunk(unsigned node, data_state *new_data, mem_chunk_t mc, unsigned is_already_in_mc_list)
{
	data_state *old_data;
	old_data = mc->data;

	/* we found an appropriate mem chunk: so we get it out
	 * of the "to free" list, and reassign it to the new
	 * piece of data */

	if (!is_already_in_mc_list)
	{
		mem_chunk_list_erase(mc_list_to_free[node], mc);
	}

	if (!mc->data_was_deleted)
	{
		old_data->per_node[node].allocated = 0;
		old_data->per_node[node].automatically_allocated = 0;
	}

	new_data->per_node[node].allocated = 1;
	new_data->per_node[node].automatically_allocated = 1;

	memcpy(&new_data->interface[node], &mc->interface, sizeof(starpu_data_interface_t));

	mc->data = new_data;
	mc->data_was_deleted = 0;
	/* mc->ops, mc->size, mc->footprint and mc->interface should be
 	 * unchanged ! */
	
	/* reinsert the mem chunk in the list of active memory chunks */
	if (!is_already_in_mc_list)
	{
		mem_chunk_list_push_front(mc_list[node], mc);
	}
}



static unsigned try_to_reuse_mem_chunk(mem_chunk_t mc, unsigned node, data_state *new_data, unsigned is_already_in_mc_list)
{
	unsigned success = 0;

	data_state *old_data;

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

/* this function looks for a memory chunk that matches a given footprint in the
 * list of mem chunk that need to be liberated */
static unsigned try_to_find_reusable_mem_chunk(unsigned node, data_state *data, uint32_t footprint)
{
	take_mutex(&mc_mutex[node]);

	/* go through all buffers for which there was a removal request */
	mem_chunk_t mc, next_mc;
	for (mc = mem_chunk_list_begin(mc_list_to_free[node]);
	     mc != mem_chunk_list_end(mc_list_to_free[node]);
	     mc = next_mc)
	{
		next_mc = mem_chunk_list_next(mc);

		if (mc->footprint == footprint)
		{

			data_state *old_data;
			old_data = mc->data;

			if (old_data->per_node[node].allocated &&
					old_data->per_node[node].automatically_allocated)
			{
				reuse_mem_chunk(node, data, mc, 0);

				release_mutex(&mc_mutex[node]);
				return 1;
			}
		}

	}

	/* now look for some non essential data in the active list */
	for (mc = mem_chunk_list_begin(mc_list[node]);
	     mc != mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* there is a risk that the memory chunk is liberated 
		   before next iteration starts: so we compute the next
		   element of the list now */
		next_mc = mem_chunk_list_next(mc);

		if (mc->data->is_not_important && (mc->footprint == footprint))
		{
//			fprintf(stderr, "found a candidate ...\n");
			if (try_to_reuse_mem_chunk(mc, node, data, 1))
			{
				release_mutex(&mc_mutex[node]);
				return 1;
			}
		}
	}

	release_mutex(&mc_mutex[node]);

	return 0;
}
#endif

/* 
 * Try to free some memory on the specified node
 * 	returns 0 if no memory was released, 1 else
 */
static size_t reclaim_memory(uint32_t node, size_t toreclaim __attribute__ ((unused)), unsigned attempts)
{
//	fprintf(stderr, "reclaim memory...\n");

	size_t liberated = 0;

	take_mutex(&mc_mutex[node]);

	/* remove all buffers for which there was a removal request */
	mem_chunk_t mc, next_mc;
	for (mc = mem_chunk_list_begin(mc_list_to_free[node]);
	     mc != mem_chunk_list_end(mc_list_to_free[node]);
	     mc = next_mc)
	{
		next_mc = mem_chunk_list_next(mc);

		liberated += liberate_memory_on_node(mc, node);

		mem_chunk_list_erase(mc_list_to_free[node], mc);

		mem_chunk_delete(mc);
	}

	/* try to free all allocated data potentially in use .. XXX */
	for (mc = mem_chunk_list_begin(mc_list[node]);
	     mc != mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		/* there is a risk that the memory chunk is liberated 
		   before next iteration starts: so we compute the next
		   element of the list now */
		next_mc = mem_chunk_list_next(mc);

		liberated += try_to_free_mem_chunk(mc, node, attempts);
		#if 0
		if (liberated > toreclaim)
			break;
		#endif
	}

//	fprintf(stderr, "got %d MB back\n", (int)liberated/(1024*1024));

	release_mutex(&mc_mutex[node]);

	return liberated;
}

static void register_mem_chunk(data_state *state, uint32_t dst_node, size_t size, unsigned automatically_allocated)
{
	mem_chunk_t mc = mem_chunk_new();

	STARPU_ASSERT(state);
	STARPU_ASSERT(state->ops);

	mc->data = state;
	mc->size = size; 
	mc->footprint = compute_data_footprint(state);
	mc->ops = state->ops;
	mc->data_was_deleted = 0;
	mc->automatically_allocated = automatically_allocated;

	/* the interface was already filled by ops->allocate_data_on_node */
	memcpy(&mc->interface, &state->interface[dst_node], sizeof(starpu_data_interface_t));

	take_mutex(&mc_mutex[dst_node]);
	mem_chunk_list_push_front(mc_list[dst_node], mc);
	release_mutex(&mc_mutex[dst_node]);
}

void request_mem_chunk_removal(data_state *state, unsigned node)
{
	take_mutex(&mc_mutex[node]);

	/* iterate over the list of memory chunks and remove the entry */
	mem_chunk_t mc, next_mc;
	for (mc = mem_chunk_list_begin(mc_list[node]);
	     mc != mem_chunk_list_end(mc_list[node]);
	     mc = next_mc)
	{
		next_mc = mem_chunk_list_next(mc);

		if (mc->data == state) {
			/* we found the data */
			mc->data_was_deleted = 1;

			/* remove it from the main list */
			mem_chunk_list_erase(mc_list[node], mc);

			/* put it in the list of buffers to be removed */
			mem_chunk_list_push_front(mc_list_to_free[node], mc);

			release_mutex(&mc_mutex[node]);

			return;
		}
	}

	/* there was no corresponding buffer ... */

	release_mutex(&mc_mutex[node]);
}

static size_t liberate_memory_on_node(mem_chunk_t mc, uint32_t node)
{
	size_t liberated = 0;

	STARPU_ASSERT(mc->ops);
	STARPU_ASSERT(mc->ops->liberate_data_on_node);

	if (mc->automatically_allocated)
	{
		mc->ops->liberate_data_on_node(&mc->interface, node);

		if (!mc->data_was_deleted)
		{
			data_state *state = mc->data;

			state->per_node[node].allocated = 0;

			/* XXX why do we need that ? */
			state->per_node[node].automatically_allocated = 0;
		}

		liberated = mc->size;
	}

	return liberated;
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
int allocate_memory_on_node(data_state *state, uint32_t dst_node)
{
	unsigned attempts = 0;
	size_t allocated_memory;

	STARPU_ASSERT(state);

	data_allocation_inc_stats(dst_node);

#ifdef USE_ALLOCATION_CACHE
	/* perhaps we can directly reuse a buffer in the free-list */
	uint32_t footprint = compute_data_footprint(state);

	TRACE_START_ALLOC_REUSE(dst_node);
	if (try_to_find_reusable_mem_chunk(dst_node, state, footprint))
	{
		allocation_cache_hit(dst_node);
		return 0;
	}
	TRACE_END_ALLOC_REUSE(dst_node);
#endif

	do {
		STARPU_ASSERT(state->ops);
		STARPU_ASSERT(state->ops->allocate_data_on_node);

		TRACE_START_ALLOC(dst_node);
		allocated_memory = state->ops->allocate_data_on_node(state, dst_node);
		TRACE_END_ALLOC(dst_node);

		if (!allocated_memory) {
			/* XXX perhaps we should find the proper granularity 
			 * not to waste our cache all the time */
			STARPU_ASSERT(state->ops->get_size);
			size_t data_size = state->ops->get_size(state);

			TRACE_START_MEMRECLAIM(dst_node);
			reclaim_memory(dst_node, 2*data_size, attempts);
			TRACE_END_MEMRECLAIM(dst_node);
		}
		
	} while(!allocated_memory && attempts++ < 2);

	/* perhaps we could really not handle that capacity misses */
	if (!allocated_memory)
		goto nomem;

	register_mem_chunk(state, dst_node, allocated_memory, 1);

	state->per_node[dst_node].allocated = 1;
	state->per_node[dst_node].automatically_allocated = 1;

	return 0;
nomem:
	STARPU_ASSERT(!allocated_memory);
	return -ENOMEM;
}
