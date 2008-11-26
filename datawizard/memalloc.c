#include "memalloc.h"

extern mem_node_descr descr;
static mutex mc_mutex[MAXNODES]; 
static mem_chunk_list_t mc_list[MAXNODES];
static mem_chunk_list_t mc_list_to_free[MAXNODES];

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

static void lock_all_subtree(data_state *data)
{
	if (data->nchildren == 0)
	{
		/* this is a leaf */	
		take_mutex(&data->header_lock);
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

unsigned may_free_subtree(data_state *data, unsigned node)
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

size_t do_free_mem_chunk(mem_chunk_t mc, unsigned node)
{
	size_t size;

	/* free the actual buffer */
	size = liberate_memory_on_node(mc, node);

	/* remove the mem_chunk from the list */
	mem_chunk_list_erase(mc_list[node], mc);
	mem_chunk_delete(mc);

	return size; 
}

void transfer_subtree_to_node(data_state *data, unsigned src_node, 
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
			ASSERT(ret == 0);

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
			ASSERT(0);
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

static size_t try_to_free_mem_chunk(mem_chunk_t mc, unsigned node)
{
	size_t liberated = 0;

	data_state *data;

	data = mc->data;

	ASSERT(data);

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

/* 
 * Try to free some memory on the specified node
 * 	returns 0 if no memory was released, 1 else
 */
static size_t reclaim_memory(uint32_t node, size_t toreclaim __attribute__ ((unused)))
{
//	fprintf(stderr, "reclaim memory...\n");

	size_t liberated = 0;

	take_mutex(&mc_mutex[node]);

	/* remove all buffers for which there was a removal request */
	mem_chunk_t mc;
	for (mc = mem_chunk_list_begin(mc_list_to_free[node]);
	     mc != mem_chunk_list_end(mc_list_to_free[node]);
	     mc = mem_chunk_list_next(mc))
	{
		liberated += liberate_memory_on_node(mc, node);

		/* XXX there is still that mem_chunk_t structure leaking */
		mem_chunk_list_erase(mc_list_to_free[node], mc);
		mem_chunk_delete(mc);
	}

	/* try to free all allocated data potentially in use .. XXX */
	for (mc = mem_chunk_list_begin(mc_list[node]);
	     mc != mem_chunk_list_end(mc_list[node]);
	     mc = mem_chunk_list_next(mc))
	{
		liberated += try_to_free_mem_chunk(mc, node);
		#if 0
		if (liberated > toreclaim)
			break;
		#endif
	}

//	fprintf(stderr, "got %d MB back\n", (int)liberated/(1024*1024));

	release_mutex(&mc_mutex[node]);

	return liberated;
}

static void register_mem_chunk(data_state *state, uint32_t dst_node, size_t size)
{
	mem_chunk_t mc = mem_chunk_new();

	ASSERT(state);
	ASSERT(state->ops);
	ASSERT(state->ops->liberate_data_on_node);

	mc->data = state;
	mc->size = size; 

	take_mutex(&mc_mutex[dst_node]);
	mem_chunk_list_push_front(mc_list[dst_node], mc);
	release_mutex(&mc_mutex[dst_node]);
}

void request_mem_chunk_removal(data_state *state, unsigned node)
{
	take_mutex(&mc_mutex[node]);

	/* iterate over the list of memory chunks and remove the entry */
	mem_chunk_t mc;
	for (mc = mem_chunk_list_begin(mc_list[node]);
	     mc != mem_chunk_list_end(mc_list[node]);
	     mc = mem_chunk_list_next(mc))
	{
		if (mc->data == state) {
			/* we found the data */
			/* remove it from the main list */
			mem_chunk_list_erase(mc_list[node], mc);

			/* put it in the list of buffers to be removed */
			mem_chunk_list_push_front(mc_list_to_free[node], mc);

			return;
		}
	}

	/* there was no corresponding buffer ... */

	release_mutex(&mc_mutex[node]);
}

size_t liberate_memory_on_node(mem_chunk_t mc, uint32_t node)
{
	size_t liberated = 0;

	data_state *state = mc->data;

	ASSERT(state->ops);
	ASSERT(state->ops->liberate_data_on_node);

	if (state->per_node[node].allocated && state->per_node[node].automatically_allocated)
	{
		state->ops->liberate_data_on_node(state, node);

		state->per_node[node].allocated = 0;

		/* XXX why do we need that ? */
		state->per_node[node].automatically_allocated = 0;

		liberated = mc->size;
	}

	return liberated;
}

int allocate_memory_on_node(data_state *state, uint32_t dst_node)
{
	unsigned attempts = 0;
	size_t allocated_memory;

	ASSERT(state);

	do {
		ASSERT(state->ops);
		ASSERT(state->ops->allocate_data_on_node);

		allocated_memory = state->ops->allocate_data_on_node(state, dst_node);

		if (!allocated_memory) {
			/* XXX perhaps we should find the proper granularity 
			 * not to waste our cache all the time */
			ASSERT(state->ops->get_size);
			size_t data_size = state->ops->get_size(state);
			reclaim_memory(dst_node, 2*data_size);
		}
		
	} while(!allocated_memory && attempts++ < 2);

	/* perhaps we could really not handle that capacity misses */
	if (!allocated_memory)
		goto nomem;

	register_mem_chunk(state, dst_node, allocated_memory);

	state->per_node[dst_node].allocated = 1;
	state->per_node[dst_node].automatically_allocated = 1;

	return 0;
nomem:
	ASSERT(!allocated_memory);
	return -1;
}
