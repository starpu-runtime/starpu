#include <pthread.h>
#include "copy-driver.h"
#include "memalloc.h"

mem_node_descr descr;
static pthread_key_t memory_node_key;

unsigned register_memory_node(node_kind kind)
{
	unsigned nnodes;
	/* ATOMIC_ADD returns the new value ... */
	nnodes = ATOMIC_ADD(&descr.nnodes, 1);

	descr.nodes[nnodes-1] = kind;

	return (nnodes-1);
}

void init_memory_nodes()
{
	/* there is no node yet, subsequent nodes will be 
	 * added using register_memory_node */
	descr.nnodes = 0;

	pthread_key_create(&memory_node_key, NULL);

	unsigned i;
	for (i = 0; i < MAXNODES; i++) 
	{
		descr.nodes[i] = UNUSED; 
	}

	init_mem_chunk_lists();
}

void set_local_memory_node_key(unsigned *node)
{
	pthread_setspecific(memory_node_key, node);
}

unsigned get_local_memory_node(void)
{
	unsigned *memory_node;
	memory_node = pthread_getspecific(memory_node_key);
	ASSERT(memory_node);
	return *memory_node;
}

inline node_kind get_node_kind(uint32_t node)
{
	return descr.nodes[node];
}

void allocate_per_node_buffer(data_state *state, uint32_t node)
{
	if (!state->per_node[node].allocated) {
		/* there is no room available for the data yet */
		allocate_memory_on_node(state, node);
	}
}

void driver_copy_data_1_to_1(data_state *state, uint32_t src_node, 
				uint32_t dst_node, unsigned donotread)
{
	/* first make sure the destination has an allocated buffer */
	allocate_per_node_buffer(state, dst_node);

	/* if there is no need to actually read the data, 
	 * we do not perform any transfer */
	if (!donotread) {
		state->copy_1_to_1_method(state, src_node, dst_node, 
			descr.nodes[src_node], descr.nodes[dst_node]);

	}

}

static uint32_t choose_src_node(uint32_t src_node_mask)
{
	unsigned src_node = 0;
	unsigned i;

	/* first find the node that will be the actual source */
	for (i = 0; i < MAXNODES; i++)
	{
		if (src_node_mask & (1<<i))
		{
			/* this is a potential candidate */
			src_node = i;

			/* however GPU are expensive sources, really !
			 * 	other should be ok */
			if (descr.nodes[i] != CUBLAS_RAM 
				&& descr.nodes[i] != CUDA_RAM)
				break;

			/* XXX do a better algorithm to distribute the memory copies */
		}
	}

	return src_node;
}

void driver_copy_data(data_state *state, uint32_t src_node_mask,
					 uint32_t dst_node)
{
	uint32_t src_node = choose_src_node(src_node_mask);

	driver_copy_data_1_to_1(state, src_node, dst_node, 0);
}
