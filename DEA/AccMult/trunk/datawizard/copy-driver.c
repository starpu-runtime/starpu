#include <pthread.h>
#include "copy-driver.h"

static mem_node_descr descr;
static pthread_key_t memory_node_key;

unsigned register_memory_node(node_kind kind)
{
	unsigned nnodes;
	/* ATOMIC_ADD returns the new value ... */
	nnodes = ATOMIC_ADD(&descr.nnodes, 1);

	descr.nodes[nnodes] = kind;

	return nnodes;
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

void driver_copy_data_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
{
//	printf("copy data from node %d to node %d\n", src_node, dst_node);

	/* first make sure the destination has an allocated buffer */
	/* XXX this should depend on the node kind */
	if (!state->per_node[dst_node].allocated) {
		/* there is no room available for the data yet */
		printf("MALLOC on node %d \n", dst_node);
		state->per_node[dst_node].ptr =
			(uintptr_t) malloc(state->length);
		state->per_node[dst_node].allocated = 1;
		state->per_node[dst_node].automatically_allocated = 1;
	}

	/* XXX this should depend on the node kind */
	memcpy((void *)state->per_node[dst_node].ptr,
		(void *)state->per_node[src_node].ptr,
		state->length);
}

void driver_copy_data(data_state *state, uint32_t src_node_mask, uint32_t dst_node)
{
	unsigned src_node;
	unsigned i;

	/* first find the node that will be the actual source */
	for (i = 0; i < MAXNODES; i++)
	{
		if (src_node_mask & (1<<i))
		{
			/* this is a potential candidate */
			src_node = i;

			/* however GPU are expensive sources, really ! other should be ok */
			if (descr.nodes[i] != CUBLAS_RAM && descr.nodes[i] != CUDA_RAM)
				break;

			/* XXX do a better algorithm to distribute the memory copies */
		}
	}

	driver_copy_data_1_to_1(state, src_node, dst_node);
}
