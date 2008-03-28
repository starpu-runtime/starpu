#include "copy-driver.h"

static mem_node_descr descr;

void register_memory_node(node_kind kind)
{
	unsigned nnodes;
	/* ATOMIC_ADD returns the new value ... */
	nnodes = ATOMIC_ADD(&descr.nnodes, 1);

	descr.nodes[nnodes] = kind;
}

void init_drivers()
{
	/* there is no node yet, subsequent nodes will be 
	 * added using register_memory_node */
	descr.nnodes = 0;

	unsigned i;
	for (i = 0; i < MAXNODES; i++) 
	{
		descr.nodes[i] = UNUSED; 
	}
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
			if (descr.nodes[i] != GPU_RAM)
				break;

			/* XXX do a better algorithm to distribute the memory copies */
		}
	}

	printf("copy data from node %d to node %d\n", src_node, dst_node);

	memcpy((void *)state->per_node[dst_node].ptr,
		(void *)state->per_node[src_node].ptr,
		state->length);
}
