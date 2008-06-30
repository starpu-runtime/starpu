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

uint32_t get_local_ld(data_state *state)
{
	unsigned *memory_node;
	memory_node = pthread_getspecific(memory_node_key);
	ASSERT(memory_node);
	ASSERT(state->per_node[*memory_node].allocated);
	return state->per_node[*memory_node].ld;
}

uintptr_t get_local_ptr(data_state *state)
{
	unsigned *memory_node;
	memory_node = pthread_getspecific(memory_node_key);
	ASSERT(memory_node);
	ASSERT(state->per_node[*memory_node].allocated);
	return state->per_node[*memory_node].ptr;
}

uint32_t get_local_nx(data_state *state)
{
	return state->nx;
}

uint32_t get_local_ny(data_state *state)
{
	return state->ny;
}

/* as not all platform easily have a BLAS lib installed ... */
void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node,
						uint32_t dst_node)
{
	unsigned y;
	for (y = 0; y < state->ny; y++)
	{
		uint32_t src_offset = 
			y*state->per_node[src_node].ld*state->elemsize;
		uint32_t dst_offset = 
			y*state->per_node[dst_node].ld*state->elemsize;
		memcpy((void *)(state->per_node[dst_node].ptr + dst_offset),
			(void *)(state->per_node[src_node].ptr + src_offset),
			state->nx*state->elemsize);
	}
}

#ifdef USE_CUBLAS
void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasGetMatrix(state->nx, state->ny, state->elemsize, 
		(uint8_t *)state->per_node[src_node].ptr,
		state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr,
		state->per_node[dst_node].ld);

}

void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasSetMatrix(state->nx, state->ny, state->elemsize, 
		(uint8_t *)state->per_node[src_node].ptr,
		state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr,
		state->per_node[dst_node].ld);
}
#endif // USE_CUBLAS

void allocate_per_node_buffer(data_state *state, uint32_t node)
{
	if (!state->per_node[node].allocated) {
		/* there is no room available for the data yet */
		allocate_memory_on_node(state, node);
	}
}

void do_copy_data_1_to_1(data_state *state, uint32_t src_node,
					    uint32_t dst_node)
{
	/* XXX clean that !  */
	switch(descr.nodes[dst_node]) {
	case RAM:
		switch(descr.nodes[src_node]) {
		case RAM:
			/* RAM -> RAM */
			dummy_copy_ram_to_ram(state, src_node, dst_node);
			break;
		case CUDA_RAM:
			/* CUDA_RAM -> RAM */
			ASSERT(0); // TODO 
			break;
#ifdef USE_CUBLAS
		case CUBLAS_RAM:
			/* CUBLAS_RAM -> RAM */
			/* only the proper CUBLAS thread can initiate this ! */
		//	ASSERT(get_local_memory_node() == src_node);
			//XXX there should be a way to control the
			copy_cublas_to_ram(state, src_node, dst_node);
			break;
#endif // USE_CUBLAS
		case SPU_LS:
			ASSERT(0); // TODO 
			break;
		case UNUSED:
			printf("error node %d UNUSED\n", src_node);
		default:
			ASSERT(0);
			break;
		}
	break;
	case CUDA_RAM:
	ASSERT(0); // TODO 
	switch(descr.nodes[src_node]) {
		case RAM:
			break;
		case CUDA_RAM:
			break;
		case CUBLAS_RAM:
			break;
		case SPU_LS:
			break;
		case UNUSED:
		default:
			ASSERT(0);
			break;
	}
	break;
#ifdef USE_CUBLAS
	case CUBLAS_RAM:
	switch(descr.nodes[src_node]) {
		case RAM:
			/* RAM -> CUBLAS_RAM */
			/* only the proper CUBLAS thread can initiate this ! */
			ASSERT(get_local_memory_node() == dst_node);
			copy_ram_to_cublas(state, src_node, dst_node);
			break;
		case CUDA_RAM:
			ASSERT(0); // TODO 
			break;
		case CUBLAS_RAM:
			ASSERT(0); // TODO 
			break;
		case SPU_LS:
			ASSERT(0); // TODO 
			break;
		case UNUSED:
		default:
			ASSERT(0);
			break;
	}
	break;
#endif // USE_CUBLAS
	case SPU_LS:
		ASSERT(0); // TODO 
		switch(descr.nodes[src_node]) {
			case RAM:
				break;
			case CUDA_RAM:
				break;
			case CUBLAS_RAM:
				break;
			case SPU_LS:
				break;
			case UNUSED:
			default:
				ASSERT(0);
				break;
		}
		break;
	case UNUSED:
	default:
		ASSERT(0);
		break;
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
		do_copy_data_1_to_1(state, src_node, dst_node);
	}

}

static uint32_t choose_src_node(data_state *state, uint32_t src_node_mask)
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
	uint32_t src_node = choose_src_node(state, src_node_mask);

	driver_copy_data_1_to_1(state, src_node, dst_node, 0);
}
