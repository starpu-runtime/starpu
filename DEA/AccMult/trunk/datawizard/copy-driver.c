#include <pthread.h>
#include "copy-driver.h"

static mem_node_descr descr;
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

static void allocate_memory_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr = 0;

	switch(descr.nodes[dst_node]) {
		case RAM:
			addr = (uintptr_t) malloc(state->nx*state->ny);
			//printf("RAM addr allocated : %p \n", addr);
			break;
#ifdef USE_CUBLAS
		case CUBLAS_RAM:
			cublasAlloc(state->nx*state->ny, 1, (void **)&addr); 
			printf("CUBLAS addr allocated : %p \n", addr);
			break;
#endif
		default:
			ASSERT(0);
	}
	
	/* TODO handle capacity misses */
	ASSERT(addr);

	state->per_node[dst_node].ptr = addr; 
	state->per_node[dst_node].ld = state->nx; 
	state->per_node[dst_node].allocated = 1;
	state->per_node[dst_node].automatically_allocated = 1;
}

/* as not all platform easily have a BLAS lib installed ... */
void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	unsigned y;
	for (y = 0; y < state->ny; y++)
	{
		uint32_t src_offset = y*state->per_node[src_node].ld;
		uint32_t dst_offset = y*state->per_node[dst_node].ld;
		memcpy((void *)(state->per_node[dst_node].ptr + dst_offset),
			(void *)(state->per_node[src_node].ptr + src_offset), state->nx);
	}
}

#ifdef USE_CUBLAS
void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasGetMatrix(state->nx, state->ny, 1, 
		(uint8_t *)state->per_node[src_node].ptr, state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr, state->per_node[dst_node].ld);

}

void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasSetMatrix(state->nx, state->ny, 1, 
		(uint8_t *)state->per_node[src_node].ptr, state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr, state->per_node[dst_node].ld);
}
#endif // USE_CUBLAS

void driver_copy_data_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	//printf("copy data from node %d to node %d\n", src_node, dst_node);

	/* first make sure the destination has an allocated buffer */
	/* TODO clean */
	if (!state->per_node[dst_node].allocated) {
		/* there is no room available for the data yet */
		allocate_memory_on_node(state, dst_node);
	}

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
			//		ASSERT(get_local_memory_node() == src_node);
			//		XXX there should be a way to control the
				//	printf("CUBLAS %p -> RAM %p \n", state->per_node[src_node].ptr, state->per_node[dst_node].ptr);
					copy_cublas_to_ram(state, src_node, dst_node);
					//printf("debug CUBLAS GET = CUBLAS %d -> RAM %d\n", *(int*)state->per_node[src_node].ptr, *(int*)state->per_node[dst_node].ptr);
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
//					printf("debug CUBLAS SET = RAM %d -> CUBLAS %d\n", *(int*)state->per_node[src_node].ptr, *(int*)state->per_node[dst_node].ptr);
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
