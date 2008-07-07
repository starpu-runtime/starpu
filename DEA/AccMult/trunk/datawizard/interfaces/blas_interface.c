#include <common/util.h>
#include "../data_parameters.h"
#include "blas_interface.h"

/* declare a new data with the BLAS interface */
void monitor_blas_data(data_state *state, uint32_t home_node,
			uintptr_t ptr, uint32_t ld, uint32_t nx,
			uint32_t ny, size_t elemsize)
{
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		blas_interface_t *local_interface = &state->interface[node].blas;

		if (node == home_node) {
			local_interface.ptr = ptr;
			local_interface.ld  = ld;
		}
		else {
			local_interface.ptr = 0;
			local_interface.ld  = 0;
		}

		local_interface.nx = nx;
		local_interface.ny = ny;
		local_interface.elemsize = elemsize;
	}

	state->interfaceid = BLAS_INTERFACE;

	state->allocation_method = allocate_blas_buffer_on_node;
	state->deallocation_method = liberate_blas_buffer_on_node;

	monitor_new_data(state, home_node);
}

/* offer an access to the data parameters */
uint32_t get_blas_local_nx(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	return (state->interface[node].blas.nx);
}

uint32_t get_blas_local_ny(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	return (state->interface[node].blas.ny);
}

uint32_t get_blas_local_ld(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	ASSERT(state->per_node[*memory_node].allocated);

	return (state->interface[node].blas.ld);
}

uintptr_t get_blas_local_ptr(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	ASSERT(state->per_node[*memory_node].allocated);

	return (state->interface[node].blas.ptr);
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
size_t allocate_blas_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr;
	size_t allocated_memory;

	uint32_t nx = state->per_node[dst_node].blas.nx;
	uint32_t ny = state->per_node[dst_node].blas.ny;
	size_t ny = state->per_node[dst_node].blas.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr = (uintptr_t)malloc(nx*ny*elemsize);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUDA_RAM:
		case CUBLAS_RAM:
			cublasAlloc(nx*ny, elemsize, (void **)&addr);
			break;
#endif
		default:
			assert(0);
	}

	if (addr) {
		/* allocation succeeded */
		allocated_memory = nx*ny*elemsize;

		/* update the data properly in consequence */
		state->per_node[dst_node].blas.addr = addr;
		state->per_node[dst_node].blas.ld = nx;
	} else {
		/* allocation failed */
		allocated_memory = 0;
	}
	
	return allocated_memory;
}

void liberate_blas_buffer_on_node(data_state *state, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)state->per_node[node].ptr);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUBLAS_RAM:
		case CUDA_RAM:
			cublasFree((void*)state->per_node[node].ptr);
			break;
#endif
		default:
			assert(0);
	}
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
static void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasGetMatrix(state->nx, state->ny, state->elemsize,
		(uint8_t *)state->per_node[src_node].ptr,
		state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr,
		state->per_node[dst_node].ld);
}

static void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	cublasSetMatrix(state->nx, state->ny, state->elemsize,
		(uint8_t *)state->per_node[src_node].ptr,
		state->per_node[src_node].ld,
		(uint8_t *)state->per_node[dst_node].ptr,
		state->per_node[dst_node].ld);
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
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


void do_copy_blas_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	node_kind src_kind = get_node_kind(src_node);
	node_kind dst_kind = get_node_kind(dst_node);

	switch (dst_kind) {
	case RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> RAM */
				 dummy_copy_ram_to_ram(state, src_node, dst_node);
				 break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
			case CUBLAS_RAM:
			case CUDA_RAM:
				/* CUBLAS_RAM -> RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				copy_cublas_to_ram(state, src_node, dst_node);
				break;
#endif
			case SPU_LS:
				ASSERT(0); // TODO
				break;
			case UNUSED:
				printf("error node %d UNUSED\n", src_node);
			default:
				assert(0);
				break;
		}
		break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	case CUDA_RAM:
	case CUBLAS_RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> CUBLAS_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				ASSERT(get_local_memory_node() == dst_node);
				copy_ram_to_cublas(state, src_node, dst_node);
				break;
			case CUDA_RAM:
			case CUBLAS_RAM:
			case SPU_LS:
				ASSERT(0); // TODO 
				break;
			case UNUSED:
			default:
				ASSERT(0);
				break;
		}
		break;
#endif
	case SPU_LS:
		ASSERT(0); // TODO
		break;
	case UNUSED:
	default:
		assert(0);
		break;
	}
}
