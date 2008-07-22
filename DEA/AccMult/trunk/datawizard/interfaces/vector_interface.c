#include <common/util.h>
#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif

size_t allocate_vector_buffer_on_node(data_state *state, uint32_t dst_node);
void liberate_vector_buffer_on_node(data_state *state, uint32_t node);
void do_copy_vector_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node);
size_t dump_vector_interface(data_interface_t *interface, void *buffer);

/* declare a new data with the BLAS interface */
void monitor_vector_data(struct data_state_t *state, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize)
{
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		vector_interface_t *local_interface = &state->interface[node].vector;

		if (node == home_node) {
			local_interface->ptr = ptr;
		}
		else {
			local_interface->ptr = 0;
		}

		local_interface->nx = nx;
		local_interface->elemsize = elemsize;
	}

	state->interfaceid = BLAS_INTERFACE;

	state->allocation_method = &allocate_vector_buffer_on_node;
	state->deallocation_method = &liberate_vector_buffer_on_node;
	state->copy_1_to_1_method = &do_copy_vector_buffer_1_to_1;
	state->dump_interface = &dump_vector_interface;

	monitor_new_data(state, home_node);
}

struct dumped_vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t elemsize;
} __attribute__ ((packed));

size_t dump_vector_interface(data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_vector_interface_s *buffer = _buffer;

	buffer->ptr = (*interface).vector.ptr;
	buffer->nx = (*interface).vector.nx;
	buffer->elemsize = (*interface).vector.elemsize;

	return (sizeof(struct dumped_vector_interface_s));
}

/* offer an access to the data parameters */
uint32_t get_vector_nx(data_state *state)
{
	return (state->interface[0].vector.nx);
}

uintptr_t get_vector_local_ptr(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	ASSERT(state->per_node[node].allocated);

	return (state->interface[node].vector.ptr);
}

/* memory allocation/deallocation primitives for the vector interface */

/* returns the size of the allocated area */
size_t allocate_vector_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr;
	size_t allocated_memory;

	uint32_t nx = state->interface[dst_node].vector.nx;
	size_t elemsize = state->interface[dst_node].vector.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr = (uintptr_t)malloc(nx*elemsize);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUDA_RAM:
		case CUBLAS_RAM:
			cublasAlloc(nx, elemsize, (void **)&addr);
			break;
#endif
		default:
			assert(0);
	}

	if (addr) {
		/* allocation succeeded */
		allocated_memory = nx*elemsize;

		/* update the data properly in consequence */
		state->interface[dst_node].vector.ptr = addr;
	} else {
		/* allocation failed */
		allocated_memory = 0;
	}
	
	return allocated_memory;
}

void liberate_vector_buffer_on_node(data_state *state, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)state->interface[node].vector.ptr);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUBLAS_RAM:
		case CUDA_RAM:
			cublasFree((void*)state->interface[node].vector.ptr);
			break;
#endif
		default:
			assert(0);
	}
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
static void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	vector_interface_t *src_vector;
	vector_interface_t *dst_vector;

	src_vector = &state->interface[src_node].vector;
	dst_vector = &state->interface[dst_node].vector;

	cublasGetVector(src_vector->nx, src_vector->elemsize,
		(uint8_t *)src_vector->ptr, 1,
		(uint8_t *)dst_vector->ptr, 1);

	TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);
}

static void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	vector_interface_t *src_vector;
	vector_interface_t *dst_vector;

	src_vector = &state->interface[src_node].vector;
	dst_vector = &state->interface[dst_node].vector;

	cublasSetVector(src_vector->nx, src_vector->elemsize,
		(uint8_t *)src_vector->ptr, 1,
		(uint8_t *)dst_vector->ptr, 1);

	TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	uint32_t nx = state->interface[dst_node].vector.nx;
	size_t elemsize = state->interface[dst_node].vector.elemsize;

	uintptr_t ptr_src = state->interface[src_node].vector.ptr;
	uintptr_t ptr_dst = state->interface[dst_node].vector.ptr;

	memcpy((void *)ptr_dst, (void *)ptr_src, nx*elemsize);

	TRACE_DATA_COPY(src_node, dst_node, nx*elemsize);
}

void do_copy_vector_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
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
