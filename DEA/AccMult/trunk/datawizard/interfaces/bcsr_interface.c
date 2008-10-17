#include <common/util.h>
#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif

/*
 * BCSR : blocked CSR, we use blocks of size (r x c)
 */
size_t allocate_bcsr_buffer_on_node(struct data_state_t *state, uint32_t dst_node);
void liberate_bcsr_buffer_on_node(data_state *state, uint32_t node);
size_t dump_bcsr_interface(data_interface_t *interface, void *_buffer);
void do_copy_bcsr_buffer_1_to_1(struct data_state_t *state, uint32_t src_node, uint32_t dst_node);

void monitor_bcsr_data(struct data_state_t *state, uint32_t home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry,  uint32_t r, uint32_t c, size_t elemsize)
{
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		bcsr_interface_t *local_interface = &state->interface[node].bcsr;

		if (node == home_node) {
			local_interface->nzval = nzval;
			local_interface->colind = colind;
			local_interface->rowptr = rowptr;
		}
		else {
			local_interface->nzval = 0;
			local_interface->colind = NULL;
			local_interface->rowptr = NULL;
		}

		local_interface->nnz = nnz;
		local_interface->nrow = nrow;
		local_interface->firstentry = firstentry;
		local_interface->r = r;
		local_interface->c = c;
		local_interface->elemsize = elemsize;
	}

	state->interfaceid = BLAS_INTERFACE;

	state->allocation_method = &allocate_bcsr_buffer_on_node;
	state->deallocation_method = &liberate_bcsr_buffer_on_node;
	state->copy_1_to_1_method = &do_copy_bcsr_buffer_1_to_1;
	state->dump_interface = &dump_bcsr_interface;

	monitor_new_data(state, home_node);
}

struct dumped_bcsr_interface_s {
	uint32_t nnz;
	uint32_t nrow;
	uintptr_t nzval;
	uint32_t *colind;
	uint32_t *rowptr;
	uint32_t firstentry;
	uint32_t r;
	uint32_t c;
	uint32_t elemsize;
}  __attribute__ ((packed));

size_t dump_bcsr_interface(data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_bcsr_interface_s *buffer = _buffer;

	buffer->nnz = (*interface).bcsr.nnz;
	buffer->nrow = (*interface).bcsr.nrow;
	buffer->nzval = (*interface).bcsr.nzval;
	buffer->colind = (*interface).bcsr.colind;
	buffer->rowptr = (*interface).bcsr.rowptr;
	buffer->firstentry = (*interface).bcsr.firstentry;
	buffer->r = (*interface).bcsr.r;
	buffer->c = (*interface).bcsr.c;
	buffer->elemsize = (*interface).bcsr.elemsize;

	return (sizeof(struct dumped_bcsr_interface_s));
}

/* offer an access to the data parameters */
uint32_t get_bcsr_nnz(struct data_state_t *state)
{
	return (state->interface[0].bcsr.nnz);
}

uint32_t get_bcsr_nrow(struct data_state_t *state)
{
	return (state->interface[0].bcsr.nrow);
}

uint32_t get_bcsr_firstentry(struct data_state_t *state)
{
	return (state->interface[0].bcsr.firstentry);
}

uint32_t get_bcsr_r(struct data_state_t *state)
{
	return (state->interface[0].bcsr.r);
}

uint32_t get_bcsr_c(struct data_state_t *state)
{
	return (state->interface[0].bcsr.c);
}

size_t get_bcsr_elemsize(struct data_state_t *state)
{
	return (state->interface[0].bcsr.elemsize);
}

uintptr_t get_bcsr_local_nzval(struct data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	ASSERT(state->per_node[node].allocated);

	return (state->interface[node].bcsr.nzval);
}

uint32_t *get_bcsr_local_colind(struct data_state_t *state)
{
//	unsigned node;
//	node = get_local_memory_node();
//
//	ASSERT(state->per_node[node].allocated);
//
//	return (state->interface[node].bcsr.colind);

	/* XXX */
	return (state->interface[0].bcsr.colind);
}

uint32_t *get_bcsr_local_rowptr(struct data_state_t *state)
{
//	unsigned node;
//	node = get_local_memory_node();
//
//	ASSERT(state->per_node[node].allocated);
//
//	return (state->interface[node].bcsr.rowptr);
	
	/* XXX */
	return (state->interface[0].bcsr.rowptr);
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
size_t allocate_bcsr_buffer_on_node(struct data_state_t *state, uint32_t dst_node)
{
	uintptr_t addr_nzval;
	uint32_t *addr_colind, *addr_rowptr;
	size_t allocated_memory;

	/* we need the 3 arrays to be allocated */

	uint32_t nnz = state->interface[dst_node].bcsr.nnz;
	uint32_t nrow = state->interface[dst_node].bcsr.nrow;
	size_t elemsize = state->interface[dst_node].bcsr.elemsize;

	uint32_t r = state->interface[dst_node].bcsr.r;
	uint32_t c = state->interface[dst_node].bcsr.c;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr_nzval = (uintptr_t)malloc(nnz*r*c*elemsize);
			if (!addr_nzval)
				goto fail_nzval;

			addr_colind = malloc(nnz*sizeof(uint32_t));
			if (!addr_colind)
				goto fail_colind;

			addr_rowptr = malloc((nrow+1)*sizeof(uint32_t));
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUDA_RAM:
		case CUBLAS_RAM:
			cublasAlloc(nnz*r*c, elemsize, (void **)&addr_nzval);
			if (!addr_nzval)
				goto fail_nzval;

			cublasAlloc(nnz, sizeof(uint32_t), (void **)&addr_colind);
			if (!addr_colind)
				goto fail_colind;

			cublasAlloc((nrow+1), sizeof(uint32_t), (void **)&addr_rowptr);
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#endif
		default:
			assert(0);
	}

	/* allocation succeeded */
	allocated_memory = 
		nnz*r*c*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	state->interface[dst_node].bcsr.nzval = addr_nzval;
	state->interface[dst_node].bcsr.colind = addr_colind;
	state->interface[dst_node].bcsr.rowptr = addr_rowptr;
	
	return allocated_memory;

fail_rowptr:
	switch(kind) {
		case RAM:
			free((void *)addr_colind);
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUDA_RAM:
		case CUBLAS_RAM:
			cublasFree((void*)addr_colind);
			break;
#endif
		default:
			assert(0);
	}

fail_colind:
	switch(kind) {
		case RAM:
			free((void *)addr_nzval);
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUDA_RAM:
		case CUBLAS_RAM:
			cublasFree((void*)addr_nzval);
			break;
#endif
		default:
			assert(0);
	}

fail_nzval:

	/* allocation failed */
	allocated_memory = 0;

	return allocated_memory;
}

void liberate_bcsr_buffer_on_node(data_state *state, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)state->interface[node].bcsr.nzval);
			free((void*)state->interface[node].bcsr.colind);
			free((void*)state->interface[node].bcsr.rowptr);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case CUBLAS_RAM:
		case CUDA_RAM:
			cublasFree((void*)state->interface[node].bcsr.nzval);
			cublasFree((void*)state->interface[node].bcsr.colind);
			cublasFree((void*)state->interface[node].bcsr.rowptr);
			break;
#endif
		default:
			assert(0);
	}
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
static void copy_cublas_to_ram(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	bcsr_interface_t *src_bcsr;
	bcsr_interface_t *dst_bcsr;

	src_bcsr = &state->interface[src_node].bcsr;
	dst_bcsr = &state->interface[dst_node].bcsr;

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	cublasGetVector(nnz*r*c, elemsize, (uint8_t *)src_bcsr->nzval, 1, 
			 		   (uint8_t *)dst_bcsr->nzval, 1);

	cublasGetVector(nnz, sizeof(uint32_t), (uint8_t *)src_bcsr->colind, 1, 
						(uint8_t *)dst_bcsr->colind, 1);

	cublasGetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_bcsr->rowptr, 1, 
						(uint8_t *)dst_bcsr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*r*c*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

}

static void copy_ram_to_cublas(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	bcsr_interface_t *src_bcsr;
	bcsr_interface_t *dst_bcsr;

	src_bcsr = &state->interface[src_node].bcsr;
	dst_bcsr = &state->interface[dst_node].bcsr;

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	cublasSetVector(nnz*r*c, elemsize, (uint8_t *)src_bcsr->nzval, 1, 
					(uint8_t *)dst_bcsr->nzval, 1);

	cublasSetVector(nnz, sizeof(uint32_t), (uint8_t *)src_bcsr->colind, 1, 
						(uint8_t *)dst_bcsr->colind, 1);

	cublasSetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_bcsr->rowptr, 1, 
						(uint8_t *)dst_bcsr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*r*c*elemsize + (nnz+nrow+1)*sizeof(uint32_t));
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static void dummy_copy_ram_to_ram(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{

	bcsr_interface_t *src_bcsr;
	bcsr_interface_t *dst_bcsr;

	src_bcsr = &state->interface[src_node].bcsr;
	dst_bcsr = &state->interface[dst_node].bcsr;

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	memcpy((void *)dst_bcsr->nzval, (void *)src_bcsr->nzval, nnz*elemsize*r*c);

	memcpy((void *)dst_bcsr->colind, (void *)src_bcsr->colind, nnz*sizeof(uint32_t));

	memcpy((void *)dst_bcsr->rowptr, (void *)src_bcsr->rowptr, (nrow+1)*sizeof(uint32_t));

	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize*r*c + (nnz+nrow+1)*sizeof(uint32_t));
}


void do_copy_bcsr_buffer_1_to_1(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
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
