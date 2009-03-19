#include <common/util.h>
#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

size_t allocate_csr_buffer_on_node(struct data_state_t *state, uint32_t dst_node);
void liberate_csr_buffer_on_node(data_state *state, uint32_t node);
size_t dump_csr_interface(data_interface_t *interface, void *_buffer);
int do_copy_csr_buffer_1_to_1(struct data_state_t *state, uint32_t src_node, uint32_t dst_node);
size_t csr_interface_get_size(struct data_state_t *state);
uint32_t footprint_csr_interface_crc32(data_state *state, uint32_t hstate);

struct data_interface_ops_t interface_csr_ops = {
	.allocate_data_on_node = allocate_csr_buffer_on_node,
	.liberate_data_on_node = liberate_csr_buffer_on_node,
	.copy_data_1_to_1 = do_copy_csr_buffer_1_to_1,
	.dump_data_interface = dump_csr_interface,
	.get_size = csr_interface_get_size,
	.interfaceid = CSR_INTERFACE,
	.footprint = footprint_csr_interface_crc32
};

/* declare a new data with the BLAS interface */
void monitor_csr_data(struct data_state_t *state, uint32_t home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize)
{
	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		csr_interface_t *local_interface = &state->interface[node].csr;

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
		local_interface->elemsize = elemsize;

	}

	state->ops = &interface_csr_ops;

	monitor_new_data(state, home_node, 0);
}

static inline uint32_t footprint_csr_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), data_state *state, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(get_csr_nnz(state), hash);

	return hash;
}

uint32_t footprint_csr_interface_crc32(data_state *state, uint32_t hstate)
{
	return footprint_csr_interface_generic(crc32_be, state, hstate);
}



struct dumped_csr_interface_s {
	uint32_t nnz;
	uint32_t nrow;
	uintptr_t nzval;
	uint32_t *colind;
	uint32_t *rowptr;
	uint32_t firstentry;
	uint32_t elemsize;
}  __attribute__ ((packed));

size_t dump_csr_interface(data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_csr_interface_s *buffer = _buffer;

	buffer->nnz = (*interface).csr.nnz;
	buffer->nrow = (*interface).csr.nrow;
	buffer->nzval = (*interface).csr.nzval;
	buffer->colind = (*interface).csr.colind;
	buffer->rowptr = (*interface).csr.rowptr;
	buffer->firstentry = (*interface).csr.firstentry;
	buffer->elemsize = (*interface).csr.elemsize;

	return (sizeof(struct dumped_csr_interface_s));
}

/* offer an access to the data parameters */
uint32_t get_csr_nnz(struct data_state_t *state)
{
	return (state->interface[0].csr.nnz);
}

uint32_t get_csr_nrow(struct data_state_t *state)
{
	return (state->interface[0].csr.nrow);
}

uint32_t get_csr_firstentry(struct data_state_t *state)
{
	return (state->interface[0].csr.firstentry);
}

size_t get_csr_elemsize(struct data_state_t *state)
{
	return (state->interface[0].csr.elemsize);
}

uintptr_t get_csr_local_nzval(struct data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.nzval);
}

uint32_t *get_csr_local_colind(struct data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.colind);
}

uint32_t *get_csr_local_rowptr(struct data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.rowptr);
}

size_t csr_interface_get_size(struct data_state_t *state)
{
	size_t size;

	uint32_t nnz = get_csr_nnz(state);
	uint32_t nrow = get_csr_nrow(state);
	size_t elemsize = get_csr_elemsize(state);

	size = nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	return size;
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
size_t allocate_csr_buffer_on_node(struct data_state_t *state, uint32_t dst_node)
{
	uintptr_t addr_nzval;
	uint32_t *addr_colind, *addr_rowptr;
	size_t allocated_memory;

	/* we need the 3 arrays to be allocated */

	uint32_t nnz = state->interface[dst_node].csr.nnz;
	uint32_t nrow = state->interface[dst_node].csr.nrow;
	size_t elemsize = state->interface[dst_node].csr.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr_nzval = (uintptr_t)malloc(nnz*elemsize);
			if (!addr_nzval)
				goto fail_nzval;

			addr_colind = malloc(nnz*sizeof(uint32_t));
			if (!addr_colind)
				goto fail_colind;

			addr_rowptr = malloc((nrow+1)*sizeof(uint32_t));
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasAlloc(nnz, elemsize, (void **)&addr_nzval);
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
		nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	state->interface[dst_node].csr.nzval = addr_nzval;
	state->interface[dst_node].csr.colind = addr_colind;
	state->interface[dst_node].csr.rowptr = addr_rowptr;
	
	return allocated_memory;

fail_rowptr:
	switch(kind) {
		case RAM:
			free((void *)addr_colind);
#ifdef USE_CUDA
		case CUDA_RAM:
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
#ifdef USE_CUDA
		case CUDA_RAM:
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

void liberate_csr_buffer_on_node(data_state *state, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)state->interface[node].csr.nzval);
			free((void*)state->interface[node].csr.colind);
			free((void*)state->interface[node].csr.rowptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)state->interface[node].csr.nzval);
			cublasFree((void*)state->interface[node].csr.colind);
			cublasFree((void*)state->interface[node].csr.rowptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static void copy_cublas_to_ram(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	csr_interface_t *src_csr;
	csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cublasGetVector(nnz, elemsize, (uint8_t *)src_csr->nzval, 1, 
					(uint8_t *)dst_csr->nzval, 1);

	cublasGetVector(nnz, sizeof(uint32_t), (uint8_t *)src_csr->colind, 1, 
						(uint8_t *)dst_csr->colind, 1);

	cublasGetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_csr->rowptr, 1, 
						(uint8_t *)dst_csr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

}

static void copy_ram_to_cublas(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	csr_interface_t *src_csr;
	csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cublasSetVector(nnz, elemsize, (uint8_t *)src_csr->nzval, 1, 
					(uint8_t *)dst_csr->nzval, 1);

	cublasSetVector(nnz, sizeof(uint32_t), (uint8_t *)src_csr->colind, 1, 
						(uint8_t *)dst_csr->colind, 1);

	cublasSetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_csr->rowptr, 1, 
						(uint8_t *)dst_csr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static void dummy_copy_ram_to_ram(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
{

	csr_interface_t *src_csr;
	csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	memcpy((void *)dst_csr->nzval, (void *)src_csr->nzval, nnz*elemsize);

	memcpy((void *)dst_csr->colind, (void *)src_csr->colind, nnz*sizeof(uint32_t));

	memcpy((void *)dst_csr->rowptr, (void *)src_csr->rowptr, (nrow+1)*sizeof(uint32_t));

	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));
}


int do_copy_csr_buffer_1_to_1(struct data_state_t *state, uint32_t src_node, uint32_t dst_node)
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
#ifdef USE_CUDA
			case CUDA_RAM:
				/* CUBLAS_RAM -> RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				if (get_local_memory_node() == src_node)
				{
					copy_cublas_to_ram(state, src_node, dst_node);
				}
				else
				{
					post_data_request(state, src_node, dst_node);
				}
				break;
#endif
			case SPU_LS:
				STARPU_ASSERT(0); // TODO
				break;
			case UNUSED:
				printf("error node %d UNUSED\n", src_node);
			default:
				assert(0);
				break;
		}
		break;
#ifdef USE_CUDA
	case CUDA_RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> CUBLAS_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				STARPU_ASSERT(get_local_memory_node() == dst_node);
				copy_ram_to_cublas(state, src_node, dst_node);
				break;
			case CUDA_RAM:
			case SPU_LS:
				STARPU_ASSERT(0); // TODO 
				break;
			case UNUSED:
			default:
				STARPU_ASSERT(0);
				break;
		}
		break;
#endif
	case SPU_LS:
		STARPU_ASSERT(0); // TODO
		break;
	case UNUSED:
	default:
		assert(0);
		break;
	}

	return 0;
}
