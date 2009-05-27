/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <common/config.h>
#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

size_t allocate_vector_buffer_on_node(data_state *state, uint32_t dst_node);
void liberate_vector_buffer_on_node(starpu_data_interface_t *interface, uint32_t node);
int do_copy_vector_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node);
size_t dump_vector_interface(starpu_data_interface_t *interface, void *buffer);
size_t vector_interface_get_size(struct starpu_data_state_t *state);
uint32_t footprint_vector_interface_crc32(data_state *state, uint32_t hstate);
void display_vector_interface(data_state *state, FILE *f);
#ifdef USE_GORDON
int convert_vector_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif

struct data_interface_ops_t interface_vector_ops = {
	.allocate_data_on_node = allocate_vector_buffer_on_node,
	.liberate_data_on_node = liberate_vector_buffer_on_node,
	.copy_data_1_to_1 = do_copy_vector_buffer_1_to_1,
	.dump_data_interface = dump_vector_interface,
	.get_size = vector_interface_get_size,
	.footprint = footprint_vector_interface_crc32,
#ifdef USE_GORDON
	.convert_to_gordon = convert_vector_to_gordon,
#endif
	.interfaceid = VECTOR_INTERFACE,
	.display = display_vector_interface
};

#ifdef USE_GORDON
int convert_vector_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	*ptr = (*interface).vector.ptr;
	(*ss).size = (*interface).vector.nx * (*interface).vector.elemsize;

	return 0;
}
#endif

/* declare a new data with the BLAS interface */
void starpu_register_vector_data(struct starpu_data_state_t **handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize)
{
	struct starpu_data_state_t *state = calloc(1, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(state);

	STARPU_ASSERT(handle);
	*handle = state;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_vector_interface_t *local_interface = &state->interface[node].vector;

		if (node == home_node) {
			local_interface->ptr = ptr;
		}
		else {
			local_interface->ptr = 0;
		}

		local_interface->nx = nx;
		local_interface->elemsize = elemsize;
	}

	state->ops = &interface_vector_ops;

	register_new_data(state, home_node, 0);
}


static inline uint32_t footprint_vector_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), data_state *state, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(starpu_get_vector_nx(state), hash);

	return hash;
}

uint32_t footprint_vector_interface_crc32(data_state *state, uint32_t hstate)
{
	return footprint_vector_interface_generic(crc32_be, state, hstate);
}

struct dumped_vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t elemsize;
} __attribute__ ((packed));

void display_vector_interface(data_state *state, FILE *f)
{
	starpu_vector_interface_t *interface;
	interface =  &state->interface[0].vector;

	fprintf(f, "%d\t", interface->nx);
}


size_t dump_vector_interface(starpu_data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_vector_interface_s *buffer = _buffer;

	buffer->ptr = (*interface).vector.ptr;
	buffer->nx = (*interface).vector.nx;
	buffer->elemsize = (*interface).vector.elemsize;

	return (sizeof(struct dumped_vector_interface_s));
}

size_t vector_interface_get_size(struct starpu_data_state_t *state)
{
	size_t size;
	starpu_vector_interface_t *interface;

	interface =  &state->interface[0].vector;

	size = interface->nx*interface->elemsize;

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_get_vector_nx(data_state *state)
{
	return (state->interface[0].vector.nx);
}

uintptr_t starpu_get_vector_local_ptr(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].vector.ptr);
}

/* memory allocation/deallocation primitives for the vector interface */

/* returns the size of the allocated area */
size_t allocate_vector_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr = 0;
	size_t allocated_memory;

	uint32_t nx = state->interface[dst_node].vector.nx;
	size_t elemsize = state->interface[dst_node].vector.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr = (uintptr_t)malloc(nx*elemsize);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
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

void liberate_vector_buffer_on_node(starpu_data_interface_t *interface, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)interface->vector.ptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)interface->vector.ptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = &state->interface[src_node].vector;
	dst_vector = &state->interface[dst_node].vector;

	cublasGetVector(src_vector->nx, src_vector->elemsize,
		(uint8_t *)src_vector->ptr, 1,
		(uint8_t *)dst_vector->ptr, 1);

	TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);
}

static void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = &state->interface[src_node].vector;
	dst_vector = &state->interface[dst_node].vector;

	cublasSetVector(src_vector->nx, src_vector->elemsize,
		(uint8_t *)src_vector->ptr, 1,
		(uint8_t *)dst_vector->ptr, 1);

	TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);
}
#endif // USE_CUDA

static void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	uint32_t nx = state->interface[dst_node].vector.nx;
	size_t elemsize = state->interface[dst_node].vector.elemsize;

	uintptr_t ptr_src = state->interface[src_node].vector.ptr;
	uintptr_t ptr_dst = state->interface[dst_node].vector.ptr;

	memcpy((void *)ptr_dst, (void *)ptr_src, nx*elemsize);

	TRACE_DATA_COPY(src_node, dst_node, nx*elemsize);
}

int do_copy_vector_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
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
					/* only the proper CUBLAS thread can initiate this directly ! */
					copy_cublas_to_ram(state, src_node, dst_node);
				}
				else
				{
					/* put a request to the corresponding GPU */
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

