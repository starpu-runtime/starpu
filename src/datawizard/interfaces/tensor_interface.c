/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#ifdef BUILDING_STARPU
#include <datawizard/memory_nodes.h>
#endif

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);
static int map_tensor(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int unmap_tensor(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int update_map_tensor(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

static const struct starpu_data_copy_methods tensor_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};


static void register_tensor_handle(starpu_data_handle_t handle, int home_node, void *data_interface);
static void *tensor_to_pointer(void *data_interface, unsigned node);
static starpu_ssize_t allocate_tensor_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_tensor_buffer_on_node(void *data_interface, unsigned node);
static size_t tensor_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_tensor_interface_crc32(starpu_data_handle_t handle);
static int tensor_compare(void *data_interface_a, void *data_interface_b);
static void display_tensor_interface(starpu_data_handle_t handle, FILE *f);
static int pack_tensor_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int peek_tensor_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static int unpack_tensor_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);

struct starpu_data_interface_ops starpu_interface_tensor_ops =
{
	.register_data_handle = register_tensor_handle,
	.allocate_data_on_node = allocate_tensor_buffer_on_node,
	.to_pointer = tensor_to_pointer,
	.free_data_on_node = free_tensor_buffer_on_node,
	.map_data = map_tensor,
	.unmap_data = unmap_tensor,
	.update_map = update_map_tensor,
	.copy_methods = &tensor_copy_data_methods_s,
	.get_size = tensor_interface_get_size,
	.footprint = footprint_tensor_interface_crc32,
	.compare = tensor_compare,
	.interfaceid = STARPU_TENSOR_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_tensor_interface),
	.display = display_tensor_interface,
	.pack_data = pack_tensor_handle,
	.peek_data = peek_tensor_handle,
	.unpack_data = unpack_tensor_handle,
	.describe = describe,
	.name = "STARPU_TENSOR_INTERFACE",
	.pack_meta = NULL,
	.unpack_meta = NULL,
	.free_meta = NULL
};

static void *tensor_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_tensor_interface *tensor_interface = data_interface;

	return (void*) tensor_interface->ptr;
}

static void register_tensor_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *) data_interface;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_tensor_interface *local_interface = (struct starpu_tensor_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = tensor_interface->ptr;
			local_interface->dev_handle = tensor_interface->dev_handle;
			local_interface->offset = tensor_interface->offset;
			local_interface->ldy  = tensor_interface->ldy;
			local_interface->ldz  = tensor_interface->ldz;
			local_interface->ldt  = tensor_interface->ldt;
		}
		else
		{
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
			local_interface->ldy  = 0;
			local_interface->ldz  = 0;
			local_interface->ldt  = 0;
		}

		local_interface->id = tensor_interface->id;
		local_interface->nx = tensor_interface->nx;
		local_interface->ny = tensor_interface->ny;
		local_interface->nz = tensor_interface->nz;
		local_interface->nt = tensor_interface->nt;
		local_interface->elemsize = tensor_interface->elemsize;
	}
}

/* declare a new data with the BLAS interface */
void starpu_tensor_data_register(starpu_data_handle_t *handleptr, int home_node,
				uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t ldt, uint32_t nx,
				uint32_t ny, uint32_t nz, uint32_t nt, size_t elemsize)
{
	STARPU_ASSERT_MSG(ldy >= nx, "ldy = %u should not be less than nx = %u.", ldy, nx);
	STARPU_ASSERT_MSG(ldz/ldy >= ny, "ldz/ldy = %u/%u = %u should not be less than ny = %u.", ldz, ldy, ldz/ldy, ny);
	STARPU_ASSERT_MSG(ldt/ldz >= nz, "ldt/ldz = %u/%u = %u should not be less than nz = %u.", ldt, ldz, ldt/ldz, nz);
	struct starpu_tensor_interface tensor_interface =
	{
		.id = STARPU_TENSOR_INTERFACE_ID,
		.ptr = ptr,
		.dev_handle = ptr,
		.offset = 0,
		.ldy = ldy,
		.ldz = ldz,
		.ldt = ldt,
		.nx = nx,
		.ny = ny,
		.nz = nz,
		.nt = nt,
		.elemsize = elemsize
	};
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (nx && ny && nz && nt && elemsize)
		{
			STARPU_ASSERT_ACCESSIBLE(ptr);
			STARPU_ASSERT_ACCESSIBLE(ptr + (nt-1)*ldt*elemsize + (nz-1)*ldz*elemsize + (ny-1)*ldy*elemsize + nx*elemsize - 1);
		}
	}
#endif

	starpu_data_register(handleptr, home_node, &tensor_interface, &starpu_interface_tensor_ops);
}

void starpu_tensor_ptr_register(starpu_data_handle_t handle, unsigned node,
				  uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ldy, uint32_t ldz, uint32_t ldt)
{
	struct starpu_tensor_interface *tensor_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	tensor_interface->ptr = ptr;
	tensor_interface->dev_handle = dev_handle;
	tensor_interface->offset = offset;
	tensor_interface->ldy = ldy;
	tensor_interface->ldz = ldz;
	tensor_interface->ldt = ldt;
}

static uint32_t footprint_tensor_interface_crc32(starpu_data_handle_t handle)
{
	uint32_t hash;

	hash = starpu_hash_crc32c_be(starpu_tensor_get_nx(handle), 0);
	hash = starpu_hash_crc32c_be(starpu_tensor_get_ny(handle), hash);
	hash = starpu_hash_crc32c_be(starpu_tensor_get_nz(handle), hash);
	hash = starpu_hash_crc32c_be(starpu_tensor_get_nt(handle), hash);

	return hash;
}

static int tensor_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_tensor_interface *tensor_a = (struct starpu_tensor_interface *) data_interface_a;
	struct starpu_tensor_interface *tensor_b = (struct starpu_tensor_interface *) data_interface_b;

	/* Two tensors are considered compatible if they have the same size */
	return (tensor_a->nx == tensor_b->nx)
		&& (tensor_a->ny == tensor_b->ny)
		&& (tensor_a->nz == tensor_b->nz)
		&& (tensor_a->nt == tensor_b->nt)
		&& (tensor_a->elemsize == tensor_b->elemsize);
}

static void display_tensor_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_tensor_interface *tensor_interface;

	tensor_interface = (struct starpu_tensor_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t%u\t%u\t%u\t", tensor_interface->nx, tensor_interface->ny, tensor_interface->nz, tensor_interface->nt);
}

#define IS_CONTIGUOUS_MATRIX(nx, ny, ldy) ((nx) == (ldy))
#define IS_CONTIGUOUS_BLOCK(nx, ny, nz, ldy, ldz) ((nx) * (ny) == (ldz))
#define IS_CONTIGUOUS_TENSOR(nx, ny, nz, nt, ldy, ldz, ldt) ((nx) * (ny) * (nz) == (ldt))

static int pack_tensor_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t ldy = tensor_interface->ldy;
	uint32_t ldz = tensor_interface->ldz;
	uint32_t ldt = tensor_interface->ldt;
	uint32_t nx = tensor_interface->nx;
	uint32_t ny = tensor_interface->ny;
	uint32_t nz = tensor_interface->nz;
	uint32_t nt = tensor_interface->nt;
	size_t elemsize = tensor_interface->elemsize;

	*count = nx*ny*nz*nt*elemsize;

	if (ptr != NULL)
	{
		uint32_t t, z, y;
		char *block = (void *)tensor_interface->ptr;

		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);

		char *cur = *ptr;
		if (IS_CONTIGUOUS_TENSOR(nx, ny, nz, nt, ldy, ldz, ldt))
			memcpy(cur, block, nx * ny * nz * nt * elemsize);
		else
		{
			char *block_t = block;
			for(t=0 ; t<nt ; t++)
			{
				if (IS_CONTIGUOUS_BLOCK(nx, ny, nz, ldy, ldz))
				{
					memcpy(cur, block_t, nx * ny * nz * elemsize);
					cur += nx*ny*nz*elemsize;
				}
				else
				{
					char *block_z = block_t;
					for(z=0 ; z<nz ; z++)
					{
						if (IS_CONTIGUOUS_MATRIX(nx, ny, ldy))
						{
							memcpy(cur, block_z, nx * ny * elemsize);
							cur += nx*ny*elemsize;
						}
						else
						{
							char *block_y = block_z;
							for(y=0 ; y<ny ; y++)
							{
								memcpy(cur, block_y, nx*elemsize);
								cur += nx*elemsize;
								block_y += ldy * elemsize;
							}
						}
						block_z += ldz * elemsize;
					}
				}
				block_t += ldt * elemsize;
			}
		}
	}

	return 0;
}

static int peek_tensor_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t ldy = tensor_interface->ldy;
	uint32_t ldz = tensor_interface->ldz;
	uint32_t ldt = tensor_interface->ldt;
	uint32_t nx = tensor_interface->nx;
	uint32_t ny = tensor_interface->ny;
	uint32_t nz = tensor_interface->nz;
	uint32_t nt = tensor_interface->nt;
	size_t elemsize = tensor_interface->elemsize;

	STARPU_ASSERT(count == elemsize * nx * ny * nz * nt);

	uint32_t t, z, y;
	char *cur = ptr;
	char *block = (void *)tensor_interface->ptr;

	if (IS_CONTIGUOUS_TENSOR(nx, ny, nz, nt, ldy, ldz, ldt))
		memcpy(block, cur, nx * ny * nz * nt * elemsize);
	else
	{
		char *block_t = block;
		for(t=0 ; t<nt ; t++)
		{
			if (IS_CONTIGUOUS_BLOCK(nx, ny, nz, ldy, ldz))
			{
				memcpy(block_t, cur, nx * ny * nz * elemsize);
				cur += nx*ny*nz*elemsize;
			}
			else
			{
				char *block_z = block_t;
				for(z=0 ; z<nz ; z++)
				{
					if (IS_CONTIGUOUS_MATRIX(nx, ny, ldy))
					{
						memcpy(block_z, cur, nx * ny * elemsize);
						cur += nx*ny*elemsize;
					}
					else
					{
						char *block_y = block_z;
						for(y=0 ; y<ny ; y++)
						{
							memcpy(block_y, cur, nx*elemsize);
							cur += nx*elemsize;
							block_y += ldy * elemsize;
						}
					}
					block_z += ldz * elemsize;
				}
			}
			block_t += ldt * elemsize;
		}
	}

	return 0;
}

static int unpack_tensor_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	peek_tensor_handle(handle, node, ptr, count);
	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}


static size_t tensor_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_tensor_interface *tensor_interface;

	tensor_interface = (struct starpu_tensor_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	size = tensor_interface->nx*tensor_interface->ny*tensor_interface->nz*tensor_interface->nt*tensor_interface->elemsize;

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_tensor_get_nx(starpu_data_handle_t handle)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->nx;
}

uint32_t starpu_tensor_get_ny(starpu_data_handle_t handle)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->ny;
}

uint32_t starpu_tensor_get_nz(starpu_data_handle_t handle)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->nz;
}

uint32_t starpu_tensor_get_nt(starpu_data_handle_t handle)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->nt;
}

uint32_t starpu_tensor_get_local_ldy(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->ldy;
}

uint32_t starpu_tensor_get_local_ldz(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->ldz;
}

uint32_t starpu_tensor_get_local_ldt(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->ldt;
}

uintptr_t starpu_tensor_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->ptr;
}

size_t starpu_tensor_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(tensor_interface->id == STARPU_TENSOR_INTERFACE_ID, "Error. The given data is not a block.");
#endif

	return tensor_interface->elemsize;
}


/* memory allocation/deallocation primitives for the BLOCK interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_tensor_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr = 0, handle;

	struct starpu_tensor_interface *dst_block = (struct starpu_tensor_interface *) data_interface_;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	uint32_t nt = dst_block->nt;
	size_t elemsize = dst_block->elemsize;

	starpu_ssize_t allocated_memory;

	handle = starpu_malloc_on_node(dst_node, nx*ny*nz*nt*elemsize);

	if (!handle)
		return -ENOMEM;

	if (!starpu_node_needs_offset(dst_node))
		addr = handle;

	allocated_memory = nx*ny*nz*nt*elemsize;

	/* update the data properly in consequence */
	dst_block->ptr = addr;
	dst_block->dev_handle = handle;
	dst_block->offset = 0;
	dst_block->ldy = nx;
	dst_block->ldz = nx*ny;
	dst_block->ldt = nx*ny*nz;

	return allocated_memory;
}

static void free_tensor_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_tensor_interface *tensor_interface = (struct starpu_tensor_interface *) data_interface;
	uint32_t nx = tensor_interface->nx;
	uint32_t ny = tensor_interface->ny;
	uint32_t nz = tensor_interface->nz;
	uint32_t nt = tensor_interface->nt;
	size_t elemsize = tensor_interface->elemsize;

	starpu_free_on_node(node, tensor_interface->dev_handle, nx*ny*nz*nt*elemsize);
	tensor_interface->ptr = 0;
	tensor_interface->dev_handle = 0;
}

static int map_tensor(void *src_interface, unsigned src_node,
		      void *dst_interface, unsigned dst_node)
{
	struct starpu_tensor_interface *src_tensor = src_interface;
	struct starpu_tensor_interface *dst_tensor = dst_interface;
	int ret;
	uintptr_t mapped;

	/* map area ldt*(nt-1) + ldz*(nz-1) + ldy*(ny-1) + nx*/
	mapped = starpu_interface_map(src_tensor->dev_handle, src_tensor->offset, src_node, dst_node, (src_tensor->ldt*(src_tensor->nt-1)+src_tensor->ldz*(src_tensor->nz-1)+src_tensor->ldy*(src_tensor->ny-1)+src_tensor->nx)*src_tensor->elemsize, &ret);
	if (mapped)
	{
		dst_tensor->dev_handle = mapped;
		dst_tensor->offset = 0;
		if (!starpu_node_needs_offset(dst_node))
			dst_tensor->ptr = mapped;
		dst_tensor->ldy = src_tensor->ldy;
		dst_tensor->ldz = src_tensor->ldz;
		dst_tensor->ldt = src_tensor->ldt;
		return 0;
	}
	return ret;
}

static int unmap_tensor(void *src_interface, unsigned src_node,
			void *dst_interface, unsigned dst_node)
{
	struct starpu_tensor_interface *src_tensor = src_interface;
	struct starpu_tensor_interface *dst_tensor = dst_interface;

	int ret = starpu_interface_unmap(src_tensor->dev_handle, src_tensor->offset, src_node, dst_tensor->dev_handle, dst_node, (src_tensor->ldt*(src_tensor->nt-1)+src_tensor->ldz*(src_tensor->nz-1)+src_tensor->ldy*(src_tensor->ny-1)+src_tensor->nx)*src_tensor->elemsize);
	dst_tensor->dev_handle = 0;

	return ret;
}

static int update_map_tensor(void *src_interface, unsigned src_node,
			     void *dst_interface, unsigned dst_node)
{
	struct starpu_tensor_interface *src_tensor = src_interface;
	struct starpu_tensor_interface *dst_tensor = dst_interface;

	return starpu_interface_update_map(src_tensor->dev_handle, src_tensor->offset, src_node, dst_tensor->dev_handle, dst_tensor->offset, dst_node, (src_tensor->ldt*(src_tensor->nt-1)+src_tensor->ldz*(src_tensor->nz-1)+src_tensor->ldy*(src_tensor->ny-1)+src_tensor->nx)*src_tensor->elemsize);
}

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_tensor_interface *src_block = (struct starpu_tensor_interface *) src_interface;
	struct starpu_tensor_interface *dst_block = (struct starpu_tensor_interface *) dst_interface;
	int ret = 0;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	uint32_t nt = dst_block->nt;
	size_t elemsize = dst_block->elemsize;

	uint32_t ldy_src = src_block->ldy;
	uint32_t ldz_src = src_block->ldz;
	uint32_t ldt_src = src_block->ldt;
	uint32_t ldy_dst = dst_block->ldy;
	uint32_t ldz_dst = dst_block->ldz;
	uint32_t ldt_dst = dst_block->ldt;

	if (starpu_interface_copy4d(src_block->dev_handle, src_block->offset, src_node,
				    dst_block->dev_handle, dst_block->offset, dst_node,
				    nx * elemsize,
				    ny, ldy_src * elemsize, ldy_dst * elemsize,
				    nz, ldz_src * elemsize, ldz_dst * elemsize,
				    nt, ldt_src * elemsize, ldt_dst * elemsize,
				    async_data))
		ret = -EAGAIN;

	starpu_interface_data_copy(src_node, dst_node, nx*ny*nz*nt*elemsize);

	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_tensor_interface *block = (struct starpu_tensor_interface *) data_interface;
	return snprintf(buf, size, "T%ux%ux%ux%ux%u",
			(unsigned) block->nx,
			(unsigned) block->ny,
			(unsigned) block->nz,
			(unsigned) block->nt,
			(unsigned) block->elemsize);
}
