/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __ALLOC_H__
#define __ALLOC_H__

#pragma GCC visibility push(hidden)

/** @file */

void _starpu_malloc_init(unsigned dst_node);
void _starpu_malloc_shutdown(unsigned dst_node);

int _starpu_malloc_flags_on_node(unsigned dst_node, void **A, size_t dim, int flags);
int _starpu_free_flags_on_node(unsigned dst_node, void *A, size_t dim, int flags);

/**
 * Returns whether when allocating data on \p dst_node, we will do pinning, i.e.
 * the allocation will be very expensive, and should thus be moved out from the
 * critical path
 */
int _starpu_malloc_willpin_on_node(unsigned dst_node);

/**
 * On CUDA which has very expensive malloc, for small sizes, allocate big
 * chunks divided in blocks, and we actually allocate segments of consecutive
 * blocks.
 *
 * We try to keep the list of chunks with increasing occupancy, so we can
 * quickly find free segments to allocate.
 */

#ifdef STARPU_USE_MAX_FPGA
// FIXME: Maxeler FPGAs want 192 byte alignment
#define CHUNK_SIZE (128*1024*192)
#define CHUNK_ALLOC_MAX (CHUNK_SIZE / 8)
#define CHUNK_ALLOC_MIN (128*192)
#else
/* Size of each chunk, 32MiB granularity brings 128 chunks to be allocated in
 * order to fill a 4GiB GPU. */
#define CHUNK_SIZE (32*1024*1024)

/* Maximum segment size we will allocate in chunks */
#define CHUNK_ALLOC_MAX (CHUNK_SIZE / 8)

/* Granularity of allocation, i.e. block size, StarPU will never allocate less
 * than this.
 * 16KiB (i.e. 64x64 float) granularity eats 2MiB RAM for managing a 4GiB GPU.
 */
#define CHUNK_ALLOC_MIN (16*1024)
#endif

/* Don't really deallocate chunks unless we have more than this many chunks
 * which are completely free. */
#define CHUNKS_NFREE 4

/* Number of blocks */
#define CHUNK_NBLOCKS (CHUNK_SIZE/CHUNK_ALLOC_MIN)

/* Linked list for available segments */
struct block
{
	int length;	/* Number of consecutive free blocks */
	int next;	/* next free segment */
};

/* One chunk */
LIST_TYPE(_starpu_chunk,
	uintptr_t base;

	/* Available number of blocks, for debugging */
	int available;

	/* Overestimation of the maximum size of available segments in this chunk */
	int available_max;

	/* Bitmap describing availability of the block */
	/* Block 0 is always empty, and is just the head of the free segments list */
	struct block bitmap[CHUNK_NBLOCKS+1];
)

#pragma GCC visibility pop

#endif
