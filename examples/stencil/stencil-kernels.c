/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2009-2010 (see AUTHORS file)
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

#include "stencil.h"
#include <sys/time.h>

/* Computation Kernels */

/*
 * There are three codeletets:
 *
 * - cl_update, which takes a block and the boundaries of its neighbours, loads
 *   the boundaries into the block and perform some update loops:
 *
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy====>#N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy    | |                  |                                            
 *   +-------------+ +------------------+ |                  |                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |                  | +----------------+ +----------------------+
 *                                        |                  | | #N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy   <====#N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * - save_cl_top, which take a block and its top boundary, and saves the top of
 *   the block into the boundary (to be given as bottom of the neighbour above
 *   this block).
 *
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy | | #N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy   <====                 |                                            
 *   +-------------+ +------------------+ |..................|                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |                  | +----------------+ +----------------------+
 *                                        |                  | | #N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy    | | #N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * - save_cl_bottom, same for the bottom
 *     comp. buffer      save. buffers        comp. buffer        save. buffers        comp. buffer
 *   |     ...     |                                                                                      
 *   |             | +------------------+ +------------------+                                            
 *   |     #N+1    | | #N+1 bottom copy | | #N+1 bottom copy |                                            
 *   +-------------+ +------------------+ +------------------+                                            
 *   | #N top copy | |   #N top copy    | |                  |                                            
 *   +-------------+ +------------------+ |                  |                                            
 *                                        | #N               |                                            
 *                                                 ...                                                    
 *                                        |..................| +----------------+ +----------------------+
 *                                        |                 ====>#N bottom copy | | block #N bottom copy |
 * ^                                      +------------------+ +----------------+ +----------------------+
 * |                                      | #N-1 top copy    | | #N-1 top copy  | |  block #N-1          |
 * |                                      +------------------+ +----------------+ |                      |
 * Z                                                                                     ...
 *
 * The idea is that the computation buffers thus don't have to move, only their
 * boundaries are copied to buffers that do move (be it CPU/GPU, GPU/GPU or via
 * MPI)
 *
 * For each of the buffers above, there are two (0/1) buffers to make new/old switch costless.
 */

#if 0
# define DEBUG(fmt, ...) fprintf(stderr,fmt,##__VA_ARGS__)
#else
# define DEBUG(fmt, ...) (void) 0
#endif

/* Record which GPU ran which block, for nice pictures */
int who_runs_what_len;
int *who_runs_what;
int *who_runs_what_index;
struct timeval *last_tick;

/* Record how many updates each worker performed */
unsigned update_per_worker[STARPU_NMAXWORKERS];

/*
 * Load a neighbour's boundary into block, CPU version
 */
static void load_subblock_from_buffer_cpu(starpu_block_interface_t *block,
					starpu_block_interface_t *boundary,
					unsigned firstz)
{
	/* Sanity checks */
	STARPU_ASSERT(block->nx == boundary->nx);
	STARPU_ASSERT(block->ny == boundary->ny);
	STARPU_ASSERT(boundary->nz == K);

	/* NB: this is not fully garanteed ... but it's *very* likely and that
	 * makes our life much simpler */
	STARPU_ASSERT(block->ldy == boundary->ldy);
	STARPU_ASSERT(block->ldz == boundary->ldz);
	
	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	memcpy(&block_data[offset], boundary_data, boundary_size);
}

/*
 * Load a neighbour's boundary into block, CUDA version
 */
#ifdef STARPU_USE_CUDA
static void load_subblock_from_buffer_cuda(starpu_block_interface_t *block,
					starpu_block_interface_t *boundary,
					unsigned firstz)
{
	/* Sanity checks */
	STARPU_ASSERT(block->nx == boundary->nx);
	STARPU_ASSERT(block->ny == boundary->ny);
	STARPU_ASSERT(boundary->nz == K);

	/* NB: this is not fully garanteed ... but it's *very* likely and that
	 * makes our life much simpler */
	STARPU_ASSERT(block->ldy == boundary->ldy);
	STARPU_ASSERT(block->ldz == boundary->ldz);
	
	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpy(&block_data[offset], boundary_data, boundary_size, cudaMemcpyDeviceToDevice);
}

/*
 * cl_update (CUDA version)
 */
static void update_func_cuda(void *descr[], void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	DEBUG( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
fprintf(stderr,"!!! DO update_func_cuda z %d CUDA%d !!!\n", block->bz, workerid);
	else
	DEBUG( "!!! DO update_func_cuda z %d CUDA%d !!!\n", block->bz, workerid);
#ifdef STARPU_USE_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG( "!!!           RANK %d              !!!\n", rank);
#endif
	DEBUG( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	unsigned block_size_z = get_block_size(block->bz);
	unsigned i;
	update_per_worker[workerid]++;

	struct timeval tv, tv2, diff, delta = {.tv_sec = 0, .tv_usec = get_ticks()*1000};
	gettimeofday(&tv, NULL);
	timersub(&tv, &start, &tv2);
	timersub(&tv2, &last_tick[block->bz], &diff);
	while (timercmp(&diff, &delta, >=)) {
		timeradd(&last_tick[block->bz], &delta, &last_tick[block->bz]);
		timersub(&tv2, &last_tick[block->bz], &diff);
		if (who_runs_what_index[block->bz] < who_runs_what_len)
			who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = -1;
	}

	if (who_runs_what_index[block->bz] < who_runs_what_len)
		who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = global_workerid(workerid);

	/*
	 *	Load neighbours' boundaries : TOP
	 */

	/* The offset along the z axis is (block_size_z + K) */
	load_subblock_from_buffer_cuda(descr[0], descr[2], block_size_z+K);
	load_subblock_from_buffer_cuda(descr[1], descr[3], block_size_z+K);

	/*
	 *	Load neighbours' boundaries : BOTTOM
	 */
	load_subblock_from_buffer_cuda(descr[0], descr[4], 0);
	load_subblock_from_buffer_cuda(descr[1], descr[5], 0);

	/*
	 *	Stencils ... do the actual work here :) TODO
	 */

	for (i=1; i<=K; i++)
	{
		starpu_block_interface_t *oldb = descr[i%2], *newb = descr[(i+1)%2];
		TYPE *old = (void*) oldb->ptr, *new = (void*) newb->ptr;

		/* Shadow data */
		cuda_shadow_host(block->bz, old, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);

		/* And perform actual computation */
#ifdef LIFE
		cuda_life_update_host(block->bz, old, new, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);
#else
		cudaMemcpy(new, old, oldb->nx * oldb->ny * oldb->nz * sizeof(*new), cudaMemcpyDeviceToDevice);
#endif /* LIFE */
	}

	cudaError_t cures;
	if ((cures = cudaThreadSynchronize()) != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(cures);

}
#endif /* STARPU_USE_CUDA */

/*
 * cl_update (CPU version)
 */
static void update_func_cpu(void *descr[], void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	DEBUG( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
fprintf(stderr,"!!! DO update_func_cpu z %d CPU%d !!!\n", block->bz, workerid);
	else
	DEBUG( "!!! DO update_func_cpu z %d CPU%d !!!\n", block->bz, workerid);
#ifdef STARPU_USE_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG( "!!!           RANK %d            !!!\n", rank);
#endif
	DEBUG( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	unsigned block_size_z = get_block_size(block->bz);
	unsigned i;
	update_per_worker[workerid]++;

	struct timeval tv, tv2, diff, delta = {.tv_sec = 0, .tv_usec = get_ticks() * 1000};
	gettimeofday(&tv, NULL);
	timersub(&tv, &start, &tv2);
	timersub(&tv2, &last_tick[block->bz], &diff);
	while (timercmp(&diff, &delta, >=)) {
		timeradd(&last_tick[block->bz], &delta, &last_tick[block->bz]);
		timersub(&tv2, &last_tick[block->bz], &diff);
		if (who_runs_what_index[block->bz] < who_runs_what_len)
			who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = -1;
	}

	if (who_runs_what_index[block->bz] < who_runs_what_len)
		who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = global_workerid(workerid);

	/*
	 *	Load neighbours' boundaries : TOP
	 */

	/* The offset along the z axis is (block_size_z + K) */
	load_subblock_from_buffer_cpu(descr[0], descr[2], block_size_z+K);
	load_subblock_from_buffer_cpu(descr[1], descr[3], block_size_z+K);

	/*
	 *	Load neighbours' boundaries : BOTTOM
	 */
	load_subblock_from_buffer_cpu(descr[0], descr[4], 0);
	load_subblock_from_buffer_cpu(descr[1], descr[5], 0);

	/*
	 *	Stencils ... do the actual work here :) TODO
	 */

	for (i=1; i<=K; i++)
	{
		starpu_block_interface_t *oldb = descr[i%2], *newb = descr[(i+1)%2];
		TYPE *old = (void*) oldb->ptr, *new = (void*) newb->ptr;

		/* Shadow data */
		unsigned ldy = oldb->ldy, ldz = oldb->ldz;
		unsigned nx = oldb->nx, ny = oldb->ny, nz = oldb->nz;
		unsigned x, y, z;
		unsigned stepx = 1;
		unsigned stepy = 1;
		unsigned stepz = 1;
		unsigned idx = 0;
		unsigned idy = 0;
		unsigned idz = 0;
		TYPE *ptr = old;

#		include "shadow.h"

		/* And perform actual computation */
#ifdef LIFE
		life_update(block->bz, old, new, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);
#else
		memcpy(new, old, oldb->nx * oldb->ny * oldb->nz * sizeof(*new));
#endif /* LIFE */
	}
}

/* Performance model and codelet structure */
static struct starpu_perfmodel_t cl_update_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "cl_update" 
};

starpu_codelet cl_update = {
	.where = 
#ifdef STARPU_USE_CUDA
		STARPU_CUDA|
#endif
		STARPU_CPU,
	.cpu_func = update_func_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = update_func_cuda,
#endif
	.model = &cl_update_model,
	.nbuffers = 6
};

/*
 * Save the block internal boundaries to give them to our neighbours.
 */

/* CPU version */
static void load_subblock_into_buffer_cpu(starpu_block_interface_t *block,
					starpu_block_interface_t *boundary,
					unsigned firstz)
{
	/* Sanity checks */
	STARPU_ASSERT(block->nx == boundary->nx);
	STARPU_ASSERT(block->ny == boundary->ny);
	STARPU_ASSERT(boundary->nz == K);

	/* NB: this is not fully garanteed ... but it's *very* likely and that
	 * makes our life much simpler */
	STARPU_ASSERT(block->ldy == boundary->ldy);
	STARPU_ASSERT(block->ldz == boundary->ldz);
	
	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	memcpy(boundary_data, &block_data[offset], boundary_size);
}

/* CUDA version */
#ifdef STARPU_USE_CUDA
static void load_subblock_into_buffer_cuda(starpu_block_interface_t *block,
					starpu_block_interface_t *boundary,
					unsigned firstz)
{
	/* Sanity checks */
	STARPU_ASSERT(block->nx == boundary->nx);
	STARPU_ASSERT(block->ny == boundary->ny);
	STARPU_ASSERT(boundary->nz == K);

	/* NB: this is not fully garanteed ... but it's *very* likely and that
	 * makes our life much simpler */
	STARPU_ASSERT(block->ldy == boundary->ldy);
	STARPU_ASSERT(block->ldz == boundary->ldz);
	
	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpy(boundary_data, &block_data[offset], boundary_size, cudaMemcpyDeviceToDevice);
}
#endif /* STARPU_USE_CUDA */

/* Record how many top/bottom saves each worker performed */
unsigned top_per_worker[STARPU_NMAXWORKERS];
unsigned bottom_per_worker[STARPU_NMAXWORKERS];

/* top save, CPU version */
static void dummy_func_top_cpu(void *descr[] __attribute__((unused)), void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	top_per_worker[workerid]++;

	DEBUG( "DO SAVE Bottom block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	unsigned block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cpu(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cpu(descr[1], descr[3], block_size_z);
}

/* bottom save, CPU version */
static void dummy_func_bottom_cpu(void *descr[] __attribute__((unused)), void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	bottom_per_worker[workerid]++;

	DEBUG( "DO SAVE Top block %d\n", block->bz);

	load_subblock_into_buffer_cpu(descr[0], descr[2], K);
	load_subblock_into_buffer_cpu(descr[1], descr[3], K);
}

/* top save, CUDA version */
#ifdef STARPU_USE_CUDA
static void dummy_func_top_cuda(void *descr[] __attribute__((unused)), void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	top_per_worker[workerid]++;

	DEBUG( "DO SAVE Top block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	unsigned block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cuda(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cuda(descr[1], descr[3], block_size_z);
	cudaThreadSynchronize();
}

/* bottom save, CUDA version */
static void dummy_func_bottom_cuda(void *descr[] __attribute__((unused)), void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id();
	bottom_per_worker[workerid]++;

	DEBUG( "DO SAVE Bottom block %d on CUDA\n", block->bz);

	load_subblock_into_buffer_cuda(descr[0], descr[2], K);
	load_subblock_into_buffer_cuda(descr[1], descr[3], K);
	cudaThreadSynchronize();
}
#endif /* STARPU_USE_CUDA */

/* Performance models and codelet for save */
static struct starpu_perfmodel_t save_cl_bottom_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "save_cl_bottom" 
};

static struct starpu_perfmodel_t save_cl_top_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "save_cl_top" 
};

starpu_codelet save_cl_bottom = {
	.where = 
#ifdef STARPU_USE_CUDA
		STARPU_CUDA|
#endif
		STARPU_CPU,
	.cpu_func = dummy_func_bottom_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = dummy_func_bottom_cuda,
#endif
	.model = &save_cl_bottom_model,
	.nbuffers = 4
};

starpu_codelet save_cl_top = {
	.where = 
#ifdef STARPU_USE_CUDA
		STARPU_CUDA|
#endif
		STARPU_CPU,
	.cpu_func = dummy_func_top_cpu,
#ifdef STARPU_USE_CUDA
	.cuda_func = dummy_func_top_cuda,
#endif
	.model = &save_cl_top_model,
	.nbuffers = 4
};
