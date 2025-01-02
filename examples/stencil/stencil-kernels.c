/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "stencil.h"

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
double *last_tick;

/* Achieved iterations */
static int achieved_iter;

/* Record how many updates each worker performed */
unsigned update_per_worker[STARPU_NMAXWORKERS];

static void record_who_runs_what(struct block_description *block)
{
	double now, now2, diff, delta = get_ticks() * 1000;
	int workerid = starpu_worker_get_id_check();

	now = starpu_timing_now();
	now2 = now - start;
	diff = now2 - last_tick[block->bz];
	while (diff >= delta)
	{
		last_tick[block->bz] += delta;
		diff = now2 - last_tick[block->bz];
		if (who_runs_what_index[block->bz] < who_runs_what_len)
			who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = -1;
	}

	if (who_runs_what_index[block->bz] < who_runs_what_len)
		who_runs_what[block->bz + (who_runs_what_index[block->bz]++) * get_nbz()] = global_workerid(workerid);
}

static void check_load(struct starpu_block_interface *block, struct starpu_block_interface *boundary)
{
	/* Sanity checks */
	STARPU_ASSERT(block->nx == boundary->nx);
	STARPU_ASSERT(block->ny == boundary->ny);
	STARPU_ASSERT(boundary->nz == K);

	/* NB: this is not fully guaranteed ... but it's *very* likely and that
	 * makes our life much simpler */
	STARPU_ASSERT(block->ldy == boundary->ldy);
	STARPU_ASSERT(block->ldz == boundary->ldz);
}

/*
 * Load a neighbour's boundary into block, CPU version
 */
static void load_subblock_from_buffer_cpu(void *_block,
					void *_boundary,
					unsigned firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

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
static void load_subblock_from_buffer_cuda(void *_block,
					void *_boundary,
					unsigned firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpyAsync(&block_data[offset], boundary_data, boundary_size, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
}

/*
 * cl_update (CUDA version)
 */
static void update_func_cuda(void *descr[], void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		FPRINTF(stderr,"!!! DO update_func_cuda z %u CUDA%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_cuda z %u CUDA%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_MASTER_SLAVE)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!           RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	unsigned block_size_z = get_block_size(block->bz);
	unsigned i;
	update_per_worker[workerid]++;

	record_who_runs_what(block);

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
		struct starpu_block_interface *oldb = descr[i%2], *newb = descr[(i+1)%2];
		TYPE *old = (void*) oldb->ptr, *newer = (void*) newb->ptr;

		/* Shadow data */
		cuda_shadow_host(block->bz, old, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);

		/* And perform actual computation */
#ifdef LIFE
		cuda_life_update_host(block->bz, old, newer, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);
#else
		cudaMemcpyAsync(newer, old, oldb->nx * oldb->ny * oldb->nz * sizeof(*newer), cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
#endif /* LIFE */
	}
}
#endif /* STARPU_USE_CUDA */

/*
 * Load a neighbour's boundary into block, OpenCL version
 */
#ifdef STARPU_USE_OPENCL
static void load_subblock_from_buffer_opencl(struct starpu_block_interface *block,
					struct starpu_block_interface *boundary,
					unsigned firstz)
{
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	cl_mem block_data = (cl_mem)block->dev_handle;
	cl_mem boundary_data = (cl_mem)boundary->dev_handle;

	cl_command_queue cq;
	starpu_opencl_get_current_queue(&cq);
	cl_int ret = clEnqueueCopyBuffer(cq, boundary_data, block_data, 0, offset, boundary_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(ret);
}

/*
 * cl_update (OpenCL version)
 */
static void update_func_opencl(void *descr[], void *arg)
{
	struct block_description *block = arg;
	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		FPRINTF(stderr,"!!! DO update_func_opencl z %u OPENCL%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_opencl z %u OPENCL%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_MASTER_SLAVE)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!           RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	unsigned block_size_z = get_block_size(block->bz);
	unsigned i;
	update_per_worker[workerid]++;

	record_who_runs_what(block);

	cl_command_queue cq;
	starpu_opencl_get_current_queue(&cq);

	/*
	 *	Load neighbours' boundaries : TOP
	 */

	/* The offset along the z axis is (block_size_z + K) */
	load_subblock_from_buffer_opencl(descr[0], descr[2], block_size_z+K);
	load_subblock_from_buffer_opencl(descr[1], descr[3], block_size_z+K);

	/*
	 *	Load neighbours' boundaries : BOTTOM
	 */
	load_subblock_from_buffer_opencl(descr[0], descr[4], 0);
	load_subblock_from_buffer_opencl(descr[1], descr[5], 0);

	/*
	 *	Stencils ... do the actual work here :) TODO
	 */

	for (i=1; i<=K; i++)
	{
		struct starpu_block_interface *oldb = descr[i%2], *newb = descr[(i+1)%2];
		TYPE *old = (void*) oldb->dev_handle, *newer = (void*) newb->dev_handle;

		/* Shadow data */
		opencl_shadow_host(block->bz, old, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);

		/* And perform actual computation */
#ifdef LIFE
		opencl_life_update_host(block->bz, old, newer, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);
#else
		cl_event event;
		cl_int ret = clEnqueueCopyBuffer(cq, old, newer, 0, 0, oldb->nx * oldb->ny * oldb->nz * sizeof(*newer), 0, NULL, &event);
		if (ret != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(ret);

#endif /* LIFE */
	}
}
#endif /* STARPU_USE_OPENCL */

/*
 * cl_update (CPU version)
 */
void update_func_cpu(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		FPRINTF(stderr,"!!! DO update_func_cpu z %u worker%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_cpu z %u worker%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_MASTER_SLAVE)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!            RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	unsigned block_size_z = get_block_size(block->bz);
	unsigned i;
	update_per_worker[workerid]++;

	record_who_runs_what(block);

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
		struct starpu_block_interface *oldb = (struct starpu_block_interface *) descr[i%2], *newb = (struct starpu_block_interface *) descr[(i+1)%2];
		TYPE *old = (TYPE*) oldb->ptr, *newer = (TYPE*) newb->ptr;

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
		life_update(block->bz, old, newer, oldb->nx, oldb->ny, oldb->nz, oldb->ldy, oldb->ldz, i);
#else
		memcpy(newer, old, oldb->nx * oldb->ny * oldb->nz * sizeof(*newer));
#endif /* LIFE */
	}
}

/* Performance model and codelet structure */
static struct starpu_perfmodel cl_update_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "cl_update"
};

struct starpu_codelet cl_update =
{
	.cpu_funcs = {update_func_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {update_func_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {update_func_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.model = &cl_update_model,
	.nbuffers = 6,
	.modes = {STARPU_RW, STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R}
};

/*
 * Save the block internal boundaries to give them to our neighbours.
 */

/* CPU version */
static void load_subblock_into_buffer_cpu(void *_block,
					void *_boundary,
					unsigned firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	memcpy(boundary_data, &block_data[offset], boundary_size);
}

/* CUDA version */
#ifdef STARPU_USE_CUDA
static void load_subblock_into_buffer_cuda(void *_block,
					void *_boundary,
					unsigned firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpyAsync(boundary_data, &block_data[offset], boundary_size, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
}
#endif /* STARPU_USE_CUDA */

/* OPENCL version */
#ifdef STARPU_USE_OPENCL
static void load_subblock_into_buffer_opencl(struct starpu_block_interface *block,
					struct starpu_block_interface *boundary,
					unsigned firstz)
{
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	unsigned offset = firstz*block->ldz;
	cl_mem block_data = (cl_mem)block->dev_handle;
	cl_mem boundary_data = (cl_mem)boundary->dev_handle;

	cl_command_queue cq;
	starpu_opencl_get_current_queue(&cq);

	cl_int ret = clEnqueueCopyBuffer(cq, block_data, boundary_data, offset, 0, boundary_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(ret);
}
#endif /* STARPU_USE_OPENCL */

/* Record how many top/bottom saves each worker performed */
unsigned top_per_worker[STARPU_NMAXWORKERS];
unsigned bottom_per_worker[STARPU_NMAXWORKERS];

/* top save, CPU version */
void dummy_func_top_cpu(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Bottom block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	unsigned block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cpu(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cpu(descr[1], descr[3], block_size_z);
}

/* bottom save, CPU version */
void dummy_func_bottom_cpu(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	(void) block;
	int workerid = starpu_worker_get_id_check();
	bottom_per_worker[workerid]++;

	DEBUG("DO SAVE Top block %d\n", block->bz);

	load_subblock_into_buffer_cpu(descr[0], descr[2], K);
	load_subblock_into_buffer_cpu(descr[1], descr[3], K);
}

/* top save, CUDA version */
#ifdef STARPU_USE_CUDA
static void dummy_func_top_cuda(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Top block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	unsigned block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cuda(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cuda(descr[1], descr[3], block_size_z);
}

/* bottom save, CUDA version */
static void dummy_func_bottom_cuda(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	(void) block;
	int workerid = starpu_worker_get_id_check();
	bottom_per_worker[workerid]++;

	DEBUG("DO SAVE Bottom block %d on CUDA\n", block->bz);

	load_subblock_into_buffer_cuda(descr[0], descr[2], K);
	load_subblock_into_buffer_cuda(descr[1], descr[3], K);
}
#endif /* STARPU_USE_CUDA */

/* top save, OpenCL version */
#ifdef STARPU_USE_OPENCL
static void dummy_func_top_opencl(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	(void) block;
	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Top block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	unsigned block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_opencl(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_opencl(descr[1], descr[3], block_size_z);
}

/* bottom save, OPENCL version */
static void dummy_func_bottom_opencl(void *descr[], void *arg)
{
	struct block_description *block = (struct block_description *) arg;
	(void) block;
	int workerid = starpu_worker_get_id_check();
	bottom_per_worker[workerid]++;

	DEBUG("DO SAVE Bottom block %d on OPENCL\n", block->bz);

	load_subblock_into_buffer_opencl(descr[0], descr[2], K);
	load_subblock_into_buffer_opencl(descr[1], descr[3], K);
}
#endif /* STARPU_USE_OPENCL */

/* Performance models and codelet for save */
static struct starpu_perfmodel save_cl_bottom_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "save_cl_bottom"
};

static struct starpu_perfmodel save_cl_top_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "save_cl_top"
};

struct starpu_codelet save_cl_bottom =
{
	.cpu_funcs = {dummy_func_bottom_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dummy_func_bottom_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {dummy_func_bottom_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.model = &save_cl_bottom_model,
	.nbuffers = 4,
	.modes = {STARPU_R, STARPU_R, STARPU_W, STARPU_W}
};

struct starpu_codelet save_cl_top =
{
	.cpu_funcs = {dummy_func_top_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dummy_func_top_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {dummy_func_top_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.model = &save_cl_top_model,
	.nbuffers = 4,
	.modes = {STARPU_R, STARPU_R, STARPU_W, STARPU_W}
};
