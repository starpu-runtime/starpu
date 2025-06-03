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

#include "implicit-stencil.h"

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
size_t update_per_worker[STARPU_NMAXWORKERS];

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
					size_t firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
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
					size_t firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpyAsync(&block_data[offset], boundary_data, boundary_size, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
}

/*
 * cl_update (CUDA version)
 */
static void update_func_cuda(void *descr[], void *arg)
{
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);

	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		FPRINTF(stderr,"!!! DO update_func_cuda z %zu CUDA%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_cuda z %zu CUDA%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_SERVER_CLIENT)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!           RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	size_t block_size_z = get_block_size(block->bz);
	size_t i;
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
					size_t firstz)
{
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
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
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);

	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		FPRINTF(stderr,"!!! DO update_func_opencl z %zu OPENCL%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_opencl z %zu OPENCL%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_SERVER_CLIENT)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!           RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	size_t block_size_z = get_block_size(block->bz);
	size_t i;
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
	size_t zz;
	starpu_codelet_unpack_args(arg, &zz);
	struct block_description *block = get_block_description(zz);

	int workerid = starpu_worker_get_id_check();
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	if (block->bz == 0)
		DEBUG("!!! DO update_func_cpu z %zu worker%d !!!\n", block->bz, workerid);
	else
		DEBUG("!!! DO update_func_cpu z %zu worker%d !!!\n", block->bz, workerid);
#if defined(STARPU_USE_MPI) && !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MPI_SERVER_CLIENT)
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DEBUG("!!!            RANK %d              !!!\n", rank);
#endif
	DEBUG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

	size_t block_size_z = get_block_size(block->bz);
	size_t i;
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
		size_t ldy = oldb->ldy, ldz = oldb->ldz;
		size_t nx = oldb->nx, ny = oldb->ny, nz = oldb->nz;
		size_t x, y, z;
		size_t stepx = 1;
		size_t stepy = 1;
		size_t stepz = 1;
		size_t idx = 0;
		size_t idy = 0;
		size_t idz = 0;
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
					size_t firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	memcpy(boundary_data, &block_data[offset], boundary_size);
}

/* CUDA version */
#ifdef STARPU_USE_CUDA
static void load_subblock_into_buffer_cuda(void *_block,
					void *_boundary,
					size_t firstz)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *)_block;
	struct starpu_block_interface *boundary = (struct starpu_block_interface *)_boundary;
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
	TYPE *block_data = (TYPE *)block->ptr;
	TYPE *boundary_data = (TYPE *)boundary->ptr;
	cudaMemcpyAsync(boundary_data, &block_data[offset], boundary_size, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
}
#endif /* STARPU_USE_CUDA */

/* OPENCL version */
#ifdef STARPU_USE_OPENCL
static void load_subblock_into_buffer_opencl(struct starpu_block_interface *block,
					struct starpu_block_interface *boundary,
					size_t firstz)
{
	check_load(block, boundary);

	/* We do a contiguous memory transfer */
	size_t boundary_size = K*block->ldz*block->elemsize;

	size_t offset = firstz*block->ldz;
	cl_mem block_data = (cl_mem)block->dev_handle;
	cl_mem boundary_data = (cl_mem)boundary->dev_handle;

	cl_command_queue cq;
	starpu_opencl_get_current_queue(&cq);

	cl_int ret = clEnqueueCopyBuffer(cq, block_data, boundary_data, offset, 0, boundary_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(ret);
}
#endif /* STARPU_USE_OPENCL */

/* Record how many top/bottom saves each worker performed */
size_t top_per_worker[STARPU_NMAXWORKERS];
size_t bottom_per_worker[STARPU_NMAXWORKERS];

/* top save, CPU version */
void dummy_func_top_cpu(void *descr[], void *arg)
{
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);

	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Bottom block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	size_t block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cpu(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cpu(descr[1], descr[3], block_size_z);
}

/* bottom save, CPU version */
void dummy_func_bottom_cpu(void *descr[], void *arg)
{
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);
	STARPU_ASSERT(block);

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
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);

	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Top block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	size_t block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_cuda(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_cuda(descr[1], descr[3], block_size_z);
}

/* bottom save, CUDA version */
static void dummy_func_bottom_cuda(void *descr[], void *arg)
{
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);
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
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);

	int workerid = starpu_worker_get_id_check();
	top_per_worker[workerid]++;

	DEBUG("DO SAVE Top block %d\n", block->bz);

	/* The offset along the z axis is (block_size_z + K)- K */
	size_t block_size_z = get_block_size(block->bz);

	load_subblock_into_buffer_opencl(descr[0], descr[2], block_size_z);
	load_subblock_into_buffer_opencl(descr[1], descr[3], block_size_z);
}

/* bottom save, OPENCL version */
static void dummy_func_bottom_opencl(void *descr[], void *arg)
{
	size_t z;
	starpu_codelet_unpack_args(arg, &z);
	struct block_description *block = get_block_description(z);
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

/* Memset a block's buffers */
void memset_func(void *descr[], void *arg)
{
	(void)descr;
	size_t sizex, sizey, bz;
	starpu_codelet_unpack_args(arg, &sizex, &sizey, &bz);
	struct block_description *block = get_block_description(bz);
	size_t size_bz = get_block_size(bz);

	size_t x,y,z;
	for (x = 0; x < sizex + 2*K; x++)
	{
		for (y = 0; y < sizey + 2*K; y++)
		{
			/* Main blocks */
			for (z = 0; z < size_bz + 2*K; z++)
			{
				block->layers[0][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;
				block->layers[1][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;
			}
			for (z = 0; z < K; z++)
			{
				/* Boundary blocks : Top */
				block->boundaries[T][0][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;
				block->boundaries[T][1][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;

				/* Boundary blocks : Bottom */
				block->boundaries[B][0][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;
				block->boundaries[B][1][(x)+(y)*(sizex + 2*K)+(z)*(sizex+2*K)*(sizey+2*K)] = 0;
			}
		}
	}
	//memset(block->layers[0], 0, (sizex + 2*K)*(sizey + 2*K)*(size_bz + 2*K)*sizeof(block->layers[0]));
	//memset(block->layers[1], 0, (sizex + 2*K)*(sizey + 2*K)*(size_bz + 2*K)*sizeof(block->layers[1]));

	//memset(block->boundaries[T][0], 0, (sizex + 2*K)*(sizey + 2*K)*K*sizeof(block->boundaries[T][0]));
	//memset(block->boundaries[T][1], 0, (sizex + 2*K)*(sizey + 2*K)*K*sizeof(block->boundaries[T][1]));

	//memset(block->boundaries[B][0], 0, (sizex + 2*K)*(sizey + 2*K)*K*sizeof(block->boundaries[B][0]));
	//memset(block->boundaries[B][1], 0, (sizex + 2*K)*(sizey + 2*K)*K*sizeof(block->boundaries[B][1]));
}

static double memset_cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 0.000001;
}

static struct starpu_perfmodel memset_model =
{
	.type = STARPU_COMMON,
	.cost_function = memset_cost_function,
	.symbol = "memset"
};

struct starpu_codelet cl_memset =
{
	.cpu_funcs = {memset_func},
	.cpu_funcs_name = {"memset_func"},
	.model = &memset_model,
	.nbuffers = 6,
	.modes = {STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W, STARPU_W}
};

/* Initialize a block's layer */
static void initlayer_func(void *descr[], void *arg)
{
	(void)descr;
	size_t sizex, sizey, bz;
	starpu_codelet_unpack_args(arg, &sizex, &sizey, &bz);
	struct block_description *block = get_block_description(bz);
	size_t size_bz = get_block_size(bz);

	/* Initialize layer with some random data */
	size_t x, y, z;
	size_t sum = 0;
	for (x = 0; x < sizex; x++)
		for (y = 0; y < sizey; y++)
			for (z = 0; z < size_bz; z++)
				sum += block->layers[0][(K+x)+(K+y)*(sizex + 2*K)+(K+z)*(sizex+2*K)*(sizey+2*K)] = (int)((x/7.+y/13.+(bz*size_bz + z)/17.) * 10.) % 2;
}

static double initlayer_cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 0.000001;
}

static struct starpu_perfmodel initlayer_model =
{
	.type = STARPU_COMMON,
	.cost_function = initlayer_cost_function,
	.symbol = "initlayer"
};

struct starpu_codelet cl_initlayer =
{
	.cpu_funcs = {initlayer_func},
	.model = &initlayer_model,
	.nbuffers = 1,
	.modes = {STARPU_W}
};

