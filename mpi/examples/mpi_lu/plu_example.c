/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <starpu.h>

#include "pxlu.h"
//#include "pxlu_kernels.h"

static unsigned long size = 16384;
static unsigned nblocks = 16;
static unsigned check = 0;
static unsigned p = 1;
static unsigned q = 1;

static starpu_data_handle *dataA_handles;
static TYPE **dataA;

/* In order to implement the distributed LU decomposition, we allocate
 * temporary buffers */
static starpu_data_handle tmp_11_block_handle;
static TYPE **tmp_11_block;
static starpu_data_handle *tmp_12_block_handles;
static TYPE **tmp_12_block;
static starpu_data_handle *tmp_21_block_handles;
static TYPE *tmp_21_block;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-check") == 0) {
			check = 1;
		}

		if (strcmp(argv[i], "-p") == 0) {
			char *argptr;
			p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-q") == 0) {
			char *argptr;
			q = strtol(argv[++i], &argptr, 10);
		}
	}
}

static void fill_block_with_random(TYPE *blockptr, unsigned size, unsigned nblocks)
{
	const unsigned block_size = (size/nblocks);

	unsigned i, j;
	for (j = 0; j < block_size; j++)
	for (i = 0; i < block_size; i++)
	{
		blockptr[i+j*block_size] = (TYPE)drand48();
//		blockptr[i+j*block_size] = (i == j)?2.0:0.0;
	}
}

starpu_data_handle STARPU_PLU(get_block_handle)(unsigned j, unsigned i)
{
	return dataA_handles[i+j*nblocks];
}

starpu_data_handle STARPU_PLU(get_tmp_11_block_handle)(void)
{
	return tmp_11_block_handle;
}

starpu_data_handle STARPU_PLU(get_tmp_12_block_handle)(unsigned j)
{
	return tmp_12_block_handles[j];
}

starpu_data_handle STARPU_PLU(get_tmp_21_block_handle)(unsigned i)
{
	return tmp_21_block_handles[i];
}

TYPE *STARPU_PLU(get_block)(unsigned j, unsigned i)
{
	return dataA[i+j*nblocks];
}

static void init_matrix(int rank)
{
	/* Allocate a grid of data handles, not all of them have to be allocated later on */
	dataA_handles = calloc(nblocks*nblocks, sizeof(starpu_data_handle));
	dataA = calloc(nblocks*nblocks, sizeof(TYPE *));

	size_t blocksize = (size_t)(size/nblocks)*(size/nblocks)*sizeof(TYPE);

	/* Allocate all the blocks that belong to this mpi node */
	unsigned long i,j;
	for (j = 0; j < nblocks; j++)
	{
		for (i = 0; i < nblocks; i++)
		{
			if (get_block_rank(i, j) == rank)
			{
				/* This blocks should be treated by the current MPI process */
				/* Allocate and fill it */
				starpu_malloc_pinned_if_possible((void **)&dataA[i+j*nblocks], blocksize);

				fill_block_with_random(STARPU_PLU(get_block)(j, i), size, nblocks);

				/* Register it to StarPU */
				starpu_register_blas_data(&dataA_handles[i+nblocks*j], 0,
					(uintptr_t)dataA[i+nblocks*j], size/nblocks,
					size/nblocks, size/nblocks, sizeof(TYPE));
			} 
		}
	}

	/* Allocate the temporary buffers required for the distributed algorithm */

	/* tmp buffer 11 */
	starpu_malloc_pinned_if_possible((void **)&tmp_11_block, blocksize);
	starpu_register_blas_data(&tmp_11_block_handle, 0, (uintptr_t)tmp_11_block,
			size/nblocks, size/nblocks, size/nblocks, sizeof(TYPE));

	/* tmp buffers 12 and 21 */
	tmp_12_block_handles = malloc(nblocks*sizeof(starpu_data_handle));
	tmp_21_block_handles = malloc(nblocks*sizeof(starpu_data_handle));
	tmp_12_block = malloc(nblocks*sizeof(TYPE *));
	tmp_21_block = malloc(nblocks*sizeof(TYPE *));
	
	unsigned k;
	for (k = 0; k < nblocks; k++)
	{
		starpu_malloc_pinned_if_possible((void **)&tmp_12_block[k], blocksize);
		starpu_register_blas_data(&tmp_12_block_handles[k], 0,
			(uintptr_t)tmp_12_block[k],
			size/nblocks, size/nblocks, size/nblocks, sizeof(TYPE));

		starpu_malloc_pinned_if_possible((void **)&tmp_21_block[k], blocksize);
		starpu_register_blas_data(&tmp_21_block_handles[k], 0,
			(uintptr_t)tmp_21_block[k],
			size/nblocks, size/nblocks, size/nblocks, sizeof(TYPE));
	}
}

int get_block_rank(unsigned i, unsigned j)
{
	/* Take a 2D block cyclic distribution */
	/* NB: p (resp. q) is for "direction" i (resp. j) */
	return (j % q) * p + (i % p);
}

int main(int argc, char **argv)
{
	int rank;
	int world_size;

	/*
	 *	Initialization
	 */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand48((long int)time(NULL));

	parse_args(argc, argv);

	STARPU_ASSERT(p*q == world_size);

	starpu_init(NULL);
	starpu_mpi_initialize();
	starpu_helper_init_cublas();

	int barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);

	/*
	 * 	Problem Init
	 */

	init_matrix(rank);

	TYPE *x, *y;

	if (check)
	{
		if (rank == 0)
			fprintf(stderr, "Compute AX = B\n");

		x = calloc(size, sizeof(TYPE));
		STARPU_ASSERT(x);

		y = calloc(size, sizeof(TYPE));
		STARPU_ASSERT(y);
		
		unsigned ind;
		for (ind = 0; ind < size; ind++)
		{
			//x[ind] = (TYPE)1.0;
			x[ind] = (TYPE)drand48();
			y[ind] = (TYPE)0.0;
		}

		STARPU_PLU(compute_ax)(size, x, y, nblocks, rank);

		if (rank == 0)
		for (ind = 0; ind < 10; ind++)
		{
			fprintf(stderr, "y[%d] = %f\n", ind, (float)y[ind]);
		}
		
	}

	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
//	fprintf(stderr, "Rank %d PID %d\n", rank, getpid());
//	sleep(10);

	STARPU_PLU(plu_main)(nblocks, rank, world_size);

	/*
	 * 	Termination
	 */
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);

	starpu_helper_shutdown_cublas();
	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return 0;
}
