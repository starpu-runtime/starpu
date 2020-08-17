/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#include "helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <starpu.h>

#include "pxlu.h"
//#include "pxlu_kernels.h"

#ifdef STARPU_HAVE_LIBNUMA
#include <numaif.h>
#endif

static unsigned long size = 4096;
static unsigned nblocks = 16;
static unsigned check = 0;
static int p = -1;
static int q = -1;
static unsigned display = 0;
static unsigned no_prio = 0;

#ifdef STARPU_HAVE_LIBNUMA
static unsigned numa = 0;
#endif

static size_t allocated_memory = 0;
static size_t allocated_memory_extra = 0;

static starpu_data_handle_t *dataA_handles;
static TYPE **dataA;

int get_block_rank(unsigned i, unsigned j);

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		if (strcmp(argv[i], "-display") == 0)
		{
			display = 1;
		}

		if (strcmp(argv[i], "-numa") == 0)
		{
#ifdef STARPU_HAVE_LIBNUMA
			numa = 1;
#else
			fprintf(stderr, "Warning: libnuma is not available\n");
#endif
		}

		if (strcmp(argv[i], "-p") == 0)
		{
			char *argptr;
			p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-q") == 0)
		{
			char *argptr;
			q = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0)
		{
			fprintf(stderr,"usage: %s [-size n] [-nblocks b] [-check] [-display] [-numa] [-p p] [-q q]\n", argv[0]);
			fprintf(stderr,"\np * q must be equal to the number of MPI nodes\n");
			exit(0);
		}
	}

#ifdef STARPU_HAVE_VALGRIND_H
	if (RUNNING_ON_VALGRIND)
		size = 16;
#endif
}

unsigned STARPU_PLU(display_flag)(void)
{
	return display;
}

static void fill_block_with_random(TYPE *blockptr, unsigned psize, unsigned pnblocks)
{
	const unsigned block_size = (psize/pnblocks);

	unsigned i, j;
	for (i = 0; i < block_size; i++)
	     for (j = 0; j < block_size; j++)
	     {
		  blockptr[j+i*block_size] = (TYPE)starpu_drand48();
	     }
}

static void init_matrix(int rank)
{
#ifdef STARPU_HAVE_LIBNUMA
	if (numa)
	{
		fprintf(stderr, "Using INTERLEAVE policy\n");
		unsigned long nodemask = ((1<<0)|(1<<1));
		int ret = set_mempolicy(MPOL_INTERLEAVE, &nodemask, 3);
		if (ret)
			perror("set_mempolicy failed");
	}
#endif

	/* Allocate a grid of data handles, not all of them have to be allocated later on */
	dataA_handles = calloc(nblocks*nblocks, sizeof(starpu_data_handle_t));
	dataA = calloc(nblocks*nblocks, sizeof(TYPE *));
	allocated_memory_extra += nblocks*nblocks*(sizeof(starpu_data_handle_t) + sizeof(TYPE *));

	size_t blocksize = (size_t)(size/nblocks)*(size/nblocks)*sizeof(TYPE);

	/* Allocate all the blocks that belong to this mpi node */
	unsigned long i,j;
	for (j = 0; j < nblocks; j++)
	{
		for (i = 0; i < nblocks; i++)
		{
			int block_rank = get_block_rank(i, j);
			TYPE **blockptr = &dataA[j+i*nblocks];
//			starpu_data_handle_t *handleptr = &dataA_handles[j+nblocks*i];
			starpu_data_handle_t *handleptr = &dataA_handles[j+nblocks*i];

			if (block_rank == rank)
			{
				/* This blocks should be treated by the current MPI process */
				/* Allocate and fill it */
				starpu_malloc((void **)blockptr, blocksize);
				allocated_memory += blocksize;

				//fprintf(stderr, "Rank %d : fill block (i = %d, j = %d)\n", rank, i, j);
				fill_block_with_random(*blockptr, size, nblocks);
				//fprintf(stderr, "Rank %d : fill block (i = %d, j = %d)\n", rank, i, j);
				if (i == j)
				{
					unsigned tmp;
					for (tmp = 0; tmp < size/nblocks; tmp++)
					{
						(*blockptr)[tmp*((size/nblocks)+1)] += (TYPE)10*nblocks;
					}
				}

				/* Register it to StarPU */
				starpu_matrix_data_register(handleptr, STARPU_MAIN_RAM,
					(uintptr_t)*blockptr, size/nblocks,
					size/nblocks, size/nblocks, sizeof(TYPE));
			}
			else
			{
				starpu_matrix_data_register(handleptr, -1,
					0, size/nblocks,
					size/nblocks, size/nblocks, sizeof(TYPE));
				*blockptr = STARPU_POISON_PTR;
			}
			starpu_data_set_coordinates(*handleptr, 2, j, i);
			starpu_mpi_data_register(*handleptr, j+i*nblocks, block_rank);
		}
	}

	//display_all_blocks(nblocks, size/nblocks);
}

TYPE *STARPU_PLU(get_block)(unsigned i, unsigned j)
{
	return dataA[j+i*nblocks];
}

int get_block_rank(unsigned i, unsigned j)
{
	/* Take a 2D block cyclic distribution */
	/* NB: p (resp. q) is for "direction" i (resp. j) */
	return (j % q) * p + (i % p);
}

starpu_data_handle_t STARPU_PLU(get_block_handle)(unsigned i, unsigned j)
{
	return dataA_handles[j+i*nblocks];
}

static void display_grid(int rank, unsigned pnblocks)
{
	if (!display)
		return;

	//if (rank == 0)
	{
		fprintf(stderr, "2D grid layout (Rank %d): \n", rank);

		unsigned i, j;
		for (j = 0; j < pnblocks; j++)
		{
			for (i = 0; i < pnblocks; i++)
			{
				TYPE *blockptr = STARPU_PLU(get_block)(i, j);
				starpu_data_handle_t handle = STARPU_PLU(get_block_handle)(i, j);

				fprintf(stderr, "%d (data %p handle %p)", get_block_rank(i, j), blockptr, handle);
			}
			fprintf(stderr, "\n");
		}
	}
}

int main(int argc, char **argv)
{
	int rank;
	int world_size;
	int ret;
	unsigned i, j;

	starpu_srand48((long int)time(NULL));

	parse_args(argc, argv);

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &world_size);

	if (p == -1 && q==-1)
	{
		fprintf(stderr, "Setting default values for p and q\n");
		p = (q % 2 == 0) ? 2 : 1;
		q = world_size / p;

	}
	STARPU_ASSERT_MSG(p*q == world_size, "p=%d, q=%d, world_size=%d\n", p, q, world_size);

	starpu_cublas_init();

	/*
	 * 	Problem Init
	 */

	init_matrix(rank);

	fprintf(stderr, "Rank %d: allocated (%d + %d) MB = %d MB\n", rank,
                        (int)(allocated_memory/(1024*1024)),
			(int)(allocated_memory_extra/(1024*1024)),
                        (int)((allocated_memory+allocated_memory_extra)/(1024*1024)));

	display_grid(rank, nblocks);

	TYPE *a_r = NULL;
//	STARPU_PLU(display_data_content)(a_r, size);

	if (check)
	{
		TYPE *x, *y;

		x = calloc(size, sizeof(TYPE));
		STARPU_ASSERT(x);

		y = calloc(size, sizeof(TYPE));
		STARPU_ASSERT(y);

		if (rank == 0)
		{
			unsigned ind;
			for (ind = 0; ind < size; ind++)
				x[ind] = (TYPE)starpu_drand48();
		}

		a_r = STARPU_PLU(reconstruct_matrix)(size, nblocks);

		if (rank == 0)
			STARPU_PLU(display_data_content)(a_r, size);

//		STARPU_PLU(compute_ax)(size, x, y, nblocks, rank);

		free(x);
		free(y);
	}

	double timing = STARPU_PLU(plu_main)(nblocks, rank, world_size, no_prio);

	/*
	 * 	Report performance
	 */

	if (rank == 0)
	{
		fprintf(stderr, "Computation took: %f ms\n", timing/1000);

		unsigned n = size;
		double flop = (2.0f*n*n*n)/3.0f;
		fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
	}

	/*
	 *	Test Result Correctness
	 */

	if (check)
	{
		/*
		 *	Compute || A - LU ||
		 */

		STARPU_PLU(compute_lu_matrix)(size, nblocks, a_r);

#if 0
		/*
		 *	Compute || Ax - LUx ||
		 */

		unsigned ind;

		y2 = calloc(size, sizeof(TYPE));
		STARPU_ASSERT(y);

		if (rank == 0)
		{
			for (ind = 0; ind < size; ind++)
			{
				y2[ind] = (TYPE)0.0;
			}
		}

		STARPU_PLU(compute_lux)(size, x, y2, nblocks, rank);

		/* Compute y2 = y2 - y */
		CPU_AXPY(size, -1.0, y, 1, y2, 1);

		TYPE err = CPU_ASUM(size, y2, 1);
		int max = CPU_IAMAX(size, y2, 1);

		fprintf(stderr, "(A - LU)X Avg error : %e\n", err/(size*size));
		fprintf(stderr, "(A - LU)X Max error : %e\n", y2[max]);
#endif
	}

	/*
	 * 	Termination
	 */
	for (j = 0; j < nblocks; j++)
	{
		for (i = 0; i < nblocks; i++)
		{
			starpu_data_unregister(dataA_handles[j+nblocks*i]);
			TYPE *blockptr = dataA[j+i*nblocks];
			if (blockptr != STARPU_POISON_PTR)
				starpu_free(blockptr);
		}
	}
	free(dataA_handles);
	free(dataA);

	starpu_cublas_shutdown();
	starpu_mpi_shutdown();

	return 0;
}
