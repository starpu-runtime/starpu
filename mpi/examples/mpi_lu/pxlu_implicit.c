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

#include "pxlu.h"
#include "pxlu_kernels.h"
#include <sys/time.h>

//#define VERBOSE_INIT	1

//#define DEBUG	1

static unsigned no_prio = 0;
static unsigned nblocks = 0;
static int rank = -1;
static int world_size = -1;

struct callback_arg
{
	unsigned i, j, k;
};

/*
 *	Task 11 (diagonal factorization)
 */

static void create_task_11(unsigned k)
{
	starpu_mpi_task_insert(MPI_COMM_WORLD,
			       &STARPU_PLU(cl11),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_RW, STARPU_PLU(get_block_handle)(k, k),
			       STARPU_PRIORITY, !no_prio ?
			       STARPU_MAX_PRIO : STARPU_MIN_PRIO,
			       0);
}

/*
 *	Task 12 (Update lower left (TRSM))
 */

static void create_task_12(unsigned k, unsigned j)
{
#ifdef STARPU_DEVEL
#warning temporary fix 
#endif
	starpu_mpi_task_insert(MPI_COMM_WORLD,
			       //&STARPU_PLU(cl12),
			       &STARPU_PLU(cl21),
			       STARPU_VALUE, &j, sizeof(j),
			       STARPU_VALUE, &j, sizeof(j),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_R, STARPU_PLU(get_block_handle)(k, k),
			       STARPU_RW, STARPU_PLU(get_block_handle)(k, j),
			       STARPU_PRIORITY, !no_prio && (j == k+1) ?
			       STARPU_MAX_PRIO : STARPU_MIN_PRIO,
			       0);
}

/*
 *	Task 21 (Update upper right (TRSM))
 */

static void create_task_21(unsigned k, unsigned i)
{
#ifdef STARPU_DEVEL
#warning temporary fix 
#endif
	starpu_mpi_task_insert(MPI_COMM_WORLD,
			       //&STARPU_PLU(cl21),
			       &STARPU_PLU(cl12),
			       STARPU_VALUE, &i, sizeof(i),
			       STARPU_VALUE, &i, sizeof(i),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_R, STARPU_PLU(get_block_handle)(k, k),
			       STARPU_RW, STARPU_PLU(get_block_handle)(i, k),
			       STARPU_PRIORITY, !no_prio && (i == k+1) ?
			       STARPU_MAX_PRIO : STARPU_MIN_PRIO,
			       0);
}

/*
 *	Task 22 (GEMM)
 */

static void create_task_22(unsigned k, unsigned i, unsigned j)
{
	starpu_mpi_task_insert(MPI_COMM_WORLD,
			       &STARPU_PLU(cl22),
			       STARPU_VALUE, &i, sizeof(i),
			       STARPU_VALUE, &j, sizeof(j),
			       STARPU_VALUE, &k, sizeof(k),
			       STARPU_R, STARPU_PLU(get_block_handle)(k, j),
			       STARPU_R, STARPU_PLU(get_block_handle)(i, k),
			       STARPU_RW, STARPU_PLU(get_block_handle)(i, j),
			       STARPU_PRIORITY, !no_prio && (i == k + 1) && (j == k +1) ?
			       STARPU_MAX_PRIO : STARPU_MIN_PRIO,
			       0);
}

/*
 *	code to bootstrap the factorization 
 */

double STARPU_PLU(plu_main)(unsigned _nblocks, int _rank, int _world_size, unsigned _no_prio)
{
	double start;
	double end;

	nblocks = _nblocks;
	rank = _rank;
	world_size = _world_size;
	no_prio = _no_prio;

	/* create all the DAG nodes */
	unsigned i,j,k;

	starpu_mpi_barrier(MPI_COMM_WORLD);

	start = starpu_timing_now();

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);

		create_task_11(k);

		for (i = k+1; i<nblocks; i++)
		{
			create_task_12(k, i);
			create_task_21(k, i);
		}

		starpu_mpi_cache_flush(MPI_COMM_WORLD, STARPU_PLU(get_block_handle)(k,k));
		if (get_block_rank(k, k) == _rank)
			starpu_data_wont_use(STARPU_PLU(get_block_handle)(k,k));

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
				create_task_22(k, i, j);
			}
		}

		for (i = k+1; i<nblocks; i++)
		{
			starpu_mpi_cache_flush(MPI_COMM_WORLD, STARPU_PLU(get_block_handle)(k,i));
			if (get_block_rank(k, i) == _rank)
				starpu_data_wont_use(STARPU_PLU(get_block_handle)(k,i));
			starpu_mpi_cache_flush(MPI_COMM_WORLD, STARPU_PLU(get_block_handle)(i,k));
			if (get_block_rank(i, k) == _rank)
				starpu_data_wont_use(STARPU_PLU(get_block_handle)(i,k));
		}
		starpu_iteration_pop();
	}

	starpu_task_wait_for_all();

	starpu_mpi_barrier(MPI_COMM_WORLD);

	end = starpu_timing_now();

	double timing = end - start;
	
//	fprintf(stderr, "RANK %d -> took %f ms\n", rank, timing/1000);
	
	return timing;
}
