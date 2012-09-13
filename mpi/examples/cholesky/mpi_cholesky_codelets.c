/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#include <starpu_mpi.h>
#include "mpi_cholesky.h"
#include "mpi_cholesky_models.h"
#include "mpi_cholesky_codelets.h"

/*
 *	Create the codelets
 */

static struct starpu_codelet cl11 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_u11, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u11, NULL},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &chol_model_11
};

static struct starpu_codelet cl21 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_u21, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u21, NULL},
#endif
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &chol_model_21
};

static struct starpu_codelet cl22 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_funcs = {chol_cpu_codelet_update_u22, NULL},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u22, NULL},
#endif
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &chol_model_22
};

extern int my_distrib(int x, int y, int nb_nodes);

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */
void dw_cholesky(float ***matA, unsigned size, unsigned ld, unsigned nblocks, int rank, int nodes)
{
	struct timeval start;
	struct timeval end;
	starpu_data_handle_t **data_handles;
	int x, y;

	/* create all the DAG nodes */
	unsigned i,j,k;

	data_handles = malloc(nblocks*sizeof(starpu_data_handle_t *));
	for(x=0 ; x<nblocks ; x++) data_handles[x] = malloc(nblocks*sizeof(starpu_data_handle_t));

	for(x = 0; x < nblocks ;  x++)
	{
		for (y = 0; y < nblocks; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (mpi_rank == rank)
			{
				//fprintf(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
				starpu_matrix_data_register(&data_handles[x][y], 0, (uintptr_t)matA[x][y],
						ld, size/nblocks, size/nblocks, sizeof(float));
			}
			/* TODO: make better test to only registering what is needed */
			else
			{
				/* I don't own that index, but will need it for my computations */
				//fprintf(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
				starpu_matrix_data_register(&data_handles[x][y], -1, (uintptr_t)NULL,
						ld, size/nblocks, size/nblocks, sizeof(float));
			}
			if (data_handles[x][y])
			{
				starpu_data_set_rank(data_handles[x][y], mpi_rank);
				starpu_data_set_tag(data_handles[x][y], (y*nblocks)+x);
			}
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);
	gettimeofday(&start, NULL);

	for (k = 0; k < nblocks; k++)
	{
		int prio = STARPU_DEFAULT_PRIO;
		if (!noprio) prio = STARPU_MAX_PRIO;

		starpu_mpi_insert_task(MPI_COMM_WORLD, &cl11,
				STARPU_PRIORITY, prio,
				STARPU_RW, data_handles[k][k],
				0);

		for (j = k+1; j<nblocks; j++)
		{
			prio = STARPU_DEFAULT_PRIO;
			if (!noprio&& (j == k+1)) prio = STARPU_MAX_PRIO;
			starpu_mpi_insert_task(MPI_COMM_WORLD, &cl21,
					STARPU_PRIORITY, prio,
					STARPU_R, data_handles[k][k],
					STARPU_RW, data_handles[k][j],
					0);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
				{
					prio = STARPU_DEFAULT_PRIO;
					if (!noprio && (i == k + 1) && (j == k +1) ) prio = STARPU_MAX_PRIO;
					starpu_mpi_insert_task(MPI_COMM_WORLD, &cl22,
							STARPU_PRIORITY, prio,
							STARPU_R, data_handles[k][i],
							STARPU_R, data_handles[k][j],
							STARPU_RW, data_handles[i][j],
							0);
				}
			}
		}
	}

	starpu_task_wait_for_all();

	for(x = 0; x < nblocks ;  x++)
	{
		for (y = 0; y < nblocks; y++)
		{
			if (data_handles[x][y])
				starpu_data_unregister(data_handles[x][y]);
		}
		free(data_handles[x]);
	}
	free(data_handles);

	starpu_mpi_barrier(MPI_COMM_WORLD);
	gettimeofday(&end, NULL);

	if (rank == 0)
	{
		double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
		fprintf(stdout, "Computation time (in ms): %2.2f\n", timing/1000);

		double flop = (1.0f*size*size*size)/3.0f;
		fprintf(stdout, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
	}
}

