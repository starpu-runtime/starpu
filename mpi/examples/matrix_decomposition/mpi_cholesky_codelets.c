/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "mpi_cholesky.h"
#include <common/blas.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>

/*
 *	Create the codelets
 */

static struct starpu_codelet cl11 =
{
	.cpu_funcs = {chol_cpu_codelet_update_u11},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u11},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &chol_model_11,
	.color = 0xffff00,
};

static struct starpu_codelet cl21 =
{
	.cpu_funcs = {chol_cpu_codelet_update_u21},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u21},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &chol_model_21,
	.color = 0x8080ff,
};

static struct starpu_codelet cl22 =
{
	.cpu_funcs = {chol_cpu_codelet_update_u22},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_u22},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW | STARPU_COMMUTE},
	.model = &chol_model_22,
	.color = 0x00ff00,
};

static void run_cholesky(starpu_data_handle_t **data_handles, int rank, int nodes)
{
	unsigned k, m, n;
	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);

		starpu_mpi_task_insert(MPI_COMM_WORLD, &cl11,
				       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
				       STARPU_RW, data_handles[k][k],
				       0);

		for (m = k+1; m<nblocks; m++)
		{
			starpu_mpi_task_insert(MPI_COMM_WORLD, &cl21,
					       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
					       STARPU_R, data_handles[k][k],
					       STARPU_RW, data_handles[m][k],
					       0);

			starpu_mpi_cache_flush(MPI_COMM_WORLD, data_handles[k][k]);
			if (my_distrib(k, k, nodes) == rank)
				starpu_data_wont_use(data_handles[k][k]);

			for (n = k+1; n<nblocks; n++)
			{
				if (n <= m)
				{
					starpu_mpi_task_insert(MPI_COMM_WORLD, &cl22,
							       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
							       STARPU_R, data_handles[n][k],
							       STARPU_R, data_handles[m][k],
							       STARPU_RW | STARPU_COMMUTE, data_handles[m][n],
							       0);
				}
			}

			starpu_mpi_cache_flush(MPI_COMM_WORLD, data_handles[m][k]);
			if (my_distrib(m, k, nodes) == rank)
				starpu_data_wont_use(data_handles[m][k]);
		}
		starpu_iteration_pop();
	}
}

/* TODO: generate from compiler polyhedral analysis of classical algorithm */
static void run_cholesky_column(starpu_data_handle_t **data_handles, int rank, int nodes)
{
	unsigned k, m, n;
	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	/* Column */
	for (n = 0; n<nblocks; n++)
	{
		starpu_iteration_push(n);

		/* Row */
		for (m = n; m<nblocks; m++)
		{
			for (k = 0; k < n; k++)
			{
				/* Accumulate updates from TRSMs */
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl22,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[n][k],
						       STARPU_R, data_handles[m][k],
						       STARPU_RW | STARPU_COMMUTE, data_handles[m][n],
						       0);
			}
			k = n;
			if (m > n)
			{
				/* non-diagonal block, solve */
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl21,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[k][k],
						       STARPU_RW, data_handles[m][k],
						       0);
			}
			else
			{
				/* diagonal block, factorize */
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl11,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
						       STARPU_RW, data_handles[k][k],
						       0);
			}
		}

		starpu_iteration_pop();
	}

	/* Submit flushes, StarPU will fit them according to the progress */
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
	for (m = 0; m < nblocks; m++)
		for (n = 0; n < nblocks ; n++)
			starpu_data_wont_use(data_handles[m][n]);
}

/* TODO: generate from compiler polyhedral analysis of classical algorithm */
static void run_cholesky_antidiagonal(starpu_data_handle_t **data_handles, int rank, int nodes)
{
	unsigned a, c;
	unsigned k, m, n;
	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

	/* double-antidiagonal number:
	 * - a=0 contains (0,0) plus (1,0)
	 * - a=1 contains (2,0), (1,1) plus (3,0), (2, 1)
	 * - etc.
	 */
	for (a = 0; a < nblocks; a++)
	{
		starpu_iteration_push(a);

		unsigned nfirst;
		if (2*a < nblocks)
			nfirst = 0;
		else
			nfirst = 2*a - (nblocks-1);

		/* column within first antidiagonal for a */
		for (n = nfirst; n <= a; n++)
		{
			/* row */
			m = 2*a-n;

			/* Accumulate updates from TRSMs */
			for (k = 0; k < n; k++)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl22,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[n][k],
						       STARPU_R, data_handles[m][k],
						       STARPU_RW | STARPU_COMMUTE, data_handles[m][n],
						       0);
			}

			/* k = n */
			if (n < a)
			{
				/* non-diagonal block, solve */
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl21,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[k][k],
						       STARPU_RW, data_handles[m][k],
						       0);
			}
			else
			{
				/* diagonal block, factorize */
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl11,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
						       STARPU_RW, data_handles[k][k],
						       0);
			}
		}

		/* column within second antidiagonal for a */
		for (n = nfirst; n <= a; n++)
		{
			/* row */
			m = 2*a-n + 1;

			if (m >= nblocks)
				/* Skip first item when even number of tiles */
				continue;

			/* Accumulate updates from TRSMs */
			for (k = 0; k < n; k++)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl22,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[n][k],
						       STARPU_R, data_handles[m][k],
						       STARPU_RW | STARPU_COMMUTE, data_handles[m][n],
						       0);
			}
			/* non-diagonal block, solve */
			k = n;
			starpu_mpi_task_insert(MPI_COMM_WORLD, &cl21,
					       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
					       STARPU_R, data_handles[k][k],
					       STARPU_RW, data_handles[m][k],
					       0);
		}

		starpu_iteration_pop();
	}

	/* Submit flushes, StarPU will fit them according to the progress */
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
	for (m = 0; m < nblocks; m++)
		for (n = 0; n < nblocks ; n++)
			starpu_data_wont_use(data_handles[m][n]);
}

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */
void dw_cholesky(float ***matA, unsigned ld, int rank, int nodes, double *timing, double *flops)
{
	double start;
	double end;
	starpu_data_handle_t **data_handles;
	unsigned k, m, n;

	/* create all the DAG nodes */

	data_handles = malloc(nblocks*sizeof(starpu_data_handle_t *));
	for(m=0 ; m<nblocks ; m++) data_handles[m] = malloc(nblocks*sizeof(starpu_data_handle_t));

	for (m = 0; m < nblocks; m++)
	{
		for(n = 0; n < nblocks ; n++)
		{
			int mpi_rank = my_distrib(m, n, nodes);
			if (mpi_rank == rank || (check && rank == 0))
			{
				//fprintf(stderr, "[%d] Owning data[%d][%d]\n", rank, n, m);
				starpu_matrix_data_register(&data_handles[m][n], STARPU_MAIN_RAM, (uintptr_t)matA[m][n],
						ld, size/nblocks, size/nblocks, sizeof(float));
			}
#ifdef STARPU_DEVEL
#warning TODO: make better test to only register what is needed
#endif
			else
			{
				/* I don't own this index, but will need it for my computations */
				//fprintf(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, n, m);
				starpu_matrix_data_register(&data_handles[m][n], -1, (uintptr_t)NULL,
						ld, size/nblocks, size/nblocks, sizeof(float));
			}
			if (data_handles[m][n])
			{
				starpu_data_set_coordinates(data_handles[m][n], 2, n, m);
				starpu_mpi_data_register(data_handles[m][n], (m*nblocks)+n, mpi_rank);
			}
		}
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);
	start = starpu_timing_now();

	switch (submission)
	{
		case TRIANGLES:		run_cholesky(data_handles, rank, nodes); break;
		case COLUMNS:		run_cholesky_column(data_handles, rank, nodes); break;
		case ANTIDIAGONALS:	run_cholesky_antidiagonal(data_handles, rank, nodes); break;
		default: STARPU_ABORT();
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);
	end = starpu_timing_now();

	for (m = 0; m < nblocks; m++)
	{
		for(n = 0; n < nblocks ; n++)
		{
			/* Get back data on node 0 for the check */
			if (check && data_handles[m][n])
				starpu_mpi_get_data_on_node(MPI_COMM_WORLD, data_handles[m][n], 0);

			if (data_handles[m][n])
				starpu_data_unregister(data_handles[m][n]);
		}
		free(data_handles[m]);
	}
	free(data_handles);

	if (rank == 0)
	{
		*timing = end - start;
		*flops = (1.0f*size*size*size)/3.0f;
	}
}

void dw_cholesky_check_computation(float ***matA, int rank, int nodes, int *correctness, double *flops, double epsilon)
{
	unsigned nn,mm,n,m;
	float *rmat = malloc(size*size*sizeof(float));

	for(n=0 ; n<nblocks ; n++)
	{
		for(m=0 ; m<nblocks ; m++)
		{
			for (nn = 0; nn < BLOCKSIZE; nn++)
			{
				for (mm = 0; mm < BLOCKSIZE; mm++)
				{
					rmat[mm+(m*BLOCKSIZE)+(nn+(n*BLOCKSIZE))*size] = matA[m][n][mm +nn*BLOCKSIZE];
				}
			}
		}
	}

	FPRINTF(stderr, "[%d] compute explicit LLt ...\n", rank);
	for (mm = 0; mm < size; mm++)
	{
		for (nn = 0; nn < size; nn++)
		{
			if (nn > mm)
			{
				rmat[mm+nn*size] = 0.0f; // debug
			}
		}
	}
	float *test_mat = malloc(size*size*sizeof(float));
	STARPU_ASSERT(test_mat);

	STARPU_SSYRK("L", "N", size, size, 1.0f,
			rmat, size, 0.0f, test_mat, size);

	FPRINTF(stderr, "[%d] comparing results ...\n", rank);
	if (display)
	{
		for (mm = 0; mm < size; mm++)
		{
			for (nn = 0; nn < size; nn++)
			{
				if (nn <= mm)
				{
					printf("%2.2f\t", test_mat[mm +nn*size]);
				}
				else
				{
					printf(".\t");
				}
			}
			printf("\n");
		}
	}

	*correctness = 1;
	for(n = 0; n < nblocks ; n++)
	{
		for (m = 0; m < nblocks; m++)
		{
			for (nn = BLOCKSIZE*n ; nn < BLOCKSIZE*(n+1); nn++)
			{
				for (mm = BLOCKSIZE*m ; mm < BLOCKSIZE*(m+1); mm++)
				{
					if (nn <= mm)
					{
						float orig = (1.0f/(1.0f+nn+mm)) + ((nn == mm)?1.0f*size:0.0f);
						float err = fabsf(test_mat[mm +nn*size] - orig) / orig;
						if (err > epsilon)
						{
							FPRINTF(stderr, "[%d] Error[%u, %u] --> %2.20f != %2.20f (err %2.20f)\n", rank, nn, mm, test_mat[mm +nn*size], orig, err);
							*correctness = 0;
							*flops = 0;
							break;
						}
					}
				}
			}
		}
	}
	free(rmat);
	free(test_mat);
}
