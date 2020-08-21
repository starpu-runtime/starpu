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

/* This is from magma

  -- Innovative Computing Laboratory
  -- Electrical Engineering and Computer Science Department
  -- University of Tennessee
  -- (C) Copyright 2009

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the University of Tennessee, Knoxville nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  */

#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))

#define FLOPS_SPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )

#define FMULS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)+1.))
#define FADDS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)-1.))

#define FMULS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FMULS_TRMM_2((__m), (__n)) :*/ FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FADDS_TRMM_2((__m), (__n)) :*/ FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

#define FLOPS_STRSM(__m, __n) (     FMULS_TRSM((__m), (__n)) +       FADDS_TRSM((__m), (__n)) )


#define FMULS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((__m), (__n), (__k)) +       FADDS_GEMM((__m), (__n), (__k)) )

/* End of magma code */

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
	unsigned nn = size/nblocks;

	unsigned unbound_prio = STARPU_MAX_PRIO == INT_MAX && STARPU_MIN_PRIO == INT_MIN;

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

	starpu_mpi_barrier(MPI_COMM_WORLD);
	start = starpu_timing_now();

	for (k = 0; k < nblocks; k++)
	{
		starpu_iteration_push(k);

		starpu_mpi_task_insert(MPI_COMM_WORLD, &cl11,
				       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k) : STARPU_MAX_PRIO,
				       STARPU_RW, data_handles[k][k],
				       STARPU_FLOPS, (double) FLOPS_SPOTRF(nn),
				       0);

		for (m = k+1; m<nblocks; m++)
		{
			starpu_mpi_task_insert(MPI_COMM_WORLD, &cl21,
					       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m) : (m == k+1)?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
					       STARPU_R, data_handles[k][k],
					       STARPU_RW, data_handles[m][k],
					       STARPU_FLOPS, (double) FLOPS_STRSM(nn, nn),
					       0);

			starpu_mpi_cache_flush(MPI_COMM_WORLD, data_handles[k][k]);
			if (my_distrib(k, k, nodes) == rank)
				starpu_data_wont_use(data_handles[k][k]);
		}

		for (n = k+1; n<nblocks; n++)
		{
			for (m = n; m<nblocks; m++)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &cl22,
						       STARPU_PRIORITY, noprio ? STARPU_DEFAULT_PRIO : unbound_prio ? (int)(2*nblocks - 2*k - m - n) : ((n == k+1) && (m == k+1))?STARPU_MAX_PRIO:STARPU_DEFAULT_PRIO,
						       STARPU_R, data_handles[n][k],
						       STARPU_R, data_handles[m][k],
						       STARPU_RW | STARPU_COMMUTE, data_handles[m][n],
						       STARPU_FLOPS, (double) FLOPS_SGEMM(nn, nn, nn),
						       0);
			}

			starpu_mpi_cache_flush(MPI_COMM_WORLD, data_handles[n][k]);
			if (my_distrib(n, k, nodes) == rank)
				starpu_data_wont_use(data_handles[n][k]);
		}
		starpu_iteration_pop();
	}

	starpu_task_wait_for_all();

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
		*flops = FLOPS_SPOTRF(size);
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
