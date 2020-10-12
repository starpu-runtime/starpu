/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This example illustrates how to distribute a pre-existing data structure to
 * a set of computing nodes using StarPU-MPI routines.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

#define VERBOSE 0

static int N  = 16; /* Matrix size */
static int BS =  4; /* Block size */

#define NB ((N)/(BS)) /* Number of blocks */

/* Matrices. Will be allocated as regular, linearized C arrays */
static double *A = NULL; /* A will be partitioned as BS rows x N  cols blocks */
static double *B = NULL; /* B will be partitioned as N  rows x BS cols blocks */
static double *C = NULL; /* C will be partitioned as BS rows x BS cols blocks */

/* Arrays of data handles for managing matrix blocks */
static starpu_data_handle_t *A_h;
static starpu_data_handle_t *B_h;
static starpu_data_handle_t *C_h;

static int comm_rank; /* mpi rank of the process */
static int comm_size; /* size of the mpi session */

static void alloc_matrices(void)
{
	/* Regular 'malloc' can also be used instead, however, starpu_malloc make sure that
	 * the area is allocated in suitably pinned memory to improve data transfers, especially
	 * with CUDA */
	starpu_malloc((void **)&A, N*N*sizeof(double));
	starpu_malloc((void **)&B, N*N*sizeof(double));
	starpu_malloc((void **)&C, N*N*sizeof(double));
}

static void free_matrices(void)
{
	starpu_free(A);
	starpu_free(B);
	starpu_free(C);
}

static void init_matrices(void)
{
	int row,col;
	for (row = 0; row < N; row++)
	{
		for (col = 0; col < N; col++)
		{
			A[row*N+col] = (row==col)?2:0;
			B[row*N+col] = row*N+col;
			C[row*N+col] = 0;
		}
	}
}

#if VERBOSE
static void disp_matrix(double *m)
{
	int row,col;
	for (row = 0; row < N; row++)
	{
		for (col = 0; col < N; col++)
		{
			printf("\t%.2lf", m[row*N+col]);
		}
		printf("\n");
	}
}
#endif

static void check_result(void)
{
	int row,col;
	for (row = 0; row < N; row++)
	{
		for (col = 0; col < N; col++)
		{
			if (fabs(C[row*N+col] - 2*(row*N+col)) > 1.0)
			{
				fprintf(stderr, "check failed\n");
				exit(1);
			}
		}
	}
#if VERBOSE
	printf("success\n");
#endif
}


/* Register the matrix blocks to StarPU and to StarPU-MPI */
static void register_matrices()
{
	A_h = calloc(NB, sizeof(starpu_data_handle_t));
	B_h = calloc(NB, sizeof(starpu_data_handle_t));
	C_h = calloc(NB*NB, sizeof(starpu_data_handle_t));

	/* Memory region, where the data being registered resides.
	 * In this example, all blocks are allocated by node 0, thus
	 * - node 0 specifies STARPU_MAIN_RAM to indicate that it owns the block in its main memory
	 * - nodes !0 specify -1 to indicate that they don't have a copy of the block initially
	 */
	int mr = (comm_rank == 0) ? STARPU_MAIN_RAM : -1;

	/* mpi tag used for the block */
	starpu_mpi_tag_t tag = 0;

	int b_row,b_col;

	for (b_row = 0; b_row < NB; b_row++)
	{
		/* Register a block to StarPU */
		starpu_matrix_data_register(&A_h[b_row],
				mr,
				(comm_rank == 0)?(uintptr_t)(A+b_row*BS*N):0, N, N, BS,
				sizeof(double));

		/* Register a block to StarPU-MPI, specifying the mpi tag to use for transfering the block
		 * and the rank of the owner node.
		 *
		 * Note: StarPU-MPI is an autonomous layer built on top of StarPU, hence the two separate
		 * registration steps.
		 */
		starpu_data_set_coordinates(A_h[b_row], 2, 0, b_row);
		starpu_mpi_data_register(A_h[b_row], tag++, 0);
	}

	for (b_col = 0; b_col < NB; b_col++)
	{
		starpu_matrix_data_register(&B_h[b_col],
				mr,
				(comm_rank == 0)?(uintptr_t)(B+b_col*BS):0, N, BS, N,
				sizeof(double));
		starpu_data_set_coordinates(B_h[b_col], 2, b_col, 0);
		starpu_mpi_data_register(B_h[b_col], tag++, 0);
	}

	for (b_row = 0; b_row < NB; b_row++)
	{
		for (b_col = 0; b_col < NB; b_col++)
		{
			starpu_matrix_data_register(&C_h[b_row*NB+b_col],
					mr,
					(comm_rank == 0)?(uintptr_t)(C+b_row*BS*N+b_col*BS):0, N, BS, BS,
					sizeof(double));
			starpu_data_set_coordinates(C_h[b_row*NB+b_col], 2, b_col, b_row);
			starpu_mpi_data_register(C_h[b_row*NB+b_col], tag++, 0);
		}
	}
}

/* Transfer ownership of the C matrix blocks following some user-defined distribution over the nodes.
 * Note: since C will be Write-accessed, it will implicitly define which node perform the task
 * associated to a given block. */
static void distribute_matrix_C(void)
{
	int b_row,b_col;
	for (b_row = 0; b_row < NB; b_row++)
	{
		for (b_col = 0; b_col < NB; b_col++)
		{
			starpu_data_handle_t h = C_h[b_row*NB+b_col]; 

			/* Select the node where the block should be computed. */
			int target_rank = (b_row+b_col)%comm_size;

			/* Move the block on to its new owner. */
			starpu_mpi_data_migrate(MPI_COMM_WORLD, h, target_rank);
		}
	}
}

/* Transfer ownership of the C matrix blocks back to node 0, for display purpose. This is not mandatory. */
static void undistribute_matrix_C(void)
{
	int b_row,b_col;
	for (b_row = 0; b_row < NB; b_row++)
	{
		for (b_col = 0; b_col < NB; b_col++)
		{
			starpu_data_handle_t h = C_h[b_row*NB+b_col]; 
			starpu_mpi_data_migrate(MPI_COMM_WORLD, h, 0);
		}
	}
}

/* Unregister matrices from the StarPU management. */
static void unregister_matrices()
{
	int b_row,b_col;

	for (b_row = 0; b_row < NB; b_row++)
	{
		starpu_data_unregister(A_h[b_row]);
	}

	for (b_col = 0; b_col < NB; b_col++)
	{
		starpu_data_unregister(B_h[b_col]);
	}

	for (b_row = 0; b_row < NB; b_row++)
	{
		for (b_col = 0; b_col < NB; b_col++)
		{
			starpu_data_unregister(C_h[b_row*NB+b_col]);
		}
	}

	free(A_h);
	free(B_h);
	free(C_h);
}

/* Perform the actual computation. In a real-life case, this would rather call a BLAS 'gemm' routine
 * instead. */
static void cpu_mult(void *handles[], void *arg)
{
	(void)arg;
	double *block_A = (double *)STARPU_MATRIX_GET_PTR(handles[0]);
	double *block_B = (double *)STARPU_MATRIX_GET_PTR(handles[1]);
	double *block_C = (double *)STARPU_MATRIX_GET_PTR(handles[2]);

	unsigned n_col_A = STARPU_MATRIX_GET_NX(handles[0]);
	unsigned n_col_B = STARPU_MATRIX_GET_NX(handles[1]);
	unsigned n_col_C = STARPU_MATRIX_GET_NX(handles[2]);

	unsigned n_row_A = STARPU_MATRIX_GET_NY(handles[0]);
	unsigned n_row_B = STARPU_MATRIX_GET_NY(handles[1]);
	unsigned n_row_C = STARPU_MATRIX_GET_NY(handles[2]);

	unsigned ld_A = STARPU_MATRIX_GET_LD(handles[0]);
	unsigned ld_B = STARPU_MATRIX_GET_LD(handles[1]);
	unsigned ld_C = STARPU_MATRIX_GET_LD(handles[2]);

	/* Sanity check, not needed in real life case */
	assert(n_col_C == n_col_B);
	assert(n_row_C == n_row_A);
	assert(n_col_A == n_row_B);

	unsigned i,j,k;
	for (k = 0; k < n_row_C; k++)
	{
		for (j = 0; j < n_col_C; j++)
		{
			for (i = 0; i < n_col_A; i++)
			{
				block_C[k*ld_C+j] += block_A[k*ld_A+i] * block_B[i*ld_B+j]; 
			}

#if VERBOSE
			/* For illustration purpose, shows which node computed
			 * the block in the decimal part of the cell */
			block_C[k*ld_C+j] += comm_rank / 100.0;
#endif
		}
	}
}

/* Define a StarPU 'codelet' structure for the matrix multiply kernel above.
 * This structure enable specifying multiple implementations for the kernel (such as CUDA or OpenCL versions)
 */
static struct starpu_codelet gemm_cl =
{
	.cpu_funcs = {cpu_mult}, /* cpu implementation(s) of the routine */
	.nbuffers = 3, /* number of data handles referenced by this routine */
	.modes = {STARPU_R, STARPU_R, STARPU_RW} /* access modes for each data handle */
};

int main(int argc, char *argv[])
{
	/* Initializes STarPU and the StarPU-MPI layer */
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	/* Parse the matrix size and block size optional args */
	if (argc > 1)
	{
		N = atoi(argv[1]);
		if (N < 1)
		{
			fprintf(stderr, "invalid matrix size\n");
			exit(1);
		}
		if (argc > 2)
		{
			BS = atoi(argv[2]);
		}
		if (BS < 1 || N % BS != 0)
		{
			fprintf(stderr, "invalid block size\n");
			exit(1);
		}
	}

	/* Get the process rank and session size */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank == 0)
	{
#if VERBOSE
		printf("N = %d\n", N);
		printf("BS = %d\n", BS);
		printf("NB = %d\n", NB);
		printf("comm_size = %d\n", comm_size);
#endif
		/* In this example, node rank 0 performs all the memory allocations and initializations,
		 * and the blocks are later distributed on the other nodes.
		 * This is not mandatory however, and blocks could be allocated on other nodes right
		 * from the beginning, depending on the application needs (in particular for the case
		 * where the session wide data footprint is larger than a single node available memory. */
		alloc_matrices();
		init_matrices();
	}

	/* Register matrices to StarPU and StarPU-MPI */
	register_matrices();
	/* Distribute C blocks */
	distribute_matrix_C();

	int b_row,b_col;

	for (b_row = 0; b_row < NB; b_row++)
	{
		for (b_col = 0; b_col < NB; b_col++)
		{
			starpu_mpi_task_insert(MPI_COMM_WORLD, &gemm_cl,
					STARPU_R,  A_h[b_row],
					STARPU_R,  B_h[b_col],
					STARPU_RW, C_h[b_row*NB+b_col],
					0);
		}
	}

	starpu_task_wait_for_all();

	undistribute_matrix_C();
	unregister_matrices();

	if (comm_rank == 0)
	{
#if VERBOSE
		disp_matrix(C);
#endif
		check_result();
		free_matrices();
	}

	starpu_mpi_shutdown();
	return 0;
}

