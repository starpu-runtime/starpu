/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example illustrates the computation of general matrices with originally
 * distributed A, B and C matrices to a set of computing nodes.
 */

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"
#include <common/blas.h>

#define VERBOSE 0


static int M  = 1024; /* Matrix size */
static int N  = 1024; /* Matrix size */
static int K  = 1024; /* Matrix size */
static int BS =  512; /* Block size */
static int P  =    2; /* height of the grid */
static int Q  =    2; /* width of the grid */
static int T  =    1; /* number of runs */
static int trace = 0; /* whether to trace */

#define MB ((M)/(BS)) /* Number of blocks */
#define NB ((N)/(BS)) /* Number of blocks */
#define KB ((K)/(BS)) /* Number of blocks */


/* Arrays of data handles for managing matrix blocks */
static starpu_data_handle_t *A_h;
static starpu_data_handle_t *B_h;
static starpu_data_handle_t *C_h;

static int comm_rank; /* mpi rank of the process */
static int comm_size; /* size of the mpi session */

struct block
{
        double* c;
        int owner;
}

struct matrix
{
        int mb, nb, b;
        struct block* blocks;
}

/* Matrices. Will be allocated as regular, linearized C arrays */
static struct matrix *A = NULL; /* A will be partitioned as MB x KB blocks */
static struct matrix *B = NULL; /* B will be partitioned as KB x NB blocks */
static struct matrix *C = NULL; /* C will be partitioned as MB x NB blocks */

struct matrix* alloc_matrix(int mb, int nb)
{
	struct matrix* X;
	X = malloc(sizeof(struct matrix));
      	X->blocks = malloc( mb*nb*sizeof(struct block));
	int i,j;
	for (i = 0; i<mb; i++)
	{
		for (j= 0; j<nb; j++)
		{
			X->blocks[i*nb+j].owner = (i%P)*Q + (j%Q);
			if (X->blocks[i*nb+j].owner == comm_rank)
  				X->blocks[i*nb+j].c = malloc(BS*BS*sizeof(double));
		}
	}
	X->mb = mb;
	X->nb = nb;
       	X->b  = BS;
	return X;
}
static void alloc_matrices(void)
{
	if (VERBOSE) printf( "Allocating matrices\n");
	A = alloc_matrix(MB,KB);
	B = alloc_matrix(KB,NB);
	C = alloc_matrix(MB,NB);
}

static void free_matrix(struct matrix* X, int mb, int nb)
{
	int i,j;
	for (i = 0; i<mb; i++)
	{
		for (j= 0; j<nb; j++)
		{
			if (X->blocks[i*nb+j].owner == comm_rank)
				free(X->blocks[i*nb+j].c);
		}
	}
	free(X->blocks);
	free(X);
}

static void free_matrices(void)
{
	if (VERBOSE) printf( "Freeing matrices\n");
  	free_matrix(A,MB,KB);
  	free_matrix(B,KB,NB);
  	free_matrix(C,MB,NB);
}

static void register_matrix(struct matrix* X, starpu_data_handle_t* X_h, starpu_mpi_tag_t *tag, int mb, int nb)
{
	int b_row, b_col;
	for (b_row = 0; b_row < mb; b_row++)
	{
		for (b_col = 0; b_col < nb; b_col++)
		{
	    		if (X->blocks[b_row*nb+b_col].owner == comm_rank)
			{
				starpu_matrix_data_register(&X_h[b_row*nb+b_col],
							    STARPU_MAIN_RAM,
							    (uintptr_t) X->blocks[b_row*nb+b_col].c, BS, BS, BS,
							    sizeof(double));
			}
			else
			{
				starpu_matrix_data_register(&X_h[b_row*nb+b_col],
							    -1, (uintptr_t) NULL, BS, BS, BS,
							    sizeof(double));
			}
//			printf("tag:%d\n",*tag);
			starpu_mpi_data_register(X_h[b_row*nb+b_col], (*tag)++, X->blocks[b_row*nb+b_col].owner);
		}
	}
}

starpu_mpi_tag_t tag = 0;
/* Register the matrix blocks to StarPU and to StarPU-MPI */
static void register_matrices()
{
	if (VERBOSE) printf("Registering matrices\n");
	A_h = calloc(MB*KB, sizeof(starpu_data_handle_t));
	B_h = calloc(KB*NB, sizeof(starpu_data_handle_t));
	C_h = calloc(MB*NB, sizeof(starpu_data_handle_t));

	/* mpi tag used for the block */
	register_matrix(A,A_h,&tag,MB,KB);
	register_matrix(B,B_h,&tag,KB,NB);
	register_matrix(C,C_h,&tag,MB,NB);
}

static void unregister_matrix(struct matrix* X, starpu_data_handle_t* X_h, int mb, int nb)
{
	int b_row,b_col;
	for (b_row = 0; b_row < mb; b_row++)
	{
		for (b_col = 0; b_col < nb; b_col++)
		{
			if (X->blocks[b_row*nb+b_col].owner == comm_rank)
 				starpu_data_unregister(X_h[b_row*nb+b_col]);
		}
	}
	free(X_h);
}

/* Unregister matrices from the StarPU management. */
static void unregister_matrices()
{
	if (VERBOSE) printf( "Unregistering matrices\n");
	unregister_matrix(A,A_h,MB,KB);
	unregister_matrix(B,B_h,KB,NB);
	unregister_matrix(C,C_h,MB,NB);
}

static void cpu_mult(void *handles[], void *arg)
{
	(void)arg;
	double *block_A = (double *)STARPU_MATRIX_GET_PTR(handles[0]);
	double *block_B = (double *)STARPU_MATRIX_GET_PTR(handles[1]);
	double *block_C = (double *)STARPU_MATRIX_GET_PTR(handles[2]);

	unsigned n_col_A = STARPU_MATRIX_GET_NX(handles[0]);
	unsigned n_col_C = STARPU_MATRIX_GET_NX(handles[2]);
	unsigned n_row_C = STARPU_MATRIX_GET_NY(handles[2]);

	unsigned ld_A = STARPU_MATRIX_GET_LD(handles[0]);
	unsigned ld_B = STARPU_MATRIX_GET_LD(handles[1]);
	unsigned ld_C = STARPU_MATRIX_GET_LD(handles[2]);

	if (VERBOSE) printf("gemm_task\n");
	STARPU_DGEMM("N", "N", n_row_C,n_col_C,n_col_A,
		     1.0, block_A, ld_A, block_B, ld_B,
		     1.0, block_C, ld_C);
}

static void cpu_fill(void *handles[], void *arg)
{
	(void)arg;
	double *block_A = (double *)STARPU_MATRIX_GET_PTR(handles[0]);

	unsigned n_col_A = STARPU_MATRIX_GET_NX(handles[0]);
	unsigned n_row_A = STARPU_MATRIX_GET_NY(handles[0]);

	unsigned i,j;
	if (VERBOSE) printf("fill_task\n");
	for (i=0;i<n_row_A;i++)
	{
    		for (j=0;j<n_col_A;j++)
		{
			block_A[i*BS+j] = 1.1;
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
	.modes = {STARPU_R, STARPU_R, STARPU_RW}, /* access modes for each data handle */
	.name = "gemm" /* to display task name in traces */
};

static struct starpu_codelet fill_cl =
{
	.cpu_funcs = {cpu_fill}, /* cpu implementation(s) of the routine */
	.nbuffers = 1, /* number of data handles referenced by this routine */
	.modes = {STARPU_W},
	.name = "fill" /* to display task name in traces */
};

static void init_matrix(struct matrix* X, starpu_data_handle_t* X_h, int mb, int nb)
{
	int row, col;
	for (row = 0; row < mb; row++)
	{
		for (col = 0; col < nb; col++)
		{
			if (X->blocks[row*nb+col].owner == comm_rank)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, &fill_cl,
					STARPU_W, X_h[row*nb+col], 0);
			}
		}
	}
}

static void init_matrices(void)
{
	if (VERBOSE) printf( "Initializing matrices\n");
	// I own all the blocks
	init_matrix(A,A_h,MB,KB);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	init_matrix(B,B_h,KB,NB);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	init_matrix(C,C_h,MB,NB);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}


int main(int argc, char *argv[])
{
	/* Initializes StarPU and the StarPU-MPI layer */
	starpu_fxt_autostart_profiling(0);
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	/* Get the process rank and session size */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank == 0) printf("Launching with %d arguments\n",argc);

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? STARPU_TEST_SKIPPED : 0;
	}

	/* Parse the matrix size and block size optional args */
 	// M, N, K, B, P, Q
	if (argc < 8)
	{
		if (comm_rank == 0) fprintf(stderr, "using default sizes for arguments\n");
	}
	else
	{
		M  = atoi(argv[1]);
		N  = atoi(argv[2]);
		K  = atoi(argv[3]);
		BS = atoi(argv[4]);
		P  = atoi(argv[5]);
		Q  = atoi(argv[6]);
		T  = atoi(argv[7]);
	}

	if (BS < 1 || M % BS != 0)
	{
		if (comm_rank == 0) fprintf(stderr, "invalid block size\n");
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? 1 : 0;
	}
	if (BS < 1 || N % BS != 0)
	{
		if (comm_rank == 0) fprintf(stderr, "invalid block size\n");
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? 1 : 0;
	}
	if (BS < 1 || K % BS != 0)
	{
		if (comm_rank == 0) fprintf(stderr, "invalid block size\n");
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? 1 : 0;
	}
	if (argc > 9)
	{
		if (comm_rank == 0) fprintf(stderr, "invalid argument size (reuqire 8 arguments, 9 if tracing ; given %d)\n",argc);
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? 1 : 0;
	}
	else if (argc == 9)
	{
		trace = 1;
	}
	if (P < 1 || Q < 1 || P*Q != comm_size)
	{
		fprintf(stderr, "invalid grid size\n");
		starpu_mpi_shutdown();
		return (comm_rank == 0) ? 1 : 0;
	}

	if (comm_rank == 0)
	{
		printf("MxNxK  = %dx%dx%d\n", M, N, K);
		printf("BS     = %d\n", BS);
		printf("MxNxKb = %dx%dx%d\n", MB,NB,KB);
		printf("comm_size = %d\n", comm_size);
		printf("PxQ = %dx%d\n", P, Q);
        }
  	int trial;
     	double start, stop;
	if (trace) starpu_fxt_start_profiling();
	for (trial =0; trial < T; trial++)
	{
	        alloc_matrices();
		register_matrices();

	        init_matrices();
	        starpu_mpi_barrier(MPI_COMM_WORLD);
		start = starpu_timing_now();

		int b_row,b_col,b_aisle;
		for (b_row = 0; b_row < MB; b_row++)
		{
			for (b_col = 0; b_col < NB; b_col++)
			{
				for (b_aisle=0;b_aisle<KB;b_aisle++)
				{
					starpu_mpi_task_insert(MPI_COMM_WORLD, &gemm_cl,
						STARPU_R,  A_h[b_row*KB+b_aisle],
						STARPU_R,  B_h[b_aisle*NB+b_col],
						STARPU_RW, C_h[b_row*NB+b_col],  0);
				}
			}
			for (b_aisle=0;b_aisle<KB;b_aisle++)
			{
				starpu_mpi_cache_flush(MPI_COMM_WORLD, A_h[b_row*KB+b_aisle]);
			}
		}

		starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	        starpu_mpi_barrier(MPI_COMM_WORLD);
		stop = starpu_timing_now();
		double timing = stop - start;
		if (comm_rank==0) printf("RANK %d -> took %f s | %f Gflop/s\n", comm_rank, timing/1000/1000, 2.0*M*N*K/(timing*1000));

		starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
		unregister_matrices();
		free_matrices();
	}

	if (trace) starpu_fxt_stop_profiling();
	starpu_mpi_shutdown();
	return 0;
}
