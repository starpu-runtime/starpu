/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_mpi_ft.h>
#include "helper.h"
#include <mpi_failure_tolerance/ulfm/starpu_mpi_ulfm_comm.h>

#include <mpi.h>

int N  = 512; /* Matrix size */
int BS =  16; /* Block size */

#define NB ((N)/(BS)) /* Number of blocks */

double *A = NULL;
double *B = NULL;
double *C = NULL;

starpu_data_handle_t *A_h;
starpu_data_handle_t *B_h;
starpu_data_handle_t *C_h;

int comm_rank; /* mpi rank of the process */
int comm_size; /* size of the mpi session */

void alloc_matrices(void)
{
	starpu_malloc((void **)&A, N*N*sizeof(double));
	starpu_malloc((void **)&B, N*N*sizeof(double));
	starpu_malloc((void **)&C, N*N*sizeof(double));
}

void free_matrices(void)
{
	starpu_free_noflag(A, N*N*sizeof(double));
	starpu_free_noflag(B, N*N*sizeof(double));
	starpu_free_noflag(C, N*N*sizeof(double));
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

int check_result(void)
{
	int row,col;
	for (row = 0; row < N; row++)
	{
		for (col = 0; col < N; col++)
		{
			if (fabs(C[row*N+col] - 2*(row*N+col)) > 1.0)
			{
				return 1;
			}
		}
	}
	return EXIT_SUCCESS;
}

void register_matrices()
{
	A_h = calloc(NB, sizeof(starpu_data_handle_t));
	B_h = calloc(NB, sizeof(starpu_data_handle_t));
	C_h = calloc(NB*NB, sizeof(starpu_data_handle_t));

	int mr = (comm_rank == 0) ? STARPU_MAIN_RAM : -1;

	starpu_mpi_tag_t tag = 0;

	int b_row,b_col;

	for (b_row = 0; b_row < NB; b_row++)
	{
		starpu_matrix_data_register(&A_h[b_row],
					    mr,
					    (comm_rank == 0)?(uintptr_t)(A+b_row*BS*N):0, N, N, BS,
					    sizeof(double));
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

void distribute_matrix_C(void)
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

void undistribute_matrix_C(void)
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

void unregister_matrices()
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

void cpu_mult(void *handles[], void *arg)
{
	(void)arg;
	double *block_A = (double *)STARPU_MATRIX_GET_PTR(handles[0]);
	double *block_B = (double *)STARPU_MATRIX_GET_PTR(handles[1]);
	double *block_C = (double *)STARPU_MATRIX_GET_PTR(handles[2]);

	size_t n_col_A = STARPU_MATRIX_GET_NX(handles[0]);
	size_t n_col_B = STARPU_MATRIX_GET_NX(handles[1]);
	size_t n_col_C = STARPU_MATRIX_GET_NX(handles[2]);

	size_t n_row_A = STARPU_MATRIX_GET_NY(handles[0]);
	size_t n_row_B = STARPU_MATRIX_GET_NY(handles[1]);
	size_t n_row_C = STARPU_MATRIX_GET_NY(handles[2]);

	size_t ld_A = STARPU_MATRIX_GET_LD(handles[0]);
	size_t ld_B = STARPU_MATRIX_GET_LD(handles[1]);
	size_t ld_C = STARPU_MATRIX_GET_LD(handles[2]);

	unsigned i,j,k;
	for (k = 0; k < n_row_C; k++)
	{
		for (j = 0; j < n_col_C; j++)
		{
			for (i = 0; i < n_col_A; i++)
			{
				block_C[k*ld_C+j] += block_A[k*ld_A+i] * block_B[i*ld_B+j];
			}
		}
	}
}

struct starpu_codelet gemm_cl =
{
	.cpu_funcs = {cpu_mult},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.name = "gemm"
};

int main(int argc, char *argv[])
{
	int ret;
	int mpi_init;
	struct starpu_conf conf;
	int status = 0;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	int level;
	MPI_Query_thread(&level);
	if(level != MPI_THREAD_SERIALIZED)
	{
		if(!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}
	MPI_Comm new_world;
	MPI_Comm_dup(MPI_COMM_WORLD, &new_world);
	MPI_Comm_set_name(new_world, "application duplicated comm");
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conf");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank == 0)
	{
		alloc_matrices();
		init_matrices();
	}

	register_matrices();
	distribute_matrix_C();

	int b_row,b_col;

	for (b_row = 0; b_row < NB/2; b_row++)
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
	//change communicator
	_starpu_mpi_ulfm_comm_test_update(MPI_COMM_WORLD, new_world);
	for (b_row = NB/2; b_row < NB; b_row++)
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
	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_task_wait_for_all();

	undistribute_matrix_C();
	unregister_matrices();
	if (comm_rank == 0)
	{
		status = check_result();
		free_matrices();
	}

	starpu_mpi_shutdown();
enodev:
	if (!mpi_init)
		MPI_Finalize();
	return status;
}
