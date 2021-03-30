/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <math.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include <common/blas.h>

/*
 * Distributed version of Conjugate Gradient implemented in examples/cg/cg.c
 *
 * Use -display-result option and compare with the non-distributed version: the
 * x vector should be the same.
 */

#include "../../../examples/cg/cg.h"

static int copy_handle(starpu_data_handle_t* dst, starpu_data_handle_t* src, unsigned nblocks);

#define HANDLE_TYPE_VECTOR starpu_data_handle_t*
#define HANDLE_TYPE_MATRIX starpu_data_handle_t**
#define TASK_INSERT(cl, ...) starpu_mpi_task_insert(MPI_COMM_WORLD, cl, ##__VA_ARGS__)
#define GET_VECTOR_BLOCK(v, i) v[i]
#define GET_MATRIX_BLOCK(m, i, j) m[i][j]
#define BARRIER() starpu_mpi_barrier(MPI_COMM_WORLD);
#define GET_DATA_HANDLE(handle) starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle)

static int block_size;

static int rank;
static int nodes_p = 2;
static int nodes_q;

static TYPE ***A;
static TYPE **x;
static TYPE **b;

static TYPE **r;
static TYPE **d;
static TYPE **q;

#define FPRINTF_SERVER(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT") && rank == 0) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#include "../../../examples/cg/cg_kernels.c"

static int my_distrib(const int y, const int x)
{
	return (y%nodes_q)*nodes_p + (x%nodes_p);
}

static int copy_handle(starpu_data_handle_t* dst, starpu_data_handle_t* src, unsigned nblocks)
{
	unsigned b;

	for (b = 0; b < nblocks; b++)
	{
		if (rank == my_distrib(b, 0))
		{
			starpu_data_cpy(dst[b], src[b], /* asynchronous */ 1, /* without callback */ NULL, NULL);
		}
	}

	return 0;
}

/*
 *	Generate Input data
 */
static void generate_random_problem(void)
{
	unsigned nn, mm, m, n, mpi_rank;

	A = malloc(nblocks * sizeof(TYPE **));
	x = malloc(nblocks * sizeof(TYPE *));
	b = malloc(nblocks * sizeof(TYPE *));

	r = malloc(nblocks * sizeof(TYPE *));
	d = malloc(nblocks * sizeof(TYPE *));
	q = malloc(nblocks * sizeof(TYPE *));

	for (m = 0; m < nblocks; m++)
	{
		A[m] = malloc(nblocks * sizeof(TYPE*));

		mpi_rank = my_distrib(m, 0);

		if (mpi_rank == rank || display_result)
		{
			starpu_malloc((void**) &x[m], block_size*sizeof(TYPE));
		}

		if (mpi_rank == rank)
		{
			starpu_malloc((void**) &b[m], block_size*sizeof(TYPE));
			starpu_malloc((void**) &r[m], block_size*sizeof(TYPE));
			starpu_malloc((void**) &d[m], block_size*sizeof(TYPE));
			starpu_malloc((void**) &q[m], block_size*sizeof(TYPE));

			for (mm = 0; mm < block_size; mm++)
			{
				x[m][mm] = (TYPE) 0.0;
				b[m][mm] = (TYPE) 1.0;
				r[m][mm] = (TYPE) 0.0;
				d[m][mm] = (TYPE) 0.0;
				q[m][mm] = (TYPE) 0.0;
			}
		}

		for (n = 0; n < nblocks; n++)
		{
			mpi_rank = my_distrib(m, n);
			if (mpi_rank == rank)
			{
				starpu_malloc((void**) &A[m][n], block_size*block_size*sizeof(TYPE));

				for (nn = 0; nn < block_size; nn++)
				{
					for (mm = 0; mm < block_size; mm++)
					{
						/* We take Hilbert matrix that is not well conditionned but definite positive: H(i,j) = 1/(1+i+j) */
						A[m][n][mm + nn*block_size] = (TYPE) (1.0/(1.0+(nn+(m*block_size)+mm+(n*block_size))));
					}
				}
			}
		}
	}
}

static void free_data(void)
{
	unsigned nn, mm, m, n, mpi_rank;

	for (m = 0; m < nblocks; m++)
	{
		mpi_rank = my_distrib(m, 0);

		if (mpi_rank == rank || display_result)
		{
			starpu_free((void*) x[m]);
		}

		if (mpi_rank == rank)
		{
			starpu_free((void*) b[m]);
			starpu_free((void*) r[m]);
			starpu_free((void*) d[m]);
			starpu_free((void*) q[m]);
		}

		for (n = 0; n < nblocks; n++)
		{
			mpi_rank = my_distrib(m, n);
			if (mpi_rank == rank)
			{
				starpu_free((void*) A[m][n]);
			}
		}

		free(A[m]);
	}

	free(A);
	free(x);
	free(b);
	free(r);
	free(d);
	free(q);
}

static void register_data(void)
{
	unsigned m, n;
	int mpi_rank;
	starpu_mpi_tag_t mpi_tag = 0;

	A_handle = malloc(nblocks*sizeof(starpu_data_handle_t*));
	x_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	b_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	r_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	d_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	q_handle = malloc(nblocks*sizeof(starpu_data_handle_t));

	for (m = 0; m < nblocks; m++)
	{
		mpi_rank = my_distrib(m, 0);
		A_handle[m] = malloc(nblocks*sizeof(starpu_data_handle_t));

		if (mpi_rank == rank || display_result)
		{
			starpu_vector_data_register(&x_handle[m], STARPU_MAIN_RAM, (uintptr_t) x[m], block_size, sizeof(TYPE));
		}
		else if (!display_result)
		{
			assert(mpi_rank != rank);
			starpu_vector_data_register(&x_handle[m], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
		}

		if (mpi_rank == rank)
		{
			starpu_vector_data_register(&b_handle[m], STARPU_MAIN_RAM, (uintptr_t) b[m], block_size, sizeof(TYPE));
			starpu_vector_data_register(&r_handle[m], STARPU_MAIN_RAM, (uintptr_t) r[m], block_size, sizeof(TYPE));
			starpu_vector_data_register(&d_handle[m], STARPU_MAIN_RAM, (uintptr_t) d[m], block_size, sizeof(TYPE));
			starpu_vector_data_register(&q_handle[m], STARPU_MAIN_RAM, (uintptr_t) q[m], block_size, sizeof(TYPE));
		}
		else
		{
			starpu_vector_data_register(&b_handle[m], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&r_handle[m], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&d_handle[m], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&q_handle[m], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
		}

		starpu_data_set_coordinates(x_handle[m], 1, m);
		starpu_mpi_data_register(x_handle[m], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(b_handle[m], 1, m);
		starpu_mpi_data_register(b_handle[m], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(r_handle[m], 1, m);
		starpu_mpi_data_register(r_handle[m], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(d_handle[m], 1, m);
		starpu_mpi_data_register(d_handle[m], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(q_handle[m], 1, m);
		starpu_mpi_data_register(q_handle[m], ++mpi_tag, mpi_rank);

		if (use_reduction)
		{
			starpu_data_set_reduction_methods(q_handle[m], &accumulate_vector_cl, &bzero_vector_cl);
			starpu_data_set_reduction_methods(r_handle[m], &accumulate_vector_cl, &bzero_vector_cl);
		}

		for (n = 0; n < nblocks; n++)
		{
			mpi_rank = my_distrib(m, n);

			if (mpi_rank == rank)
			{
				starpu_matrix_data_register(&A_handle[m][n], STARPU_MAIN_RAM, (uintptr_t) A[m][n], block_size, block_size, block_size, sizeof(TYPE));
			}
			else
			{
				starpu_matrix_data_register(&A_handle[m][n], -1, (uintptr_t) NULL, block_size, block_size, block_size, sizeof(TYPE));
			}

			starpu_data_set_coordinates(A_handle[m][n], 2, n, m);
			starpu_mpi_data_register(A_handle[m][n], ++mpi_tag, mpi_rank);
		}
	}

	starpu_variable_data_register(&dtq_handle, STARPU_MAIN_RAM, (uintptr_t)&dtq, sizeof(TYPE));
	starpu_variable_data_register(&rtr_handle, STARPU_MAIN_RAM, (uintptr_t)&rtr, sizeof(TYPE));
	starpu_mpi_data_register(rtr_handle, ++mpi_tag, 0);
	starpu_mpi_data_register(dtq_handle, ++mpi_tag, 0);

	if (use_reduction)
	{
		starpu_data_set_reduction_methods(dtq_handle, &accumulate_variable_cl, &bzero_variable_cl);
		starpu_data_set_reduction_methods(rtr_handle, &accumulate_variable_cl, &bzero_variable_cl);
	}
}

static void unregister_data(void)
{
	unsigned m, n;

	for (m = 0; m < nblocks; m++)
	{
		starpu_data_unregister(x_handle[m]);
		starpu_data_unregister(b_handle[m]);
		starpu_data_unregister(r_handle[m]);
		starpu_data_unregister(d_handle[m]);
		starpu_data_unregister(q_handle[m]);

		for (n = 0; n < nblocks; n++)
		{
			starpu_data_unregister(A_handle[m][n]);
		}

		free(A_handle[m]);
	}

	starpu_data_unregister(dtq_handle);
	starpu_data_unregister(rtr_handle);

	free(A_handle);
	free(x_handle);
	free(b_handle);
	free(r_handle);
	free(d_handle);
	free(q_handle);
}

static void display_x_result(void)
{
	int j, i;

	for (j = 0; j < nblocks; j++)
	{
		starpu_mpi_get_data_on_node(MPI_COMM_WORLD, x_handle[j], 0);
	}

	if (rank == 0)
	{
		FPRINTF_SERVER(stderr, "Computed X vector:\n");
		for (j = 0; j < nblocks; j++)
		{
			starpu_data_acquire(x_handle[j], STARPU_R);
			for (i = 0; i < block_size; i++)
			{
				FPRINTF(stderr, "% 02.2e\n", x[j][i]);
			}
			starpu_data_release(x_handle[j]);
		}
	}
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-p") == 0)
		{
			nodes_p = atoi(argv[++i]);
			continue;
		}

		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0)
		{
			FPRINTF_SERVER(stderr, "usage: %s [-h] [-nblocks #blocks] [-display-result] [-p node_grid_width] [-n problem_size] [-no-reduction] [-maxiter i]\n", argv[0]);
			exit(-1);
		}
	}

	parse_common_args(argc, argv);
}

int main(int argc, char **argv)
{
	int worldsize, ret;
	double start, end;

	/* Not supported yet */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return 77;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	parse_args(argc, argv);

	if (worldsize % nodes_p != 0)
	{
		FPRINTF_SERVER(stderr, "Node grid (%d) width must divide the number of nodes (%d).\n", nodes_p, worldsize);
		starpu_mpi_shutdown();
		return 1;
	}
	nodes_q = worldsize / nodes_p;

	if (n % nblocks != 0)
	{
		FPRINTF_SERVER(stderr, "The number of blocks (%d) must divide the matrix size (%lld).\n", nblocks, n);
		starpu_mpi_shutdown();
		return 1;
	}
	block_size = n / nblocks;

	starpu_cublas_init();

	FPRINTF_SERVER(stderr, "************** PARAMETERS ***************\n");
	FPRINTF_SERVER(stderr, "%d nodes (%dx%d)\n", worldsize, nodes_p, nodes_q);
	FPRINTF_SERVER(stderr, "Problem size (-n): %lld\n", n);
	FPRINTF_SERVER(stderr, "Maximum number of iterations (-maxiter): %d\n", i_max);
	FPRINTF_SERVER(stderr, "Number of blocks (-nblocks): %d\n", nblocks);
	FPRINTF_SERVER(stderr, "Reduction (-no-reduction): %s\n", use_reduction ? "enabled" : "disabled");

	starpu_mpi_barrier(MPI_COMM_WORLD);
	start = starpu_timing_now();
	generate_random_problem();
	register_data();
	starpu_mpi_barrier(MPI_COMM_WORLD);
	end = starpu_timing_now();

	FPRINTF_SERVER(stderr, "Problem initialization timing : %2.2f seconds\n", (end-start)/1e6);

	ret = cg();
	if (ret == -ENODEV)
	{
		ret = 77;
		goto enodev;
	}

	starpu_task_wait_for_all();

	if (display_result)
	{
		display_x_result();
	}

enodev:
	unregister_data();
	free_data();
	starpu_cublas_shutdown();
	starpu_mpi_shutdown();
	return ret;
}
