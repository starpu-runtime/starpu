/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static unsigned block_size;

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

static int my_distrib(const int yy, const int xx)
{
	return (yy%nodes_q)*nodes_p + (xx%nodes_p);
}

static int copy_handle(starpu_data_handle_t* dst, starpu_data_handle_t* src, unsigned nb)
{
	unsigned block;

	for (block = 0; block < nb; block++)
	{
		if (rank == my_distrib(block, 0))
		{
			starpu_data_cpy(dst[block], src[block], /* asynchronous */ 1, /* without callback */ NULL, NULL);
		}
	}

	return 0;
}

/*
 *	Generate Input data
 */
static void generate_random_problem(void)
{
	unsigned ii, jj, j, i;
	int mpi_rank;

	A = malloc(nblocks * sizeof(TYPE **));
	x = malloc(nblocks * sizeof(TYPE *));
	b = malloc(nblocks * sizeof(TYPE *));

	r = malloc(nblocks * sizeof(TYPE *));
	d = malloc(nblocks * sizeof(TYPE *));
	q = malloc(nblocks * sizeof(TYPE *));

	for (j = 0; j < nblocks; j++)
	{
		A[j] = malloc(nblocks * sizeof(TYPE*));

		mpi_rank = my_distrib(j, 0);

		if (mpi_rank == rank || display_result)
		{
			starpu_malloc((void**) &x[j], block_size*sizeof(TYPE));
		}

		if (mpi_rank == rank)
		{
			starpu_malloc((void**) &b[j], block_size*sizeof(TYPE));
			starpu_malloc((void**) &r[j], block_size*sizeof(TYPE));
			starpu_malloc((void**) &d[j], block_size*sizeof(TYPE));
			starpu_malloc((void**) &q[j], block_size*sizeof(TYPE));

			for (jj = 0; jj < block_size; jj++)
			{
				x[j][jj] = (TYPE) 0.0;
				b[j][jj] = (TYPE) 1.0;
				r[j][jj] = (TYPE) 0.0;
				d[j][jj] = (TYPE) 0.0;
				q[j][jj] = (TYPE) 0.0;
			}
		}

		for (i = 0; i < nblocks; i++)
		{
			mpi_rank = my_distrib(j, i);
			if (mpi_rank == rank)
			{
				starpu_malloc((void**) &A[j][i], block_size*block_size*sizeof(TYPE));

				for (ii = 0; ii < block_size; ii++)
				{
					for (jj = 0; jj < block_size; jj++)
					{
						/* We take Hilbert matrix that is not well conditioned but definite positive: H(i,j) = 1/(1+i+j) */
						A[j][i][jj + ii*block_size] = (TYPE) (1.0/(1.0+(ii+(j*block_size)+jj+(i*block_size))));
					}
				}
			}
		}
	}
}

static void free_data(void)
{
	unsigned j, i;
	int mpi_rank;

	for (j = 0; j < nblocks; j++)
	{
		mpi_rank = my_distrib(j, 0);

		if (mpi_rank == rank || display_result)
		{
			starpu_free_noflag((void*) x[j], block_size*sizeof(TYPE));
		}

		if (mpi_rank == rank)
		{
			starpu_free_noflag((void*) b[j], block_size*sizeof(TYPE));
			starpu_free_noflag((void*) r[j], block_size*sizeof(TYPE));
			starpu_free_noflag((void*) d[j], block_size*sizeof(TYPE));
			starpu_free_noflag((void*) q[j], block_size*sizeof(TYPE));
		}

		for (i = 0; i < nblocks; i++)
		{
			mpi_rank = my_distrib(j, i);
			if (mpi_rank == rank)
			{
				starpu_free_noflag((void*) A[j][i], block_size*block_size*sizeof(TYPE));
			}
		}

		free(A[j]);
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
	unsigned j, i;
	int mpi_rank;
	starpu_mpi_tag_t mpi_tag = 0;

	A_handle = malloc(nblocks*sizeof(starpu_data_handle_t*));
	x_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	b_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	r_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	d_handle = malloc(nblocks*sizeof(starpu_data_handle_t));
	q_handle = malloc(nblocks*sizeof(starpu_data_handle_t));

	for (j = 0; j < nblocks; j++)
	{
		mpi_rank = my_distrib(j, 0);
		A_handle[j] = malloc(nblocks*sizeof(starpu_data_handle_t));

		if (mpi_rank == rank || display_result)
		{
			starpu_vector_data_register(&x_handle[j], STARPU_MAIN_RAM, (uintptr_t) x[j], block_size, sizeof(TYPE));
		}
		else if (!display_result)
		{
			assert(mpi_rank != rank);
			starpu_vector_data_register(&x_handle[j], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
		}

		if (mpi_rank == rank)
		{
			starpu_vector_data_register(&b_handle[j], STARPU_MAIN_RAM, (uintptr_t) b[j], block_size, sizeof(TYPE));
			starpu_vector_data_register(&r_handle[j], STARPU_MAIN_RAM, (uintptr_t) r[j], block_size, sizeof(TYPE));
			starpu_vector_data_register(&d_handle[j], STARPU_MAIN_RAM, (uintptr_t) d[j], block_size, sizeof(TYPE));
			starpu_vector_data_register(&q_handle[j], STARPU_MAIN_RAM, (uintptr_t) q[j], block_size, sizeof(TYPE));
		}
		else
		{
			starpu_vector_data_register(&b_handle[j], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&r_handle[j], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&d_handle[j], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
			starpu_vector_data_register(&q_handle[j], -1, (uintptr_t) NULL, block_size, sizeof(TYPE));
		}

		starpu_data_set_coordinates(x_handle[j], 1, j);
		starpu_mpi_data_register(x_handle[j], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(b_handle[j], 1, j);
		starpu_mpi_data_register(b_handle[j], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(r_handle[j], 1, j);
		starpu_mpi_data_register(r_handle[j], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(d_handle[j], 1, j);
		starpu_mpi_data_register(d_handle[j], ++mpi_tag, mpi_rank);
		starpu_data_set_coordinates(q_handle[j], 1, j);
		starpu_mpi_data_register(q_handle[j], ++mpi_tag, mpi_rank);

		if (use_reduction)
		{
			starpu_data_set_reduction_methods(q_handle[j], &accumulate_vector_cl, &bzero_vector_cl);
			starpu_data_set_reduction_methods(r_handle[j], &accumulate_vector_cl, &bzero_vector_cl);
		}

		for (i = 0; i < nblocks; i++)
		{
			mpi_rank = my_distrib(j, i);

			if (mpi_rank == rank)
			{
				starpu_matrix_data_register(&A_handle[j][i], STARPU_MAIN_RAM, (uintptr_t) A[j][i], block_size, block_size, block_size, sizeof(TYPE));
			}
			else
			{
				starpu_matrix_data_register(&A_handle[j][i], -1, (uintptr_t) NULL, block_size, block_size, block_size, sizeof(TYPE));
			}

			starpu_data_set_coordinates(A_handle[j][i], 2, i, j);
			starpu_mpi_data_register(A_handle[j][i], ++mpi_tag, mpi_rank);
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
	unsigned j, i;

	for (j = 0; j < nblocks; j++)
	{
		starpu_data_unregister(x_handle[j]);
		starpu_data_unregister(b_handle[j]);
		starpu_data_unregister(r_handle[j]);
		starpu_data_unregister(d_handle[j]);
		starpu_data_unregister(q_handle[j]);

		for (i = 0; i < nblocks; i++)
		{
			starpu_data_unregister(A_handle[j][i]);
		}

		free(A_handle[j]);
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
	unsigned j, i;

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
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
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
		FPRINTF_SERVER(stderr, "The number of blocks (%u) must divide the matrix size (%lld).\n", nblocks, n);
		starpu_mpi_shutdown();
		return 1;
	}
	block_size = n / nblocks;

	starpu_cublas_init();

	FPRINTF_SERVER(stderr, "************** PARAMETERS ***************\n");
	FPRINTF_SERVER(stderr, "%d nodes (%dx%d)\n", worldsize, nodes_p, nodes_q);
	FPRINTF_SERVER(stderr, "Problem size (-n): %lld\n", n);
	FPRINTF_SERVER(stderr, "Maximum number of iterations (-maxiter): %d\n", i_max);
	FPRINTF_SERVER(stderr, "Number of blocks (-nblocks): %u\n", nblocks);
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
