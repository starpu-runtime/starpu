/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <limits.h>
#include <common/blas.h>
#include "../../examples/mult/simple.h"
#include "helper.h"
#include "gemm_helper.h"


#define CHECK_TASK_SUBMIT(ret) do {				\
	if (ret == -ENODEV)					\
	{							\
		return -ENODEV;					\
	}							\
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");	\
} while(0)


unsigned nslices = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
unsigned matrix_dim = 256;
#else
unsigned matrix_dim = 320 * 4;
#endif
unsigned check = 0;
int comm_thread_cpuid = -1;

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

static void check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", matrix_dim, matrix_dim, matrix_dim, (TYPE)-1.0f, A, matrix_dim, B, matrix_dim, (TYPE)1.0f, C, matrix_dim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(matrix_dim*matrix_dim, C, 1);

	if (err < matrix_dim*matrix_dim*0.001)
	{
		FPRINTF(stderr, "Results are OK\n");
	}
	else
	{
		int max;
		max = CPU_IAMAX(matrix_dim*matrix_dim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
	}
}


static void partition_mult_data(void)
{
	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
		matrix_dim, matrix_dim, matrix_dim, sizeof(TYPE));

	struct starpu_data_filter vert;
	memset(&vert, 0, sizeof(vert));
	vert.filter_func = starpu_matrix_filter_vertical_block;
	vert.nchildren = nslices;

	struct starpu_data_filter horiz;
	memset(&horiz, 0, sizeof(horiz));
	horiz.filter_func = starpu_matrix_filter_block;
	horiz.nchildren = nslices;

	starpu_data_partition(B_handle, &vert);
	starpu_data_partition(A_handle, &horiz);

	starpu_data_map_filters(C_handle, 2, &vert, &horiz);
}


static void cpu_init_matrix_random(void *descr[], void *arg)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[0]);
	unsigned i = 0;

	for (i = 0; i < nx *ny; i++)
	{
		subA[i] = (TYPE) (starpu_drand48());
		subB[i] = (TYPE) (starpu_drand48());
	}
}


static void cpu_init_matrix_zero(void *descr[], void *arg)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	unsigned nx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned ny = STARPU_MATRIX_GET_NY(descr[0]);
	unsigned i = 0;

	for (i = 0; i < nx *ny; i++)
	{
		subA[i] = (TYPE) (0);
	}
}


static void cpu_mult(void *descr[], void *arg)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	int worker_size = starpu_combined_worker_get_size();

	if (worker_size == 1)
	{
		/* Sequential CPU task */
		CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, (TYPE)0.0, subC, ldC);
	}
	else
	{
		/* Parallel CPU task */
		unsigned rank = starpu_combined_worker_get_rank();

		unsigned block_size = (nyC + worker_size - 1)/worker_size;
		unsigned new_nyC = STARPU_MIN(nyC, block_size*(rank+1)) - block_size*rank;

		STARPU_ASSERT(nyC == STARPU_MATRIX_GET_NY(descr[1]));

		TYPE *new_subB = &subB[block_size*rank];
		TYPE *new_subC = &subC[block_size*rank];

		CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, subA, ldA, new_subB, ldB, (TYPE)0.0, new_subC, ldC);
	}
}

static struct starpu_perfmodel starpu_gemm_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

static struct starpu_codelet cl =
{
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_mult},
	.cpu_funcs_name = {"cpu_mult"},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &starpu_gemm_model
};

static struct starpu_codelet cl_init_matrix_random =
{
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_init_matrix_random},
	.cpu_funcs_name = {"cpu_init_matrix_random"},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_W},
	.name = "init_matrix_random",
	.color = 0xffa500 // orange
};

static struct starpu_codelet cl_init_matrix_zero =
{
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_init_matrix_zero},
	.cpu_funcs_name = {"cpu_init_matrix_zero"},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "init_matrix_zero",
	.color = 0x808000 // olive
};

/* Allocate and partition buffers */
void gemm_alloc_data()
{
	starpu_malloc_flags((void **)&A, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	partition_mult_data();
}

/* Submit tasks to initialize matrices: fill them with zeros or random numbers */
int gemm_init_data()
{
#ifndef STARPU_SIMGRID
	int ret;
	unsigned x, y;

	for (x = 0; x < nslices; x++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl_init_matrix_random;
		task->handles[0] = starpu_data_get_sub_data(A_handle, 1, x);
		task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
		ret = starpu_task_submit(task);
		CHECK_TASK_SUBMIT(ret);

		for (y = 0; y < nslices; y++)
		{
			task = starpu_task_create();
			task->cl = &cl_init_matrix_zero;
			task->handles[0] = starpu_data_get_sub_data(C_handle, 2, x, y);
			ret = starpu_task_submit(task);
			CHECK_TASK_SUBMIT(ret);
		}
	}
#endif
	return 0;
}

/* Submit tasks to compute the GEMM */
int gemm_submit_tasks()
{
	return gemm_submit_tasks_with_tags(/* by default, disable task tags */ 0);
}

int gemm_submit_tasks_with_tags(int with_tags)
{
	int ret;
	unsigned x, y;
	starpu_tag_t task_tag = 0;

	for (x = 0; x < nslices; x++)
	for (y = 0; y < nslices; y++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		task->handles[0] = starpu_data_get_sub_data(A_handle, 1, y);
		task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
		task->handles[2] = starpu_data_get_sub_data(C_handle, 2, x, y);
		task->flops = 2ULL * (matrix_dim/nslices) * (matrix_dim/nslices) * matrix_dim;

		if (with_tags)
		{
			task->use_tag = 1;
			task->tag_id = ++task_tag;
		}

		ret = starpu_task_submit(task);
		CHECK_TASK_SUBMIT(ret);
		starpu_data_wont_use(starpu_data_get_sub_data(C_handle, 2, x, y));
	}

	return 0;
}

/* Add dependencies between GEMM tasks to see the impact of polling workers which will at the end get a task.
 * The new dependency graph has the following shape:
 * - the same number of GEMMs as the number of workers are executed in parallel on all workers ("a column of tasks")
 * - then a GEMM waits all tasks of the previous column of tasks, and is executed on a worker
 * - the next column of tasks waits for the previous GEMM
 * - and so on...
 *
 * worker 0 |  1  |  4  |  5  |  8  |  9  |
 * worker 1 |  2  |     |  6  |     | 10  |  ...
 * worker 2 |  3  |     |  7  |     | 11  |
 *
 * This function has to be called before gemm_submit_tasks_with_tags(1).
 */
void gemm_add_polling_dependencies()
{
	starpu_tag_t nb_tasks = (starpu_tag_t) nslices * (starpu_tag_t) nslices;
	unsigned nb_workers = starpu_worker_get_count();
	starpu_tag_t synchro_tag;
	starpu_tag_t previous_tag;
	starpu_tag_t next_tag;

	for (synchro_tag = nb_workers+1; synchro_tag <= nb_tasks; synchro_tag += (nb_workers+1))
	{
		// this synchro tag depends on tasks of previous column of tasks:
		for (previous_tag = synchro_tag - nb_workers; previous_tag < synchro_tag; previous_tag++)
		{
			starpu_tag_declare_deps(synchro_tag, 1, previous_tag);
		}

		// tasks of the next column of tasks depend on this synchro tag:
		// this actually allows workers to poll for new tasks, while no task is available
		for (next_tag = synchro_tag+1; next_tag < (synchro_tag + nb_workers + 1) && next_tag <= nb_tasks; next_tag++)
		{
			starpu_tag_declare_deps(next_tag, 1, synchro_tag);
		}
	}

}

void gemm_release()
{
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

	if (check)
		check_output();

	starpu_free_flags(A, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, matrix_dim*matrix_dim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
}


