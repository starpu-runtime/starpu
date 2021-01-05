/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "helper.h"

void func_cpu(void *descr[], void *_args)
{
	int rank;
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	starpu_codelet_unpack_args(_args, &rank);

	FPRINTF(stdout, "[%d] VALUES: %d %d\n", rank, *x, *y);
	*x = *x * *y;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

int test(int rank, int node, int *before, int *after, int task_insert, int data_array)
{
	int ok, ret, i, x[2];
	starpu_data_handle_t data_handles[2];
	struct starpu_data_descr descrs[2];
	int barrier_ret;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	if (starpu_cpu_worker_get_count() == 0)
	{
		// If there is no cpu to execute the codelet, mpi will block trying to do the post-execution communication
		ret = -ENODEV;
		FPRINTF_MPI(stderr, "No CPU is available\n");
		goto nodata;
	}

	FPRINTF_MPI(stderr, "Testing with node=%d - task_insert=%d - data_array=%d - \n", node, task_insert, data_array);

	for(i=0 ; i<2 ; i++)
	{
		if (rank <= 1)
		{
			x[i] = before[rank*2+i];
			FPRINTF_MPI(stderr, "before computation x[%d] = %d\n", i, x[i]);
		}
		else
			x[i] = rank*2+i;
		if (rank == i)
			starpu_variable_data_register(&data_handles[i], 0, (uintptr_t)&x[i], sizeof(int));
		else
			starpu_variable_data_register(&data_handles[i], -1, (uintptr_t)NULL, sizeof(int));
		starpu_mpi_data_register(data_handles[i], i, i);
		descrs[i].handle = data_handles[i];
	}
	descrs[0].mode = STARPU_RW;
	descrs[1].mode = STARPU_R;

	switch(task_insert)
	{
		case 0:
		{
			struct starpu_task *task = NULL;
			switch(data_array)
			{
				case 0:
				{
					task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet,
								     STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 1:
				{
					task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet,
								     STARPU_DATA_ARRAY, data_handles, 2,
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 2:
				{
					task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet,
								     STARPU_DATA_MODE_ARRAY, descrs, 2,
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
			}

			if (task)
			{
				ret = starpu_task_submit(task);
				if (ret == -ENODEV)
					goto enodev;
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			}

			switch(data_array)
			{
				case 0:
				{
					starpu_mpi_task_post_build(MPI_COMM_WORLD, &mycodelet,
								   STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
								   STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 1:
				{
					starpu_mpi_task_post_build(MPI_COMM_WORLD, &mycodelet,
								   STARPU_DATA_ARRAY, data_handles, 2,
								   STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 2:
				{
					starpu_mpi_task_post_build(MPI_COMM_WORLD, &mycodelet,
								   STARPU_DATA_MODE_ARRAY, descrs, 2,
								   STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
			}

			break;
		}
		case 1:
		{
			switch(data_array)
			{
				case 0:
				{
					ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
								     STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 1:
				{
					ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
								     STARPU_DATA_ARRAY, data_handles, 2,
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
				case 2:
				{
					ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
								     STARPU_DATA_MODE_ARRAY, descrs, 2,
								     STARPU_VALUE, &rank, sizeof(rank),
								     STARPU_EXECUTE_ON_NODE, node, 0);
					break;
				}
			}
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
			break;
		}
	}

	starpu_task_wait_for_all();

enodev:
	for(i=0; i<2; i++)
	{
		starpu_data_unregister(data_handles[i]);
	}

	ok = 1;
#ifndef STARPU_SIMGRID
	if (rank <= 1)
	{
		for(i=0; i<2; i++)
		{
			ok = ok && (x[i] == after[rank*2+i]);
			FPRINTF_MPI(stderr, "after computation x[%d] = %d, should be %d\n", i, x[i], after[rank*2+i]);
		}
		FPRINTF_MPI(stderr, "result is %s\n", ok?"CORRECT":"NOT CORRECT");
	}
#endif

nodata:
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
	starpu_mpi_shutdown();

	return ret == -ENODEV ? ret : !ok;
}

int main(int argc, char **argv)
{
	int rank;
	int global_ret, ret;
	int before[4] = {10, 20, 11, 22};
	int after_node[2][4] = {{220, 20, 11, 22}, {220, 20, 11, 22}};
	int node, insert_task, data_array;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	global_ret = 0;
	for(node=0 ; node<=1 ; node++)
	{
		for(insert_task=0 ; insert_task<=1 ; insert_task++)
		{
			for(data_array=0 ; data_array<=2 ; data_array++)
			{
				ret = test(rank, node, before, after_node[node], insert_task, data_array);
				if (ret == -ENODEV || ret)
					global_ret = ret;
			}
		}
	}

	MPI_Finalize();
	return global_ret==-ENODEV?STARPU_TEST_SKIPPED:global_ret;
}
