/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013, 2014  Centre National de la Recherche Scientifique
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

void func_cpu(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	FPRINTF(stdout, "VALUES: %u %u\n", *x, *y);
	*x = *x * *y;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu, NULL},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R}
};

int test(int rank, int node, int *before, int *after, int task_insert)
{
	int ok, ret, i, x[2];
	starpu_data_handle_t data_handles[2];

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (starpu_cpu_worker_get_count() <= 0)
	{
		// If there is no cpu to execute the codelet, mpi will block trying to do the post-execution communication
		ret = -ENODEV;
		goto nodata;
	}

	for(i=0 ; i<2 ; i++)
	{
		x[i] = before[rank*2+i];
		if (rank <= 1)
			FPRINTF_MPI("before computation x[%d] = %d\n", i, x[i]);
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(int));
		starpu_mpi_data_register(data_handles[i], i, i);
	}

	if (task_insert)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
					     STARPU_EXECUTE_ON_NODE, node, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}
	else
	{
		struct starpu_task *task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
								 STARPU_EXECUTE_ON_NODE, node, 0);
		if (task)
		{
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
		starpu_mpi_task_post_build(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
					   STARPU_EXECUTE_ON_NODE, node, 0);
	}

	starpu_task_wait_for_all();

enodev:
	for(i=0; i<2; i++)
	{
		starpu_data_unregister(data_handles[i]);
	}

	ok = 1;
	if (rank <= 1)
	{
		for(i=0; i<2; i++)
		{
			ok = ok && (x[i] == after[rank*2+i]);
			FPRINTF_MPI("after computation x[%d] = %d, should be %d\n", i, x[i], after[rank*2+i]);
		}
		FPRINTF_MPI("result is %s\n", ok?"CORRECT":"NOT CORRECT");
	}

nodata:
	starpu_mpi_shutdown();
	starpu_shutdown();

	return ret == -ENODEV ? ret : !ok;
}

int main(int argc, char **argv)
{
	int rank, size;
	int ret;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int before[4] = {10, 20, 11, 22};
	int after_node1[4] = {220, 20, 220, 22};
	int after_node0[4] = {220, 22, 11, 22};

	ret = test(rank, 0, before, after_node0, 1);
	if (ret == -ENODEV || ret) goto end;

	ret = test(rank, 0, before, after_node0, 0);
	if (ret == -ENODEV || ret) goto end;

	ret = test(rank, 1, before, after_node1, 1);
	if (ret == -ENODEV || ret) goto end;

	ret = test(rank, 1, before, after_node1, 0);
	if (ret == -ENODEV || ret) goto end;

end:
	MPI_Finalize();
	return ret==-ENODEV?STARPU_TEST_SKIPPED:ret;
}
