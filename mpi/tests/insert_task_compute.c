/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	FPRINTF(stdout, "[%d] VALUES: %u %u\n", rank, *x, *y);
	*x = *x * *y;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R}
};

int test(int rank, int node, int *before, int *after, int data_array)
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

	FPRINTF_MPI(stderr, "Testing with data_array=%d and node=%d\n", data_array, node);

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
	}

	switch(data_array)
	{
		case 0:
		{
			ret = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet,
						     STARPU_RW, data_handles[0], STARPU_R, data_handles[1],
						     STARPU_VALUE, &rank, sizeof(rank),
						     STARPU_EXECUTE_ON_NODE, node, 0);
			break;
		}
		case 1:
		{
			ret = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet,
						     STARPU_DATA_ARRAY, data_handles, 2,
						     STARPU_VALUE, &rank, sizeof(rank),
						     STARPU_EXECUTE_ON_NODE, node, 0);
			break;
		}
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	starpu_task_wait_for_all();

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
			FPRINTF_MPI(stderr, "after computation x[%d] = %d, should be %d\n", i, x[i], after[rank*2+i]);
		}
		FPRINTF_MPI(stderr, "result is %s\n", ok?"CORRECT":"NOT CORRECT");
	}

nodata:
	starpu_mpi_shutdown();
	starpu_shutdown();

	return ret == -ENODEV ? ret : !ok;
}

int main(int argc, char **argv)
{
	int rank;
	int ret;
	int before[4] = {10, 20, 11, 22};
	int after_node[2][4] = {{220, 20, 11, 22}, {220, 20, 11, 22}};
	int node, insert_task, data_array;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for(node=0 ; node<=1 ; node++)
	{
		for(data_array=0 ; data_array<=1 ; data_array++)
		{
			ret = test(rank, node, before, after_node[node], data_array);
			if (ret == -ENODEV || ret) goto end;
		}
	}

end:
	MPI_Finalize();
	return ret==-ENODEV?STARPU_TEST_SKIPPED:ret;
}
