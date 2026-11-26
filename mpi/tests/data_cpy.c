/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define N 10

void setvalue_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int tid, delay;

	starpu_codelet_unpack_args(_args, &tid, &delay);

	if (delay)
	{
		usleep(10000 * (N-tid)); /* make sure tasks complete out-of-order */
	}

	*x0 = tid + 1;
}

struct starpu_codelet setvalue =
{
	.modes = {STARPU_W},
	.cpu_funcs = {setvalue_cpu},
	.cpu_funcs_name = {"setvalue"},
	.nbuffers = 1
};

void checkvalue_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int tid;

	starpu_codelet_unpack_args(_args, &tid);
	if (*x0 == tid+1)
	{
		FPRINTF_MPI(stderr, "in checkvalue_cpu: expected %d, value is %d\n", tid+1, *x0);
	}
	else
	{
		FPRINTF_MPI(stderr, "iN CHECKVALUE_CPU: EXPECTED %d, VALUE IS %d\n", tid+1, *x0);
	}
	STARPU_ASSERT_MSG(*x0 == tid+1, "received value %d should be %d\n", *x0, tid+1);
}

struct starpu_codelet checkvalue =
{
	.modes = {STARPU_R},
	.cpu_funcs = {checkvalue_cpu},
	.cpu_funcs_name = {"checkvalue"},
	.nbuffers = 1
};

void callback(void *arg)
{
	FPRINTF_MPI(stderr, "value in callback: %d\n", *((int *)arg));
}

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	int value;
	struct starpu_conf conf;
	starpu_data_handle_t src_handle, dst_handle;
	int zero = 0;
	int one  = 1;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	/* Do a synchronous mpi_data_cpy from Rank 0 to Rank 1, test that Rank 1 gets Rank 0 value */
	FPRINTF_MPI(stderr, "Check synchronous mpi_data_cpy\n");
	{
		value = rank;
		starpu_variable_data_register(&src_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(src_handle, 12, 0);
		starpu_variable_data_register(&dst_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(dst_handle, 42, 1);

		FPRINTF_MPI(stderr, "value before copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == rank, "before copy value %d should be %d\n", value, rank);
		starpu_mpi_data_cpy(dst_handle, src_handle, MPI_COMM_WORLD, 0, callback, &value);
		starpu_data_unregister(src_handle);
		starpu_data_unregister(dst_handle);

		FPRINTF_MPI(stderr, "value after copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == 0, "after copy value %d should be %d\n", value, 0);
	}

	/* Do an asynchronous mpi_data_cpy from Rank 0 to Rank 1, test that Rank 1 gets Rank 0 value */
	FPRINTF_MPI(stderr, "Check asynchronous mpi_data_cpy\n");
	{
		value = rank+12;
		starpu_variable_data_register(&src_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(src_handle, 12, 0);
		starpu_variable_data_register(&dst_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(dst_handle, 42, 1);

		FPRINTF_MPI(stderr, "value before copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == rank+12, "before copy value %d should be %d\n", value, rank+12);
		starpu_mpi_data_cpy(dst_handle, src_handle, MPI_COMM_WORLD, 1, callback, &value);
		starpu_data_unregister(src_handle);
		starpu_data_unregister(dst_handle);

		FPRINTF_MPI(stderr, "value after copy: %d\n", value);
		if (rank == 1) STARPU_ASSERT_MSG(value == 12, "after copy value %d should be %d\n", value, 12);
	}

	/* Test if send order is respected: have Rank 0 do many sends whose data become available out-of-order, and
	 * check that Rank 1 receives them in-order. */
	for(int prune = 0; prune < 2; prune++)
	{
		int values[N];
		starpu_data_handle_t src_handles[N];
		int i;

		starpu_mpi_wait_for_all(MPI_COMM_WORLD);
		FPRINTF_MPI(stderr, "Check send order in asynchronous mpi_data_cpy (%s)\n", prune ? "w/ pruning" : "w/o pruning");

		for(i = 0; i < N; i++)
		{
			starpu_variable_data_register(&src_handles[i], STARPU_MAIN_RAM, (uintptr_t)&values[i], sizeof(values[i]));
			starpu_mpi_data_register(src_handles[i], 12+i, 0);
		}
		value = 0;
		starpu_variable_data_register(&dst_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(dst_handle, 11, 1);

		for(i = 0; i < N; i++)
		{
			if(0 == prune)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &setvalue,
					               STARPU_W, src_handles[i],
						       STARPU_VALUE, &i, sizeof(i),
						       STARPU_VALUE, &one, sizeof(one),
						       0);
			}
			else if (0 == rank)
			{
				starpu_task_insert(&setvalue,
						   STARPU_W, src_handles[i],
						   STARPU_VALUE, &i, sizeof(i),
						   0);
			}
			starpu_mpi_data_cpy(dst_handle, src_handles[i], MPI_COMM_WORLD, 1, NULL, NULL);
			if (0 == prune)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &checkvalue,
						       STARPU_R, dst_handle,
						       STARPU_VALUE, &i, sizeof(i),
						       0);
			}
			else if(1 == rank)
			{
				starpu_task_insert(&checkvalue,
						   STARPU_R, dst_handle,
						   STARPU_VALUE, &i, sizeof(i),
						   0);
			}
		}

		starpu_task_wait_for_all();

		for (i = 0; i < N; i++)
			starpu_data_unregister(src_handles[i]);
		starpu_data_unregister(dst_handle);
	}

	/* Test if recv order is respected: have Rank 0 do many recvs whose data become available out-of-order, and
	 * check that Rank 1 sends them in-order. */
	for(int prune = 0; prune < 2; prune++)
	{
		int values[N];
		starpu_data_handle_t dst_handles[N];
		int i;

		starpu_mpi_wait_for_all(MPI_COMM_WORLD);
		FPRINTF_MPI(stderr, "Check recv order in asynchronous mpi_data_cpy (%s)\n", prune ? "w/ pruning" : "w/o pruning");

		for (i = 0; i < N; i++)
		{
			starpu_variable_data_register(&dst_handles[i], STARPU_MAIN_RAM, (uintptr_t)&values[i], sizeof(values[i]));
			starpu_mpi_data_register(dst_handles[i], 12+i, 0);
		}
		value = 0;
		starpu_variable_data_register(&src_handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));
		starpu_mpi_data_register(src_handle, 11, 1);

		for(i = 0; i < N; i++)
		{
			if(0 == prune)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &setvalue,
					               STARPU_W, src_handle,
						       STARPU_VALUE, &i,sizeof(i),
						       STARPU_VALUE, &zero, sizeof(zero),
						       0);
				starpu_mpi_insert_task(MPI_COMM_WORLD, &setvalue,
					               STARPU_W, dst_handles[i],
						       STARPU_VALUE, &zero, sizeof(zero),
						       STARPU_VALUE, &one, sizeof(one),
						       0);
			}
			else if(0 == rank)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &setvalue,
						       STARPU_W, dst_handles[i],
						       STARPU_VALUE, &zero, sizeof(zero),
						       STARPU_VALUE, &one, sizeof(one),
						       0);
			}
			else if(1 == rank)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &setvalue,
						       STARPU_W, src_handle,
						       STARPU_VALUE, &i,sizeof(i),
							       STARPU_VALUE, &zero, sizeof(zero),
						       0);
			}
			starpu_mpi_data_cpy(dst_handles[i], src_handle, MPI_COMM_WORLD, 1, NULL, NULL);
			if(0 == prune)
			{
				starpu_mpi_insert_task(MPI_COMM_WORLD, &checkvalue,
						       STARPU_R, dst_handles[i],
						       STARPU_VALUE, &i, sizeof(i),
						       0);
			}
			else if(0 == rank)
			{
				starpu_task_insert(&checkvalue,
						   STARPU_R, dst_handles[i],
						   STARPU_VALUE, &i, sizeof(i),
						   0);
			}
		}

		starpu_task_wait_for_all();

		for (i = 0; i < N; i++)
			starpu_data_unregister(dst_handles[i]);
		starpu_data_unregister(src_handle);
	}

	starpu_mpi_shutdown();

enodev:
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
