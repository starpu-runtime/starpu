/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_mpi.h>
#include <math.h>
#include "helper.h"
#include "../../tests/vector/memset.h"

#define SIZE (1<<20)
#define NPARTS 16

/* This tests the combination of MPI and partitioning */

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	int rank, size;
	void *ptr = NULL;
	starpu_data_handle_t handle, handles[NPARTS];
	int mpi_init, ret, i;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size < 2)
	{
		FPRINTF(stderr, "We need at least 2 processes.\n");
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;
	conf.nopencl = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	/* Initial data on rank 0 */
	if (rank == 0)
		ptr = calloc(SIZE, 1);
	starpu_vector_data_register(&handle, rank == 0 ? STARPU_MAIN_RAM : -1, rank == 0 ? (uintptr_t) ptr : 0, SIZE, 1);
	starpu_mpi_data_register(handle, 0, 0);

	/* Migrate it to 1 */
	starpu_mpi_data_migrate(MPI_COMM_WORLD, handle, 1);

	/* Partition it there */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NPARTS,
	};
	starpu_data_partition_plan(handle, &f, handles);
	for (i = 0; i < NPARTS; i++)
		starpu_mpi_data_register(handles[i], i+1, 1);

	/* Initialize the pieces */
	for (i = 0; i < NPARTS; i++)
		if (size >= 3 && i % 2)
			/* Just for fun */
			starpu_mpi_task_insert(MPI_COMM_WORLD, &memset_cl, STARPU_W, handles[i], STARPU_EXECUTE_ON_NODE, 2, 0);
		else
			starpu_mpi_task_insert(MPI_COMM_WORLD, &memset_cl, STARPU_W, handles[i], 0);

	/* Check it from somewhere else */
	starpu_mpi_task_insert(MPI_COMM_WORLD, &memset_check_content_cl, STARPU_R, handle, STARPU_EXECUTE_ON_NODE, 0, 0);

	/* Clean up */
	starpu_data_partition_clean(handle, NPARTS, handles);
	starpu_data_unregister(handle);
	if (rank == 0)
		free(ptr);

	starpu_mpi_shutdown();

enodev:
	MPI_Finalize();
	return 0;
}
