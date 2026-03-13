/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This test focuses on the following algorithm:
 * - A is owned by P0 with value 1.0
 * - P1 becomes an alternative source of A for P2
 * - A is written over by P0 with the value 2.0
 * - A is read by P1 and P2
 * This test asserts that P1 and P2 gets A with value 2.0 i.e.
 * Writing on A has removed alternative sources.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

// This task is used to test the value on P1 and P2
static void cpu_test(void *handles[], void *arg)
{
	(void)arg;
	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
 	STARPU_ASSERT( *tA == 2.0 );
}

static struct starpu_codelet test_cl =
{
	.cpu_funcs = {cpu_test},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

// This task is used to write the tested value on P0
static void cpu_write(void *handles[], void *arg)
{
	(void)arg;
	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
	*tA = 2.0;
}

static struct starpu_codelet write_cl =
{
	.cpu_funcs = {cpu_write},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

int main(int argc, char *argv[])
{
	float A;
	starpu_data_handle_t A_h;
	int comm_rank; /* mpi rank of the process */
	int comm_size; /* size of the mpi session */
	starpu_mpi_tag_t tag = 0;

	/* Initializes StarPU and the StarPU-MPI layer */
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_size < 3)
	{
		FPRINTF(stderr, "We need at least 3 nodes.\n");
		starpu_mpi_shutdown();
		return comm_rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}
	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return comm_rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

 	// - A is owned by P0 with value 1.0
	if (comm_rank == 0)
	{
		starpu_variable_data_register(&A_h, STARPU_MAIN_RAM, (uintptr_t)&A, sizeof(A));
		starpu_data_acquire(A_h, STARPU_W);
		A = 1.0;
		starpu_data_release(A_h);
		FPRINTF(stderr, "[%d] I own A=%f (%p)\n", comm_rank,A,A_h);
	}
	else
	{
		starpu_variable_data_register(&A_h, -1, (uintptr_t) NULL, sizeof(A));
		FPRINTF(stderr, "[%d] I do not own A (%p)\n", comm_rank,A_h);
	}
	starpu_mpi_data_register(A_h, tag++, 0);

 	// - P1 becomes an alternative source of A for P2
	starpu_mpi_data_set_source(A_h, 2, 1);

 	// - A is written over by P0 with the value 2.0
	starpu_mpi_task_insert(MPI_COMM_WORLD, &write_cl,
			       STARPU_W, A_h,
			       0);
 	// - A is read by P1 and P2
	starpu_mpi_task_insert(MPI_COMM_WORLD, &test_cl,
			       STARPU_R, A_h,
			       STARPU_EXECUTE_ON_NODE, 1,
			       0);
	starpu_mpi_task_insert(MPI_COMM_WORLD, &test_cl,
			       STARPU_R, A_h,
			       STARPU_EXECUTE_ON_NODE, 2,
			       0);
	starpu_data_unregister_submit(A_h);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);

	starpu_mpi_shutdown();
	return 0;
}
