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
 * - A is read by P2
 * - P1 becomes an alternative source of A for P2
 * - A is read by P2
 * This test asserts that either:
 * - P2 receives the value from P1 as setting the
 *   source flushes the cache
 * - P2 do not receive the value as P1 knows from P0
 *   that P2 has a correct value of A.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

// This task is used to test the value on P1 and P2
static void cpu_read(void *handles[], void *arg)
{
	(void)arg;
	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
}

static struct starpu_codelet read_cl =
{
	.cpu_funcs = {cpu_read},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(int argc, char *argv[])
{
	starpu_data_handle_t A_h;
	float A;
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
		FPRINTF(stderr, "[%d] I own A=%f (%p)\n", comm_rank, A, A_h);
	}
	else
	{
		starpu_variable_data_register(&A_h, -1, (uintptr_t) NULL, sizeof(A));
		FPRINTF(stderr, "[%d] I do not own A (%p)\n", comm_rank, A_h);
	}
	starpu_mpi_data_register(A_h, tag++, 0);

 	// - A is read by P2
	starpu_mpi_task_insert(MPI_COMM_WORLD, &read_cl,
			       STARPU_R, A_h,
			       STARPU_EXECUTE_ON_NODE, 2,
			       0);
 	// - P1 becomes an alternative source of A for P2
	starpu_mpi_data_set_source(A_h, 2, 1);
 	// - A is read by P2
	starpu_mpi_task_insert(MPI_COMM_WORLD, &read_cl,
			       STARPU_R, A_h,
			       STARPU_EXECUTE_ON_NODE, 2,
			       0);
	starpu_data_unregister_submit(A_h);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);

	starpu_mpi_shutdown();
	return 0;
}
