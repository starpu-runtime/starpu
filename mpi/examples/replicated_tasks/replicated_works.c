/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example illustrates how to replicate the same task across different nodes
 * to be able to manipulate the same value across them.
 * Node 0 owns A. All the other nodes replicate computation using A.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

/* The replicated work on each node. Here, we simply multiply A by 2.
 */
static void cpu_repl_work(void *handles[], void *arg)
{
	(void)arg;

	int cl_rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &cl_rank);

	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
 	float newA = 2 * (*tA);
	printf("[%d] received A = %f, now A = %f\n", cl_rank, *tA, newA);
	*tA = newA;
}

/* Define a StarPU 'codelet' structure for the replicated work above.
 */
static struct starpu_codelet repl_work_cl =
{
	.cpu_funcs = {cpu_repl_work}, /* cpu implementation(s) of the routine */
	.nbuffers = 1, /* number of data handles referenced by this routine */
	.modes = {STARPU_RW},
	.name = "repl_work", /* to display task name in traces */
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char *argv[])
{
	float A  = 1.1;
	starpu_data_handle_t A_h; /* handle for data A */
	int comm_rank; /* mpi rank of the process */
	int comm_size; /* size of the mpi session */

	starpu_mpi_tag_t tag = 0;
	/* Initializes StarPU and the StarPU-MPI layer */
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	/* Get the process rank and session size */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	if (comm_rank == 0)
	{
		starpu_variable_data_register(&A_h, STARPU_MAIN_RAM, (uintptr_t)&A, sizeof(A));
		starpu_data_acquire(A_h, STARPU_W);
		A = 1.1;
		starpu_data_release(A_h);
		FPRINTF(stderr, "[%d] I own A=%f (%p)\n", comm_rank,A,A_h);
	}
	else
	{
		starpu_variable_data_register(&A_h, -1, (uintptr_t) NULL, sizeof(A));
		FPRINTF(stderr, "[%d] I do not own A (%p)\n", comm_rank,A_h);
	}
	starpu_mpi_data_register(A_h, tag++, 0);

	int* work_rks = malloc(comm_size*sizeof(int));
	int work_rk;
 	for (work_rk = 0; work_rk < comm_size; work_rk++)
	{
		work_rks[work_rk] = work_rk;
	}
	starpu_mpi_tasks_insert(MPI_COMM_WORLD, &repl_work_cl,
				comm_size, work_rks,
				STARPU_RW, A_h,
				0);
	free(work_rks);
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
	starpu_data_unregister_submit(A_h);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	starpu_mpi_shutdown();
	return 0;
}
