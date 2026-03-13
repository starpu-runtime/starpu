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
 * This example illustrates the combination of replicating tasks and setting up
 * alternative sources : a data A, owned by 0, is send to 1 and 2. 0, 1 and 2
 * use A to compute a value of A. Then all the other nodes receive A either from
 * 0, 1 or 2 based on their rank. They use A to locally compute another result.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

static void cpu_repl_work(void *handles[], void *arg)
{
	(void)arg;

	int cl_rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &cl_rank);

	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
	// this helps track where the data came from
	float newA = 2 * (*tA) + cl_rank/1000.0;
	printf("[%d] received A = %f, now A = %f\n", cl_rank, *tA, newA);
	*tA = newA;
}

static void cpu_work(void *handles[], void *arg)
{
	(void)arg;
	int cl_rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &cl_rank);

	float *tA = (float *)STARPU_VARIABLE_GET_PTR(handles[0]);
   	float B = *tA + cl_rank/100.0;
	printf("[%d] using %f, now my value is %f\n", cl_rank, *tA, B);
}

static struct starpu_codelet repl_work_cl =
{
	.cpu_funcs = {cpu_repl_work}, /* cpu implementation(s) of the routine */
	.nbuffers = 1, /* number of data handles referenced by this routine */
	.modes = {STARPU_RW},
	.name = "repl_work" /* to display task name in traces */
};

static struct starpu_codelet work_cl =
{
	.cpu_funcs = {cpu_work}, /* cpu implementation(s) of the routine */
	.nbuffers = 1, /* number of data handles referenced by this routine */
	.modes = {STARPU_R}, /* access modes for each data handle */
	.name = "work" /* to display task name in traces */
};

int main(int argc, char *argv[])
{
	int i;
	float A  = 1.1;
	starpu_data_handle_t A_h;

	int comm_rank; /* mpi rank of the process */
	int comm_size; /* size of the mpi session */

	starpu_mpi_tag_t tag = 0;
	/* Initializes StarPU and the StarPU-MPI layer */
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	/* Get the process rank and session size */
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank == 0) FPRINTF(stderr, "Launching with %d arguments\n",argc);

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	if (comm_size < 5)
	{
		FPRINTF(stderr, "We need at least 5 processes.\n");
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

	int rk;
 	for (rk = 0; rk < 2; rk ++)
	{
		if (rk != 0)
		{
			starpu_mpi_task_insert(MPI_COMM_WORLD, &repl_work_cl,
					       STARPU_RW | STARPU_MPI_SAME, A_h,
					       STARPU_EXECUTE_ON_NODE, rk,
					       0);
		}
	}
	starpu_mpi_task_insert(MPI_COMM_WORLD, &repl_work_cl,
			       STARPU_RW | STARPU_MPI_SAME, A_h,
			       STARPU_EXECUTE_ON_NODE, 0,
			       0);
	for (rk = 0; rk < comm_size; rk++)
	{
		starpu_mpi_data_set_source(A_h, rk, rk % 2);
		FPRINTF(stderr, "[%d] %d sources A from %d\n", comm_rank, rk, starpu_mpi_data_get_source(A_h,rk));
		starpu_mpi_task_insert(MPI_COMM_WORLD, &work_cl,
				       STARPU_R, A_h,
				       STARPU_EXECUTE_ON_NODE, rk,
				       0);
		//if (rk != 1) starpu_mpi_data_reset_source(A_h, rk);
		FPRINTF(stderr, "[%d] %d resets A from %d\n", comm_rank, rk, starpu_mpi_data_get_source(A_h,rk));
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
	starpu_data_unregister(A_h);

	starpu_mpi_shutdown();
	return 0;
}
