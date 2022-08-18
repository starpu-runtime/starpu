/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example illustrates how to use the STARPU_MPI_REDUX mode
 * and compare it with the standard STARPU_REDUX.
 *
 * In order to make this comparison salliant, the init codelet is not
 * a task that set the handle to a neutral element but rather depends
 * on the working node.
 * This is not a proper way to use a reduction pattern however it
 * can be analogous to the cost/weight of each contribution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"
#include <unistd.h>

static void cl_cpu_work(void *handles[], void*arg)
{
	(void)arg;
	double *a = (double *)STARPU_VARIABLE_GET_PTR(handles[0]);
	double *b = (double *)STARPU_VARIABLE_GET_PTR(handles[1]);
	sleep(2);
	printf("work_cl (rank:%d,worker:%d) %f =>",starpu_mpi_world_rank(), starpu_worker_get_id(), *a);
	*a = 3.0 + *a + *b;
	printf("%f\n",*a);
}

static struct starpu_codelet work_cl =
{
	.cpu_funcs = { cl_cpu_work },
	.nbuffers = 2,
	.modes = { STARPU_REDUX, STARPU_R },
	.name = "task_init"
};

static struct starpu_codelet mpi_work_cl =
{
	.cpu_funcs = { cl_cpu_work },
	.nbuffers = 2,
	.modes = { STARPU_RW | STARPU_COMMUTE, STARPU_R },
	.name = "task_init-mpi"
};

static void cl_cpu_task_init(void *handles[], void*arg)
{
	(void) arg;
	double *a = (double *)STARPU_VARIABLE_GET_PTR(handles[0]);
	sleep(1);
	printf("init_cl (rank:%d,worker:%d) %d (was %f)\n", starpu_mpi_world_rank(), starpu_worker_get_id(), starpu_mpi_world_rank(),
#ifdef STARPU_HAVE_VALGRIND_H
			RUNNING_ON_VALGRIND ? 0. :
#endif
			*a);
	*a = starpu_mpi_world_rank();
}

static struct starpu_codelet task_init_cl =
{
	.cpu_funcs = { cl_cpu_task_init },
	.nbuffers = 1,
	.modes = { STARPU_W },
	.name = "task_init"
};

static void cl_cpu_task_red(void *handles[], void*arg)
{
	(void) arg;
	double *ad = (double *)STARPU_VARIABLE_GET_PTR(handles[0]);
	double *as = (double *)STARPU_VARIABLE_GET_PTR(handles[1]);
	sleep(2);
	printf("red_cl (rank:%d,worker:%d) %f ; %f --> %f\n", starpu_mpi_world_rank(), starpu_worker_get_id(), *as, *ad, *as+*ad);
	*ad = *ad + *as;
}

static struct starpu_codelet task_red_cl =
{
	.cpu_funcs = { cl_cpu_task_red },
	.nbuffers = 2,
	.modes = { STARPU_RW|STARPU_COMMUTE, STARPU_R },
	.name = "task_red"
};

int main(int argc, char *argv[])
{
	int comm_rank, comm_size;
	/* Initializes STarPU and the StarPU-MPI layer */
	starpu_fxt_autostart_profiling(0);
	int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_ini_conft");

	int nworkers = starpu_cpu_worker_get_count();
	if (nworkers < 2)
	{
		FPRINTF(stderr, "We need at least 2 CPU worker per node.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);
	if (comm_size < 2)
	{
		FPRINTF(stderr, "We need at least 2 nodes.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &comm_rank);

	double a, b[comm_size];
	starpu_data_handle_t a_h, b_h[comm_size];
	double work_coef = 2;
	enum starpu_data_access_mode task_mode;
	int i,j,work_node;
	starpu_mpi_tag_t tag = 0;
	for (i = 0 ; i < 2 ; i++)
	{
		starpu_mpi_barrier(MPI_COMM_WORLD);
		if (i==0)
			task_mode = STARPU_MPI_REDUX;
		else
			task_mode = STARPU_REDUX;
		if (comm_rank == 0)
		{
			a = 1.0;
			printf("init a = %f\n", a);
			starpu_variable_data_register(&a_h, STARPU_MAIN_RAM, (uintptr_t)&a, sizeof(double));
			for (j=0;j<comm_size;j++)
				starpu_variable_data_register(&b_h[j], -1, 0, sizeof(double));
		}
		else
		{
			b[comm_rank] = 1.0 / (comm_rank + 1.0);
			printf("init b_%d = %f\n", comm_rank, b[comm_rank]);
			starpu_variable_data_register(&a_h, -1, 0, sizeof(double));
			for (j=0;j<comm_size;j++)
			{
				if (j == comm_rank)
					starpu_variable_data_register(&b_h[j], STARPU_MAIN_RAM, (uintptr_t)&b[j], sizeof(double));
				else
					starpu_variable_data_register(&b_h[j], -1, 0, sizeof(double));
			}
		}
		starpu_mpi_data_register(a_h, tag++, 0);
		for (j=0;j<comm_size;j++)
			starpu_mpi_data_register(b_h[j], tag++, j);

		starpu_data_set_reduction_methods(a_h, &task_red_cl, &task_init_cl);
		starpu_fxt_start_profiling();
		for (work_node=1; work_node < comm_size;work_node++)
		{
			for (j=1;j<=work_coef*nworkers;j++)
			{
				if (i == 0)
				    starpu_mpi_task_insert(MPI_COMM_WORLD,
					&mpi_work_cl,
					task_mode, a_h,
					STARPU_R, b_h[work_node],
					STARPU_EXECUTE_ON_NODE, work_node,
					0);
				else
				    starpu_mpi_task_insert(MPI_COMM_WORLD,
					&work_cl,
					task_mode, a_h,
					STARPU_R, b_h[work_node],
					STARPU_EXECUTE_ON_NODE, work_node,
					0);
			}
		}
		ret = starpu_mpi_redux_data(MPI_COMM_WORLD, a_h);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_redux_data");
		starpu_mpi_wait_for_all(MPI_COMM_WORLD);
		starpu_mpi_barrier(MPI_COMM_WORLD);
		if (comm_rank == 0)
		{
			double tmp = 0.0;
			for (work_node = 1; work_node < comm_size ; work_node++)
				tmp += 1.0 / (work_node + 1.0);
			printf("computed result ---> %f expected %f\n", a, 1.0 + (comm_size - 1.0)*(comm_size)/2.0 + work_coef*nworkers*((comm_size-1)*3.0 + tmp));
		}
		starpu_data_unregister(a_h);
		for (work_node=0; work_node < comm_size;work_node++)
			starpu_data_unregister(b_h[work_node]);
		starpu_mpi_barrier(MPI_COMM_WORLD);
	}
	starpu_mpi_shutdown();
	return 0;
}
