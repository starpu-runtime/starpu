/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <omp.h>

#if !defined(STARPU_PARALLEL_WORKER)
int main(void)
{
	return 77;
}
#else

#ifdef STARPU_QUICK_CHECK
#define NTASKS 8
#else
#define NTASKS 32
#endif
#define SIZE 4000

/* Codelet SUM */
static void sum_cpu(void * descr[], void *cl_arg)
{
	double * v_dst = (double *) STARPU_VECTOR_GET_PTR(descr[0]);
	double * v_src0 = (double *) STARPU_VECTOR_GET_PTR(descr[1]);
	double * v_src1 = (double *) STARPU_VECTOR_GET_PTR(descr[1]);

	int size;
	starpu_codelet_unpack_args(cl_arg, &size);
	fprintf(stderr, "sum_cpu\n");
	int i, k;
#pragma omp parallel
	fprintf(stderr, "hello from the task %d\n", omp_get_thread_num());
	for (k=0;k<10;k++)
	{
#pragma omp parallel for
		for (i=0; i<size; i++)
		{
			v_dst[i]+=v_src0[i]+v_src1[i];
		}
	}
}

static struct starpu_codelet sum_cl =
{
	.cpu_funcs = {sum_cpu, NULL},
	.nbuffers = 3,
	.modes={STARPU_RW,STARPU_R, STARPU_R}
};

int main(void)
{
	int ntasks = NTASKS;
	int ret, i;
	struct starpu_parallel_worker_config *parallel_workers;

	setenv("STARPU_NMPI_MS","0",1);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* We regroup resources under each sockets into a parallel worker. We express a partition
	 * of one socket to create two internal parallel workers */
	parallel_workers = starpu_parallel_worker_init(HWLOC_OBJ_SOCKET,
						       STARPU_PARALLEL_WORKER_POLICY_NAME, "dmdas",
						       STARPU_PARALLEL_WORKER_PARTITION_ONE,
						       STARPU_PARALLEL_WORKER_NEW,
//						       STARPU_PARALLEL_WORKER_TYPE, STARPU_PARALLEL_WORKER_OPENMP,
//						       STARPU_PARALLEL_WORKER_TYPE, STARPU_PARALLEL_WORKER_INTEL_OPENMP_MKL,
						       STARPU_PARALLEL_WORKER_NB, 2,
						       STARPU_PARALLEL_WORKER_NCORES, 1,
					  0);
	if (parallel_workers == NULL)
		goto enodev;
	starpu_parallel_worker_print(parallel_workers);

	/* Data preparation */
	double array1[SIZE];
	double array2[SIZE];

	memset(array1, 0, sizeof(double));
	for (i=0;i<SIZE;i++)
	{
		array2[i]=i*2;
	}

	starpu_data_handle_t handle1;
	starpu_data_handle_t handle2;

	starpu_vector_data_register(&handle1, 0, (uintptr_t)array1, SIZE, sizeof(double));
	starpu_vector_data_register(&handle2, 0, (uintptr_t)array2, SIZE, sizeof(double));

	int size = SIZE;

	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_insert(&sum_cl,
					 STARPU_RW, handle1,
					 STARPU_R, handle2,
					 STARPU_R, handle1,
					 STARPU_VALUE, &size, sizeof(int),

					 /* For two tasks, try out the case when the task isn't parallel and expect
					    the configuration to be sequential due to this, then automatically changed
					    back to the parallel one */
					 STARPU_POSSIBLY_PARALLEL, (i<=4 || i > 6) ? 1 : 0,

					 /* Note that this mode requires that you put a prologue callback managing
					    this on all tasks to be taken into account. */
					 STARPU_PROLOGUE_CALLBACK_POP, &starpu_parallel_worker_openmp_prologue,

					 0);

		if (ret == -ENODEV)
			goto out;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}



out:
	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();

	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_parallel_worker_shutdown(parallel_workers);

	starpu_shutdown();
	return (ret == -ENODEV) ? 77 : 0 ;

enodev:
	starpu_shutdown();
	return 77;
}
#endif
