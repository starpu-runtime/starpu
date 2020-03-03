/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "../helper.h"

/*
 * Test that STARPU_RW vs STARPU_RW|STARPU_COMMUTE get proper dependency
 */

static unsigned cnt;

static void cpu_memcpy(void *descr[], void *cl_arg)
{
	int me = (uintptr_t)cl_arg;
	int res;

	(void)descr;

	FPRINTF(stderr,"%d\n", me);

	if (me == 0)
	{
		/* let commute tasks potentially happen */
		usleep(100000);
		res = STARPU_ATOMIC_ADD(&cnt,1);
		STARPU_ASSERT(res == 1);
	}
	else
	{
		res = STARPU_ATOMIC_ADD(&cnt,1);
		STARPU_ASSERT(res != 1);
	}
}

static struct starpu_codelet my_cl =
{
	.where =  STARPU_CPU,
	.cpu_funcs = {cpu_memcpy},
	.nbuffers = STARPU_VARIABLE_NBUFFERS
};

int main(void)
{
	double *res, *a;
	unsigned n=100000, i;
	starpu_data_handle_t res_handle, a_handle;
	unsigned nb_tasks = 10, worker;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void**)&res, n*sizeof(double));
	starpu_malloc((void**)&a,   n*sizeof(double));

	for(i=0; i < n; i++)
		res[i] = a[i] = 1.0;

	starpu_vector_data_register(&res_handle, 0, (uintptr_t)res, (uint32_t)n, sizeof(double));
	starpu_vector_data_register(&a_handle,   0, (uintptr_t)a,   (uint32_t)n, sizeof(double));

	starpu_data_acquire(a_handle, STARPU_RW);
	for (i = 0; i < nb_tasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl=&my_cl;
		task->nbuffers = i == 0 ? 2 : 1;
		task->handles[0] = res_handle;

		if (i == 0)
			task->modes[0]   = STARPU_RW;
		else
			task->modes[0]   = STARPU_RW | STARPU_COMMUTE;

		task->handles[1] = a_handle;
		task->modes[1]   = STARPU_R;
		task->cl_arg = (void*)(uintptr_t)i;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
		{
			starpu_data_release(a_handle);
			goto enodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* let commute tasks potentially happen */
	usleep(100000);
	starpu_data_release(a_handle);

	starpu_task_wait_for_all ();

enodev:
	starpu_data_unregister(res_handle);
	starpu_data_unregister(a_handle);

	starpu_free(res);
	starpu_free(a);

	starpu_shutdown();
	return ret == -ENODEV ? STARPU_TEST_SKIPPED : EXIT_SUCCESS;
}
