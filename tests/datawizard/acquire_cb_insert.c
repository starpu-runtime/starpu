/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include "../helper.h"

/*
 * Test that inserting a task from the callback of a starpu_data_acquire_cb
 * call, with proper dependency with an already-submitted task
 */

#define N 16
#define M 4
#define X 2

void which_index_cpu(void *descr[], void *_args)
{
	(void)_args;

	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* A real case would actually compute something */
	*x0 = X;
}

struct starpu_codelet which_index =
{
	.cpu_funcs = {which_index_cpu},
	.cpu_funcs_name = {"which_index_cpu"},
        .nbuffers = 1,
	.modes = {STARPU_W}
};

void work_cpu(void *descr[], void *_args)
{
	int i, n = STARPU_VECTOR_GET_NX(descr[0]);
	float *x0 = (float *)STARPU_VECTOR_GET_PTR(descr[0]);

	(void)_args;

	for (i = 0; i < n; i++)
		x0[i] = i + 1;
}

struct starpu_codelet work =
{
	.cpu_funcs = {work_cpu},
	.cpu_funcs_name = {"work_cpu"},
        .nbuffers = 1,
	.modes = {STARPU_W}
};

static int x;
static starpu_data_handle_t x_handle, f_handle;

static
void callback(void *arg)
{
	(void)arg;
	starpu_task_insert(&work, STARPU_W, starpu_data_get_sub_data(f_handle, 1, x), 0);
	starpu_data_release(x_handle);
}

int main(int argc, char **argv)
{
        int i, ret;
	float *f;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if(starpu_cpu_worker_get_count() == 0) return STARPU_TEST_SKIPPED;

	/* Declare x */
	starpu_variable_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));

	/* Allocate and Declare f */
	ret = starpu_malloc((void**)&f, N * sizeof(*f));
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");
	memset(f, 0, N * sizeof(*f));
	starpu_vector_data_register(&f_handle, STARPU_MAIN_RAM, (uintptr_t)f, N, sizeof(*f));

	/* Partition f */
	struct starpu_data_filter filter =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = M,
	};
	starpu_data_partition(f_handle, &filter);

	/* Compute which portion we will work on */
        ret = starpu_task_insert(&which_index, STARPU_W, x_handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* And submit the corresponding task */
#ifdef __GCC__
	STARPU_DATA_ACQUIRE_CB(
			x_handle,
			STARPU_R,
			starpu_task_insert(&work, STARPU_W, starpu_data_get_sub_data(f_handle, 1, x), 0)
			);
#else
	starpu_data_acquire_cb(x_handle, STARPU_R, callback, NULL);
#endif

	/* Wait for acquisition (and thus insertion) */
	starpu_data_acquire(x_handle, STARPU_W);
	starpu_data_release(x_handle);

	/* Now wait for the inserted task */
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	/* Can now clean */
	starpu_data_unpartition(f_handle, STARPU_MAIN_RAM);
	starpu_data_unregister(f_handle);
	starpu_data_unregister(x_handle);

        FPRINTF(stderr, "VALUES: %d", x);
        for(i=0 ; i<N ; i++)
	{
		FPRINTF(stderr, " %f", f[i]);
        }
	FPRINTF(stderr, "\n");

	ret = EXIT_SUCCESS;
	if (f[X*(N/M)] != 1 || f[X*(N/M)+1] != 2 ||
	    f[X*(N/M)+2] != 3 || f[X*(N/M)+3] != 4)
		ret = EXIT_FAILURE;

	starpu_free(f);
	starpu_shutdown();
	return ret;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
