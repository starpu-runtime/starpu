/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

#define N 16
#define M 4
#define X 2

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

void which_index_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* A real case would actually compute something */
	*x0 = X;
}

starpu_codelet which_index = {
	.where = STARPU_CPU,
	.cpu_func = which_index_cpu,
        .nbuffers = 1
};

void work_cpu(void *descr[], void *_args)
{
	int i, n = STARPU_VECTOR_GET_NX(descr[0]);
	float *x0 = (float *)STARPU_VECTOR_GET_PTR(descr[0]);

	for (i = 0; i < n; i++)
		x0[i] = i + 1;
}

starpu_codelet work = {
	.where = STARPU_CPU,
	.cpu_func = work_cpu,
        .nbuffers = 1
};

static int x;
static starpu_data_handle x_handle, f_handle;

void callback(void *arg) {
	starpu_insert_task(&work, STARPU_W, starpu_data_get_sub_data(f_handle, 1, x), 0);
	starpu_data_release(x_handle);
}

int main(int argc, char **argv)
{
        int i, ret;
	float *f;

	starpu_init(NULL);

	/* Declare x */
	starpu_variable_data_register(&x_handle, 0, (uintptr_t)&x, sizeof(x));

	/* Allocate and Declare f */
	starpu_malloc((void**)&f, N * sizeof(*f));
	memset(f, 0, N * sizeof(*f));
	starpu_vector_data_register(&f_handle, 0, (uintptr_t)f, N, sizeof(*f));

	/* Partition f */
	struct starpu_data_filter filter = {
		.filter_func = starpu_block_filter_func_vector,
		.nchildren = M,
	};
	starpu_data_partition(f_handle, &filter);

	/* Compute which portion we will work on */
        ret = starpu_insert_task(&which_index, STARPU_W, x_handle, 0);
	if (ret == -ENODEV) goto enodev;

	/* And submit the corresponding task */
#ifdef __GCC__
	STARPU_DATA_ACQUIRE_CB(
			x_handle,
			STARPU_R,
			starpu_insert_task(&work, STARPU_W, starpu_data_get_sub_data(f_handle, 1, x), 0)
			);
#else
	starpu_data_acquire_cb(x_handle, STARPU_W, callback, NULL);
#endif

	starpu_task_wait_for_all();
	starpu_data_unpartition(f_handle, 0);
	starpu_data_unregister(f_handle);
	starpu_data_unregister(x_handle);

        FPRINTF(stderr, "VALUES: %d", x);

        for(i=0 ; i<N ; i++) {
		FPRINTF(stderr, " %f", f[i]);
        }

	STARPU_ASSERT(f[X*(N/M)] == 1);
	STARPU_ASSERT(f[X*(N/M)+1] == 2);
	STARPU_ASSERT(f[X*(N/M)+2] == 3);
	STARPU_ASSERT(f[X*(N/M)+3] == 4);

	FPRINTF(stderr, "\n");

	starpu_shutdown();
	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 77;
}
