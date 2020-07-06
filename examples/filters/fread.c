/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX    20
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void display_func(void *buffers[], void *cl_arg)
{
        unsigned i;

        /* length of the vector */
        unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
        /* local copy of the vector pointer */
        int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	FPRINTF(stderr, "vector with n=%u : ", n);
        for (i = 0; i < n; i++)
		FPRINTF(stderr, "%5d ", val[i]);
	FPRINTF(stderr, "\n");
}

void cpu_func(void *buffers[], void *cl_arg)
{
        unsigned i;

        /* length of the vector */
        unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
        /* local copy of the vector pointer */
        int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	FPRINTF(stderr, "computing on vector with n=%u\n", n);
        for (i = 0; i < n; i++)
                val[i] *= 2;
}

int main(void)
{
	int i;
        int vector[NX];
        starpu_data_handle_t handle;
	starpu_data_handle_t subhandles[PARTS];
	int ret;

        struct starpu_codelet cl =
	{
                .cpu_funcs = {cpu_func},
                .cpu_funcs_name = {"cpu_func"},
                .nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "vector_scal"
        };
        struct starpu_codelet print_cl =
	{
                .cpu_funcs = {display_func},
                .cpu_funcs_name = {"display_func"},
                .nbuffers = 1,
		.modes = {STARPU_R},
		.name = "vector_display"
        };

        for(i=0 ; i<NX ; i++) vector[i] = i;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare data to StarPU */
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

        /* Partition the vector in PARTS sub-vectors */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition_plan(handle, &f, subhandles);

	ret = starpu_task_insert(&print_cl,
				 STARPU_R, handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        /* Submit a task on each sub-vector */
	for (i=0; i<PARTS; i++)
	{
		ret = starpu_task_insert(&cl,
					 STARPU_RW, subhandles[i],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Submit a read on the whole vector */
	ret = starpu_task_insert(&print_cl,
				 STARPU_R, handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

        /* Submit a read on each sub-vector */
	for (i=0; i<PARTS; i++)
	{
		ret = starpu_task_insert(&print_cl,
					 STARPU_R, subhandles[i],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Submit a read on the whole vector */
	ret = starpu_task_insert(&print_cl,
				 STARPU_R, handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_data_partition_clean(handle, PARTS, subhandles);
        starpu_data_unregister(handle);
	starpu_shutdown();

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
