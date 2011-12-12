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
#include "../helper.h"

void func_cpu(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *x1 = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int ifactor;
	float ffactor;

	starpu_unpack_cl_args(_args, &ifactor, &ffactor);
        *x0 = *x0 * ifactor;
        *x1 = *x1 * ffactor;
}

struct starpu_codelet mycodelet = {
	.where = STARPU_CPU,
	.cpu_funcs = {func_cpu, NULL},
        .nbuffers = 2
};

int main(int argc, char **argv)
{
        int x; float f;
        int i, ret;
	int ifactor=12;
	float ffactor=10.0;
        starpu_data_handle_t data_handles[2];

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	x = 1;
	starpu_variable_data_register(&data_handles[0], 0, (uintptr_t)&x, sizeof(x));
	f = 2.0;
	starpu_variable_data_register(&data_handles[1], 0, (uintptr_t)&f, sizeof(f));

        FPRINTF(stderr, "VALUES: %d (%d) %f (%f)\n", x, ifactor, f, ffactor);

        ret = starpu_insert_task(&mycodelet,
				 STARPU_VALUE, &ifactor, sizeof(ifactor),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task");

        ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

        for(i=0 ; i<2 ; i++) {
                ret = starpu_data_acquire(data_handles[i], STARPU_R);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
        }
        FPRINTF(stderr, "VALUES: %d %f\n", x, f);

        for(i=0 ; i<2 ; i++) {
                starpu_data_release(data_handles[i]);
        }

	struct starpu_task *task = starpu_task_create();
	task->cl = &mycodelet;
	task->buffers[0].handle = data_handles[0];
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = data_handles[1];
	task->buffers[1].mode = STARPU_RW;
	char *arg_buffer;
	size_t arg_buffer_size;
	starpu_pack_cl_args(&arg_buffer, &arg_buffer_size,
			    STARPU_VALUE, &ifactor, sizeof(ifactor),
			    STARPU_VALUE, &ffactor, sizeof(ffactor),
			    0);
	task->cl_arg = arg_buffer;
	task->cl_arg_size = arg_buffer_size;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task");

        ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

        for(i=0 ; i<2 ; i++) {
                ret = starpu_data_acquire(data_handles[i], STARPU_R);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
        }
        FPRINTF(stderr, "VALUES: %d %f\n", x, f);

        for(i=0 ; i<2 ; i++) {
                starpu_data_release(data_handles[i]);
		starpu_data_unregister(data_handles[i]);
        }

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
