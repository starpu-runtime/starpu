/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <pthread.h>

static unsigned niter = 50000;

extern void cpu_codelet(void *descr[], __attribute__ ((unused)) void *_args);

#ifdef STARPU_USE_CUDA
extern void cuda_codelet(void *descr[], __attribute__ ((unused)) void *_args);
#endif

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
extern void opencl_codelet(void *descr[], __attribute__ ((unused)) void *_args);
struct starpu_opencl_program opencl_code;
#endif

int main(int argc, char **argv)
{
	unsigned i;
        float foo;
	starpu_data_handle float_array_handle;
	starpu_codelet cl;

	starpu_init(NULL);
        if (argc == 2) niter = atoi(argv[1]);
        foo = 0.0f;

	starpu_variable_data_register(&float_array_handle, 0 /* home node */,
                                      (uintptr_t)&foo, sizeof(float));

#ifdef STARPU_USE_OPENCL
        starpu_opencl_load_opencl_from_file("examples/basic_examples/variable_kernels_opencl_codelet.cl", &opencl_code);
#endif

	cl.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL;
        cl.cpu_func = cpu_codelet;
#ifdef STARPU_USE_CUDA
        cl.cuda_func = cuda_codelet;
#endif
#ifdef STARPU_USE_OPENCL
        cl.opencl_func = opencl_codelet;
#endif
        cl.nbuffers = 1;
        cl.model = NULL;

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = starpu_task_create();
                int ret;

		task->cl = &cl;

		task->callback_func = NULL;

		task->buffers[0].handle = float_array_handle;
		task->buffers[0].mode = STARPU_RW;

		ret = starpu_task_submit(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			fprintf(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_task_wait_for_all();

	/* update the array in RAM */
	starpu_data_acquire(float_array_handle, STARPU_R);

	fprintf(stderr, "variable -> %f\n", foo);

	starpu_data_release(float_array_handle);

	starpu_shutdown();

	return 0;
}
