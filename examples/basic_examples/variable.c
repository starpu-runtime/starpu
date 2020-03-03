/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 500;
#elif !defined(STARPU_LONG_CHECK)
static unsigned niter = 5000;
#else
static unsigned niter = 50000;
#endif

extern void cpu_codelet(void *descr[], void *_args);

#ifdef STARPU_USE_CUDA
extern void cuda_codelet(void *descr[], void *_args);
#endif

#ifdef STARPU_USE_OPENCL
extern void opencl_codelet(void *descr[], void *_args);
struct starpu_opencl_program opencl_program;
#endif

int main(int argc, char **argv)
{
	unsigned i;
        float foo;
	starpu_data_handle_t float_array_handle;
	struct starpu_codelet cl;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

        if (argc == 2) niter = atoi(argv[1]);
        foo = 0.0f;

	starpu_variable_data_register(&float_array_handle, STARPU_MAIN_RAM /* home node */,
                                      (uintptr_t)&foo, sizeof(float));

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_load_opencl_from_file("examples/basic_examples/variable_kernels_opencl_kernel.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	starpu_codelet_init(&cl);
        cl.cpu_funcs[0] = cpu_codelet;
        cl.cpu_funcs_name[0] = "cpu_codelet";
#ifdef STARPU_USE_CUDA
        cl.cuda_funcs[0] = cuda_codelet;
#endif
#ifdef STARPU_USE_OPENCL
        cl.opencl_funcs[0] = opencl_codelet;
#endif
        cl.nbuffers = 1;
	cl.modes[0] = STARPU_RW;
        cl.model = NULL;
	cl.name = "variable_inc";

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;

		task->callback_func = NULL;

		task->handles[0] = float_array_handle;

		ret = starpu_task_submit(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			starpu_data_unregister(float_array_handle);
			goto enodev;
		}
	}

	starpu_task_wait_for_all();

	/* update the array in RAM */
	starpu_data_unregister(float_array_handle);

	FPRINTF(stderr, "variable -> %f\n", foo);
	FPRINTF(stderr, "result is %scorrect\n", foo==niter?"":"IN");

	starpu_shutdown();

	return (foo == niter) ? EXIT_SUCCESS:EXIT_FAILURE;

enodev:
	return 77;
}
