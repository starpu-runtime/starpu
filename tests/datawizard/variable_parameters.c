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
#include "../helper.h"

/*
 * Test the variable interface
 */

static starpu_data_handle_t handle1, handle2, handle3, handle4;

/*
 *	Increment codelet
 */

#ifdef STARPU_USE_OPENCL
/* dummy OpenCL implementation */
static void increment_opencl_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	int num = starpu_task_get_current()->nbuffers;
	int i;

	for (i = 0; i < num; i++)
	{
		cl_mem d_token = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[i]);
		unsigned h_token;

		cl_command_queue queue;
		starpu_opencl_get_current_queue(&queue);

		clEnqueueReadBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
		h_token++;
		clEnqueueWriteBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
		clFinish(queue);
	}
}
#endif

#ifdef STARPU_USE_CUDA
static void increment_cuda_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	int num = starpu_task_get_current()->nbuffers;
	int i;

	for (i = 0; i < num; i++)
	{
		unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[i]);
		unsigned host_token;

		/* This is a dummy technique of course */
		cudaMemcpyAsync(&host_token, tokenptr, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
		cudaStreamSynchronize(starpu_cuda_get_local_stream());

		host_token++;

		cudaMemcpyAsync(tokenptr, &host_token, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	}
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

void increment_cpu_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	int num = starpu_task_get_current()->nbuffers;
	int i;

	for (i = 0; i < num; i++)
	{
		unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[i]);
		*tokenptr = *tokenptr + 1;
	}
}

static struct starpu_codelet increment_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {increment_cuda_kernel},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {increment_opencl_kernel},
#endif
	.cpu_funcs = {increment_cpu_kernel},

	/* starpu_task_get_current() doesn't work on MIC */
	/*.cpu_funcs_name = {"increment_cpu_kernel"},*/
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

int main(void)
{
	unsigned *pvar = NULL;
	int ret;
	unsigned var1 = 0, var2 = 0, var3 = 0, var4 = 0;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)&var1, sizeof(unsigned));
	starpu_variable_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)&var2, sizeof(unsigned));
	starpu_variable_data_register(&handle3, STARPU_MAIN_RAM, (uintptr_t)&var3, sizeof(unsigned));
	starpu_variable_data_register(&handle4, STARPU_MAIN_RAM, (uintptr_t)&var4, sizeof(unsigned));

#ifdef STARPU_QUICK_CHECK
	unsigned nloops = 4;
#else
	unsigned nloops = 16;
#endif

	unsigned loop;
	unsigned t;

	for (loop = 0; loop < nloops; loop++)
	{
		for (t = 0; t <= 4; t++)
		{
			struct starpu_task *task = starpu_task_create();
			unsigned i;

			task->cl = &increment_cl;
			task->handles[0] = handle1;
			task->handles[1] = handle2;
			task->handles[2] = handle3;
			task->handles[3] = handle4;
			for (i = 0; i < t; i++)
				task->modes[i] = STARPU_RW;
			task->nbuffers = t;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		starpu_task_insert(&increment_cl,
				STARPU_RW, handle1,
				0);
		starpu_task_insert(&increment_cl,
				STARPU_RW, handle1,
				STARPU_RW, handle2,
				0);
		starpu_task_insert(&increment_cl,
				STARPU_RW, handle1,
				STARPU_RW, handle2,
				STARPU_RW, handle3,
				0);
		starpu_task_insert(&increment_cl,
				STARPU_RW, handle1,
				STARPU_RW, handle2,
				STARPU_RW, handle3,
				STARPU_RW, handle4,
				0);
	}

	ret = starpu_data_acquire(handle1, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
	if (var1 != 8*nloops)
	{
		FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", var1, 8*nloops);
		starpu_data_release(handle1);
		goto err;
	}
	starpu_data_release(handle1);

	ret = starpu_data_acquire(handle2, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
	if (var2 != 6*nloops)
	{
		FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", var2, 6*nloops);
		starpu_data_release(handle2);
		goto err;
	}
	starpu_data_release(handle2);

	ret = starpu_data_acquire(handle3, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
	if (var3 != 4*nloops)
	{
		FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", var3, 4*nloops);
		starpu_data_release(handle3);
		goto err;
	}
	starpu_data_release(handle3);

	ret = starpu_data_acquire(handle4, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
	if (var4 != 2*nloops)
	{
		FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", var4, 2*nloops);
		starpu_data_release(handle4);
		goto err;
	}
	starpu_data_release(handle4);

	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_data_unregister(handle3);
	starpu_data_unregister(handle4);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_data_unregister(handle3);
	starpu_data_unregister(handle4);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;

err:
	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_data_unregister(handle3);
	starpu_data_unregister(handle4);
	starpu_shutdown();
	return EXIT_FAILURE;
}
