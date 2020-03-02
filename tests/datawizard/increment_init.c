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
 * Check that the initializer passed to starpu_data_set_reduction_methods
 * is used to initialize a handle when it is registered from NULL, and when
 * starpu_data_invalidate is called
 */

static starpu_data_handle_t handle;

/*
 *	Reduction methods
 */

#ifdef STARPU_USE_CUDA
static void neutral_cuda_kernel(void *descr[], void *arg)
{
	(void)arg;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* This is a dummy technique of course */
	unsigned host_dst = 0;
	cudaMemcpyAsync(dst, &host_dst, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
static void neutral_opencl_kernel(void *descr[], void *arg)
{
	(void)arg;

	unsigned h_dst = 0;
	cl_mem d_dst = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueWriteBuffer(queue, d_dst, CL_TRUE, 0, sizeof(unsigned), (void *)&h_dst, 0, NULL, NULL);
	clFinish(queue);
}
#endif

void neutral_cpu_kernel(void *descr[], void *arg)
{
	(void)arg;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dst = 0;
}

static struct starpu_codelet neutral_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {neutral_cuda_kernel},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {neutral_opencl_kernel},
#endif
	.cpu_funcs = {neutral_cpu_kernel},
	.cpu_funcs_name = {"neutral_cpu_kernel"},
	.modes = {STARPU_W},
	.nbuffers = 1
};

/*
 *	Increment codelet
 */

#ifdef STARPU_USE_OPENCL
/* dummy OpenCL implementation */
static void increment_opencl_kernel(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	cl_mem d_token = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned h_token;

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueReadBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
	h_token++;
	clEnqueueWriteBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
	clFinish(queue);
}
#endif


#ifdef STARPU_USE_CUDA
static void increment_cuda_kernel(void *descr[], void *arg)
{
	(void)arg;
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned host_token;

	/* This is a dummy technique of course */
	cudaMemcpyAsync(&host_token, tokenptr, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	host_token++;

	cudaMemcpyAsync(tokenptr, &host_token, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

void increment_cpu_kernel(void *descr[], void *arg)
{
	(void)arg;
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*tokenptr = *tokenptr + 1;
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
	.cpu_funcs_name = {"increment_cpu_kernel"},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int main(void)
{
	unsigned *pvar = NULL;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, -1, 0, sizeof(unsigned));

	starpu_data_set_reduction_methods(handle, NULL, &neutral_cl);

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 32;
	unsigned nloops = 4;
#else
	unsigned ntasks = 1024;
	unsigned nloops = 16;
#endif

	unsigned loop;
	unsigned t;

	for (loop = 0; loop < nloops; loop++)
	{
		for (t = 0; t < ntasks; t++)
		{
			struct starpu_task *task = starpu_task_create();

			task->cl = &increment_cl;
			task->handles[0] = handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		ret = starpu_data_acquire(handle, STARPU_R);
		pvar = starpu_data_handle_to_pointer(handle, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
		if (*pvar != ntasks)
		{
			FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", *pvar, ntasks * (loop+1));
			starpu_data_release(handle);
			starpu_data_unregister(handle);
			goto err;
		}
		starpu_data_release(handle);
		starpu_data_invalidate(handle);
	}

	starpu_data_unregister(handle);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;

err:
	starpu_shutdown();
	return EXIT_FAILURE;
}
