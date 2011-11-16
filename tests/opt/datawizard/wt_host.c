/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
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
#include "../../common/helper.h"

#ifdef STARPU_USE_CUDA
#include <starpu_cuda.h>
#endif
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif


static unsigned var = 0;
static starpu_data_handle handle;
/*
 *	Increment codelet
 */

#ifdef STARPU_USE_OPENCL
/* dummy OpenCL implementation */
static void increment_opencl_kernel(void *descr[], void *cl_arg __attribute__((unused)))
{
	cl_mem d_token = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned h_token;

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueReadBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
	h_token++;
	clEnqueueWriteBuffer(queue, d_token, CL_TRUE, 0, sizeof(unsigned), (void *)&h_token, 0, NULL, NULL);
}
#endif


#ifdef STARPU_USE_CUDA
static void increment_cuda_kernel(void *descr[], void *arg)
{
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned host_token;

	/* This is a dummy technique of course */
	cudaMemcpy(&host_token, tokenptr, sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	host_token++;

	cudaMemcpy(tokenptr, &host_token, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}
#endif

static void increment_cpu_kernel(void *descr[], void *arg)
{
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*tokenptr = *tokenptr + 1;
}

static starpu_codelet increment_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
#ifdef STARPU_USE_CUDA
	.cuda_func = increment_cuda_kernel,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = increment_opencl_kernel,
#endif
	.cpu_func = increment_cpu_kernel,
	.nbuffers = 1
};

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, 0, (uintptr_t)&var, sizeof(unsigned));

	/* Copy the handle in main memory every time it is modified */
	uint32_t wt_mask = (1<<0);
	starpu_data_set_wt_mask(handle, wt_mask);

	unsigned ntasks = 2;

	unsigned t;

	for (t = 0; t < ntasks; t++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &increment_cl;

		task->buffers[0].mode = STARPU_RW;
		task->buffers[0].handle = handle;

		int ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(handle);

	if (var != ntasks)
		fprintf(stderr, "VAR is %d should be %d\n", var, ntasks);

	STARPU_ASSERT(var == ntasks);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return 77;
}
