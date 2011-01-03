/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif
#ifdef STARPU_USE_OPENCL
#include <CL/cl.h>
#endif


static unsigned var = 0;
static starpu_data_handle handle;

/*
 *	Reduction methods
 */

#ifdef STARPU_USE_CUDA
static void redux_cuda_kernel(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	unsigned host_dst, host_src;

	/* This is a dummy technique of course */
	cudaMemcpy(&host_src, src, sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_dst, dst, sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	host_dst += host_src;

	cudaMemcpy(dst, &host_dst, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

static void neutral_cuda_kernel(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* This is a dummy technique of course */
	unsigned host_dst = 0;
	cudaMemcpy(dst, &host_dst, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}
#endif

#ifdef STARPU_USE_OPENCL
static void redux_opencl_kernel(void *descr[], void *arg)
{
	unsigned h_dst, h_src;

	cl_mem d_dst = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
	cl_mem d_src = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[1]);

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	/* This is a dummy technique of course */
	clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0, sizeof(unsigned), (void *)&h_dst, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_src, CL_TRUE, 0, sizeof(unsigned), (void *)&h_src, 0, NULL, NULL);

	h_dst += h_src;

	clEnqueueWriteBuffer(queue, d_dst, CL_TRUE, 0, sizeof(unsigned), (void *)&h_dst, 0, NULL, NULL); 
}

static void neutral_opencl_kernel(void *descr[], void *arg)
{
	unsigned h_dst = 0;
	cl_mem d_dst = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueWriteBuffer(queue, d_dst, CL_TRUE, 0, sizeof(unsigned), (void *)&h_dst, 0, NULL, NULL); 
}
#endif



static void redux_cpu_kernel(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*dst = *dst + *src;
}

static void neutral_cpu_kernel(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dst = 0;
}

static starpu_codelet redux_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
#ifdef STARPU_USE_CUDA
	.cuda_func = redux_cuda_kernel,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = redux_opencl_kernel,
#endif
	.cpu_func = redux_cpu_kernel,
	.nbuffers = 2
};

static starpu_codelet neutral_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
#ifdef STARPU_USE_CUDA
	.cuda_func = neutral_cuda_kernel,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func = neutral_opencl_kernel,
#endif
	.cpu_func = neutral_cpu_kernel,
	.nbuffers = 1
};

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
	starpu_init(NULL);

	starpu_variable_data_register(&handle, 0, (uintptr_t)&var, sizeof(unsigned));

	starpu_data_set_reduction_methods(handle, &redux_cl, &neutral_cl);

	unsigned ntasks = 1024;
	unsigned nloops = 16;

	unsigned loop;
	unsigned t;

	for (loop = 0; loop < nloops; loop++)
	{
		for (t = 0; t < ntasks; t++)
		{
			struct starpu_task *task = starpu_task_create();
	
			task->cl = &increment_cl;
	
			task->buffers[0].mode = (t % 10 == 0)?STARPU_RW:STARPU_REDUX;
			task->buffers[0].handle = handle;
	
			int ret = starpu_task_submit(task);
			STARPU_ASSERT(!ret);

		}

		starpu_data_acquire(handle, STARPU_R);
		STARPU_ASSERT(var == ntasks*(loop + 1));
		starpu_data_release(handle);
	}

	starpu_data_unregister(handle);
	STARPU_ASSERT(var == ntasks*nloops);
	
	starpu_shutdown();

	return 0;
}
