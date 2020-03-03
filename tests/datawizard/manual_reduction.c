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
 * Allocate one buffer per worker, doing computations with it, and
 * eventually reducing it into a single buffer
 */

#define INIT_VALUE	42
#define NTASKS		10000

static unsigned variable;
static starpu_data_handle_t variable_handle;

static uintptr_t per_worker[STARPU_NMAXWORKERS];
static starpu_data_handle_t per_worker_handle[STARPU_NMAXWORKERS];

static unsigned ndone;

/* Create per-worker handles */
static void initialize_per_worker_handle(void *arg)
{
	(void)arg;
	int workerid = starpu_worker_get_id_check();

	/* Allocate memory on the worker, and initialize it to 0 */
	switch (starpu_worker_get_type(workerid))
	{
		case STARPU_CPU_WORKER:
			per_worker[workerid] = (uintptr_t)calloc(1, sizeof(variable));
			break;
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_WORKER:
			{
			cl_context context;
			cl_command_queue queue;
			starpu_opencl_get_current_context(&context);
			starpu_opencl_get_current_queue(&queue);

			cl_mem ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(variable), NULL, NULL);
			/* Poor's man memset */
			unsigned zero = 0;
			clEnqueueWriteBuffer(queue, ptr, CL_FALSE, 0, sizeof(variable), (void *)&zero, 0, NULL, NULL);
			clFinish(queue);
			per_worker[workerid] = (uintptr_t)ptr;
			}

			break;
#endif
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:
		{
		     	cudaError_t status;
			status = cudaMalloc((void **)&per_worker[workerid], sizeof(variable));
			if (!per_worker[workerid] || (status != cudaSuccess))
			{
				STARPU_CUDA_REPORT_ERROR(status);
			}
			status = cudaMemsetAsync((void *)per_worker[workerid], 0, sizeof(variable), starpu_cuda_get_local_stream());
			if (!status)
				status = cudaStreamSynchronize(starpu_cuda_get_local_stream());
			if (status)
				STARPU_CUDA_REPORT_ERROR(status);
			break;
		}
#endif
		default:
			STARPU_ABORT();
			break;
	}
	FPRINTF(stderr, "worker %d got data %lx\n", workerid, (unsigned long) per_worker[workerid]);

	STARPU_ASSERT(per_worker[workerid]);
}

/*
 *	Implement reduction method
 */

void cpu_redux_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned *a = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *b = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	FPRINTF(stderr, "%u = %u + %u\n", *a + *b, *a, *b);

	*a = *a + *b;
}

static struct starpu_codelet reduction_codelet =
{
	.cpu_funcs = {cpu_redux_func},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = NULL
};

/*
 *	Use per-worker local copy
 */

void cpu_func_incr(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*val = *val + 1;
	STARPU_ATOMIC_ADD(&ndone, 1);
}

#ifdef STARPU_USE_CUDA
/* dummy CUDA implementation */
static void cuda_func_incr(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	unsigned h_val, h_val2;
	cudaError_t status;
	status = cudaMemcpyAsync(&h_val, val, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);
	status = cudaStreamSynchronize(starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);
	h_val++;
	status = cudaMemcpyAsync(val, &h_val, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);
	status = cudaStreamSynchronize(starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);

	status = cudaMemcpyAsync(&h_val2, val, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);
	status = cudaStreamSynchronize(starpu_cuda_get_local_stream());
	if (status)
		STARPU_CUDA_REPORT_ERROR(status);
	STARPU_ASSERT_MSG(h_val2 == h_val, "%lx should be %u, not %u, I have just written it ?!\n", (unsigned long)(uintptr_t) val, h_val, h_val2);

	STARPU_ATOMIC_ADD(&ndone, 1);
}
#endif

#ifdef STARPU_USE_OPENCL
/* dummy OpenCL implementation */
static void opencl_func_incr(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	STARPU_SKIP_IF_VALGRIND;

	cl_mem d_val = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned h_val;

	cl_command_queue queue;
	starpu_opencl_get_current_queue(&queue);

	clEnqueueReadBuffer(queue, d_val, CL_FALSE, 0, sizeof(unsigned), (void *)&h_val, 0, NULL, NULL);
	clFinish(queue);
	h_val++;
	clEnqueueWriteBuffer(queue, d_val, CL_FALSE, 0, sizeof(unsigned), (void *)&h_val, 0, NULL, NULL);
	clFinish(queue);
	STARPU_ATOMIC_ADD(&ndone, 1);
}
#endif

static struct starpu_codelet use_data_on_worker_codelet =
{
	.cpu_funcs = {cpu_func_incr},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_func_incr},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_func_incr},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = NULL
};

int main(int argc, char **argv)
{
	unsigned worker;
	unsigned i;
	int ret;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	variable = INIT_VALUE;

        ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned nworkers = starpu_worker_get_count();

	starpu_variable_data_register(&variable_handle, STARPU_MAIN_RAM, (uintptr_t)&variable, sizeof(unsigned));

	/* Allocate a per-worker handle on each worker (and initialize it to 0) */
	starpu_execute_on_each_worker(initialize_per_worker_handle, NULL, STARPU_CPU|STARPU_CUDA|STARPU_OPENCL);

	/* Register all per-worker handles */
	for (worker = 0; worker < nworkers; worker++)
	{
		STARPU_ASSERT(per_worker[worker]);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		starpu_variable_data_register(&per_worker_handle[worker], memory_node,
						per_worker[worker], sizeof(variable));
	}

	/* Submit NTASKS tasks to the different worker to simulate the usage of a data in reduction */
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &use_data_on_worker_codelet;

		int workerid = (i % nworkers);
		task->handles[0] = per_worker_handle[workerid];

		task->execute_on_a_specific_worker = 1;
		task->workerid = (unsigned)workerid;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}


	/* Perform the reduction of all per-worker handles into the variable_handle */
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &reduction_codelet;

		task->handles[0] = variable_handle;
		task->handles[1] = per_worker_handle[worker];

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(variable_handle);

	/* Destroy all per-worker handles */
	for (worker = 0; worker < nworkers; worker++)
	{
		starpu_data_unregister_no_coherency(per_worker_handle[worker]);
		switch(starpu_worker_get_type(worker))
		{
		case STARPU_CPU_WORKER:
			free((void*)per_worker[worker]);
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_WORKER:
			cudaFree((void*)per_worker[worker]);
			break;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_WORKER:
			clReleaseMemObject((void*)per_worker[worker]);
			break;
#endif /* !STARPU_USE_OPENCL */
		default:
			STARPU_ABORT();
		}
	}

	starpu_shutdown();

	if (variable == INIT_VALUE + NTASKS)
		ret = EXIT_SUCCESS;
	else
	{
		FPRINTF(stderr, "%u != %d + %d\n", variable, INIT_VALUE, NTASKS);
		FPRINTF(stderr, "ndone: %u\n", ndone);
		ret = EXIT_FAILURE;
	}
	STARPU_RETURN(ret);

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	starpu_task_wait_for_all();
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
