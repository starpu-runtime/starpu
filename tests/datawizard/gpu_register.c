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
#include <starpu_opencl.h>
#include "../helper.h"

void scal_func_cpu(void *buffers[], void *cl_arg)
{
	unsigned i;

	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];
	unsigned *val = (unsigned *) STARPU_VECTOR_GET_PTR(vector);
	unsigned n = STARPU_VECTOR_GET_NX(vector);

	/* scale the vector */
	for (i = 0; i < n; i++)
		val[i] *= 2;
}

#ifdef STARPU_USE_CUDA
extern void scal_func_cuda(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_OPENCL
static struct starpu_opencl_program opencl_program;

void scal_func_opencl(void *buffers[], void *_args)
{
	int id, devid;
        cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;
	cl_event event;

	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
	cl_mem val = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned offset = STARPU_VECTOR_GET_OFFSET(buffers[0]);

	id = starpu_worker_get_id();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "vector_mult_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(val), &val);
	err |= clSetKernelArg(kernel, 1, sizeof(offset), &offset);
	err |= clSetKernelArg(kernel, 2, sizeof(n), &n);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=n;
		size_t local;
                size_t s;
                cl_device_id device;

                starpu_opencl_get_device(devid, &device);

                err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
                if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
                if (local > global) local=global;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);
	starpu_opencl_collect_stats(event);
	clReleaseEvent(event);

	starpu_opencl_release_kernel(kernel);
}
#endif

static struct starpu_codelet codelet =
{
        .where = STARPU_CPU
#ifdef STARPU_USE_CUDA
		| STARPU_CUDA
#endif
#ifdef STARPU_USE_OPENCL
		| STARPU_OPENCL
#endif
		,
	.cpu_funcs = { scal_func_cpu, NULL },
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = { scal_func_opencl, NULL },
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = { scal_func_cuda, NULL },
#endif
	.modes = { STARPU_RW },
        .model = NULL,
        .nbuffers = 1
};


int main(int argc, char **argv)
{
	unsigned *foo_gpu;
	unsigned *foo;
	starpu_data_handle_t handle;
	int ret;
	int n, i, size;
	unsigned workerid;
	int chosen = -1;
	int devid;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* TODO OpenCL, too */
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++) {
		if (starpu_worker_get_type(workerid) == STARPU_CUDA_WORKER) {
			chosen = workerid;
			break;
		}
	}

	if (chosen == -1)
		return STARPU_TEST_SKIPPED;

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_file("tests/datawizard/scal_opencl.cl", &opencl_program, NULL);
#endif

	n = starpu_worker_get_count();
	size = 10 * n;

	devid = starpu_worker_get_devid(chosen);
	cudaSetDevice(devid);
	cudaMalloc((void**)&foo_gpu, size * sizeof(*foo_gpu));

	foo = calloc(size, sizeof(*foo));
	for (i = 0; i < size; i++)
		foo[i] = i;

	cudaMemcpy(foo_gpu, foo, size * sizeof(*foo_gpu), cudaMemcpyHostToDevice);

	starpu_vector_data_register(&handle, starpu_worker_get_memory_node(chosen), (uintptr_t)foo_gpu, size, sizeof(*foo_gpu));

	/* Broadcast the data to force in-place partitioning */
	for (i = 0; i < n; i++)
		starpu_data_prefetch_on_node(handle, starpu_worker_get_memory_node(i), 0);

	struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_func_vector,
		.nchildren = n > 1 ? n : 2,
	};

	starpu_data_partition(handle, &f);

	for (i = 0; i < n; i++) {
		struct starpu_task *task = starpu_task_create();

		task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
		task->cl = &codelet;
		task->execute_on_a_specific_worker = 1;
		task->workerid = i;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unpartition(handle, 0);
	starpu_data_unregister(handle);

	cudaMemcpy(foo, foo_gpu, size * sizeof(*foo_gpu), cudaMemcpyDeviceToHost);

	starpu_shutdown();

	for (i = 0; i < size; i++) {
		if (foo[i] != i*2) {
			fprintf(stderr,"value %d is %d instead of %d\n", i, foo[i], 2*i);
			return EXIT_FAILURE;
		}
	}

        return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
