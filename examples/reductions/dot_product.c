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

/*
 * This computes the dot product of a big vector, using data reduction to
 * optimize the dot reduction.
 */

#include <starpu.h>
#include <assert.h>
#include <math.h>

#include <reductions/dot_product.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static float *_x;
static float *_y;
static starpu_data_handle_t *_x_handles;
static starpu_data_handle_t *_y_handles;
#ifdef STARPU_USE_OPENCL
static struct starpu_opencl_program _opencl_program;
#endif

#ifdef STARPU_QUICK_CHECK
static unsigned _nblocks = 512;
#else
static unsigned _nblocks = 4096;
#endif
static unsigned _entries_per_block = 1024;

static DOT_TYPE _dot = 0.0f;
static starpu_data_handle_t _dot_handle;

#ifdef STARPU_USE_CUDA
static int cublas_version;
#endif

static int can_execute(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	(void)task;
	(void)nimpl;
	enum starpu_worker_archtype type = starpu_worker_get_type(workerid);
	if (type == STARPU_CPU_WORKER || type == STARPU_OPENCL_WORKER || type == STARPU_MIC_WORKER)
		return 1;

#ifdef STARPU_USE_CUDA
#ifdef STARPU_SIMGRID
	/* We don't know, let's assume it can */
	return 1;
#else
	/* Cuda device */
	const struct cudaDeviceProp *props;
	props = starpu_cuda_get_device_properties(workerid);
	if (props->major >= 2 || props->minor >= 3)
		/* At least compute capability 1.3, supports doubles */
		return 1;
#endif
#endif
	/* Old card, does not support doubles */
	return 0;
}

/*
 *	Codelet to create a neutral element
 */

void init_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dot = 0.0f;
}

#ifdef STARPU_USE_CUDA
void init_cuda_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	cudaMemsetAsync(dot, 0, sizeof(DOT_TYPE), starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
void init_opencl_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        cl_int err;
	cl_command_queue queue;

	cl_mem dot = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
	starpu_opencl_get_current_queue(&queue);
	DOT_TYPE zero = (DOT_TYPE) 0.0;

	err = clEnqueueWriteBuffer(queue,
			dot,
			CL_TRUE,
			0,
			sizeof(DOT_TYPE),
			&zero,
			0,
			NULL,
			NULL);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

}
#endif

static struct starpu_codelet init_codelet =
{
	.can_execute = can_execute,
	.cpu_funcs = {init_cpu_func},
	.cpu_funcs_name = {"init_cpu_func"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {init_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {init_opencl_func},
#endif
	.modes = {STARPU_W},
	.nbuffers = 1,
	.name = "init",
};

/*
 *	Codelet to perform the reduction of two elements
 */

void redux_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	DOT_TYPE *dota = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	DOT_TYPE *dotb = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*dota = *dota + *dotb;
}

#ifdef STARPU_USE_CUDA
extern void redux_cuda_func(void *descr[], void *_args);
#endif

#ifdef STARPU_USE_OPENCL
void redux_opencl_func(void *buffers[], void *args)
{
	(void)args;
	int id, devid;
        cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_mem dota = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[0]);
	cl_mem dotb = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[1]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &_opencl_program, "_redux_opencl", devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(dota), &dota);
	err|= clSetKernelArg(kernel, 1, sizeof(dotb), &dotb);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
                size_t local=1;
                size_t s;
                cl_device_id device;

                starpu_opencl_get_device(devid, &device);

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
#endif

static struct starpu_codelet redux_codelet =
{
	.can_execute = can_execute,
	.cpu_funcs = {redux_cpu_func},
	.cpu_funcs_name = {"redux_cpu_func"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {redux_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {redux_opencl_func},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.modes = {STARPU_RW, STARPU_R},
	.nbuffers = 2,
	.name = "redux"
};

/*
 *	Dot product codelet
 */

void dot_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	float *local_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *local_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	DOT_TYPE local_dot = 0.0;

	unsigned i;
	for (i = 0; i < n; i++)
	{
		local_dot += (DOT_TYPE)local_x[i]*(DOT_TYPE)local_y[i];
	}

	*dot = *dot + local_dot;
}

#ifdef STARPU_USE_CUDA
void dot_cuda_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	DOT_TYPE current_dot;
	float local_dot;

	float *local_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *local_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	DOT_TYPE *dot = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[2]);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemcpyAsync(&current_dot, dot, sizeof(DOT_TYPE), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	cublasStatus_t status = cublasSdot(starpu_cublas_get_local_handle(), n, local_x, 1, local_y, 1, &local_dot);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	/* FPRINTF(stderr, "current_dot %f local dot %f -> %f\n", current_dot, local_dot, current_dot + local_dot); */
	current_dot += local_dot;

	cudaMemcpyAsync(dot, &current_dot, sizeof(DOT_TYPE), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
void dot_opencl_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	int id, devid;
        cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_mem x = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	cl_mem y = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[1]);
	cl_mem dot = (cl_mem) STARPU_VARIABLE_GET_PTR(buffers[2]);
	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &_opencl_program, "_dot_opencl", devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(x), &x);
	err|= clSetKernelArg(kernel, 1, sizeof(y), &y);
	err|= clSetKernelArg(kernel, 2, sizeof(dot), &dot);
	err|= clSetKernelArg(kernel, 3, sizeof(n), &n);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
                size_t local=1;
                size_t s;
                cl_device_id device;

                starpu_opencl_get_device(devid, &device);

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
#endif

static struct starpu_codelet dot_codelet =
{
	.can_execute = can_execute,
	.cpu_funcs = {dot_cpu_func},
	.cpu_funcs_name = {"dot_cpu_func"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {dot_cuda_func},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {dot_opencl_func},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_REDUX},
	.name = "dot"
};

/*
 *	Tasks initialization
 */

int main(void)
{
	int ret;

	/* Not supported yet */
	if (starpu_get_env_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return 77;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/reductions/dot_product_opencl_kernels.cl",
						  &_opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

#ifdef STARPU_USE_CUDA
	unsigned devices = starpu_cuda_worker_get_count();
	if (devices)
	{
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasGetVersion(handle, &cublas_version);
		cublasDestroy(handle);
		if (cublas_version >= 7050)
			starpu_cublas_init();
		else
			/* Disable the sdot cublas kernel, it is bogus with a
			 * non-blocking stream (Nvidia bugid 1669886) */
			dot_codelet.cuda_funcs[0] = NULL;
	}
#endif

	unsigned long nelems = _nblocks*_entries_per_block;
	size_t size = nelems*sizeof(float);

	_x = (float *) malloc(size);
	_y = (float *) malloc(size);

	_x_handles = (starpu_data_handle_t *) calloc(_nblocks, sizeof(starpu_data_handle_t));
	_y_handles = (starpu_data_handle_t *) calloc(_nblocks, sizeof(starpu_data_handle_t));

	assert(_x && _y);

        starpu_srand48(0);

	DOT_TYPE reference_dot = 0.0;

	unsigned long i;
	for (i = 0; i < nelems; i++)
	{
		_x[i] = (float)starpu_drand48();
		_y[i] = (float)starpu_drand48();

		reference_dot += (DOT_TYPE)_x[i]*(DOT_TYPE)_y[i];
	}

	unsigned block;
	for (block = 0; block < _nblocks; block++)
	{
		starpu_vector_data_register(&_x_handles[block], STARPU_MAIN_RAM,
			(uintptr_t)&_x[_entries_per_block*block], _entries_per_block, sizeof(float));
		starpu_vector_data_register(&_y_handles[block], STARPU_MAIN_RAM,
			(uintptr_t)&_y[_entries_per_block*block], _entries_per_block, sizeof(float));
	}

	starpu_variable_data_register(&_dot_handle, STARPU_MAIN_RAM, (uintptr_t)&_dot, sizeof(DOT_TYPE));

	/*
	 *	Compute dot product with StarPU
	 */
	starpu_data_set_reduction_methods(_dot_handle, &redux_codelet, &init_codelet);

	for (block = 0; block < _nblocks; block++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &dot_codelet;
		task->destroy = 1;

		task->handles[0] = _x_handles[block];
		task->handles[1] = _y_handles[block];
		task->handles[2] = _dot_handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_ASSERT(!ret);
	}

	for (block = 0; block < _nblocks; block++)
	{
		starpu_data_unregister(_x_handles[block]);
		starpu_data_unregister(_y_handles[block]);
	}
	starpu_data_unregister(_dot_handle);

	FPRINTF(stderr, "Reference : %e vs. %e (Delta %e)\n", reference_dot, _dot, reference_dot - _dot);

#ifdef STARPU_USE_CUDA
	if (cublas_version >= 7050)
		starpu_cublas_shutdown();
#endif

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_unload_opencl(&_opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	starpu_shutdown();

	free(_x);
	free(_y);
	free(_x_handles);
	free(_y_handles);

	if (fabs(reference_dot - _dot) < reference_dot * 1e-6)
		return EXIT_SUCCESS;
	else
	{
		FPRINTF(stderr, "ERROR: fabs(%e - %e) >= %e * 1e-6\n", reference_dot, _dot, reference_dot);
		return EXIT_FAILURE;
	}

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 77;
}
