/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "scal.h"

/*
 * Register a handle from a GPU buffer, and performs a partitioned operation
 */

#if ! (defined(STARPU_USE_OPENCL) || defined(STARPU_USE_CUDA))
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

static int
submit_tasks(starpu_data_handle_t handle, int pieces, int n)
{
	int i, ret;

	for (i = 0; i < pieces; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
		task->cl = &scal_codelet;
		task->execute_on_a_specific_worker = 1;
		task->workerid = i%n;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
			return -ENODEV;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	return 0;
}

static int
find_a_worker(enum starpu_worker_archtype type)
{
	int worker[STARPU_NMAXWORKERS];
	int ret = starpu_worker_get_ids_by_type(type, worker, STARPU_NMAXWORKERS);
	if (ret == 0)
		return -ENODEV;
	if (ret == -ERANGE)
		return worker[STARPU_NMAXWORKERS-1];
	return worker[ret-1];
}

static int
check_result(unsigned *t, size_t size)
{
	unsigned i;
	for (i = 0; i < size; i++)
	{
		if (t[i] != i*2)
		{
			FPRINTF(stderr,"t[%u] is %u instead of %u\n", i, t[i], 2*i);
			return 1;
		}
	}
	return 0;
}

#ifdef STARPU_USE_CUDA
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
static int
test_cuda(void)
{
	int ret;
	unsigned *foo_gpu;
	unsigned *foo;
	int n, i, size, pieces;
	int devid;
	int chosen;
	cudaError_t cures;
	starpu_data_handle_t handle;

	/* Find a CUDA worker */
	chosen = find_a_worker(STARPU_CUDA_WORKER);
	if (chosen == -ENODEV)
		return -ENODEV;

	n = starpu_worker_get_count();
	size = 10 * n;

	devid = starpu_worker_get_devid(chosen);
	foo_gpu = (void*) starpu_malloc_on_node(starpu_worker_get_memory_node(chosen), size * sizeof(*foo_gpu));

	foo = calloc(size, sizeof(*foo));
	for (i = 0; i < size; i++)
		foo[i] = i;

	cures = cudaMemcpy(foo_gpu, foo, size * sizeof(*foo_gpu), cudaMemcpyHostToDevice);
	if (!cures)
		cures = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	starpu_vector_data_register(&handle, starpu_worker_get_memory_node(chosen), (uintptr_t)foo_gpu, size, sizeof(*foo_gpu));

	/* Broadcast the data to force in-place partitioning */
	for (i = 0; i < n; i++)
		starpu_data_prefetch_on_node(handle, starpu_worker_get_memory_node(i), 0);

	/* Even with just one worker, split in at least two */
	if (n == 1)
		pieces = 2;
	else
		pieces = n;

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = pieces,
	};

	starpu_data_partition(handle, &f);

	ret = submit_tasks(handle, pieces, n);
	if (ret == -ENODEV)
	{
		starpu_free_on_node(starpu_worker_get_memory_node(chosen), (uintptr_t) foo_gpu, size * sizeof(*foo_gpu));
		free(foo);
		return -ENODEV;
	}

	starpu_data_unpartition(handle, starpu_worker_get_memory_node(chosen));
	starpu_data_unregister(handle);

	starpu_cuda_set_device(devid);
	cures = cudaMemcpy(foo, foo_gpu, size * sizeof(*foo_gpu), cudaMemcpyDeviceToHost);
	if (!cures)
		cures = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
	{
		starpu_free_on_node(starpu_worker_get_memory_node(chosen), (uintptr_t) foo_gpu, size * sizeof(*foo_gpu));
		free(foo);
		STARPU_CUDA_REPORT_ERROR(cures);
		return 1;
	}

	ret = check_result(foo, size);
	starpu_free_on_node(starpu_worker_get_memory_node(chosen), (uintptr_t) foo_gpu, size * sizeof(*foo_gpu));
	free(foo);
	return ret;
}
#endif
#endif

#ifdef STARPU_USE_OPENCL
static int
test_opencl(void)
{
	int i;
	int ret;
	int chosen;
	int n;
	int size;
	int pieces;
	cl_mem foo_gpu;
	starpu_data_handle_t handle;

	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/scal_opencl.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");

	/* Find an OpenCL worker */
	chosen = find_a_worker(STARPU_OPENCL_WORKER);
	if (chosen == -ENODEV)
		return -ENODEV;

	n = starpu_worker_get_count();
	size = 10 * n;

	int devid;
	cl_int err;
	cl_context context;
	cl_command_queue queue;

	devid = starpu_worker_get_devid(chosen);

	starpu_opencl_get_context(devid, &context);
	starpu_opencl_get_queue(devid, &queue);

	foo_gpu = (void*) starpu_malloc_on_node(starpu_worker_get_memory_node(chosen), size * sizeof(int));

	unsigned int *foo = malloc(size*sizeof(*foo));
	for (i = 0; i < size; i++)
		foo[i] = i;

	err = clEnqueueWriteBuffer(queue,
				   foo_gpu,
				   CL_FALSE,
				   0,
				   size*sizeof(int),
				   foo,
				   0,
				   NULL,
				   NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	clFinish(queue);

	starpu_vector_data_register(&handle,
				    starpu_worker_get_memory_node(chosen),
				    (uintptr_t)foo_gpu,
				    size,
				    sizeof(int));

	/* Broadcast the data to force in-place partitioning */
	for (i = 0; i < n; i++)
		starpu_data_prefetch_on_node(handle, starpu_worker_get_memory_node(i), 0);

	/* Even with just one worker, split in at least two */
	if (n == 1)
		pieces = 2;
	else
		pieces = n;

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = pieces,
	};

	starpu_data_partition(handle, &f);

	ret = submit_tasks(handle, pieces, n);
	if (ret == -ENODEV)
		return -ENODEV;

	starpu_data_unpartition(handle, starpu_worker_get_memory_node(chosen));
	starpu_data_unregister(handle);

	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");

	err = clEnqueueReadBuffer(queue,
				  foo_gpu,
				  CL_FALSE,
				  0,
				  size*sizeof(*foo),
				  foo,
				  0,
				  NULL,
				  NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	clFinish(queue);
	ret = check_result(foo, size);
	starpu_free_on_node(starpu_worker_get_memory_node(chosen), (uintptr_t) foo_gpu, size * sizeof(int));
	free(foo);
	return ret;
}
#endif /* !STARPU_USE_OPENCL */

int main(int argc, char **argv)
{
	int skipped_cuda = 1, skipped_opencl = 1;
	int ret;
	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/scal_opencl.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

#ifdef STARPU_USE_CUDA
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	ret = test_cuda();
	if (ret == 1)
		goto fail;
	else if (ret == 0)
		skipped_cuda = 0;
#endif
#endif

#ifdef STARPU_USE_OPENCL
	ret = test_opencl();
	if (ret == 1)
		goto fail;
	else if (ret == 0)
		skipped_opencl = 0;
#endif

	starpu_shutdown();

	if (skipped_cuda == 1 && skipped_opencl == 1)
		return STARPU_TEST_SKIPPED;

	return EXIT_SUCCESS;

fail:
	starpu_shutdown();
	return EXIT_FAILURE;
}

#endif /* defined(STARPU_USE_OPENCL) || defined(STARPU_USE_CUDA) */
