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

#include <errno.h>

#include <core/workers.h>
#include <common/config.h>
#include <starpu.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

#ifdef STARPU_USE_OPENCL
#include <drivers/opencl/driver_opencl.h>
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
struct malloc_pinned_codelet_struct {
	void **ptr;
	size_t dim;
};
#endif

//#ifdef STARPU_USE_OPENCL
//static void malloc_pinned_opencl_codelet(void *buffers[] __attribute__((unused)), void *arg)
//{
//	struct malloc_pinned_codelet_struct *s = arg;
//        //        *(s->ptr) = malloc(s->dim);
//        _starpu_opencl_allocate_memory((void **)(s->ptr), s->dim, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR);
//}
//#endif

#ifdef STARPU_USE_CUDA
static void malloc_pinned_cuda_codelet(void *buffers[] __attribute__((unused)), void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;

	cudaError_t cures;
	cures = cudaHostAlloc((void **)(s->ptr), s->dim, cudaHostAllocPortable);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

#if defined(STARPU_USE_CUDA)// || defined(STARPU_USE_OPENCL)
static starpu_codelet malloc_pinned_cl = {
#ifdef STARPU_USE_CUDA
	.cuda_func = malloc_pinned_cuda_codelet,
#endif
//#ifdef STARPU_USE_OPENCL
//	.opencl_func = malloc_pinned_opencl_codelet,
//#endif
	.model = NULL,
	.nbuffers = 0
};
#endif

int starpu_data_malloc_pinned_if_possible(void **A, size_t dim)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	STARPU_ASSERT(A);

	if (_starpu_may_submit_cuda_task())
	{
#ifdef STARPU_USE_CUDA
		int push_res;

		struct malloc_pinned_codelet_struct s = {
			.ptr = A,
			.dim = dim
		};

                malloc_pinned_cl.where = STARPU_CUDA;
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL;
			task->cl = &malloc_pinned_cl;
			task->cl_arg = &s;

		task->synchronous = 1;

		push_res = starpu_task_submit(task);
		STARPU_ASSERT(push_res != -ENODEV);
#endif
	}
//	else if (_starpu_may_submit_opencl_task())
//	{
//#ifdef STARPU_USE_OPENCL
//		int push_res;
//
//		struct malloc_pinned_codelet_struct s = {
//			.ptr = A,
//			.dim = dim
//		};
//
//                malloc_pinned_cl.where = STARPU_OPENCL;
//		struct starpu_task *task = starpu_task_create();
//			task->callback_func = NULL;
//			task->cl = &malloc_pinned_cl;
//			task->cl_arg = &s;
//
//		task->synchronous = 1;
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//#endif
//        }
        else {
		*A = malloc(dim);
	}

	STARPU_ASSERT(*A);

	return 0;
}

#ifdef STARPU_USE_CUDA
static void free_pinned_cuda_codelet(void *buffers[] __attribute__((unused)), void *arg)
{
	cudaError_t cures;
	cures = cudaFreeHost(arg);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

//#ifdef STARPU_USE_OPENCL
//static void free_pinned_opencl_codelet(void *buffers[] __attribute__((unused)), void *arg)
//{
//        //        free(arg);
//        int err = clReleaseMemObject(arg);
//        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
//}
//#endif

#if defined(STARPU_USE_CUDA) // || defined(STARPU_USE_OPENCL)
static starpu_codelet free_pinned_cl = {
#ifdef STARPU_USE_CUDA
	.cuda_func = free_pinned_cuda_codelet,
#endif
//#ifdef STARPU_USE_OPENCL
//	.opencl_func = free_pinned_opencl_codelet,
//#endif
	.model = NULL,
	.nbuffers = 0
};
#endif

int starpu_data_free_pinned_if_possible(void *A)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	if (_starpu_may_submit_cuda_task())
	{
#ifdef STARPU_USE_CUDA
		int push_res;

                free_pinned_cl.where = STARPU_CUDA;
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL;
			task->cl = &free_pinned_cl;
			task->cl_arg = A;

		task->synchronous = 1;

		push_res = starpu_task_submit(task);
		STARPU_ASSERT(push_res != -ENODEV);
#endif
	}
//	else if (_starpu_may_submit_opencl_task())
//	{
//#ifdef STARPU_USE_OPENCL
//		int push_res;
//
//                free_pinned_cl.where = STARPU_OPENCL;
//		struct starpu_task *task = starpu_task_create();
//			task->callback_func = NULL;
//			task->cl = &free_pinned_cl;
//			task->cl_arg = A;
//
//		task->synchronous = 1;
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//#endif
//	}
	else {
		free(A);
	}

	return 0;
}
