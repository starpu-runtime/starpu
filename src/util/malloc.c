/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <errno.h>

#include <core/workers.h>
#include <common/config.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <drivers/opencl/driver_opencl.h>

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
struct malloc_pinned_codelet_struct
{
	void **ptr;
	size_t dim;
};
#endif

//#ifdef STARPU_USE_OPENCL
//static void malloc_pinned_opencl_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
//{
//	struct malloc_pinned_codelet_struct *s = arg;
//        //        *(s->ptr) = malloc(s->dim);
//        _starpu_opencl_allocate_memory((void **)(s->ptr), s->dim, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR);
//}
//#endif

#ifdef STARPU_USE_CUDA
static void malloc_pinned_cuda_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;

	cudaError_t cures;
	cures = cudaHostAlloc((void **)(s->ptr), s->dim, cudaHostAllocPortable);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

#if defined(STARPU_USE_CUDA)// || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel malloc_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "malloc_pinned"
};

static struct starpu_codelet malloc_pinned_cl =
{
	.cuda_func = malloc_pinned_cuda_codelet,
//#ifdef STARPU_USE_OPENCL
//	.opencl_func = malloc_pinned_opencl_codelet,
//#endif
	.nbuffers = 0,
	.model = &malloc_pinned_model
};
#endif

int starpu_malloc(void **A, size_t dim)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	STARPU_ASSERT(A);

	if (_starpu_can_submit_cuda_task())
	{
#ifdef STARPU_USE_CUDA
		int push_res;

		struct malloc_pinned_codelet_struct s =
		{
			.ptr = A,
			.dim = dim
		};

                malloc_pinned_cl.where = STARPU_CUDA;
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL;
			task->cl = &malloc_pinned_cl;
			task->cl_arg = &s;

		task->synchronous = 1;

		_starpu_exclude_task_from_dag(task);

		push_res = starpu_task_submit(task);
		STARPU_ASSERT(push_res != -ENODEV);
#endif
	}
//	else if (_starpu_can_submit_opencl_task())
//	{
//#ifdef STARPU_USE_OPENCL
//		int push_res;
//
//		struct malloc_pinned_codelet_struct s =
//		{
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
//		_starpu_exclude_task_from_dag(task);
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//#endif
//        }
        else
	{
		*A = malloc(dim);
	}

	STARPU_ASSERT(*A);

	return 0;
}

#ifdef STARPU_USE_CUDA
static void free_pinned_cuda_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	cudaError_t cures;
	cures = cudaFreeHost(arg);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

//#ifdef STARPU_USE_OPENCL
//static void free_pinned_opencl_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
//{
//        //        free(arg);
//        int err = clReleaseMemObject(arg);
//        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
//}
//#endif

#if defined(STARPU_USE_CUDA) // || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel free_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "free_pinned"
};

static struct starpu_codelet free_pinned_cl =
{
	.cuda_func = free_pinned_cuda_codelet,
//#ifdef STARPU_USE_OPENCL
//	.opencl_func = free_pinned_opencl_codelet,
//#endif
	.nbuffers = 0,
	.model = &free_pinned_model
};
#endif

int starpu_free(void *A)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	if (_starpu_can_submit_cuda_task())
	{
#ifdef STARPU_USE_CUDA
		int push_res;

                free_pinned_cl.where = STARPU_CUDA;
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL;
			task->cl = &free_pinned_cl;
			task->cl_arg = A;

		task->synchronous = 1;

		_starpu_exclude_task_from_dag(task);

		push_res = starpu_task_submit(task);
		STARPU_ASSERT(push_res != -ENODEV);
#endif
	}
//	else if (_starpu_can_submit_opencl_task())
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
//		_starpu_exclude_task_from_dag(task);
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//#endif
//	}
	else
	{
		free(A);
	}

	return 0;
}

/* Internal convenience function, used by code generated by the GCC
 * plug-in.  */
void _starpu_free_unref(void *p)
{
	(void)starpu_free(* (void **)p);
}
