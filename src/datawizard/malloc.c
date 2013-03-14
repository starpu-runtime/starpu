/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2010, 2012-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <drivers/opencl/driver_opencl.h>
#include <datawizard/memory_manager.h>

static size_t malloc_align = sizeof(void*);

void starpu_malloc_set_align(size_t align)
{
	STARPU_ASSERT_MSG(!(align & (align - 1)), "Alignment given to starpu_malloc_set_align must be a power of two");
	if (malloc_align < align)
		malloc_align = align;
}

#if (defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER))// || defined(STARPU_USE_OPENCL)
struct malloc_pinned_codelet_struct
{
	void **ptr;
	size_t dim;
};
#endif

/* Would be difficult to do it this way, we need to remember the cl_mem to be able to free it later... */

//#ifdef STARPU_USE_OPENCL
//static void malloc_pinned_opencl_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
//{
//	struct malloc_pinned_codelet_struct *s = arg;
//        //        *(s->ptr) = malloc(s->dim);
//        starpu_opencl_allocate_memory((void **)(s->ptr), s->dim, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR);
//}
//#endif

#if defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
static void malloc_pinned_cuda_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;

	cudaError_t cures;
	cures = cudaHostAlloc((void **)(s->ptr), s->dim, cudaHostAllocPortable);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

#if (defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER)) && !defined(STARPU_SIMGRID)// || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel malloc_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "malloc_pinned"
};

static struct starpu_codelet malloc_pinned_cl =
{
	.cuda_funcs = {malloc_pinned_cuda_codelet, NULL},
//#ifdef STARPU_USE_OPENCL
//	.opencl_funcs = {malloc_pinned_opencl_codelet, NULL},
//#endif
	.nbuffers = 0,
	.model = &malloc_pinned_model
};
#endif

int starpu_malloc(void **A, size_t dim)
{
#ifdef STARPU_DEVEL
#warning TODO: we need to request _starpu_memory_manager_can_allocate_size()
#warning TODO: if it fails, we should reclaim memory
#endif

	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	STARPU_ASSERT(A);

#ifndef STARPU_SIMGRID
	if (_starpu_can_submit_cuda_task())
	{
#ifdef STARPU_USE_CUDA
#ifdef HAVE_CUDA_MEMCPY_PEER
		cudaError_t cures;
		cures = cudaHostAlloc(A, dim, cudaHostAllocPortable);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
#else
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

		push_res = _starpu_task_submit_internally(task);
		STARPU_ASSERT(push_res != -ENODEV);
#endif /* HAVE_CUDA_MEMCPY_PEER */
#endif /* STARPU_USE_CUDA */
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
//		push_res = _starpu_task_submit_internally(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//#endif /* STARPU_USE_OPENCL */
//        }
        else
#endif /* STARPU_SIMGRID */
	{
#ifdef STARPU_HAVE_POSIX_MEMALIGN
		if (malloc_align != sizeof(void*))
		{
			if (posix_memalign(A, malloc_align, dim))
				*A = NULL;
		}
		else
#elif defined(STARPU_HAVE_MEMALIGN)
		if (malloc_align != sizeof(void*))
		{
			*A = memalign(malloc_align, dim);
		}
		else
#endif /* STARPU_HAVE_POSIX_MEMALIGN */
		{
			*A = malloc(dim);
		}
	}

	STARPU_ASSERT(*A);

	return 0;
}

#if defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
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

#if defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID) // || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel free_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "free_pinned"
};

static struct starpu_codelet free_pinned_cl =
{
	.cuda_funcs = {free_pinned_cuda_codelet, NULL},
//#ifdef STARPU_USE_OPENCL
//	.opencl_funcs = {free_pinned_opencl_codelet, NULL},
//#endif
	.nbuffers = 0,
	.model = &free_pinned_model
};
#endif

int starpu_free(void *A)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

#ifndef STARPU_SIMGRID
#ifdef STARPU_USE_CUDA
	if (_starpu_can_submit_cuda_task())
	{
#ifndef HAVE_CUDA_MEMCPY_PEER
	if (!_starpu_is_initialized())
	{
#endif
		/* This is especially useful when starpu_free is called from
 		 * the GCC-plugin. starpu_shutdown will probably have already
		 * been called, so we will not be able to submit a task. */
		cudaError_t err = cudaFreeHost(A);
		if (STARPU_UNLIKELY(err))
			STARPU_CUDA_REPORT_ERROR(err);
#ifndef HAVE_CUDA_MEMCPY_PEER
	}
	else
	{
		int push_res;

                free_pinned_cl.where = STARPU_CUDA;
		struct starpu_task *task = starpu_task_create();
		task->callback_func = NULL;
		task->cl = &free_pinned_cl;
		task->cl_arg = A;

		task->synchronous = 1;

		_starpu_exclude_task_from_dag(task);

		push_res = _starpu_task_submit_internally(task);
		STARPU_ASSERT(push_res != -ENODEV);
	}
#endif
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
	} else
#endif
#endif
	{
		free(A);
	}

	return 0;
}

#ifdef STARPU_SIMGRID
static _starpu_pthread_mutex_t cuda_alloc_mutex = _STARPU_PTHREAD_MUTEX_INITIALIZER;
static _starpu_pthread_mutex_t opencl_alloc_mutex = _STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif

uintptr_t
starpu_malloc_on_node(unsigned dst_node, size_t size)
{
	uintptr_t addr = 0;

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif

	if (_starpu_memory_manager_can_allocate_size(size, dst_node) == 0)
		return 0;

	switch(starpu_node_get_kind(dst_node))
	{
		case STARPU_CPU_RAM:
		{
			addr = (uintptr_t)malloc(size);
			break;
		}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		case STARPU_CUDA_RAM:
#ifdef STARPU_SIMGRID
#ifdef STARPU_DEVEL
#warning TODO: record used memory, using a simgrid property to know the available memory
#endif
			/* Sleep 10µs for the allocation */
			_STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
			MSG_process_sleep(0.000010);
			addr = 1;
			_STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
#else
			status = cudaMalloc((void **)&addr, size);
			if (!addr || (status != cudaSuccess))
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					STARPU_CUDA_REPORT_ERROR(status);
				addr = 0;
			}
#endif
			break;
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	        case STARPU_OPENCL_RAM:
			{
#ifdef STARPU_SIMGRID
				/* Sleep 10µs for the allocation */
				_STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
				MSG_process_sleep(0.000010);
				addr = 1;
				_STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
#else
                                int ret;
				cl_mem ptr;

				ret = starpu_opencl_allocate_memory(&ptr, size, CL_MEM_READ_WRITE);
				if (ret)
				{
					addr = 0;
				}
				else
				{
					addr = (uintptr_t)ptr;
				}
				break;
#endif
			}
#endif
		default:
			STARPU_ABORT();
	}

	if (addr == 0)
	{
		// Allocation failed, gives the memory back to the memory manager
		_starpu_memory_manager_deallocate_size(size, dst_node);
	}
	return addr;
}

void
starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size)
{
	enum starpu_node_kind kind = starpu_node_get_kind(dst_node);
	switch(kind)
	{
#ifdef STARPU_DEVEL
#warning TODO we need to call starpu_free
#endif
		case STARPU_CPU_RAM:
			free((void*)addr);
			_starpu_memory_manager_deallocate_size(size, dst_node);
			break;
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		case STARPU_CUDA_RAM:
		{
#ifdef STARPU_SIMGRID
			_STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
			/* Sleep 10µs for the free */
			MSG_process_sleep(0.000010);
			_STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
#else
			cudaError_t err;
			err = cudaFree((void*)addr);
			if (STARPU_UNLIKELY(err != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(err);
			_starpu_memory_manager_deallocate_size(size, dst_node);
#endif
			break;
		}
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
                case STARPU_OPENCL_RAM:
		{
#ifdef STARPU_SIMGRID
			_STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
			/* Sleep 10µs for the free */
			MSG_process_sleep(0.000010);
			_STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
#else
			cl_int err;
                        err = clReleaseMemObject((void*)addr);
			if (STARPU_UNLIKELY(err != CL_SUCCESS))
				STARPU_OPENCL_REPORT_ERROR(err);
			_starpu_memory_manager_deallocate_size(size, dst_node);
#endif
                        break;
		}
#endif
		default:
			STARPU_ABORT();
	}
}

