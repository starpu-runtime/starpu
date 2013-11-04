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
#include <core/disk.h>
#include <common/config.h>
#include <common/fxt.h>
#include <starpu.h>
#include <drivers/opencl/driver_opencl.h>
#include <datawizard/memory_manager.h>
#include <datawizard/malloc.h>

static size_t _malloc_align = sizeof(void*);

void starpu_malloc_set_align(size_t align)
{
	STARPU_ASSERT_MSG(!(align & (align - 1)), "Alignment given to starpu_malloc_set_align (%lu) must be a power of two", (unsigned long) align);
	if (_malloc_align < align)
		_malloc_align = align;
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

int starpu_malloc_flags(void **A, size_t dim, int flags)
{
	int ret=0;

	STARPU_ASSERT(A);

	if (flags & STARPU_MALLOC_COUNT)
	{
		if (_starpu_memory_manager_can_allocate_size(dim, STARPU_MAIN_RAM) == 0)
		{
			size_t freed;
			size_t reclaim = 2 * dim;
			_STARPU_DEBUG("There is not enough memory left, we are going to reclaim %ld\n", reclaim);
			_STARPU_TRACE_START_MEMRECLAIM(0,0);
			freed = _starpu_memory_reclaim_generic(0, 0, reclaim);
			_STARPU_TRACE_END_MEMRECLAIM(0,0);
			if (freed < dim)
			{
				// We could not reclaim enough memory
				*A = NULL;
				return -ENOMEM;
			}
		}
	}

	if (flags & STARPU_MALLOC_PINNED)
	{
#ifdef STARPU_SIMGRID
		/* FIXME: CUDA seems to be taking 650µs every 1MiB.
		 * Ideally we would simulate this batching in 1MiB requests
		 * instead of computing an average value.
		 */
		MSG_process_sleep((float) dim * 0.000650 / 1048576.);
#else /* STARPU_SIMGRID */
		if (_starpu_can_submit_cuda_task())
		{
#ifdef STARPU_USE_CUDA
#ifdef HAVE_CUDA_MEMCPY_PEER
			cudaError_t cures;
			cures = cudaHostAlloc(A, dim, cudaHostAllocPortable);
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
			goto end;
#else
			int push_res;

			if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
				return -EDEADLK;

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
			goto end;
#endif /* HAVE_CUDA_MEMCPY_PEER */
#endif /* STARPU_USE_CUDA */
		}
//		else if (_starpu_can_submit_opencl_task())
//		{
//#ifdef STARPU_USE_OPENCL
//			int push_res;
//
//			if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
//				return -EDEADLK;
//
//			struct malloc_pinned_codelet_struct s =
//				{
//					.ptr = A,
//					.dim = dim
//				};
//
//			malloc_pinned_cl.where = STARPU_OPENCL;
//			struct starpu_task *task = starpu_task_create();
//			task->callback_func = NULL;
//			task->cl = &malloc_pinned_cl;
//			task->cl_arg = &s;
//			task->synchronous = 1;
//
//			_starpu_exclude_task_from_dag(task);
//
//			push_res = _starpu_task_submit_internally(task);
//			STARPU_ASSERT(push_res != -ENODEV);
//			goto end;
//#endif /* STARPU_USE_OPENCL */
//		}
#endif /* STARPU_SIMGRID */
	}

	if (_starpu_can_submit_scc_task())
	{
#ifdef STARPU_USE_SCC
		_starpu_scc_allocate_shared_memory(A, dim);
#endif
	}
	else
#ifdef STARPU_HAVE_POSIX_MEMALIGN
	if (_malloc_align != sizeof(void*))
	{
		if (posix_memalign(A, _malloc_align, dim))
		{
			ret = -ENOMEM;
			*A = NULL;
		}
	}
	else
#elif defined(STARPU_HAVE_MEMALIGN)
		if (_malloc_align != sizeof(void*))
		{
			*A = memalign(_malloc_align, dim);
		}
		else
#endif /* STARPU_HAVE_POSIX_MEMALIGN */
		{
			*A = malloc(dim);
		}

end:
	if (ret == 0)
	{
		STARPU_ASSERT(*A);
	}

	return ret;
}

int starpu_malloc(void **A, size_t dim)
{
	return starpu_malloc_flags(A, dim, STARPU_MALLOC_PINNED);
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

int starpu_free_flags(void *A, size_t dim, int flags)
{
#ifndef STARPU_SIMGRID
	if (flags & STARPU_MALLOC_PINNED)
	{
		if (_starpu_can_submit_cuda_task())
		{
#ifdef STARPU_USE_CUDA
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
				goto out;
#ifndef HAVE_CUDA_MEMCPY_PEER
			}
			else
			{
				int push_res;

				if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
					return -EDEADLK;

				free_pinned_cl.where = STARPU_CUDA;
				struct starpu_task *task = starpu_task_create();
				task->callback_func = NULL;
				task->cl = &free_pinned_cl;
				task->cl_arg = A;
				task->synchronous = 1;

				_starpu_exclude_task_from_dag(task);

				push_res = _starpu_task_submit_internally(task);
				STARPU_ASSERT(push_res != -ENODEV);
				goto out;
			}
#endif /* HAVE_CUDA_MEMCPY_PEER */
#endif /* STARPU_USE_CUDA */
		}
//	else if (_starpu_can_submit_opencl_task())
//	{
//#ifdef STARPU_USE_OPENCL
//		int push_res;
//
//		if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
//			return -EDEADLK;
//
//                free_pinned_cl.where = STARPU_OPENCL;
//		struct starpu_task *task = starpu_task_create();
//		task->callback_func = NULL;
//		task->cl = &free_pinned_cl;
//		task->cl_arg = A;
//		task->synchronous = 1;
//
//		_starpu_exclude_task_from_dag(task);
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//		goto out;
//	}
//#endif
	}
#endif /* STARPU_SIMGRID */

	if (_starpu_can_submit_scc_task())
	{
#ifdef STARPU_USE_SCC
		_starpu_scc_free_shared_memory(A);
#endif
	} else
	free(A);

out:
	if (flags & STARPU_MALLOC_COUNT)
	{
		_starpu_memory_manager_deallocate_size(dim, STARPU_MAIN_RAM);
	}

	return 0;
}

int starpu_free(void *A)
{
	return starpu_free_flags(A, 0, STARPU_MALLOC_PINNED);
}

#ifdef STARPU_SIMGRID
static starpu_pthread_mutex_t cuda_alloc_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_mutex_t opencl_alloc_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif

static uintptr_t
_starpu_malloc_on_node(unsigned dst_node, size_t size)
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
			starpu_malloc_flags((void**) &addr, size,
#if defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
					0
#else
					STARPU_MALLOC_PINNED
#endif
					);
			break;
		}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		case STARPU_CUDA_RAM:
		{
#ifdef STARPU_SIMGRID
			static uintptr_t last[STARPU_MAXNODES];
#ifdef STARPU_DEVEL
#warning TODO: record used memory, using a simgrid property to know the available memory
#endif
			/* Sleep for the allocation */
			STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
			MSG_process_sleep(0.000175);
			if (!last[dst_node])
				last[dst_node] = 1<<10;
			addr = last[dst_node];
			last[dst_node]+=size;
			STARPU_ASSERT(last[dst_node] >= addr);
			STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
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
		}
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	        case STARPU_OPENCL_RAM:
		{
#ifdef STARPU_SIMGRID
				static uintptr_t last[STARPU_MAXNODES];
				/* Sleep for the allocation */
				STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
				MSG_process_sleep(0.000175);
				if (!last[dst_node])
					last[dst_node] = 1<<10;
				addr = last[dst_node];
				last[dst_node]+=size;
				STARPU_ASSERT(last[dst_node] >= addr);
				STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
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
	        case STARPU_DISK_RAM:
		{
			addr = (uintptr_t) _starpu_disk_alloc(dst_node, size);
			break;
		}

#ifdef STARPU_USE_MIC
		case STARPU_MIC_RAM:
			if (_starpu_mic_allocate_memory((void **)(&addr), size, dst_node))
				addr = 0;
			break;
#endif
#ifdef STARPU_USE_SCC
		case STARPU_SCC_RAM:
			if (_starpu_scc_allocate_memory((void **)(&addr), size, dst_node))
				addr = 0;
			break;
#endif
		default:
			STARPU_ABORT();
	}

	if (addr == 0)
	{
		// Allocation failed, gives the memory back to the memory manager
		const char* file;					
		file = strrchr(__FILE__,'/');							
		file += sizeof(char);										
		_STARPU_TRACE_MEMORY_FULL(size);
		_starpu_memory_manager_deallocate_size(size, dst_node);
	}
	return addr;
}

void
_starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size)
{
	enum starpu_node_kind kind = starpu_node_get_kind(dst_node);
	switch(kind)
	{
		case STARPU_CPU_RAM:
			starpu_free_flags((void*)addr, size,
#if defined(STARPU_USE_CUDA) && !defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
					0
#else
					STARPU_MALLOC_PINNED
#endif
					);
			break;
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		case STARPU_CUDA_RAM:
		{
#ifdef STARPU_SIMGRID
			STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
			/* Sleep for the free */
			MSG_process_sleep(0.000750);
			STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
#else
			cudaError_t err;
			err = cudaFree((void*)addr);
			if (STARPU_UNLIKELY(err != cudaSuccess))
				STARPU_CUDA_REPORT_ERROR(err);
#endif
			break;
		}
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
                case STARPU_OPENCL_RAM:
		{
#ifdef STARPU_SIMGRID
			STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
			/* Sleep for the free */
			MSG_process_sleep(0.000750);
			STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
#else
			cl_int err;
                        err = clReleaseMemObject((void*)addr);
			if (STARPU_UNLIKELY(err != CL_SUCCESS))
				STARPU_OPENCL_REPORT_ERROR(err);
#endif
                        break;
		}
#endif
	        case STARPU_DISK_RAM:
		{
			_starpu_disk_free (dst_node, (void *) addr , size);
			break;
		}

#ifdef STARPU_USE_MIC
		case STARPU_MIC_RAM:
			_starpu_mic_free_memory((void*) addr, size, dst_node);
			break;
#endif
#ifdef STARPU_USE_SCC
		case STARPU_SCC_RAM:
			_starpu_scc_free_memory((void *) addr, dst_node);
			break;
#endif
		default:
			STARPU_ABORT();
	}
	_starpu_memory_manager_deallocate_size(size, dst_node);

}

/*
 * On CUDA which has very expensive malloc, for small sizes, allocate big
 * chunks divided in blocks, and we actually allocate segments of consecutive
 * blocks.
 *
 * We try to keep the list of chunks with increasing occupancy, so we can
 * quickly find free segments to allocate.
 */

/* Size of each chunk, 32MiB granularity brings 128 chunks to be allocated in
 * order to fill a 4GiB GPU. */
#define CHUNK_SIZE (32*1024*1024)

/* Maximum segment size we will allocate in chunks */
#define CHUNK_ALLOC_MAX (CHUNK_SIZE / 8)

/* Granularity of allocation, i.e. block size, StarPU will never allocate less
 * than this.
 * 16KiB (i.e. 64x64 float) granularity eats 2MiB RAM for managing a 4GiB GPU.
 */
#define CHUNK_ALLOC_MIN (16*1024)

/* Number of blocks */
#define CHUNK_NBLOCKS (CHUNK_SIZE/CHUNK_ALLOC_MIN)

/* Linked list for available segments */
struct block {
	int length;	/* Number of consecutive free blocks */
	int next;	/* next free segment */
};

/* One chunk */
LIST_TYPE(_starpu_chunk,
	uintptr_t base;

	/* Available number of blocks, for debugging */
	int available;

	/* Overestimation of the maximum size of available segments in this chunk */
	int available_max;

	/* Bitmap describing availability of the block */
	/* Block 0 is always empty, and is just the head of the free segments list */
	struct block bitmap[CHUNK_NBLOCKS+1];
)

/* One list of chunks per node */
static struct _starpu_chunk_list *chunks[STARPU_MAXNODES];
/* Number of completely free chunks */
static int nfreechunks[STARPU_MAXNODES];
/* This protects chunks and nfreechunks */
static starpu_pthread_mutex_t chunk_mutex[STARPU_MAXNODES];

void
_starpu_malloc_init(unsigned dst_node)
{
	chunks[dst_node] = _starpu_chunk_list_new();
	nfreechunks[dst_node] = 0;
	STARPU_PTHREAD_MUTEX_INIT(&chunk_mutex[dst_node], NULL);
}

void
_starpu_malloc_shutdown(unsigned dst_node)
{
	struct _starpu_chunk *chunk, *next_chunk;

	if (!chunks[dst_node])
		return;

	STARPU_PTHREAD_MUTEX_LOCK(&chunk_mutex[dst_node]);
	for (chunk = _starpu_chunk_list_begin(chunks[dst_node]);
	     chunk != _starpu_chunk_list_end(chunks[dst_node]);
	     chunk = next_chunk)
	{
		next_chunk = _starpu_chunk_list_next(chunk);
		_starpu_free_on_node(dst_node, chunk->base, CHUNK_SIZE);
		_starpu_chunk_list_erase(chunks[dst_node], chunk);
		free(chunk);
	}
	_starpu_chunk_list_delete(chunks[dst_node]);
	chunks[dst_node] = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&chunk_mutex[dst_node]);
	STARPU_PTHREAD_MUTEX_DESTROY(&chunk_mutex[dst_node]);
}

/* Create a new chunk */
static struct _starpu_chunk *_starpu_new_chunk(unsigned dst_node)
{
	struct _starpu_chunk *chunk;
	uintptr_t base = _starpu_malloc_on_node(dst_node, CHUNK_SIZE);

	if (!base)
		return NULL;

	/* Create a new chunk */
	chunk = _starpu_chunk_new();
	chunk->base = base;

	/* First block is just a fake block pointing to the free segments list */
	chunk->bitmap[0].length = 0;
	chunk->bitmap[0].next = 1;

	/* At first we have only one big segment for the whole chunk */
	chunk->bitmap[1].length = CHUNK_NBLOCKS;
	chunk->bitmap[1].next = -1;

	chunk->available_max = CHUNK_NBLOCKS;
	chunk->available = CHUNK_NBLOCKS;
	return chunk;
}

uintptr_t
starpu_malloc_on_node(unsigned dst_node, size_t size)
{
	/* Big allocation, allocate normally */
	if (size > CHUNK_ALLOC_MAX || starpu_node_get_kind(dst_node) != STARPU_CUDA_RAM)
		return _starpu_malloc_on_node(dst_node, size);

	/* Round up allocation to block size */
	int nblocks = (size + CHUNK_ALLOC_MIN - 1) / CHUNK_ALLOC_MIN;

	struct _starpu_chunk *chunk;
	int prevblock, block;
	int available_max;
	struct block *bitmap;

	STARPU_PTHREAD_MUTEX_LOCK(&chunk_mutex[dst_node]);

	/* Try to find a big enough segment among the chunks */
	for (chunk = _starpu_chunk_list_begin(chunks[dst_node]);
	     chunk != _starpu_chunk_list_end(chunks[dst_node]);
	     chunk = _starpu_chunk_list_next(chunk))
	{
		if (chunk->available_max < nblocks)
			continue;

		bitmap = chunk->bitmap;
		available_max = 0;
		for (prevblock = block = 0;
			block != -1;
			prevblock = block, block = bitmap[prevblock].next)
		{
			STARPU_ASSERT(block >= 0 && block <= CHUNK_NBLOCKS);
			int length = bitmap[block].length;
			if (length >= nblocks) {

				if (length >= 2*nblocks)
				{
					/* This one this has quite some room,
					 * put it front, to make finding it
					 * easier next time. */
					_starpu_chunk_list_erase(chunks[dst_node], chunk);
					_starpu_chunk_list_push_front(chunks[dst_node], chunk);
				}
				if (chunk->available == CHUNK_NBLOCKS)
					/* This one was empty, it's not empty any more */
					nfreechunks[dst_node]--;
				goto found;
			}
			if (length > available_max)
				available_max = length;
		}

		/* Didn't find a big enough segment in this chunk, its
		 * available_max is out of date */
		chunk->available_max = available_max;
	}

	/* Didn't find a big enough segment, create another chunk.  */
	chunk = _starpu_new_chunk(dst_node);
	if (!chunk)
	{
		/* Really no memory any more, fail */
		STARPU_PTHREAD_MUTEX_UNLOCK(&chunk_mutex[dst_node]);
		errno = ENOMEM;
		return 0;
	}

	/* And make it easy to find. */
	_starpu_chunk_list_push_front(chunks[dst_node], chunk);
	bitmap = chunk->bitmap;
	prevblock = 0;
	block = 1;

found:

	chunk->available -= nblocks;
	STARPU_ASSERT(bitmap[block].length >= nblocks);
	STARPU_ASSERT(block <= CHUNK_NBLOCKS);
	if (bitmap[block].length == nblocks)
	{
		/* Fits exactly, drop this segment from the skip list */
		bitmap[prevblock].next = bitmap[block].next;
	}
	else
	{
		/* Still some room */
		STARPU_ASSERT(block + nblocks <= CHUNK_NBLOCKS);
		bitmap[prevblock].next = block + nblocks;
		bitmap[block + nblocks].length = bitmap[block].length - nblocks;
		bitmap[block + nblocks].next = bitmap[block].next;
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&chunk_mutex[dst_node]);

	return chunk->base + (block-1) * CHUNK_ALLOC_MIN;
}

void
starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size)
{
	/* Big allocation, deallocate normally */
	if (size > CHUNK_ALLOC_MAX || starpu_node_get_kind(dst_node) != STARPU_CUDA_RAM)
	{
		_starpu_free_on_node(dst_node, addr, size);
		return;
	}

	struct _starpu_chunk *chunk;

	/* Round up allocation to block size */
	int nblocks = (size + CHUNK_ALLOC_MIN - 1) / CHUNK_ALLOC_MIN;

	STARPU_PTHREAD_MUTEX_LOCK(&chunk_mutex[dst_node]);
	for (chunk = _starpu_chunk_list_begin(chunks[dst_node]);
	     chunk != _starpu_chunk_list_end(chunks[dst_node]);
	     chunk = _starpu_chunk_list_next(chunk))
		if (addr >= chunk->base && addr < chunk->base + CHUNK_SIZE)
			break;
	STARPU_ASSERT(chunk != _starpu_chunk_list_end(chunks[dst_node]));

	struct block *bitmap = chunk->bitmap;
	int block = ((addr - chunk->base) / CHUNK_ALLOC_MIN) + 1, prevblock, nextblock;

	/* Look for free segment just before this one */
	for (prevblock = 0;
		prevblock != -1;
		prevblock = nextblock)
	{
		STARPU_ASSERT(prevblock >= 0 && prevblock <= CHUNK_NBLOCKS);
		nextblock = bitmap[prevblock].next;
		STARPU_ASSERT_MSG(nextblock != block, "It seems data 0x%lx (size %u) on node %u is being freed a second time\n", (unsigned long) addr, (unsigned) size, dst_node);
		if (nextblock > block || nextblock == -1)
			break;
	}
	STARPU_ASSERT(prevblock != -1);

	chunk->available += nblocks;

	/* Insert in free segments list */
	bitmap[block].next = nextblock;
	bitmap[prevblock].next = block;
	bitmap[block].length = nblocks;

	STARPU_ASSERT(nextblock >= -1 && nextblock <= CHUNK_NBLOCKS);
	if (nextblock == block + nblocks)
	{
		/* This freed segment is just before a free segment, merge them */
		bitmap[block].next = bitmap[nextblock].next;
		bitmap[block].length += bitmap[nextblock].length;

		if (bitmap[block].length > chunk->available_max)
			chunk->available_max = bitmap[block].length;
	}

	if (prevblock > 0 && prevblock + bitmap[prevblock].length == block)
	{
		/* This free segment is just after a free segment, merge them */
		bitmap[prevblock].next = bitmap[block].next;
		bitmap[prevblock].length += bitmap[block].length;

		if (bitmap[prevblock].length > chunk->available_max)
			chunk->available_max = bitmap[prevblock].length;

		block = prevblock;
	}

	if (chunk->available == CHUNK_NBLOCKS)
	{
		/* This chunk is now empty, but avoid chunk free/alloc
		 * ping-pong by keeping some of these.  */
		if (nfreechunks[dst_node] >= 1) {
			/* We already have free chunks, release this one */
			_starpu_free_on_node(dst_node, chunk->base, CHUNK_SIZE);
			_starpu_chunk_list_erase(chunks[dst_node], chunk);
			free(chunk);
		} else
			nfreechunks[dst_node]++;
	}
	else
	{
		/* Freed some room, put this first in chunks list */
		_starpu_chunk_list_erase(chunks[dst_node], chunk);
		_starpu_chunk_list_push_front(chunks[dst_node], chunk);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&chunk_mutex[dst_node]);
}
