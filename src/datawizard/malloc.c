/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018       Federal University of Rio Grande do Sul (UFRGS)
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
#include <drivers/driver_common/driver_common.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <core/simgrid.h>
#include <core/task.h>

#ifdef STARPU_SIMGRID
#include <sys/mman.h>
#include <fcntl.h>
#include <smpi/smpi.h>
#endif

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

static size_t _malloc_align = sizeof(void*);
static int disable_pinning;
static int enable_suballocator;

/* This file is used for implementing "folded" allocation */
#ifdef STARPU_SIMGRID
static int bogusfile = -1;
static unsigned long _starpu_malloc_simulation_fold;
/* Table to control unique simulation mallocs */
#include <common/uthash.h>
struct unique_shared_alloc
{
        size_t id;
        int count;
        void* addr;
        UT_hash_handle hh;
};
static struct unique_shared_alloc* unique_shared_alloc_table = NULL;
#endif

static starpu_malloc_hook malloc_hook;
static starpu_free_hook free_hook;

void starpu_malloc_set_hooks(starpu_malloc_hook _malloc_hook, starpu_free_hook _free_hook)
{
	malloc_hook = _malloc_hook;
	free_hook = _free_hook;
}

void starpu_malloc_set_align(size_t align)
{
	STARPU_ASSERT_MSG(!(align & (align - 1)), "Alignment given to starpu_malloc_set_align (%lu) must be a power of two", (unsigned long) align);
	if (_malloc_align < align)
		_malloc_align = align;
}

/* Driver porters: adding your driver here is optional, only needed for pinning host memory.  */

#if (defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER))// || defined(STARPU_USE_OPENCL)
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
//        //        _STARPU_MALLOC(*(s->ptr), s->dim);
//        starpu_opencl_allocate_memory(devid, (void **)(s->ptr), s->dim, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR);
//}
//#endif

#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
static void malloc_pinned_cuda_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;
	cudaError_t cures = cudaErrorMemoryAllocation;
#if 0 //defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_MNGMEM)
	/* FIXME: check if devices actually support cudaMallocManaged or fallback to cudaHostAlloc() */
	cures = cudaMallocManaged((void **)(s->ptr), s->dim, cudaMemAttachGlobal);
#endif
#if defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_CANMAPHOST)
	if (cures != cudaSuccess)
		cures = cudaHostAlloc((void **)(s->ptr), s->dim, cudaHostAllocPortable|cudaHostAllocMapped);
#endif
	if (cures != cudaSuccess)
		cures = cudaHostAlloc((void **)(s->ptr), s->dim, cudaHostAllocPortable);

	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

#if (defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER)) && !defined(STARPU_SIMGRID)// || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel malloc_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "malloc_pinned"
};

static struct starpu_codelet malloc_pinned_cl =
{
	.cuda_funcs = {malloc_pinned_cuda_codelet},
//#ifdef STARPU_USE_OPENCL
//	.opencl_funcs = {malloc_pinned_opencl_codelet},
//#endif
	.nbuffers = 0,
	.model = &malloc_pinned_model
};
#endif

/* Allocation in CPU RAM */
int starpu_malloc_flags(void **A, size_t dim, int flags)
{
	return _starpu_malloc_flags_on_node(STARPU_MAIN_RAM, A, dim, flags);
}

/* Return whether we should pin the allocated data */
static int _starpu_malloc_should_pin(int flags)
{
	if (flags & STARPU_MALLOC_PINNED && disable_pinning <= 0)
	{
		if (_starpu_can_submit_cuda_task())
		{
			return 1;
		}
		if (_starpu_can_submit_hip_task())
		{
			return 1;
		}
//		if (_starpu_can_submit_opencl_task())
//			return 1;
	}
	return 0;
}

int _starpu_malloc_willpin_on_node(unsigned dst_node)
{
	int flags = _starpu_get_node_struct(dst_node)->malloc_on_node_default_flags;
	return (_starpu_malloc_should_pin(flags) && STARPU_RUNNING_ON_VALGRIND == 0
			&& (_starpu_can_submit_cuda_task()
			    || _starpu_can_submit_hip_task()
			    /* || _starpu_can_submit_opencl_task() */
			));
}

int _starpu_malloc_flags_on_node(unsigned dst_node, void **A, size_t dim, int flags)
{
	int ret=0;

	STARPU_ASSERT_MSG(A, "starpu_malloc needs to be passed the address of the pointer to be filled");
	if (!starpu_is_initialized())
		_STARPU_DISP("Warning: starpu_malloc needs to be called after starpu is initialized, to be able to pin memory for CUDA\n");

	if (dim == 0)
		/* Make sure we succeed */
		dim = 1;

	if (flags & STARPU_MALLOC_COUNT)
	{
		if (!(flags & STARPU_MALLOC_NORECLAIM))
			while (starpu_memory_allocate(dst_node, dim, flags) != 0)
			{
				size_t freed;
				size_t reclaim = 2 * dim;
				_STARPU_DEBUG("There is not enough memory left, we are going to reclaim %ld\n", (long)reclaim);
				_STARPU_TRACE_START_MEMRECLAIM(dst_node,0);
				freed = _starpu_memory_reclaim_generic(dst_node, 0, reclaim, STARPU_FETCH);
				_STARPU_TRACE_END_MEMRECLAIM(dst_node,0);
				if (freed < dim && !(flags & STARPU_MEMORY_WAIT))
				{
					// We could not reclaim enough memory
					*A = NULL;
					return -ENOMEM;
				}
			}
		else if (flags & STARPU_MEMORY_WAIT)
			starpu_memory_allocate(dst_node, dim, flags);
		else
			starpu_memory_allocate(dst_node, dim, flags | STARPU_MEMORY_OVERFLOW);
	}

	if (malloc_hook)
	{
		ret = malloc_hook(dst_node, A, dim, flags);
		goto end;
	}

	/* Note: synchronize this test with _starpu_malloc_willpin_on_node */
	if (_starpu_malloc_should_pin(flags) && STARPU_RUNNING_ON_VALGRIND == 0)
	{
		if (_starpu_can_submit_cuda_task())
		{
#ifdef STARPU_SIMGRID
		/* FIXME: CUDA seems to be taking 650µs every 1MiB.
		 * Ideally we would simulate this batching in 1MiB requests
		 * instead of computing an average value.
		 */
			if (_starpu_simgrid_cuda_malloc_cost())
				starpu_sleep((float) dim * 0.000650 / 1048576.);
#else /* STARPU_SIMGRID */
#ifdef STARPU_USE_CUDA
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
			cudaError_t cures = cudaErrorMemoryAllocation;

#if 0 //defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_MNGMEM)
			/* FIXME: check if devices actually support cudaMallocManaged or fallback to cudaHostAlloc() */
			cures = cudaMallocManaged(A, dim, cudaMemAttachGlobal);
#endif

#if defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_CANMAPHOST)
			if (cures != cudaSuccess)
				cures = cudaHostAlloc(A, dim, cudaHostAllocPortable|cudaHostAllocMapped);
#endif

			if (cures != cudaSuccess)
				cures = cudaHostAlloc(A, dim, cudaHostAllocPortable);

			if (STARPU_UNLIKELY(cures))
			{
				STARPU_CUDA_REPORT_ERROR(cures);
				ret = -ENOMEM;
			}
			goto end;
#else
			int push_res;

			/* Old versions of CUDA are not thread-safe, we have to
			 * run cudaHostAlloc from CUDA workers */
			STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "without CUDA peer allocation support, pinned allocation must not be done from task or callback");

			struct malloc_pinned_codelet_struct s =
			{
				.ptr = A,
				.dim = dim
			};

			malloc_pinned_cl.where = STARPU_CUDA;
			struct starpu_task *task = starpu_task_create();
			task->name = "cuda_malloc_pinned";
			task->callback_func = NULL;
			task->cl = &malloc_pinned_cl;
			task->cl_arg = &s;
			task->type = STARPU_TASK_TYPE_INTERNAL;

			task->synchronous = 1;

			_starpu_exclude_task_from_dag(task);

			push_res = _starpu_task_submit_internally(task);
			STARPU_ASSERT(push_res != -ENODEV);
			goto end;
#endif /* STARPU_HAVE_CUDA_MEMCPY_PEER */
#endif /* STARPU_USE_CUDA */
		}
		if (_starpu_can_submit_hip_task())
		{
#ifdef STARPU_USE_HIP
			hipError_t hipres = hipErrorMemoryAllocation;

#if 0 //defined(STARPU_USE_HIP_MAP) && defined(STARPU_HAVE_HIP_MNGMEM)
			/* FIXME: check if devices actually support hipMallocManaged or fallback to hipHostAlloc() */
			hipres = hipMallocManaged(A, dim, hipMemAttachGlobal);
#endif

#if defined(STARPU_USE_HIP_MAP) && defined(STARPU_HAVE_HIP_CANMAPHOST)
			if (hipres != hipSuccess)
				hipres = hipHostMalloc(A, dim, hipHostMallocPortable|hipHostMallocMapped);
#endif

			if (hipres != hipSuccess)
			{
				hipres = hipHostMalloc(A, dim, hipHostMallocPortable);
			}

			if (STARPU_UNLIKELY(hipres != hipSuccess))
			{
				STARPU_HIP_REPORT_ERROR(hipres);
				ret = -ENOMEM;
			}
			goto end;
#endif /* STARPU_USE_HIP */
//		}
//		else if (_starpu_can_submit_opencl_task())
//		{
//#ifdef STARPU_USE_OPENCL
//			int push_res;
//
//			STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "pinned OpenCL allocation must not be done from task or callback");
//
//			struct malloc_pinned_codelet_struct s =
//				{
//					.ptr = A,
//					.dim = dim
//				};
//
//			malloc_pinned_cl.where = STARPU_OPENCL;
//			struct starpu_task *task = starpu_task_create();
//		        task->name = "opencl_malloc_pinned";
//			task->callback_func = NULL;
//			task->cl = &malloc_pinned_cl;
//			task->cl_arg = &s;
//			task->synchronous = 1;
//			task->type = STARPU_TASK_TYPE_INTERNAL;
//
//			_starpu_exclude_task_from_dag(task);
//
//			push_res = _starpu_task_submit_internally(task);
//			STARPU_ASSERT(push_res != -ENODEV);
//			goto end;
//#endif /* STARPU_USE_OPENCL */
#endif /* STARPU_SIMGRID */
		}
	}

#ifdef STARPU_SIMGRID
	if (flags & STARPU_MALLOC_SIMULATION_FOLDED)
	{
#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
		if(_starpu_simgrid_running_smpi())
		{
			if(flags & STARPU_MALLOC_SIMULATION_UNIQUE)
			{
				struct unique_shared_alloc *block;
				HASH_FIND(hh, unique_shared_alloc_table, &dim, sizeof(dim), block);
				if(block==NULL)
				{
					block = (struct unique_shared_alloc*)malloc(sizeof(struct unique_shared_alloc));
					block->addr = SMPI_SHARED_MALLOC(dim);
					block->count = 1;
					block->id = dim;
					HASH_ADD(hh, unique_shared_alloc_table, id, sizeof(dim), block);
				}
				else
				{
					block->count++;
				}
				*A = block->addr;
			}
			else
			{
				*A = SMPI_SHARED_MALLOC(dim);
			}
		}
		else
#endif
		{
		/* Use "folded" allocation: the same file is mapped several
		 * times contiguously, to get a memory area one can read/write,
		 * without consuming memory */

		/* First reserve memory area */
		void *buf = mmap (NULL, dim, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
		unsigned i;
		if (buf == MAP_FAILED)
		{
			_STARPU_DISP("Warning: could not allocate %luMiB of memory, you need to run \"sysctl vm.overcommit_memory=1\" as root to allow so big allocations\n", (unsigned long) (dim >> 20));
			ret = -ENOMEM;
			*A = NULL;
		}
		else
		{
			if (bogusfile == -1)
			{
				char *path = starpu_getenv("TMPDIR");
				if (!path)
					path = starpu_getenv("TEMP");
				if (!path)
					path = starpu_getenv("TMP");
				if (!path)
					path = "/tmp";
				/* Create bogus file if not done already */
				char *name = _starpu_mktemp(path, O_RDWR | O_BINARY, &bogusfile);
				char *dumb;
				if (!name)
				{
					ret = errno;
					munmap(buf, dim);
					*A = NULL;
					goto end;
				}
				unlink(name);
				free(name);
				_STARPU_CALLOC(dumb, 1,_starpu_malloc_simulation_fold);
				int res = write(bogusfile, dumb, _starpu_malloc_simulation_fold);
				STARPU_ASSERT_MSG(res > 0, "Cannot write");
				free(dumb);
			}
			/* Map the bogus file in place of the anonymous memory */
			for (i = 0; i < dim / _starpu_malloc_simulation_fold; i++)
			{
				void *pos = (void*) ((unsigned long) buf + i * _starpu_malloc_simulation_fold);
				void *res = mmap(pos, _starpu_malloc_simulation_fold, PROT_READ|PROT_WRITE, MAP_FIXED|MAP_SHARED|MAP_POPULATE, bogusfile, 0);
				STARPU_ASSERT_MSG(res == pos, "Could not map folded virtual memory (%s). Do you perhaps need to increase the STARPU_MALLOC_SIMULATION_FOLD environment variable or the sysctl vm.max_map_count?", strerror(errno));
			}

			if (dim % _starpu_malloc_simulation_fold)
			{
				void *pos = (void*) ((unsigned long) buf + i * _starpu_malloc_simulation_fold);
				void *res = mmap(pos, dim % _starpu_malloc_simulation_fold, PROT_READ|PROT_WRITE, MAP_FIXED|MAP_SHARED|MAP_POPULATE, bogusfile, 0);
				STARPU_ASSERT_MSG(res == pos, "Could not map folded virtual memory (%s). Do you perhaps need to increase the STARPU_MALLOC_SIMULATION_FOLD environment variable or the sysctl vm.max_map_count?", strerror(errno));
			}
			*A = buf;
		}
	    }
	}
#endif

#ifdef HAVE_MMAP
#ifdef STARPU_USE_MP
	if(_starpu_can_submit_ms_task())
	{
		*A = _starpu_map_allocate(dim, dst_node);

		if (!*A)
			ret = -ENOMEM;
		else
		{
#ifdef STARPU_HAVE_HWLOC
			struct _starpu_machine_config *config = _starpu_get_machine_config();
			hwloc_topology_t hwtopology = config->topology.hwtopology;
			hwloc_obj_t numa_node_obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, starpu_memory_nodes_numa_id_to_hwloclogid(dst_node));
			if (numa_node_obj)
			{
				hwloc_bitmap_t nodeset = numa_node_obj->nodeset;
#if HWLOC_API_VERSION >= 0x00020000
				hwloc_set_area_membind(hwtopology, *A, dim, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_NOCPUBIND);
#else
				hwloc_set_area_membind_nodeset(hwtopology, *A, dim, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND);
#endif
			}
#endif
		}
	}
	else
#endif
#endif
#ifdef STARPU_HAVE_HWLOC
	if (starpu_memory_nodes_get_numa_count() > 1)
	{
		struct _starpu_machine_config *config = _starpu_get_machine_config();
		hwloc_topology_t hwtopology = config->topology.hwtopology;
		hwloc_obj_t numa_node_obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, starpu_memory_nodes_numa_id_to_hwloclogid(dst_node));
		hwloc_bitmap_t nodeset = numa_node_obj->nodeset;
#if HWLOC_API_VERSION >= 0x00020000
		*A = hwloc_alloc_membind(hwtopology, dim, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_NOCPUBIND);
#else
		*A = hwloc_alloc_membind_nodeset(hwtopology, dim, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND);
#endif
		//fprintf(stderr, "Allocation %lu bytes on NUMA node %d [%p]\n", (unsigned long) dim, starpu_memnode_get_numaphysid(dst_node), *A);
		if (!*A)
			ret = -ENOMEM;
	}
#endif /* STARPU_HAVE_HWLOC */
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
			if (!*A)
				ret = -ENOMEM;
		}
		else
#endif /* STARPU_HAVE_POSIX_MEMALIGN */
		{
			*A = malloc(dim);
			if (!*A)
				ret = -ENOMEM;
		}

end:
	if (ret == 0)
	{
		STARPU_ASSERT_MSG(*A, "Failed to allocated memory of size %lu b\n", (unsigned long)dim);
	}
	else if (flags & STARPU_MALLOC_COUNT)
	{
		starpu_memory_deallocate(dst_node, dim);
	}

	return ret;
}

int starpu_malloc(void **A, size_t dim)
{
	return starpu_malloc_flags(A, dim, STARPU_MALLOC_PINNED);
}

#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
static void free_pinned_cuda_codelet(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	cudaError_t cures;
#if 0 //defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_MNGMEM)
	/* FIXME: check if devices actually support cudaMallocManaged or fallback to cudaHostAlloc() */
	cures = cudaFree(arg);
#else
	cures = cudaFreeHost(arg);
#endif
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

#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID) // || defined(STARPU_USE_OPENCL)
static struct starpu_perfmodel free_pinned_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "free_pinned"
};

static struct starpu_codelet free_pinned_cl =
{
	.cuda_funcs = {free_pinned_cuda_codelet},
//#ifdef STARPU_USE_OPENCL
//	.opencl_funcs = {free_pinned_opencl_codelet},
//#endif
	.nbuffers = 0,
	.model = &free_pinned_model
};
#endif

int starpu_free_flags(void *A, size_t dim, int flags)
{
	return _starpu_free_flags_on_node(STARPU_MAIN_RAM, A, dim, flags);
}

int _starpu_free_flags_on_node(unsigned dst_node, void *A, size_t dim, int flags)
{
	if (dim == 0)
		dim = 1;

	if (free_hook)
	{
		free_hook(dst_node, A, dim, flags);
		goto out;
	}

	if (_starpu_malloc_should_pin(flags) && STARPU_RUNNING_ON_VALGRIND == 0)
	{
		if (_starpu_can_submit_cuda_task())
		{
#ifdef STARPU_SIMGRID
			/* TODO: simulate CUDA barrier */
#else /* !STARPU_SIMGRID */
#ifdef STARPU_USE_CUDA
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
			if (!starpu_is_initialized())
			{
#endif
				/* This is especially useful when starpu_free is called even
				 * though starpu_shutdown has already
				 * been called, so we will not be able to submit a task. */
				cudaError_t cures;
#if 0 //defined(STARPU_USE_CUDA_MAP) && defined(STARPU_HAVE_CUDA_MNGMEM)
				/* FIXME: check if devices actually support cudaMallocManaged or fallback to cudaHostAlloc() */
				cures = cudaFree(A);
#else
				cures = cudaFreeHost(A);
#endif
				if (STARPU_UNLIKELY(cures))
					STARPU_CUDA_REPORT_ERROR(cures);
				goto out;
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
			}
			else
			{
				int push_res;

				STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "without CUDA peer allocation support, pinned deallocation must not be done from task or callback");

				free_pinned_cl.where = STARPU_CUDA;
				struct starpu_task *task = starpu_task_create();
				task->name = "cuda_free_pinned";
				task->callback_func = NULL;
				task->cl = &free_pinned_cl;
				task->cl_arg = A;
				task->synchronous = 1;
				task->type = STARPU_TASK_TYPE_INTERNAL;

				_starpu_exclude_task_from_dag(task);

				push_res = _starpu_task_submit_internally(task);
				STARPU_ASSERT(push_res != -ENODEV);
				goto out;
			}
#endif /* STARPU_HAVE_CUDA_MEMCPY_PEER */
#endif /* STARPU_USE_CUDA */
		}
		if (_starpu_can_submit_hip_task())
		{
#ifdef STARPU_USE_HIP
			/* TODO: submit task */
			/* This is especially useful when starpu_free is called even
			 * though starpu_shutdown has already
			 * been called, so we will not be able to submit a task. */
			hipError_t hipres;
			hipres = hipHostFree(A);
			if (STARPU_UNLIKELY(hipres))
				STARPU_HIP_REPORT_ERROR(hipres);
			goto out;
#endif /* STARPU_USE_HIP */
#endif /* STARPU_SIMGRID */
		}
//	else if (_starpu_can_submit_opencl_task())
//	{
//#ifdef STARPU_USE_OPENCL
//		int push_res;
//
//		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "pinned OpenCL deallocation must not be done from task or callback");
//
//                free_pinned_cl.where = STARPU_OPENCL;
//		struct starpu_task *task = starpu_task_create();
//              task->name = "opencl_free_pinned";
//		task->callback_func = NULL;
//		task->cl = &free_pinned_cl;
//		task->cl_arg = A;
//		task->synchronous = 1;
//		task->type = STARPU_TASK_TYPE_INTERNAL;
//
//		_starpu_exclude_task_from_dag(task);
//
//		push_res = starpu_task_submit(task);
//		STARPU_ASSERT(push_res != -ENODEV);
//		goto out;
//	}
//#endif
	}

#ifdef STARPU_SIMGRID
	if (flags & STARPU_MALLOC_SIMULATION_FOLDED)
	{
#if SIMGRID_VERSION >= 31500 && SIMGRID_VERSION != 31559
		if(_starpu_simgrid_running_smpi())
		{
			if(flags & STARPU_MALLOC_SIMULATION_UNIQUE)
			{
				struct unique_shared_alloc *block;
				HASH_FIND(hh, unique_shared_alloc_table, &dim, sizeof(dim), block);
				STARPU_ASSERT(block != NULL);
				block->count--;
				if(block->count == 0)
				{
					SMPI_SHARED_FREE(block->addr);
					HASH_DEL(unique_shared_alloc_table, block);
					free(block);
				}
			}
			else
			{
				SMPI_SHARED_FREE(A);
			}
		}
		else
#endif
			munmap(A, dim);
	}
#endif
#ifdef HAVE_MMAP
#ifdef STARPU_USE_MP
	else if(_starpu_can_submit_ms_task())
	{
		_starpu_map_deallocate(A, dim);
	}
#endif
#endif
#ifdef STARPU_HAVE_HWLOC
	else if (starpu_memory_nodes_get_numa_count() > 1)
	{
		struct _starpu_machine_config *config = _starpu_get_machine_config();
		hwloc_topology_t hwtopology = config->topology.hwtopology;
		hwloc_free(hwtopology, A, dim);
	}
#endif /* STARPU_HAVE_HWLOC */
	else
		free(A);

out:
	if (flags & STARPU_MALLOC_COUNT)
	{
		starpu_memory_deallocate(dst_node, dim);
	}

	return 0;
}

int starpu_free(void *A)
{
	return starpu_free_flags(A, 0, STARPU_MALLOC_PINNED);
}

int starpu_free_noflag(void *A, size_t dim)
{
	return starpu_free_flags(A, dim, STARPU_MALLOC_PINNED);
}

static uintptr_t _starpu_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	uintptr_t addr = 0;

	if (size == 0)
		size = 1;

	/* Handle count first */
	if (flags & STARPU_MALLOC_COUNT)
	{
		if (starpu_memory_allocate(dst_node, size, flags) != 0)
			return 0;
		/* And prevent double-count in starpu_malloc_flags */
		flags &= ~STARPU_MALLOC_COUNT;
	}

	const struct _starpu_node_ops *node_ops = _starpu_memory_node_get_node_ops(dst_node);
	if (node_ops && node_ops->malloc_on_device)
	{
		int devid = starpu_memory_node_get_devid(dst_node);
		addr = node_ops->malloc_on_device(devid, size, flags & ~STARPU_MALLOC_COUNT);
	}
	else
		STARPU_ABORT_MSG("No malloc_on_device function defined for node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));

	if (addr == 0)
	{
		// Allocation failed, gives the memory back to the memory manager
		_STARPU_TRACE_MEMORY_FULL(size);
		if (flags & STARPU_MALLOC_COUNT)
			starpu_memory_deallocate(dst_node, size);
	}
	return addr;
}

void _starpu_free_on_node_flags(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	int count = flags & STARPU_MALLOC_COUNT;
	flags &= ~STARPU_MALLOC_COUNT;

	if (size == 0)
		size = 1;

	const struct _starpu_node_ops *node_ops = _starpu_memory_node_get_node_ops(dst_node);
	if (node_ops && node_ops->free_on_device)
	{
		int devid = starpu_memory_node_get_devid(dst_node);
		node_ops->free_on_device(devid, addr, size, flags);
	}
	else
		STARPU_ABORT_MSG("No free_on_device function defined for node %s\n", _starpu_node_get_prefix(starpu_node_get_kind(dst_node)));

	if (count)
		starpu_memory_deallocate(dst_node, size);
}

int
starpu_memory_pin(void *addr STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	if (STARPU_MALLOC_PINNED && disable_pinning <= 0 && STARPU_RUNNING_ON_VALGRIND == 0)
	{
#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
		if (cudaHostRegister(addr, size, cudaHostRegisterPortable) != cudaSuccess)
			return -1;
#endif
#if defined(STARPU_USE_HIP)
		if (hipHostRegister(addr, size, hipHostRegisterPortable) != hipSuccess)
			return -1;
#endif
	}
	return 0;
}

int
starpu_memory_unpin(void *addr STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED)
{
	if (STARPU_MALLOC_PINNED && disable_pinning <= 0 && STARPU_RUNNING_ON_VALGRIND == 0)
	{
#if defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
		if (cudaHostUnregister(addr) != cudaSuccess)
			return -1;
#endif
#if defined(STARPU_USE_HIP)
		if (hipHostUnregister(addr) != hipSuccess)
			return -1;
#endif
	}
	return 0;
}

void
_starpu_malloc_init(unsigned dst_node)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(dst_node);
	_starpu_chunk_list_init(&node_struct->chunks);
	node_struct->nfreechunks = 0;
	STARPU_PTHREAD_MUTEX_INIT(&node_struct->chunk_mutex, NULL);
	disable_pinning = starpu_getenv_number("STARPU_DISABLE_PINNING");
	enable_suballocator = starpu_getenv_number_default("STARPU_SUBALLOCATOR", 1);
	node_struct->malloc_on_node_default_flags = STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT;
#ifdef STARPU_SIMGRID
	/* Reasonably "costless" */
	_starpu_malloc_simulation_fold = starpu_getenv_number_default("STARPU_MALLOC_SIMULATION_FOLD", 1) << 20;
#endif
}

void
_starpu_malloc_shutdown(unsigned dst_node)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(dst_node);
	struct _starpu_chunk *chunk, *next_chunk;

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->chunk_mutex);
	for (chunk = _starpu_chunk_list_begin(&node_struct->chunks);
	     chunk != _starpu_chunk_list_end(&node_struct->chunks);
	     chunk = next_chunk)
	{
		next_chunk = _starpu_chunk_list_next(chunk);
		_starpu_free_on_node_flags(dst_node, chunk->base, CHUNK_SIZE, node_struct->malloc_on_node_default_flags);
		_starpu_chunk_list_erase(&node_struct->chunks, chunk);
		free(chunk);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->chunk_mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&node_struct->chunk_mutex);
}

/* Create a new chunk */
static struct _starpu_chunk *_starpu_new_chunk(unsigned dst_node, int flags)
{
	struct _starpu_chunk *chunk;
	uintptr_t base = _starpu_malloc_on_node(dst_node, CHUNK_SIZE, flags);

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

/* Return whether we should use our suballocator */
static int _starpu_malloc_should_suballoc(unsigned dst_node, size_t size, int flags)
{
	return (enable_suballocator &&
		(size <= CHUNK_ALLOC_MAX &&
		(starpu_node_get_kind(dst_node) == STARPU_CUDA_RAM
		 || starpu_node_get_kind(dst_node) == STARPU_HIP_RAM
		 || (starpu_node_get_kind(dst_node) == STARPU_CPU_RAM
		     && _starpu_malloc_should_pin(flags))
		 )))
	       || starpu_node_get_kind(dst_node) == STARPU_MAX_FPGA_RAM;
}

uintptr_t
starpu_malloc_on_node_flags(unsigned dst_node, size_t size, int flags)
{
	/* Big allocation, allocate normally */
	if (!_starpu_malloc_should_suballoc(dst_node, size, flags))
		return _starpu_malloc_on_node(dst_node, size, flags);

	struct _starpu_node *node_struct = _starpu_get_node_struct(dst_node);

	/* Round up allocation to block size */
	int nblocks = (size + CHUNK_ALLOC_MIN - 1) / CHUNK_ALLOC_MIN;
	if (!nblocks)
		nblocks = 1;

	struct _starpu_chunk *chunk;
	int prevblock, block;
	int available_max;
	struct block *bitmap;

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->chunk_mutex);

	/* Try to find a big enough segment among the chunks */
	for (chunk = _starpu_chunk_list_begin(&node_struct->chunks);
	     chunk != _starpu_chunk_list_end(&node_struct->chunks);
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
			if (length >= nblocks)
			{

				if (length >= 2*nblocks)
				{
					/* This one this has quite some room,
					 * put it front, to make finding it
					 * easier next time. */
					_starpu_chunk_list_erase(&node_struct->chunks, chunk);
					_starpu_chunk_list_push_front(&node_struct->chunks, chunk);
				}
				if (chunk->available == CHUNK_NBLOCKS)
					/* This one was empty, it's not empty any more */
					node_struct->nfreechunks--;
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
	chunk = _starpu_new_chunk(dst_node, flags);
	if (!chunk)
	{
		/* Really no memory any more, fail */
		STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->chunk_mutex);
		errno = ENOMEM;
		return 0;
	}

	/* And make it easy to find. */
	_starpu_chunk_list_push_front(&node_struct->chunks, chunk);
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

	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->chunk_mutex);

	return chunk->base + (block-1) * CHUNK_ALLOC_MIN;
}

void
starpu_free_on_node_flags(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	/* Big allocation, deallocate normally */
	if (!_starpu_malloc_should_suballoc(dst_node, size, flags))
	{
		_starpu_free_on_node_flags(dst_node, addr, size, flags);
		return;
	}

	struct _starpu_node *node_struct = _starpu_get_node_struct(dst_node);
	struct _starpu_chunk *chunk;

	/* Round up allocation to block size */
	int nblocks = (size + CHUNK_ALLOC_MIN - 1) / CHUNK_ALLOC_MIN;
	if (!nblocks)
		nblocks = 1;

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->chunk_mutex);
	for (chunk = _starpu_chunk_list_begin(&node_struct->chunks);
	     chunk != _starpu_chunk_list_end(&node_struct->chunks);
	     chunk = _starpu_chunk_list_next(chunk))
		if (addr >= chunk->base && addr < chunk->base + CHUNK_SIZE)
			break;
	STARPU_ASSERT(chunk != _starpu_chunk_list_end(&node_struct->chunks));

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
		if (node_struct->nfreechunks >= CHUNKS_NFREE &&
		     starpu_node_get_kind(dst_node) != STARPU_MAX_FPGA_RAM)
		{
			/* We already have free chunks, release this one */
			_starpu_free_on_node_flags(dst_node, chunk->base, CHUNK_SIZE, flags);
			_starpu_chunk_list_erase(&node_struct->chunks, chunk);
			free(chunk);
		}
		else
			node_struct->nfreechunks++;
	}
	else
	{
		/* Freed some room, put this first in chunks list */
		_starpu_chunk_list_erase(&node_struct->chunks, chunk);
		_starpu_chunk_list_push_front(&node_struct->chunks, chunk);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->chunk_mutex);
}

void starpu_malloc_on_node_set_default_flags(unsigned node, int flags)
{
	STARPU_ASSERT_MSG(node < STARPU_MAXNODES, "bogus node value %u given to starpu_malloc_on_node_set_default_flags\n", node);
	_starpu_get_node_struct(node)->malloc_on_node_default_flags = flags;
}

uintptr_t
starpu_malloc_on_node(unsigned dst_node, size_t size)
{
	return starpu_malloc_on_node_flags(dst_node, size, _starpu_get_node_struct(dst_node)->malloc_on_node_default_flags);
}

void
starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size)
{
	starpu_free_on_node_flags(dst_node, addr, size, _starpu_get_node_struct(dst_node)->malloc_on_node_default_flags);
}
