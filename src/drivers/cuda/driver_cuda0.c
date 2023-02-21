/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

/* This is a version of the CUDA driver with very minimal features:
 * - synchronous kernel execution
 * - synchronous data transfers
 *
 * This is not meant to be actually used :)
 *
 * It is only meant as a basic driver sample, easy to get inspired from for
 * writing other drivers.
 */

#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <core/devices.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_cuda.h"
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <common/knobs.h>
#ifdef STARPU_USE_CUDA
#include <cublas.h>
#endif

/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

/* the number of CUDA devices */
static int ncudagpus = -1;

static size_t global_mem[STARPU_MAXCUDADEVS];
int _starpu_cuda_bus_ids[STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES][STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES];
/* Note: streams are not thread-safe, so we define them for each CUDA worker
 * emitting a GPU-GPU transfer */
static struct cudaDeviceProp props[STARPU_MAXCUDADEVS];

static unsigned cuda_bindid_init[STARPU_MAXCUDADEVS];
static unsigned cuda_bindid[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_init[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_nodes[STARPU_MAXCUDADEVS];

int _starpu_nworker_per_cuda = 1;

static size_t _starpu_cuda_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}

cudaStream_t starpu_cuda_get_local_stream(void)
{
	return NULL;
}

const struct cudaDeviceProp *starpu_cuda_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}


/* Early library initialization, before anything else, just initialize data */
void _starpu_cuda_init(void)
{
	memset(&cuda_bindid_init, 0, sizeof(cuda_bindid_init));
	memset(&cuda_memory_init, 0, sizeof(cuda_memory_init));
}

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXCUDADEVS */

unsigned _starpu_get_cuda_device_count(void)
{
	int cnt;
	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;

	if (cnt > STARPU_MAXCUDADEVS)
	{
		_STARPU_MSG("# Warning: %d CUDA devices available. Only %d enabled. Use configure option --enable-maxcudadev=xxx to update the maximum value of supported CUDA devices.\n", cnt, STARPU_MAXCUDADEVS);
		cnt = STARPU_MAXCUDADEVS;
	}
	return (unsigned)cnt;
}

/* This is run from initialize to determine the number of CUDA devices */
void _starpu_init_cuda(void)
{
	if (ncudagpus < 0)
	{
		ncudagpus = _starpu_get_cuda_device_count();
		STARPU_ASSERT(ncudagpus <= STARPU_MAXCUDADEVS);
	}
}

/* This is called to return the real (non-clamped) number of devices */
void _starpu_cuda_discover_devices(struct _starpu_machine_config *config)
{
	/* Discover the number of CUDA devices. Fill the result in CONFIG. */

	int cnt;
	cudaError_t cures;

	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures != cudaSuccess))
		cnt = 0;
	config->topology.nhwdevices[STARPU_CUDA_WORKER] = cnt;
}

static void _starpu_initialize_workers_cuda_gpuid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(uconf->use_explicit_workers_cuda_gpuid == 0
					    ? NULL
					    : (int *)uconf->workers_cuda_gpuid,
					    &(config->current_devid[STARPU_CUDA_WORKER]),
					    (int *)topology->workers_devid[STARPU_CUDA_WORKER],
					    "STARPU_WORKERS_CUDAID",
					    topology->nhwdevices[STARPU_CUDA_WORKER],
					    STARPU_CUDA_WORKER);
	_starpu_devices_drop_duplicate(topology->workers_devid[STARPU_CUDA_WORKER]);
}

/* Determine which devices we will use */
void _starpu_init_cuda_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config)
{
	int ncuda = config->conf.ncuda;

	if (ncuda != 0)
	{
		/* The user did not disable CUDA. We need to
		 * initialize CUDA early to count the number of
		 * devices
		 */
		_starpu_init_cuda();
		int nb_devices = _starpu_get_cuda_device_count();

		_starpu_topology_check_ndevices(&ncuda, nb_devices, 0, STARPU_MAXCUDADEVS, 0, "ncuda", "CUDA", "maxcudadev");
	}

	/* Now we know how many CUDA devices will be used */
	topology->ndevices[STARPU_CUDA_WORKER] = ncuda;

	_starpu_initialize_workers_cuda_gpuid(config);

	unsigned cudagpu;
	for (cudagpu = 0; (int) cudagpu < ncuda; cudagpu++)
	{
		int devid = _starpu_get_next_devid(topology, config, STARPU_CUDA_WORKER);

		if (devid == -1)
		{
			// There is no more devices left
			topology->ndevices[STARPU_CUDA_WORKER] = cudagpu;
			break;
		}

		_starpu_topology_configure_workers(topology, config,
					STARPU_CUDA_WORKER,
					cudagpu, devid, 0, 0,
					1, 1, NULL, NULL);
	}

	/* Don't copy this, just here for other code to work fine */
	topology->cuda_th_per_stream = 0;
	topology->cuda_th_per_dev = 1;
}

/* Bind the driver on a CPU core */
void _starpu_cuda_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

	if (cuda_bindid_init[devid])
	{
		workerarg->bindid = cuda_bindid[devid];
	}
	else
	{
		cuda_bindid_init[devid] = 1;

		workerarg->bindid = cuda_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
	}
}

/* Set up memory and buses */
void _starpu_cuda_init_worker_memory(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (cuda_memory_init[devid])
	{
		memory_node = cuda_memory_nodes[devid];
	}
	else
	{
		cuda_memory_init[devid] = 1;

		memory_node = cuda_memory_nodes[devid] = _starpu_memory_node_register(STARPU_CUDA_RAM, devid);

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_cuda_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
			_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
		}
	}
	_starpu_memory_node_add_nworkers(memory_node);

	//This worker can also manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(workerarg, numa);

	_starpu_worker_drives_memory_node(workerarg, memory_node);

	workerarg->memory_node = memory_node;
}

/* Set the current CUDA device */
void starpu_cuda_set_device(int devid STARPU_ATTRIBUTE_UNUSED)
{
	cudaError_t cures;

	cures = cudaSetDevice(devid);

	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void _starpu_cuda_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

	/* Find the size of the memory on the device */
	totalGlobalMem = props[devid].totalGlobalMem;

	limit = totalGlobalMem / (1024*1024) * FREE_MARGIN;

	global_mem[devid] = limit * 1024*1024;
}
/* Really initialize one device */
static void init_device_context(unsigned devid, unsigned memnode)
{
	cudaError_t cures;

	starpu_cuda_set_device(devid);

	/* force CUDA to initialize the context for real */
	cures = cudaFree(0);
	if (STARPU_UNLIKELY(cures))
	{
		if (cures == cudaErrorDevicesUnavailable)
		{
			_STARPU_MSG("All CUDA-capable devices are busy or unavailable\n");
			exit(77);
		}
		STARPU_CUDA_REPORT_ERROR(cures);
	}

	cures = cudaGetDeviceProperties(&props[devid], devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	_starpu_cuda_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cuda_get_global_mem_size(devid));
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
}

/* This is run from the driver thread to initialize the driver CUDA context */
int _starpu_cuda_driver_init(struct _starpu_worker *worker)
{
	_starpu_driver_start(worker, STARPU_CUDA_WORKER, 0);
	_starpu_set_local_worker_key(worker);

	unsigned devid = worker->devid;
	unsigned memnode = worker->memory_node;

	init_device_context(devid, memnode);

	unsigned workerid = worker->workerid;

	float size = (float) global_mem[devid] / (1<<30);
	/* get the device's name */
	char devname[64];
	strncpy(devname, props[devid].name, 63);
	devname[63] = 0;

	snprintf(worker->name, sizeof(worker->name), "CUDA0 %u (%s %.1f GiB)", devid, devname, size);
	snprintf(worker->short_name, sizeof(worker->short_name), "CUDA %u", devid);
	_STARPU_DEBUG("cuda (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

	_STARPU_TRACE_WORKER_INIT_END(workerid);

	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "CUDA0 %u", worker->devid);
		starpu_pthread_setname(thread_name);
	}

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	worker->status = STATUS_UNKNOWN;
	worker->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);

	return 0;
}

int _starpu_cuda_driver_deinit(struct _starpu_worker *worker)
{
	_STARPU_TRACE_WORKER_DEINIT_START;

	unsigned devid = worker->devid;
	unsigned memnode = worker->memory_node;

	/* I'm last, deinitialize device */
	_starpu_datawizard_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	_starpu_malloc_shutdown(memnode);

	deinit_device_context(devid);

	worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_CUDA_WORKER);

	return 0;
}

uintptr_t _starpu_cuda_malloc_on_device(int devid, size_t size, int flags)
{
	uintptr_t addr = 0;
	(void) flags;

	starpu_cuda_set_device(devid);

	/* Check if there is free memory */
	size_t cuda_mem_free, cuda_mem_total;
	cudaError_t status;
	status = cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total);
	if (status == cudaSuccess && cuda_mem_free * FREE_MARGIN < size)
	{
		addr = 0;
	}
	else
	{
		status = cudaMalloc((void **)&addr, size);
		if (!addr || (status != cudaSuccess))
		{
			if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
				STARPU_CUDA_REPORT_ERROR(status);
			addr = 0;
		}
	}
	return addr;
}

void _starpu_cuda_free_on_device(int devid, uintptr_t addr, size_t size, int flags)
{
	(void) devid;
	(void) addr;
	(void) size;
	(void) flags;

	cudaError_t err;
	starpu_cuda_set_device(devid);
	err = cudaFree((void*)addr);
	if (STARPU_UNLIKELY(err != cudaSuccess))
		STARPU_CUDA_REPORT_ERROR(err);
}

int starpu_cuda_copy_async_sync_devid(void *src_ptr, int src_devid, enum starpu_node_kind src_kind STARPU_ATTRIBUTE_UNUSED,
				      void *dst_ptr, int dst_devid, enum starpu_node_kind dst_kind STARPU_ATTRIBUTE_UNUSED,
				      size_t ssize, cudaStream_t stream STARPU_ATTRIBUTE_UNUSED,
				      enum cudaMemcpyKind kind)
{
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_devid != dst_devid)
	{
		STARPU_ABORT();
	}

	/* do it in a synchronous fashion */
	cures = cudaMemcpy((char *)dst_ptr, (char *)src_ptr, ssize, kind);
	(void) cudaGetLastError();

	if (!cures)
		cures = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	return 0;
}

/* Driver porters: this is optional but really recommended */
int starpu_cuda_copy2d_async_sync_devid(void *src_ptr, int src_devid, enum starpu_node_kind src_kind STARPU_ATTRIBUTE_UNUSED,
					void *dst_ptr, int dst_devid, enum starpu_node_kind dst_kind STARPU_ATTRIBUTE_UNUSED,
					size_t blocksize,
					size_t numblocks, size_t ld_src, size_t ld_dst,
					cudaStream_t stream STARPU_ATTRIBUTE_UNUSED, enum cudaMemcpyKind kind)
{
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_devid != dst_devid)
	{
		STARPU_ABORT_MSG("CUDA memcpy 3D peer not available, but core triggered one ?!");
	}

	cures = cudaMemcpy2D((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
			blocksize, numblocks, kind);
	if (!cures)
		cures = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	return 0;
}

int _starpu_cuda_copy_interface(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	(void) req;

	int ret = 1;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	STARPU_ASSERT(copy_methods->any_to_any);
	copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	return ret;
}

int _starpu_cuda_copy_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	return starpu_cuda_copy_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CUDA_RAM,
						 (void*) (dst + dst_offset), dst_devid, STARPU_CPU_RAM,
						 size,
						 NULL,
						 cudaMemcpyDeviceToHost);
}

int _starpu_cuda_copy_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
	STARPU_ASSERT(src_devid == dst_devid);
#endif

	return starpu_cuda_copy_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CUDA_RAM,
						 (void*) (dst + dst_offset), dst_devid, STARPU_CUDA_RAM,
						 size,
						 NULL,
						 cudaMemcpyDeviceToDevice);
}

int _starpu_cuda_copy_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	return starpu_cuda_copy_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CPU_RAM,
						 (void*) (dst + dst_offset), dst_devid, STARPU_CUDA_RAM,
						 size,
						 NULL,
						 cudaMemcpyHostToDevice);
}

/* Driver porters: these are optional but really recommended */
int _starpu_cuda_copy2d_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, int src_devid,
					      uintptr_t dst, size_t dst_offset, int dst_devid,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	return starpu_cuda_copy2d_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CUDA_RAM,
						   (void*) (dst + dst_offset), dst_devid, STARPU_CPU_RAM,
						   blocksize, numblocks, ld_src, ld_dst,
						   NULL,
						   cudaMemcpyDeviceToHost);
}

int _starpu_cuda_copy2d_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, int src_devid,
					       uintptr_t dst, size_t dst_offset, int dst_devid,
					       size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					       struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
	STARPU_ASSERT(src_devid == dst_devid);
#endif

	return starpu_cuda_copy2d_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CUDA_RAM,
						   (void*) (dst + dst_offset), dst_devid, STARPU_CUDA_RAM,
						   blocksize, numblocks, ld_src, ld_dst,
						   NULL,
						   cudaMemcpyDeviceToDevice);
}

int _starpu_cuda_copy2d_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, int src_devid,
					      uintptr_t dst, size_t dst_offset, int dst_devid,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	return starpu_cuda_copy2d_async_sync_devid((void*) (src + src_offset), src_devid, STARPU_CPU_RAM,
						   (void*) (dst + dst_offset), dst_devid, STARPU_CUDA_RAM,
						   blocksize, numblocks, ld_src, ld_dst,
						   NULL,
						   cudaMemcpyHostToDevice);
}

void _starpu_cuda_init_device_context(int devid)
{
	starpu_cuda_set_device(devid);
	/* hack to force the initialization */
	cudaFree(0);
}

void _starpu_cuda_device_name(int devid, char *name, size_t size)
{
	struct cudaDeviceProp prop;
	cudaError_t cures;
	cures = cudaGetDeviceProperties(&prop, devid);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	strncpy(name, prop.name, size);
	name[size-1] = 0;
}

size_t _starpu_cuda_total_memory(int devid)
{
	struct cudaDeviceProp prop;
	cudaError_t cures;
	cures = cudaGetDeviceProperties(&prop, devid);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	return prop.totalGlobalMem;
}

void _starpu_cuda_reset_device(int devid)
{
	starpu_cuda_set_device(devid);
#if CUDART_VERSION >= 4000
	cudaDeviceReset();
#else
	cudaThreadExit();
#endif
}

static int start_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);
	j->workerid = worker->workerid;

	/* Fetch data input synchronously */
	int ret = _starpu_fetch_task_input(task, j, 0);
	if (ret != 0)
	{
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return -EAGAIN;
	}

	_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);

	starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_CUDA is defined in 'where', cuda_func or cuda_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING(j);
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		_STARPU_TRACE_END_EXECUTING(j);
	}

	return 0;
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static int execute_job_on_cuda(struct starpu_task *task, struct _starpu_worker *worker)
{
	int res;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	res = start_job_on_cuda(j, worker);

	if (res)
	{
		switch (res)
		{
			case -EAGAIN:
				_STARPU_DISP("ouch, CUDA could not actually run task %p, putting it back...\n", task);
				_starpu_push_task_to_workers(task);
				return -EAGAIN;
			default:
				STARPU_ABORT();
		}
	}

	finish_job_on_cuda(j, worker);

	return 0;
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker)
{
	int profiling = starpu_profiling_status_get();

	worker->current_task = NULL;

	_starpu_driver_end_job(worker, j, &worker->perf_arch, 0, profiling);

	_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

	_starpu_push_task_output(j);

	_starpu_set_current_task(NULL);

	_starpu_handle_job_termination(j);
}

/* One iteration of the main driver loop */
int _starpu_cuda_driver_run_once(struct _starpu_worker *worker)
{
	struct starpu_task *task;
	struct _starpu_job *j;
	int res;

	unsigned memnode = worker->memory_node;

	/* Make some progress */
	_starpu_datawizard_progress(1);
	if (memnode != STARPU_MAIN_RAM)
	{
		_starpu_datawizard_progress(1);
	}

	/* And pull a task */
	task = _starpu_get_worker_task(worker, worker->workerid, worker->memory_node);

	if (!task)
		return 0;

	j = _starpu_get_job_associated_to_task(task);

	/* can CUDA do that task ? */
	if (!_STARPU_MAY_PERFORM(j, CUDA))
	{
		/* this is neither a cuda or a cublas task */
		_starpu_worker_refuse_task(worker, task);
		return 0;
	}

	worker->current_task = task;

	res = execute_job_on_cuda(task, worker);

	if (res)
	{
		switch (res)
		{
		case -EAGAIN:
			_starpu_push_task_to_workers(task);
			return 0;
		default:
			STARPU_ABORT();
		}
	}

	return 0;
}

void *_starpu_cuda_worker(void *_arg)
{
	struct _starpu_worker *worker = _arg;

	_starpu_cuda_driver_init(worker);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_cuda_driver_run_once(worker);
	}
	_starpu_cuda_driver_deinit(worker);

	return NULL;
}


void starpu_cublas_report_error(const char *func, const char *file, int line, int status)
{
	char *errormsg;
	switch (status)
	{
		case CUBLAS_STATUS_SUCCESS:
			errormsg = "success";
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			errormsg = "not initialized";
			break;
		case CUBLAS_STATUS_ALLOC_FAILED:
			errormsg = "alloc failed";
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			errormsg = "invalid value";
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			errormsg = "arch mismatch";
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			errormsg = "execution failed";
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			errormsg = "internal error";
			break;
		default:
			errormsg = "unknown error";
			break;
	}
	_STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}

void starpu_cuda_report_error(const char *func, const char *file, int line, cudaError_t status)
{
	const char *errormsg = cudaGetErrorString(status);
	_STARPU_ERROR("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

int _starpu_cuda_run_from_worker(struct _starpu_worker *worker)
{
	/* Let's go ! */
	_starpu_cuda_worker(worker);

	return 0;
}

struct _starpu_driver_ops _starpu_driver_cuda_ops =
{
	.init = _starpu_cuda_driver_init,
	.run = _starpu_cuda_run_from_worker,
	.run_once = _starpu_cuda_driver_run_once,
	.deinit = _starpu_cuda_driver_deinit,
};

struct _starpu_node_ops _starpu_driver_cuda_node_ops =
{
	.name = "cuda0 driver",
	.malloc_on_device = _starpu_cuda_malloc_on_device,
	.free_on_device = _starpu_cuda_free_on_device,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_cuda_copy_interface,
	.copy_interface_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_cuda_copy_interface,
	.copy_interface_from[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_cuda_copy_data_from_cuda_to_cpu,
	.copy_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cuda_to_cuda,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_cuda_copy_data_from_cpu_to_cuda,
	.copy_data_from[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cuda_to_cuda,

	.copy2d_data_to[STARPU_CPU_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cpu,
	.copy2d_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cuda,

	.copy2d_data_from[STARPU_CPU_RAM] = _starpu_cuda_copy2d_data_from_cpu_to_cuda,
	.copy2d_data_from[STARPU_CUDA_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cuda,

	.device_name = _starpu_cuda_device_name,
	.total_memory = _starpu_cuda_total_memory,
	.max_memory = _starpu_cuda_total_memory,
	.set_device = starpu_cuda_set_device,
	.init_device = _starpu_cuda_init_device_context,
	.reset_device = _starpu_cuda_reset_device,
};
