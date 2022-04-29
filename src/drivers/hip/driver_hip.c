/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_hip.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_hip.h"
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <common/knobs.h>

#if !defined(STARPU_HIP_REPORT_ERROR)
	#error "No STARPU_HIP_REPORT_ERROR!!!!"
#endif
/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

/* the number of HIP devices */
static int nhipgpus = -1;

static size_t global_mem[STARPU_MAXHIPDEVS];
int _starpu_hip_bus_ids[STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES][STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES];
/* Note: streams are not thread-safe, so we define them for each HIP worker
 * emitting a GPU-GPU transfer */
static struct hipDeviceProp_t props[STARPU_MAXHIPDEVS];

static unsigned hip_init[STARPU_MAXHIPDEVS];
static unsigned hip_memory_nodes[STARPU_MAXHIPDEVS];
static unsigned hip_bindid[STARPU_MAXHIPDEVS];

int _starpu_nworker_per_hip = 1;

static size_t _starpu_hip_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}
/* Streams not needed right now
// hipStream_t starpu_hip_get_local_stream(void)
// {
// 	return NULL;
// }
*/
const struct hipDeviceProp_t *starpu_hip_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}


/* Early library initialization, before anything else, just initialize data */
void _starpu_hip_init(void)
{
	memset(&hip_init, 0, sizeof(hip_init));
}

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXHIPDEVS */

unsigned _starpu_get_hip_device_count(void)
{
	int cnt;
	hipError_t cures;
	cures = hipGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;

	if (cnt > STARPU_MAXHIPDEVS)
	{
		_STARPU_MSG("# Warning: %d HIP devices available. Only %d enabled. Use configure option --enable-maxhipdev=xxx to update the maximum value of supported HIP devices.\n", cnt, STARPU_MAXHIPDEVS);
		cnt = STARPU_MAXHIPDEVS;
	}
	return (unsigned)cnt;
}

/* This is run from initialize to determine the number of HIP devices */
void _starpu_init_hip(void)
{
	if (nhipgpus < 0)
	{
		nhipgpus = _starpu_get_hip_device_count();
		STARPU_ASSERT(nhipgpus <= STARPU_MAXHIPDEVS);
	}
}

/* This is called to really discover the hardware */
void
_starpu_hip_discover_devices (struct _starpu_machine_config *config)
{
/* Discover the number of HIP devices. Fill the result in CONFIG. */

	int cnt;
	hipError_t cures;

	cures = hipGetDeviceCount (&cnt);
	if (STARPU_UNLIKELY(cures != hipSuccess))
		cnt = 0;
	config->topology.nhwdevices[STARPU_HIP_WORKER] = cnt;
}

static void _starpu_initialize_workers_hip_gpuid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

        _starpu_initialize_workers_deviceid(uconf->use_explicit_workers_hip_gpuid == 0
					    ? NULL
					    : (int *)uconf->workers_hip_gpuid,
					    &(config->current_devid[STARPU_HIP_WORKER]),
					    (int *)topology->workers_devid[STARPU_HIP_WORKER],
					    "STARPU_WORKERS_HIPID",
					    topology->nhwdevices[STARPU_HIP_WORKER],
					    STARPU_HIP_WORKER);
	_starpu_topology_drop_duplicate(topology->workers_devid[STARPU_HIP_WORKER]);
}

/* Determine which devices we will use */
void _starpu_init_hip_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config)
{
	int nhip = config->conf.nhip;

	if (nhip != 0)
	{
		/* The user did not disable HIP. We need to initialize HIP
 		 * early to count the number of devices */
		_starpu_init_hip();
		int nb_devices = _starpu_get_hip_device_count();

		_starpu_topology_check_ndevices(&nhip, nb_devices, 0, STARPU_MAXHIPDEVS, "nhip", "HIP", "maxhipdev");
	}

	/* Now we know how many HIP devices will be used */
	topology->ndevices[STARPU_HIP_WORKER] = nhip;

	_starpu_initialize_workers_hip_gpuid(config);

	unsigned hipgpu;
	for (hipgpu = 0; (int) hipgpu < nhip; hipgpu++)
	{
		int devid = _starpu_get_next_devid(topology, config, STARPU_HIP_WORKER);

		if (devid == -1)
		{
			// There is no more devices left
			topology->ndevices[STARPU_HIP_WORKER] = hipgpu;
			break;
		}

		_starpu_topology_configure_workers(topology, config,
					STARPU_HIP_WORKER,
					hipgpu, devid, 0, 0,
					1, 1, NULL, NULL);
        }
}

/* Bind the driver on a CPU core, set up memory and buses */
int _starpu_hip_init_workers_binding_and_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (hip_init[devid])
	{
		memory_node = hip_memory_nodes[devid];
		workerarg->bindid = hip_bindid[devid];
	}
	else
	{
		hip_init[devid] = 1;

		workerarg->bindid = hip_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
		memory_node = hip_memory_nodes[devid] = _starpu_memory_node_register(STARPU_HIP_RAM, devid);

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_hip_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
			_starpu_hip_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
		}
	}
	_starpu_memory_node_add_nworkers(memory_node);

	//This worker can also manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(workerarg, numa);

	_starpu_worker_drives_memory_node(workerarg, memory_node);

	return memory_node;
}

/* Set the current HIP device */
void starpu_hip_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	hipError_t cures;

	cures = hipSetDevice(devid);

	if (STARPU_UNLIKELY(cures))
		STARPU_HIP_REPORT_ERROR(cures);
}

static void _starpu_hip_limit_gpu_mem_if_needed(unsigned devid)
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
	hipError_t cures;

	starpu_hip_set_device(devid);

	/* force HIP to initialize the context for real */
	#warning "Not sure about the argument of hipInit(flag=0)"
	cures = hipInit(0);
	if (STARPU_UNLIKELY(cures))
	{
		if (cures != hipSuccess)
		{
			_STARPU_MSG("Failed to initialize HIP runtime\n");
			exit(77);
		}
		STARPU_HIP_REPORT_ERROR(cures);
	}

	cures = hipGetDeviceProperties(&props[devid], devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_HIP_REPORT_ERROR(cures);

	_starpu_hip_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_hip_get_global_mem_size(devid));
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
}

/* This is run from the driver thread to initialize the driver HIP context */
int _starpu_hip_driver_init(struct _starpu_worker *worker)
{
	_starpu_driver_start(worker, STARPU_HIP_WORKER, 0);
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

	snprintf(worker->name, sizeof(worker->name), "HIP %u (%s %.1f GiB)", devid, devname, size);
	snprintf(worker->short_name, sizeof(worker->short_name), "HIP %u", devid);
	_STARPU_DEBUG("hip (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

	_STARPU_TRACE_WORKER_INIT_END(workerid);

	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "HIP %u", worker->devid);
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

int _starpu_hip_driver_deinit(struct _starpu_worker *worker)
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
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_HIP_WORKER);

	return 0;
}

uintptr_t _starpu_hip_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	uintptr_t addr = 0;
	(void) flags;

	unsigned devid = starpu_memory_node_get_devid(dst_node);

	starpu_hip_set_device(devid);

	/* Check if there is free memory */
	size_t hip_mem_free, hip_mem_total;
	hipError_t status;
	status = hipMemGetInfo(&hip_mem_free, &hip_mem_total);
	if (status == hipSuccess && hip_mem_free * FREE_MARGIN < size)
	{
		addr = 0;
	}
	else
	{
		status = hipMalloc((void **)&addr, size);
		if (!addr || (status != hipSuccess))
		{
			if (STARPU_UNLIKELY(status != hipErrorOutOfMemory))
				STARPU_HIP_REPORT_ERROR(status);
			addr = 0;
		}
	}
	return addr;
}

void _starpu_hip_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void) dst_node;
	(void) addr;
	(void) size;
	(void) flags;

	hipError_t err;
	unsigned devid = starpu_memory_node_get_devid(dst_node);
	starpu_hip_set_device(devid);
	err = hipFree((void*)addr);
	if (STARPU_UNLIKELY(err != hipSuccess))
		STARPU_HIP_REPORT_ERROR(err);
}

int
starpu_hip_copy_async_sync(void *src_ptr, unsigned src_node,
			    void *dst_ptr, unsigned dst_node,
			    size_t ssize, hipStream_t stream STARPU_ATTRIBUTE_UNUSED,
			    enum hipMemcpyKind kind)
{
	hipError_t cures = 0;

	if (kind == hipMemcpyDeviceToDevice && src_node != dst_node)
	{
		STARPU_ABORT();
	}

	cures = hipMemcpy((char *)dst_ptr, (char *)src_ptr, ssize, kind);
	(void) hipGetLastError();

	if (!cures)
		cures = hipDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_HIP_REPORT_ERROR(cures);

	return 0;
}

int
starpu_hip_copy2d_async_sync(void *src_ptr, unsigned src_node,
			      void *dst_ptr, unsigned dst_node,
			      size_t blocksize,
			      size_t numblocks, size_t ld_src, size_t ld_dst,
			      hipStream_t stream STARPU_ATTRIBUTE_UNUSED, enum hipMemcpyKind kind)
{
	hipError_t cures = 0;

	if (kind == hipMemcpyDeviceToDevice && src_node != dst_node)
	{
		STARPU_ABORT_MSG("HIP memcpy 3D peer not available, but core triggered one ?!");
	}

	cures = hipMemcpy2D((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
			blocksize, numblocks, kind);
	if (!cures)
		cures = hipDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_HIP_REPORT_ERROR(cures);

	return 0;
}

int _starpu_hip_copy_interface(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	(void) req;

	int ret = 1;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	STARPU_ASSERT(copy_methods->any_to_any);
	copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	return ret;
}

int _starpu_hip_copy_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_hip_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   NULL,
					   hipMemcpyDeviceToHost);
}

int _starpu_hip_copy_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_HIP_RAM);
#ifndef STARPU_HAVE_HIP_MEMCPY_PEER
	STARPU_ASSERT(src_node == dst_node);
#endif

	return starpu_hip_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   NULL,
					   hipMemcpyDeviceToDevice);
}

int _starpu_hip_copy_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_HIP_RAM);

	return starpu_hip_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   NULL,
					   hipMemcpyHostToDevice);
}

int _starpu_hip_copy2d_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node,
					      uintptr_t dst, size_t dst_offset, unsigned dst_node,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_hip_copy2d_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   blocksize, numblocks, ld_src, ld_dst,
					   NULL,
					   hipMemcpyDeviceToHost);
}

int _starpu_hip_copy2d_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node,
					       uintptr_t dst, size_t dst_offset, unsigned dst_node,
					       size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					       struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_HIP_RAM);
#ifndef STARPU_HAVE_HIP_MEMCPY_PEER
	STARPU_ASSERT(src_node == dst_node);
#endif

	return starpu_hip_copy2d_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   blocksize, numblocks, ld_src, ld_dst,
					   NULL,
					   hipMemcpyDeviceToDevice);
}

int _starpu_hip_copy2d_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node,
					      uintptr_t dst, size_t dst_offset, unsigned dst_node,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel STARPU_ATTRIBUTE_UNUSED)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_HIP_RAM);

	return starpu_hip_copy2d_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   blocksize, numblocks, ld_src, ld_dst,
					   NULL,
					   hipMemcpyHostToDevice);
}

int _starpu_hip_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	/* Direct GPU-GPU transfers are not allowed in general */
	(void) node;
	(void) handling_node;
	return 0;
}

static int start_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);

	/* Fetch data input synchronously */
	int ret = _starpu_fetch_task_input(task, j, 0);
	if (ret != 0)
	{
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return -EAGAIN;
	}

	_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);

	starpu_hip_func_t func = _starpu_task_get_hip_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_HIP is defined in 'where', hip_func or hip_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		_STARPU_TRACE_END_EXECUTING();
	}

	return 0;
}

static void finish_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static int execute_job_on_hip(struct starpu_task *task, struct _starpu_worker *worker)
{
	int res;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	res = start_job_on_hip(j, worker);

	if (res)
	{
		switch (res)
		{
			case -EAGAIN:
				_STARPU_DISP("ouch, HIP could not actually run task %p, putting it back...\n", task);
				_starpu_push_task_to_workers(task);
				return -EAGAIN;
			default:
				STARPU_ABORT();
		}
	}

	finish_job_on_hip(j, worker);

	return 0;
}

static void finish_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker)
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
int _starpu_hip_driver_run_once(struct _starpu_worker *worker)
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

	/* can HIP do that task ? */
	if (!_STARPU_MAY_PERFORM(j, HIP))
	{
		/* this is neither a hip or a hipblas task */
		_starpu_worker_refuse_task(worker, task);
		return 0;
	}

	worker->current_task = task;

	res = execute_job_on_hip(task, worker);

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

void *_starpu_hip_worker(void *_arg)
{
	struct _starpu_worker *worker = _arg;

	_starpu_hip_driver_init(worker);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_hip_driver_run_once(worker);
	}
	_starpu_hip_driver_deinit(worker);

	return NULL;
}

void starpu_hip_report_error(const char *func, const char *file, int line, hipError_t status)
{
	const char *errormsg = hipGetErrorString(status);
	_STARPU_ERROR("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

int _starpu_hip_run_from_worker(struct _starpu_worker *worker)
{
	_starpu_hip_worker(worker);

	return 0;
}

struct _starpu_driver_ops _starpu_driver_hip_ops =
{
	.init = _starpu_hip_driver_init,
	.run = _starpu_hip_run_from_worker,
	.run_once = _starpu_hip_driver_run_once,
	.deinit = _starpu_hip_driver_deinit,
};

struct _starpu_node_ops _starpu_driver_hip_node_ops =
{
	.name = "hip driver",
	.malloc_on_node = _starpu_hip_malloc_on_node,
	.free_on_node = _starpu_hip_free_on_node,

	.is_direct_access_supported = _starpu_hip_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_hip_copy_interface,
	.copy_interface_to[STARPU_HIP_RAM] = _starpu_hip_copy_interface,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_hip_copy_interface,
	.copy_interface_from[STARPU_HIP_RAM] = _starpu_hip_copy_interface,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_hip_copy_data_from_hip_to_cpu,
	.copy_data_to[STARPU_HIP_RAM] = _starpu_hip_copy_data_from_hip_to_hip,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_hip_copy_data_from_cpu_to_hip,
	.copy_data_from[STARPU_HIP_RAM] = _starpu_hip_copy_data_from_hip_to_hip,

	.copy2d_data_to[STARPU_CPU_RAM] = _starpu_hip_copy2d_data_from_hip_to_cpu,
	.copy2d_data_to[STARPU_HIP_RAM] = _starpu_hip_copy2d_data_from_hip_to_hip,

	.copy2d_data_from[STARPU_CPU_RAM] = _starpu_hip_copy2d_data_from_cpu_to_hip,
	.copy2d_data_from[STARPU_HIP_RAM] = _starpu_hip_copy2d_data_from_hip_to_hip,
};
