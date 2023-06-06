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

/* This is a version of the CUDA driver with reduced features:
 * - asynchronous kernel execution
 * - asynchronous data transfers
 * - peer2peer transfers
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

#if HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/cuda.h>
#endif
#ifdef STARPU_USE_CUDA
#include <cublas.h>
#endif

#if CUDART_VERSION >= 5000
/* Avoid letting our streams spuriously synchonize with the NULL stream */
#define starpu_cudaStreamCreate(stream) cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#else
#define starpu_cudaStreamCreate(stream) cudaStreamCreate(stream)
#endif

/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

/* the number of CUDA devices */
static int ncudagpus = -1;

static size_t global_mem[STARPU_MAXCUDADEVS];
int _starpu_cuda_bus_ids[STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES][STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES];
static cudaStream_t streams[STARPU_NMAXWORKERS];
static char used_stream[STARPU_NMAXWORKERS];
static cudaStream_t out_transfer_streams[STARPU_MAXCUDADEVS];
static cudaStream_t in_transfer_streams[STARPU_MAXCUDADEVS];
/* Note: streams are not thread-safe, so we define them for each CUDA worker
 * emitting a GPU-GPU transfer */
static cudaStream_t in_peer_transfer_streams[STARPU_MAXCUDADEVS][STARPU_MAXCUDADEVS];
static struct cudaDeviceProp props[STARPU_MAXCUDADEVS];
static cudaEvent_t task_events[STARPU_NMAXWORKERS];

static unsigned cuda_bindid_init[STARPU_MAXCUDADEVS];
static unsigned cuda_bindid[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_init[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_nodes[STARPU_MAXCUDADEVS];

int _starpu_nworker_per_cuda = 1;

static size_t _starpu_cuda_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}

cudaStream_t starpu_cuda_get_local_in_transfer_stream()
{
	int worker = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(worker);
	cudaStream_t stream;

	stream = in_transfer_streams[devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_in_transfer_stream(unsigned dst_node)
{
	int dst_devid = starpu_memory_node_get_devid(dst_node);
	cudaStream_t stream;

	stream = in_transfer_streams[dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_local_out_transfer_stream()
{
	int worker = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(worker);
	cudaStream_t stream;

	stream = out_transfer_streams[devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_out_transfer_stream(unsigned src_node)
{
	int src_devid = starpu_memory_node_get_devid(src_node);
	cudaStream_t stream;

	stream = out_transfer_streams[src_devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_peer_transfer_stream(unsigned src_node, unsigned dst_node)
{
	int src_devid = starpu_memory_node_get_devid(src_node);
	int dst_devid = starpu_memory_node_get_devid(dst_node);
	cudaStream_t stream;

	stream = in_peer_transfer_streams[src_devid][dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_local_stream(void)
{
	int worker = starpu_worker_get_id_check();

	used_stream[worker] = 1;
	return streams[worker];
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

/* This is called to really discover the hardware */
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
void _starpu_cuda_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
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

		int worker2;
		for (worker2 = 0; worker2 < workerarg->workerid; worker2++)
		{
			struct _starpu_worker *workerarg2 = &config->workers[worker2];
			int devid2 = workerarg2->devid;
			if (workerarg2->arch == STARPU_CUDA_WORKER)
			{
				unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
				int bus21 STARPU_ATTRIBUTE_UNUSED = _starpu_cuda_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node2, memory_node);
				int bus12 STARPU_ATTRIBUTE_UNUSED = _starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node, memory_node2);
				STARPU_ASSERT(bus21 >= 0);
				STARPU_ASSERT(bus12 >= 0);
#if HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
				{
					hwloc_obj_t obj, obj2, ancestor;
					obj = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, devid);
					obj2 = hwloc_cuda_get_device_osdev_by_index(config->topology.hwtopology, devid2);
					ancestor = hwloc_get_common_ancestor_obj(config->topology.hwtopology, obj, obj2);
					if (ancestor)
					{
						struct _starpu_hwloc_userdata *data = ancestor->userdata;
#ifdef STARPU_VERBOSE
						{
							char name[64];
							hwloc_obj_type_snprintf(name, sizeof(name), ancestor, 0);
							_STARPU_DEBUG("CUDA%u and CUDA%u are linked through %s, along %u GPUs\n", devid, devid2, name, data->ngpus);
						}
#endif
						starpu_bus_set_ngpus(bus21, data->ngpus);
						starpu_bus_set_ngpus(bus12, data->ngpus);
					}
				}
#endif
			}
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
void starpu_cuda_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
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
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (props[devid].computeMode == cudaComputeModeExclusive)
	{
		_STARPU_MSG("CUDA is in EXCLUSIVE-THREAD mode, but StarPU was built with multithread GPU control support, please either ask your administrator to use EXCLUSIVE-PROCESS mode (which should really be fine), or reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#endif

	cures = starpu_cudaStreamCreate(&in_transfer_streams[devid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = starpu_cudaStreamCreate(&out_transfer_streams[devid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	int i;
	for (i = 0; i < ncudagpus; i++)
	{
		cures = starpu_cudaStreamCreate(&in_peer_transfer_streams[i][devid]);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}

	_starpu_cuda_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cuda_get_global_mem_size(devid));
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	int i;
	starpu_cuda_set_device(devid);

	cudaStreamDestroy(in_transfer_streams[devid]);
	cudaStreamDestroy(out_transfer_streams[devid]);

	for (i = 0; i < ncudagpus; i++)
	{
		cudaStreamDestroy(in_peer_transfer_streams[i][devid]);
	}
}

static void init_worker_context(unsigned workerid, unsigned devid)
{
	cudaError_t cures;
	starpu_cuda_set_device(devid);

	cures = cudaEventCreateWithFlags(&task_events[workerid], cudaEventDisableTiming);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = starpu_cudaStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void deinit_worker_context(unsigned workerid, unsigned devid)
{
	starpu_cuda_set_device(devid);
	cudaEventDestroy(task_events[workerid]);
	cudaStreamDestroy(streams[workerid]);
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

	snprintf(worker->name, sizeof(worker->name), "CUDA1 %u (%s %.1f GiB)", devid, devname, size);
	snprintf(worker->short_name, sizeof(worker->short_name), "CUDA %u", devid);
	_STARPU_DEBUG("cuda (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

	init_worker_context(workerid, worker->devid);

	_STARPU_TRACE_WORKER_INIT_END(workerid);

	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "CUDA1 %u", worker->devid);
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

	unsigned workerid = worker->workerid;

	deinit_worker_context(workerid, worker->devid);

	worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_CUDA_WORKER);

	return 0;
}

uintptr_t _starpu_cuda_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	uintptr_t addr = 0;
	(void) flags;

	unsigned devid = starpu_memory_node_get_devid(dst_node);

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

void _starpu_cuda_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void) dst_node;
	(void) addr;
	(void) size;
	(void) flags;

	cudaError_t err;
	unsigned devid = starpu_memory_node_get_devid(dst_node);
	starpu_cuda_set_device(devid);
	err = cudaFree((void*)addr);
	if (STARPU_UNLIKELY(err != cudaSuccess))
		STARPU_CUDA_REPORT_ERROR(err);
}

int starpu_cuda_copy_async_sync(void *src_ptr, unsigned src_node,
				void *dst_ptr, unsigned dst_node,
				size_t ssize, cudaStream_t stream,
				enum cudaMemcpyKind kind)
{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	int peer_copy = 0;
	int src_dev = -1, dst_dev = -1;
#endif
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		peer_copy = 1;
		src_dev = starpu_memory_node_get_devid(src_node);
		dst_dev = starpu_memory_node_get_devid(dst_node);
#else
		STARPU_ABORT();
#endif
	}

	if (stream)
	{
		double start;
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		if (peer_copy)
		{
			cures = cudaMemcpyPeerAsync((char *) dst_ptr, dst_dev,
						    (char *) src_ptr, src_dev,
						    ssize, stream);
		}
		else
#endif
		{
			cures = cudaMemcpyAsync((char *)dst_ptr, (char *)src_ptr, ssize, kind, stream);
		}
		(void) cudaGetLastError();
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	}

	/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
	if (stream == NULL || cures)
	{
	/* do it in a synchronous fashion */
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		if (peer_copy)
		{
			cures = cudaMemcpyPeer((char *) dst_ptr, dst_dev,
					       (char *) src_ptr, src_dev,
					       ssize);
		}
		else
#endif
		{
	cures = cudaMemcpy((char *)dst_ptr, (char *)src_ptr, ssize, kind);
		}
	(void) cudaGetLastError();

	if (!cures)
		cures = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	return 0;
	}

	return -EAGAIN;
}

/* Driver porters: this is optional but really recommended */
int
starpu_cuda_copy2d_async_sync(void *src_ptr, unsigned src_node,
			      void *dst_ptr, unsigned dst_node,
			      size_t blocksize,
			      size_t numblocks, size_t ld_src, size_t ld_dst,
			      cudaStream_t stream, enum cudaMemcpyKind kind)
{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	int peer_copy = 0;
	int src_dev = -1, dst_dev = -1;
#endif
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
#  ifdef BUGGED_MEMCPY3D
		STARPU_ABORT_MSG("CUDA memcpy 3D peer buggy, but core triggered one?!");
#  endif
		peer_copy = 1;
		src_dev = starpu_memory_node_get_devid(src_node);
		dst_dev = starpu_memory_node_get_devid(dst_node);
#else
		STARPU_ABORT_MSG("CUDA memcpy 3D peer not available, but core triggered one ?!");
#endif
	}

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (peer_copy)
	{
		struct cudaMemcpy3DPeerParms p;
		memset(&p, 0, sizeof(p));

		p.srcDevice = src_dev;
		p.dstDevice = dst_dev;
		p.srcPtr = make_cudaPitchedPtr((char *)src_ptr, ld_src, blocksize, numblocks);
		p.dstPtr = make_cudaPitchedPtr((char *)dst_ptr, ld_dst, blocksize, numblocks);
		p.extent = make_cudaExtent(blocksize, numblocks, 1);


		if (stream)
		{
			double start;
			starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
			cures = cudaMemcpy3DPeerAsync(&p, stream);
			(void) cudaGetLastError();
		}

		/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
		if (stream == NULL || cures)
		{
			cures = cudaMemcpy3DPeer(&p);
			(void) cudaGetLastError();

			if (!cures)
				cures = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);

			return 0;
		}
	}
	else
#endif
	{
		if (stream)
		{
			double start;
			starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
			cures = cudaMemcpy2DAsync((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
				blocksize, numblocks, kind, stream);
			starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	}

		/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
		if (stream == NULL || cures)
		{
			cures = cudaMemcpy2D((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
					     blocksize, numblocks, kind);
			if (!cures)
				cures = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);

			return 0;
		}
	}

	return -EAGAIN;
}

static inline cudaEvent_t *_starpu_cuda_event(union _starpu_async_channel_event *_event)
{
	cudaEvent_t *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (void *) _event;
	return event;
}

unsigned _starpu_cuda_test_request_completion(struct _starpu_async_channel *async_channel)
{
	cudaEvent_t event;
	cudaError_t cures;
	unsigned success;

	event = *_starpu_cuda_event(&async_channel->event);
	cures = cudaEventQuery(event);
	success = (cures == cudaSuccess);

	if (success)
		cudaEventDestroy(event);
	else if (cures != cudaErrorNotReady)
		STARPU_CUDA_REPORT_ERROR(cures);

	return success;
}

void _starpu_cuda_wait_request_completion(struct _starpu_async_channel *async_channel)
{
	cudaEvent_t event;
	cudaError_t cures;

	event = *_starpu_cuda_event(&async_channel->event);

	cures = cudaEventSynchronize(event);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaEventDestroy(event);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
static void
starpu_cuda_set_copy_device(unsigned src_node, unsigned dst_node)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	unsigned devid;
	if ((src_kind == STARPU_CUDA_RAM) && (dst_kind == STARPU_CUDA_RAM))
	{
		/* GPU-GPU transfer, issue it from the destination */
		devid = starpu_memory_node_get_devid(dst_node);
	}
	else
	{
		unsigned node = (dst_kind == STARPU_CUDA_RAM)?dst_node:src_node;
		devid = starpu_memory_node_get_devid(node);
	}
	starpu_cuda_set_device(devid);
}
#endif

int _starpu_cuda_copy_interface_from_cuda_to_cuda(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CUDA_RAM);

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	starpu_cuda_set_copy_device(src_node, dst_node);
#else
	STARPU_ASSERT(src_node == dst_node);
#endif

	int ret = 1;
	cudaError_t cures;
	cudaStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
/* CUDA - CUDA transfer */
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() || !copy_methods->any_to_any)
	{
		STARPU_ASSERT(copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_peer_transfer_stream(src_node, dst_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		cures = cudaEventRecord(*_starpu_cuda_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
	}
	return ret;
}

int _starpu_cuda_copy_interface_from_cuda_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CPU_RAM);

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	starpu_cuda_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	cudaError_t cures;
	cudaStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* only the proper CUBLAS thread can initiate this directly ! */
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == src_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() || !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_out_transfer_stream(src_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		cures = cudaEventRecord(*_starpu_cuda_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);
	}
	return ret;
}

int _starpu_cuda_copy_interface_from_cpu_to_cuda(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_CUDA_RAM);

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	starpu_cuda_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	cudaError_t cures;
	cudaStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* STARPU_CPU_RAM -> CUBLAS_RAM */
	/* only the proper CUBLAS thread can initiate this ! */
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == dst_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() ||
	    !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess))
			STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_in_transfer_stream(dst_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		cures = cudaEventRecord(*_starpu_cuda_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(cures != cudaSuccess))
			STARPU_CUDA_REPORT_ERROR(cures);
	}
	return ret;
}

int _starpu_cuda_copy_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_cuda_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   async_channel?starpu_cuda_get_out_transfer_stream(src_node):NULL,
					   cudaMemcpyDeviceToHost);
}

int _starpu_cuda_copy_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CUDA_RAM);
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
	STARPU_ASSERT(src_node == dst_node);
#endif

	return starpu_cuda_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   async_channel?starpu_cuda_get_peer_transfer_stream(src_node, dst_node):NULL,
					   cudaMemcpyDeviceToDevice);
}

int _starpu_cuda_copy_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_CUDA_RAM);

	return starpu_cuda_copy_async_sync((void*) (src + src_offset), src_node,
					   (void*) (dst + dst_offset), dst_node,
					   size,
					   async_channel?starpu_cuda_get_in_transfer_stream(dst_node):NULL,
					   cudaMemcpyHostToDevice);
}

int _starpu_cuda_copy2d_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node,
					      uintptr_t dst, size_t dst_offset, unsigned dst_node,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_cuda_copy2d_async_sync((void*) (src + src_offset), src_node,
					     (void*) (dst + dst_offset), dst_node,
					     blocksize, numblocks, ld_src, ld_dst,
					     async_channel?starpu_cuda_get_out_transfer_stream(src_node):NULL,
					     cudaMemcpyDeviceToHost);
}

int _starpu_cuda_copy2d_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node,
					       uintptr_t dst, size_t dst_offset, unsigned dst_node,
					       size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					       struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CUDA_RAM && dst_kind == STARPU_CUDA_RAM);
#ifndef STARPU_HAVE_CUDA_MEMCPY_PEER
	STARPU_ASSERT(src_node == dst_node);
#endif

	return starpu_cuda_copy2d_async_sync((void*) (src + src_offset), src_node,
					     (void*) (dst + dst_offset), dst_node,
					     blocksize, numblocks, ld_src, ld_dst,
					     async_channel?starpu_cuda_get_peer_transfer_stream(src_node, dst_node):NULL,
					     cudaMemcpyDeviceToDevice);
}

int _starpu_cuda_copy2d_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node,
					      uintptr_t dst, size_t dst_offset, unsigned dst_node,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_CUDA_RAM);

	return starpu_cuda_copy2d_async_sync((void*) (src + src_offset), src_node,
					     (void*) (dst + dst_offset), dst_node,
					     blocksize, numblocks, ld_src, ld_dst,
					     async_channel?starpu_cuda_get_in_transfer_stream(dst_node):NULL,
					     cudaMemcpyHostToDevice);
}

int _starpu_cuda_is_direct_access_supported(unsigned node, unsigned handling_node)
{
#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return kind == STARPU_CUDA_RAM;
#else /* STARPU_HAVE_CUDA_MEMCPY_PEER */
	/* Direct GPU-GPU transfers are not allowed in general */
	(void) node;
	(void) handling_node;
	return 0;
#endif /* STARPU_HAVE_CUDA_MEMCPY_PEER */
}

static void start_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);
	j->workerid = worker->workerid;

	_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);

#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	/* We make sure we do manipulate the proper device */
	starpu_cuda_set_device(worker->devid);
#endif

	starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_CUDA is defined in 'where', cuda_func or cuda_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		_STARPU_TRACE_END_EXECUTING();
	}
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static void execute_job_on_cuda(struct starpu_task *task, struct _starpu_worker *worker)
{
	int workerid = worker->workerid;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	start_job_on_cuda(j, worker);

	if (!used_stream[workerid])
	{
		used_stream[workerid] = 1;
		_STARPU_DISP("Warning: starpu_cuda_get_local_stream() was not used to submit kernel to CUDA on worker %d. CUDA will thus introduce a lot of useless synchronizations, which will prevent proper overlapping of data transfers and kernel execution. See the CUDA-specific part of the 'Check List When Performance Are Not There' of the StarPU handbook\n", workerid);
	}

	if (task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC)
	{
		/* Record event to synchronize with task termination later */
		cudaError_t cures = cudaEventRecord(task_events[workerid], starpu_cuda_get_local_stream());
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
#ifdef STARPU_USE_FXT
		_STARPU_TRACE_START_EXECUTING();
#endif
	}
	else
	/* Synchronous execution */
	{
#if !defined(STARPU_SIMGRID)
		STARPU_ASSERT_MSG(cudaStreamQuery(starpu_cuda_get_local_stream()) == cudaSuccess, "Unless when using the STARPU_CUDA_ASYNC flag, CUDA codelets have to wait for termination of their kernels on the starpu_cuda_get_local_stream() stream");
#endif
		finish_job_on_cuda(j, worker);
	}
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker)
{
	int profiling = starpu_profiling_status_get();

	worker->current_task = NULL;
	worker->ntasks--;

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

	int idle_tasks, idle_transfers;

	/* First poll for completed jobs */
	idle_tasks = 0;
	idle_transfers = 0;
	int workerid = worker->workerid;
	unsigned memnode = worker->memory_node;

	do /* This do {} while (0) is only to match the cuda driver worker for look */
	{
		if (!worker->ntasks)
			idle_tasks++;
		if (!worker->task_transferring)
			idle_transfers++;

		if (!worker->ntasks && !worker->task_transferring)
		{
			/* Even nothing to test */
			continue;
		}

		/* First test for transfers pending for next task */
		task = worker->task_transferring;
		if (task && worker->nb_buffers_transferred == worker->nb_buffers_totransfer)
		{
			STARPU_RMB();
			_STARPU_TRACE_END_PROGRESS(memnode);
			j = _starpu_get_job_associated_to_task(task);

			_starpu_fetch_task_input_tail(task, j, worker);
			/* Reset it */
			worker->task_transferring = NULL;

			execute_job_on_cuda(task, worker);
			_STARPU_TRACE_START_PROGRESS(memnode);
		}

		/* Then test for termination of queued tasks */
		if (!worker->ntasks)
			/* No queued task */
			continue;

		task = worker->current_task;
		if (task == worker->task_transferring)
			/* Next task is still pending transfer */
			continue;

		/* On-going asynchronous task, check for its termination first */
		cudaError_t cures = cudaEventQuery(task_events[workerid]);

		if (cures != cudaSuccess)
		{
			STARPU_ASSERT_MSG(cures == cudaErrorNotReady, "CUDA error on task %p, codelet %p (%s): %s (%d)", task, task->cl, _starpu_codelet_get_model_name(task->cl), cudaGetErrorString(cures), cures);
		}
		else
		{
			_STARPU_TRACE_END_PROGRESS(memnode);
			/* Asynchronous task completed! */
			finish_job_on_cuda(_starpu_get_job_associated_to_task(task), worker);
#ifdef STARPU_USE_FXT
			_STARPU_TRACE_END_EXECUTING()
#endif
			_STARPU_TRACE_START_PROGRESS(memnode);
		}
		if (worker->ntasks < 1)
			idle_tasks++;
	} while(0);

#if defined(STARPU_NON_BLOCKING_DRIVERS)
	if (!idle_tasks)
	{
		/* No task ready yet, no better thing to do than waiting */
		__starpu_datawizard_progress(_STARPU_DATAWIZARD_DO_ALLOC, !idle_transfers);
		return 0;
	}
#endif

	/* Something done, make some progress */
	res = __starpu_datawizard_progress(_STARPU_DATAWIZARD_DO_ALLOC, 1);

	if (worker->ntasks >= 1)
		return 0;

	/* And pull a task */
	task = _starpu_get_worker_task(worker, worker->workerid, worker->memory_node);

	if (!task)
		return 0;

	worker->ntasks++;

	j = _starpu_get_job_associated_to_task(task);

	/* can CUDA do that task ? */
	if (!_STARPU_MAY_PERFORM(j, CUDA))
	{
		/* this is neither a cuda or a cublas task */
		_starpu_worker_refuse_task(worker, task);
		return 0;
	}

	worker->current_task = task;

	/* Fetch data asynchronously */
	_STARPU_TRACE_END_PROGRESS(memnode);
	_starpu_set_local_worker_key(worker);
	res = _starpu_fetch_task_input(task, j, 1);
	STARPU_ASSERT(res == 0);
	_STARPU_TRACE_START_PROGRESS(memnode);

	return 0;
}

void *_starpu_cuda_worker(void *_arg)
{
	struct _starpu_worker *worker = _arg;

	_starpu_cuda_driver_init(worker);
	_STARPU_TRACE_START_PROGRESS(worker->memory_node);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_cuda_driver_run_once(worker);
	}
	_STARPU_TRACE_END_PROGRESS(worker->memory_node);
	_starpu_cuda_driver_deinit(worker);

	return NULL;
}


#ifdef STARPU_HAVE_HWLOC
hwloc_obj_t _starpu_cuda_get_hwloc_obj(hwloc_topology_t topology, int devid)
{
#if !defined(STARPU_SIMGRID) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
	return hwloc_cuda_get_device_osdev_by_index(topology, devid);
#else
	return NULL;
#endif
}
#endif

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
	.name = "cuda1 driver",
	.malloc_on_node = _starpu_cuda_malloc_on_node,
	.free_on_node = _starpu_cuda_free_on_node,

	.is_direct_access_supported = _starpu_cuda_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_cuda_copy_interface_from_cuda_to_cpu,
	.copy_interface_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface_from_cuda_to_cuda,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_cuda_copy_interface_from_cpu_to_cuda,
	.copy_interface_from[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface_from_cuda_to_cuda,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_cuda_copy_data_from_cuda_to_cpu,
	.copy_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cuda_to_cuda,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_cuda_copy_data_from_cpu_to_cuda,
	.copy_data_from[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cuda_to_cuda,

	.copy2d_data_to[STARPU_CPU_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cpu,
	.copy2d_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cuda,

	.copy2d_data_from[STARPU_CPU_RAM] = _starpu_cuda_copy2d_data_from_cpu_to_cuda,
	.copy2d_data_from[STARPU_CUDA_RAM] = _starpu_cuda_copy2d_data_from_cuda_to_cuda,

	.wait_request_completion = _starpu_cuda_wait_request_completion,
	.test_request_completion = _starpu_cuda_test_request_completion,
};
