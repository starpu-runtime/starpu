/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021-2021  Federal University of Rio Grande do Sul (UFRGS)
 * Copyright (C) 2016-2016  Uppsala University
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2011-2011  Télécom Sud Paris
 * Copyright (C) 2010-2010  Mehdi Juhoor
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

#define DPCT_COMPAT_RT_VERSION 12060

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

extern "C" {
#include <starpu.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/devices.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_sycl.h"
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <common/knobs.h>
}

#include <starpu_sycl.hpp>

#ifdef STARPU_USE_SYCL
#include <starpu_syclblas.h>
#endif

/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

static size_t global_mem[STARPU_MAXSYCLDEVS];
int _starpu_sycl_bus_ids[STARPU_MAXSYCLDEVS+STARPU_MAXNUMANODES][STARPU_MAXSYCLDEVS+STARPU_MAXNUMANODES];
static dpct::queue_ptr streams[STARPU_NMAXWORKERS];
static dpct::queue_ptr streams_bus[STARPU_NMAXWORKERS];
static char used_stream[STARPU_NMAXWORKERS];
static dpct::queue_ptr out_transfer_streams[STARPU_MAXSYCLDEVS];
static dpct::queue_ptr in_transfer_streams[STARPU_MAXSYCLDEVS];
/* Note: streams are not thread-safe, so we define them for each SYCL worker
 * emitting a GPU-GPU transfer */
static dpct::queue_ptr in_peer_transfer_streams[STARPU_MAXSYCLDEVS][STARPU_MAXSYCLDEVS];
static dpct::device_info props[STARPU_MAXSYCLDEVS];
static dpct::event_ptr task_events[STARPU_NMAXWORKERS];

static unsigned sycl_bindid_init[STARPU_MAXSYCLDEVS];
static unsigned sycl_bindid[STARPU_MAXSYCLDEVS];
static unsigned sycl_memory_init[STARPU_MAXSYCLDEVS];
static unsigned sycl_memory_nodes[STARPU_MAXSYCLDEVS];

static bool workers_initialized = false;

int _starpu_nworker_per_sycl = 1;
static int nsyclgpus = -1;

static size_t _starpu_sycl_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}

static dpct::queue_ptr starpu_sycl_get_in_transfer_stream(int dst_devid)
{
	dpct::queue_ptr stream;

	stream = in_transfer_streams[dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static dpct::queue_ptr starpu_sycl_get_out_transfer_stream(int src_devid)
{
	dpct::queue_ptr stream;

	stream = out_transfer_streams[src_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static dpct::queue_ptr starpu_sycl_get_peer_transfer_stream(int src_devid, int dst_devid)
{
	dpct::queue_ptr stream;

	stream = in_peer_transfer_streams[src_devid][dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

const dpct::queue_ptr& starpu_sycl_get_local_stream(int devid)
{
	if (workers_initialized)
	{
		int worker = starpu_worker_get_id_check();

		used_stream[worker] = 1;
		return streams[worker];
	}
	else
	{
                // starpu not initialized yet
                STARPU_ASSERT(devid >= 0);
		return streams_bus[devid];
	}
}

const dpct::device_info *starpu_sycl_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}

/* Early library initialization, before anything else, just initialize data */
void _starpu_sycl_early_init(void)
{
        workers_initialized = true;
	memset(&sycl_bindid_init, 0, sizeof(sycl_bindid_init));
	memset(&sycl_memory_init, 0, sizeof(sycl_memory_init));
}

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXSYCLDEVS */
static unsigned _starpu_get_sycl_device_count(void) try
{
	int cnt;
	dpct::err0 syclres;
	syclres = DPCT_CHECK_ERROR(cnt = dpct::device_count());
	if (STARPU_UNLIKELY(syclres))
		return 0;

	if (cnt > STARPU_MAXSYCLDEVS)
	{
		_STARPU_MSG("# Warning: %d SYCL devices available. Only %d enabled. Use configure option --enable-maxsycldev=xxx to update the maximum value of supported SYCL devices.\n", cnt, STARPU_MAXSYCLDEVS);
		cnt = STARPU_MAXSYCLDEVS;
	}
	return (unsigned)cnt;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

/* This is run from initialize to determine the number of SYCL devices */
void _starpu_init_sycl(void)
{
	if(nsyclgpus < 0)
	{
		nsyclgpus = _starpu_get_sycl_device_count();
		STARPU_ASSERT(nsyclgpus <= STARPU_MAXSYCLDEVS);
	}
}

/* This is called to really discover the hardware */
void _starpu_sycl_discover_devices(struct _starpu_machine_config *config) try
{
	/* Discover the number of SYCL devices. Fill the result in CONFIG. */

	int cnt;
	dpct::err0 syclres;

	syclres = DPCT_CHECK_ERROR(cnt = dpct::device_count());
	if (STARPU_UNLIKELY(syclres != 0))
		cnt = 0;
	config->topology.nhwdevices[STARPU_SYCL_WORKER] = cnt;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void _starpu_initialize_workers_sycl_gpuid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(uconf->use_explicit_workers_sycl_gpuid == 0
					    ? NULL
					    : (int *)uconf->workers_sycl_gpuid,
					    &(config->current_devid[STARPU_SYCL_WORKER]),
					    (int *)topology->workers_devid[STARPU_SYCL_WORKER],
					    "STARPU_WORKERS_SYCLID",
					    topology->nhwdevices[STARPU_SYCL_WORKER],
					    STARPU_SYCL_WORKER);
	_starpu_devices_drop_duplicate(topology->workers_devid[STARPU_SYCL_WORKER]);
}

/* Determine which devices we will use */
void _starpu_init_sycl_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED)
{
	int nsycl = config->conf.nsycl;

	if (nsycl != 0)
	{
		/* The user did not disable SYCL. We need to
		 * initialize SYCL early to count the number of
		 * devices
		 */
		_starpu_init_sycl();
		int nb_devices = _starpu_get_sycl_device_count();

		_starpu_topology_check_ndevices(&nsycl, nb_devices, 0, STARPU_MAXSYCLDEVS, 0, "nsycl", "SYCL", "NSYCL", "maxsycldev");
	}

	/* Now we know how many SYCL devices will be used */
	topology->ndevices[STARPU_SYCL_WORKER] = nsycl;

	_starpu_initialize_workers_sycl_gpuid(config);

	unsigned syclgpu;
	for (syclgpu = 0; (int) syclgpu < nsycl; syclgpu++)
	{
		int devid = _starpu_get_next_devid(topology, config, STARPU_SYCL_WORKER);

		if (devid == -1)
		{
			// There is no more devices left
			topology->ndevices[STARPU_SYCL_WORKER] = syclgpu;
			break;
		}

		_starpu_topology_configure_workers(topology, config,
				STARPU_SYCL_WORKER,
				syclgpu, devid, 0, 0,
				1, 1, NULL, NULL);
	}

	/* Don't copy this, just here for other code to work fine */
	topology->sycl_th_per_stream = 0;
	topology->sycl_th_per_dev = 1;
}

/* Bind the driver on a CPU core */
void _starpu_sycl_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

	if (sycl_bindid_init[devid])
	{
		workerarg->bindid = sycl_bindid[devid];
	}
	else
	{
		sycl_bindid_init[devid] = 1;

		workerarg->bindid = sycl_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
	}
}

/* Set up memory and buses */
void _starpu_sycl_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (sycl_memory_init[devid])
	{
		memory_node = sycl_memory_nodes[devid];
	}
	else
	{
		sycl_memory_init[devid] = 1;

		memory_node = sycl_memory_nodes[devid] = _starpu_memory_node_register(STARPU_SYCL_RAM, devid);

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_sycl_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
			_starpu_sycl_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
		}

		int worker2;
		for (worker2 = 0; worker2 < workerarg->workerid; worker2++)
		{
			struct _starpu_worker *workerarg2 = &config->workers[worker2];
			int devid2 = workerarg2->devid;
			if (workerarg2->arch == STARPU_SYCL_WORKER)
			{
				unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
				int bus21 STARPU_ATTRIBUTE_UNUSED = _starpu_sycl_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node2, memory_node);
				int bus12 STARPU_ATTRIBUTE_UNUSED = _starpu_sycl_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node, memory_node2);
				STARPU_ASSERT(bus21 >= 0);
				STARPU_ASSERT(bus12 >= 0);
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

/* Set the current SYCL device */
void starpu_sycl_set_device(int devid STARPU_ATTRIBUTE_UNUSED) try
{
	dpct::err0 syclres;

	/* DPCT_ORIG 	syclres = syclSetDevice(devid);*/
	/*
	  DPCT1093:1: The "devid" device may be not the one intended for use.
	  Adjust the selected device if needed.
	*/
	syclres = DPCT_CHECK_ERROR(dpct::select_device(devid));

	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void _starpu_sycl_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

	/* Find the size of the memory on the device */
	totalGlobalMem = props[devid].get_global_mem_size();

	limit = totalGlobalMem / (1024*1024) * FREE_MARGIN;

	global_mem[devid] = limit * 1024*1024;
}

/* hack to force the initialization of a queue */
static void force_queue_init(const dpct::queue_ptr& stream)
{
	char *a = (char*)sycl::malloc_device(sizeof(*a), *stream);
	char b = 0;
	dpct::err0 syclres = DPCT_CHECK_ERROR(stream->memcpy(a, &b, sizeof(*a)));
	if (STARPU_UNLIKELY(syclres))
	{
		STARPU_SYCL_REPORT_ERROR(syclres);
	}
	sycl::free(a, *stream);
}

/* Really initialize one device */
static void init_device_context(unsigned devid, unsigned memnode) try
{
	STARPU_ASSERT(devid < STARPU_MAXSYCLDEVS);

	dpct::err0 syclres;

	starpu_sycl_set_device(devid);

	syclres = DPCT_CHECK_ERROR(dpct::get_device(devid).get_device_info(props[devid]));

	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	syclres = DPCT_CHECK_ERROR(in_transfer_streams[devid] = dpct::get_current_device().create_in_order_queue());
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	syclres = DPCT_CHECK_ERROR(out_transfer_streams[devid] = dpct::get_current_device().create_in_order_queue());
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	force_queue_init(in_transfer_streams[devid]);
	force_queue_init(out_transfer_streams[devid]);

	int nworkers = starpu_worker_get_count();
	int workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		if (worker->arch == STARPU_SYCL_WORKER && worker->subworkerid == 0)
		{
			syclres = DPCT_CHECK_ERROR(in_peer_transfer_streams[worker->devid][devid] = dpct::get_current_device().create_in_order_queue());
			if (STARPU_UNLIKELY(syclres))
				STARPU_SYCL_REPORT_ERROR(syclres);
			force_queue_init(in_peer_transfer_streams[worker->devid][devid]);
		}
	}

	_starpu_sycl_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_sycl_get_global_mem_size(devid));
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	starpu_sycl_set_device(devid);

	dpct::get_current_device().destroy_queue(in_transfer_streams[devid]);
	dpct::get_current_device().destroy_queue(out_transfer_streams[devid]);

	int nworkers = starpu_worker_get_count();
	int workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
		if (worker->arch == STARPU_SYCL_WORKER && worker->subworkerid == 0)
		{
			dpct::get_current_device().destroy_queue(in_peer_transfer_streams[worker->devid][devid]);
		}
	}
}

static void init_worker_context(unsigned workerid, unsigned devid) try
{
	dpct::err0 syclres;
	starpu_sycl_set_device(devid);

	syclres = DPCT_CHECK_ERROR(task_events[workerid] = new sycl::event());
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	syclres = DPCT_CHECK_ERROR(streams[workerid] = dpct::get_current_device().create_in_order_queue());
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	force_queue_init(streams[workerid]);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void deinit_worker_context(unsigned workerid, unsigned devid)
{
	starpu_sycl_set_device(devid);
	dpct::destroy_event(task_events[workerid]);
	dpct::get_current_device().destroy_queue(streams[workerid]);
}

/* This is run from the driver thread to initialize the driver SYCL context */
static int _starpu_sycl_driver_init(struct _starpu_worker *worker)
{
	_starpu_driver_start(worker, STARPU_SYCL_WORKER, 0);
	_starpu_set_local_worker_key(worker);

	unsigned devid = worker->devid;
	unsigned memnode = worker->memory_node;

	init_device_context(devid, memnode);

	unsigned workerid = worker->workerid;

	float size = (float) global_mem[devid] / (1<<30);
	/* get the device's name */
	char devname[64];
	strncpy(devname, props[devid].get_name(), 63);
	devname[63] = 0;

	snprintf(worker->name, sizeof(worker->name), "SYCL %u (%s %.1f GiB)", devid, devname, size);
	snprintf(worker->short_name, sizeof(worker->short_name), "SYCL %u", devid);
	_STARPU_DEBUG("sycl (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

	init_worker_context(workerid, worker->devid);

	_starpu_trace_worker_init_end(worker, STARPU_SYCL_WORKER);

	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "SYCL %u", worker->devid);
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

static int _starpu_sycl_driver_deinit(struct _starpu_worker *worker)
{
	_starpu_trace_worker_deinit_start;

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
	_starpu_trace_worker_deinit_end(workerid, STARPU_SYCL_WORKER);

	return 0;
}

uintptr_t _starpu_sycl_malloc_on_device(int devid, size_t size, int flags) try
{
	uintptr_t addr = 0;
	(void) flags;

	starpu_sycl_set_device(devid);

	auto sycl_mem_total = dpct::get_current_device().get_device_info().get_global_mem_size();
	if (sycl_mem_total > size)
	{

		addr = (unsigned long) sycl::malloc_device(size, *starpu_sycl_get_local_stream(devid));
	}
	else
	{
		addr = 0;
	}

	return addr;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

uintptr_t _starpu_sycl_malloc_on_host(size_t size) try
{
	uintptr_t addr = 0;
	addr = (unsigned long)sycl::malloc_host(size, dpct::get_default_queue());
	return addr;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void _starpu_sycl_memset_on_device(uintptr_t ptr, int c, size_t size) try
{
	dpct::err0 status;
	status = DPCT_CHECK_ERROR(starpu_sycl_get_local_stream()->memset((void *)ptr, c, size).wait());
	if (STARPU_UNLIKELY(status != 0))
		STARPU_SYCL_REPORT_ERROR(status);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void _starpu_sycl_free_on_device(int devid, uintptr_t addr, size_t size, int flags) try
{
	(void) devid;
	(void) addr;
	(void) size;
	(void) flags;

	dpct::err0 err;
	starpu_sycl_set_device(devid);
	err = DPCT_CHECK_ERROR(dpct::dpct_free((void *)addr, *starpu_sycl_get_local_stream(devid)));
	if (STARPU_UNLIKELY(err != 0))
		STARPU_SYCL_REPORT_ERROR(err);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

void _starpu_sycl_free_on_host(uintptr_t addr) try
{
	dpct::err0 err;
        // can be called outside worker so dont use `starpu_sycl_get_local_stream`
	err = DPCT_CHECK_ERROR(dpct::dpct_free((void *)addr, dpct::get_default_queue()));
	if (STARPU_UNLIKELY(err != 0))
		STARPU_SYCL_REPORT_ERROR(err);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

int starpu_sycl_copy_async_sync_devid(void *src_ptr, int src_devid,
				      enum starpu_node_kind src_kind,
				      void *dst_ptr, int dst_devid,
				      enum starpu_node_kind dst_kind,
				      size_t ssize, dpct::queue_ptr stream,
				      dpct::memcpy_direction kind) try
{
	dpct::err0 syclres = 0;

	if (kind == dpct::memcpy_direction::device_to_device && src_devid != dst_devid)
	{
#ifndef STARPU_HAVE_SYCL_MEMCPY_PEER
		STARPU_ABORT();
#endif
	}

	if (stream)
	{
		double start;
		starpu_interface_start_driver_copy_async_devid(src_devid, src_kind, dst_devid, dst_kind, &start);
                // Also handle peer memcpy
		{
			syclres = DPCT_CHECK_ERROR(stream->memcpy((char *)dst_ptr, (char *)src_ptr, ssize));
		}
		starpu_interface_end_driver_copy_async_devid(src_devid, src_kind, dst_devid, dst_kind, start);
	}

	/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
	if (stream == NULL || syclres)
	{
	/* do it in a synchronous fashion */
		{
			// workers might be uninitialized, use queue associated to devid
                        int devid = (dst_kind == STARPU_SYCL_RAM) ? dst_devid : src_devid;
			sycl::queue q = *starpu_sycl_get_local_stream(devid);

			syclres = DPCT_CHECK_ERROR(q.memcpy((char *)dst_ptr, (char *)src_ptr, ssize).wait());
		}
		if (!syclres)
			syclres = DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
		if (STARPU_UNLIKELY(syclres))
			STARPU_SYCL_REPORT_ERROR(syclres);
		return 0;
	}

	return -EAGAIN;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

int starpu_sycl_copy2d_async_sync(void *src_ptr, unsigned src_node, void *dst_ptr, unsigned dst_node,
				  size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
				  dpct::queue_ptr stream, dpct::memcpy_direction kind)
{
	return starpu_sycl_copy2d_async_sync_devid(src_ptr,
						   starpu_memory_node_get_devid(src_node),
						   starpu_node_get_kind(src_node),
						   dst_ptr,
						   starpu_memory_node_get_devid(dst_node),
						   starpu_node_get_kind(dst_node),
						   blocksize, numblocks,
						   ld_src, ld_dst,
						   stream, kind);
}

int starpu_sycl_copy2d_async_sync_devid(void *src_ptr, int src_devid, enum starpu_node_kind src_kind, void *dst_ptr,
					int dst_devid, enum starpu_node_kind dst_kind, size_t blocksize,
					size_t numblocks, size_t ld_src, size_t ld_dst,
					dpct::queue_ptr stream, dpct::memcpy_direction kind) try
{
	dpct::err0 syclres = 0;
	if (kind == dpct::memcpy_direction::device_to_device && src_devid != dst_devid)
	{
		STARPU_ABORT_MSG("SYCL memcpy 3D peer not available, but core triggered one ?!");
	}

	{
		if (stream)
		{
			double start;
			starpu_interface_start_driver_copy_async_devid(src_devid, src_kind, dst_devid, dst_kind, &start);
			auto event = stream->ext_oneapi_memcpy2d(dst_ptr, ld_dst, src_ptr, ld_src, blocksize, numblocks);
			starpu_interface_end_driver_copy_async_devid(src_devid, src_kind, dst_devid, dst_kind, start);
		}

		/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
		if (stream ==  NULL || syclres)
		/* do it in a synchronous fashion */
		{
			// workers might be uninitialized, use queue associated to devid
			int devid = (dst_kind == STARPU_SYCL_RAM) ? dst_devid : src_devid;
			sycl::queue q = *starpu_sycl_get_local_stream(devid);
			syclres = DPCT_CHECK_ERROR(q.ext_oneapi_memcpy2d(dst_ptr, ld_dst, src_ptr, ld_src, blocksize, numblocks).wait());
		}
		if (!syclres)
			syclres = DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
		if (STARPU_UNLIKELY(syclres))
			STARPU_SYCL_REPORT_ERROR(syclres);
		return 0;
	}
	return -EAGAIN;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static inline dpct::event_ptr *_starpu_sycl_event(union _starpu_async_channel_event *_event)
{
        dpct::event_ptr *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (dpct::event_ptr*) _event;
	return event;
}

unsigned _starpu_sycl_test_request_completion(struct _starpu_async_channel *async_channel) try
{
	dpct::event_ptr event;
	sycl::info::event_command_status syclres;
	unsigned success;

	event = *_starpu_sycl_event(&async_channel->event);
	syclres = event->get_info<sycl::info::event::command_execution_status>();
	success = (syclres == sycl::info::event_command_status::complete);

	if (success)
		dpct::destroy_event(event);
	else if (syclres != sycl::info::event_command_status::submitted
		 && syclres != sycl::info::event_command_status::running)
		STARPU_SYCL_REPORT_ERROR(-1);
	return success;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

/* Only used at starpu_shutdown */
void _starpu_sycl_wait_request_completion(struct _starpu_async_channel *async_channel) try
{
	dpct::event_ptr event;
	dpct::err0 syclres;

	event = *_starpu_sycl_event(&async_channel->event);

	syclres = DPCT_CHECK_ERROR(event->wait_and_throw());
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);

	syclres = DPCT_CHECK_ERROR(dpct::destroy_event(event));
	if (STARPU_UNLIKELY(syclres))
		STARPU_SYCL_REPORT_ERROR(syclres);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

sycl::event enqueue_empty_task(dpct::queue_ptr q)
{
	sycl::event e;
	dpct::err0 syclres = DPCT_CHECK_ERROR(
		//e = q->ext_oneapi_submit_barrier()); // introduces unwanted synchronization?
		e = q->single_task([=]() {})); // only works for in_order queue, which is fine
	if (STARPU_UNLIKELY(syclres != 0))
		STARPU_SYCL_REPORT_ERROR(syclres);
	return e;
}

#ifdef STARPU_HAVE_SYCL_MEMCPY_PEER
static void starpu_sycl_set_copy_device(unsigned src_node, unsigned dst_node)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	unsigned devid;
	if ((src_kind == STARPU_SYCL_RAM) && (dst_kind == STARPU_SYCL_RAM))
	{
		/* GPU-GPU transfer, issue it from the source queue */
		devid = starpu_memory_node_get_devid(src_node);
	}
	else
	{
		unsigned node = (dst_kind == STARPU_SYCL_RAM)?dst_node:src_node;
		devid = starpu_memory_node_get_devid(node);
	}
	starpu_sycl_set_device(devid);
}
#endif

int _starpu_sycl_copy_interface_from_sycl_to_sycl( starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req) try
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_SYCL_RAM && dst_kind == STARPU_SYCL_RAM);

#ifdef STARPU_HAVE_SYCL_MEMCPY_PEER
	starpu_sycl_set_copy_device(src_node, dst_node);
#else
	STARPU_ASSERT(src_node == dst_node);
#endif

	int ret = 1;
	dpct::err0 syclres;
	dpct::queue_ptr stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	/* SYCL - SYCL transfer */
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_sycl_copy_disabled() || !copy_methods->any_to_any)
	{
		STARPU_ASSERT(copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_sycl_node_ops;
		syclres = DPCT_CHECK_ERROR(*_starpu_sycl_event(&req->async_channel.event) = new sycl::event());
		if (STARPU_UNLIKELY(syclres != 0)) STARPU_SYCL_REPORT_ERROR(syclres);

		unsigned src_devid = starpu_memory_node_get_devid(src_node);
		unsigned dst_devid = starpu_memory_node_get_devid(dst_node);
		stream = starpu_sycl_get_peer_transfer_stream(src_devid, dst_devid);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		**_starpu_sycl_event(&req->async_channel.event) = enqueue_empty_task(stream);
	}
	return ret;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

int _starpu_sycl_copy_interface_from_sycl_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node,
						 void *dst_interface, unsigned dst_node,
						 struct _starpu_data_request *req) try
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_SYCL_RAM && dst_kind == STARPU_CPU_RAM);

#ifdef STARPU_HAVE_SYCL_MEMCPY_PEER
	starpu_sycl_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	dpct::err0 syclres;
	dpct::queue_ptr stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* only the proper SYCLBLAS thread can initiate this directly ! */
#if !defined(STARPU_HAVE_SYCL_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == src_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_sycl_copy_disabled() || !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_sycl_node_ops;
		syclres = DPCT_CHECK_ERROR(*_starpu_sycl_event(&req->async_channel.event) =
					   new sycl::event());
		if (STARPU_UNLIKELY(syclres != 0)) STARPU_SYCL_REPORT_ERROR(syclres);

		unsigned src_devid = starpu_memory_node_get_devid(src_node);
		stream = starpu_sycl_get_out_transfer_stream(src_devid);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		**_starpu_sycl_event(&req->async_channel.event) = enqueue_empty_task(stream);
	}
	return ret;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

int _starpu_sycl_copy_interface_from_cpu_to_sycl(starpu_data_handle_t handle, void *src_interface, unsigned src_node,
						 void *dst_interface, unsigned dst_node,
						 struct _starpu_data_request *req) try
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_SYCL_RAM);

#ifdef STARPU_HAVE_SYCL_MEMCPY_PEER
	starpu_sycl_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	dpct::err0 syclres;
	dpct::queue_ptr stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* STARPU_CPU_RAM -> SYCLBLAS_RAM */
	/* only the proper SYCLBLAS thread can initiate this ! */
#if !defined(STARPU_HAVE_SYCL_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == dst_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_sycl_copy_disabled() ||
	    !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_sycl_node_ops;
		syclres = DPCT_CHECK_ERROR(*_starpu_sycl_event(&req->async_channel.event) =
					   new sycl::event());
		if (STARPU_UNLIKELY(syclres != 0)) STARPU_SYCL_REPORT_ERROR(syclres);
		unsigned dst_devid = starpu_memory_node_get_devid(dst_node);
		stream = starpu_sycl_get_in_transfer_stream(dst_devid);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		**_starpu_sycl_event(&req->async_channel.event) = enqueue_empty_task(stream);
	}
	return ret;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

int _starpu_sycl_copy_data_from_sycl_to_cpu(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel)
{
	return starpu_sycl_copy_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_SYCL_RAM,
						 (void *)(dst + dst_offset), dst_devid, STARPU_CPU_RAM, size,
						 async_channel ? starpu_sycl_get_out_transfer_stream(src_devid)
						 : NULL,
						 dpct::memcpy_direction::device_to_host);
}

int _starpu_sycl_copy_data_from_sycl_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel)
{
#ifndef STARPU_HAVE_SYCL_MEMCPY_PEER
	STARPU_ASSERT(src_devid == dst_devid);
#endif

	return starpu_sycl_copy_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_SYCL_RAM,
						 (void *)(dst + dst_offset), dst_devid, STARPU_SYCL_RAM, size,
						 async_channel
						 ? starpu_sycl_get_peer_transfer_stream(src_devid, dst_devid)
						 : NULL,
						 dpct::memcpy_direction::device_to_device);
}

int _starpu_sycl_copy_data_from_cpu_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel)
{
	return starpu_sycl_copy_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_CPU_RAM,
						 (void *)(dst + dst_offset), dst_devid, STARPU_SYCL_RAM, size,
						 async_channel ? starpu_sycl_get_in_transfer_stream(dst_devid)
						 : NULL,
						 dpct::memcpy_direction::host_to_device);
}

int _starpu_sycl_copy2d_data_from_sycl_to_cpu(uintptr_t src, size_t src_offset, int src_devid,
					      uintptr_t dst, size_t dst_offset, int dst_devid,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel)
{
	return starpu_sycl_copy2d_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_SYCL_RAM,
						   (void *)(dst + dst_offset), dst_devid, STARPU_CPU_RAM, blocksize,
						   numblocks, ld_src, ld_dst,
						   async_channel ? starpu_sycl_get_out_transfer_stream(src_devid)
						   : NULL,
						   dpct::memcpy_direction::device_to_host);
}

int _starpu_sycl_copy2d_data_from_sycl_to_sycl(uintptr_t src, size_t src_offset, int src_devid,
					       uintptr_t dst, size_t dst_offset, int dst_devid,
					       size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					       struct _starpu_async_channel *async_channel)
{
#ifndef STARPU_HAVE_SYCL_MEMCPY_PEER
	STARPU_ASSERT(src_devid == dst_devid);
#endif

	return starpu_sycl_copy2d_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_SYCL_RAM,
						   (void *)(dst + dst_offset), dst_devid, STARPU_SYCL_RAM, blocksize,
						   numblocks, ld_src, ld_dst,
						   async_channel
						   ? starpu_sycl_get_peer_transfer_stream(src_devid, dst_devid)
						   : NULL,
						   dpct::memcpy_direction::device_to_device);
}

int _starpu_sycl_copy2d_data_from_cpu_to_sycl(uintptr_t src, size_t src_offset, int src_devid,
					      uintptr_t dst, size_t dst_offset, int dst_devid,
					      size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					      struct _starpu_async_channel *async_channel)
{
	return starpu_sycl_copy2d_async_sync_devid((void *)(src + src_offset), src_devid, STARPU_CPU_RAM,
						   (void *)(dst + dst_offset), dst_devid, STARPU_SYCL_RAM, blocksize,
						   numblocks, ld_src, ld_dst,
						   async_channel ? starpu_sycl_get_in_transfer_stream(dst_devid)
						   : NULL,
						   dpct::memcpy_direction::host_to_device);
}

int _starpu_sycl_is_direct_access_supported(unsigned node, unsigned handling_node)
{
#if defined(STARPU_HAVE_SYCL_MEMCPY_PEER)
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return kind == STARPU_SYCL_RAM;
#else /* STARPU_HAVE_SYCL_MEMCPY_PEER */
	/* Direct GPU-GPU transfers are not allowed in general */
	(void) node;
	(void) handling_node;
	return 0;
#endif /* STARPU_HAVE_SYCL_MEMCPY_PEER */
}

void _starpu_sycl_init_device_context(int devid)
{
	streams_bus[devid] = dpct::get_current_device().create_in_order_queue();
	force_queue_init(starpu_sycl_get_local_stream(devid));
}

void _starpu_sycl_device_name(int devid, char *name, size_t size) try
{
	dpct::device_info prop;
	dpct::err0 syclres;
	syclres = DPCT_CHECK_ERROR(dpct::get_device(devid).get_device_info(prop));
	if (STARPU_UNLIKELY(syclres)) STARPU_SYCL_REPORT_ERROR(syclres);
	strncpy(name, prop.get_name(), size);
	name[size-1] = 0;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

size_t _starpu_sycl_total_memory(int devid) try
{
	dpct::device_info prop;
	dpct::err0 syclres;
	syclres = DPCT_CHECK_ERROR(dpct::get_device(devid).get_device_info(prop));
	if (STARPU_UNLIKELY(syclres)) STARPU_SYCL_REPORT_ERROR(syclres);
	return prop.get_global_mem_size();
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

void _starpu_sycl_reset_device(int devid)
{
	starpu_sycl_set_device(devid);
#if DPCT_COMPAT_RT_VERSION >= 4000
	dpct::get_current_device().reset();
#else
	syclThreadExit();
#endif
}

static void start_job_on_sycl(struct _starpu_job *j, struct _starpu_worker *worker)
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

#if defined(STARPU_HAVE_SYCL_MEMCPY_PEER)
	/* We make sure we do manipulate the proper device */
	starpu_sycl_set_device(worker->devid);
#endif

	starpu_sycl_func_t func = _starpu_task_get_sycl_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_SYCL is defined in 'where', sycl_func or sycl_funcs has to be defined");

	void *func_ptr = reinterpret_cast<void *>(func);
	if (_starpu_get_disable_kernels() <= 0)
	{
		_starpu_trace_start_executing(j, task, worker, func_ptr);
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		_starpu_trace_end_executing(j, worker);
	}
}

static void finish_job_on_sycl(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static void execute_job_on_sycl(struct starpu_task *task, struct _starpu_worker *worker) try
{
	int workerid = worker->workerid;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	start_job_on_sycl(j, worker);

	if (!used_stream[workerid])
	{
		used_stream[workerid] = 1;
		_STARPU_DISP("Warning: starpu_sycl_get_local_stream() was not used to submit kernel to SYCL on worker %d. SYCL will thus introduce a lot of useless synchronizations, which will prevent proper overlapping of data transfers and kernel execution. See the SYCL-specific part of the 'Check List When Performance Are Not There' of the StarPU handbook\n", workerid);
	}

	if (task->cl->sycl_flags[j->nimpl] & STARPU_SYCL_ASYNC)
	{
		/* Record event to synchronize with task termination later */
		*task_events[workerid] = enqueue_empty_task(starpu_sycl_get_local_stream());
	}
	else
	/* Synchronous execution */
	{
#if !defined(STARPU_SIMGRID)
		STARPU_ASSERT_MSG(DPCT_CHECK_ERROR((starpu_sycl_get_local_stream()->ext_oneapi_empty())) == 0,
				  "Unless when using the STARPU_SYCL_ASYNC flag, SYCL "
				  "codelets have to wait for termination of their kernels on "
				  "the starpu_sycl_get_local_stream() stream");
#endif
		finish_job_on_sycl(j, worker);
	}
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void finish_job_on_sycl(struct _starpu_job *j, struct _starpu_worker *worker)
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
static int _starpu_sycl_driver_run_once(struct _starpu_worker *worker) try
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

	do /* This do {} while (0) is only to match the sycl driver worker for look */
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
			_starpu_trace_end_progress(memnode, worker);
			j = _starpu_get_job_associated_to_task(task);

			_starpu_fetch_task_input_tail(task, j, worker);
			/* Reset it */
			worker->task_transferring = NULL;

			execute_job_on_sycl(task, worker);
			_starpu_trace_start_progress(memnode, worker);
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
		dpct::err0 syclres = dpct::sycl_event_query(task_events[workerid]);

		if (syclres != 0)
		{
			STARPU_ASSERT_MSG(syclres == 1,
					  "SYCL error on task %p, codelet %p (%s): %s (%d)",
					  task, task->cl,
					  _starpu_codelet_get_model_name(task->cl),
					  "No codelet error", syclres);
		}
		else
		{
			_starpu_trace_end_progress(memnode, worker);
			/* Asynchronous task completed! */
			finish_job_on_sycl(_starpu_get_job_associated_to_task(task), worker);
			_starpu_trace_start_progress(memnode, worker);
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

	/* can SYCL do that task ? */
	if (!_STARPU_MAY_PERFORM(j, SYCL))
	{
		/* this is neither a sycl or a syclblas task */
		_starpu_worker_refuse_task(worker, task);
		return 0;
	}

	worker->current_task = task;

	/* Fetch data asynchronously */
	_starpu_trace_end_progress(memnode, worker);
	_starpu_set_local_worker_key(worker);
	res = _starpu_fetch_task_input(task, j, 1);
	STARPU_ASSERT(res == 0);
	_starpu_trace_start_progress(memnode, worker);

	return 0;
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

void *_starpu_sycl_worker(void *_arg)
{
	struct _starpu_worker *worker = (_starpu_worker*)_arg;

	_starpu_sycl_driver_init(worker);
	_starpu_trace_start_progress(worker->memory_node, worker);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_sycl_driver_run_once(worker);
	}
	_starpu_trace_end_progress(worker->memory_node, worker);
	_starpu_sycl_driver_deinit(worker);

	return NULL;
}

#ifdef STARPU_HAVE_HWLOC
hwloc_obj_t _starpu_sycl_get_hwloc_obj(hwloc_topology_t topology, int devid)
{
	return NULL;
}
#endif

void starpu_syclblas_report_error(const char *func, const char *file, int line, int status)
{
	const char *errormsg = "No status error";
	_STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}

void starpu_sycl_report_error(const char *func, const char *file, int line, dpct::err0 status)
{
	const char *errormsg = "No status error";
	_STARPU_ERROR("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

static int _starpu_sycl_run_from_worker(struct _starpu_worker *worker)
{
	/* Let's go ! */
	_starpu_sycl_worker(worker);

	return 0;
}

struct _starpu_driver_ops _starpu_driver_sycl_ops =
{
	.init = _starpu_sycl_driver_init,
	.run = _starpu_sycl_run_from_worker,
	.run_once = _starpu_sycl_driver_run_once,
	.deinit = _starpu_sycl_driver_deinit,
};

struct _starpu_node_ops _starpu_driver_sycl_node_ops =
{
	.name = "sycl driver",
	.malloc_on_device = _starpu_sycl_malloc_on_device,
	.malloc_on_host = _starpu_sycl_malloc_on_host,
	.memset_on_device = _starpu_sycl_memset_on_device,
	.free_on_device = _starpu_sycl_free_on_device,
	.free_on_host = _starpu_sycl_free_on_host,

	.is_direct_access_supported = _starpu_sycl_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_sycl_copy_interface_from_sycl_to_cpu,
	.copy_interface_to[STARPU_SYCL_RAM] = _starpu_sycl_copy_interface_from_sycl_to_sycl,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_sycl_copy_interface_from_cpu_to_sycl,
	.copy_interface_from[STARPU_SYCL_RAM] = _starpu_sycl_copy_interface_from_sycl_to_sycl,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_sycl_copy_data_from_sycl_to_cpu,
	.copy_data_to[STARPU_SYCL_RAM] = _starpu_sycl_copy_data_from_sycl_to_sycl,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_sycl_copy_data_from_cpu_to_sycl,
	.copy_data_from[STARPU_SYCL_RAM] = _starpu_sycl_copy_data_from_sycl_to_sycl,

	.copy2d_data_to[STARPU_CPU_RAM] = _starpu_sycl_copy2d_data_from_sycl_to_cpu,
	.copy2d_data_to[STARPU_SYCL_RAM] = _starpu_sycl_copy2d_data_from_sycl_to_sycl,

	.copy2d_data_from[STARPU_CPU_RAM] = _starpu_sycl_copy2d_data_from_cpu_to_sycl,
	.copy2d_data_from[STARPU_SYCL_RAM] = _starpu_sycl_copy2d_data_from_sycl_to_sycl,

	.wait_request_completion = _starpu_sycl_wait_request_completion,
	.test_request_completion = _starpu_sycl_test_request_completion,

	.calibrate_bus = 1,
	.device_name = _starpu_sycl_device_name,
	.total_memory = _starpu_sycl_total_memory,
	.max_memory = _starpu_sycl_total_memory,
	.set_device = starpu_sycl_set_device,
	.init_device = _starpu_sycl_init_device_context,
	.reset_device = _starpu_sycl_reset_device,
};
