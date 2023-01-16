/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010	    Mehdi Juhoor
 * Copyright (C) 2011	    Télécom-SudParis
 * Copyright (C) 2013	    Thibaut Lambert
 * Copyright (C) 2016	    Uppsala University
 * Copyright (C) 2021	    Federal University of Rio Grande do Sul (UFRGS)
 * Copyright (C) 2022	    Camille Coti
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

#include "starpu_config.h"
#include <starpu.h>
#include <starpu_hip.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <core/devices.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/driver_common/driver_common.h>
#include <drivers/hip/driver_hip.h>
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <common/knobs.h>
#include <profiling/callbacks.h>

#if HAVE_DECL_HWLOC_HIP_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/hip/hip_runtime.h>
#endif

#define starpu_hipStreamCreate(stream) hipStreamCreateWithFlags(stream, hipStreamNonBlocking)

/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

/* the number of HIP devices */
static int nhipgpus = -1;

static size_t global_mem[STARPU_MAXHIPDEVS];
int _starpu_hip_bus_ids[STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES][STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES];
static hipStream_t streams[STARPU_NMAXWORKERS];
static char used_stream[STARPU_NMAXWORKERS];
static hipStream_t out_transfer_streams[STARPU_MAXHIPDEVS];
static hipStream_t in_transfer_streams[STARPU_MAXHIPDEVS];
/* Note: streams are not thread-safe, so we define them for each HIP worker
 * emitting a GPU-GPU transfer */
static hipStream_t in_peer_transfer_streams[STARPU_MAXHIPDEVS][STARPU_MAXHIPDEVS];
static struct hipDeviceProp_t props[STARPU_MAXHIPDEVS];
static hipEvent_t task_events[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];

static unsigned hip_bindid_init[STARPU_MAXHIPDEVS];
static unsigned hip_bindid[STARPU_MAXHIPDEVS];
static unsigned hip_memory_init[STARPU_MAXHIPDEVS];
static unsigned hip_memory_nodes[STARPU_MAXHIPDEVS];

static struct _starpu_worker_set hip_worker_set[STARPU_MAXHIPDEVS];
static enum initialization hip_device_init[STARPU_MAXHIPDEVS];
static int hip_device_users[STARPU_MAXHIPDEVS];
static starpu_pthread_mutex_t hip_device_init_mutex[STARPU_MAXHIPDEVS];
static starpu_pthread_cond_t hip_device_init_cond[STARPU_MAXHIPDEVS];
static int hip_globalbindid;


int _starpu_nworker_per_hip;

static size_t _starpu_hip_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}

static hipStream_t starpu_hip_get_in_transfer_stream(unsigned dst_node)
{
	int dst_devid = starpu_memory_node_get_devid(dst_node);
	hipStream_t stream;

	stream = in_transfer_streams[dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static hipStream_t starpu_hip_get_out_transfer_stream(unsigned src_node)
{
	int src_devid = starpu_memory_node_get_devid(src_node);
	hipStream_t stream;

	stream = out_transfer_streams[src_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static hipStream_t starpu_hip_get_peer_transfer_stream(unsigned src_node, unsigned dst_node)
{
	int src_devid = starpu_memory_node_get_devid(src_node);
	int dst_devid = starpu_memory_node_get_devid(dst_node);
	hipStream_t stream;

	stream = in_peer_transfer_streams[src_devid][dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

hipStream_t starpu_hip_get_local_stream(void)
{
	int worker = starpu_worker_get_id_check();

	used_stream[worker] = 1;
	return streams[worker];
}

const struct hipDeviceProp_t *starpu_hip_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}

/* Early library initialization, before anything else, just initialize data */
void _starpu_hip_init(void)
{
	int i;
	for (i = 0; i < STARPU_MAXHIPDEVS; i++)
	{
		STARPU_PTHREAD_MUTEX_INIT(&hip_device_init_mutex[i], NULL);
		STARPU_PTHREAD_COND_INIT(&hip_device_init_cond[i], NULL);
	}
	memset(&hip_bindid_init, 0, sizeof(hip_bindid_init));
	memset(&hip_memory_init, 0, sizeof(hip_memory_init));
	hip_globalbindid = -1;
}

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXHIPDEVS */
unsigned _starpu_get_hip_device_count(void)
{
	int cnt;
	hipError_t hipres;
	hipres = hipGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(hipres))
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
void _starpu_hip_discover_devices(struct _starpu_machine_config *config)
{
	/* Discover the number of HIP devices. Fill the result in CONFIG. */

	int cnt;
	hipError_t hipres;

	hipres = hipGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(hipres != hipSuccess))
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

	_starpu_devices_gpu_clear(config, STARPU_HIP_WORKER);
	_starpu_devices_drop_duplicate(topology->workers_devid[STARPU_HIP_WORKER]);
}

/* Determine which devices we will use */
void _starpu_init_hip_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config)
{
	int i;

	for (i = 0; i < (int) (sizeof(hip_worker_set)/sizeof(hip_worker_set[0])); i++)
		hip_worker_set[i].workers = NULL;

	int nhip = config->conf.nhip;

	if (nhip != 0)
	{
		/* The user did not disable HIP. We need to initialize HIP
		 * early to count the number of devices */
		_starpu_init_hip();
		int nb_devices = _starpu_get_hip_device_count();

		_starpu_topology_check_ndevices(&nhip, nb_devices, 0, STARPU_MAXHIPDEVS, "nhip", "HIP", "maxhipdev");
	}

	int nworker_per_hip = starpu_get_env_number_default("STARPU_NWORKER_PER_HIP", 1);

	STARPU_ASSERT_MSG(nworker_per_hip > 0, "STARPU_NWORKER_PER_HIP has to be > 0");
	STARPU_ASSERT_MSG(nworker_per_hip < STARPU_NMAXWORKERS, "STARPU_NWORKER_PER_HIP (%d) cannot be higher than STARPU_NMAXWORKERS (%d)\n", nworker_per_hip, STARPU_NMAXWORKERS);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	if (nworker_per_hip > 1)
	{
		_STARPU_DISP("Warning: reducing STARPU_NWORKER_PER_HIP to 1 because blocking drivers are enabled\n");
		nworker_per_hip = 1;
	}
	_starpu_nworker_per_hip = nworker_per_hip;
#endif

	/* Now we know how many HIP devices will be used */
	topology->ndevices[STARPU_HIP_WORKER] = nhip;

	_starpu_initialize_workers_hip_gpuid(config);

	/* allow having one worker per stream */
	topology->hip_th_per_stream = starpu_get_env_number_default("STARPU_HIP_THREAD_PER_WORKER", -1);
	topology->hip_th_per_dev = starpu_get_env_number_default("STARPU_HIP_THREAD_PER_DEV", -1);

	STARPU_ASSERT_MSG(!(topology->hip_th_per_stream == 1 && topology->hip_th_per_dev != -1), "It does not make sense to set both STARPU_HIP_THREAD_PER_WORKER to 1 and to set STARPU_HIP_THREAD_PER_DEV, please choose either per worker or per device or none");

	/* per device by default */
	if (topology->hip_th_per_dev == -1)
	{
		if (topology->hip_th_per_stream == 1)
			topology->hip_th_per_dev = 0;
		else
			topology->hip_th_per_dev = 1;
	}
	/* Not per stream by default */
	if (topology->hip_th_per_stream == -1)
	{
		topology->hip_th_per_stream = 0;
	}

	if (!topology->hip_th_per_dev)
	{
		hip_worker_set[0].workers = &config->workers[topology->nworkers];
		hip_worker_set[0].nworkers = nhip * nworker_per_hip;
	}

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

		struct _starpu_worker_set *worker_set;

		if(topology->hip_th_per_stream)
		{
			worker_set = ALLOC_WORKER_SET;
		}
		else if (topology->hip_th_per_dev)
		{
			worker_set = &hip_worker_set[devid];
			worker_set->workers = &config->workers[topology->nworkers];
			worker_set->nworkers = nworker_per_hip;
		}
		else
		{
			/* Same worker set for all devices */
			worker_set = &hip_worker_set[0];
		}

		_starpu_topology_configure_workers(topology, config,
						   STARPU_HIP_WORKER,
						   hipgpu, devid, 0, 0,
						   nworker_per_hip,
						   // TODO: fix perfmodels etc.
						   // nworker_per_hip - 1,
						   1,
						   worker_set, NULL);

		_starpu_devices_gpu_set_used(devid);

/* TODO: move this to generic place */
#ifdef STARPU_HAVE_HWLOC
		{
			hwloc_obj_t obj = NULL;
			if (starpu_driver_info[STARPU_HIP_WORKER].get_hwloc_obj)
				obj = starpu_driver_info[STARPU_HIP_WORKER].get_hwloc_obj(topology, devid);

			if (obj)
			{
				struct _starpu_hwloc_userdata *data = obj->userdata;
				data->ngpus++;
			}
			else
			{
				_STARPU_DEBUG("Warning: could not find location of HIP%u, do you have the hwloc HIP plugin installed?\n", devid);
			}
		}
#endif
        }
}

/* Bind the driver on a CPU core */
void _starpu_hip_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

	if (hip_bindid_init[devid])
	{
		if (config->topology.hip_th_per_stream == 0)
			workerarg->bindid = hip_bindid[devid];
		else
			workerarg->bindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
	}
	else
	{
		hip_bindid_init[devid] = 1;

		if (config->topology.hip_th_per_dev == 0 && config->topology.hip_th_per_stream == 0)
		{
			if (hip_globalbindid == -1)
				hip_globalbindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
			workerarg->bindid = hip_bindid[devid] = hip_globalbindid;
		}
		else
		{
			workerarg->bindid = hip_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
		}
	}
}

/* Set up memory and buses */
void _starpu_hip_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (hip_memory_init[devid])
	{
		memory_node = hip_memory_nodes[devid];
	}
	else
	{
		hip_memory_init[devid] = 1;

		memory_node = hip_memory_nodes[devid] = _starpu_memory_node_register(STARPU_HIP_RAM, devid);

#ifdef STARPU_USE_HIP_MAP
		/* TODO: check node capabilities */
		_starpu_memory_node_set_mapped(memory_node);
#endif

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_hip_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
			_starpu_hip_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
		}

		if (
#if defined(STARPU_HAVE_HIP_MEMCPY_PEER)
			1
#else /* MEMCPY_PEER */
			0
#endif /* MEMCPY_PEER */
		   )
		{
			int worker2;
			for (worker2 = 0; worker2 < workerarg->workerid; worker2++)
			{
				struct _starpu_worker *workerarg2 = &config->workers[worker2];
				int devid2 = workerarg2->devid;
				if (workerarg2->arch == STARPU_HIP_WORKER)
				{
					unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
					_starpu_hip_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node2, memory_node);
					_starpu_hip_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES] = _starpu_register_bus(memory_node, memory_node2);
#if HAVE_DECL_HWLOC_HIP_GET_DEVICE_OSDEV_BY_INDEX
					{
						hwloc_obj_t obj, obj2, ancestor;
						obj = hwloc_hip_get_device_osdev_by_index(config->topology.hwtopology, devid);
						obj2 = hwloc_hip_get_device_osdev_by_index(config->topology.hwtopology, devid2);
						ancestor = hwloc_get_common_ancestor_obj(config->topology.hwtopology, obj, obj2);
						if (ancestor)
						{
							struct _starpu_hwloc_userdata *data = ancestor->userdata;
#ifdef STARPU_VERBOSE
							{
								char name[64];
								hwloc_obj_type_snprintf(name, sizeof(name), ancestor, 0);
								_STARPU_DEBUG("HIP%u and HIP%u are linked through %s, along %u GPUs\n", devid, devid2, name, data->ngpus);
							}
#endif
							starpu_bus_set_ngpus(_starpu_hip_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES], data->ngpus);
							starpu_bus_set_ngpus(_starpu_hip_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES], data->ngpus);
						}
					}
#endif
				}
			}
		}
	}
	_starpu_memory_node_add_nworkers(memory_node);

	//This worker can also manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(&workerarg->set->workers[0], numa);

	_starpu_worker_drives_memory_node(&workerarg->set->workers[0], memory_node);

	workerarg->memory_node = memory_node;
}

/* Set the current HIP device */
void starpu_hip_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	hipError_t hipres;
	int attempts = 0;

	hipres = hipSetDevice(devid);
	while (hipres == hipErrorDeinitialized && ++attempts < 10)
	{
		usleep(100000);
		hipres = hipSetDevice(devid);
	}

	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);
}

static void _starpu_hip_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

	/* Find the size of the memory on the device */
	totalGlobalMem = props[devid].totalGlobalMem;

	limit = starpu_getenv_number("STARPU_LIMIT_HIP_MEM");
	if (limit == -1)
	{
		char name[30];
		snprintf(name, sizeof(name), "STARPU_LIMIT_HIP_%u_MEM", devid);
		limit = starpu_getenv_number(name);
	}
#if defined(STARPU_USE_HIP)
	if (limit == -1)
	{
		limit = totalGlobalMem / (1024*1024) * FREE_MARGIN;
	}
#endif

	global_mem[devid] = limit * 1024*1024;
}

/* Really initialize one device */
static void init_device_context(unsigned devid, unsigned memnode)
{
	hipError_t hipres;
	int attempts = 0;

	starpu_hip_set_device(devid);

	STARPU_PTHREAD_MUTEX_LOCK(&hip_device_init_mutex[devid]);
	hip_device_users[devid]++;
	if (hip_device_init[devid] == UNINITIALIZED)
		/* Nobody started initialization yet, do it */
		hip_device_init[devid] = CHANGING;
	else
	{
		/* Somebody else is doing initialization, wait for it */
		while (hip_device_init[devid] != INITIALIZED)
			STARPU_PTHREAD_COND_WAIT(&hip_device_init_cond[devid], &hip_device_init_mutex[devid]);
		STARPU_PTHREAD_MUTEX_UNLOCK(&hip_device_init_mutex[devid]);
		return;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&hip_device_init_mutex[devid]);

	/* force HIP to initialize the context for real */
	hipres = hipInit(0);
	while (hipres == hipErrorDeinitialized && ++attempts < 100)
	{
		usleep(100000);
		hipres = hipInit(0);
	}

	if (STARPU_UNLIKELY(hipres))
	{
		if (hipres != hipSuccess)
		{
			_STARPU_MSG("Failed to initialize HIP runtime\n");
			exit(77);
		}
		STARPU_HIP_REPORT_ERROR(hipres);
	}

	hipres = hipGetDeviceProperties(&props[devid], devid);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
	if (props[devid].computeMode == hipComputeModeExclusive)
	{
		_STARPU_MSG("HIP is in EXCLUSIVE-THREAD mode, but StarPU was built with multithread GPU control support, please either ask your administrator to use EXCLUSIVE-PROCESS mode (which should really be fine), or reconfigure with --disable-hip-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#endif

	hipres = starpu_hipStreamCreate(&in_transfer_streams[devid]);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);

	hipres = starpu_hipStreamCreate(&out_transfer_streams[devid]);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);

	int i;
	for (i = 0; i < nhipgpus; i++)
	{
		hipres = starpu_hipStreamCreate(&in_peer_transfer_streams[i][devid]);
		if (STARPU_UNLIKELY(hipres))
			STARPU_HIP_REPORT_ERROR(hipres);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&hip_device_init_mutex[devid]);
	hip_device_init[devid] = INITIALIZED;
	STARPU_PTHREAD_COND_BROADCAST(&hip_device_init_cond[devid]);
	STARPU_PTHREAD_MUTEX_UNLOCK(&hip_device_init_mutex[devid]);

	_starpu_hip_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_hip_get_global_mem_size(devid));
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	int i;
	starpu_hip_set_device(devid);

	hipStreamDestroy(in_transfer_streams[devid]);
	hipStreamDestroy(out_transfer_streams[devid]);

	for (i = 0; i < nhipgpus; i++)
	{
		hipStreamDestroy(in_peer_transfer_streams[i][devid]);
	}
}

static void init_worker_context(unsigned workerid, unsigned devid)
{
	int j;
	hipError_t hipres;
	starpu_hip_set_device(devid);

	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
	{
		hipres = hipEventCreateWithFlags(&task_events[workerid][j], hipEventDisableTiming);
		if (STARPU_UNLIKELY(hipres))
			STARPU_HIP_REPORT_ERROR(hipres);
	}

	hipres = starpu_hipStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);
}

static void deinit_worker_context(unsigned workerid, unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	unsigned j;
	starpu_hip_set_device(devid);
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		hipEventDestroy(task_events[workerid][j]);
	hipStreamDestroy(streams[workerid]);
}

/* This is run from the driver thread to initialize the driver HIP context */
int _starpu_hip_driver_init(struct _starpu_worker *worker)
{
	struct _starpu_worker_set *worker_set = worker->set;
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	int lastdevid = -1;
	unsigned i;

	_starpu_driver_start(worker0, STARPU_HIP_WORKER, 0);
	_starpu_set_local_worker_key(worker);

#ifdef STARPU_PROF_TOOL
		struct starpu_prof_tool_info pi;
#endif

#ifdef STARPU_USE_FXT
	for (i = 1; i < worker_set->nworkers; i++)
		_starpu_worker_start(&worker_set->workers[i], STARPU_HIP_WORKER, 0);
#endif

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned memnode = worker->memory_node;

#ifdef STARPU_PROF_TOOL
		struct starpu_prof_tool_info pi;
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init, devid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init(&pi, NULL, NULL);
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_start, devid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start(&pi, NULL, NULL);
#endif

		if ((int) devid == lastdevid)
		{
			/* Already initialized */
			continue;
		}
		lastdevid = devid;
		init_device_context(devid, memnode);

		if (worker->config->topology.nworker[STARPU_HIP_WORKER][devid] > 1 && props[devid].concurrentKernels == 0)
			_STARPU_DISP("Warning: STARPU_NWORKER_PER_HIP is %u, but HIP device %u does not support concurrent kernel execution!\n", worker_set->nworkers, devid);
	}

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(worker0->bindid, worker0->workerid, NULL);

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned workerid = worker->workerid;

		float size = (float) global_mem[devid] / (1<<30);
		/* get the device's name */
		char devname[64];
		strncpy(devname, props[devid].name, 63);
		devname[63] = 0;

		snprintf(worker->name, sizeof(worker->name), "HIP %u (%s %.1f GiB)", devid, devname, size);
		snprintf(worker->short_name, sizeof(worker->short_name), "HIP %u", devid);
		_STARPU_DEBUG("hip (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

		worker->pipeline_length = starpu_get_env_number_default("STARPU_HIP_PIPELINE", 2);
		if (worker->pipeline_length > STARPU_MAX_PIPELINE)
		{
			_STARPU_DISP("Warning: STARPU_HIP_PIPELINE is %u, but STARPU_MAX_PIPELINE is only %u\n", worker->pipeline_length, STARPU_MAX_PIPELINE);
			worker->pipeline_length = STARPU_MAX_PIPELINE;
		}	/* tell the main thread that this one is ready */
#if !defined(STARPU_NON_BLOCKING_DRIVERS)
		if (worker->pipeline_length >= 1)
		{
			/* We need non-blocking drivers, to poll for HIP task
			 * termination */
			_STARPU_DISP("Warning: reducing STARPU_HIP_PIPELINE to 0 because blocking drivers are enabled (and simgrid is not supported with this driver)\n");
			worker->pipeline_length = 0;
		}
#endif
		init_worker_context(workerid, worker->devid);

		_STARPU_TRACE_WORKER_INIT_END(workerid);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_end, devid, worker->workerid, starpu_prof_tool_driver_gpu, 0, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_end(&pi, NULL, NULL);
#endif
	}
	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "HIP %u", worker->devid);
		starpu_pthread_setname(thread_name);
	}


	STARPU_PTHREAD_MUTEX_LOCK(&worker0->mutex);
	worker0->status = STATUS_UNKNOWN;
	worker0->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);

	/* tell the main thread that this one is also ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
	worker_set->set_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

	return 0;
}

int _starpu_hip_driver_deinit(struct _starpu_worker *worker)
{
	struct _starpu_worker_set *worker_set = worker->set;
	int lastdevid = -1;
	unsigned i;
	_STARPU_TRACE_WORKER_DEINIT_START;

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned memnode = worker->memory_node;
		unsigned usersleft;
		if ((int) devid == lastdevid)
			/* Already initialized */
			continue;
		lastdevid = devid;

		STARPU_PTHREAD_MUTEX_LOCK(&hip_device_init_mutex[devid]);
		usersleft = --hip_device_users[devid];
		STARPU_PTHREAD_MUTEX_UNLOCK(&hip_device_init_mutex[devid]);

		if (!usersleft)
		{
			/* I'm last, deinitialize device */
			_starpu_datawizard_handle_all_pending_node_data_requests(memnode);

			/* In case there remains some memory that was automatically
			 * allocated by StarPU, we release it now. Note that data
			 * coherency is not maintained anymore at that point ! */
			_starpu_free_all_automatically_allocated_buffers(memnode);

			_starpu_malloc_shutdown(memnode);

			deinit_device_context(devid);
		}
		STARPU_PTHREAD_MUTEX_LOCK(&hip_device_init_mutex[devid]);
		hip_device_init[devid] = UNINITIALIZED;
		STARPU_PTHREAD_MUTEX_UNLOCK(&hip_device_init_mutex[devid]);

	}

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned workerid = worker->workerid;
		unsigned memnode = worker->memory_node;

		deinit_worker_context(workerid, worker->devid);

#ifdef STARPU_PROF_TOOL
		struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_deinit, workerid, worker->workerid, starpu_prof_tool_driver_hip, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit(&pi, NULL, NULL);
#endif
	}

	worker_set->workers[0].worker_is_initialized = 0;
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
	(void) size;
	(void) flags;

	hipError_t err;
	unsigned devid = starpu_memory_node_get_devid(dst_node);
	starpu_hip_set_device(devid);
	err = hipFree((void*)addr);
	if (STARPU_UNLIKELY(err != hipSuccess))
		STARPU_HIP_REPORT_ERROR(err);
}

int starpu_hip_copy_async_sync(void *src_ptr, unsigned src_node,
			       void *dst_ptr, unsigned dst_node,
			       size_t ssize, hipStream_t stream,
			       hipMemcpyKind kind)
{
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
	int peer_copy = 0;
	int src_dev = -1, dst_dev = -1;
#endif
	hipError_t hipres = 0;

	if (kind == hipMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
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
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
		if (peer_copy)
		{
			hipres = hipMemcpyPeerAsync((char *) dst_ptr, dst_dev,
						   (char *) src_ptr, src_dev,
						   ssize, stream);
		}
		else
#endif
		{
			hipres = hipMemcpyAsync((char *)dst_ptr, (char *)src_ptr, ssize, kind, stream);
		}
		(void) hipGetLastError();
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	}

	/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
	if (stream == NULL || hipres)
	{
	/* do it in a synchronous fashion */
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
		if (peer_copy)
		{
			hipres = hipMemcpyPeer((char *) dst_ptr, dst_dev,
					      (char *) src_ptr, src_dev,
					      ssize);
		}
		else
#endif
		{
			hipres = hipMemcpy((char *)dst_ptr, (char *)src_ptr, ssize, kind);
		}
		(void) hipGetLastError();

		if (!hipres)
			hipres = hipDeviceSynchronize();
		if (STARPU_UNLIKELY(hipres))
			STARPU_HIP_REPORT_ERROR(hipres);

		return 0;
	}

	return -EAGAIN;
}

/* Driver porters: this is optional but really recommended */
int starpu_hip_copy2d_async_sync(void *src_ptr, unsigned src_node,
				 void *dst_ptr, unsigned dst_node,
				 size_t blocksize,
				 size_t numblocks, size_t ld_src, size_t ld_dst,
				 hipStream_t stream, hipMemcpyKind kind)
{
	hipError_t hipres = 0;

	if (kind == hipMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
#  ifdef BUGGED_MEMCPY3D
		STARPU_ABORT_MSG("HIP memcpy 3D peer buggy, but core triggered one?!");
#  endif
#else
		STARPU_ABORT_MSG("HIP memcpy 3D peer not available, but core triggered one ?!");
#endif
	}

	if (stream)
	{
		double start;
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
		hipres = hipMemcpy2DAsync((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
					 blocksize, numblocks, kind, stream);
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	}

	/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
	if (stream == NULL || hipres)
	{
		hipres = hipMemcpy2D((char *)dst_ptr, ld_dst, (char *)src_ptr, ld_src,
				    blocksize, numblocks, kind);
		if (!hipres)
			hipres = hipDeviceSynchronize();
		if (STARPU_UNLIKELY(hipres))
			STARPU_HIP_REPORT_ERROR(hipres);

		return 0;
	}

	return -EAGAIN;
}

static inline hipEvent_t *_starpu_hip_event(union _starpu_async_channel_event *_event)
{
	hipEvent_t *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (void *) _event;
	return event;
}

unsigned _starpu_hip_test_request_completion(struct _starpu_async_channel *async_channel)
{
	hipEvent_t event;
	hipError_t hipres;
	unsigned success;

	event = *_starpu_hip_event(&async_channel->event);
	hipres = hipEventQuery(event);
	success = (hipres == hipSuccess);

	if (success)
		hipEventDestroy(event);
	else if (hipres != hipErrorNotReady)
		STARPU_HIP_REPORT_ERROR(hipres);

	return success;
}

void _starpu_hip_wait_request_completion(struct _starpu_async_channel *async_channel)
{
	hipEvent_t event;
	hipError_t hipres;

	event = *_starpu_hip_event(&async_channel->event);

	hipres = hipEventSynchronize(event);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);

	hipres = hipEventDestroy(event);
	if (STARPU_UNLIKELY(hipres))
		STARPU_HIP_REPORT_ERROR(hipres);
}

#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
static void starpu_hip_set_copy_device(unsigned src_node, unsigned dst_node)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	unsigned devid;
	if ((src_kind == STARPU_HIP_RAM) && (dst_kind == STARPU_HIP_RAM))
	{
		/* GPU-GPU transfer, issue it from the destination */
		devid = starpu_memory_node_get_devid(dst_node);
	}
	else
	{
		unsigned node = (dst_kind == STARPU_HIP_RAM)?dst_node:src_node;
		devid = starpu_memory_node_get_devid(node);
	}
	starpu_hip_set_device(devid);
}
#endif

int _starpu_hip_copy_interface_from_hip_to_hip(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_HIP_RAM);

#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
	starpu_hip_set_copy_device(src_node, dst_node);
#else
	STARPU_ASSERT(src_node == dst_node);
#endif

	int ret = 1;
	hipError_t hipres;
	hipStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	/* HIP - HIP transfer */
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_hip_copy_disabled() || !copy_methods->any_to_any)
	{
		STARPU_ASSERT(copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_hip_node_ops;
		hipres = hipEventCreateWithFlags(_starpu_hip_event(&req->async_channel.event), hipEventDisableTiming);
		if (STARPU_UNLIKELY(hipres != hipSuccess)) STARPU_HIP_REPORT_ERROR(hipres);

		stream = starpu_hip_get_peer_transfer_stream(src_node, dst_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		hipres = hipEventRecord(*_starpu_hip_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(hipres != hipSuccess)) STARPU_HIP_REPORT_ERROR(hipres);
	}
	return ret;
}

int _starpu_hip_copy_interface_from_hip_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_CPU_RAM);

#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
	starpu_hip_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	hipError_t hipres;
	hipStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* only the proper CUBLAS thread can initiate this directly ! */
#if !defined(STARPU_HAVE_HIP_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == src_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_hip_copy_disabled() || !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_hip_node_ops;
		hipres = hipEventCreateWithFlags(_starpu_hip_event(&req->async_channel.event), hipEventDisableTiming);
		if (STARPU_UNLIKELY(hipres != hipSuccess)) STARPU_HIP_REPORT_ERROR(hipres);

		stream = starpu_hip_get_out_transfer_stream(src_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		hipres = hipEventRecord(*_starpu_hip_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(hipres != hipSuccess)) STARPU_HIP_REPORT_ERROR(hipres);
	}
	return ret;
}

int _starpu_hip_copy_interface_from_cpu_to_hip(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_HIP_RAM);

#ifdef STARPU_HAVE_HIP_MEMCPY_PEER
	starpu_hip_set_copy_device(src_node, dst_node);
#endif

	int ret = 1;
	hipError_t hipres;
	hipStream_t stream;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	/* STARPU_CPU_RAM -> CUBLAS_RAM */
	/* only the proper CUBLAS thread can initiate this ! */
#if !defined(STARPU_HAVE_HIP_MEMCPY_PEER)
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == dst_node);
#endif
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_hip_copy_disabled() ||
	    !copy_methods->any_to_any)
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->any_to_any);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_hip_node_ops;
		hipres = hipEventCreateWithFlags(_starpu_hip_event(&req->async_channel.event), hipEventDisableTiming);
		if (STARPU_UNLIKELY(hipres != hipSuccess))
			STARPU_HIP_REPORT_ERROR(hipres);

		stream = starpu_hip_get_in_transfer_stream(dst_node);
		STARPU_ASSERT(copy_methods->any_to_any);
		ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);

		hipres = hipEventRecord(*_starpu_hip_event(&req->async_channel.event), stream);
		if (STARPU_UNLIKELY(hipres != hipSuccess))
			STARPU_HIP_REPORT_ERROR(hipres);
	}
	return ret;
}

int _starpu_hip_copy_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_hip_copy_async_sync((void*) (src + src_offset), src_node,
					  (void*) (dst + dst_offset), dst_node,
					  size,
					  async_channel?starpu_hip_get_out_transfer_stream(src_node):NULL,
					  hipMemcpyDeviceToHost);
}

int _starpu_hip_copy_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
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
					  async_channel?starpu_hip_get_peer_transfer_stream(src_node, dst_node):NULL,
					  hipMemcpyDeviceToDevice);
}

int _starpu_hip_copy_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_HIP_RAM);

	return starpu_hip_copy_async_sync((void*) (src + src_offset), src_node,
					  (void*) (dst + dst_offset), dst_node,
					  size,
					  async_channel?starpu_hip_get_in_transfer_stream(dst_node):NULL,
					  hipMemcpyHostToDevice);
}

int _starpu_hip_copy2d_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node,
					    uintptr_t dst, size_t dst_offset, unsigned dst_node,
					    size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					    struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_HIP_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_hip_copy2d_async_sync((void*) (src + src_offset), src_node,
					    (void*) (dst + dst_offset), dst_node,
					    blocksize, numblocks, ld_src, ld_dst,
					    async_channel?starpu_hip_get_out_transfer_stream(src_node):NULL,
					    hipMemcpyDeviceToHost);
}

int _starpu_hip_copy2d_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node,
					    uintptr_t dst, size_t dst_offset, unsigned dst_node,
					    size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					    struct _starpu_async_channel *async_channel)
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
					    async_channel?starpu_hip_get_peer_transfer_stream(src_node, dst_node):NULL,
					    hipMemcpyDeviceToDevice);
}

int _starpu_hip_copy2d_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node,
					    uintptr_t dst, size_t dst_offset, unsigned dst_node,
					    size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst,
					    struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_HIP_RAM);

	return starpu_hip_copy2d_async_sync((void*) (src + src_offset), src_node,
					    (void*) (dst + dst_offset), dst_node,
					    blocksize, numblocks, ld_src, ld_dst,
					    async_channel?starpu_hip_get_in_transfer_stream(dst_node):NULL,
					    hipMemcpyHostToDevice);
}

int _starpu_hip_is_direct_access_supported(unsigned node, unsigned handling_node)
{
#if defined(STARPU_HAVE_HIP_MEMCPY_PEER)
	(void) node;
	enum starpu_node_kind kind = starpu_node_get_kind(handling_node);
	return kind == STARPU_HIP_RAM;
#else /* STARPU_HAVE_HIP_MEMCPY_PEER */
	/* Direct GPU-GPU transfers are not allowed in general */
	(void) node;
	(void) handling_node;
	return 0;
#endif /* STARPU_HAVE_HIP_MEMCPY_PEER */
}

static void start_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker, unsigned char pipeline_idx STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);
	j->workerid = worker->workerid;

	if (worker->ntasks == 1)
	{
		/* We are alone in the pipeline, the kernel will start now, record it */
		_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);
	}

#if defined(STARPU_HAVE_HIP_MEMCPY_PEER)
	/* We make sure we do manipulate the proper device */
	starpu_hip_set_device(worker->devid);
#endif

	starpu_hip_func_t func = _starpu_task_get_hip_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_HIP is defined in 'where', hip_func or hip_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_gpu_exec, worker->devid, worker->workerid, starpu_prof_tool_driver_hip, -1, (void*)func);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec(&pi, NULL, NULL);
#endif

		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);

#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_gpu_exec, worker->devid, worker->workerid, starpu_prof_tool_driver_hip, -1, (void*)func);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec(&pi, NULL, NULL);
#endif
		_STARPU_TRACE_END_EXECUTING();
	}
}

static void finish_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static void execute_job_on_hip(struct starpu_task *task, struct _starpu_worker *worker)
{
	int workerid = worker->workerid;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	unsigned char pipeline_idx = (worker->first_task + worker->ntasks - 1)%STARPU_MAX_PIPELINE;

	start_job_on_hip(j, worker, pipeline_idx);

	if (!used_stream[workerid])
	{
		used_stream[workerid] = 1;
		_STARPU_DISP("Warning: starpu_hip_get_local_stream() was not used to submit kernel to HIP on worker %d. HIP will thus introduce a lot of useless synchronizations, which will prevent proper overlapping of data transfers and kernel execution. See the HIP-specific part of the 'Check List When Performance Are Not There' of the StarPU handbook\n", workerid);
	}

	if (task->cl->hip_flags[j->nimpl] & STARPU_HIP_ASYNC)
	{
		if (worker->pipeline_length == 0)
		{
			/* Forced synchronous execution */
			hipStreamSynchronize(starpu_hip_get_local_stream());
			finish_job_on_hip(j, worker);
		}
		else
		{
			/* Record event to synchronize with task termination later */
			hipError_t hipres = hipEventRecord(task_events[workerid][pipeline_idx], starpu_hip_get_local_stream());
			if (STARPU_UNLIKELY(hipres))
				STARPU_HIP_REPORT_ERROR(hipres);
#ifdef STARPU_USE_FXT
			_STARPU_TRACE_START_EXECUTING();
#endif
		}
	}
	else /* Synchronous execution */
	{
		STARPU_ASSERT_MSG(hipStreamQuery(starpu_hip_get_local_stream()) == hipSuccess, "Unless when using the STARPU_HIP_ASYNC flag, HIP codelets have to wait for termination of their kernels on the starpu_hip_get_local_stream() stream");
		finish_job_on_hip(j, worker);
	}
}

static void finish_job_on_hip(struct _starpu_job *j, struct _starpu_worker *worker)
{
	int profiling = starpu_profiling_status_get();

	if (worker->pipeline_length)
		worker->current_tasks[worker->first_task] = NULL;
	else
		worker->current_task = NULL;
	worker->first_task = (worker->first_task + 1) % STARPU_MAX_PIPELINE;
	worker->ntasks--;

	_starpu_driver_end_job(worker, j, &worker->perf_arch, 0, profiling);

	_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

	_starpu_push_task_output(j);

	_starpu_set_current_task(NULL);

	_starpu_handle_job_termination(j);
}

/* One iteration of the main driver loop */
int _starpu_hip_driver_run_once(struct _starpu_worker *worker)
{
	struct _starpu_worker_set *worker_set = worker->set;
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	struct starpu_task *tasks[worker_set->nworkers];
	struct starpu_task *task;
	struct _starpu_job *j;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif
	int i, res;


	int idle_tasks, idle_transfers;

	/* First poll for completed jobs */
	idle_tasks = 0;
	idle_transfers = 0;
	for (i = 0; i < (int) worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		int workerid = worker->workerid;
		unsigned memnode = worker->memory_node;

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
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_hip, memnode, NULL);
			starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
			j = _starpu_get_job_associated_to_task(task);

			_starpu_set_local_worker_key(worker);
			_starpu_fetch_task_input_tail(task, j, worker);
			/* Reset it */
			worker->task_transferring = NULL;
			if (worker->ntasks > 1 && !(task->cl->hip_flags[j->nimpl] & STARPU_HIP_ASYNC))
			{
				/* We have to execute a non-asynchronous task but we
				 * still have tasks in the pipeline...  Record it to
				 * prevent more tasks from coming, and do it later */
				worker->pipeline_stuck = 1;
			}
			else
			{
				execute_job_on_hip(task, worker);
			}
			_STARPU_TRACE_START_PROGRESS(memnode);
		}

		/* Then test for termination of queued tasks */
		if (!worker->ntasks)
			/* No queued task */
			continue;

		if (worker->pipeline_length)
			task = worker->current_tasks[worker->first_task];
		else
			task = worker->current_task;
		if (task == worker->task_transferring)
			/* Next task is still pending transfer */
			continue;

		/* On-going asynchronous task, check for its termination first */
		hipError_t hipres = hipEventQuery(task_events[workerid][worker->first_task]);

		if (hipres != hipSuccess)
		{
			STARPU_ASSERT_MSG(hipres == hipErrorNotReady, "HIP error on task %p, codelet %p (%s): %s (%d)", task, task->cl, _starpu_codelet_get_model_name(task->cl), hipGetErrorString(hipres), hipres);
		}
		else
		{
			_STARPU_TRACE_END_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_hip, memnode, NULL);
			starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
			/* Asynchronous task completed! */
			_starpu_set_local_worker_key(worker);
			finish_job_on_hip(_starpu_get_job_associated_to_task(task), worker);
			/* See next task if any */
			if (worker->ntasks)
			{
				if (worker->current_tasks[worker->first_task] != worker->task_transferring)
				{
					task = worker->current_tasks[worker->first_task];
					j = _starpu_get_job_associated_to_task(task);
					if (task->cl->hip_flags[j->nimpl] & STARPU_HIP_ASYNC)
					{
						/* An asynchronous task, it was already
						 * queued, it's now running, record its start time.  */
						_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, starpu_profiling_status_get());
					}
					else
					{
						/* A synchronous task, we have finished
						 * flushing the pipeline, we can now at
						 * last execute it.  */

						_STARPU_TRACE_EVENT("sync_task");
						execute_job_on_hip(task, worker);
						_STARPU_TRACE_EVENT("end_sync_task");
						worker->pipeline_stuck = 0;
					}
				}
				else
					/* Data for next task didn't have time to finish transferring :/ */
					_STARPU_TRACE_WORKER_START_FETCH_INPUT(NULL, workerid);
			}
#ifdef STARPU_USE_FXT
			_STARPU_TRACE_END_EXECUTING()
#endif
			_STARPU_TRACE_START_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_hip, worker->memory_node, NULL);
			starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
		}
		if (!worker->pipeline_length || worker->ntasks < worker->pipeline_length)
			idle_tasks++;
	}

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

	/* And pull tasks */
	res |= _starpu_get_multi_worker_task(worker_set->workers, tasks, worker_set->nworkers, worker0->memory_node);

	for (i = 0; i < (int) worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned memnode STARPU_ATTRIBUTE_UNUSED = worker->memory_node;

		task = tasks[i];
		if (!task)
			continue;


		j = _starpu_get_job_associated_to_task(task);

		/* can HIP do that task ? */
		if (!_STARPU_MAY_PERFORM(j, HIP))
		{
			/* this is neither a cuda or a cublas task */
			_starpu_worker_refuse_task(worker, task);
			continue;
		}

                /* Fetch data asynchronously */
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
		_STARPU_TRACE_END_PROGRESS(memnode);
		_starpu_set_local_worker_key(worker);
		res = _starpu_fetch_task_input(task, j, 1);
		STARPU_ASSERT(res == 0);
		_STARPU_TRACE_START_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
	}

	return 0;
}

void *_starpu_hip_worker(void *_arg)
{
	struct _starpu_worker *worker = _arg;
	struct _starpu_worker_set* worker_set = worker->set;

#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif
	unsigned i;

	_starpu_hip_driver_init(worker);
	for (i = 0; i < worker_set->nworkers; i++)
	{
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_transfer, worker_set->workers[i].workerid, worker_set->workers[i].workerid, starpu_prof_tool_driver_gpu, worker_set->workers[i].memory_node, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
		_STARPU_TRACE_START_PROGRESS(worker_set->workers[i].memory_node);
	}
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_hip_driver_run_once(worker);
	}
	for (i = 0; i < worker_set->nworkers; i++)
	{
		_STARPU_TRACE_END_PROGRESS(worker_set->workers[i].memory_node);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, worker_set->workers[i].workerid, worker_set->workers[i].workerid, starpu_prof_tool_driver_gpu, worker_set->workers[i].memory_node, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
	}
	_starpu_hip_driver_deinit(worker);

	return NULL;
}

#ifdef STARPU_HAVE_HWLOC
hwloc_obj_t _starpu_hip_get_hwloc_obj(hwloc_topology_t topology, int devid)
{
#if HAVE_DECL_HWLOC_HIP_GET_DEVICE_OSDEV_BY_INDEX
	return hwloc_hip_get_device_osdev_by_index(topology, devid);
#else
	(void)topology;
	(void)devid;
	return NULL;
#endif
}
#endif

void starpu_hipblas_report_error(const char *func, const char *file, int line, int status)
{
        char *errormsg;
        switch (status)
        {
                case HIPBLAS_STATUS_SUCCESS:
                        errormsg = "success";
                        break;
                case HIPBLAS_STATUS_NOT_INITIALIZED:
                        errormsg = "not initialized";
                        break;
                case HIPBLAS_STATUS_ALLOC_FAILED:
                        errormsg = "alloc failed";
                        break;
                case HIPBLAS_STATUS_INVALID_VALUE:
                        errormsg = "invalid value";
                        break;
                case HIPBLAS_STATUS_ARCH_MISMATCH:
                        errormsg = "arch mismatch";
                        break;
                case HIPBLAS_STATUS_EXECUTION_FAILED:
                        errormsg = "execution failed";
                        break;
                case HIPBLAS_STATUS_INTERNAL_ERROR:
                        errormsg = "internal error";
                        break;
                default:
                        errormsg = "unknown error";
                        break;
        }
        _STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
        STARPU_ABORT();
}

void starpu_hip_report_error(const char *func, const char *file, int line, hipError_t status)
{
	const char *errormsg = hipGetErrorString(status);
	_STARPU_ERROR("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

int _starpu_hip_run_from_worker(struct _starpu_worker *worker)
{
	/* Let's go ! */
	_starpu_hip_worker(worker);

	return 0;
}

int _starpu_hip_driver_set_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	driver->id.hip_id = worker->devid;
	return 0;
}

int _starpu_hip_driver_is_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	return driver->id.hip_id == worker->devid;
}

struct _starpu_driver_ops _starpu_driver_hip_ops =
{
	.init = _starpu_hip_driver_init,
	.run = _starpu_hip_run_from_worker,
	.run_once = _starpu_hip_driver_run_once,
	.deinit = _starpu_hip_driver_deinit,
	.set_devid = _starpu_hip_driver_set_devid,
	.is_devid = _starpu_hip_driver_is_devid,
};

struct _starpu_node_ops _starpu_driver_hip_node_ops =
{
	.name = "hip driver",
	.malloc_on_node = _starpu_hip_malloc_on_node,
	.free_on_node = _starpu_hip_free_on_node,

	.is_direct_access_supported = _starpu_hip_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_hip_copy_interface_from_hip_to_cpu,
	.copy_interface_to[STARPU_HIP_RAM] = _starpu_hip_copy_interface_from_hip_to_hip,

	.copy_interface_from[STARPU_CPU_RAM] = _starpu_hip_copy_interface_from_cpu_to_hip,
	.copy_interface_from[STARPU_HIP_RAM] = _starpu_hip_copy_interface_from_hip_to_hip,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_hip_copy_data_from_hip_to_cpu,
	.copy_data_to[STARPU_HIP_RAM] = _starpu_hip_copy_data_from_hip_to_hip,

	.copy_data_from[STARPU_CPU_RAM] = _starpu_hip_copy_data_from_cpu_to_hip,
	.copy_data_from[STARPU_HIP_RAM] = _starpu_hip_copy_data_from_hip_to_hip,

	.copy2d_data_to[STARPU_CPU_RAM] = _starpu_hip_copy2d_data_from_hip_to_cpu,
	.copy2d_data_to[STARPU_HIP_RAM] = _starpu_hip_copy2d_data_from_hip_to_hip,

	.copy2d_data_from[STARPU_CPU_RAM] = _starpu_hip_copy2d_data_from_cpu_to_hip,
	.copy2d_data_from[STARPU_HIP_RAM] = _starpu_hip_copy2d_data_from_hip_to_hip,

	.wait_request_completion = _starpu_hip_wait_request_completion,
	.test_request_completion = _starpu_hip_test_request_completion,
};
