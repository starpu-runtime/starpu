/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <core/devices.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/driver_common/driver_common.h>
#include <drivers/cuda/driver_cuda.h>
#include <core/sched_policy.h>
#ifdef HAVE_CUDA_GL_INTEROP_H
#include <cuda_gl_interop.h>
#endif
#ifdef STARPU_HAVE_LIBNVIDIA_ML
#include <nvml.h>
#endif
#ifdef STARPU_USE_CUDA
#include <cublas.h>
#endif
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/task.h>
#include <common/knobs.h>
#include <profiling/callbacks.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

#if HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/cuda.h>
#endif

#ifdef STARPU_USE_CUDA
#if CUDART_VERSION >= 5000
/* Avoid letting our streams spuriously synchonize with the NULL stream */
#define starpu_cudaStreamCreate(stream) cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#else
#define starpu_cudaStreamCreate(stream) cudaStreamCreate(stream)
#endif

/* At least CUDA 4.2 still didn't have working memcpy3D */
#if CUDART_VERSION < 5000
#define BUGGED_MEMCPY3D
#endif
#endif

/* Consider a rough 10% overhead cost */
#define FREE_MARGIN 0.9

/* the number of CUDA devices */
static int ncudagpus = -1;

static size_t global_mem[STARPU_MAXCUDADEVS];
#ifdef STARPU_HAVE_LIBNVIDIA_ML
static nvmlDevice_t nvmlDev[STARPU_MAXCUDADEVS];
#endif
int _starpu_cuda_bus_ids[STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES][STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES];
#ifdef STARPU_USE_CUDA
static cudaStream_t streams[STARPU_NMAXWORKERS];
static char used_stream[STARPU_NMAXWORKERS];
/* TODO: ideally we'd have different streams for idle, prefetch and fetch, but apparently CUDA doesn't take priorities into account for transfers anyway? */
static cudaStream_t out_transfer_streams[STARPU_MAXCUDADEVS];
static cudaStream_t in_transfer_streams[STARPU_MAXCUDADEVS];
/* Note: streams are not thread-safe, so we define them for each CUDA worker
 * emitting a GPU-GPU transfer */
static cudaStream_t in_peer_transfer_streams[STARPU_MAXCUDADEVS][STARPU_MAXCUDADEVS];
static struct cudaDeviceProp props[STARPU_MAXCUDADEVS];
#ifndef STARPU_SIMGRID
static cudaEvent_t task_events[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
#endif
#endif /* STARPU_USE_CUDA */
#ifdef STARPU_SIMGRID
static unsigned task_finished[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
static starpu_pthread_mutex_t cuda_alloc_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif /* STARPU_SIMGRID */

static enum initialization cuda_device_init[STARPU_MAXCUDADEVS];
static int cuda_device_users[STARPU_MAXCUDADEVS];
static starpu_pthread_mutex_t cuda_device_init_mutex[STARPU_MAXCUDADEVS];
static starpu_pthread_cond_t cuda_device_init_cond[STARPU_MAXCUDADEVS];

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
static struct _starpu_worker_set cuda_worker_set[STARPU_MAXCUDADEVS];

static unsigned cuda_bindid_init[STARPU_MAXCUDADEVS];
static unsigned cuda_bindid[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_init[STARPU_MAXCUDADEVS];
static unsigned cuda_memory_nodes[STARPU_MAXCUDADEVS];
static int cuda_globalbindid;
#endif

int _starpu_nworker_per_cuda;

static size_t _starpu_cuda_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}

#ifdef STARPU_USE_CUDA
static cudaStream_t starpu_cuda_get_in_transfer_stream(unsigned dst_node)
{
	int dst_devid = starpu_memory_node_get_devid(dst_node);
	cudaStream_t stream;

	stream = in_transfer_streams[dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static cudaStream_t starpu_cuda_get_out_transfer_stream(unsigned src_node)
{
	int src_devid = starpu_memory_node_get_devid(src_node);
	cudaStream_t stream;

	stream = out_transfer_streams[src_devid];
	STARPU_ASSERT(stream);
	return stream;
}

static cudaStream_t starpu_cuda_get_peer_transfer_stream(unsigned src_node, unsigned dst_node)
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
#endif /* STARPU_USE_CUDA */


/* Early library initialization, before anything else, just initialize data */
void _starpu_cuda_init(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXCUDADEVS; i++)
	{
		STARPU_PTHREAD_MUTEX_INIT(&cuda_device_init_mutex[i], NULL);
		STARPU_PTHREAD_COND_INIT(&cuda_device_init_cond[i], NULL);
	}
	memset(&cuda_bindid_init, 0, sizeof(cuda_bindid_init));
	memset(&cuda_memory_init, 0, sizeof(cuda_memory_init));
	cuda_globalbindid = -1;
}

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXCUDADEVS */

unsigned _starpu_get_cuda_device_count(void)
{
	int cnt;
#ifdef STARPU_SIMGRID
	cnt = _starpu_simgrid_get_nbhosts("CUDA");
#else
	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;
#endif

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

#ifdef STARPU_SIMGRID
	config->topology.nhwdevices[STARPU_CUDA_WORKER] = _starpu_simgrid_get_nbhosts("CUDA");
#else
	int cnt;
	cudaError_t cures;

	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures != cudaSuccess))
		cnt = 0;
	config->topology.nhwdevices[STARPU_CUDA_WORKER] = cnt;
#ifdef STARPU_HAVE_LIBNVIDIA_ML
	nvmlInit();
#endif
#endif
}

#ifdef STARPU_HAVE_LIBNVIDIA_ML
static int _starpu_cuda_direct_link(unsigned devid1, unsigned devid2)
{
	unsigned i;
	struct cudaDeviceProp props_dev1;
	struct cudaDeviceProp props_dev2;
	cudaError_t cures;

	cures = cudaGetDeviceProperties(&props_dev1, devid1);
	if (cures != cudaSuccess)
		return 0;
	cures = cudaGetDeviceProperties(&props_dev2, devid2);
	if (cures != cudaSuccess)
		return 0;

	nvmlDevice_t nvml_dev1 = _starpu_cuda_get_nvmldev(&props_dev1);

	if (!nvml_dev1)
		return 0;

	for (i = 0; i < NVML_NVLINK_MAX_LINKS; i++) {
		nvmlEnableState_t active;
		nvmlReturn_t ret;
		ret = nvmlDeviceGetNvLinkState(nvml_dev1, i, &active);
		if (ret == NVML_ERROR_NOT_SUPPORTED)
			continue;
		if (active != NVML_FEATURE_ENABLED)
			continue;

		nvmlPciInfo_t pci;
		nvmlDeviceGetNvLinkRemotePciInfo(nvml_dev1, i, &pci);
		if ((int) pci.domain == props_dev2.pciDomainID &&
		    (int) pci.bus == props_dev2.pciBusID &&
		    (int) pci.device == props_dev2.pciDeviceID)
			/* We have a direct NVLink! */
			return 1;
	}

	/* No direct NVLink found */
	return 0;
}
#endif

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
	int i;

	for (i = 0; i < (int) (sizeof(cuda_worker_set)/sizeof(cuda_worker_set[0])); i++)
		cuda_worker_set[i].workers = NULL;

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

	int nworker_per_cuda = starpu_getenv_number_default("STARPU_NWORKER_PER_CUDA", 1);

	STARPU_ASSERT_MSG(nworker_per_cuda > 0, "STARPU_NWORKER_PER_CUDA has to be > 0");
	STARPU_ASSERT_MSG(nworker_per_cuda < STARPU_NMAXWORKERS, "STARPU_NWORKER_PER_CUDA (%d) cannot be higher than STARPU_NMAXWORKERS (%d)\n", nworker_per_cuda, STARPU_NMAXWORKERS);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	if (nworker_per_cuda > 1)
	{
		_STARPU_DISP("Warning: reducing STARPU_NWORKER_PER_CUDA to 1 because blocking drivers are enabled\n");
		nworker_per_cuda = 1;
	}
	_starpu_nworker_per_cuda = nworker_per_cuda;
#endif
	/* Now we know how many CUDA devices will be used */
	topology->ndevices[STARPU_CUDA_WORKER] = ncuda;

	_starpu_initialize_workers_cuda_gpuid(config);

	/* allow having one worker per stream */
	topology->cuda_th_per_stream = starpu_getenv_number_default("STARPU_CUDA_THREAD_PER_WORKER", -1);
	topology->cuda_th_per_dev = starpu_getenv_number_default("STARPU_CUDA_THREAD_PER_DEV", -1);

	STARPU_ASSERT_MSG(!(topology->cuda_th_per_stream == 1 && topology->cuda_th_per_dev != -1), "It does not make sense to set both STARPU_CUDA_THREAD_PER_WORKER to 1 and to set STARPU_CUDA_THREAD_PER_DEV, please choose either per worker or per device or none");

	/* per device by default */
	if (topology->cuda_th_per_dev == -1)
	{
		if (topology->cuda_th_per_stream == 1)
			topology->cuda_th_per_dev = 0;
		else
			topology->cuda_th_per_dev = 1;
	}
	/* Not per stream by default */
	if (topology->cuda_th_per_stream == -1)
	{
		topology->cuda_th_per_stream = 0;
	}

	if (!topology->cuda_th_per_dev)
	{
		cuda_worker_set[0].workers = &config->workers[topology->nworkers];
		cuda_worker_set[0].nworkers = ncuda * nworker_per_cuda;
	}

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

		struct _starpu_worker_set *worker_set;

		if(topology->cuda_th_per_stream)
		{
			worker_set = ALLOC_WORKER_SET;
		}
		else if (topology->cuda_th_per_dev)
		{
			worker_set = &cuda_worker_set[devid];
			worker_set->workers = &config->workers[topology->nworkers];
			worker_set->nworkers = nworker_per_cuda;
		}
		else
		{
			/* Same worker set for all devices */
			worker_set = &cuda_worker_set[0];
		}

		_starpu_topology_configure_workers(topology, config,
						   STARPU_CUDA_WORKER,
						   cudagpu, devid, 0, 0,
						   nworker_per_cuda,
						   // TODO: fix perfmodels etc.
						   // nworker_per_cuda - 1,
						   1,
						   worker_set, NULL);

		_starpu_devices_gpu_set_used(devid);

		/* TODO: move this to generic place */
#ifdef STARPU_HAVE_HWLOC
		{
			hwloc_obj_t obj = NULL;
			if (starpu_driver_info[STARPU_CUDA_WORKER].get_hwloc_obj)
				obj = starpu_driver_info[STARPU_CUDA_WORKER].get_hwloc_obj(topology->hwtopology, devid);

			if (obj)
			{
				struct _starpu_hwloc_userdata *data = obj->userdata;
				data->ngpus++;
			}
			else
			{
				_STARPU_DISP("Warning: could not find location of CUDA%u, do you have the hwloc CUDA plugin installed?\n", devid);
			}
		}
#endif
	}
}

/* Bind the driver on a CPU core */
void _starpu_cuda_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Perhaps the worker has some "favourite" bindings (logical core) */
	unsigned preferred_binding[STARPU_NMAXWORKERS];
	unsigned npreferred = 0;
	unsigned devid = workerarg->devid;

#ifndef STARPU_SIMGRID
	if (_starpu_may_bind_automatically[STARPU_CUDA_WORKER])
	{
		/* StarPU is allowed to bind threads automatically */
		unsigned *preferred_numa_binding = _starpu_get_cuda_affinity_vector(devid);
		unsigned npreferred_numa = _starpu_topology_get_nhwnumanodes(config);
		npreferred = _starpu_topology_get_numa_core_binding(config, preferred_numa_binding, npreferred_numa, preferred_binding, STARPU_NMAXWORKERS);
	}
#endif /* SIMGRID */
	if (cuda_bindid_init[devid])
	{
		if (config->topology.cuda_th_per_stream == 0)
			workerarg->bindid = cuda_bindid[devid];
		else
			workerarg->bindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
	}
	else
	{
		cuda_bindid_init[devid] = 1;

		if (config->topology.cuda_th_per_dev == 0 && config->topology.cuda_th_per_stream == 0)
		{
			if (cuda_globalbindid == -1)
				cuda_globalbindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
			workerarg->bindid = cuda_bindid[devid] = cuda_globalbindid;
		}
		else
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

#ifdef STARPU_USE_CUDA_MAP
		/* TODO: check node capabilities */
		_starpu_memory_node_set_mapped(memory_node);
#endif

		for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
		{
			_starpu_cuda_bus_ids[numa][devid+STARPU_MAXNUMANODES] = _starpu_register_bus(numa, memory_node);
			_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][numa] = _starpu_register_bus(memory_node, numa);
		}

#ifdef STARPU_SIMGRID
		const char* cuda_memcpy_peer;
		char name[16];
		snprintf(name, sizeof(name), "CUDA%u", devid);
		starpu_sg_host_t host = _starpu_simgrid_get_host_by_name(name);
		STARPU_ASSERT(host);
		_starpu_simgrid_memory_node_set_host(memory_node, host);
#  ifdef STARPU_HAVE_SIMGRID_ACTOR_H
		cuda_memcpy_peer = sg_host_get_property_value(host, "memcpy_peer");
#  else
		cuda_memcpy_peer = MSG_host_get_property_value(host, "memcpy_peer");
#  endif
#endif /* SIMGRID */

		if (
#ifdef STARPU_SIMGRID
			cuda_memcpy_peer && atoll(cuda_memcpy_peer)
#elif defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
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
				if (workerarg2->arch == STARPU_CUDA_WORKER)
				{
					unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
					int bus21 = _starpu_register_bus(memory_node2, memory_node);
					int bus12 = _starpu_register_bus(memory_node, memory_node2);
					if (bus21 < 0 || bus12 < 0)
						/* Already registered because of e.g. several workers per CUDA */
						continue;
					_starpu_cuda_bus_ids[devid2+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES] = bus21;
					_starpu_cuda_bus_ids[devid+STARPU_MAXNUMANODES][devid2+STARPU_MAXNUMANODES] = bus12;
#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_LIBNVIDIA_ML
					if (_starpu_cuda_direct_link(devid, devid2))
					{
						starpu_bus_set_ngpus(bus21, 1);
						starpu_bus_set_ngpus(bus12, 1);
					}
					else
#endif
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

/* Set the current CUDA device */
void starpu_cuda_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	STARPU_ABORT();
#else
	cudaError_t cures;
	struct starpu_conf *conf = &_starpu_get_machine_config()->conf;
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
	unsigned i;
#endif

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (conf->n_cuda_opengl_interoperability)
	{
		_STARPU_MSG("OpenGL interoperability was requested, but StarPU was built with multithread GPU control support, please reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#elif !defined(HAVE_CUDA_GL_INTEROP_H)
	if (conf->n_cuda_opengl_interoperability)
	{
		_STARPU_MSG("OpenGL interoperability was requested, but cuda_gl_interop.h could not be compiled, please make sure that OpenGL headers were available before ./configure run.");
		STARPU_ABORT();
	}
#else
	for (i = 0; i < conf->n_cuda_opengl_interoperability; i++)
	{
		if (conf->cuda_opengl_interoperability[i] == devid)
		{
			cures = cudaGLSetGLDevice(devid);
			goto done;
		}
	}
#endif

	cures = cudaSetDevice(devid);

#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
done:
#endif
#ifdef STARPU_OPENMP
	/* When StarPU is used as Open Runtime support,
	 * starpu_omp_shutdown() will usually be called from a
	 * destructor, in which case cudaThreadExit() reports a
	 * cudaErrorCudartUnloading here. There should not
	 * be any remaining tasks running at this point so
	 * we can probably ignore it without much consequences. */
	if (STARPU_UNLIKELY(cures && cures != cudaErrorCudartUnloading))
		STARPU_CUDA_REPORT_ERROR(cures);
#else
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
#endif /* STARPU_OPENMP */
#endif
}

/* In case we want to cap the amount of memory available on the GPUs by the
 * mean of the STARPU_LIMIT_CUDA_MEM, we decrease the value of
 * global_mem[devid] which is the value returned by
 * _starpu_cuda_get_global_mem_size() to indicate how much memory can
 * be allocated on the device
 */
static void _starpu_cuda_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

#ifdef STARPU_SIMGRID
	totalGlobalMem = _starpu_simgrid_get_memsize("CUDA", devid);
#elif defined(STARPU_USE_CUDA)
	/* Find the size of the memory on the device */
	totalGlobalMem = props[devid].totalGlobalMem;
#endif

	limit = starpu_getenv_number("STARPU_LIMIT_CUDA_MEM");
	if (limit == -1)
	{
		char name[30];
		snprintf(name, sizeof(name), "STARPU_LIMIT_CUDA_%u_MEM", devid);
		limit = starpu_getenv_number(name);
	}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (limit == -1)
	{
		limit = totalGlobalMem / (1024*1024) * FREE_MARGIN;
	}
#endif

	global_mem[devid] = limit * 1024*1024;

#ifdef STARPU_USE_CUDA
	/* How much memory to waste ? */
	to_waste = totalGlobalMem - global_mem[devid];

	props[devid].totalGlobalMem -= to_waste;
#endif /* STARPU_USE_CUDA */

	_STARPU_DEBUG("CUDA device %u: Wasting %ld MB / Limit %ld MB / Total %ld MB / Remains %ld MB\n",
			devid, (long) to_waste/(1024*1024), (long) limit, (long) totalGlobalMem/(1024*1024),
			(long) (totalGlobalMem - to_waste)/(1024*1024));
}

/* Really initialize one device */
static void init_device_context(unsigned devid, unsigned memnode)
{
#ifndef STARPU_SIMGRID
	cudaError_t cures;

	/* TODO: cudaSetDeviceFlag(cudaDeviceMapHost) */

	starpu_cuda_set_device(devid);
#endif /* !STARPU_SIMGRID */

	STARPU_PTHREAD_MUTEX_LOCK(&cuda_device_init_mutex[devid]);
	cuda_device_users[devid]++;
	if (cuda_device_init[devid] == UNINITIALIZED)
		/* Nobody started initialization yet, do it */
		cuda_device_init[devid] = CHANGING;
	else
	{
		/* Somebody else is doing initialization, wait for it */
		while (cuda_device_init[devid] != INITIALIZED)
			STARPU_PTHREAD_COND_WAIT(&cuda_device_init_cond[devid], &cuda_device_init_mutex[devid]);
		STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_device_init_mutex[devid]);
		return;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_device_init_mutex[devid]);

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (starpu_getenv_number("STARPU_ENABLE_CUDA_GPU_GPU_DIRECT") != 0)
	{
		int nworkers = starpu_worker_get_count();
		int workerid;
		for (workerid = 0; workerid < nworkers; workerid++)
		{
			struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
			if (worker->arch == STARPU_CUDA_WORKER && worker->devid != devid)
			{
				int can;
				cures = cudaDeviceCanAccessPeer(&can, devid, worker->devid);
				(void) cudaGetLastError();

				if (!cures && can)
				{
					cures = cudaDeviceEnablePeerAccess(worker->devid, 0);
					(void) cudaGetLastError();

					if (!cures)
					{
						_STARPU_DEBUG("Enabled GPU-Direct %d -> %d\n", worker->devid, devid);
						/* direct copies are made from the destination, see link_supports_direct_transfers */
						starpu_bus_set_direct(_starpu_cuda_bus_ids[worker->devid+STARPU_MAXNUMANODES][devid+STARPU_MAXNUMANODES], 1);
					}
				}
			}
		}
	}
#endif

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
#endif /* !STARPU_SIMGRID */

	STARPU_PTHREAD_MUTEX_LOCK(&cuda_device_init_mutex[devid]);
	cuda_device_init[devid] = INITIALIZED;
	STARPU_PTHREAD_COND_BROADCAST(&cuda_device_init_cond[devid]);
	STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_device_init_mutex[devid]);

	_starpu_cuda_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cuda_get_global_mem_size(devid));
}

/* De-initialize one device */
static void deinit_device_context(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
#ifndef STARPU_SIMGRID
	int i;
	starpu_cuda_set_device(devid);

	cudaStreamDestroy(in_transfer_streams[devid]);
	cudaStreamDestroy(out_transfer_streams[devid]);

	for (i = 0; i < ncudagpus; i++)
	{
		cudaStreamDestroy(in_peer_transfer_streams[i][devid]);
	}
#endif /* !STARPU_SIMGRID */
}

static void init_worker_context(unsigned workerid, unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	int j;
#ifdef STARPU_SIMGRID
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		task_finished[workerid][j] = 0;
#else /* !STARPU_SIMGRID */
	cudaError_t cures;
	starpu_cuda_set_device(devid);

	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
	{
		cures = cudaEventCreateWithFlags(&task_events[workerid][j], cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}

	cures = starpu_cudaStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

#endif /* !STARPU_SIMGRID */
}

static void deinit_worker_context(unsigned workerid, unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
	unsigned j;
#ifdef STARPU_SIMGRID
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		task_finished[workerid][j] = 0;
#else /* STARPU_SIMGRID */
	starpu_cuda_set_device(devid);
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		cudaEventDestroy(task_events[workerid][j]);
	cudaStreamDestroy(streams[workerid]);
#endif /* STARPU_SIMGRID */
}

#ifdef STARPU_HAVE_LIBNVIDIA_ML
nvmlDevice_t _starpu_cuda_get_nvmldev(struct cudaDeviceProp *dev_props)
{
	char busid[13];
	nvmlDevice_t ret;

	snprintf(busid, sizeof(busid), "%04x:%02x:%02x.0", dev_props->pciDomainID, dev_props->pciBusID, dev_props->pciDeviceID);
	if (nvmlDeviceGetHandleByPciBusId(busid, &ret) != NVML_SUCCESS)
		ret = NULL;

	return ret;
}

nvmlDevice_t starpu_cuda_get_nvmldev(unsigned devid)
{
	return nvmlDev[devid];
}
#endif

/* This is run from the driver thread to initialize the driver CUDA context */
int _starpu_cuda_driver_init(struct _starpu_worker *worker)
{
	struct _starpu_worker_set *worker_set = worker->set;
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	int lastdevid = -1;
	unsigned i;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif

	_starpu_driver_start(worker0, STARPU_CUDA_WORKER, 0);
	_starpu_set_local_worker_set_key(worker_set);

#ifdef STARPU_USE_FXT
	for (i = 1; i < worker_set->nworkers; i++)
		_starpu_worker_start(&worker_set->workers[i], STARPU_CUDA_WORKER, 0);
#endif

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned memnode = worker->memory_node;

#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init, devid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init(&pi, NULL, NULL);
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_start, devid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start(&pi, NULL, NULL);
#endif

		if ((int) devid == lastdevid)
		{
#ifdef STARPU_SIMGRID
			STARPU_ASSERT_MSG(0, "Simgrid mode does not support concurrent kernel execution yet\n");
#endif /* !STARPU_SIMGRID */

			/* Already initialized */
			continue;
		}
		lastdevid = devid;
		init_device_context(devid, memnode);

#ifndef STARPU_SIMGRID
		if (worker->config->topology.nworker[STARPU_CUDA_WORKER][devid] > 1 && props[devid].concurrentKernels == 0)
			_STARPU_DISP("Warning: STARPU_NWORKER_PER_CUDA is %u, but CUDA device %u does not support concurrent kernel execution!\n", worker_set->nworkers, devid);
#endif /* !STARPU_SIMGRID */
	}

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(worker0->bindid, worker0->workerid, NULL);

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned workerid = worker->workerid;
		unsigned subdev = worker->subworkerid;

		float size = (float) global_mem[devid] / (1<<30);
#ifdef STARPU_SIMGRID
		const char *devname = _starpu_simgrid_get_devname("CUDA", devid);
		if (!devname)
			devname = "Simgrid";
#else
		/* get the device's name */
		char devname[64];
		strncpy(devname, props[devid].name, 63);
		devname[63] = 0;
#endif

#if defined(STARPU_HAVE_BUSID) && !defined(STARPU_SIMGRID)
#if defined(STARPU_HAVE_DOMAINID) && !defined(STARPU_SIMGRID)
#ifdef STARPU_HAVE_LIBNVIDIA_ML
		nvmlDev[devid] = _starpu_cuda_get_nvmldev(&props[devid]);
#endif
		if (props[devid].pciDomainID)
			snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB %04x:%02x:%02x.0)", devid, subdev, devname, size, props[devid].pciDomainID, props[devid].pciBusID, props[devid].pciDeviceID);
		else
#endif
			snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB %02x:%02x.0)", devid, subdev, devname, size, props[devid].pciBusID, props[devid].pciDeviceID);
#else
		snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB)", devid, subdev, devname, size);
#endif
		snprintf(worker->short_name, sizeof(worker->short_name), "CUDA %u.%u", devid, subdev);
		_STARPU_DEBUG("cuda (%s) dev id %u worker %u thread is ready to run on CPU %d !\n", devname, devid, subdev, worker->bindid);

		worker->pipeline_length = starpu_getenv_number_default("STARPU_CUDA_PIPELINE", 2);
		if (worker->pipeline_length > STARPU_MAX_PIPELINE)
		{
			_STARPU_DISP("Warning: STARPU_CUDA_PIPELINE is %u, but STARPU_MAX_PIPELINE is only %u\n", worker->pipeline_length, STARPU_MAX_PIPELINE);
			worker->pipeline_length = STARPU_MAX_PIPELINE;
		}
#if !defined(STARPU_SIMGRID) && !defined(STARPU_NON_BLOCKING_DRIVERS)
		if (worker->pipeline_length >= 1)
		{
			/* We need non-blocking drivers, to poll for CUDA task
			 * termination */
			_STARPU_DISP("Warning: reducing STARPU_CUDA_PIPELINE to 0 because blocking drivers are enabled (and simgrid is not enabled)\n");
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
		snprintf(thread_name, sizeof(thread_name), "CUDA %u", worker0->devid);
		starpu_pthread_setname(thread_name);
	}

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker0->mutex);
	worker0->status = STATUS_UNKNOWN;
	worker0->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker0->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker0->mutex);

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
	worker_set->set_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

	return 0;
}

int _starpu_cuda_driver_deinit(struct _starpu_worker *worker)
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

		STARPU_PTHREAD_MUTEX_LOCK(&cuda_device_init_mutex[devid]);
		usersleft = --cuda_device_users[devid];
		STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_device_init_mutex[devid]);

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
		STARPU_PTHREAD_MUTEX_LOCK(&cuda_device_init_mutex[devid]);
		cuda_device_init[devid] = UNINITIALIZED;
		STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_device_init_mutex[devid]);

	}

	for (i = 0; i < worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned workerid = worker->workerid;
		unsigned memnode = worker->memory_node;

		deinit_worker_context(workerid, worker->devid);

#ifdef STARPU_PROF_TOOL
		struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_deinit, workerid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit(&pi, NULL, NULL);
#endif
	}

	worker_set->workers[0].worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_CUDA_WORKER);

	return 0;
}

uintptr_t _starpu_cuda_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	uintptr_t addr = 0;
	(void) flags;

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)

#ifdef STARPU_SIMGRID
	static uintptr_t last[STARPU_MAXNODES];
#ifdef STARPU_DEVEL
#warning TODO: record used memory, using a simgrid property to know the available memory
#endif
	/* Sleep for the allocation */
	STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
	if (_starpu_simgrid_cuda_malloc_cost())
		starpu_sleep(0.000175);
	if (!last[dst_node])
		last[dst_node] = 1<<10;
	addr = last[dst_node];
	last[dst_node]+=size;
	STARPU_ASSERT(last[dst_node] >= addr);
	STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
#else
	unsigned devid = starpu_memory_node_get_devid(dst_node);
#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	starpu_cuda_set_device(devid);
#else
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	if (!worker || worker->arch != STARPU_CUDA_WORKER || worker->devid != devid)
		STARPU_ASSERT_MSG(0, "CUDA peer access is not available with this version of CUDA");
#endif
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
#endif
#endif
	return addr;
}

void _starpu_cuda_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void) dst_node;
	(void) addr;
	(void) size;
	(void) flags;

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_LOCK(&cuda_alloc_mutex);
	/* Sleep for the free */
	if (_starpu_simgrid_cuda_malloc_cost())
		starpu_sleep(0.000750);
	STARPU_PTHREAD_MUTEX_UNLOCK(&cuda_alloc_mutex);
	/* CUDA also synchronizes roughly everything on cudaFree */
	_starpu_simgrid_sync_gpus();
#else
	cudaError_t err;
	unsigned devid = starpu_memory_node_get_devid(dst_node);
#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	starpu_cuda_set_device(devid);
#else
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	if (!worker || worker->arch != STARPU_CUDA_WORKER || worker->devid != devid)
		STARPU_ASSERT_MSG(0, "CUDA peer access is not available with this version of CUDA");
#endif /* STARPU_HAVE_CUDA_MEMCPY_PEER */
	err = cudaFree((void*)addr);
#ifdef STARPU_OPENMP
	/* When StarPU is used as Open Runtime support,
	 * starpu_omp_shutdown() will usually be called from a
	 * destructor, in which case cudaThreadExit() reports a
	 * cudaErrorCudartUnloading here. There should not
	 * be any remaining tasks running at this point so
	 * we can probably ignore it without much consequences. */
	if (STARPU_UNLIKELY(err != cudaSuccess && err != cudaErrorCudartUnloading))
		STARPU_CUDA_REPORT_ERROR(err);
#else
	if (STARPU_UNLIKELY(err != cudaSuccess))
		STARPU_CUDA_REPORT_ERROR(err);
#endif /* STARPU_OPENMP */
#endif /* STARPU_SIMGRID */
#endif
}

#ifdef STARPU_USE_CUDA
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

#if 0
/* CUDA doesn't seem to be providing a way to set ld2?? */
int
starpu_cuda_copy3d_async_sync(void *src_ptr, unsigned src_node,
			      void *dst_ptr, unsigned dst_node,
			      size_t blocksize,
			      size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
			      size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
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
		p.srcPtr = make_cudaPitchedPtr((char *)src_ptr, ld1_src, blocksize, numblocks);
		p.dstPtr = make_cudaPitchedPtr((char *)dst_ptr, ld1_dst, blocksize, numblocks);
		// FIXME: how to pass ld2_src / ld2_dst ??
		p.extent = make_cudaExtent(blocksize, numblocks_1, numblocks_2);


		if (stream)
		{
			double start;
			starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
			cures = cudaMemcpy3DPeerAsync(&p, stream);
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
		struct cudaMemcpy3DParms p;
		memset(&p, 0, sizeof(p));

		p.srcPtr = make_cudaPitchedPtr((char *)src_ptr, ld1_src, blocksize, numblocks);
		p.dstPtr = make_cudaPitchedPtr((char *)dst_ptr, ld1_dst, blocksize, numblocks);
		// FIXME: how to pass ld2_src / ld2_dst ??
		p.extent = make_cudaExtent(blocksize, numblocks, 1);
		p.kind = kind;

		if (stream)
		{
			double start;
			starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
			cures = cudaMemcpy3DAsync(&p, stream);
			starpu_interface_end_driver_copy_async(src_node, dst_node, start);
		}

		/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
		if (stream == NULL || cures)
		{
			cures = cudaMemcpy3D(&p);
			if (!cures)
				cures = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);

			return 0;
		}
	}


	return -EAGAIN;
}
#endif

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
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() || !(copy_methods->cuda_to_cuda_async || copy_methods->any_to_any))
	{
		STARPU_ASSERT(copy_methods->cuda_to_cuda || copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		if (copy_methods->cuda_to_cuda)
			copy_methods->cuda_to_cuda(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_peer_transfer_stream(src_node, dst_node);
		if (copy_methods->cuda_to_cuda_async)
			ret = copy_methods->cuda_to_cuda_async(src_interface, src_node, dst_interface, dst_node, stream);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}

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
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_cuda_copy_disabled() || !(copy_methods->cuda_to_ram_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->cuda_to_ram || copy_methods->any_to_any);
		if (copy_methods->cuda_to_ram)
			copy_methods->cuda_to_ram(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess)) STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_out_transfer_stream(src_node);
		if (copy_methods->cuda_to_ram_async)
			ret = copy_methods->cuda_to_ram_async(src_interface, src_node, dst_interface, dst_node, stream);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}

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
	    !(copy_methods->ram_to_cuda_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->ram_to_cuda || copy_methods->any_to_any);
		if (copy_methods->ram_to_cuda)
			copy_methods->ram_to_cuda(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_cuda_node_ops;
		cures = cudaEventCreateWithFlags(_starpu_cuda_event(&req->async_channel.event), cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures != cudaSuccess))
			STARPU_CUDA_REPORT_ERROR(cures);

		stream = starpu_cuda_get_in_transfer_stream(dst_node);
		if (copy_methods->ram_to_cuda_async)
			ret = copy_methods->ram_to_cuda_async(src_interface, src_node, dst_interface, dst_node, stream);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}

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

#ifdef STARPU_USE_CUDA_MAP
uintptr_t _starpu_cuda_map_ram(uintptr_t src_ptr STARPU_ATTRIBUTE_UNUSED, size_t src_offset, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
			       unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
			       size_t size STARPU_ATTRIBUTE_UNUSED, int *ret STARPU_ATTRIBUTE_UNUSED)
{
	/* TODO */
	/*
	 * Old interface:
	 *
	 * cudaHostAllocMapped and cudaHostGetDevicePointer
	 * cudaSetDeviceFlags() must have been called with the cudaDeviceMapHost flag in order
	 * for the cudaHostAllocMapped flag to have any effect.
	 *
	 *
	 *
	 * New interface: Unified Addressing
	 *
	 * Whether or not a device supports unified addressing may be queried
	 * by calling cudaGetDeviceProperties() with the device property
	 * cudaDeviceProp::unifiedAddressing.
	 * Unified addressing is automatically enabled in 64-bit processes.
	 *
	 * Upon enabling direct access from a device that supports unified
	 * addressing to another peer device that supports unified addressing
	 * using cudaDeviceEnablePeerAccess() all memory allocated in the peer
	 * device using cudaMalloc() and cudaMallocPitch() will immediately be
	 * accessible by the current device.
	 */

	*ret = -EIO;

	if (starpu_node_get_kind(src_node) != STARPU_CPU_RAM)
		return 0;

	/*
	 * mapping relevant cudaDeviceProps fields:
	 * - .canMapHostMemory: "Can map host memory with cudaHostAlloc/cudaHostGetDevicePointer"
	 * - .unifiedAddressing: "Device shares a unified address space with the host"
	 * - .managedMemory: "Device supports allocating memory that will be automatically managed by the Unified Memory system"
	 * - .pageableMemoryAccess: "Device supports coherently accessing pageable memory without calling cudaHostRegister on it"
	 * - .concurrentManagedAccess: "Device can coherently access managed memory concurrently with the CPU"
	 */

	struct _starpu_worker *worker = _starpu_get_local_worker_key();
#ifdef STARPU_HAVE_CUDA_CANMAPHOST
	const int cuda_canMapHostMemory = props[worker->devid].canMapHostMemory;
#else
	const int cuda_canMapHostMemory = 0;
#endif

#ifdef STARPU_HAVE_CUDA_UNIFIEDADDR
	const int cuda_unifiedAddressing = props[worker->devid].unifiedAddressing;
#else
	const int cuda_unifiedAddressing = 0;
#endif

#ifdef STARPU_HAVE_CUDA_MNGMEM
	const int cuda_managedMemory = props[worker->devid].managedMemory;
#else
	const int cuda_managedMemory = 0;
#endif

#ifdef STARPU_HAVE_CUDA_PAGEABLEMEM
	const int cuda_pageableMemoryAccess = props[worker->devid].pageableMemoryAccess;
#else
	const int cuda_pageableMemoryAccess = 0;
#endif
	uintptr_t dst_addr;
	if (cuda_pageableMemoryAccess)
	{
		dst_addr = (uintptr_t)(src_ptr+src_offset);
		*ret = 0;
	}
	else if (cuda_unifiedAddressing || cuda_managedMemory)
	{
		struct cudaPointerAttributes cuda_ptrattr;
		cudaError_t cures;
		cures = cudaPointerGetAttributes(&cuda_ptrattr, (void *)(src_ptr+src_offset));
		if (STARPU_UNLIKELY(cures != cudaSuccess))
		{
			if (cures == cudaErrorInvalidValue)
			{
				cudaGetLastError();
				/* pointer does not support mapping */
				return (uintptr_t)NULL;
			}

			STARPU_CUDA_REPORT_ERROR(cures);
		}
#ifdef STARPU_HAVE_CUDA_POINTER_TYPE
		if (!(cuda_ptrattr.type == cudaMemoryTypeHost || cuda_ptrattr.type == cudaMemoryTypeManaged))
			return 0;
#else
		if (!(cuda_ptrattr.memoryType == cudaMemoryTypeHost
#if CUDART_VERSION >= 10000
				|| cuda_ptrattr.memoryType == cudaMemoryTypeManaged
#endif
				))
			return 0;
#endif
		dst_addr = (uintptr_t)cuda_ptrattr.devicePointer;
		*ret = 0;
	}
	else if (cuda_canMapHostMemory)
	{
		cudaError_t cures;
		void *pDevice;
		cures = cudaHostGetDevicePointer(&pDevice, (void*)(src_ptr+src_offset), 0);
		if (STARPU_UNLIKELY(cures != cudaSuccess))
		{
			STARPU_CUDA_REPORT_ERROR(cures);
		}
		dst_addr = (uintptr_t)pDevice;
		*ret = 0;
	}
	else
	{
		dst_addr = (uintptr_t)NULL;
	}
	return dst_addr;
}

int _starpu_cuda_unmap_ram(uintptr_t src_ptr STARPU_ATTRIBUTE_UNUSED, size_t src_offset STARPU_ATTRIBUTE_UNUSED, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
			   uintptr_t dst_ptr STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
			   size_t size STARPU_ATTRIBUTE_UNUSED)
{
#if defined(STARPU_HAVE_CUDA_CANMAPHOST) || defined(STARPU_HAVE_CUDA_UNIFIEDADDR) || defined(STARPU_HAVE_CUDA_MNGMEM)
	/* TODO */
	return 0;
#else
	return -EIO;
#endif
}

int _starpu_cuda_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	(void) src;
	(void) src_offset;
	(void) src_node;
	(void) dst;
	(void) dst_offset;
	(void) dst_node;
	(void) size;

	/* CUDA mappings are coherent */
	/* FIXME: not necessarily, depends on board capabilities */
	return 0;
}

#endif /* STARPU_USE_CUDA_MAP */

#endif /* STARPU_USE_CUDA */

int _starpu_cuda_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	/* GPUs not always allow direct remote access: if CUDA4
	 * is enabled, we allow two CUDA devices to communicate. */
#ifdef STARPU_SIMGRID
	(void) node;
	if (starpu_node_get_kind(handling_node) == STARPU_CUDA_RAM)
	{
		starpu_sg_host_t host = _starpu_simgrid_get_memnode_host(handling_node);
#  ifdef STARPU_HAVE_SIMGRID_ACTOR_H
		const char* cuda_memcpy_peer = sg_host_get_property_value(host, "memcpy_peer");
#  else
		const char* cuda_memcpy_peer = MSG_host_get_property_value(host, "memcpy_peer");
#  endif
		return cuda_memcpy_peer && atoll(cuda_memcpy_peer);
	}
	else
		return 0;
#elif defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
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

static void start_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker, unsigned char pipeline_idx STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();
#if !defined(STARPU_SIMGRID) && defined(STARPU_PROF_TOOL)
	struct starpu_prof_tool_info pi;
#endif

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_local_worker_key(worker);
	_starpu_set_current_task(task);
	j->workerid = worker->workerid;

	if (worker->ntasks == 1)
	{
		/* We are alone in the pipeline, the kernel will start now, record it */
		_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);
	}

#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
	/* We make sure we do manipulate the proper device */
	starpu_cuda_set_device(worker->devid);
#endif

	starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_CUDA is defined in 'where', cuda_func or cuda_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
#ifdef STARPU_SIMGRID
		int async = task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC;
		unsigned workerid = worker->workerid;
		if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE && !async)
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		else if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT && !async)
			{
				_SIMGRID_TIMER_BEGIN(1);
				func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
				_SIMGRID_TIMER_END;
			}
		else
		{
			struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(worker, j);
			_starpu_simgrid_submit_job(workerid, sched_ctx->id, j, &worker->perf_arch, NAN, NAN,
				async ? &task_finished[workerid][pipeline_idx] : NULL);
		}
#else
#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
		unsigned long long energy_start = 0;
		nvmlReturn_t nvmlRet = -1;
		if (profiling && task->profiling_info)
		{
			nvmlRet = nvmlDeviceGetTotalEnergyConsumption(nvmlDev[worker->devid], &energy_start);
			if (nvmlRet == NVML_SUCCESS)
				task->profiling_info->energy_consumed = energy_start / 1000.;
		}
#endif

#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_gpu_exec, worker->devid, worker->workerid, starpu_prof_tool_driver_gpu, -1, (void*)func);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec(&pi, NULL, NULL);
#endif

		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);

#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_gpu_exec, worker->devid, worker->workerid, starpu_prof_tool_driver_gpu, -1, (void*)func);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec(&pi, NULL, NULL);
#endif

#endif
		_STARPU_TRACE_END_EXECUTING();
	}
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker);

/* Execute a job, up to completion for synchronous jobs */
static void execute_job_on_cuda(struct starpu_task *task, struct _starpu_worker *worker)
{
	int workerid = worker->workerid;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	unsigned char pipeline_idx = (worker->first_task + worker->ntasks - 1)%STARPU_MAX_PIPELINE;

	start_job_on_cuda(j, worker, pipeline_idx);

#ifndef STARPU_SIMGRID
	if (!used_stream[workerid])
	{
		used_stream[workerid] = 1;
		_STARPU_DISP("Warning: starpu_cuda_get_local_stream() was not used to submit kernel to CUDA on worker %d. CUDA will thus introduce a lot of useless synchronizations, which will prevent proper overlapping of data transfers and kernel execution. See the CUDA-specific part of the 'Check List When Performance Are Not There' of the StarPU handbook\n", workerid);
	}
#endif

	if (task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC)
	{
		if (worker->pipeline_length == 0)
		{
#ifdef STARPU_SIMGRID
			_starpu_simgrid_wait_tasks(workerid);
#else
			/* Forced synchronous execution */
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
#endif
			finish_job_on_cuda(j, worker);
		}
		else
		{
#ifndef STARPU_SIMGRID
			/* Record event to synchronize with task termination later */
			cudaError_t cures = cudaEventRecord(task_events[workerid][pipeline_idx], starpu_cuda_get_local_stream());
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
#endif
#ifdef STARPU_USE_FXT
			if (fut_active)
			{
				int k;
				for (k = 0; k < (int) worker->set->nworkers; k++)
					if (worker->set->workers[k].ntasks == worker->set->workers[k].pipeline_length)
						break;
				if (k == (int) worker->set->nworkers)
					/* Everybody busy */
					_STARPU_TRACE_START_EXECUTING();
			}
#endif
		}
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


#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
	if (profiling && j->task->profiling_info && j->task->profiling_info->energy_consumed)
	{
		unsigned long long energy_end;
		nvmlReturn_t nvmlRet;
		nvmlRet = nvmlDeviceGetTotalEnergyConsumption(nvmlDev[worker->devid], &energy_end);
#ifdef STARPU_DEVEL
#warning TODO: measure idle consumption to subtract it
#endif
		if (nvmlRet == NVML_SUCCESS)
			j->task->profiling_info->energy_consumed =
				(energy_end / 1000. - j->task->profiling_info->energy_consumed);
	}
#endif
	if (worker->pipeline_length)
		worker->current_tasks[worker->first_task] = NULL;
	else
		worker->current_task = NULL;
	worker->first_task = (worker->first_task + 1) % STARPU_MAX_PIPELINE;
	worker->ntasks--;

	_starpu_driver_end_job(worker, j, &worker->perf_arch, 0, profiling);

	struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(worker, j);
	if(!sched_ctx)
		sched_ctx = _starpu_get_sched_ctx_struct(j->task->sched_ctx);

	if(!sched_ctx->sched_policy)
		_starpu_driver_update_job_feedback(j, worker, &sched_ctx->perf_arch, profiling);
	else
		_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

	_starpu_push_task_output(j);

	_starpu_set_current_task(NULL);

	_starpu_handle_job_termination(j);
}

/* One iteration of the main driver loop */
int _starpu_cuda_driver_run_once(struct _starpu_worker *worker)
{
	struct _starpu_worker_set *worker_set = worker->set;
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	struct starpu_task *tasks[worker_set->nworkers], *task;
	struct _starpu_job *j;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif
	int i, res;

	int idle_tasks, idle_transfers;

#ifdef STARPU_SIMGRID
	starpu_pthread_wait_reset(&worker0->wait);
#endif
	_starpu_set_local_worker_key(worker0);

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
			pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_gpu, memnode, worker->nb_buffers_totransfer, worker->nb_buffers_transferred);
			starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
			j = _starpu_get_job_associated_to_task(task);

			_starpu_set_local_worker_key(worker);
			_starpu_fetch_task_input_tail(task, j, worker);
			/* Reset it */
			worker->task_transferring = NULL;

			if (worker->ntasks > 1 && !(task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC))
			{
				/* We have to execute a non-asynchronous task but we
				 * still have tasks in the pipeline...  Record it to
				 * prevent more tasks from coming, and do it later */
				worker->pipeline_stuck = 1;
			}
			else
			{
				execute_job_on_cuda(task, worker);
			}
			_STARPU_TRACE_START_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, worker->workerid, workerid, starpu_prof_tool_driver_gpu, memnode, worker->nb_buffers_totransfer, worker->nb_buffers_transferred);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
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
#ifdef STARPU_SIMGRID
		if (task_finished[workerid][worker->first_task])
#else /* !STARPU_SIMGRID */
		cudaError_t cures = cudaEventQuery(task_events[workerid][worker->first_task]);

		if (cures != cudaSuccess)
		{
			STARPU_ASSERT_MSG(cures == cudaErrorNotReady, "CUDA error on task %p, codelet %p (%s): %s (%d)", task, task->cl, _starpu_codelet_get_model_name(task->cl), cudaGetErrorString(cures), cures);
		}
		else
#endif /* !STARPU_SIMGRID */
		{
#ifdef STARPU_PROF_TOOL
            pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
            starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
			_STARPU_TRACE_END_PROGRESS(memnode);
			/* Asynchronous task completed! */
			_starpu_set_local_worker_key(worker);
			finish_job_on_cuda(_starpu_get_job_associated_to_task(task), worker);
			/* See next task if any */
			if (worker->ntasks)
			{
				if (worker->current_tasks[worker->first_task] != worker->task_transferring)
				{
					task = worker->current_tasks[worker->first_task];
					j = _starpu_get_job_associated_to_task(task);
					if (task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC)
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
						execute_job_on_cuda(task, worker);
						_STARPU_TRACE_EVENT("end_sync_task");
						worker->pipeline_stuck = 0;
					}
				}
				else
					/* Data for next task didn't have time to finish transferring :/ */
					_STARPU_TRACE_WORKER_START_FETCH_INPUT(NULL, workerid);
			}
#ifdef STARPU_USE_FXT
			if (fut_active)
			{
				int k;
				for (k = 0; k < (int) worker_set->nworkers; k++)
					if (worker_set->workers[k].ntasks)
						break;
				if (k == (int) worker_set->nworkers)
					/* Everybody busy */
					_STARPU_TRACE_END_EXECUTING()
			}
#endif
			_STARPU_TRACE_START_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
            pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_gpu, memnode, NULL);
            starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
            
		}

		if (!worker->pipeline_length || worker->ntasks < worker->pipeline_length)
			idle_tasks++;
	}

#if defined(STARPU_NON_BLOCKING_DRIVERS) && !defined(STARPU_SIMGRID)
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

#ifdef STARPU_SIMGRID
	if (!res)
		starpu_pthread_wait_wait(&worker0->wait);
#endif

	for (i = 0; i < (int) worker_set->nworkers; i++)
	{
		worker = &worker_set->workers[i];
		unsigned memnode STARPU_ATTRIBUTE_UNUSED = worker->memory_node;

		task = tasks[i];
		if (!task)
			continue;


		j = _starpu_get_job_associated_to_task(task);

		/* can CUDA do that task ? */
		if (!_STARPU_MAY_PERFORM(j, CUDA))
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
        //	_STARPU_TRACE_END_PROGRESS(memnode);
	}

	return 0;
}

void *_starpu_cuda_worker(void *_arg)
{
	struct _starpu_worker *worker = _arg;
	struct _starpu_worker_set* worker_set = worker->set;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif
	unsigned i;

	_starpu_cuda_driver_init(worker);
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
		_starpu_cuda_driver_run_once(worker);
	}
	for (i = 0; i < worker_set->nworkers; i++)
	{
		_STARPU_TRACE_END_PROGRESS(worker_set->workers[i].memory_node);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, worker_set->workers[i].workerid, worker_set->workers[i].workerid, starpu_prof_tool_driver_gpu, worker_set->workers[i].memory_node, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif

	}
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

#ifdef STARPU_USE_CUDA
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
#endif /* STARPU_USE_CUDA */

int _starpu_cuda_run_from_worker(struct _starpu_worker *worker)
{
	/* Let's go ! */
	_starpu_cuda_worker(worker);

	return 0;
}

int _starpu_cuda_driver_set_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	driver->id.cuda_id = worker->devid;
	return 0;
}

int _starpu_cuda_driver_is_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	return driver->id.cuda_id == worker->devid;
}

struct _starpu_driver_ops _starpu_driver_cuda_ops =
{
	.init = _starpu_cuda_driver_init,
	.run = _starpu_cuda_run_from_worker,
	.run_once = _starpu_cuda_driver_run_once,
	.deinit = _starpu_cuda_driver_deinit,
	.set_devid = _starpu_cuda_driver_set_devid,
	.is_devid = _starpu_cuda_driver_is_devid,
};

struct _starpu_node_ops _starpu_driver_cuda_node_ops =
{
	.name = "cuda driver",
	.malloc_on_node = _starpu_cuda_malloc_on_node,
	.free_on_node = _starpu_cuda_free_on_node,

	.is_direct_access_supported = _starpu_cuda_is_direct_access_supported,

#ifndef STARPU_SIMGRID
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

#ifdef STARPU_USE_CUDA_MAP
	.map[STARPU_CPU_RAM] = _starpu_cuda_map_ram,
	.unmap[STARPU_CPU_RAM] = _starpu_cuda_unmap_ram,
	.update_map[STARPU_CPU_RAM] = _starpu_cuda_update_map,
#endif

	.wait_request_completion = _starpu_cuda_wait_request_completion,
	.test_request_completion = _starpu_cuda_test_request_completion,
#endif
};
