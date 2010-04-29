/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdlib.h>
#include <stdio.h>
#include <common/config.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <common/hash.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#endif

#if defined(__MINGW32__) || defined(__CYGWIN__)
#include <windows.h>
#endif
		
static unsigned topology_is_initialized = 0;

static void _starpu_initialize_workers_bindid(struct starpu_machine_config_s *config);

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
#  ifdef STARPU_USE_CUDA
static void _starpu_initialize_workers_cuda_gpuid(struct starpu_machine_config_s *config);
static struct starpu_htbl32_node_s *devices_using_cuda = NULL;
#  endif
#  ifdef STARPU_USE_OPENCL
static void _starpu_initialize_workers_opencl_gpuid(struct starpu_machine_config_s *config);
#  endif
static void _starpu_initialize_workers_gpuid(int use_explicit_workers_gpuid, int *explicit_workers_gpuid,
                                             int *current, int *workers_gpuid, const char *varname, unsigned nhwgpus);
static unsigned may_bind_automatically = 0;
#endif

/*
 * Discover the topology of the machine
 */

#ifdef STARPU_USE_CUDA
static void _starpu_initialize_workers_cuda_gpuid(struct starpu_machine_config_s *config)
{
        _starpu_initialize_workers_gpuid(config->user_conf==NULL?0:config->user_conf->use_explicit_workers_cuda_gpuid,
                                         config->user_conf==NULL?NULL:(int *)config->user_conf->workers_cuda_gpuid,
                                         &(config->current_cuda_gpuid), (int *)config->workers_cuda_gpuid, "STARPU_WORKERS_CUDAID",
                                         config->nhwcudagpus);
}
#endif

#ifdef STARPU_USE_OPENCL
static void _starpu_initialize_workers_opencl_gpuid(struct starpu_machine_config_s *config)
{
        _starpu_initialize_workers_gpuid(config->user_conf==NULL?0:config->user_conf->use_explicit_workers_opencl_gpuid,
                                         config->user_conf==NULL?NULL:(int *)config->user_conf->workers_opencl_gpuid,
                                         &(config->current_opencl_gpuid), (int *)config->workers_opencl_gpuid, "STARPU_WORKERS_OPENCLID",
                                         config->nhwopenclgpus);

#ifdef STARPU_USE_CUDA
        // Detect devices which are already used with CUDA
        {
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;
                for(i=0 ; i<STARPU_NMAXWORKERS ; i++) {
                        uint32_t key = _starpu_crc32_be(config->workers_opencl_gpuid[i], 0);
                        if (_starpu_htbl_search_32(devices_using_cuda, key) == NULL) {
                                tmp[nb] = config->workers_opencl_gpuid[i];
                                nb++;
                        }
                }
                for(i=nb ; i<STARPU_NMAXWORKERS ; i++) tmp[i] = -1;
                memcpy(config->workers_opencl_gpuid, tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
#endif /* STARPU_USE_CUDA */
        {
                // Detect identical devices
                struct starpu_htbl32_node_s *devices_already_used = NULL;
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;

                for(i=0 ; i<STARPU_NMAXWORKERS ; i++) {
                        uint32_t key = _starpu_crc32_be(config->workers_opencl_gpuid[i], 0);
                        if (_starpu_htbl_search_32(devices_already_used, key) == NULL) {
                                _starpu_htbl_insert_32(&devices_already_used, key, config);
                                tmp[nb] = config->workers_opencl_gpuid[i];
                                nb ++;
                        }
                }
                for(i=nb ; i<STARPU_NMAXWORKERS ; i++) tmp[i] = -1;
                memcpy(config->workers_opencl_gpuid, tmp, sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
}
#endif


#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static void _starpu_initialize_workers_gpuid(int use_explicit_workers_gpuid, int *explicit_workers_gpuid,
                                             int *current, int *workers_gpuid, const char *varname, unsigned nhwgpus)
{
	char *strval;
	unsigned i;

	*current = 0;

	/* conf->workers_bindid indicates the successive cpu identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cpus. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if (use_explicit_workers_gpuid)
	{
		/* we use the explicit value from the user */
		memcpy(workers_gpuid,
                       explicit_workers_gpuid,
                       STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else if ((strval = getenv(varname)))
	{
		/* STARPU_WORKERS_CUDAID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round robin
		 * fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CUDAID env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap) {
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					workers_gpuid[i] = (unsigned)val;
					strval = endptr;
				}
				else {
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;
	
					/* there is no more values in the string */
					wrap = 1;

					workers_gpuid[i] = workers_gpuid[0];
				}
			}
			else {
				workers_gpuid[i] = workers_gpuid[i % number_of_entries];
			}
		}
	}
	else
	{
		/* by default, we take a round robin policy */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
			workers_gpuid[i] = (unsigned)(i % nhwgpus);

		/* StarPU can use sampling techniques to bind threads correctly */
		may_bind_automatically = 1;
	}
}
#endif

static inline int _starpu_get_next_cuda_gpuid(struct starpu_machine_config_s *config)
{
	unsigned i = ((config->current_cuda_gpuid++) % config->ncudagpus);

	return (int)config->workers_cuda_gpuid[i];
}

static inline int _starpu_get_next_opencl_gpuid(struct starpu_machine_config_s *config)
{
	unsigned i = ((config->current_opencl_gpuid++) % config->nopenclgpus);

	return (int)config->workers_opencl_gpuid[i];
}

static void _starpu_init_topology(struct starpu_machine_config_s *config)
{
	if (!topology_is_initialized)
	{
#ifdef STARPU_HAVE_HWLOC
		hwloc_topology_init(&config->hwtopology);
		hwloc_topology_load(config->hwtopology);

		config->cpu_depth = hwloc_get_type_depth(config->hwtopology, HWLOC_OBJ_CORE);

		/* Would be very odd */
		STARPU_ASSERT(config->cpu_depth != HWLOC_TYPE_DEPTH_MULTIPLE);

		if (config->cpu_depth == HWLOC_TYPE_DEPTH_UNKNOWN)
			/* unknown, using logical procesors as fallback */
			config->cpu_depth = hwloc_get_type_depth(config->hwtopology, HWLOC_OBJ_PU);

		config->nhwcpus = hwloc_get_nbobjs_by_depth(config->hwtopology, config->cpu_depth);
#elif defined(__MINGW32__) || defined(__CYGWIN__)
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		config->nhwcpus += sysinfo.dwNumberOfProcessors;
#elif defined(HAVE_SYSCONF)
		config->nhwcpus = sysconf(_SC_NPROCESSORS_ONLN);
#else
#warning no way to know number of cores, assuming 1
		config->nhwcpus = 1;
#endif

#ifdef STARPU_USE_CUDA
                config->nhwcudagpus = _starpu_get_cuda_device_count();
#endif
#ifdef STARPU_USE_OPENCL
                config->nhwopenclgpus = _starpu_opencl_get_device_count();
#endif

		topology_is_initialized = 1;
	}
}

unsigned _starpu_topology_get_nhwcpu(struct starpu_machine_config_s *config)
{
	_starpu_init_topology(config);
	
	return config->nhwcpus;
}

static int _starpu_init_machine_config(struct starpu_machine_config_s *config,
				struct starpu_conf *user_conf)
{
	int explicitval __attribute__((unused));
	unsigned use_accelerator = 0;

	config->nworkers = 0;

	_starpu_init_topology(config);

	_starpu_initialize_workers_bindid(config);

#ifdef STARPU_USE_CUDA
	if (user_conf && (user_conf->ncuda == 0))
	{
		/* the user explicitely disabled CUDA */
		config->ncudagpus = 0;
	}
	else {
		/* we need to initialize CUDA early to count the number of devices */
		_starpu_init_cuda();

		if (user_conf && (user_conf->ncuda != -1))
		{
			explicitval = user_conf->ncuda;
		}
		else {
			explicitval = starpu_get_env_number("STARPU_NCUDA");
		}

		if (explicitval < 0) {
			config->ncudagpus =
				STARPU_MIN(_starpu_get_cuda_device_count(), STARPU_MAXCUDADEVS);
		} else {
			/* use the specified value */
			config->ncudagpus = (unsigned)explicitval;
			STARPU_ASSERT(config->ncudagpus <= STARPU_MAXCUDADEVS);
		}
		STARPU_ASSERT(config->ncudagpus + config->nworkers <= STARPU_NMAXWORKERS);
	}

	if (config->ncudagpus > 0)
		use_accelerator = 1;

	_starpu_initialize_workers_cuda_gpuid(config);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < config->ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = STARPU_CUDA_WORKER;
		int devid = _starpu_get_next_cuda_gpuid(config);
		enum starpu_perf_archtype arch = STARPU_CUDA_DEFAULT + devid;
		config->workers[config->nworkers + cudagpu].devid = devid;
		config->workers[config->nworkers + cudagpu].perf_arch = arch; 
		config->workers[config->nworkers + cudagpu].worker_mask = STARPU_CUDA;
		config->worker_mask |= STARPU_CUDA;

                uint32_t key = _starpu_crc32_be(devid, 0);
                _starpu_htbl_insert_32(&devices_using_cuda, key, config);
        }

	config->nworkers += config->ncudagpus;
#endif

#ifdef STARPU_USE_OPENCL
	if (user_conf && (user_conf->nopencl == 0))
	{
		/* the user explicitely disabled OpenCL */
		config->nopenclgpus = 0;
	}
	else {
		/* we need to initialize OpenCL early to count the number of devices */
		_starpu_opencl_init();

		if (user_conf && (user_conf->nopencl != -1))
		{
			explicitval = user_conf->nopencl;
		}
		else {
			explicitval = starpu_get_env_number("STARPU_NOPENCL");
		}

		if (explicitval < 0) {
			config->nopenclgpus =
				STARPU_MIN(_starpu_opencl_get_device_count(), STARPU_MAXOPENCLDEVS);
		} else {
			/* use the specified value */
			config->nopenclgpus = (unsigned)explicitval;
			STARPU_ASSERT(config->nopenclgpus <= STARPU_MAXOPENCLDEVS);
		}
		STARPU_ASSERT(config->nopenclgpus + config->nworkers <= STARPU_NMAXWORKERS);
	}

	if (config->nopenclgpus > 0)
		use_accelerator = 1;
	// TODO: use_accelerator pour les OpenCL?

	_starpu_initialize_workers_opencl_gpuid(config);

	unsigned openclgpu;
	for (openclgpu = 0; openclgpu < config->nopenclgpus; openclgpu++)
	{
		int devid = _starpu_get_next_opencl_gpuid(config);
                if (devid == -1) { // There is no more devices left
                  config->nopenclgpus = openclgpu;
                  break;
                }
		config->workers[config->nworkers + openclgpu].arch = STARPU_OPENCL_WORKER;
		enum starpu_perf_archtype arch = STARPU_OPENCL_DEFAULT + devid;
		config->workers[config->nworkers + openclgpu].devid = devid;
		config->workers[config->nworkers + openclgpu].perf_arch = arch; 
		config->workers[config->nworkers + openclgpu].worker_mask = STARPU_OPENCL;
		config->worker_mask |= STARPU_OPENCL;
	}

	config->nworkers += config->nopenclgpus;
#endif
	
#ifdef STARPU_USE_GORDON
	if (user_conf && (user_conf->ncuda != -1)) {
		explicitval = user_conf->ncuda;
	}
	else {
		explicitval = starpu_get_env_number("STARPU_NGORDON");
	}

	if (explicitval < 0) {
		config->ngordon_spus = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	} else {
		/* use the specified value */
		config->ngordon_spus = (unsigned)explicitval;
		STARPU_ASSERT(config->ngordon_spus <= NMAXGORDONSPUS);
	}
	STARPU_ASSERT(config->ngordon_spus + config->nworkers <= STARPU_NMAXWORKERS);

	if (config->ngordon_spus > 0)
		use_accelerator = 1;

	unsigned spu;
	for (spu = 0; spu < config->ngordon_spus; spu++)
	{
		config->workers[config->nworkers + spu].arch = STARPU_GORDON_WORKER;
		config->workers[config->nworkers + spu].perf_arch = STARPU_GORDON_DEFAULT;
		config->workers[config->nworkers + spu].id = spu;
		config->workers[config->nworkers + spu].worker_is_running = 0;
		config->workers[config->nworkers + spu].worker_mask = STARPU_GORDON;
		config->worker_mask |= STARPU_GORDON;
	}

	config->nworkers += config->ngordon_spus;
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one cpu */
#ifdef STARPU_USE_CPU
	if (user_conf && (user_conf->ncpus != -1)) {
		explicitval = user_conf->ncpus;
	}
	else {
		explicitval = starpu_get_env_number("STARPU_NCPUS");
	}

	if (explicitval < 0) {
		unsigned already_busy_cpus = (config->ngordon_spus?1:0) + config->ncudagpus;
		long avail_cpus = config->nhwcpus - (use_accelerator?already_busy_cpus:0);
		config->ncpus = STARPU_MIN(avail_cpus, STARPU_NMAXCPUS);
	} else {
		/* use the specified value */
		config->ncpus = (unsigned)explicitval;
		STARPU_ASSERT(config->ncpus <= STARPU_NMAXCPUS);
	}
	STARPU_ASSERT(config->ncpus + config->nworkers <= STARPU_NMAXWORKERS);

	unsigned cpu;
	for (cpu = 0; cpu < config->ncpus; cpu++)
	{
		config->workers[config->nworkers + cpu].arch = STARPU_CPU_WORKER;
		config->workers[config->nworkers + cpu].perf_arch = STARPU_CPU_DEFAULT;
		config->workers[config->nworkers + cpu].devid = cpu;
		config->workers[config->nworkers + cpu].worker_mask = STARPU_CPU;
		config->worker_mask |= STARPU_CPU;
	}

	config->nworkers += config->ncpus;
#endif

	if (config->nworkers == 0)
	{
#ifdef STARPU_VERBOSE
		fprintf(stderr, "No worker found, aborting ...\n");
#endif
		return -ENODEV;
	}

	return 0;
}

/*
 * Bind workers on the different processors
 */
static void _starpu_initialize_workers_bindid(struct starpu_machine_config_s *config)
{
	char *strval;
	unsigned i;

	config->current_bindid = 0;

	/* conf->workers_bindid indicates the successive cpu identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cpus. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if (config->user_conf && config->user_conf->use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(config->workers_bindid,
			config->user_conf->workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else if ((strval = getenv("STARPU_WORKERS_CPUID")))
	{
		/* STARPU_WORKERS_CPUID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round robin
		 * fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CUDAID env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap) {
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					config->workers_bindid[i] = (unsigned)(val % config->nhwcpus);
					strval = endptr;
				}
				else {
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;

					/* there is no more values in the string */
					wrap = 1;

					config->workers_bindid[i] = config->workers_bindid[0];
				}
			}
			else {
				config->workers_bindid[i] = config->workers_bindid[i % number_of_entries];
			}
		}
	}
	else
	{
		/* by default, we take a round robin policy */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
			config->workers_bindid[i] = (unsigned)(i % config->nhwcpus);
	}
}

/* This function gets the identifier of the next cpu on which to bind a
 * worker. In case a list of preferred cpus was specified, we look for a an
 * available cpu among the list if possible, otherwise a round-robin policy is
 * used. */
static inline int _starpu_get_next_bindid(struct starpu_machine_config_s *config,
				int *preferred_binding, int npreferred)
{
	unsigned found = 0;
	int current_preferred;

	for (current_preferred = 0; current_preferred < npreferred; current_preferred++)
	{
		if (found)
			break;

		unsigned requested_cpu = preferred_binding[current_preferred];

		/* can we bind the worker on the requested cpu ? */
		unsigned ind;
		for (ind = config->current_bindid; ind < config->nhwcpus; ind++)
		{
			if (config->workers_bindid[ind] == requested_cpu)
			{
				/* the cpu is available, we  use it ! In order
				 * to make sure that it will not be used again
				 * later on, we remove the entry from the list
				 * */
				config->workers_bindid[ind] =
					config->workers_bindid[config->current_bindid];
				config->workers_bindid[config->current_bindid] = requested_cpu;

				found = 1;

				break;
			}
		}
	}

	unsigned i = ((config->current_bindid++) % STARPU_NMAXWORKERS);

	return (int)config->workers_bindid[i];
}

void _starpu_bind_thread_on_cpu(struct starpu_machine_config_s *config __attribute__((unused)), unsigned cpuid)
{
#ifdef STARPU_HAVE_HWLOC
	int ret;
	_starpu_init_topology(config);

	hwloc_obj_t obj = hwloc_get_obj_by_depth(config->hwtopology, config->cpu_depth, cpuid);
	hwloc_cpuset_t set = obj->cpuset;
	hwloc_cpuset_singlify(set);
	ret = hwloc_set_cpubind(config->hwtopology, set, HWLOC_CPUBIND_THREAD);
	if (ret)
	{
		perror("binding thread");
		STARPU_ABORT();
	}

#elif defined(HAVE_PTHREAD_SETAFFINITY_NP)
	int ret;
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(cpuid, &aff_mask);

	pthread_t self = pthread_self();

	ret = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
	if (ret)
	{
		perror("binding thread");
		STARPU_ABORT();
	}

#elif defined(__MINGW32__) || defined(__CYGWIN__)
	DWORD mask = 1 << cpuid;
	if (!SetThreadAffinityMask(GetCurrentThread(), mask)) {
		fprintf(stderr,"SetThreadMaskAffinity(%lx) failed\n", mask);
		STARPU_ABORT();
	}
#else
#warning no CPU binding support
#endif
}

static void _starpu_init_workers_binding(struct starpu_machine_config_s *config)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	/* a single cpu is dedicated for the accelerators */
	int accelerator_bindid = -1;

	/* note that even if the CPU cpu are not used, we always have a RAM node */
	/* TODO : support NUMA  ;) */
	ram_memory_node = _starpu_register_memory_node(STARPU_RAM);

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		unsigned memory_node = -1;
		unsigned is_a_set_of_accelerators = 0;
		struct starpu_worker_s *workerarg = &config->workers[worker];

		/* Perhaps the worker has some "favourite" bindings  */
		int *preferred_binding = NULL;
		int npreferred = 0;
		
		/* select the memory node that contains worker's memory */
		switch (workerarg->arch) {
			case STARPU_CPU_WORKER:
			/* "dedicate" a cpu cpu to that worker */
				is_a_set_of_accelerators = 0;
				memory_node = ram_memory_node;
				break;
#ifdef STARPU_USE_GORDON
			case STARPU_GORDON_WORKER:
				is_a_set_of_accelerators = 1;
				memory_node = ram_memory_node;
				break;
#endif
#ifdef STARPU_USE_CUDA
			case STARPU_CUDA_WORKER:
				if (may_bind_automatically)
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_cuda_affinity_vector(workerarg->devid);
					npreferred = config->nhwcpus;
				}
				is_a_set_of_accelerators = 0;
				memory_node = _starpu_register_memory_node(STARPU_CUDA_RAM);
				break;
#endif

#ifdef STARPU_USE_OPENCL
		        case STARPU_OPENCL_WORKER:
				if (may_bind_automatically)
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_opencl_affinity_vector(workerarg->devid);
					npreferred = config->nhwcpus;
				}
				is_a_set_of_accelerators = 0;
				memory_node = _starpu_register_memory_node(STARPU_OPENCL_RAM);
				break;
#endif

			default:
				STARPU_ABORT();
		}

		if (is_a_set_of_accelerators) {
			if (accelerator_bindid == -1)
				accelerator_bindid = _starpu_get_next_bindid(config, preferred_binding, npreferred);

			workerarg->bindid = accelerator_bindid;
		}
		else {
			workerarg->bindid = _starpu_get_next_bindid(config, preferred_binding, npreferred);
		}

		workerarg->memory_node = memory_node;
	}
}


int _starpu_build_topology(struct starpu_machine_config_s *config)
{
	int ret;

	struct starpu_conf *user_conf = config->user_conf;

	ret = _starpu_init_machine_config(config, user_conf);
	if (ret)
		return ret;

	/* for the data management library */
	_starpu_init_memory_nodes();

	_starpu_init_workers_binding(config);

	return 0;
}

void _starpu_destroy_topology(struct starpu_machine_config_s *config __attribute__ ((unused)))
{
	/* cleanup StarPU internal data structures */
	_starpu_deinit_memory_nodes();

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_destroy(config->hwtopology);
#endif

	topology_is_initialized = 0;
}
