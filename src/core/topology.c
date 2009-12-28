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

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif
		
static unsigned topology_is_initialized = 0;

static unsigned may_bind_automatically = 0;

static void initialize_workers_bindid(struct machine_config_s *config);

#ifdef USE_CUDA
static void initialize_workers_gpuid(struct machine_config_s *config);
#endif

/*
 * Discover the topology of the machine
 */

#ifdef USE_CUDA
static void initialize_workers_gpuid(struct machine_config_s *config)
{
	char *strval;
	unsigned i;

	config->current_gpuid = 0;

	/* conf->workers_bindid indicates the successive core identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if (config->user_conf && config->user_conf->use_explicit_workers_gpuid)
	{
		/* we use the explicit value from the user */
		memcpy(config->workers_gpuid,
			config->user_conf->workers_gpuid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else if ((strval = getenv("WORKERS_GPUID")))
	{
		/* WORKERS_GPUID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round robin
		 * fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the WORKERS_GPUID env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap) {
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					config->workers_gpuid[i] = (unsigned)val;
					strval = endptr;
				}
				else {
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;
	
					/* there is no more values in the string */
					wrap = 1;

					config->workers_gpuid[i] = config->workers_gpuid[0];
				}
			}
			else {
				config->workers_gpuid[i] = config->workers_gpuid[i % number_of_entries];
			}
		}
	}
	else
	{
		/* by default, we take a round robin policy */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
			config->workers_gpuid[i] = (unsigned)i;

		/* StarPU can use sampling techniques to bind threads correctly */
		may_bind_automatically = 1;
	}
}
#endif

static inline int get_next_gpuid(struct machine_config_s *config)
{
	unsigned i = ((config->current_gpuid++) % config->ncudagpus);

	return (int)config->workers_gpuid[i];
}

static void init_topology(struct machine_config_s *config)
{
	if (!topology_is_initialized)
	{
#ifdef HAVE_HWLOC
		hwloc_topology_init(&config->hwtopology);
		hwloc_topology_load(config->hwtopology);

		config->core_depth = hwloc_get_type_depth(config->hwtopology, HWLOC_OBJ_CORE);

		/* Would be very odd */
		STARPU_ASSERT(config->core_depth != HWLOC_TYPE_DEPTH_MULTIPLE);

		if (config->core_depth == HWLOC_TYPE_DEPTH_UNKNOWN)
			/* unknown, using logical procesors as fallback */
			config->core_depth = hwloc_get_type_depth(config->hwtopology, HWLOC_OBJ_PROC);

		config->nhwcores = hwloc_get_nbobjs_by_depth(config->hwtopology, config->core_depth);
#else
		config->nhwcores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
	
		topology_is_initialized = 1;
	}
}

unsigned topology_get_nhwcore(struct machine_config_s *config)
{
	init_topology(config);
	
	return config->nhwcores;
}

static int init_machine_config(struct machine_config_s *config,
				struct starpu_conf *user_conf)
{
	int explicitval __attribute__((unused));
	unsigned use_accelerator = 0;

	config->nworkers = 0;

	init_topology(config);

	initialize_workers_bindid(config);

#ifdef USE_CUDA
	if (user_conf && (user_conf->ncuda == 0))
	{
		/* the user explicitely disabled CUDA */
		config->ncudagpus = 0;
	}
	else {
		/* we need to initialize CUDA early to count the number of devices */
		init_cuda();

		if (user_conf && (user_conf->ncuda != -1))
		{
			explicitval = user_conf->ncuda;
		}
		else {
			explicitval = starpu_get_env_number("NCUDA");
		}

		if (explicitval < 0) {
			config->ncudagpus =
				STARPU_MIN(get_cuda_device_count(), MAXCUDADEVS);
		} else {
			/* use the specified value */
			config->ncudagpus = (unsigned)explicitval;
			STARPU_ASSERT(config->ncudagpus <= MAXCUDADEVS);
		}
		STARPU_ASSERT(config->ncudagpus + config->nworkers <= STARPU_NMAXWORKERS);
	}

	if (config->ncudagpus > 0)
		use_accelerator = 1;

	initialize_workers_gpuid(config);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < config->ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = STARPU_CUDA_WORKER;
		int devid = get_next_gpuid(config);
		enum starpu_perf_archtype arch = STARPU_CUDA_DEFAULT + devid;
		config->workers[config->nworkers + cudagpu].id = devid;
		config->workers[config->nworkers + cudagpu].perf_arch = arch; 
		config->workers[config->nworkers + cudagpu].worker_mask = CUDA;
		config->worker_mask |= CUDA;
	}

	config->nworkers += config->ncudagpus;
#endif
	
#ifdef USE_GORDON
	if (user_conf && (user_conf->ncuda != -1)) {
		explicitval = user_conf->ncuda;
	}
	else {
		explicitval = starpu_get_env_number("NGORDON");
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
		config->workers[config->nworkers + spu].worker_mask = GORDON;
		config->worker_mask |= GORDON;
	}

	config->nworkers += config->ngordon_spus;
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one core */
#ifdef USE_CPUS
	if (user_conf && (user_conf->ncpus != -1)) {
		explicitval = user_conf->ncpus;
	}
	else {
		explicitval = starpu_get_env_number("NCPUS");
	}

	if (explicitval < 0) {
		unsigned already_busy_cores = (config->ngordon_spus?1:0) + config->ncudagpus;
		long avail_cores = config->nhwcores - (use_accelerator?already_busy_cores:0);
		config->ncores = STARPU_MIN(avail_cores, NMAXCORES);
	} else {
		/* use the specified value */
		config->ncores = (unsigned)explicitval;
		STARPU_ASSERT(config->ncores <= NMAXCORES);
	}
	STARPU_ASSERT(config->ncores + config->nworkers <= STARPU_NMAXWORKERS);

	unsigned core;
	for (core = 0; core < config->ncores; core++)
	{
		config->workers[config->nworkers + core].arch = STARPU_CORE_WORKER;
		config->workers[config->nworkers + core].perf_arch = STARPU_CORE_DEFAULT;
		config->workers[config->nworkers + core].id = core;
		config->workers[config->nworkers + core].worker_mask = CORE;
		config->worker_mask |= CORE;
	}

	config->nworkers += config->ncores;
#endif

	if (config->nworkers == 0)
	{
#ifdef VERBOSE
		fprintf(stderr, "No worker found, aborting ...\n");
#endif
		return -ENODEV;
	}

	return 0;
}

/*
 * Bind workers on the different processors
 */
static void initialize_workers_bindid(struct machine_config_s *config)
{
	char *strval;
	unsigned i;

	config->current_bindid = 0;

	/* conf->workers_bindid indicates the successive core identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if (config->user_conf && config->user_conf->use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(config->workers_bindid,
			config->user_conf->workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else if ((strval = getenv("WORKERS_CPUID")))
	{
		/* WORKERS_CPUID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round robin
		 * fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the WORKERS_GPUID env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap) {
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					config->workers_bindid[i] = (unsigned)(val % config->nhwcores);
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
			config->workers_bindid[i] = (unsigned)(i % config->nhwcores);
	}
}

/* This function gets the identifier of the next core on which to bind a
 * worker. In case a list of preferred cores was specified, we look for a an
 * available core among the list if possible, otherwise a round-robin policy is
 * used. */
static inline int get_next_bindid(struct machine_config_s *config,
				int *preferred_binding, int npreferred)
{
	unsigned found = 0;
	int current_preferred;

	for (current_preferred = 0; current_preferred < npreferred; current_preferred++)
	{
		if (found)
			break;

		unsigned requested_core = preferred_binding[current_preferred];

		/* can we bind the worker on the requested core ? */
		unsigned ind;
		for (ind = config->current_bindid; ind < config->nhwcores; ind++)
		{
			if (config->workers_bindid[ind] == requested_core)
			{
				/* the core is available, we  use it ! In order
				 * to make sure that it will not be used again
				 * later on, we remove the entry from the list
				 * */
				config->workers_bindid[ind] =
					config->workers_bindid[config->current_bindid];
				config->workers_bindid[config->current_bindid] = requested_core;

				found = 1;

				break;
			}
		}
	}

	unsigned i = ((config->current_bindid++) % STARPU_NMAXWORKERS);

	return (int)config->workers_bindid[i];
}

void bind_thread_on_cpu(struct machine_config_s *config __attribute__((unused)), unsigned coreid)
{
	int ret;

#ifdef HAVE_HWLOC
	init_topology(config);

	hwloc_obj_t obj = hwloc_get_obj_by_depth(config->hwtopology, config->core_depth, coreid);
	hwloc_cpuset_t set = obj->cpuset;
	hwloc_cpuset_singlify(set);
	ret = hwloc_set_cpubind(config->hwtopology, set, HWLOC_CPUBIND_THREAD);
	if (ret)
	{
		perror("binding thread");
		STARPU_ABORT();
	}

#elif defined(HAVE_PTHREAD_SETAFFINITY_NP)
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(coreid, &aff_mask);

	pthread_t self = pthread_self();

	ret = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
	if (ret)
	{
		perror("binding thread");
		STARPU_ABORT();
	}

#else
#warning no CPU binding support
#endif
}

static void init_workers_binding(struct machine_config_s *config)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	/* a single core is dedicated for the accelerators */
	int accelerator_bindid = -1;

	/* note that even if the CPU core are not used, we always have a RAM node */
	/* TODO : support NUMA  ;) */
	ram_memory_node = register_memory_node(RAM);

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		unsigned memory_node = -1;
		unsigned is_a_set_of_accelerators = 0;
		struct worker_s *workerarg = &config->workers[worker];

		/* Perhaps the worker has some "favourite" bindings  */
		int *preferred_binding = NULL;
		int npreferred = 0;
		
		/* select the memory node that contains worker's memory */
		switch (workerarg->arch) {
			case STARPU_CORE_WORKER:
			/* "dedicate" a cpu core to that worker */
				is_a_set_of_accelerators = 0;
				memory_node = ram_memory_node;
				break;
#ifdef USE_GORDON
			case STARPU_GORDON_WORKER:
				is_a_set_of_accelerators = 1;
				memory_node = ram_memory_node;
				break;
#endif
#ifdef USE_CUDA
			case STARPU_CUDA_WORKER:
				if (may_bind_automatically)
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = get_gpu_affinity_vector(workerarg->id);
					npreferred = config->nhwcores;
				}
				is_a_set_of_accelerators = 0;
				memory_node = register_memory_node(CUDA_RAM);
				break;
#endif
			default:
				STARPU_ABORT();
		}

		if (is_a_set_of_accelerators) {
			if (accelerator_bindid == -1)
				accelerator_bindid = get_next_bindid(config, preferred_binding, npreferred);

			workerarg->bindid = accelerator_bindid;
		}
		else {
			workerarg->bindid = get_next_bindid(config, preferred_binding, npreferred);
		}

		workerarg->memory_node = memory_node;
	}
}


int starpu_build_topology(struct machine_config_s *config)
{
	int ret;

	struct starpu_conf *user_conf = config->user_conf;

	ret = init_machine_config(config, user_conf);
	if (ret)
		return ret;

	/* for the data management library */
	init_memory_nodes();

	init_workers_binding(config);

	return 0;
}

void starpu_destroy_topology(struct machine_config_s *config __attribute__ ((unused)))
{
	/* cleanup StarPU internal data structures */
	deinit_memory_nodes();

#ifdef HAVE_HWLOC
	hwloc_topology_destroy(config->hwtopology);
#endif

	topology_is_initialized = 0;
}
