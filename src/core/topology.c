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

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

/*
 * Discover the topology of the machine
 */

#ifdef USE_CUDA
extern unsigned get_cuda_device_count(void);
#endif

static int current_bindid;
static unsigned get_next_bindid_is_initialized;
static unsigned get_next_bindid_use_envvar;
static char *get_next_bindid_strval;

static int current_gpuid;
static unsigned get_next_gpuid_is_initialized;
static unsigned get_next_gpuid_use_envvar;
static char *get_next_gpuid_strval;

static inline int get_next_gpuid(void)
{
	int gpuid;

	if (!get_next_gpuid_is_initialized)
	{
		char *strval;
		strval = getenv("WORKERS_GPUID");
		if (strval) {
			get_next_gpuid_strval = strval;
			get_next_gpuid_use_envvar = 1;
		}

		get_next_gpuid_is_initialized = 1;
	}
	
	if (get_next_gpuid_use_envvar)
	{
		/* read the value from the WORKERS_GPUID env variable */
		long int val;
		char *endptr;
		val = strtol(get_next_gpuid_strval, &endptr, 10);
		if (endptr != get_next_gpuid_strval)
		{
			gpuid = val;

			get_next_gpuid_strval = endptr;
		}
		else {
			/* there was no valid value so we use a round robin */
			gpuid = current_gpuid++;
		}
	}
	else {
		/* the user did not specify any worker distribution so we use a
 		 * round robin distribution by default */
		gpuid = current_gpuid++;
	}

	return gpuid;
}




static void init_machine_config(struct machine_config_s *config,
				struct starpu_conf *user_conf)
{
	int explicitval __attribute__((unused));
	unsigned use_accelerator = 0;

	config->nworkers = 0;

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

#ifdef USE_CUDA
	if (user_conf && (user_conf->ncuda == 0))
	{
		/* the user explicitely disabled CUDA */
		config->ncudagpus = 0;
	}
	else {
		/* we need to initialize CUDA early to count the number of devices */
		current_gpuid = 0;
		get_next_gpuid_is_initialized = 0;
		get_next_gpuid_use_envvar = 0;
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

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < config->ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = STARPU_CUDA_WORKER;
		/* XXX could be cleaner, we something like STARPU_CUDA_DEFAULT + gpuid */
		int devid = get_next_gpuid();
		enum starpu_perf_archtype arch;
		switch (devid) {
			case 0:
			default:
				arch = STARPU_CUDA_DEFAULT;
				break;
			case 1:
				arch = STARPU_CUDA_2;
				break;
			case 2:
				arch = STARPU_CUDA_3;
				break;
			case 3:
				arch = STARPU_CUDA_4;
				break;
		}
		
		config->workers[config->nworkers + cudagpu].id = devid;
		config->workers[config->nworkers + cudagpu].perf_arch = arch; 
		config->worker_mask |= (CUDA|CUBLAS);
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
		config->worker_mask |= CORE;
	}

	config->nworkers += config->ncores;
#endif


	if (config->nworkers == 0)
	{
		fprintf(stderr, "No worker found, aborting ...\n");
		exit(-1);
	}
}

/*
 * Bind workers on the different processors
 */


static inline int get_next_bindid(struct machine_config_s *config)
{
	int bindid;

	/* do we use a round robin policy to distribute the workers on the
 	 * cores, or do we another distribution ? */
	if (!get_next_bindid_is_initialized)
	{
		char *strval;
		strval = getenv("WORKERS_CPUID");
		if (strval) {
			get_next_bindid_strval = strval;
			get_next_bindid_use_envvar = 1;
		}

		get_next_bindid_is_initialized = 1;
	}
	
	if (get_next_bindid_use_envvar)
	{
		/* read the value from the WORKERS_CPUID env variable */
		long int val;
		char *endptr;
		val = strtol(get_next_bindid_strval, &endptr, 10);
		if (endptr != get_next_bindid_strval)
		{
			bindid = (int)(val % config->nhwcores);

			get_next_bindid_strval = endptr;
		}
		else {
			/* there was no valid value so we use a round robin */
			bindid = (current_bindid++) % config->nhwcores;
		}
	}
	else {
		/* the user did not specify any worker distribution so we use a
 		 * round robin distribution by default */
		bindid = (current_bindid++) % config->nhwcores;
	}

	return bindid;
}



void bind_thread_on_cpu(struct machine_config_s *config __attribute__((unused)), unsigned coreid)
{
	int ret;

#ifdef HAVE_HWLOC
	hwloc_obj_t obj = hwloc_get_obj_by_depth(config->hwtopology, config->core_depth, coreid);
	hwloc_cpuset_t set = obj->cpuset;
	hwloc_cpuset_singlify(&set);
	ret = hwloc_set_cpubind(config->hwtopology, &set, HWLOC_CPUBIND_THREAD);
#elif defined(HAVE_PTHREAD_SETAFFINITY_NP)
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(coreid, &aff_mask);

	pthread_t self = pthread_self();

	ret = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
#else
#warning no CPU binding support
#endif
	if (ret)
	{
		perror("binding thread");
		STARPU_ASSERT(0);
	}
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

	/* we need to reinitialize these value in case StarPU was restarted */
	current_bindid = 0;
	get_next_bindid_is_initialized = 0;
	get_next_bindid_use_envvar = 0;

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		unsigned memory_node = -1;
		unsigned is_a_set_of_accelerators = 0;
		struct worker_s *workerarg = &config->workers[worker];
		
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
				is_a_set_of_accelerators = 0;
				memory_node = register_memory_node(CUDA_RAM);
				break;
#endif
			default:
				STARPU_ASSERT(0);
		}

		if (is_a_set_of_accelerators) {
			if (accelerator_bindid == -1)
				accelerator_bindid = get_next_bindid(config);
			workerarg->bindid = accelerator_bindid;
		}
		else {
			workerarg->bindid = get_next_bindid(config);
		}

		workerarg->memory_node = memory_node;
	}
}


void starpu_build_topology(struct machine_config_s *config,
			   struct starpu_conf *user_conf)
{
	init_machine_config(config, user_conf);

	/* for the data management library */
	init_memory_nodes();

	init_workers_binding(config);
}

void starpu_destroy_topology(struct machine_config_s *config)
{
	/* cleanup StarPU internal data structures */
	deinit_memory_nodes();

#ifdef HAVE_HWLOC
	hwloc_topology_destroy(config->hwtopology);
#endif
}
