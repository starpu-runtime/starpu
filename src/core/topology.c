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

/*
 * Discover the topology of the machine
 */


#ifdef USE_CPUS
static unsigned ncores;
#endif
#ifdef USE_CUDA
static unsigned ncudagpus;
#endif
#ifdef USE_GORDON
static unsigned ngordon_spus;
#endif

#ifdef USE_CUDA
extern unsigned get_cuda_device_count(void);
#endif

static void init_machine_config(struct machine_config_s *config,
				struct starpu_conf *user_conf)
{
	int explicitval __attribute__((unused));
	unsigned use_accelerator = 0;

	config->nworkers = 0;

#ifdef USE_CUDA
	if (user_conf && (user_conf->ncuda == 0))
	{
		/* the user explicitely disabled CUDA */
		ncudagpus = 0;
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
			ncudagpus = STARPU_MIN(get_cuda_device_count(), MAXCUDADEVS);
		} else {
			/* use the specified value */
			ncudagpus = (unsigned)explicitval;
			STARPU_ASSERT(ncudagpus <= MAXCUDADEVS);
		}
		STARPU_ASSERT(ncudagpus + config->nworkers <= NMAXWORKERS);
	}

	if (ncudagpus > 0)
		use_accelerator = 1;

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < ncudagpus; cudagpu++)
	{
		config->workers[config->nworkers + cudagpu].arch = CUDA_WORKER;
		config->workers[config->nworkers + cudagpu].perf_arch = STARPU_CUDA_DEFAULT;
		config->workers[config->nworkers + cudagpu].id = cudagpu;
		config->worker_mask |= (CUDA|CUBLAS);
	}

	config->nworkers += ncudagpus;
#endif
	
#ifdef USE_GORDON
	if (user_conf && (user_conf->ncuda != -1)) {
		explicitval = user_conf->ncuda;
	}
	else {
		explicitval = starpu_get_env_number("NGORDON");
	}

	if (explicitval < 0) {
		ngordon_spus = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
	} else {
		/* use the specified value */
		ngordon_spus = (unsigned)explicitval;
		STARPU_ASSERT(ngordon_spus <= NMAXGORDONSPUS);
	}
	STARPU_ASSERT(ngordon_spus + config->nworkers <= NMAXWORKERS);

	if (ngordon_spus > 0)
		use_accelerator = 1;

	unsigned spu;
	for (spu = 0; spu < ngordon_spus; spu++)
	{
		config->workers[config->nworkers + spu].arch = GORDON_WORKER;
		config->workers[config->nworkers + spu].perf_arch = STARPU_GORDON_DEFAULT;
		config->workers[config->nworkers + spu].id = spu;
		config->workers[config->nworkers + spu].worker_is_running = 0;
		config->worker_mask |= GORDON;
	}

	config->nworkers += ngordon_spus;
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
		long avail_cores = sysconf(_SC_NPROCESSORS_ONLN) 
						- (use_accelerator?1:0);
		ncores = STARPU_MIN(avail_cores, NMAXCORES);
	} else {
		/* use the specified value */
		ncores = (unsigned)explicitval;
		STARPU_ASSERT(ncores <= NMAXCORES);
	}
	STARPU_ASSERT(ncores + config->nworkers <= NMAXWORKERS);

	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		config->workers[config->nworkers + core].arch = CORE_WORKER;
		config->workers[config->nworkers + core].perf_arch = STARPU_CORE_DEFAULT;
		config->workers[config->nworkers + core].id = core;
		config->worker_mask |= CORE;
	}

	config->nworkers += ncores;
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

static int current_bindid = 0;
static unsigned get_next_bindid_is_initialized = 0;
static unsigned get_next_bindid_use_envvar = 0;
static char *get_next_bindid_strval;

static inline int get_next_bindid(void)
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
			bindid = (int)(val % sysconf(_SC_NPROCESSORS_ONLN));

			get_next_bindid_strval = endptr;
		}
		else {
			/* there was no valid value so we use a round robin */
			bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
		}
	}
	else {
		/* the user did not specify any worker distribution so we use a
 		 * round robin distribution by default */
		bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
	}

	return bindid;
}



void bind_thread_on_cpu(unsigned coreid)
{
#ifndef DONTBIND
	int ret;

	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(coreid, &aff_mask);

	pthread_t self = pthread_self();

	ret = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
	if (ret)
	{
		perror("pthread_setaffinity_np");
		STARPU_ASSERT(0);
	}
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
		unsigned is_an_accelerator = 0;
		struct worker_s *workerarg = &config->workers[worker];
		
		/* select the memory node that contains worker's memory */
		switch (workerarg->arch) {
			case CORE_WORKER:
			/* "dedicate" a cpu core to that worker */
				is_an_accelerator = 0;
				memory_node = ram_memory_node;
				break;
#ifdef USE_GORDON
			case GORDON_WORKER:
				is_an_accelerator = 1;
				memory_node = ram_memory_node;
				break;
#endif
#ifdef USE_CUDA
			case CUDA_WORKER:
				is_an_accelerator = 1;
				memory_node = register_memory_node(CUDA_RAM);
				break;
#endif
			default:
				STARPU_ASSERT(0);
		}

		if (is_an_accelerator) {
			if (accelerator_bindid == -1)
				accelerator_bindid = get_next_bindid();
			workerarg->bindid = accelerator_bindid;
		}
		else {
			workerarg->bindid = get_next_bindid();
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
