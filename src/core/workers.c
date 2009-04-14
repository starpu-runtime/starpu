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

#include <core/workers.h>

/* XXX quick and dirty implementation for now ... */
pthread_key_t local_workers_key;

static struct machine_config_s config;

/* in case a task is submitted, we may check whether there exists a worker
   that may execute the task or not */
static uint32_t worker_mask = 0;

inline uint32_t worker_exists(uint32_t task_mask)
{
	return (task_mask & worker_mask);
} 

inline uint32_t may_submit_cuda_task(void)
{
	return ((CUDA|CUBLAS) & worker_mask);
}

inline uint32_t may_submit_core_task(void)
{
	return (CORE & worker_mask);
}

#ifdef USE_CPUS
static unsigned ncores;
#endif
#ifdef USE_CUDA
static unsigned ncudagpus;
#endif
#ifdef USE_GORDON
static unsigned ngordon_spus;
#endif

/*
 * Runtime initialization methods
 */

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
		worker_mask |= (CUDA|CUBLAS);
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
		worker_mask |= GORDON;
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
		worker_mask |= CORE;
	}

	config->nworkers += ncores;
#endif


	if (config->nworkers == 0)
	{
		fprintf(stderr, "No worker found, aborting ...\n");
		exit(-1);
	}
}

void bind_thread_on_cpu(unsigned coreid)
{
#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(coreid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif
}

static void init_workers_binding(struct machine_config_s *config)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	int current_bindid = 0;

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
				accelerator_bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
			workerarg->bindid = accelerator_bindid;
		}
		else {
			workerarg->bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
		}

		workerarg->memory_node = memory_node;
	}
}

#ifdef USE_GORDON
static unsigned gordon_inited = 0;	
static struct worker_set_s gordon_worker_set;
#endif

static void init_workers(struct machine_config_s *config)
{
	config->running = 1;

	pthread_key_create(&local_workers_key, NULL);

	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct worker_s *workerarg = &config->workers[worker];

		pthread_mutex_init(&workerarg->mutex, NULL);
		pthread_cond_init(&workerarg->ready_cond, NULL);

		/* if some codelet's termination cannot be handled directly :
		 * for instance in the Gordon driver, Gordon tasks' callbacks
		 * may be executed by another thread than that of the Gordon
		 * driver so that we cannot call the push_codelet_output method
		 * directly */
		workerarg->terminated_jobs = job_list_new();
	
		switch (workerarg->arch) {
#ifdef USE_CPUS
			case CORE_WORKER:
				workerarg->set = NULL;
				workerarg->worker_is_initialized = 0;
				pthread_create(&workerarg->worker_thread, 
						NULL, core_worker, workerarg);

				pthread_mutex_lock(&workerarg->mutex);
				if (!workerarg->worker_is_initialized)
					pthread_cond_wait(&workerarg->ready_cond, &workerarg->mutex);
				pthread_mutex_unlock(&workerarg->mutex);

				break;
#endif
#ifdef USE_CUDA
			case CUDA_WORKER:
				workerarg->set = NULL;
				workerarg->worker_is_initialized = 0;
				pthread_create(&workerarg->worker_thread, 
						NULL, cuda_worker, workerarg);

				pthread_mutex_lock(&workerarg->mutex);
				if (!workerarg->worker_is_initialized)
					pthread_cond_wait(&workerarg->ready_cond, &workerarg->mutex);
				pthread_mutex_unlock(&workerarg->mutex);

				break;
#endif
#ifdef USE_GORDON
			case GORDON_WORKER:
				/* we will only launch gordon once, but it will handle 
				 * the different SPU workers */
				if (!gordon_inited)
				{
					gordon_worker_set.nworkers = ngordon_spus; 
					gordon_worker_set.workers = &config->workers[worker];

					gordon_worker_set.set_is_initialized = 0;

					pthread_create(&gordon_worker_set.worker_thread, NULL, 
							gordon_worker, &gordon_worker_set);

					pthread_mutex_lock(&gordon_worker_set.mutex);
					if (!gordon_worker_set.set_is_initialized)
						pthread_cond_wait(&gordon_worker_set.ready_cond,
									&gordon_worker_set.mutex);
					pthread_mutex_unlock(&gordon_worker_set.mutex);

					gordon_inited = 1;
				}
				
				workerarg->set = &gordon_worker_set;
				gordon_worker_set.joined = 0;
				workerarg->worker_is_running = 1;

				break;
#endif
			default:
				STARPU_ASSERT(0);
		}
	}
}

void starpu_init(struct starpu_conf *user_conf)
{
	srand(2008);

#ifdef USE_FXT
	start_fxt_profiling();
#endif

	timing_init();

	init_machine_config(&config, user_conf);

	/* for the data wizard */
	init_memory_nodes();

	init_workers_binding(&config);

	/* initialize the scheduler */

	/* initialize the queue containing the jobs */
	init_sched_policy(&config, user_conf);

	init_workers(&config);
}

/*
 * Handle runtime termination 
 */

static void terminate_workers(struct machine_config_s *config)
{
	int status;
	unsigned workerid;

	for (workerid = 0; workerid < config->nworkers; workerid++)
	{
		wake_all_blocked_workers();
		
#ifdef VERBOSE
		fprintf(stderr, "wait for worker %d\n", workerid);
#endif

		struct worker_set_s *set = config->workers[workerid].set;

		/* in case StarPU termination code is called from a callback,
 		 * we have to check if pthread_self() is the worker itself */
		if (set){ 
			if (!set->joined) {
				if (pthread_self() != set->worker_thread)
				{
					status = pthread_join(set->worker_thread, NULL);
#ifdef VERBOSE
					if (status)
						fprintf(stderr, "pthread_join -> %d\n", status);
#endif
				}

				set->joined = 1;
			}
		}
		else {
			struct worker_s *worker = &config->workers[workerid];
			if (pthread_self() != worker->worker_thread)
			{
				status = pthread_join(worker->worker_thread, NULL);
#ifdef VERBOSE
				if (status)
					fprintf(stderr, "pthread_join -> %d\n", status);
#endif
			}
		}
	}
}

unsigned machine_is_running(void)
{
	return config.running;
}

typedef enum {
	BROADCAST,
	LOCK,
	UNLOCK
} queue_op;

/* XXX we should use an accessor */
extern mem_node_descr descr;

static void operate_on_all_queues_attached_to_node(unsigned nodeid, queue_op op)
{
	unsigned q_id;
	struct jobq_s *q;

	take_mutex(&descr.attached_queues_mutex);

	unsigned nqueues = descr.queues_count[nodeid];

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr.attached_queues_per_node[nodeid][q_id];
		switch (op) {
			case BROADCAST:
				pthread_cond_broadcast(&q->activity_cond);
				break;
			case LOCK:
				pthread_mutex_lock(&q->activity_mutex);
				break;
			case UNLOCK:
				pthread_mutex_unlock(&q->activity_mutex);
				break;
		}
	}

	release_mutex(&descr.attached_queues_mutex);
}

inline void lock_all_queues_attached_to_node(unsigned node)
{
	operate_on_all_queues_attached_to_node(node, LOCK);
}

inline void unlock_all_queues_attached_to_node(unsigned node)
{
	operate_on_all_queues_attached_to_node(node, UNLOCK);
}

inline void broadcast_all_queues_attached_to_node(unsigned node)
{
	operate_on_all_queues_attached_to_node(node, BROADCAST);
}

static void operate_on_all_queues(queue_op op)
{
	unsigned q_id;
	struct jobq_s *q;

	take_mutex(&descr.attached_queues_mutex);

	unsigned nqueues = descr.total_queues_count;

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr.attached_queues_all[q_id];
		switch (op) {
			case BROADCAST:
				pthread_cond_broadcast(&q->activity_cond);
				break;
			case LOCK:
				pthread_mutex_lock(&q->activity_mutex);
				break;
			case UNLOCK:
				pthread_mutex_unlock(&q->activity_mutex);
				break;
		}
	}

	release_mutex(&descr.attached_queues_mutex);
}

static void kill_all_workers(struct machine_config_s *config)
{
	/* lock all workers and the scheduler (in the proper order) to make
	   sure everyone will notice the termination */
	/* WARNING: here we make the asumption that a queue is not attached to
 	 * different memory nodes ! */

	struct sched_policy_s *sched = get_sched_policy();

	operate_on_all_queues(LOCK);
	pthread_mutex_lock(&sched->sched_activity_mutex);
	
	/* set the flag which will tell workers to stop */
	config->running = 0;

	operate_on_all_queues(BROADCAST);
	pthread_cond_broadcast(&sched->sched_activity_cond);

	pthread_mutex_unlock(&sched->sched_activity_mutex);
	operate_on_all_queues(UNLOCK);
}

void starpu_shutdown(void)
{
	display_msi_stats();
	display_alloc_cache_stats();

	/* tell all workers to shutdown */
	kill_all_workers(&config);

#ifdef DATA_STATS
	display_comm_ammounts();
#endif

	if (starpu_get_env_number("CALIBRATE") != -1)
		dump_registered_models();

	/* wait for their termination */
	terminate_workers(&config);

	/* cleanup StarPU internal data structures */
	deinit_memory_nodes();

	deinit_sched_policy(&config);
}
