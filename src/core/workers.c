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

/* XXX quick and dirty implementation for now ... */
pthread_key_t local_workers_key;

static struct machine_config_s config;

/* in case a task is submitted, we may check whether there exists a worker
   that may execute the task or not */

inline uint32_t worker_exists(uint32_t task_mask)
{
	return (task_mask & config.worker_mask);
} 

inline uint32_t may_submit_cuda_task(void)
{
	return ((CUDA|CUBLAS) & config.worker_mask);
}

inline uint32_t may_submit_core_task(void)
{
	return (CORE & config.worker_mask);
}

/*
 * Runtime initialization methods
 */

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
					gordon_worker_set.nworkers = config->ngordon_spus; 
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
	
	open_debug_logfile();

	timing_init();

	starpu_build_topology(&config, user_conf);

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

static void operate_on_all_queues_attached_to_node(unsigned nodeid, queue_op op)
{
	unsigned q_id;
	struct jobq_s *q;

	mem_node_descr * const descr = get_memory_node_description();

	pthread_rwlock_rdlock(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->queues_count[nodeid];

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr->attached_queues_per_node[nodeid][q_id];
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

	pthread_rwlock_unlock(&descr->attached_queues_rwlock);
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

	mem_node_descr * const descr = get_memory_node_description();

	pthread_rwlock_rdlock(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->total_queues_count;

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr->attached_queues_all[q_id];
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

	pthread_rwlock_unlock(&descr->attached_queues_rwlock);
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

	close_debug_logfile();
}
