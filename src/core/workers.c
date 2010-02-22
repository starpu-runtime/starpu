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

#ifdef __MINGW32__
#include <windows.h>
#endif

static pthread_key_t worker_key;

static struct starpu_machine_config_s config;

struct starpu_machine_config_s *_starpu_get_machine_config(void)
{
	return &config;
}

/* in case a task is submitted, we may check whether there exists a worker
   that may execute the task or not */

inline uint32_t _starpu_worker_exists(uint32_t task_mask)
{
	return (task_mask & config.worker_mask);
} 

inline uint32_t _starpu_may_submit_cuda_task(void)
{
	return (STARPU_CUDA & config.worker_mask);
}

inline uint32_t _starpu_may_submit_cpu_task(void)
{
	return (STARPU_CPU & config.worker_mask);
}

inline uint32_t _starpu_worker_may_execute_task(unsigned workerid, uint32_t where)
{
	return (where & config.workers[workerid].worker_mask);
}

/*
 * Runtime initialization methods
 */

#ifdef STARPU_USE_GORDON
static unsigned gordon_inited = 0;	
static struct starpu_worker_set_s gordon_worker_set;
#endif

static void _starpu_init_worker_queue(struct starpu_worker_s *workerarg)
{
	struct starpu_jobq_s *jobq = workerarg->jobq;

	/* warning : in case there are multiple workers on the same
	  queue, we overwrite this value so that it is meaningless */
	jobq->arch = workerarg->perf_arch;
		
	jobq->who |= workerarg->worker_mask;

	switch (workerarg->arch) {
		case STARPU_CPU_WORKER:
			jobq->alpha = STARPU_CPU_ALPHA;
			break;
		case STARPU_CUDA_WORKER:
			jobq->alpha = STARPU_CUDA_ALPHA;
			break;
		case STARPU_GORDON_WORKER:
			jobq->alpha = STARPU_GORDON_ALPHA;
			break;
		default:
			STARPU_ABORT();
	}
		
	_starpu_memory_node_attach_queue(jobq, workerarg->memory_node);
}

static void _starpu_init_workers(struct starpu_machine_config_s *config)
{
	config->running = 1;

	pthread_key_create(&worker_key, NULL);

	/* Launch workers asynchronously (except for SPUs) */
	unsigned worker;
	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct starpu_worker_s *workerarg = &config->workers[worker];

		workerarg->config = config;

		pthread_mutex_init(&workerarg->mutex, NULL);
		pthread_cond_init(&workerarg->ready_cond, NULL);

		workerarg->workerid = (int)worker;

		/* if some codelet's termination cannot be handled directly :
		 * for instance in the Gordon driver, Gordon tasks' callbacks
		 * may be executed by another thread than that of the Gordon
		 * driver so that we cannot call the push_codelet_output method
		 * directly */
		workerarg->terminated_jobs = starpu_job_list_new();

		workerarg->local_jobs = starpu_job_list_new();
		pthread_mutex_init(&workerarg->local_jobs_mutex, NULL);
	
		workerarg->status = STATUS_INITIALIZING;

		_starpu_init_worker_queue(workerarg);

		switch (workerarg->arch) {
#ifdef STARPU_USE_CPU
			case STARPU_CPU_WORKER:
				workerarg->set = NULL;
				workerarg->worker_is_initialized = 0;
				pthread_create(&workerarg->worker_thread, 
						NULL, _starpu_cpu_worker, workerarg);
				break;
#endif
#ifdef STARPU_USE_CUDA
			case STARPU_CUDA_WORKER:
				workerarg->set = NULL;
				workerarg->worker_is_initialized = 0;
				pthread_create(&workerarg->worker_thread, 
						NULL, _starpu_cuda_worker, workerarg);

				break;
#endif
#ifdef STARPU_USE_GORDON
			case STARPU_GORDON_WORKER:
				/* we will only launch gordon once, but it will handle 
				 * the different SPU workers */
				if (!gordon_inited)
				{
					gordon_worker_set.nworkers = config->ngordon_spus; 
					gordon_worker_set.workers = &config->workers[worker];

					gordon_worker_set.set_is_initialized = 0;

					pthread_create(&gordon_worker_set.worker_thread, NULL, 
							_starpu_gordon_worker, &gordon_worker_set);

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
				STARPU_ABORT();
		}
	}

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct starpu_worker_s *workerarg = &config->workers[worker];

		switch (workerarg->arch) {
			case STARPU_CPU_WORKER:
			case STARPU_CUDA_WORKER:
				pthread_mutex_lock(&workerarg->mutex);
				if (!workerarg->worker_is_initialized)
					pthread_cond_wait(&workerarg->ready_cond, &workerarg->mutex);
				pthread_mutex_unlock(&workerarg->mutex);
				break;
#ifdef STARPU_USE_GORDON
			case STARPU_GORDON_WORKER:
				/* the initialization of Gordon worker is
				 * synchronous for now */
				break;
#endif
			default:
				STARPU_ABORT();
		}
	}

}

void _starpu_set_local_worker_key(struct starpu_worker_s *worker)
{
	pthread_setspecific(worker_key, worker);
}

struct starpu_worker_s *_starpu_get_local_worker_key(void)
{
	return pthread_getspecific(worker_key);
}

int starpu_init(struct starpu_conf *user_conf)
{
	int ret;

#ifdef __MINGW32__
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
#endif

	srand(2008);
	
#ifdef STARPU_USE_FXT
	start_fxt_profiling();
#endif
	
	_starpu_open_debug_logfile();

	_starpu_timing_init();

	_starpu_load_bus_performance_files();

	/* store the pointer to the user explicit configuration during the
	 * initialization */
	config.user_conf = user_conf;

	ret = _starpu_build_topology(&config);
	if (ret)
		return ret;

	/* initialize the scheduler */

	/* initialize the queue containing the jobs */
	_starpu_init_sched_policy(&config);

	_starpu_init_workers(&config);

	return 0;
}

/*
 * Handle runtime termination 
 */

static void _starpu_terminate_workers(struct starpu_machine_config_s *config)
{
	int status;
	unsigned workerid;

	for (workerid = 0; workerid < config->nworkers; workerid++)
	{
		starpu_wake_all_blocked_workers();
		
#ifdef STARPU_VERBOSE
		fprintf(stderr, "wait for worker %d\n", workerid);
#endif

		struct starpu_worker_set_s *set = config->workers[workerid].set;
		struct starpu_worker_s *worker = &config->workers[workerid];

		/* in case StarPU termination code is called from a callback,
 		 * we have to check if pthread_self() is the worker itself */
		if (set){ 
			if (!set->joined) {
				if (!pthread_equal(pthread_self(), set->worker_thread))
				{
					status = pthread_join(set->worker_thread, NULL);
#ifdef STARPU_VERBOSE
					if (status)
						fprintf(stderr, "pthread_join -> %d\n", status);
#endif
				}

				set->joined = 1;
			}
		}
		else {
			if (!pthread_equal(pthread_self(), worker->worker_thread))
			{
				status = pthread_join(worker->worker_thread, NULL);
#ifdef STARPU_VERBOSE
				if (status)
					fprintf(stderr, "pthread_join -> %d\n", status);
#endif
			}
		}

		starpu_job_list_delete(worker->local_jobs);
		starpu_job_list_delete(worker->terminated_jobs);
	}
}

unsigned _starpu_machine_is_running(void)
{
	return config.running;
}

unsigned _starpu_worker_can_block(unsigned memnode)
{
	unsigned can_block = 1;

	if (!_starpu_check_that_no_data_request_exists(memnode))
		can_block = 0;

	if (!_starpu_machine_is_running())
		can_block = 0;

	if (!_starpu_execute_registered_progression_hooks())
		can_block = 0;

	return can_block;
}

typedef enum {
	BROADCAST,
	LOCK,
	UNLOCK
} queue_op;

static void _starpu_operate_on_all_queues_attached_to_node(unsigned nodeid, queue_op op)
{
	unsigned q_id;
	struct starpu_jobq_s *q;

	starpu_mem_node_descr * const descr = _starpu_get_memory_node_description();

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

inline void _starpu_lock_all_queues_attached_to_node(unsigned node)
{
	_starpu_operate_on_all_queues_attached_to_node(node, LOCK);
}

inline void _starpu_unlock_all_queues_attached_to_node(unsigned node)
{
	_starpu_operate_on_all_queues_attached_to_node(node, UNLOCK);
}

inline void _starpu_broadcast_all_queues_attached_to_node(unsigned node)
{
	_starpu_operate_on_all_queues_attached_to_node(node, BROADCAST);
}

static void _starpu_operate_on_all_queues(queue_op op)
{
	unsigned q_id;
	struct starpu_jobq_s *q;

	starpu_mem_node_descr * const descr = _starpu_get_memory_node_description();

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

static void _starpu_kill_all_workers(struct starpu_machine_config_s *config)
{
	/* lock all workers and the scheduler (in the proper order) to make
	   sure everyone will notice the termination */
	/* WARNING: here we make the asumption that a queue is not attached to
 	 * different memory nodes ! */

	struct starpu_sched_policy_s *sched = _starpu_get_sched_policy();

	_starpu_operate_on_all_queues(LOCK);
	pthread_mutex_lock(&sched->sched_activity_mutex);
	
	/* set the flag which will tell workers to stop */
	config->running = 0;

	_starpu_operate_on_all_queues(BROADCAST);
	pthread_cond_broadcast(&sched->sched_activity_cond);

	pthread_mutex_unlock(&sched->sched_activity_mutex);
	_starpu_operate_on_all_queues(UNLOCK);
}

void starpu_shutdown(void)
{
	_starpu_display_msi_stats();
	_starpu_display_alloc_cache_stats();

	/* tell all workers to shutdown */
	_starpu_kill_all_workers(&config);

#ifdef STARPU_DATA_STATS
	_starpu_display_comm_amounts();
#endif

	if (starpu_get_env_number("STARPU_CALIBRATE") != -1)
		_starpu_dump_registered_models();

	/* wait for their termination */
	_starpu_terminate_workers(&config);

	_starpu_deinit_sched_policy(&config);

	_starpu_destroy_topology(&config);

#ifdef STARPU_USE_FXT
	stop_fxt_profiling();
#endif

	_starpu_close_debug_logfile();
}

unsigned starpu_get_worker_count(void)
{
	return config.nworkers;
}

unsigned starpu_get_cpu_worker_count(void)
{
	return config.ncpus;
}

unsigned starpu_get_cuda_worker_count(void)
{
	return config.ncudagpus;
}

unsigned starpu_get_spu_worker_count(void)
{
	return config.ngordon_spus;
}

/* When analyzing performance, it is useful to see what is the processing unit
 * that actually performed the task. This function returns the id of the
 * processing unit actually executing it, therefore it makes no sense to use it
 * within the callbacks of SPU functions for instance. If called by some thread
 * that is not controlled by StarPU, starpu_get_worker_id returns -1. */
int starpu_get_worker_id(void)
{
	struct starpu_worker_s * worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->workerid;
	}
	else {
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

struct starpu_worker_s *_starpu_get_worker_struct(unsigned id)
{
	return &config.workers[id];
}

enum starpu_archtype starpu_get_worker_type(int id)
{
	return config.workers[id].arch;
}

void starpu_get_worker_name(int id, char *dst, size_t maxlen)
{
	char *name = config.workers[id].name;

	snprintf(dst, maxlen, "%s", name);
}
