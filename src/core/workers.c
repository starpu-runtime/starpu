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
#include <common/utils.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/task.h>
#include <profiling/profiling.h>

#ifdef __MINGW32__
#include <windows.h>
#endif

/* acquire/release semantic for concurrent initialization/de-initialization */
static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t init_cond = PTHREAD_COND_INITIALIZER;
static int init_count;
static enum { UNINITIALIZED, CHANGING, INITIALIZED } initialized = UNINITIALIZED;

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

inline uint32_t _starpu_may_submit_opencl_task(void)
{
	return (STARPU_OPENCL & config.worker_mask);
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

	PTHREAD_MUTEX_LOCK(&jobq->activity_mutex);

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
		case STARPU_OPENCL_WORKER:
			jobq->alpha = STARPU_OPENCL_ALPHA;
			break;
		case STARPU_GORDON_WORKER:
			jobq->alpha = STARPU_GORDON_ALPHA;
			break;
		default:
			STARPU_ABORT();
	}

	/* This is only useful (and meaningful) is there is a single memory
	 * node "related" to that queue */
	jobq->memory_node = workerarg->memory_node;

	jobq->total_computation_time = 0.0;
	jobq->total_communication_time = 0.0;
	jobq->total_computation_time_error = 0.0;
	jobq->total_job_performed = 0;
		
	PTHREAD_MUTEX_UNLOCK(&jobq->activity_mutex);

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

		PTHREAD_MUTEX_INIT(&workerarg->mutex, NULL);
		PTHREAD_COND_INIT(&workerarg->ready_cond, NULL);

		workerarg->workerid = (int)worker;

		/* if some codelet's termination cannot be handled directly :
		 * for instance in the Gordon driver, Gordon tasks' callbacks
		 * may be executed by another thread than that of the Gordon
		 * driver so that we cannot call the push_codelet_output method
		 * directly */
		workerarg->terminated_jobs = starpu_job_list_new();

		workerarg->local_jobs = starpu_job_list_new();
		PTHREAD_MUTEX_INIT(&workerarg->local_jobs_mutex, NULL);
	
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
#ifdef STARPU_USE_OPENCL
			case STARPU_OPENCL_WORKER:
				workerarg->set = NULL;
				workerarg->worker_is_initialized = 0;
				pthread_create(&workerarg->worker_thread, 
						NULL, _starpu_opencl_worker, workerarg);

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

					PTHREAD_MUTEX_LOCK(&gordon_worker_set.mutex);
					while (!gordon_worker_set.set_is_initialized)
						PTHREAD_COND_WAIT(&gordon_worker_set.ready_cond,
									&gordon_worker_set.mutex);
					PTHREAD_MUTEX_UNLOCK(&gordon_worker_set.mutex);

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
			case STARPU_OPENCL_WORKER:			  
				PTHREAD_MUTEX_LOCK(&workerarg->mutex);
				while (!workerarg->worker_is_initialized)
					PTHREAD_COND_WAIT(&workerarg->ready_cond, &workerarg->mutex);
				PTHREAD_MUTEX_UNLOCK(&workerarg->mutex);
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

	PTHREAD_MUTEX_LOCK(&init_mutex);
	while (initialized == CHANGING)
		/* Wait for the other one changing it */
		PTHREAD_COND_WAIT(&init_cond, &init_mutex);
	init_count++;
	if (initialized == INITIALIZED)
		/* He initialized it, don't do it again */
		return 0;
	/* initialized == UNINITIALIZED */
	initialized = CHANGING;
	PTHREAD_MUTEX_UNLOCK(&init_mutex);

#ifdef __MINGW32__
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
#endif

	srand(2008);
	
#ifdef STARPU_USE_FXT
	_starpu_start_fxt_profiling();
#endif
	
	_starpu_open_debug_logfile();

	_starpu_timing_init();

	_starpu_load_bus_performance_files();

	/* store the pointer to the user explicit configuration during the
	 * initialization */
	config.user_conf = user_conf;

	ret = _starpu_build_topology(&config);
	if (ret) {
		PTHREAD_MUTEX_LOCK(&init_mutex);
		init_count--;
		initialized = UNINITIALIZED;
		/* Let somebody else try to do it */
		PTHREAD_COND_SIGNAL(&init_cond);
		PTHREAD_MUTEX_UNLOCK(&init_mutex);
		return ret;
	}

	/* We need to store the current task handled by the different
	 * threads */
	_starpu_initialize_current_task_key();	

	/* initialize the scheduler */

	/* initialize the queue containing the jobs */
	_starpu_init_sched_policy(&config);

	_starpu_initialize_registered_performance_models();

	_starpu_init_workers(&config);

	PTHREAD_MUTEX_LOCK(&init_mutex);
	initialized = INITIALIZED;
	/* Tell everybody that we initialized */
	PTHREAD_COND_BROADCAST(&init_cond);
	PTHREAD_MUTEX_UNLOCK(&init_mutex);

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

	PTHREAD_RWLOCK_RDLOCK(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->queues_count[nodeid];

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr->attached_queues_per_node[nodeid][q_id];
		switch (op) {
			case BROADCAST:
				PTHREAD_COND_BROADCAST(&q->activity_cond);
				break;
			case LOCK:
				PTHREAD_MUTEX_LOCK(&q->activity_mutex);
				break;
			case UNLOCK:
				PTHREAD_MUTEX_UNLOCK(&q->activity_mutex);
				break;
		}
	}

	PTHREAD_RWLOCK_UNLOCK(&descr->attached_queues_rwlock);
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

	PTHREAD_RWLOCK_RDLOCK(&descr->attached_queues_rwlock);

	unsigned nqueues = descr->total_queues_count;

	for (q_id = 0; q_id < nqueues; q_id++)
	{
		q  = descr->attached_queues_all[q_id];
		switch (op) {
			case BROADCAST:
				PTHREAD_COND_BROADCAST(&q->activity_cond);
				break;
			case LOCK:
				PTHREAD_MUTEX_LOCK(&q->activity_mutex);
				break;
			case UNLOCK:
				PTHREAD_MUTEX_UNLOCK(&q->activity_mutex);
				break;
		}
	}

	PTHREAD_RWLOCK_UNLOCK(&descr->attached_queues_rwlock);
}

static void _starpu_kill_all_workers(struct starpu_machine_config_s *config)
{
	/* lock all workers and the scheduler (in the proper order) to make
	   sure everyone will notice the termination */
	/* WARNING: here we make the asumption that a queue is not attached to
 	 * different memory nodes ! */

	struct starpu_sched_policy_s *sched = _starpu_get_sched_policy();

	_starpu_operate_on_all_queues(LOCK);
	PTHREAD_MUTEX_LOCK(&sched->sched_activity_mutex);
	
	/* set the flag which will tell workers to stop */
	config->running = 0;

	_starpu_operate_on_all_queues(BROADCAST);
	PTHREAD_COND_BROADCAST(&sched->sched_activity_cond);

	PTHREAD_MUTEX_UNLOCK(&sched->sched_activity_mutex);
	_starpu_operate_on_all_queues(UNLOCK);
}

void starpu_shutdown(void)
{
	PTHREAD_MUTEX_LOCK(&init_mutex);
	init_count--;
	if (init_count)
		/* Still somebody needing StarPU, don't deinitialize */
		return;
	/* We're last */
	initialized = CHANGING;
	PTHREAD_MUTEX_UNLOCK(&init_mutex);

	_starpu_display_msi_stats();
	_starpu_display_alloc_cache_stats();

	/* tell all workers to shutdown */
	_starpu_kill_all_workers(&config);

#ifdef STARPU_DATA_STATS
	_starpu_display_comm_amounts();
#endif

	_starpu_deinitialize_registered_performance_models();

	/* wait for their termination */
	_starpu_terminate_workers(&config);

	_starpu_deinit_sched_policy(&config);

	_starpu_destroy_topology(&config);

#ifdef STARPU_USE_FXT
	_starpu_stop_fxt_profiling();
#endif

	_starpu_close_debug_logfile();

	PTHREAD_MUTEX_LOCK(&init_mutex);
	initialized = UNINITIALIZED;
	/* Let someone else that wants to initialize it again do it */
	pthread_cond_signal(&init_cond);
	PTHREAD_MUTEX_UNLOCK(&init_mutex);
}

unsigned starpu_worker_get_count(void)
{
	return config.nworkers;
}

unsigned starpu_cpu_worker_get_count(void)
{
	return config.ncpus;
}

unsigned starpu_cuda_worker_get_count(void)
{
	return config.ncudagpus;
}

unsigned starpu_opencl_worker_get_count(void)
{
	return config.nopenclgpus;
}

unsigned starpu_spu_worker_get_count(void)
{
	return config.ngordon_spus;
}

/* When analyzing performance, it is useful to see what is the processing unit
 * that actually performed the task. This function returns the id of the
 * processing unit actually executing it, therefore it makes no sense to use it
 * within the callbacks of SPU functions for instance. If called by some thread
 * that is not controlled by StarPU, starpu_worker_get_id returns -1. */
int starpu_worker_get_id(void)
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

int starpu_worker_get_devid(int id)
{
	return config.workers[id].devid;
}

struct starpu_worker_s *_starpu_get_worker_struct(unsigned id)
{
	return &config.workers[id];
}

enum starpu_archtype starpu_worker_get_type(int id)
{
	return config.workers[id].arch;
}

void starpu_worker_get_name(int id, char *dst, size_t maxlen)
{
	char *name = config.workers[id].name;

	snprintf(dst, maxlen, "%s", name);
}

starpu_worker_status _starpu_worker_get_status(int workerid)
{
	return config.workers[workerid].status;
}

void _starpu_worker_set_status(int workerid, starpu_worker_status status)
{
	config.workers[workerid].status = status;
}

/* TODO move in some driver/common/ directory */
/* Workers may block when there is no work to do at all. We assume that the
 * mutex is hold when that function is called. */
void _starpu_block_worker(int workerid, pthread_cond_t *cond, pthread_mutex_t *mutex)
{
	struct timespec start_time, end_time;

	STARPU_TRACE_WORKER_SLEEP_START
	config.workers[workerid].status = STATUS_SLEEPING;

	starpu_clock_gettime(&start_time);
	_starpu_worker_register_sleeping_start_date(workerid, &start_time);

	PTHREAD_COND_WAIT(cond, mutex);

	config.workers[workerid].status = STATUS_UNKNOWN;
	STARPU_TRACE_WORKER_SLEEP_END
	starpu_clock_gettime(&end_time);

	int profiling = starpu_profiling_status_get();
	if (profiling)
	{
		struct timespec sleeping_time;
		starpu_timespec_sub(&end_time, &start_time, &sleeping_time);
		_starpu_worker_update_profiling_info_sleeping(workerid, &start_time, &end_time);
	}
}
