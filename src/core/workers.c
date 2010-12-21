/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <starpu_task_list.h>

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

int starpu_worker_may_execute_task(unsigned workerid, struct starpu_task *task)
{
	/* TODO: check that the task operand sizes will fit on that device */
	/* TODO: call application-provided function for various cases like
	 * double support, shared memory size limit, etc. */
	return !!(task->cl->where & config.workers[workerid].worker_mask);
}



int starpu_combined_worker_may_execute_task(unsigned workerid, struct starpu_task *task)
{
	/* TODO: check that the task operand sizes will fit on that device */
	/* TODO: call application-provided function for various cases like
	 * double support, shared memory size limit, etc. */

	struct starpu_codelet_t *cl = task->cl;
	unsigned nworkers = config.topology.nworkers;

	/* Is this a parallel worker ? */
	if (workerid < nworkers)
	{
		return !!(task->cl->where & config.workers[workerid].worker_mask);
	}
	else {
		if ((cl->type == STARPU_SPMD) || (cl->type == STARPU_FORKJOIN))
		{
			/* TODO we should add other types of constraints */

			/* Is the worker larger than requested ? */
			int worker_size = (int)config.combined_workers[workerid - nworkers].worker_size;
			return !!(worker_size <= task->cl->max_parallelism);
		}
		else
		{
			/* We have a sequential task but a parallel worker */
			return 0;
		}
	}
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
	pthread_cond_t *cond = workerarg->sched_cond;
	pthread_mutex_t *mutex = workerarg->sched_mutex;

	unsigned memory_node = workerarg->memory_node;

	_starpu_memory_node_register_condition(cond, mutex, memory_node);
}

static void _starpu_launch_drivers(struct starpu_machine_config_s *config)
{
	config->running = 1;

	pthread_key_create(&worker_key, NULL);

	unsigned nworkers = config->topology.nworkers;

	/* Launch workers asynchronously (except for SPUs) */
	unsigned worker;
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_worker_s *workerarg = &config->workers[worker];

		workerarg->config = config;

		PTHREAD_MUTEX_INIT(&workerarg->mutex, NULL);
		PTHREAD_COND_INIT(&workerarg->ready_cond, NULL);

		workerarg->workerid = (int)worker;
		workerarg->worker_size = 1;
		workerarg->combined_workerid = workerarg->workerid;
		workerarg->current_rank = 0;

		/* if some codelet's termination cannot be handled directly :
		 * for instance in the Gordon driver, Gordon tasks' callbacks
		 * may be executed by another thread than that of the Gordon
		 * driver so that we cannot call the push_codelet_output method
		 * directly */
		workerarg->terminated_jobs = starpu_job_list_new();

		starpu_task_list_init(&workerarg->local_tasks);
		PTHREAD_MUTEX_INIT(&workerarg->local_tasks_mutex, NULL);
	
		workerarg->status = STATUS_INITIALIZING;

		_STARPU_DEBUG("initialising worker %d\n", worker);

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

	for (worker = 0; worker < nworkers; worker++)
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

/* Initialize the starpu_conf with default values */
int starpu_conf_init(struct starpu_conf *conf)
{
	if (!conf)
		return -EINVAL;

	conf->sched_policy_name = NULL;
	conf->sched_policy = NULL;
	conf->ncpus = -1;
	conf->ncuda = -1;
	conf->nopencl = -1;
	conf->nspus = -1;
	conf->use_explicit_workers_bindid = 0;
	conf->use_explicit_workers_cuda_gpuid = 0;
	conf->use_explicit_workers_opencl_gpuid = 0;
	conf->calibrate = -1;

	return 0;
};

int starpu_init(struct starpu_conf *user_conf)
{
	int ret;

	PTHREAD_MUTEX_LOCK(&init_mutex);
	while (initialized == CHANGING)
		/* Wait for the other one changing it */
		PTHREAD_COND_WAIT(&init_cond, &init_mutex);
	init_count++;
	if (initialized == INITIALIZED) {
	  /* He initialized it, don't do it again, and let the others get the mutex */
	  PTHREAD_MUTEX_UNLOCK(&init_mutex);
	  return 0;
	  }
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

	_starpu_profiling_init();

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

	/* initialize the scheduling policy */
	_starpu_init_sched_policy(&config);

	_starpu_initialize_registered_performance_models();

	/* Launch "basic" workers (ie. non-combined workers) */
	_starpu_launch_drivers(&config);

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
	int status __attribute__((unused));
	unsigned workerid;

	for (workerid = 0; workerid < config->topology.nworkers; workerid++)
	{
		starpu_wake_all_blocked_workers();
		
		_STARPU_DEBUG("wait for worker %u\n", workerid);

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
					if (status) {
						_STARPU_DEBUG("pthread_join -> %d\n", status);
                                        }
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
				if (status) {
					_STARPU_DEBUG("pthread_join -> %d\n", status);
                                }
#endif
			}
		}

		STARPU_ASSERT(starpu_task_list_empty(&worker->local_tasks));
		starpu_job_list_delete(worker->terminated_jobs);
	}
}

unsigned _starpu_machine_is_running(void)
{
	return config.running;
}

unsigned _starpu_worker_can_block(unsigned memnode __attribute__((unused)))
{
#ifdef STARPU_NON_BLOCKING_DRIVERS
	return 0;
#else
	unsigned can_block = 1;

	if (!_starpu_check_that_no_data_request_exists(memnode))
		can_block = 0;

	if (!_starpu_machine_is_running())
		can_block = 0;

	if (!_starpu_execute_registered_progression_hooks())
		can_block = 0;

	return can_block;
#endif
}

static void _starpu_kill_all_workers(struct starpu_machine_config_s *config)
{
	/* set the flag which will tell workers to stop */
	config->running = 0;
	starpu_wake_all_blocked_workers();
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
	PTHREAD_COND_SIGNAL(&init_cond);
	PTHREAD_MUTEX_UNLOCK(&init_mutex);
}

unsigned starpu_worker_get_count(void)
{
	return config.topology.nworkers;
}

unsigned starpu_combined_worker_get_count(void)
{
	return config.topology.ncombinedworkers;
}

unsigned starpu_cpu_worker_get_count(void)
{
	return config.topology.ncpus;
}

unsigned starpu_cuda_worker_get_count(void)
{
	return config.topology.ncudagpus;
}

unsigned starpu_opencl_worker_get_count(void)
{
	return config.topology.nopenclgpus;
}

unsigned starpu_spu_worker_get_count(void)
{
	return config.topology.ngordon_spus;
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

int starpu_combined_worker_get_id(void)
{
	struct starpu_worker_s *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->combined_workerid;
	}
	else {
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

int starpu_combined_worker_get_size(void)
{
	struct starpu_worker_s *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->worker_size;
	}
	else {
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

int starpu_combined_worker_get_rank(void)
{
	struct starpu_worker_s *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->current_rank;
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

struct starpu_combined_worker_s *_starpu_get_combined_worker_struct(unsigned id)
{
	unsigned basic_worker_count = starpu_worker_get_count();

	STARPU_ASSERT(id >= basic_worker_count);
	return &config.combined_workers[id - basic_worker_count];
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

/* Retrieve the status which indicates what the worker is currently doing. */
starpu_worker_status _starpu_worker_get_status(int workerid)
{
	return config.workers[workerid].status;
}

/* Change the status of the worker which indicates what the worker is currently
 * doing (eg. executing a callback). */
void _starpu_worker_set_status(int workerid, starpu_worker_status status)
{
	config.workers[workerid].status = status;
}

void starpu_worker_set_sched_condition(int workerid, pthread_cond_t *sched_cond, pthread_mutex_t *sched_mutex)
{
	config.workers[workerid].sched_cond = sched_cond;
	config.workers[workerid].sched_mutex = sched_mutex;
}
