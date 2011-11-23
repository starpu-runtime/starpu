/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

/* Work stealing policy */

#include <core/workers.h>
#include <sched_policies/deque_queues.h>

static unsigned nworkers;
static unsigned rr_worker;
static struct starpu_deque_jobq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_mutex_t global_sched_mutex;
static pthread_cond_t global_sched_cond;

/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to select when stealing or deferring work
 */
static unsigned performed_total = 0;

#ifdef USE_OVERLOAD
static float overload_metric(unsigned id)
{
	float execution_ratio = 0.0f;
	if (performed_total > 0) {
		execution_ratio = _starpu_get_deque_nprocessed(queue_array[id])/performed_total;
	}

	unsigned performed_queue;
	performed_queue = _starpu_get_deque_nprocessed(queue_array[id]);

	float current_ratio = 0.0f;
	if (performed_queue > 0) {
		current_ratio = _starpu_get_deque_njobs(queue_array[id])/performed_queue;
	}
	
	return (current_ratio - execution_ratio);
}

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(void)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = rr_worker;
	do {
		if (overload_metric(worker) > 0.0f)
		{
			q = queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = queue_array[rr_worker];
	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

static struct starpu_deque_jobq_s *select_workerq(void)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = rr_worker;
	do {
		if (overload_metric(worker) < 0.0f)
		{
			q = queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = queue_array[rr_worker];
	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

#else

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(void)
{
	struct starpu_deque_jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}


/* when anonymous threads submit tasks, 
 * we need to select a queue where to dispose them */
static struct starpu_deque_jobq_s *select_workerq(void)
{
	struct starpu_deque_jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

#endif

#ifdef STARPU_DEVEL
#warning TODO rewrite ... this will not scale at all now
#endif
static struct starpu_task *ws_pop_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();

	struct starpu_deque_jobq_s *q;

	q = queue_array[workerid];

	_STARPU_PTHREAD_MUTEX_LOCK(&global_sched_mutex);

	task = _starpu_deque_pop_task(q, -1);
	if (task) {
		/* there was a local task */
		performed_total++;
		_STARPU_PTHREAD_MUTEX_UNLOCK(&global_sched_mutex);
		return task;
	}
	
	/* we need to steal someone's job */
	struct starpu_deque_jobq_s *victimq;
	victimq = select_victimq();

	task = _starpu_deque_pop_task(victimq, workerid);
	if (task) {
		STARPU_TRACE_WORK_STEALING(q, victimq);
		performed_total++;
	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(&global_sched_mutex);

	return task;
}

static int ws_push_task(struct starpu_task *task)
{
	starpu_job_t j = _starpu_get_job_associated_to_task(task);

	int workerid = starpu_worker_get_id();

        struct starpu_deque_jobq_s *deque_queue;
	deque_queue = queue_array[workerid];

        _STARPU_PTHREAD_MUTEX_LOCK(&global_sched_mutex);
	// XXX reuse ?
        //total_number_of_jobs++;

        STARPU_TRACE_JOB_PUSH(task, 0);
        starpu_job_list_push_front(deque_queue->jobq, j);
        deque_queue->njobs++;
        deque_queue->nprocessed++;

        _STARPU_PTHREAD_COND_SIGNAL(&global_sched_cond);
        _STARPU_PTHREAD_MUTEX_UNLOCK(&global_sched_mutex);

        return 0;
}

static void initialize_ws_policy(struct starpu_machine_topology_s *topology, 
				__attribute__ ((unused)) struct starpu_sched_policy *_policy) 
{
	nworkers = topology->nworkers;
	rr_worker = 0;

	_STARPU_PTHREAD_MUTEX_INIT(&global_sched_mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&global_sched_cond, NULL);

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_deque();
		starpu_worker_set_sched_condition(workerid, &global_sched_cond, &global_sched_mutex);
	}
}

struct starpu_sched_policy _starpu_sched_ws_policy = {
	.init_sched = initialize_ws_policy,
	.deinit_sched = NULL,
	.push_task = ws_push_task,
	.pop_task = ws_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "ws",
	.policy_description = "work stealing"
};
