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

/* Distributed queues using performance modeling to assign tasks */

#include <limits.h>

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/perfmodel/perfmodel.h>

static unsigned nworkers;
static struct starpu_fifo_taskq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static double alpha = 1.0;
static double beta = 1.0;

#ifdef STARPU_VERBOSE
static long int total_task_cnt = 0;
static long int ready_task_cnt = 0;
#endif

static int count_non_ready_buffers(struct starpu_task *task, uint32_t node)
{
	int cnt = 0;

	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_buffer_descr *descr;
		starpu_data_handle handle;

		descr = &descrs[index];
		handle = descr->handle;
		
		int is_valid;
		starpu_data_query_status(handle, node, NULL, &is_valid, NULL);

		if (!is_valid)
			cnt++;
	}

	return cnt;
}

static struct starpu_task *_starpu_fifo_pop_first_ready_task(struct starpu_fifo_taskq_s *fifo_queue, unsigned node)
{
	struct starpu_task *task = NULL, *current;

	if (fifo_queue->ntasks == 0)
		return NULL;

	if (fifo_queue->ntasks > 0) 
	{
		fifo_queue->ntasks--;

		task = starpu_task_list_back(&fifo_queue->taskq);

		int first_task_priority = task->priority;

		current = task;

		int non_ready_best = INT_MAX;

		while (current)
		{
			int priority = current->priority;

			if (priority <= first_task_priority)
			{
				int non_ready = count_non_ready_buffers(current, node);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					task = current;

					if (non_ready == 0)
						break;
				}
			}

			current = current->prev;
		}
		
		starpu_task_list_erase(&fifo_queue->taskq, task);

		STARPU_TRACE_JOB_POP(task, 0);
	}
	
	return task;
}

static struct starpu_task *dmda_pop_ready_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	unsigned node = starpu_worker_get_memory_node(workerid);

	task = _starpu_fifo_pop_first_ready_task(fifo, node);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, starpu_worker_get_memory_node(workerid));
			if (non_ready == 0)
				ready_task_cnt++;
		}

		total_task_cnt++;
#endif
	}

	return task;
}

static struct starpu_task *dmda_pop_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	task = _starpu_fifo_pop_task(fifo, -1);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, starpu_worker_get_memory_node(workerid));
			if (non_ready == 0)
				ready_task_cnt++;
		}

		total_task_cnt++;
#endif
	}

	return task;
}



static struct starpu_task *dmda_pop_every_task(void)
{
	struct starpu_task *new_list;

	int workerid = starpu_worker_get_id();

	struct starpu_fifo_taskq_s *fifo = queue_array[workerid];

	new_list = _starpu_fifo_pop_every_task(fifo, &sched_mutex[workerid], workerid);

	while (new_list)
	{
		double model = new_list->predicted;

		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	
		new_list = new_list->next;
	}

	return new_list;
}

int _starpu_fifo_push_sorted_task(struct starpu_fifo_taskq_s *fifo_queue, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond, struct starpu_task *task)
{
	struct starpu_task_list *list = &fifo_queue->taskq;

	PTHREAD_MUTEX_LOCK(sched_mutex);

	STARPU_TRACE_JOB_PUSH(task, 0);

	if (list->head == NULL)
	{
		list->head = task;
		list->tail = task;
		task->prev = NULL;
		task->next = NULL;
	}
	else {
		struct starpu_task *current = list->head;
		struct starpu_task *prev = NULL;

		while (current)
		{
			if (current->priority >= task->priority)
				break;

			prev = current;
			current = current->next;
		}

		if (prev == NULL)
		{
			/* Insert at the front of the list */
			list->head->prev = task;
			task->prev = NULL;
			task->next = list->head;
			list->head = task;
		}
		else {
			if (current)
			{
				/* Insert between prev and current */
				task->prev = prev;
				prev->next = task;
				task->next = current;
				current->prev = task;
			}
			else {
				/* Insert at the tail of the list */
				list->tail->next = task;
				task->next = NULL;
				task->prev = list->tail;
				list->tail = task;
			}
		}
	}

	fifo_queue->ntasks++;
	fifo_queue->nprocessed++;

	PTHREAD_COND_SIGNAL(sched_cond);
	PTHREAD_MUTEX_UNLOCK(sched_mutex);

	return 0;
}



static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	struct starpu_fifo_taskq_s *fifo;
	fifo = queue_array[best_workerid];

	fifo->exp_end += predicted;
	fifo->exp_len += predicted;

	task->predicted = predicted;

	unsigned memory_node = starpu_worker_get_memory_node(best_workerid);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, memory_node);

	switch (prio) {
		case 1:
			return _starpu_fifo_push_prio_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
		case 2:
			return _starpu_fifo_push_sorted_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
		default:
			return _starpu_fifo_push_task(queue_array[best_workerid],
				&sched_mutex[best_workerid], &sched_cond[best_workerid], task);
	}
}

static int _dm_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		double exp_end;
		
		fifo = queue_array[worker];

		fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
		fifo->exp_end = STARPU_MAX(fifo->exp_end, starpu_timing_now());

		if (!starpu_worker_may_execute_task(worker, task))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		double local_length = starpu_task_expected_length(task, perf_arch);

		if (local_length == -1.0) 
		{
			/* there is no prediction available for that task
			 * with that arch we want to speed-up calibration time 
			 * so we force this measurement */
			/* XXX assert we are benchmarking ! */
			best = worker;
			model_best = 0.0;
			exp_end = fifo->exp_start + fifo->exp_len;
			break;
		}


		exp_end = fifo->exp_start + fifo->exp_len + local_length;

		if (best == -1 || exp_end < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end;
			best = worker;
			model_best = local_length;
		}
	}
	
	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio);
}

static int _dmda_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker;
	int best = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	double local_task_length[nworkers];
	double local_data_penalty[nworkers];
	double exp_end[nworkers];

	double fitness[nworkers];

	double best_exp_end = 10e240;
	double model_best = 0.0;
	double penality_best = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		fifo = queue_array[worker];

		fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
		fifo->exp_end = STARPU_MAX(fifo->exp_end, starpu_timing_now());

		if (!starpu_worker_may_execute_task(worker, task))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		local_task_length[worker] = starpu_task_expected_length(task, perf_arch);

		unsigned memory_node = starpu_worker_get_memory_node(worker);
		local_data_penalty[worker] = starpu_data_expected_penalty(memory_node, task);

		if (local_task_length[worker] == -1.0)
		{
			forced_best = worker;
			break;
		}

		exp_end[worker] = fifo->exp_start + fifo->exp_len + local_task_length[worker];

		if (exp_end[worker] < best_exp_end)
		{
			/* a better solution was found */
			best_exp_end = exp_end[worker];
		}
	}

	double best_fitness = -1;
	
	if (forced_best == -1)
	{
		for (worker = 0; worker < nworkers; worker++)
		{
			fifo = queue_array[worker];
	
			if (!starpu_worker_may_execute_task(worker, task))
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker] = alpha*(exp_end[worker] - best_exp_end) 
					+ beta*(local_data_penalty[worker]);

			if (best == -1 || fitness[worker] < best_fitness)
			{
				/* we found a better solution */
				best_fitness = fitness[worker];
				best = worker;

	//			_STARPU_DEBUG("best fitness (worker %d) %le = alpha*(%le) + beta(%le) \n", worker, best_fitness, exp_end[worker] - best_exp_end, local_data_penalty[worker]);
			}
		}
	}

	STARPU_ASSERT(forced_best != -1 || best != -1);
	
	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = worker;
		model_best = 0.0;
		penality_best = 0.0;
	}
	else 
	{
		model_best = local_task_length[best];
		penality_best = local_data_penalty[best];
	}

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio);
}

static int dmda_push_sorted_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 2);
}

static int dm_push_prio_task(struct starpu_task *task)
{
	return _dm_push_task(task, 1);
}

static int dm_push_task(struct starpu_task *task)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _dm_push_task(task, 1);

	return _dm_push_task(task, 0);
}

static int dmda_push_prio_task(struct starpu_task *task)
{
	return _dmda_push_task(task, 1);
}

static int dmda_push_task(struct starpu_task *task)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _dmda_push_task(task, 1);

	return _dmda_push_task(task, 0);
}

static void initialize_dmda_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	nworkers = topology->nworkers;

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		beta = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_fifo();
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

static void initialize_dmda_sorted_policy(struct starpu_machine_topology_s *topology,
					struct starpu_sched_policy_s *_policy)
{
	initialize_dmda_policy(topology, _policy);

	/* The application may use any integer */
	starpu_sched_set_min_priority(INT_MIN);
	starpu_sched_set_max_priority(INT_MAX);
}

static void deinitialize_dmda_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	unsigned workerid;
	for (workerid = 0; workerid < topology->nworkers; workerid++)
		_starpu_destroy_fifo(queue_array[workerid]);

	_STARPU_DEBUG("total_task_cnt %ld ready_task_cnt %ld -> %f\n", total_task_cnt, ready_task_cnt, (100.0f*ready_task_cnt)/total_task_cnt);
}

struct starpu_sched_policy_s _starpu_sched_dm_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dm_push_task, 
	.push_prio_task = dm_push_prio_task,
	.pop_task = dmda_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dm",
	.policy_description = "performance model"
};

struct starpu_sched_policy_s _starpu_sched_dmda_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_task, 
	.push_prio_task = dmda_push_prio_task, 
	.pop_task = dmda_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmda",
	.policy_description = "data-aware performance model"
};

struct starpu_sched_policy_s _starpu_sched_dmda_sorted_policy = {
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_sorted_task, 
	.push_prio_task = dmda_push_sorted_task, 
	.pop_task = dmda_pop_ready_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdas",
	.policy_description = "data-aware performance model (sorted)"
};

struct starpu_sched_policy_s _starpu_sched_dmda_ready_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_task, 
	.push_prio_task = dmda_push_prio_task, 
	.pop_task = dmda_pop_ready_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdar",
	.policy_description = "data-aware performance model (ready)"
};
