/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2016       Uppsala University
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

/* FIFO queues, ready for use by schedulers */

#include <starpu_scheduler.h>

#include <sched_policies/fifo_queues.h>
#include <common/fxt.h>
#include <core/topology.h>
#include <limits.h>
/*
static int is_sorted_task_list(struct starpu_task * task)
{
	if(!task)
		return 1;
	struct starpu_task * next = task->next;
	if(!next)
		return 1;
	while(next)
	{
		if(task->priority < next->priority)
			return 0;
		task = next;
		next = next->next;
	}
	return 1;
}
*/

struct _starpu_fifo_taskq *_starpu_create_fifo(void)
{
	struct _starpu_fifo_taskq *fifo;
	_STARPU_MALLOC(fifo, sizeof(struct _starpu_fifo_taskq));

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	starpu_task_list_init(&fifo->taskq);
	fifo->ntasks = 0;
	STARPU_HG_DISABLE_CHECKING(fifo->ntasks);
	fifo->nprocessed = 0;

	fifo->exp_start = starpu_timing_now();
	fifo->exp_len = 0.0;
	fifo->exp_end = fifo->exp_start;
	fifo->exp_len_per_priority = NULL;
	fifo->pipeline_len = 0.0;

	return fifo;
}

void _starpu_destroy_fifo(struct _starpu_fifo_taskq *fifo)
{
	free(fifo);
}

int _starpu_fifo_empty(struct _starpu_fifo_taskq *fifo)
{
	return fifo->ntasks == 0;
}

double 
_starpu_fifo_get_exp_len_prev_task_list(struct _starpu_fifo_taskq *fifo_queue, struct starpu_task *task, int workerid, int nimpl, int *fifo_ntasks)
{
	struct starpu_task_list *list = &fifo_queue->taskq;
	struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerid, task->sched_ctx);
	double exp_len = 0.0;
	
	if (list->_head != NULL)
	{
		struct starpu_task *current = list->_head;
		struct starpu_task *prev = NULL;

		if (list->_head->priority == task->priority &&
		    list->_head->priority == list->_tail->priority)
		{
			/* They all have the same priority, the task's place is at the end */
			prev = list->_tail;
			current = NULL;
		}
		else
		while (current)
		{
			if (current->priority < task->priority)
				break;

			prev = current;
			current = current->next;
		}

		if (prev != NULL)
		{
			if (current)
			{
				/* the task's place is between prev and current */
				struct starpu_task *it;
				for(it = list->_head; it != current; it = it->next)
				{
					exp_len += starpu_task_expected_length(it, perf_arch, nimpl);
					(*fifo_ntasks) ++;
				}
			}
			else
			{
				/* the task's place is at the _tail of the list */
				exp_len = fifo_queue->exp_len;
				*fifo_ntasks = fifo_queue->ntasks;
			}
		}
	}


	return exp_len;
}

int
_starpu_fifo_push_sorted_task(struct _starpu_fifo_taskq *fifo_queue, struct starpu_task *task)
{
	struct starpu_task_list *list = &fifo_queue->taskq;

	if (list->_head == NULL)
	{
		list->_head = task;
		list->_tail = task;
		task->prev = NULL;
		task->next = NULL;
	}
	else if (list->_head->priority == task->priority &&
		 list->_head->priority == list->_tail->priority)
	{
		/* They all have the same priority, just put at the end */
		list->_tail->next = task;
		task->next = NULL;
		task->prev = list->_tail;
		list->_tail = task;
	}
	else
	{
		struct starpu_task *current = list->_head;
		struct starpu_task *prev = NULL;

		while (current)
		{
			if (current->priority < task->priority)
				break;

			prev = current;
			current = current->next;
		}

		if (prev == NULL)
		{
			/* Insert at the front of the list */
			list->_head->prev = task;
			task->prev = NULL;
			task->next = list->_head;
			list->_head = task;
		}
		else
		{
			if (current)
			{
				/* Insert between prev and current */
				task->prev = prev;
				prev->next = task;
				task->next = current;
				current->prev = task;
			}
			else
			{
				/* Insert at the _tail of the list */
				list->_tail->next = task;
				task->next = NULL;
				task->prev = list->_tail;
				list->_tail = task;
			}
		}
	}

	fifo_queue->ntasks++;
	fifo_queue->nprocessed++;

	return 0;
}

int _starpu_fifo_push_task(struct _starpu_fifo_taskq *fifo_queue, struct starpu_task *task)
{

	if (task->priority > 0)
	{
		_starpu_fifo_push_sorted_task(fifo_queue, task);
	}
	else
	{
		starpu_task_list_push_back(&fifo_queue->taskq, task);

		fifo_queue->ntasks++;
		fifo_queue->nprocessed++;
	}

	return 0;
}

int _starpu_fifo_push_back_task(struct _starpu_fifo_taskq *fifo_queue, struct starpu_task *task)
{

	if (task->priority > 0)
	{
		_starpu_fifo_push_sorted_task(fifo_queue, task);
	}
	else
	{
		starpu_task_list_push_front(&fifo_queue->taskq, task);

		fifo_queue->ntasks++;
	}

	return 0;
}

int _starpu_fifo_pop_this_task(struct _starpu_fifo_taskq *fifo_queue, int workerid, struct starpu_task *task)
{
	unsigned nimpl = 0;
	STARPU_ASSERT(task);
#ifdef STARPU_DEBUG
	STARPU_ASSERT(starpu_task_list_ismember(&fifo_queue->taskq, task));
#endif

	if (workerid < 0 || starpu_worker_can_execute_task_first_impl(workerid, task, &nimpl))
	{
		starpu_task_set_implementation(task, nimpl);
		starpu_task_list_erase(&fifo_queue->taskq, task);
		fifo_queue->ntasks--;
		return 1;
	}

	return 0;
}

struct starpu_task *_starpu_fifo_pop_task(struct _starpu_fifo_taskq *fifo_queue, int workerid)
{
	struct starpu_task *task;

	for (task  = starpu_task_list_begin(&fifo_queue->taskq);
	     task != starpu_task_list_end(&fifo_queue->taskq);
	     task  = starpu_task_list_next(task))
	{
		if (_starpu_fifo_pop_this_task(fifo_queue, workerid, task))
			return task;
	}

	return NULL;
}

/* This is the same as _starpu_fifo_pop_task, but without checking that the
 * worker will be able to execute this task. This is useful when the scheduler
 * has already checked it. */
struct starpu_task *_starpu_fifo_pop_local_task(struct _starpu_fifo_taskq *fifo_queue)
{
	struct starpu_task *task = NULL;

	if (!starpu_task_list_empty(&fifo_queue->taskq))
	{
		task = starpu_task_list_pop_front(&fifo_queue->taskq);
		fifo_queue->ntasks--;
	}

	return task;
}

/* pop every task that can be executed on the calling driver */
struct starpu_task *_starpu_fifo_pop_every_task(struct _starpu_fifo_taskq *fifo_queue, int workerid)
{
	unsigned size = fifo_queue->ntasks;
	struct starpu_task *new_list = NULL;

	if (size > 0)
	{
		struct starpu_task_list *old_list = &fifo_queue->taskq;
		struct starpu_task *new_list_tail = NULL;
		unsigned new_list_size = 0;

		struct starpu_task *task, *next_task;
		/* note that this starts at the _head_ of the list, so we put
 		 * elements at the back of the new list */
		task = starpu_task_list_front(old_list);
		while (task)
		{
			unsigned nimpl;
			next_task = task->next;

			if (starpu_worker_can_execute_task_first_impl(workerid, task, &nimpl))
			{
				/* this elements can be moved into the new list */
				new_list_size++;

				starpu_task_list_erase(old_list, task);

				if (new_list_tail)
				{
					new_list_tail->next = task;
					task->prev = new_list_tail;
					task->next = NULL;
					new_list_tail = task;
				}
				else
				{
					new_list = task;
					new_list_tail = task;
					task->prev = NULL;
					task->next = NULL;
				}
				starpu_task_set_implementation(task, nimpl);
			}

			task = next_task;
		}

		fifo_queue->ntasks -= new_list_size;
	}

	return new_list;
}

int _starpu_normalize_prio(int priority, int num_priorities, unsigned sched_ctx_id)
{
	int min = starpu_sched_ctx_get_min_priority(sched_ctx_id);
	int max = starpu_sched_ctx_get_max_priority(sched_ctx_id);
	return ((num_priorities-1)/(max-min)) * (priority - min);
}

int _starpu_count_non_ready_buffers(struct starpu_task *task, unsigned worker)
{
	int cnt = 0;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned index;

	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle;
		unsigned buffer_node = _starpu_task_data_get_node_on_worker(task, index, worker);

		handle = STARPU_TASK_GET_HANDLE(task, index);

		int is_valid;
		starpu_data_query_status(handle, buffer_node, NULL, &is_valid, NULL);

		if (!is_valid)
			cnt++;
	}

	return cnt;
}

struct starpu_task *_starpu_fifo_pop_first_ready_task(struct _starpu_fifo_taskq *fifo_queue, unsigned workerid, int num_priorities)
{
	struct starpu_task *task = NULL, *current;

	if (fifo_queue->ntasks == 0)
		return NULL;

	if (fifo_queue->ntasks > 0)
	{
		fifo_queue->ntasks--;

		task = starpu_task_list_front(&fifo_queue->taskq);
		if (STARPU_UNLIKELY(!task))
			return NULL;

		int first_task_priority = task->priority;

		int non_ready_best = INT_MAX;

		for (current = task; current; current = current->next)
		{
			int priority = current->priority;

			if (priority >= first_task_priority)
			{
				int non_ready = _starpu_count_non_ready_buffers(current, workerid);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					task = current;

					if (non_ready == 0)
						break;
				}
			}
		}

		if(num_priorities != -1)
		{
			int i;
			int task_prio = _starpu_normalize_prio(task->priority, num_priorities, task->sched_ctx);
			for(i = 0; i <= task_prio; i++)
				fifo_queue->ntasks_per_priority[i]--;
		}

		starpu_task_list_erase(&fifo_queue->taskq, task);
	}

	return task;
}

