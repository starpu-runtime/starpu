/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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
	fifo = (struct _starpu_fifo_taskq *) malloc(sizeof(struct _starpu_fifo_taskq));
	
	/* note that not all mechanisms (eg. the semaphore) have to be used */
	starpu_task_list_init(&fifo->taskq);
	fifo->ntasks = 0;
	fifo->nprocessed = 0;
	
	fifo->exp_start = starpu_timing_now();
	fifo->exp_len = 0.0;
	fifo->exp_end = fifo->exp_start;
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

int
_starpu_fifo_push_sorted_task(struct _starpu_fifo_taskq *fifo_queue, struct starpu_task *task)
{
	struct starpu_task_list *list = &fifo_queue->taskq;
	if (list->head == NULL)
	{
		list->head = task;
		list->tail = task;
		task->prev = NULL;
		task->next = NULL;
	}
	else
	{
		struct starpu_task *current = list->head;
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
			list->head->prev = task;
			task->prev = NULL;
			task->next = list->head;
			list->head = task;
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

struct starpu_task *_starpu_fifo_pop_task(struct _starpu_fifo_taskq *fifo_queue, int workerid)
{
	struct starpu_task *task;
	for (task  = starpu_task_list_begin(&fifo_queue->taskq);
	     task != starpu_task_list_end(&fifo_queue->taskq);
	     task  = starpu_task_list_next(task))
	{
		unsigned nimpl;
		STARPU_ASSERT(task);

		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			if (starpu_worker_can_execute_task(workerid, task, nimpl))
			{
				starpu_task_set_implementation(task, nimpl);
				starpu_task_list_erase(&fifo_queue->taskq, task);
				fifo_queue->ntasks--;
				_STARPU_TRACE_JOB_POP(task, 0);
				return task;
			}
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
		_STARPU_TRACE_JOB_POP(task, 0);
	}

	return task;
}

/* pop every task that can be executed on the calling driver */
struct starpu_task *_starpu_fifo_pop_every_task(struct _starpu_fifo_taskq *fifo_queue, int workerid)
{
	struct starpu_task_list *old_list;
	unsigned size;

	struct starpu_task *new_list = NULL;
	struct starpu_task *new_list_tail = NULL;

	size = fifo_queue->ntasks;

	if (size > 0)
	{
		old_list = &fifo_queue->taskq;
		unsigned new_list_size = 0;

		struct starpu_task *task, *next_task;
		/* note that this starts at the _head_ of the list, so we put
 		 * elements at the back of the new list */
		task = starpu_task_list_front(old_list);
		while (task)
		{
			unsigned nimpl;
			next_task = task->next;

			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			if (starpu_worker_can_execute_task(workerid, task, nimpl))
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
				break;
			}

			task = next_task;
		}

		fifo_queue->ntasks -= new_list_size;
	}

	return new_list;
}
