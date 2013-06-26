/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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

#ifndef __TASK_FIFO_H__
#define __TASK_FIFO_H__

#include <pthread.h>

#include <starpu.h>
#include <common/config.h>


struct mp_task{
	struct _starpu_mp_node *node;
	void (*kernel)(void **, void *);
	void *interfaces[STARPU_NMAXBUFS]; 
	void *cl_arg;
	unsigned coreid;
	enum starpu_codelet_type type;
	int is_parallel_task;
	int combined_worker_size;
	int combined_worker[STARPU_NMAXWORKERS];

	/*the next task of the fifo*/
	struct mp_task * next;
};


struct mp_task_fifo{
	/*the first task of the fifo*/
	struct mp_task * first;
  
	/*the last task of the fifo*/
	struct mp_task * last;

	/*mutex to protect concurrent access on the fifo*/
	pthread_mutex_t mutex;
};


void task_fifo_init(struct mp_task_fifo* fifo);

int task_fifo_is_empty(struct mp_task_fifo* fifo);

void task_fifo_append(struct mp_task_fifo* fifo, struct mp_task * task);

void task_fifo_pop(struct mp_task_fifo* fifo);

#endif /*__TASK_FIFO_H__*/
