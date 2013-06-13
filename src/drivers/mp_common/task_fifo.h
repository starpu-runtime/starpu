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


struct task{
  struct _starpu_mp_node *node;
  void (*kernel)(void **, void *);
  void *interfaces[STARPU_NMAXBUFS]; 
  void *cl_arg;
  unsigned coreid;

  /*the next task of the fifo*/
  struct task * next;
};


struct task_fifo{
  /*the first task of the fifo*/
  struct task * first;
  
  /*the last task of the fifo*/
  struct task * last;

  /*mutex to protect concurrent access on the fifo*/
  pthread_mutex_t mutex;
};


void task_fifo_init(struct task_fifo* fifo);

int task_fifo_is_empty(struct task_fifo* fifo);

void task_fifo_append(struct task_fifo* fifo, struct task * task);

void task_fifo_pop(struct task_fifo* fifo);

#endif /* __TASK_FIFO_H__*/
