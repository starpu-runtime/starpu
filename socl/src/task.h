/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

#ifndef SOCL_TASK_H
#define SOCL_TASK_H

#include "socl.h"

starpu_task * task_create(cl_command_type type);
starpu_task * task_create_with_event(cl_command_type type, cl_event event);
void task_dependency_add(starpu_task * task, cl_uint num, const cl_event *events);
starpu_task * task_create_cpu(cl_command_type type, void (*callback)(void*), void *arg, int free_arg);

/** 
 * Return event associated to a task
 */
cl_event task_event(starpu_task *task);

/**
 * Submit "task" with "events" dependencies
 */
cl_int task_submit(starpu_task * task, cl_int num_events, cl_event * events);

#endif /* SOCL_TASK_H */
