/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Université de Bordeaux 1
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

#ifndef __SIMGRID_H__
#define __SIMGRID_H__

#ifdef STARPU_SIMGRID
#include <msg/msg.h>

void _starpu_simgrid_execute_job(struct _starpu_job *job, enum starpu_perf_archtype perf_arch);
msg_task_t _starpu_simgrid_transfer_task_create(unsigned src_node, unsigned dst_node, size_t size);
void _starpu_simgrid_post_task(msg_task_t task, unsigned *finished, _starpu_pthread_mutex_t *mutex, _starpu_pthread_cond_t *cond);
#endif

#endif // __SIMGRID_H__
