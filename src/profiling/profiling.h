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

#ifndef __PROFILING_H__
#define __PROFILING_H__

#include <sys/time.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <common/config.h>

struct starpu_task_profiling_info *_starpu_allocate_profiling_info_if_needed(void);
void _starpu_worker_reset_profiling_info(int workerid);
void _starpu_worker_update_profiling_info_executing(int workerid, struct timespec *executing_time, int executed_tasks);
void _starpu_worker_update_profiling_info_sleeping(int workerid, struct timespec *sleeping_start, struct timespec *sleeping_end);
void _starpu_worker_register_sleeping_start_date(int workerid, struct timespec *sleeping_start);
void _starpu_worker_register_executing_start_date(int workerid, struct timespec *executing_start);

void _starpu_initialize_busid_matrix(void);
int _starpu_register_bus(int src_node, int dst_node);
void _starpu_bus_update_profiling_info(int src_node, int dst_node, size_t size);

#endif // __PROFILING_H__
