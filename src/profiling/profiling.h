/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <starpu_profiling.h>
#include <common/config.h>

struct starpu_task_profiling_info *_starpu_allocate_profiling_info_if_needed(void);
void _starpu_worker_reset_profiling_info(int workerid);
void _starpu_worker_update_profiling_info(int workerid, int64_t executing_time, int64_t sleeping_time, int executed_tasks);

#endif // __PROFILING_H__
