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

#ifndef __STARPU_PROFILING_H__
#define __STARPU_PROFILING_H__

#include <errno.h>
#include <starpu.h>

#define STARPU_PROFILING_DISABLE	0
#define STARPU_PROFILING_ENABLE		1

/* -ENOSYS is returned in case the info is not available. Timing are shown in
 * microseconds. */
struct starpu_task_profiling_info {
	int64_t submit_time;
	int64_t start_time;
	int64_t end_time;
	/* TODO add expected length, expected start/end ? */
	int workerid;
};

/* The timing is provided since the previous call to starpu_worker_get_profiling_info */
struct starpu_worker_profiling_info {
	int64_t start_time;
	int64_t total_time;
	int64_t executing_time;
	int64_t sleeping_time;
	int executed_tasks;
};

/* This function sets the profiling status:
 * - enable with STARPU_PROFILING_ENABLE
 * - disable with STARPU_PROFILING_DISABLE 
 * Negative return values indicate an error, otherwise the previous status is
 * returned. Calling this function resets the profiling measurements. */
int starpu_profiling_status_set(int status);

/* Return the current profiling status or a negative value in case there was an
 * error. */
int starpu_profiling_status_get(void);

/* Get the profiling info associated to a worker, and reset the profiling
 * measurements. If worker_info is NULL, we only reset the counters. */
int starpu_worker_get_profiling_info(int workerid, struct starpu_worker_profiling_info *worker_info);

#endif // __STARPU_PROFILING_H__
