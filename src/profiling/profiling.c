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

#include <starpu.h>
#include <starpu_profiling.h>
#include <profiling/profiling.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/timing.h>
#include <errno.h>

static struct starpu_worker_profiling_info worker_info[STARPU_NMAXWORKERS];
static pthread_mutex_t worker_info_mutex[STARPU_NMAXWORKERS];

/*
 *	Global control of profiling
 */

/* Disabled by default */
static int profiling = 0;

int starpu_profiling_status_set(int status)
{
	int prev_value = profiling;
	profiling = status;

	/* If we enable profiling, we reset the counters. */
	if (status == STARPU_PROFILING_ENABLE)
	{
		int worker;
		for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
			_starpu_worker_reset_profiling_info(worker);
	}

	return prev_value;
}

int starpu_profiling_status_get(void)
{
	return profiling;
}

void starpu_profiling_init(void)
{
	int worker;
	for (worker = 0; worker < STARPU_NMAXWORKERS; worker++)
	{
		PTHREAD_MUTEX_INIT(&worker_info_mutex[worker], NULL);
		_starpu_worker_reset_profiling_info(worker);
	}
}

void starpu_profiling_terminate(void)
{

}

/*
 *	Task profiling
 */

struct starpu_task_profiling_info *_starpu_allocate_profiling_info_if_needed(void)
{
	struct starpu_task_profiling_info *info = NULL;

	if (profiling)
	{
		info = calloc(1, sizeof(struct starpu_task_profiling_info));
		STARPU_ASSERT(info);

		info->submit_time = -ENOSYS;
		info->start_time = -ENOSYS;
		info->end_time = -ENOSYS;
	}

	return info;
}

/*
 *	Worker profiling
 */

static void _do_starpu_worker_reset_profiling_info(int workerid)
{
	worker_info[workerid].start_time = (int64_t)_starpu_timing_now();

	/* This is computed in a lazy fashion when the application queries
	 * profiling info. */
	worker_info[workerid].total_time = -ENOSYS;

	worker_info[workerid].executing_time = 0;
	worker_info[workerid].sleeping_time = 0;
	worker_info[workerid].executed_tasks = 0;
}

void _starpu_worker_reset_profiling_info(int workerid)
{
	PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);
	_do_starpu_worker_reset_profiling_info(workerid);
	PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
}

void _starpu_worker_update_profiling_info(int workerid, int64_t executing_time,
					int64_t sleeping_time, int executed_tasks)
{
	if (profiling)
	{
		PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);
	
		worker_info[workerid].executing_time += executing_time;
		worker_info[workerid].sleeping_time += sleeping_time;
		worker_info[workerid].executed_tasks += executed_tasks;
	
		PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);
	}
}

int starpu_worker_get_profiling_info(int workerid, struct starpu_worker_profiling_info *info)
{
	if (!profiling)
		return -EINVAL;

	PTHREAD_MUTEX_LOCK(&worker_info_mutex[workerid]);

	if (info)
	{
		/* The total time is computed in a lazy fashion */
		int64_t total_time = ((int64_t)_starpu_timing_now()) - worker_info[workerid].start_time;
		worker_info[workerid].total_time = total_time;

		memcpy(info, &worker_info[workerid], sizeof(struct starpu_worker_profiling_info));
	}

	_do_starpu_worker_reset_profiling_info(workerid);

	PTHREAD_MUTEX_UNLOCK(&worker_info_mutex[workerid]);

	return 0;
}
