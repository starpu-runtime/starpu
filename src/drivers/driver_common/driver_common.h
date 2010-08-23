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

#ifndef __DRIVER_COMMON_H__
#define __DRIVER_COMMON_H__

#include <sys/time.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <core/jobs.h>
#include <profiling/profiling.h>
#include <common/utils.h>

void _starpu_driver_update_job_feedback(starpu_job_t j, struct starpu_worker_s *worker_args,
		struct starpu_task_profiling_info *profiling_info,
		unsigned calibrate_model,
		struct timespec *codelet_start, struct timespec *codelet_end);

void _starpu_block_worker(int workerid, pthread_cond_t *cond, pthread_mutex_t *mutex);

#endif // __DRIVER_COMMON_H__
