/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __DRIVER_COMMON_H__
#define __DRIVER_COMMON_H__

/** @file */

#include <starpu.h>
#include <starpu_util.h>
#include <core/jobs.h>
#include <common/utils.h>

void _starpu_driver_start_job(struct _starpu_worker *args, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch,
			      int rank, int profiling);
void _starpu_driver_end_job(struct _starpu_worker *args, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch,
			    int rank, int profiling);
void _starpu_driver_update_job_feedback(struct _starpu_job *j, struct _starpu_worker *worker_args,
					struct starpu_perfmodel_arch* perf_arch, int profiling);

struct starpu_task *_starpu_get_worker_task(struct _starpu_worker *args, int workerid, unsigned memnode);
int _starpu_get_multi_worker_task(struct _starpu_worker *workers, struct starpu_task ** tasks, int nworker, unsigned memnode);
#endif // __DRIVER_COMMON_H__
