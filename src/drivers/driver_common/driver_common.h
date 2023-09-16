/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/** The task job is about to start (or has already started when kernels are
 * queued in a pipeline), record profiling and trace information. */
void _starpu_driver_start_job(struct _starpu_worker *args, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch,
			      int rank, int profiling);
/** The task job has ended, record profiling and trace information. */
void _starpu_driver_end_job(struct _starpu_worker *args, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch,
			    int rank, int profiling);
/** Feed performance model with the terminated job statistics */
void _starpu_driver_update_job_feedback(struct _starpu_job *j, struct _starpu_worker *worker_args,
					struct starpu_perfmodel_arch* perf_arch, int profiling);

#pragma GCC visibility push(hidden)

/** Get from the scheduler a task to be executed on the worker \p workerid */
struct starpu_task *_starpu_get_worker_task(struct _starpu_worker *args, int workerid, unsigned memnode);
/** Get from the scheduler tasks to be executed on the workers \p workers */
int _starpu_get_multi_worker_task(struct _starpu_worker *workers, struct starpu_task ** tasks, int nworker, unsigned memnode);

void *_starpu_map_allocate(size_t length, unsigned node);
int _starpu_map_deallocate(void* map_addr, size_t length);
char* _starpu_get_fdname_from_mapaddr(uintptr_t map_addr, size_t *offset, size_t length);
void *_starpu_sink_map(char *fd_name, size_t offset, size_t length);
int _starpu_sink_unmap(uintptr_t map_addr, size_t length);

#pragma GCC visibility pop

#endif // __DRIVER_COMMON_H__
