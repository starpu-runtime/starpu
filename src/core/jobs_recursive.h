/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __JOBS_SPLITTER_H__
#define __JOBS_SPLITTER_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <sched_policies/splitter.h>

#pragma GCC visibility push(hidden)

extern int (*splitter_policy) (struct _starpu_job*);
extern unsigned call_splitter_on_scheduler;
extern unsigned liberate_deps_on_first_exec;

void _starpu_job_splitter_policy_init();
void _starpu_rec_task_deinit(void);
void _starpu_job_splitter_termination(struct _starpu_job *j);
void _starpu_job_splitter_destroy(struct _starpu_job *j);
void _starpu_job_splitter_liberate_parent(struct _starpu_job *j);

long _starpu_get_cuda_executor_id(struct starpu_task *t);
void _starpu_set_cuda_executor_id(struct starpu_task *t, long id);
void _starpu_recursive_job_destroy(struct _starpu_job *j);

#pragma GCC visibility pop

#endif // __JOBS_SPLITTER_H__
