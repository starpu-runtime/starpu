/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#ifndef __BOUND_H__
#define __BOUND_H__

#include <starpu.h>
#include <starpu_bound.h>
#include <core/jobs.h>

/* Are we recording? */
extern int _starpu_bound_recording;

/* Record task for bound computation */
extern void _starpu_bound_record(starpu_job_t j);

/* Record tag dependency: id depends on dep_id */
extern void _starpu_bound_tag_dep(starpu_tag id, starpu_tag dep_id);

/* Record task dependency: j depends on dep_j */
extern void _starpu_bound_task_dep(starpu_job_t j, starpu_job_t dep_j);

/* Record job id dependency: j depends on job_id */
extern void _starpu_bound_job_id_dep(starpu_job_t dep_j, unsigned long job_id);

#endif // __BOUND_H__
