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

#ifndef __BOUND_H__
#define __BOUND_H__

#include <starpu.h>
#include <starpu_bound.h>
#include <core/jobs.h>

/* Record task for bound computation */
extern void _starpu_bound_record(starpu_job_t j);

/* Record tag dependency */
extern void _starpu_bound_tag_dep(starpu_tag_t id, starpu_tag_t dep_id);

/* Record task dependency */
extern void _starpu_bound_task_dep(starpu_job_t j, starpu_job_t dep_j);

#endif // __BOUND_H__
