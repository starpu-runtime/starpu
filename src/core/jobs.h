/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#ifndef __JOBS_H__
#define __JOBS_H__

#include <starpu.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <common/config.h>
#include <common/timing.h>
#include <common/list.h>
#include <common/fxt.h>
#include <core/dependencies/tags.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>
#include <core/errorcheck.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

struct worker_s;

/* codelet function */
typedef void (*cl_func)(void **, void *);
typedef void (*callback)(void *);

#define CORE_MAY_PERFORM(j)	((j)->task->cl->where & CORE)
#define CUDA_MAY_PERFORM(j)     ((j)->task->cl->where & CUDA)
#define SPU_MAY_PERFORM(j)	((j)->task->cl->where & SPU)
#define GORDON_MAY_PERFORM(j)	((j)->task->cl->where & GORDON)

/* a job is the internal representation of a task */
LIST_TYPE(job,
	struct starpu_task *task;

	pthread_mutex_t sync_mutex;
	pthread_cond_t sync_cond;

	struct tag_s *tag;

	double predicted;
	double penality;

	unsigned footprint_is_computed;
	uint32_t footprint;

	unsigned terminated;
);

job_t __attribute__((malloc)) _starpu_job_create(struct starpu_task *task);
void starpu_wait_job(job_t j);

/* try to submit job j, enqueue it if it's not schedulable yet */
unsigned _starpu_enforce_deps_and_schedule(job_t j);

//#warning this must not be exported anymore ... 
//job_t _starpu_job_create(struct starpu_task *task);
void _starpu_handle_job_termination(job_t j);
size_t _starpu_job_get_data_size(job_t j);

job_t _starpu_pop_local_task(struct worker_s *worker);
int _starpu_push_local_task(struct worker_s *worker, job_t j);

#endif // __JOBS_H__
