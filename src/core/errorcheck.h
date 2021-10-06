/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __ERRORCHECK_H__
#define __ERRORCHECK_H__

/** @file */

#include <starpu.h>

#pragma GCC visibility push(hidden)

/** This type enumerates the actions that can be done by a worker.
 * Some can be happening during others, that is why
 * enum _starpu_worker_status
 * is a bitset indexed by the values of enum _starpu_worker_status_index.
 */
enum _starpu_worker_status_index
{
	STATUS_INDEX_INITIALIZING = 0,
	STATUS_INDEX_EXECUTING,
	STATUS_INDEX_CALLBACK,
	STATUS_INDEX_WAITING,
	STATUS_INDEX_SLEEPING,
	STATUS_INDEX_SCHEDULING,
	STATUS_INDEX_NR,
};

/** This type describes in which state a worker may be. */
enum _starpu_worker_status
{
	/** invalid status (for instance if we request the status of some thread
	 * that is not controlled by StarPU */
	STATUS_INVALID = -1,
	/** Nothing particular, thus just overhead */
	STATUS_UNKNOWN = 0,
	/** during the initialization */
	STATUS_INITIALIZING = 1 << STATUS_INDEX_INITIALIZING,
	/** during the execution of a codelet */
	STATUS_EXECUTING = 1 << STATUS_INDEX_EXECUTING,
	/** during the execution of the callback */
	STATUS_CALLBACK = 1 << STATUS_INDEX_CALLBACK,
	/** while waiting for a data transfer */
	STATUS_WAITING = 1 << STATUS_INDEX_WAITING,
	/** while sleeping because there is no task to do */
	STATUS_SLEEPING = 1 << STATUS_INDEX_SLEEPING,
	/** while executing the scheduler code */
	STATUS_SCHEDULING = 1 << STATUS_INDEX_SCHEDULING,
};

struct _starpu_worker;
/** Specify what the local worker is currently doing (eg. executing a callback).
 * This permits to detect if this is legal to do a blocking call for instance. */
void _starpu_add_worker_status(struct _starpu_worker *worker, enum _starpu_worker_status_index st, struct timespec *time);
void _starpu_add_local_worker_status(enum _starpu_worker_status_index st, struct timespec *time);

/** Clear the fact that the local worker was currently doing something(eg. executing a callback). */
void _starpu_clear_worker_status(struct _starpu_worker *worker, enum _starpu_worker_status_index st, struct timespec *time);
void _starpu_clear_local_worker_status(enum _starpu_worker_status_index st, struct timespec *time);

/** Indicate what type of operation the worker is currently doing. */
enum _starpu_worker_status _starpu_get_local_worker_status(void);

/** It is forbidden to do blocking calls during some operations such as callback
 * or during the execution of a task. This function indicates whether it is
 * legal to call a blocking operation in the current context. */
unsigned _starpu_worker_may_perform_blocking_calls(void);

#pragma GCC visibility pop

#endif // __ERRORCHECK_H__
