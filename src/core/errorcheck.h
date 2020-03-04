/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/** This type describes in which state a worker may be. */
enum _starpu_worker_status
{
	/** invalid status (for instance if we request the status of some thread
	 * that is not controlled by StarPU */
	STATUS_INVALID,
	/** everything that does not fit the other status */
	STATUS_UNKNOWN,
	/** during the initialization */
	STATUS_INITIALIZING,
	/** during the execution of a codelet */
	STATUS_EXECUTING,
	/** during the execution of the callback */
	STATUS_CALLBACK,
	/** while executing the scheduler code */
	STATUS_SCHEDULING,
	/** while waiting for a data transfer */
	STATUS_WAITING,
	/** while sleeping because there is nothing to do, but looking for tasks to do */
	STATUS_SLEEPING_SCHEDULING,
	/** while sleeping because there is nothing to do, and not even scheduling */
	STATUS_SLEEPING
};

struct _starpu_worker;
/** Specify what the local worker is currently doing (eg. executing a callback).
 * This permits to detect if this is legal to do a blocking call for instance.
 * */
void _starpu_set_worker_status(struct _starpu_worker *worker, enum _starpu_worker_status st);
void _starpu_set_local_worker_status(enum _starpu_worker_status st);

/** Indicate what type of operation the worker is currently doing. */
enum _starpu_worker_status _starpu_get_local_worker_status(void);

/** It is forbidden to do blocking calls during some operations such as callback
 * or during the execution of a task. This function indicates whether it is
 * legal to call a blocking operation in the current context. */
unsigned _starpu_worker_may_perform_blocking_calls(void);

#endif // __ERRORCHECK_H__
