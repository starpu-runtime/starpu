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

#ifndef __ERRORCHECK_H__
#define __ERRORCHECK_H__

#include <starpu.h>

typedef enum {
	/* invalid status (for instance if we request the status of some thread
	 * that is not controlled by StarPU */
	STATUS_INVALID,
	/* everything that does not fit the other status */
	STATUS_UNKNOWN,
	/* during the initialization */
	STATUS_INITIALIZING,
	/* during the execution of a codelet */
	STATUS_EXECUTING,
	/* during the execution of the callback */
	STATUS_CALLBACK
} worker_status;

void _starpu_set_local_worker_status(worker_status st);
worker_status _starpu_get_local_worker_status(void);

unsigned _starpu_worker_may_perform_blocking_calls(void);

#endif // __ERRORCHECK_H__
