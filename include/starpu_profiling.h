/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#ifndef __STARPU_PROFILING_H__
#define __STARPU_PROFILING_H__

#include <errno.h>
#include <starpu.h>

/* -ENOSYS is returned in case the info is not available */
struct starpu_task_profiling_info {
	int64_t submit_time;
	int64_t start_time;
	int64_t end_time;
	int workerid;
};

/* Enable the profiling and return the previous profiling status (0 if
 * disabled, 1 if enabled). */
unsigned starpu_enable_profiling(void);

/* Disable the profiling and return the previous profiling status (0 if
 * disabled, 1 if enabled). */
unsigned starpu_disable_profiling(void);

#endif // __STARPU_PROFILING_H__
