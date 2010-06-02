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

#include <starpu.h>
#include <starpu_profiling.h>
#include <common/config.h>

/* Disabled by default */
static unsigned profiling = 0;

unsigned starpu_enable_profiling(void)
{
	unsigned prev_value = profiling;
	profiling = 1;

	return prev_value;
}

unsigned starpu_disable_profiling(void)
{
	unsigned prev_value = profiling;
	profiling = 0;

	return prev_value;
}

struct starpu_task_profiling_info *_starpu_allocate_profiling_info_if_needed(void)
{
	struct starpu_task_profiling_info *info = NULL;

	if (profiling)
	{
		info = calloc(1, sizeof(struct starpu_task_profiling_info));
		STARPU_ASSERT(info);
	}

	return info;
}
