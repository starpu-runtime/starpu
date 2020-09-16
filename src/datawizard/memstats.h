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

#ifndef __MEMSTATS_H__
#define __MEMSTATS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>

#ifdef STARPU_MEMORY_STATS
struct _starpu_memory_stats
{
	/** Handle access stats per node */
	unsigned direct_access[STARPU_MAXNODES];
	unsigned loaded_shared[STARPU_MAXNODES];
	unsigned loaded_owner[STARPU_MAXNODES];
	unsigned shared_to_owner[STARPU_MAXNODES];
	unsigned invalidated[STARPU_MAXNODES];
};

typedef	struct _starpu_memory_stats * _starpu_memory_stats_t;
#else
typedef void * _starpu_memory_stats_t;
#endif

void _starpu_memory_stats_init(starpu_data_handle_t handle);
void _starpu_memory_stats_init_per_node(starpu_data_handle_t handle, unsigned node);

void _starpu_memory_stats_free(starpu_data_handle_t handle);

void _starpu_memory_display_handle_stats(FILE *stream, starpu_data_handle_t handle);

void _starpu_memory_handle_stats_cache_hit(starpu_data_handle_t handle, unsigned node);
void _starpu_memory_handle_stats_loaded_shared(starpu_data_handle_t handle, unsigned node);
void _starpu_memory_handle_stats_loaded_owner(starpu_data_handle_t handle, unsigned node);
void _starpu_memory_handle_stats_shared_to_owner(starpu_data_handle_t handle, unsigned node);
void _starpu_memory_handle_stats_invalidated(starpu_data_handle_t handle, unsigned node);

#endif /* __MEMSTATS_H__ */
