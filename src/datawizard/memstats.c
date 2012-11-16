/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <datawizard/memstats.h>
#include <common/config.h>
#include <datawizard/coherency.h>

void _starpu_memory_stats_init(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_MEMORY_STATS
	handle->memory_stats = calloc(1, sizeof(struct _starpu_memory_stats));
#endif
}

void _starpu_memory_stats_init_per_node(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_MEMORY_STATS
	/* Stats initilization */
	handle->memory_stats->direct_access[node]=0;
	handle->memory_stats->loaded_shared[node]=0;
	handle->memory_stats->shared_to_owner[node]=0;
	handle->memory_stats->loaded_owner[node]=0;
	handle->memory_stats->invalidated[node]=0;
#endif
}

void _starpu_memory_stats_free(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_MEMORY_STATS
	free(handle->memory_stats);
#endif
}

#ifdef STARPU_MEMORY_STATS
void _starpu_memory_display_handle_stats(starpu_data_handle_t handle)
{
	unsigned node;

	fprintf(stderr, "#-----\n");
	fprintf(stderr, "Data : %p\n", handle);
	fprintf(stderr, "Size : %d\n", (int)handle->data_size);
	fprintf(stderr, "\n");

	fprintf(stderr, "#--\n");
	fprintf(stderr, "Data access stats\n");
	fprintf(stderr, "/!\\ Work Underway\n");
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (handle->memory_stats->direct_access[node]+handle->memory_stats->loaded_shared[node]
		    +handle->memory_stats->invalidated[node]+handle->memory_stats->loaded_owner[node])
		{
			fprintf(stderr, "Node #%d\n", node);
			fprintf(stderr, "\tDirect access : %d\n", handle->memory_stats->direct_access[node]);
			/* XXX Not Working yet. */
			if (handle->memory_stats->shared_to_owner[node])
				fprintf(stderr, "\t\tShared to Owner : %d\n", handle->memory_stats->shared_to_owner[node]);
			fprintf(stderr, "\tLoaded (Owner) : %d\n", handle->memory_stats->loaded_owner[node]);
			fprintf(stderr, "\tLoaded (Shared) : %d\n", handle->memory_stats->loaded_shared[node]);
			fprintf(stderr, "\tInvalidated (was Owner) : %d\n\n", handle->memory_stats->invalidated[node]);
		}
	}
}

void _starpu_memory_handle_stats_cache_hit(starpu_data_handle_t handle, unsigned node)
{
	handle->memory_stats->direct_access[node]++;
}

void _starpu_memory_handle_stats_loaded_shared(starpu_data_handle_t handle, unsigned node)
{
	handle->memory_stats->loaded_shared[node]++;
}

void _starpu_memory_handle_stats_loaded_owner(starpu_data_handle_t handle, unsigned node)
{
	handle->memory_stats->loaded_owner[node]++;
}

void _starpu_memory_handle_stats_shared_to_owner(starpu_data_handle_t handle, unsigned node)
{
	handle->memory_stats->shared_to_owner[node]++;
}

void _starpu_memory_handle_stats_invalidated(starpu_data_handle_t handle, unsigned node)
{
	handle->memory_stats->invalidated[node]++;
}

#endif



