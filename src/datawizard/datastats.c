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

#include <starpu.h>
#include <datawizard/datastats.h>
#include <datawizard/coherency.h>
#include <datawizard/memory_nodes.h>
#include <common/config.h>

int _starpu_enable_stats = 0;

void _starpu_datastats_init()
{
	_starpu_enable_stats = !!starpu_getenv("STARPU_ENABLE_STATS");
}

/* measure the cache hit ratio for each node */
static unsigned hit_cnt[STARPU_MAXNODES];
static unsigned miss_cnt[STARPU_MAXNODES];

void __starpu_msi_cache_hit(unsigned node)
{
	STARPU_HG_DISABLE_CHECKING(hit_cnt[node]);
	hit_cnt[node]++;
}

void __starpu_msi_cache_miss(unsigned node)
{
	STARPU_HG_DISABLE_CHECKING(miss_cnt[node]);
	miss_cnt[node]++;
}

void _starpu_display_msi_stats(FILE *stream)
{
	if (!starpu_enable_stats())
		return;

	unsigned node;
	unsigned total_hit_cnt = 0;
	unsigned total_miss_cnt = 0;

	fprintf(stream, "\n#---------------------\n");
	fprintf(stream, "MSI cache stats :\n");

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		total_hit_cnt += hit_cnt[node];
		total_miss_cnt += miss_cnt[node];
	}

	fprintf(stream, "TOTAL MSI stats\thit %u (%2.2f %%)\tmiss %u (%2.2f %%)\n", total_hit_cnt, (100.0f*total_hit_cnt)/(total_hit_cnt+total_miss_cnt), total_miss_cnt, (100.0f*total_miss_cnt)/(total_hit_cnt+total_miss_cnt));

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (hit_cnt[node]+miss_cnt[node])
		{
			char name[128];
			starpu_memory_node_get_name(node, name, sizeof(name));
			fprintf(stream, "memory node %s\n", name);
			fprintf(stream, "\thit : %u (%2.2f %%)\n", hit_cnt[node], (100.0f*hit_cnt[node])/(hit_cnt[node]+miss_cnt[node]));
			fprintf(stream, "\tmiss : %u (%2.2f %%)\n", miss_cnt[node], (100.0f*miss_cnt[node])/(hit_cnt[node]+miss_cnt[node]));
		}
	}
	fprintf(stream, "#---------------------\n");
}

/* measure the efficiency of our allocation cache */
static unsigned alloc_cnt[STARPU_MAXNODES];
static unsigned alloc_cache_hit_cnt[STARPU_MAXNODES];

void __starpu_allocation_cache_hit(unsigned node)
{
	STARPU_HG_DISABLE_CHECKING(alloc_cache_hit_cnt[node]);
	alloc_cache_hit_cnt[node]++;
}

void __starpu_data_allocation_inc_stats(unsigned node)
{
	STARPU_HG_DISABLE_CHECKING(alloc_cnt[node]);
	alloc_cnt[node]++;
}

void _starpu_display_alloc_cache_stats(FILE *stream)
{
	if (!starpu_enable_stats())
		return;

	fprintf(stream, "\n#---------------------\n");
	fprintf(stream, "Allocation cache stats:\n");
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (alloc_cnt[node])
		{
			char name[128];
			starpu_memory_node_get_name(node, name, sizeof(name));
			fprintf(stream, "memory node %s\n", name);
			fprintf(stream, "\ttotal alloc : %u\n", alloc_cnt[node]);
			fprintf(stream, "\tcached alloc: %u (%2.2f %%)\n",
				alloc_cache_hit_cnt[node], (100.0f*alloc_cache_hit_cnt[node])/(alloc_cnt[node]));
		}
	}
	fprintf(stream, "#---------------------\n");
}
