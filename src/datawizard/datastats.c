/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2013  Université de Bordeaux 1
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
#include <datawizard/datastats.h>
#include <datawizard/coherency.h>
#include <common/config.h>

#ifdef STARPU_ENABLE_STATS
/* measure the cache hit ratio for each node */
static unsigned hit_cnt[STARPU_MAXNODES];
static unsigned miss_cnt[STARPU_MAXNODES];
#endif

void _starpu_msi_cache_hit(unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_ENABLE_STATS
	STARPU_HG_DISABLE_CHECKING(hit_cnt[node]);
	hit_cnt[node]++;
#endif
}

void _starpu_msi_cache_miss(unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_ENABLE_STATS
	STARPU_HG_DISABLE_CHECKING(miss_cnt[node]);
	miss_cnt[node]++;
#endif
}

void _starpu_display_msi_stats(void)
{
#ifdef STARPU_ENABLE_STATS
	unsigned node;
	unsigned total_hit_cnt = 0;
	unsigned total_miss_cnt = 0;

	fprintf(stderr, "\n#---------------------\n");
	fprintf(stderr, "MSI cache stats :\n");

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		total_hit_cnt += hit_cnt[node];
		total_miss_cnt += miss_cnt[node];
	}

	fprintf(stderr, "TOTAL MSI stats\thit %u (%2.2f \%%)\tmiss %u (%2.2f \%%)\n", total_hit_cnt, (100.0f*total_hit_cnt)/(total_hit_cnt+total_miss_cnt), total_miss_cnt, (100.0f*total_miss_cnt)/(total_hit_cnt+total_miss_cnt));

	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (hit_cnt[node]+miss_cnt[node])
		{
			fprintf(stderr, "memory node %d\n", node);
			fprintf(stderr, "\thit : %u (%2.2f \%%)\n", hit_cnt[node], (100.0f*hit_cnt[node])/(hit_cnt[node]+miss_cnt[node]));
			fprintf(stderr, "\tmiss : %u (%2.2f \%%)\n", miss_cnt[node], (100.0f*miss_cnt[node])/(hit_cnt[node]+miss_cnt[node]));
		}
	}
	fprintf(stderr, "#---------------------\n");
#endif
}

/* measure the efficiency of our allocation cache */

#ifdef STARPU_ENABLE_STATS
static unsigned alloc_cnt[STARPU_MAXNODES];
static unsigned alloc_cache_hit_cnt[STARPU_MAXNODES];
#endif

void _starpu_allocation_cache_hit(unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_ENABLE_STATS
	STARPU_HG_DISABLE_CHECKING(alloc_cache_hit_cnt[node]);
	alloc_cache_hit_cnt[node]++;
#endif
}

void _starpu_data_allocation_inc_stats(unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_ENABLE_STATS
	STARPU_HG_DISABLE_CHECKING(alloc_cnt[node]);
	alloc_cnt[node]++;
#endif
}

void _starpu_display_alloc_cache_stats(void)
{
#ifdef STARPU_ENABLE_STATS
	fprintf(stderr, "\n#---------------------\n");
	fprintf(stderr, "Allocation cache stats:\n");
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		if (alloc_cnt[node])
		{
			fprintf(stderr, "memory node %d\n", node);
			fprintf(stderr, "\ttotal alloc : %u\n", alloc_cnt[node]);
			fprintf(stderr, "\tcached alloc: %u (%2.2f \%%)\n",
				alloc_cache_hit_cnt[node], (100.0f*alloc_cache_hit_cnt[node])/(alloc_cnt[node]));
		}
		else
			fprintf(stderr, "No allocation on node %d\n", node);
	}
	fprintf(stderr, "#---------------------\n");
#endif
}

/* measure the amount of data transfers between each pair of nodes */
#ifdef STARPU_ENABLE_STATS
static size_t comm_amount[STARPU_MAXNODES][STARPU_MAXNODES];
#endif /* STARPU_ENABLE_STATS */

void _starpu_comm_amounts_inc(unsigned src  STARPU_ATTRIBUTE_UNUSED, unsigned dst  STARPU_ATTRIBUTE_UNUSED, size_t size  STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_ENABLE_STATS
	STARPU_HG_DISABLE_CHECKING(comm_amount[src][dst]);
	comm_amount[src][dst] += size;
#endif /* STARPU_ENABLE_STATS */
}

void _starpu_display_comm_amounts(void)
{
#ifdef STARPU_DEVEL
#  warning TODO. The information displayed here seems to be similar to the one displayed by starpu_profiling_bus_helper_display_summary()
#endif

#ifdef STARPU_ENABLE_STATS
	unsigned src, dst;
	size_t sum = 0;

	fprintf(stderr, "\n#---------------------\n");
	fprintf(stderr, "Data transfer stats:\n");

	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		for (src = 0; src < STARPU_MAXNODES; src++)
		{
			sum += comm_amount[src][dst];
			sum += comm_amount[dst][src];
		}

	fprintf(stderr, "TOTAL transfers %f MB\n", (float)sum/1024/1024);

	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		for (src = dst + 1; src < STARPU_MAXNODES; src++)
		{
			if (comm_amount[src][dst])
				fprintf(stderr, "\t%d <-> %d\t%f MB\n\t\t%d -> %d\t%f MB\n\t\t%d -> %d\t%f MB\n",
					src, dst, ((float)comm_amount[src][dst] + (float)comm_amount[dst][src])/(1024*1024),
					src, dst, ((float)comm_amount[src][dst])/(1024*1024),
					dst, src, ((float)comm_amount[dst][src])/(1024*1024));
		}
	fprintf(stderr, "#---------------------\n");
#endif
}

