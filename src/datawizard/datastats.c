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

#include <stdio.h>
#include <datawizard/datastats.h>
#include <common/config.h>
#include <starpu.h>

/* measure the cache hit ratio for each node */

#ifdef STARPU_DATA_STATS
static unsigned hit_cnt[STARPU_MAXNODES];
static unsigned miss_cnt[STARPU_MAXNODES];
#endif

inline void starpu_msi_cache_hit(unsigned node __attribute__ ((unused)))
{
#ifdef STARPU_DATA_STATS
	hit_cnt[node]++;
#endif
}

inline void starpu_msi_cache_miss(unsigned node __attribute__ ((unused)))
{
#ifdef STARPU_DATA_STATS
	miss_cnt[node]++;
#endif
}

void starpu_display_msi_stats(void)
{
#ifdef STARPU_DATA_STATS
	unsigned node;
	unsigned total_hit_cnt = 0;
	unsigned total_miss_cnt = 0;

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
#endif
}

/* measure the efficiency of our allocation cache */

#ifdef STARPU_DATA_STATS
static unsigned alloc_cnt[STARPU_MAXNODES];
static unsigned alloc_cache_hit_cnt[STARPU_MAXNODES];
#endif

inline void starpu_allocation_cache_hit(unsigned node __attribute__ ((unused)))
{
#ifdef STARPU_DATA_STATS
	alloc_cache_hit_cnt[node]++;
#endif
}

inline void starpu_data_allocation_inc_stats(unsigned node __attribute__ ((unused)))
{
#ifdef STARPU_DATA_STATS
	alloc_cnt[node]++;
#endif
}

void starpu_display_alloc_cache_stats(void)
{
#ifdef STARPU_DATA_STATS
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
	}
#endif
}

/* measure the amount of data transfers between each pair of nodes */
#ifdef STARPU_DATA_STATS

static size_t comm_ammount[STARPU_MAXNODES][STARPU_MAXNODES];

void starpu_display_comm_amounts(void)
{
	unsigned src, dst;

	unsigned long sum = 0;

	for (dst = 0; dst < STARPU_MAXNODES; dst++)
	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		sum += (unsigned long)comm_ammount[src][dst];
	}

	fprintf(stderr, "\nData transfers stats:\nTOTAL transfers %ld MB\n", sum/(1024*1024));

	for (dst = 0; dst < STARPU_MAXNODES; dst++)
	for (src = dst + 1; src < STARPU_MAXNODES; src++)
	{
		if (comm_ammount[src][dst])
			fprintf(stderr, "\t%d <-> %d\t%ld MB\n\t\t%d -> %d\t%ld MB\n\t\t%d -> %d\t%ld MB\n",
				src, dst, ((unsigned long)comm_ammount[src][dst] + (unsigned long)comm_ammount[dst][src])/(1024*1024),
				src, dst, ((unsigned long)comm_ammount[src][dst])/(1024*1024),
				dst, src, ((unsigned long)comm_ammount[dst][src])/(1024*1024));
	}
}

inline void starpu_update_comm_amount(uint32_t src_node, uint32_t dst_node, size_t size)
{
	comm_ammount[src_node][dst_node] += size;
}

#else

inline void starpu_display_comm_amounts(void)
{
}

#endif

