#include <datawizard/datastats.h>
#include <common/util.h>
#include <stdio.h>

/* measure the cache hit ratio for each node */

#ifdef DATA_STATS
static unsigned hit_cnt[16];
static unsigned miss_cnt[16];
#endif

inline void msi_cache_hit(unsigned node __attribute__ ((unused)))
{
#ifdef DATA_STATS
	hit_cnt[node]++;
#endif
}

inline void msi_cache_miss(unsigned node __attribute__ ((unused)))
{
#ifdef DATA_STATS
	miss_cnt[node]++;
#endif
}

void display_msi_stats(void)
{
#ifdef DATA_STATS
	fprintf(stderr, "MSI cache stats :\n");
	unsigned node;
	for (node = 0; node < 4; node++) 
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

/* measure the amount of data transfers between each pair of nodes */
#ifdef DATA_STATS

static size_t comm_ammount[8][8];

void display_comm_ammounts(void)
{
	unsigned src, dst;

	if (get_env_number("CALIBRATE") != -1)
	for (dst = 0; dst < 8; dst++)
	for (src = 0; src < 8; src++)
	{
		if (comm_ammount[src][dst])
			fprintf(stderr, "Total comm from %d to %d \t%dMB\n", src, dst, ((unsigned)comm_ammount[src][dst])/(1024*1024));
	}
}

inline void update_comm_ammount(uint32_t src_node, uint32_t dst_node, size_t size)
{
	comm_ammount[src_node][dst_node] += size;
}

#else

inline void display_comm_ammounts(void)
{
}

#endif

