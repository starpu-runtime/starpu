#include <datawizard/datastats.h>
#include <stdio.h>

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


