#ifndef __DATASTATS_H__
#define __DATASTATS_H__

#include <stdint.h>
#include <stdlib.h>


inline void msi_cache_hit(unsigned node);
inline void msi_cache_miss(unsigned node);

void display_msi_stats(void);

inline void allocation_cache_hit(unsigned node __attribute__ ((unused)));
inline void data_allocation_inc_stats(unsigned node __attribute__ ((unused)));


void display_comm_ammounts(void);
void display_alloc_cache_stats(void);

#ifdef DATA_STATS
inline void update_comm_ammount(uint32_t src_node, uint32_t dst_node, size_t size);
#endif

#endif
