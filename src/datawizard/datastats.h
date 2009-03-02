#ifndef __DATASTATS_H__
#define __DATASTATS_H__

#include <stdint.h>
#include <stdlib.h>


inline void msi_cache_hit(unsigned node);
inline void msi_cache_miss(unsigned node);

void display_msi_stats(void);


void display_comm_ammounts(void);

#ifdef DATA_STATS
inline void update_comm_ammount(uint32_t src_node, uint32_t dst_node, size_t size);
#endif

#endif
