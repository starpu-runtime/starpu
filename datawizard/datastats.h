#ifndef __DATASTATS_H__
#define __DATASTATS_H__

inline void msi_cache_hit(unsigned node);
inline void msi_cache_miss(unsigned node);

void display_msi_stats(void);

#endif
