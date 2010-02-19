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

#ifndef __DATASTATS_H__
#define __DATASTATS_H__

#include <stdint.h>
#include <stdlib.h>


inline void _starpu_msi_cache_hit(unsigned node);
inline void _starpu_msi_cache_miss(unsigned node);

void _starpu_display_msi_stats(void);

inline void _starpu_allocation_cache_hit(unsigned node __attribute__ ((unused)));
inline void _starpu_data_allocation_inc_stats(unsigned node __attribute__ ((unused)));


void _starpu_display_comm_amounts(void);
void _starpu_display_alloc_cache_stats(void);

#ifdef STARPU_DATA_STATS
inline void _starpu_update_comm_amount(uint32_t src_node, uint32_t dst_node, size_t size);
#endif

#endif
