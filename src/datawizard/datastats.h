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

#ifndef __DATASTATS_H__
#define __DATASTATS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <stdint.h>
#include <stdlib.h>

extern int _starpu_enable_stats;

void _starpu_datastats_init();

static inline int starpu_enable_stats(void)
{
	return _starpu_enable_stats;
}

void __starpu_msi_cache_hit(unsigned node);
void __starpu_msi_cache_miss(unsigned node);

#define _starpu_msi_cache_hit(node) do { \
	if (starpu_enable_stats()) \
		__starpu_msi_cache_hit(node); \
} while (0)

#define _starpu_msi_cache_miss(node) do { \
	if (starpu_enable_stats()) \
		__starpu_msi_cache_miss(node); \
} while (0)

void _starpu_display_msi_stats(FILE *stream);

void __starpu_allocation_cache_hit(unsigned node STARPU_ATTRIBUTE_UNUSED);
void __starpu_data_allocation_inc_stats(unsigned node STARPU_ATTRIBUTE_UNUSED);

#define _starpu_allocation_cache_hit(node) do { \
	if (starpu_enable_stats()) \
		__starpu_allocation_cache_hit(node); \
} while (0)

#define _starpu_data_allocation_inc_stats(node) do { \
	if (starpu_enable_stats()) \
		__starpu_data_allocation_inc_stats(node); \
} while (0)

void _starpu_display_alloc_cache_stats(FILE *stream);

#endif // __DATASTATS_H__
