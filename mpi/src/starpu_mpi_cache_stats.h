/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_CACHE_STATS_H__
#define __STARPU_MPI_CACHE_STATS_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

void _starpu_mpi_cache_stats_init();
void _starpu_mpi_cache_stats_shutdown();

void _starpu_mpi_cache_stats_update(unsigned dst, starpu_data_handle_t data_handle, int count);

#define _starpu_mpi_cache_stats_inc(dst, data_handle) _starpu_mpi_cache_stats_update(dst, data_handle, +1)
#define _starpu_mpi_cache_stats_dec(dst, data_handle) _starpu_mpi_cache_stats_update(dst, data_handle, -1)

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_CACHE_STATS_H__
