/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_CACHE_H__
#define __STARPU_MPI_CACHE_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern int _starpu_cache_enabled;
void _starpu_mpi_cache_init(MPI_Comm comm);
void _starpu_mpi_cache_free();

/*
 * If the data is already available in the cache, return a pointer to the data
 * If the data is NOT available in the cache, add it to the cache and return NULL
 */
void *_starpu_mpi_cache_received_data_set(starpu_data_handle_t data, int mpi_rank);
void *_starpu_mpi_cache_received_data_get(starpu_data_handle_t data, int mpi_rank);
void _starpu_mpi_cache_received_data_clear(starpu_data_handle_t data);

/*
 * If the data is already available in the cache, return a pointer to the data
 * If the data is NOT available in the cache, add it to the cache and return NULL
 */
void *_starpu_mpi_cache_sent_data_set(starpu_data_handle_t data, int dest);
void *_starpu_mpi_cache_sent_data_get(starpu_data_handle_t data, int dest);
void _starpu_mpi_cache_sent_data_clear(MPI_Comm comm, starpu_data_handle_t data);

void _starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_CACHE_H__
