/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2011-2014  Université de Bordeaux
 * Copyright (C) 2014 INRIA
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
extern "C" {
#endif

extern int _starpu_cache_enabled;
void _starpu_mpi_cache_init(MPI_Comm comm);
void *_starpu_mpi_already_received(int src, starpu_data_handle_t data, int mpi_rank);
void *_starpu_mpi_already_sent(starpu_data_handle_t data, int dest);
void _starpu_mpi_cache_flush_sent(MPI_Comm comm, starpu_data_handle_t data);
void _starpu_mpi_cache_flush_recv(starpu_data_handle_t data, int me);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_CACHE_H__
