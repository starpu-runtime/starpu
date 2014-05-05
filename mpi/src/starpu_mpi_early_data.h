/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_MPI_EARLY_DATA_H__
#define __STARPU_MPI_EARLY_DATA_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>
#include <common/config.h>
#include <common/list.h>

#ifdef __cplusplus
extern "C" {
#endif

LIST_TYPE(_starpu_mpi_copy_handle,
	  starpu_data_handle_t handle;
	  struct _starpu_mpi_envelope *env;
	  struct _starpu_mpi_req *req;
	  void *buffer;
	  int mpi_tag;
	  int source;
	  int req_ready;
	  starpu_pthread_mutex_t req_mutex;
	  starpu_pthread_cond_t req_cond;
);

void _starpu_mpi_early_data_init(int world_size);
void _starpu_mpi_early_data_check_termination();
void _starpu_mpi_early_data_free(int world_size);

struct _starpu_mpi_copy_handle *find_chandle(int mpi_tag, int source);
void add_chandle(struct _starpu_mpi_copy_handle *chandle);
void delete_chandle(struct _starpu_mpi_copy_handle *chandle);
struct _starpu_mpi_copy_handle *pop_chandle(int mpi_tag, int source, int delete);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_EARLY_DATA_H__ */
