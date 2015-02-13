/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_MPI_SYNC_DATA_H__
#define __STARPU_MPI_SYNC_DATA_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>
#include <common/config.h>
#include <common/list.h>

#ifdef __cplusplus
extern "C" {
#endif

LIST_TYPE(_starpu_mpi_sync_data_handle,
	  struct _starpu_mpi_req *req;
	  int data_tag;
	  int source;
);

void _starpu_mpi_sync_data_init(int world_size);
void _starpu_mpi_sync_data_check_termination();
void _starpu_mpi_sync_data_free(int world_size);

struct _starpu_mpi_sync_data_handle *_starpu_mpi_sync_data_create(struct _starpu_mpi_req *req);
struct _starpu_mpi_sync_data_handle *_starpu_mpi_sync_data_find(int data_tag, int source);
void _starpu_mpi_sync_data_add(struct _starpu_mpi_sync_data_handle *sync_data_handle);
int _starpu_mpi_sync_data_count();

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_SYNC_DATA_H__ */
