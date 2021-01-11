/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi_private.h>

#ifdef __cplusplus
extern "C"
{
#endif

LIST_TYPE(_starpu_mpi_early_data_handle,
	  starpu_data_handle_t handle;
	  struct _starpu_mpi_envelope *env;
	  struct _starpu_mpi_req *req;
	  void *buffer;
	  int req_ready;
	  struct _starpu_mpi_node_tag node_tag;
	  starpu_pthread_mutex_t req_mutex;
	  starpu_pthread_cond_t req_cond;
);

void _starpu_mpi_early_data_init(void);
void _starpu_mpi_early_data_check_termination(void);
void _starpu_mpi_early_data_free(void);

struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_create(struct _starpu_mpi_envelope *envelope, int source, MPI_Comm comm) STARPU_ATTRIBUTE_MALLOC;
struct _starpu_mpi_early_data_handle *_starpu_mpi_early_data_find(struct _starpu_mpi_node_tag *node_tag);
void _starpu_mpi_early_data_add(struct _starpu_mpi_early_data_handle *early_data_handle);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_EARLY_DATA_H__ */
