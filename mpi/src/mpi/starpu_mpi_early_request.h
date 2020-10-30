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

#ifndef __STARPU_MPI_EARLY_REQUEST_H__
#define __STARPU_MPI_EARLY_REQUEST_H__

#include <starpu.h>
#include <stdlib.h>
#include <mpi.h>
#include <common/config.h>
#include <common/list.h>

/** @file */

#ifdef STARPU_USE_MPI_MPI

#ifdef __cplusplus
extern "C"
{
#endif

struct _starpu_mpi_early_request_tag_hashlist
{
	struct _starpu_mpi_req_list list;
	UT_hash_handle hh;
	starpu_mpi_tag_t data_tag;
};

void _starpu_mpi_early_request_init(void);
void _starpu_mpi_early_request_shutdown(void);
int _starpu_mpi_early_request_count(void);
void _starpu_mpi_early_request_check_termination(void);

void _starpu_mpi_early_request_enqueue(struct _starpu_mpi_req *req);
struct _starpu_mpi_req* _starpu_mpi_early_request_dequeue(starpu_mpi_tag_t data_tag, int source, MPI_Comm comm);

struct _starpu_mpi_early_request_tag_hashlist *_starpu_mpi_early_request_extract(starpu_mpi_tag_t data_tag, int source, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_MPI_MPI */
#endif /* __STARPU_MPI_EARLY_REQUEST_H__ */
