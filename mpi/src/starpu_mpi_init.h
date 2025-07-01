/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_INIT_H__
#define __STARPU_MPI_INIT_H__

#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <common/uthash.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

struct comm_size_entry
{
	UT_hash_handle hh;
	MPI_Comm comm;
	int size;
	int rank;
	int *translated_ranks;
};

extern struct comm_size_entry *registered_comms;
void _starpu_mpi_do_initialize(struct _starpu_mpi_argc_argv *argc_argv);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_INIT_H__ */
