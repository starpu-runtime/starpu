/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_ULFM_COMM_H__
#define __STARPU_MPI_ULFM_COMM_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <mpi.h>
#include <common/config.h>
#include <starpu_mpi_private.h>

#if defined(STARPU_USE_MPI_FT)

void _starpu_mpi_ulfm_comm_init();
void _starpu_mpi_ulfm_comm_shutdown();

void _starpu_mpi_ulfm_comm_register(MPI_Comm key_comm);
void _starpu_mpi_ulfm_comm_update(MPI_Comm key_comm,  starpu_mpi_comm comm);
void _starpu_mpi_ulfm_comm_delete(MPI_Comm comm);

MPI_Comm _starpu_mpi_ulfm_get_key_comm(starpu_mpi_comm comm);
MPI_Comm _starpu_mpi_ulfm_get_mpi_comm_from_key(MPI_Comm key_comm);
void _starpu_mpi_ulfm_comm_test_update(MPI_Comm key_comm, MPI_Comm comm);

#else

static inline void _starpu_mpi_ulfm_comm_init() {}
static inline void _starpu_mpi_ulfm_comm_shutdown() {}

static inline void _starpu_mpi_ulfm_comm_register(MPI_Comm key_comm STARPU_ATTRIBUTE_UNUSED) {}
static inline void _starpu_mpi_ulfm_comm_update(MPI_Comm key_comm STARPU_ATTRIBUTE_UNUSED, starpu_mpi_comm comm STARPU_ATTRIBUTE_UNUSED) {}
static inline void _starpu_mpi_ulfm_comm_delete(MPI_Comm comm STARPU_ATTRIBUTE_UNUSED) {}

static inline MPI_Comm _starpu_mpi_ulfm_get_key_comm(starpu_mpi_comm comm) {return comm;}
static inline MPI_Comm _starpu_mpi_ulfm_get_mpi_comm_from_key(MPI_Comm key_comm) {return key_comm;}
static inline void _starpu_mpi_ulfm_comm_test_update(MPI_Comm key_comm STARPU_ATTRIBUTE_UNUSED, MPI_Comm comm STARPU_ATTRIBUTE_UNUSED) { return ;}

#endif // STARPU_USE_MPI_FT

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_MPI_ULFM_COMM_H__ */
