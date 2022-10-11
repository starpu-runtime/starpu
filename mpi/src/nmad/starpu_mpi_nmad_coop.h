/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_NMAD_COOP_H__
#define __STARPU_MPI_NMAD_COOP_H__

#include <common/config.h>

#ifdef STARPU_USE_MPI_NMAD

#ifdef __cplusplus
extern "C"
{
#endif

#include <starpu_config.h>
#include <nm_sendrecv_interface.h>

void _starpu_mpi_nmad_coop_init(void);
void _starpu_mpi_nmad_coop_shutdown(void);
void _starpu_mpi_nmad_end_coop_callback(void* arg);

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI_NMAD
#endif // __STARPU_MPI_NMAD_COOP_H__
