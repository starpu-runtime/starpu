/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_MS_H__
#define __STARPU_MPI_MS_H__

#include <starpu_config.h>

#ifdef STARPU_USE_MPI_MASTER_SLAVE

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Master_Slave Master Slave Extension
   @{
*/

typedef void *starpu_mpi_ms_func_symbol_t;

int starpu_mpi_ms_register_kernel(starpu_mpi_ms_func_symbol_t *symbol, const char *func_name);

starpu_mpi_ms_kernel_t starpu_mpi_ms_get_kernel(starpu_mpi_ms_func_symbol_t symbol);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#endif /* __STARPU_MPI_MS_H__ */
