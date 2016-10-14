/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Mathieu Lirzin <mthl@openmailbox.org>
 * Copyright (C) 2016  Inria
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

#ifndef __DRIVER_MPI_COMMON_H__
#define __DRIVER_MPI_COMMON_H__

#include <drivers/mp_common/mp_common.h>
#include <drivers/mpi/driver_mpi_source.h>

#ifdef STARPU_USE_MPI_MASTER_SLAVE

int _starpu_mpi_common_mp_init();
void _starpu_mpi_src_mp_deinit();
int _starpu_mpi_common_is_src_node();
int _starpu_mpi_common_is_mp_initialized();


#endif  /* STARPU_USE_MPI_MASTER_SLAVE */

#endif	/* __DRIVER_MPI_COMMON_H__ */
