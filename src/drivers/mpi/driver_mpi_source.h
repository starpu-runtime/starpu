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

#ifndef __DRIVER_MPI_SOURCE_H__
#define __DRIVER_MPI_SOURCE_H__

#include <drivers/mp_common/mp_common.h>
#include <starpu_mpi_ms.h>

#ifdef STARPU_USE_MPI_MASTER_SLAVE

/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
extern struct _starpu_mp_node *mpi_ms_nodes[STARPU_MAXMICDEVS];

unsigned _starpu_mpi_src_get_device_count();
void *_starpu_mpi_src_worker(void *arg);
void _starpu_mpi_exit_useless_node(int devid);

void _starpu_mpi_source_init(struct _starpu_mp_node *node);
void _starpu_mpi_source_deinit(struct _starpu_mp_node *node);

void(* _starpu_mpi_ms_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void);

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#endif	/* __DRIVER_MPI_SOURCE_H__ */
