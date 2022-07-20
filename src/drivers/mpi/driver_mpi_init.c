/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/workers.h>
#include <drivers/mpi/driver_mpi_source.h>

static struct _starpu_driver_info driver_info =
{
	.name_upper = "MPI_MS",
	.name_var = "MPI_MS",
	.name_lower = "mpi_ms",
	.memory_kind = STARPU_MPI_MS_RAM,
	.alpha = 1.0f,
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	.run_worker = _starpu_mpi_src_worker,
	.init_worker_binding = _starpu_mpi_init_worker_binding,
	.init_worker_memory = _starpu_mpi_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "MPI_MS",
	.worker_archtype = STARPU_MPI_MS_WORKER,
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	.ops = &_starpu_driver_mpi_ms_node_ops,
#endif
};

void _starpu_mpi_ms_preinit(void)
{
	_starpu_driver_info_register(STARPU_MPI_MS_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_MPI_MS_RAM, &memory_driver_info);
}
