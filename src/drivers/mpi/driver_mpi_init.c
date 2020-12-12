/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static struct starpu_driver_info driver_info = {
	.name_upper = "MPI_MS",
	.name_var = "MPI_MS",
	.name_lower = "mpi_ms",
	.memory_kind = STARPU_MPI_MS_RAM,
	.alpha = 1.0f,
};

static struct starpu_memory_driver_info memory_driver_info = {
	.name_upper = "MPI_MS",
	.worker_archtype = STARPU_MPI_MS_WORKER,
};

void _starpu_mpi_ms_preinit(void)
{
	starpu_driver_info_register(STARPU_MPI_MS_WORKER, &driver_info);
	starpu_memory_driver_info_register(STARPU_MPI_MS_RAM, &memory_driver_info);
}
