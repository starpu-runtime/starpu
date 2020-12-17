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
#include <drivers/opencl/driver_opencl.h>

static struct starpu_driver_info driver_info =
{
	.name_upper = "OpenCL",
	.name_var = "OPENCL",
	.name_lower = "opencl",
	.memory_kind = STARPU_OPENCL_RAM,
	.alpha = 12.22f,
};

static struct starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "OpenCL",
	.worker_archtype = STARPU_OPENCL_WORKER,
};

void _starpu_opencl_preinit(void)
{
	starpu_driver_info_register(STARPU_OPENCL_WORKER, &driver_info);
	starpu_memory_driver_info_register(STARPU_OPENCL_RAM, &memory_driver_info);
}
