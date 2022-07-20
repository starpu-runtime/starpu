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
#include <drivers/opencl/driver_opencl.h>

static struct _starpu_driver_info driver_info =
{
	.name_upper = "OpenCL",
	.name_var = "OPENCL",
	.name_lower = "opencl",
	.memory_kind = STARPU_OPENCL_RAM,
	.alpha = 12.22f,
	.wait_for_worker_initialization = 1,
#if defined(STARPU_USE_OPENCL)
	.driver_ops = &_starpu_driver_opencl_ops,
#ifdef STARPU_HAVE_HWLOC
	.get_hwloc_obj = _starpu_opencl_get_hwloc_obj,
#endif
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	.run_worker = _starpu_opencl_worker,
	.init_worker_binding = _starpu_opencl_init_worker_binding,
	.init_worker_memory = _starpu_opencl_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "OpenCL",
	.worker_archtype = STARPU_OPENCL_WORKER,
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	.ops = &_starpu_driver_opencl_node_ops,
#endif
};

void _starpu_opencl_preinit(void)
{
	_starpu_driver_info_register(STARPU_OPENCL_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_OPENCL_RAM, &memory_driver_info);
}
