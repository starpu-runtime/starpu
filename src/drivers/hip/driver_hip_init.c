/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "driver_hip.h"

static struct _starpu_driver_info driver_info =
{
	.name_upper = "HIP",
	.name_var = "HIP",
	.name_lower = "hip",
	.memory_kind = STARPU_HIP_RAM,
	.alpha = 13.33f,
#if defined(STARPU_USE_HIP)
	.driver_ops = &_starpu_driver_hip_ops,
	.run_worker = _starpu_hip_worker,
#if defined(STARPU_HAVE_HWLOC)
	.get_hwloc_obj = _starpu_hip_get_hwloc_obj,
#endif
	.init_worker_binding = _starpu_hip_init_worker_binding,
	.init_worker_memory = _starpu_hip_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "HIP",
	.worker_archtype = STARPU_HIP_WORKER,
#if defined(STARPU_USE_HIP)
	.ops = &_starpu_driver_hip_node_ops,
#endif
};

void _starpu_hip_preinit(void)
{
	_starpu_driver_info_register(STARPU_HIP_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_HIP_RAM, &memory_driver_info);
}
