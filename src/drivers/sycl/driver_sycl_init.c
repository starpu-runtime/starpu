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
#include "driver_sycl.h"

static struct _starpu_driver_info driver_info =
{
        .name_upper = "SYCL",
        .name_var = "SYCL",
        .name_lower = "sycl",
        .memory_kind = STARPU_SYCL_RAM,
        .alpha = 13.33f,
#if defined(STARPU_USE_SYCL)
	.early_init = _starpu_sycl_early_init,
	.discover_devices = _starpu_sycl_discover_devices,
	.init_config = _starpu_init_sycl_config,
        .driver_ops = &_starpu_driver_sycl_ops,
        .run_worker = _starpu_sycl_worker,
#if defined(STARPU_HAVE_HWLOC)
        //.get_hwloc_obj = _starpu_sycl_get_hwloc_obj,
#endif
        .init_worker_binding = _starpu_sycl_init_worker_binding,
        .init_worker_memory = _starpu_sycl_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
        .name_upper = "SYCL",
        .worker_archtype = STARPU_SYCL_WORKER,
#if defined(STARPU_USE_SYCL)
        .ops = &_starpu_driver_sycl_node_ops,
#endif
};

void _starpu_sycl_preinit(void)
{
        _starpu_driver_info_register(STARPU_SYCL_WORKER, &driver_info);
        _starpu_memory_driver_info_register(STARPU_SYCL_RAM, &memory_driver_info);
}
