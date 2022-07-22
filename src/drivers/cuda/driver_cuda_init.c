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
#include <drivers/cuda/driver_cuda.h>

static struct _starpu_driver_info driver_info =
{
	.name_upper = "CUDA",
	.name_var = "CUDA",
	.name_lower = "cuda",
	.memory_kind = STARPU_CUDA_RAM,
	.alpha = 13.33f,
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	.driver_ops = &_starpu_driver_cuda_ops,
	.run_worker = _starpu_cuda_worker,
#if defined(STARPU_HAVE_HWLOC) && !defined(STARPU_USE_CUDA0)
	.get_hwloc_obj = _starpu_cuda_get_hwloc_obj,
#endif
	.init_worker_binding = _starpu_cuda_init_worker_binding,
	.init_worker_memory = _starpu_cuda_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "CUDA",
	.worker_archtype = STARPU_CUDA_WORKER,
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	.ops = &_starpu_driver_cuda_node_ops,
#endif
};

void _starpu_cuda_preinit(void)
{
	_starpu_driver_info_register(STARPU_CUDA_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_CUDA_RAM, &memory_driver_info);
}
