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
#include <drivers/max/driver_max_fpga.h>

static struct _starpu_driver_info driver_info =
{
	.name_upper = "FPGA",
	.name_var = "FPGA",
	.name_lower = "fpga",
	.memory_kind = STARPU_MAX_FPGA_RAM,
	.alpha = 0.5,
	.wait_for_worker_initialization = 1,
#ifdef STARPU_USE_MAX_FPGA
	.driver_ops = &_starpu_driver_max_fpga_ops,
	.run_worker = _starpu_max_fpga_worker,
	.init_worker_binding = _starpu_max_fpga_init_worker_binding,
	.init_worker_memory = _starpu_max_fpga_init_worker_memory,
#endif
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "FPGA",
	.worker_archtype = STARPU_MAX_FPGA_WORKER,
#ifdef STARPU_USE_MAX_FPGA
	.ops = &_starpu_driver_max_fpga_node_ops,
#endif
};

void _starpu_max_fpga_preinit(void)
{
	_starpu_driver_info_register(STARPU_MAX_FPGA_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_MAX_FPGA_RAM, &memory_driver_info);
}
