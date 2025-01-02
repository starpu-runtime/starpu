/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_CPU_H__
#define __DRIVER_CPU_H__

/** @file */

#include <core/workers.h>
#include <common/config.h>

#pragma GCC visibility push(hidden)

void _starpu_cpu_preinit(void);

extern struct _starpu_driver_ops _starpu_driver_cpu_ops;

/* Reserve one CPU core as busy for starting a driver thread */
void _starpu_cpu_busy_cpu(unsigned num);

void _starpu_init_cpu_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config);

#pragma GCC visibility pop

#endif //  __DRIVER_CPU_H__
