/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2012  Centre National de la Recherche Scientifique
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

#include <common/config.h>
#include <core/jobs.h>
#include <core/task.h>

#include <core/perfmodel/perfmodel.h>
#include <common/fxt.h>
#include <datawizard/datawizard.h>

#include <starpu.h>

void *_starpu_cpu_worker(void *);
int _starpu_run_cpu(struct starpu_driver *);
int _starpu_cpu_driver_init(struct starpu_driver *);
int _starpu_cpu_driver_run_once(struct starpu_driver *);
int _starpu_cpu_driver_deinit(struct starpu_driver *);

#endif //  __DRIVER_CPU_H__
