/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

#ifndef __DRIVERS_H__
#define __DRIVERS_H__

/** @file */

struct _starpu_driver_ops
{
	int (*init)(struct _starpu_worker *worker);
	int (*run)(struct _starpu_worker *worker);
	int (*run_once)(struct _starpu_worker *worker);
	int (*deinit)(struct _starpu_worker *worker);
};

#endif // __DRIVERS_H__
