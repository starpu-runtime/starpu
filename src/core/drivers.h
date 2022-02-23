/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#pragma GCC visibility push(hidden)

/** @file */

struct _starpu_driver_ops
{
	int (*init)(struct _starpu_worker *worker);	/**< Initialize the thread for running the worker */
	int (*run)(struct _starpu_worker *worker);	/**< Actually run the worker */
	int (*run_once)(struct _starpu_worker *worker);	/**< Run just one loop of the worker */
	int (*deinit)(struct _starpu_worker *worker);	/**< Deinitialize the thread after running a worker */
	int (*set_devid)(struct starpu_driver *driver, struct _starpu_worker *worker);
							/**< Sets into \p driver the id for worker \p worker */
	int (*is_devid)(struct starpu_driver *driver, struct _starpu_worker *worker);
							/**< Tests whether \p driver has the id for worker \p worker */
};

#pragma GCC visibility pop

#endif // __DRIVERS_H__
