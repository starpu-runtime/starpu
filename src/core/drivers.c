/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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

#include <stdlib.h>
#include <stdio.h>
#include <common/config.h>
#include <core/debug.h>

int starpu_driver_init(struct starpu_driver *d)
{
	STARPU_ASSERT(d);
	struct _starpu_worker *worker = _starpu_get_worker_from_driver(d);

	if (worker->driver_ops == NULL)
		return -EINVAL;
	else
		return worker->driver_ops->init(worker);
}

int starpu_driver_run(struct starpu_driver *d)
{
	if (!d)
	{
		_STARPU_DEBUG("Invalid argument\n");
		return -EINVAL;
	}

	struct _starpu_worker *worker = _starpu_get_worker_from_driver(d);
	if (worker->driver_ops == NULL)
		return -EINVAL;
	else
		return worker->driver_ops->run(worker);
}

int starpu_driver_run_once(struct starpu_driver *d)
{
	STARPU_ASSERT(d);
	struct _starpu_worker *worker = _starpu_get_worker_from_driver(d);

	if (worker->driver_ops == NULL)
		return -EINVAL;
	else
		return worker->driver_ops->run_once(worker);
}

int starpu_driver_deinit(struct starpu_driver *d)
{
	STARPU_ASSERT(d);
	struct _starpu_worker *worker = _starpu_get_worker_from_driver(d);

	if (worker->driver_ops == NULL)
		return -EINVAL;
	else
		return worker->driver_ops->deinit(worker);
}

