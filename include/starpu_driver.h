/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
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

#ifndef __STARPU_DRIVER_H__
#define __STARPU_DRIVER_H__

#include <starpu_config.h>
#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
#include <starpu_opencl.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_driver
{
	enum starpu_worker_archtype type;
	union
	{
		unsigned cpu_id;
		unsigned cuda_id;
#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
		cl_device_id opencl_id;
#elif defined(STARPU_SIMGRID)
		unsigned opencl_id;
#endif
		/*
		 * HOWTO: add a new kind of device to the starpu_driver structure.
		 * 1) Add a member to this union.
		 * 2) Edit _starpu_launch_drivers() to make sure the driver is
		 *    not always launched.
		 * 3) Edit starpu_driver_run() so that it can handle another
		 *    kind of architecture.
		 * 4) Write _starpu_run_foobar() in the corresponding driver.
		 * 5) Test the whole thing :)
		 */
	} id;
};

int starpu_driver_run(struct starpu_driver *d);
void starpu_drivers_request_termination(void);

int starpu_driver_init(struct starpu_driver *d);
int starpu_driver_run_once(struct starpu_driver *d);
int starpu_driver_deinit(struct starpu_driver *d);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DRIVER_H__ */
