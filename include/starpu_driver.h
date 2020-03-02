/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/**
   @defgroup API_Running_Drivers Running Drivers
   @{
*/

/**
   structure for a driver
*/
struct starpu_driver
{
	/**
	    Type of the driver. Only ::STARPU_CPU_WORKER, ::STARPU_CUDA_WORKER
	    and ::STARPU_OPENCL_WORKER are currently supported.
	*/
	enum starpu_worker_archtype type;
	/**
	   Identifier of the driver.
	*/
	union
	{
		unsigned cpu_id;
		unsigned cuda_id;
#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
		cl_device_id opencl_id;
#elif defined(STARPU_SIMGRID)
		unsigned opencl_id;
#endif
	} id;
};

/**
   Initialize the given driver, run it until it receives a request to
   terminate, deinitialize it and return 0 on success. Return
   <c>-EINVAL</c> if starpu_driver::type is not a valid StarPU device type
   (::STARPU_CPU_WORKER, ::STARPU_CUDA_WORKER or ::STARPU_OPENCL_WORKER).

   This is the same as using the following functions: calling
   starpu_driver_init(), then calling starpu_driver_run_once() in a loop,
   and finally starpu_driver_deinit().
*/
int starpu_driver_run(struct starpu_driver *d);

/**
   Notify all running drivers that they should terminate.
*/
void starpu_drivers_request_termination(void);

/**
   Initialize the given driver. Return 0 on success, <c>-EINVAL</c>
   if starpu_driver::type is not a valid ::starpu_worker_archtype.
*/
int starpu_driver_init(struct starpu_driver *d);

/**
   Run the driver once, then return 0 on success, <c>-EINVAL</c> if
   starpu_driver::type is not a valid ::starpu_worker_archtype.
*/
int starpu_driver_run_once(struct starpu_driver *d);

/**
   Deinitialize the given driver. Return 0 on success, <c>-EINVAL</c> if
   starpu_driver::type is not a valid ::starpu_worker_archtype.
*/
int starpu_driver_deinit(struct starpu_driver *d);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DRIVER_H__ */
