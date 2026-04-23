/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SYCLBLAS_H__
#define __STARPU_SYCLBLAS_H__

#ifdef __cplusplus
//#include <starpu_sycl.hpp>
//SYCL_ENABLE_WARNINGS
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
//SYCL_DISABLE_WARNINGS
//#include <dpct/blas_utils.hpp>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_SYCL_Extensions
   @{
*/

/**
   Initialize SYCLBLAS on every SYCL device. The
   SYCLBLAS library does not have to be be initialized prior to any SYCLBLAS call.
   Calling starpu_syclblas_init() will initialize SYCLBLAS on every SYCL device
   controlled by StarPU. This call blocks until SYCLBLAS has been properly
   initialized on every device.
*/
void starpu_syclblas_init(void);

/**
   Synchronously deinitialize the SYCLBLAS library on
   every SYCL device.
*/
void starpu_syclblas_shutdown(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SYCLBLAS_H__ */
