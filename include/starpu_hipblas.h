/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_HIPBLAS_H__
#define __STARPU_HIPBLAS_H__

#if defined STARPU_USE_HIP && !defined STARPU_DONT_INCLUDE_HIP_HEADERS

#include <hipblas/hipblas.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_HIP_Extensions
   @{
 */

/**
   Initialize HIPBLAS on every HIPdevice. The
   HIPBLAS library must be initialized prior to any HIPBLAS call. Calling
   starpu_hipblas_init() will initialize HIPBLAS on every HIP device
   controlled by StarPU. This call blocks until HIPBLAS has been properly
   initialized on every device.
*/
void starpu_hipblas_init(void);

/**
   Return the HIPBLAS handle to be used to queue HIPBLAS kernels. It
   is properly initialized and configured for multistream by
   starpu_cublas_init().
*/
hipblasHandle_t starpu_hipblas_get_local_handle(void);

/**
   Synchronously deinitialize the HIPBLAS library on
   every HIP device.
*/
void starpu_hipblas_shutdown(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

#endif /* __STARPU_HIPBLAS_H__ */
