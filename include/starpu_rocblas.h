/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_ROCBLAS_H__
#define __STARPU_ROCBLAS_H__

#ifdef STARPU_USE_HIP
#ifdef STARPU_USE_ROCBLAS
#include <rocblas/rocblas.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_HIP_Extensions
   @{
*/

/**
   Initialize ROCBLAS on every HIP device. The
   ROCBLAS library must be initialized prior to any ROCBLAS call. Calling
   starpu_rocblas_init() will initialize ROCBLAS on every HIP device
   controlled by StarPU. This call blocks until ROCBLAS has been properly
   initialized on every device.
*/
void starpu_rocblas_init(void);

#ifdef STARPU_USE_HIP
#ifdef STARPU_USE_ROCBLAS
/**
   Return the ROCBLAS handle to be used to queue ROCBLAS kernels. It
   is properly initialized and configured for multistream by
   starpu_rocblas_init().
*/
rocblas_handle starpu_rocblas_get_local_handle(void);
#endif
#endif

/**
   Synchronously deinitialize the ROCBLAS library on
   every HIP device.
*/
void starpu_rocblas_shutdown(void);

/**
   Rebind the ROCBLAS handle of the current worker to the current StarPU
   HIP stream. Must be called from a HIP worker.
*/
void starpu_rocblas_set_stream(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_ROCBLAS_H__ */
