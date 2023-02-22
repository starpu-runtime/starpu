/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_CUSOLVER_H__
#define __STARPU_CUSOLVER_H__

#ifdef STARPU_USE_CUDA
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverRf.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_CUDA_Extensions
   @{
*/

/**
   Initialize CUSOLVER on every CUDA device
   controlled by StarPU. This call blocks until CUSOLVER has been properly
   initialized on every device.

   See \ref CUDA-specificOptimizations
*/
void starpu_cusolver_init(void);

/**
   Synchronously deinitialize the CUSOLVER library on
   every CUDA device.

   See \ref CUDA-specificOptimizations
*/
void starpu_cusolver_shutdown(void);

#ifdef STARPU_USE_CUDA
/**
   Return the CUSOLVER Dense handle to be used to queue CUSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_cusolver_init().

   See \ref CUDA-specificOptimizations
*/
cusolverDnHandle_t starpu_cusolverDn_get_local_handle(void);

/**
   Return the CUSOLVER Sparse handle to be used to queue CUSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_cusolver_init().

   See \ref CUDA-specificOptimizations
*/
cusolverSpHandle_t starpu_cusolverSp_get_local_handle(void);

/**
   Return the CUSOLVER Refactorization handle to be used to queue CUSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_cusolver_init().

   See \ref CUDA-specificOptimizations
*/
cusolverRfHandle_t starpu_cusolverRf_get_local_handle(void);
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_CUSOLVER_H__ */
