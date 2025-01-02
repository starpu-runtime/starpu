/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_CUBLASLT_H__
#define __STARPU_CUBLASLT_H__

#ifdef STARPU_USE_CUDA
#ifdef STARPU_HAVE_LIBCUBLASLT
#include <cublasLt.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_CUDA_Extensions
   @{
*/

/**
   Initialize CUBLASLT on every CUDA device
   controlled by StarPU. This call blocks until CUBLASLT has been properly
   initialized on every device. See \ref CUDA-specificOptimizations for more details.
*/
void starpu_cublasLt_init(void);

/**
   Synchronously deinitialize the CUBLASLT library on
   every CUDA device. See \ref CUDA-specificOptimizations for more details.
*/
void starpu_cublasLt_shutdown(void);

#ifdef STARPU_USE_CUDA
#ifdef STARPU_HAVE_LIBCUBLASLT
/**
   Return the CUBLASLT handle to be used to queue CUBLASLT
   kernels. It is properly initialized and configured for multistream by
   starpu_cublasLt_init(). See \ref CUDA-specificOptimizations for more details.
*/
cublasLtHandle_t starpu_cublasLt_get_local_handle(void);
#endif
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_CUBLASLT_H__ */
