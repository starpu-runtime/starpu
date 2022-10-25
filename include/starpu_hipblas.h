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

#ifndef __STARPU_HIPBLAS_V2_H__
#define __STARPU_HIPBLAS_V2_H__

#if defined STARPU_USE_HIP && !defined STARPU_DONT_INCLUDE_HIP_HEADERS

#include <hipblas.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_HIP_Extensions
   @{
 */

/**
   Return the HIPBLAS handle to be used to queue HIPBLAS kernels. It
   is properly initialized and configured for multistream by
   starpu_cublas_init().
*/hipblasHandle_t starpu_hipblas_get_local_handle(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

#endif /* __STARPU_HIPBLAS_V2_H__ */
