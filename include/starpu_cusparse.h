/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_CUSPARSE_H__
#define __STARPU_CUSPARSE_H__

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
#include <cusparse.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_CUDA_Extensions
   @{
*/

/**
   Initialize CUSPARSE on every CUDA device
   controlled by StarPU. This call blocks until CUSPARSE has been properly
   initialized on every device.
*/
void starpu_cusparse_init(void);

/**
   Synchronously deinitialize the CUSPARSE library on
   every CUDA device.
*/
void starpu_cusparse_shutdown(void);

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
/**
   Return the CUSPARSE handle to be used to queue CUSPARSE
   kernels. It is properly initialized and configured for multistream by
   starpu_cusparse_init().
*/
cusparseHandle_t starpu_cusparse_get_local_handle(void);
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_CUSPARSE_H__ */
