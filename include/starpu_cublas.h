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

#ifndef __STARPU_CUBLAS_H__
#define __STARPU_CUBLAS_H__

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @ingroup API_CUDA_Extensions
   @{
 */

/**
   Initialize CUBLAS on every CUDA device. The
   CUBLAS library must be initialized prior to any CUBLAS call. Calling
   starpu_cublas_init() will initialize CUBLAS on every CUDA device
   controlled by StarPU. This call blocks until CUBLAS has been properly
   initialized on every device.
*/
void starpu_cublas_init(void);

/**
   Set the proper CUBLAS stream for CUBLAS v1. This must be called
   from the CUDA codelet before calling CUBLAS v1 kernels, so that
   they are queued on the proper CUDA stream. When using one thread
   per CUDA worker, this function does not do anything since the
   CUBLAS stream does not change, and is set once by
   starpu_cublas_init().
*/
void starpu_cublas_set_stream(void);

/**
   Synchronously deinitialize the CUBLAS library on
   every CUDA device.
*/
void starpu_cublas_shutdown(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_CUBLAS_H__ */
