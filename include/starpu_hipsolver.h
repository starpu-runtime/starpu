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

#ifndef __STARPU_HIPSOLVER_H__
#define __STARPU_HIPSOLVER_H__

#ifdef STARPU_HAVE_LIBHIPSOLVER
#include <hipsolver/hipsolver.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @ingroup API_HIP_Extensions
   @{
*/

/**
   Initialize HIPSOLVER on every HIP device
   controlled by StarPU. This call blocks until HIPSOLVER has been properly
   initialized on every device.

   See \ref HIP-specificOptimizations
*/
void starpu_hipsolver_init(void);

/**
   Synchronously deinitialize the HIPSOLVER library on
   every HIP device.

   See \ref HIP-specificOptimizations
*/
void starpu_hipsolver_shutdown(void);

#if defined(STARPU_USE_HIP) && defined(STARPU_HAVE_LIBHIPSOLVER)
/**
   Return the HIPSOLVER Dense handle to be used to queue HIPSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_hipsolver_init().
   See \ref HIP-specificOptimizations
*/
hipsolverDnHandle_t starpu_hipsolverDn_get_local_handle(void);

#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
/**
   Return the HIPSOLVER Sparse handle to be used to queue HIPSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_hipsolver_init().

   See \ref HIP-specificOptimizations
*/
hipsolverSpHandle_t starpu_hipsolverSp_get_local_handle(void);
#endif

#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
/**
   Return the HIPSOLVER Refactorization handle to be used to queue HIPSOLVER
   kernels. It is properly initialized and configured for multistream by
   starpu_hipsolver_init().

   See \ref HIP-specificOptimizations
*/
hipsolverRfHandle_t starpu_hipsolverRf_get_local_handle(void);
#endif

/**
   Report a HIPSOLVER error.
   See \ref HIPSupport for more details.
*/
void starpu_hipsolver_report_error(const char *func, const char *file, int line, hipError_t error);

/**
   Call starpu_hipsolver_report_error(), passing the current function, file and line position.
*/
#define STARPU_HIPSOLVER_REPORT_ERROR(error) starpu_hipsolver_report_error(__starpu_func__, __FILE__, __LINE__, error)

/**
   Report a HIPSOLVER status.
   See \ref HIPSupport for more details.
*/
void starpu_hipsolver_report_status(const char *func, const char *file, int line, hipsolverStatus_t status);

/**
   Call starpu_hipsolver_report_status(), passing the current function, file and line position.
*/
#define STARPU_HIPSOLVER_REPORT_STATUS(status) starpu_hipsolver_report_status(__starpu_func__, __FILE__, __LINE__, status)

#endif
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_HIPSOLVER_H__ */
