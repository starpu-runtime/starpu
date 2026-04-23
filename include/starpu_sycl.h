/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SYCL_H__
#define __STARPU_SYCL_H__

#ifdef STARPU_USE_SYCL

/**
   @ingroup API_SYCL_Extensions
   todo
*/
void starpu_sycl_set_device(int devid);

/**
   Report a SYCLBLAS error.
   See \ref SYCLSupport for more details.
*/
void starpu_syclblas_report_error(const char *func, const char *file, int line, int status);

/**
   Call starpu_syclblas_report_error(), passing the current function, file and line position.
*/
#define STARPU_SYCLBLAS_REPORT_ERROR(status) starpu_syclblas_report_error(__starpu_func__, __FILE__, __LINE__, status)

#endif /* STARPU_USE_SYCL */

#endif /* __STARPU_SYCL_H__ */
