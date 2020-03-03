/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_cuda.h>
#include <common/config.h>

#ifdef STARPU_USE_CUDA
#include <cublas.h>
#include <starpu_cublas_v2.h>

static cublasHandle_t cublas_handles[STARPU_NMAXWORKERS];
static cublasHandle_t main_handle;

static void init_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasStatus cublasst = cublasInit();
	if (STARPU_UNLIKELY(cublasst))
		STARPU_CUBLAS_REPORT_ERROR(cublasst);

	cublasCreate(&cublas_handles[starpu_worker_get_id_check()]);
	cublasSetStream(cublas_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());

	cublasSetKernelStream(starpu_cuda_get_local_stream());
}

static void shutdown_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasShutdown();

	cublasDestroy(cublas_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_cublas_init(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(init_cublas_func, NULL, STARPU_CUDA);

	if (cublasCreate(&main_handle) != CUBLAS_STATUS_SUCCESS)
		main_handle = NULL;
#endif
}

void starpu_cublas_shutdown(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(shutdown_cublas_func, NULL, STARPU_CUDA);

	if (main_handle)
		cublasDestroy(main_handle);
#endif
}

#ifdef STARPU_USE_CUDA
cublasHandle_t starpu_cublas_get_local_handle(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cublas_handles[workerid];
	else
		return main_handle;
}
#endif
