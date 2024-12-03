/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>

#include <starpu.h>
#include <starpu_cuda.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_LIBCUBLASLT
#include <starpu_cublasLt.h>

static cublasLtHandle_t cublasLt_handles[STARPU_NMAXWORKERS];
static cublasLtHandle_t main_handle;

static void init_cublasLt_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasLtCreate(&cublasLt_handles[starpu_worker_get_id_check()]);
	// No need for setting streams, because the cublasLt handles are not bundled with streams
}

static void shutdown_cublasLt_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasLtDestroy(cublasLt_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_cublasLt_init(void)
{
#ifdef STARPU_HAVE_LIBCUBLASLT
	if (!starpu_cuda_worker_get_count())
		return;
	starpu_execute_on_each_worker_ex(init_cublasLt_func, NULL, STARPU_CUDA, "init_cublasLt");

	if (cublasLtCreate(&main_handle) != CUBLAS_STATUS_SUCCESS)
		main_handle = NULL;
#endif
}

void starpu_cublasLt_shutdown(void)
{
#ifdef STARPU_HAVE_LIBCUBLASLT
	if (!starpu_cuda_worker_get_count())
		return;
	starpu_execute_on_each_worker_ex(shutdown_cublasLt_func, NULL, STARPU_CUDA, "shutdown_cublasLt");

	if (main_handle)
		cublasLtDestroy(main_handle);
#endif
}

#ifdef STARPU_HAVE_LIBCUBLASLT
cublasLtHandle_t starpu_cublasLt_get_local_handle(void)
{
	if (!starpu_cuda_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cublasLt_handles[workerid];
	else
		return main_handle;
}
#endif
