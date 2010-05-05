/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>

#ifdef STARPU_USE_CUDA
static void init_cublas_func(void *args __attribute__((unused)))
{
	cublasStatus cublasst = cublasInit();
	if (STARPU_UNLIKELY(cublasst))
		STARPU_CUBLAS_REPORT_ERROR(cublasst);
}

static void shutdown_cublas_func(void *args __attribute__((unused)))
{
	cublasShutdown();
}
#endif

void starpu_helper_cublas_init(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(init_cublas_func, NULL, STARPU_CUDA);
#endif
}

void starpu_helper_cublas_shutdown(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(shutdown_cublas_func, NULL, STARPU_CUDA);
#endif
}
