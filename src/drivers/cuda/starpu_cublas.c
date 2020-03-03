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
#include <core/workers.h>

#ifdef STARPU_USE_CUDA
#include <cublas.h>
#include <starpu_cublas_v2.h>

static int cublas_initialized[STARPU_NMAXWORKERS];
static cublasHandle_t cublas_handles[STARPU_NMAXWORKERS];
static cublasHandle_t main_handle;
static starpu_pthread_mutex_t mutex;

static unsigned get_idx(void)
{
	unsigned workerid = starpu_worker_get_id_check();
	unsigned th_per_dev = _starpu_get_machine_config()->topology.cuda_th_per_dev;
	unsigned th_per_stream = _starpu_get_machine_config()->topology.cuda_th_per_stream;

	if (th_per_dev)
		return starpu_worker_get_devid(workerid);
	else if (th_per_stream)
		return workerid;
	else
		/* same thread for all devices */
		return 0;
}

static void init_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (!(cublas_initialized[idx]++))
	{
		cublasStatus cublasst = cublasInit();
		if (STARPU_UNLIKELY(cublasst))
			STARPU_CUBLAS_REPORT_ERROR(cublasst);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	cublasCreate(&cublas_handles[starpu_worker_get_id_check()]);
	cublasSetStream(cublas_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
}

static void set_cublas_stream_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasSetKernelStream(starpu_cuda_get_local_stream());
}

static void shutdown_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (!--cublas_initialized[idx])
		cublasShutdown();
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	cublasDestroy(cublas_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_cublas_init(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(init_cublas_func, NULL, STARPU_CUDA);
	starpu_execute_on_each_worker(set_cublas_stream_func, NULL, STARPU_CUDA);

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

void starpu_cublas_set_stream(void)
{
#ifdef STARPU_USE_CUDA
	if (!_starpu_get_machine_config()->topology.cuda_th_per_dev ||
		(!_starpu_get_machine_config()->topology.cuda_th_per_stream &&
		 _starpu_get_machine_config()->topology.nworkerpercuda > 1))
		cublasSetKernelStream(starpu_cuda_get_local_stream());
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
