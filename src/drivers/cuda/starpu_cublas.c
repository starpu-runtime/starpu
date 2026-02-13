/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if defined(STARPU_USE_CUDA) && defined(STARPU_USE_CUBLAS)
#include <drivers/cuda/driver_cuda.h>

//#ifdef CUBLAS_V2_H_
//#error oops
//#endif

static int cublas_initialized[STARPU_NMAXWORKERS];
static starpu_pthread_mutex_t mutex[STARPU_MAXCUDADEVS];

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
	unsigned devid = starpu_worker_get_devid(starpu_worker_get_id_check());
	STARPU_PTHREAD_MUTEX_LOCK(&mutex[devid]);
	if (!(cublas_initialized[idx]++))
	{
		cublasStatus_t cublasst = cublasInit();
		if (STARPU_UNLIKELY(cublasst))
			STARPU_CUBLAS_REPORT_ERROR(cublasst);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex[devid]);

	_starpu_init_cublas_v2_func();
}

static void set_cublas_stream_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasSetKernelStream(starpu_cuda_get_local_stream());
}

static void shutdown_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	unsigned devid = starpu_worker_get_devid(starpu_worker_get_id_check());
	STARPU_PTHREAD_MUTEX_LOCK(&mutex[devid]);
	if (!--cublas_initialized[idx])
		cublasShutdown();
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex[devid]);

	_starpu_shutdown_cublas_v2_func();
}

void starpu_cublas_init(void)
{
	if (!starpu_cuda_worker_get_count())
		return;
	unsigned i;
	for (i = 0; i < STARPU_MAXCUDADEVS; i++)
		STARPU_PTHREAD_MUTEX_INIT0(&mutex[i], NULL);

	starpu_execute_on_each_worker_ex(init_cublas_func, NULL, STARPU_CUDA, "init_cublas");
	starpu_execute_on_each_worker_ex(set_cublas_stream_func, NULL, STARPU_CUDA, "set_cublas_stream_func");

	_starpu_cublas_v2_init();
}

void starpu_cublas_shutdown(void)
{
	if (!starpu_cuda_worker_get_count())
		return;
	starpu_execute_on_each_worker_ex(shutdown_cublas_func, NULL, STARPU_CUDA, "shutdown_cublas");

	_starpu_cublas_v2_shutdown();
}

void starpu_cublas_set_stream(void)
{
	if (!starpu_cuda_worker_get_count())
		return;
	unsigned workerid = starpu_worker_get_id_check();
	int devnum = starpu_worker_get_devnum(workerid);
	if (!_starpu_get_machine_config()->topology.cuda_th_per_dev ||
		(!_starpu_get_machine_config()->topology.cuda_th_per_stream &&
		 _starpu_get_machine_config()->topology.nworker[STARPU_CUDA_WORKER][devnum] > 1))
		cublasSetKernelStream(starpu_cuda_get_local_stream());
}
#else
void starpu_cublas_init(void)
{
}

void starpu_cublas_shutdown(void)
{
}
#endif

