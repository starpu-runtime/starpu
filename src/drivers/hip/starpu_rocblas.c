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
#include <common/config.h>
#include <core/workers.h>

#if defined(STARPU_USE_HIP) && defined(STARPU_USE_ROCBLAS)
#include <rocblas/rocblas.h>

static int rocblas_initialized[STARPU_NMAXWORKERS];
static rocblas_handle rocblas_handles[STARPU_NMAXWORKERS];
static rocblas_handle main_handle;
static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

static unsigned get_idx(void)
{
	unsigned workerid = starpu_worker_get_id_check();
	unsigned th_per_dev = _starpu_get_machine_config()->topology.hip_th_per_dev;
	unsigned th_per_stream = _starpu_get_machine_config()->topology.hip_th_per_stream;

	if (th_per_dev)
		return starpu_worker_get_devid(workerid);
	else if (th_per_stream)
		return workerid;
	else
		/* same thread for all devices */
		return 0;
}
#define STARPU_ROCBLAS_REPORT_ERROR(status) do { \
		_STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", "init_rocblas_func", __FILE__, __LINE__, status, rocblas_status_to_string(status)); \
		STARPU_ABORT(); \
	} while(0)
static void init_rocblas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	rocblas_status status = rocblas_create_handle(&rocblas_handles[starpu_worker_get_id_check()]);
	STARPU_ASSERT_MSG(status != rocblas_status_invalid_handle, "rocblas initialization failed\n");
	if (status != rocblas_status_success)
		STARPU_ROCBLAS_REPORT_ERROR(status);
	status=rocblas_set_stream(rocblas_handles[starpu_worker_get_id_check()], starpu_hip_get_local_stream());
	if (status != rocblas_status_success)
		STARPU_ROCBLAS_REPORT_ERROR(status);

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (!(rocblas_initialized[idx]++))
		rocblas_initialize();
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

static void shutdown_rocblas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	--rocblas_initialized[idx];
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	rocblas_destroy_handle(rocblas_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_rocblas_init(void)
{
#if defined(STARPU_USE_HIP) && defined(STARPU_USE_ROCBLAS)
	starpu_execute_on_each_worker_ex(init_rocblas_func, NULL, STARPU_HIP, "init_rocblas");

	if (rocblas_create_handle(&main_handle) != rocblas_status_success)
		main_handle = NULL;
#endif
}

void starpu_rocblas_shutdown(void)
{
#if defined(STARPU_USE_HIP) && defined(STARPU_USE_ROCBLAS)
	starpu_execute_on_each_worker_ex(shutdown_rocblas_func, NULL, STARPU_HIP, "shutdown_rocblas");

	if (main_handle)
		rocblas_destroy_handle(main_handle);
#endif
}

void starpu_rocblas_set_stream(void)
{
#if defined(STARPU_USE_HIP) && defined(STARPU_USE_ROCBLAS)
	unsigned workerid = starpu_worker_get_id_check();
	int devnum = starpu_worker_get_devnum(workerid);
	if (!_starpu_get_machine_config()->topology.hip_th_per_dev ||
		(!_starpu_get_machine_config()->topology.hip_th_per_stream &&
		 _starpu_get_machine_config()->topology.nworker[STARPU_HIP_WORKER][devnum] > 1))
		rocblas_set_stream(rocblas_handles[starpu_worker_get_id_check()], starpu_hip_get_local_stream());
#endif
}

#if defined(STARPU_USE_HIP) && defined(STARPU_USE_ROCBLAS)
rocblas_handle starpu_rocblas_get_local_handle(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return rocblas_handles[workerid];
	else
		return main_handle;
}
#endif
