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

#ifdef STARPU_USE_SYCL
#ifdef STARPU_USE_SYCLBLAS
#include <core/workers.h>

#include <mkl.h>
#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <starpu_syclblas.h>

static int syclblas_initialized[STARPU_NMAXWORKERS];
static dpct::queue_ptr syclblas_handles[STARPU_NMAXWORKERS];
static starpu_pthread_mutex_t mutex[STARPU_MAXSYCLDEVS];

#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef STARPU_USE_SYCL
#ifdef STARPU_USE_SYCLBLAS
static unsigned get_idx(void)
{
	unsigned workerid = starpu_worker_get_id_check();
	unsigned th_per_dev = _starpu_get_machine_config()->topology.sycl_th_per_dev;
	unsigned th_per_stream = _starpu_get_machine_config()->topology.sycl_th_per_stream;

	if (th_per_dev)
		return starpu_worker_get_devid(workerid);
	else if (th_per_stream)
		return workerid;
	else
		/* same thread for all devices */
		return 0;
}

static void dummy_init_syclblas(dpct::queue_ptr stream)
{
	int n = 10;
	float* x = sycl::malloc_device<float>(n, *stream);
	float* y = sycl::malloc_device<float>(n, *stream);
	oneapi::mkl::blas::row_major::axpy(*stream, n, 2.0f, x, 1, y, 1).wait();
	sycl::free(x, *stream);
	sycl::free(y, *stream);
}

static dpct::queue_ptr starpu_syclblas_get_local_handle(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return syclblas_handles[workerid];
	else
		return NULL;
}

static void init_syclblas_func(void *args STARPU_ATTRIBUTE_UNUSED) try
{
        unsigned idx = get_idx();
	unsigned devid = starpu_worker_get_devid(starpu_worker_get_id_check());
	STARPU_PTHREAD_MUTEX_LOCK(&mutex[devid]);
	syclblas_handles[starpu_worker_get_id_check()] = starpu_sycl_get_local_stream();
	if (!(syclblas_initialized[idx]++))
	{
		// dummy initialization
		dummy_init_syclblas(starpu_syclblas_get_local_handle());
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex[devid]);
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

static void shutdown_syclblas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	unsigned devid = starpu_worker_get_devid(starpu_worker_get_id_check());
	STARPU_PTHREAD_MUTEX_LOCK(&mutex[devid]);
	if (!--syclblas_initialized[idx])
	{
		// nothing to do
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex[devid]);
	syclblas_handles[devid] = NULL;
}
#endif
#endif

void starpu_syclblas_init(void)
{
#ifdef STARPU_USE_SYCL
#ifdef STARPU_USE_SYCLBLAS
	if (!starpu_sycl_worker_get_count())
		return;
	unsigned i;
	for (i = 0; i < STARPU_MAXSYCLDEVS; i++)
		STARPU_PTHREAD_MUTEX_INIT0(&mutex[i], NULL);

	starpu_execute_on_each_worker(init_syclblas_func, NULL, STARPU_SYCL);
#endif
#endif
}

void starpu_syclblas_shutdown(void)
{
#ifdef STARPU_USE_SYCL
#ifdef STARPU_USE_SYCLBLAS
	if (!starpu_sycl_worker_get_count())
		return;
	starpu_execute_on_each_worker(shutdown_syclblas_func, NULL, STARPU_SYCL);
#endif
#endif
}

#ifdef __cplusplus
}
#endif
