/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_cusolver.h>
#include <starpu_cuda.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_LIBCUSOLVER
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverRf.h>

static cusolverDnHandle_t cusolverDn_handles[STARPU_NMAXWORKERS];
static cusolverSpHandle_t cusolverSp_handles[STARPU_NMAXWORKERS];
static cusolverRfHandle_t cusolverRf_handles[STARPU_NMAXWORKERS];
static cusolverDnHandle_t mainDn_handle;
static cusolverSpHandle_t mainSp_handle;
static cusolverRfHandle_t mainRf_handle;

static void init_cusolver_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cusolverDnCreate(&cusolverDn_handles[starpu_worker_get_id_check()]);
	cusolverDnSetStream(cusolverDn_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
	cusolverSpCreate(&cusolverSp_handles[starpu_worker_get_id_check()]);
	cusolverSpSetStream(cusolverSp_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
	cusolverRfCreate(&cusolverRf_handles[starpu_worker_get_id_check()]);
	// Not available?
	//cusolverRfSetStream(cusolverRf_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
}

static void shutdown_cusolver_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cusolverDnDestroy(cusolverDn_handles[starpu_worker_get_id_check()]);
	cusolverSpDestroy(cusolverSp_handles[starpu_worker_get_id_check()]);
	cusolverRfDestroy(cusolverRf_handles[starpu_worker_get_id_check()]);
}
#endif

void starpu_cusolver_init(void)
{
#ifdef STARPU_HAVE_LIBCUSOLVER
	if (!starpu_cuda_worker_get_count())
		return;
	starpu_execute_on_each_worker(init_cusolver_func, NULL, STARPU_CUDA);

	if (cusolverDnCreate(&mainDn_handle) != CUSOLVER_STATUS_SUCCESS)
		mainDn_handle = NULL;
	if (cusolverSpCreate(&mainSp_handle) != CUSOLVER_STATUS_SUCCESS)
		mainSp_handle = NULL;
	if (cusolverRfCreate(&mainRf_handle) != CUSOLVER_STATUS_SUCCESS)
		mainRf_handle = NULL;
#endif
}

void starpu_cusolver_shutdown(void)
{
#ifdef STARPU_HAVE_LIBCUSOLVER
	if (!starpu_cuda_worker_get_count())
		return;
	starpu_execute_on_each_worker(shutdown_cusolver_func, NULL, STARPU_CUDA);

	if (mainDn_handle)
		cusolverDnDestroy(mainDn_handle);
	if (mainSp_handle)
		cusolverSpDestroy(mainSp_handle);
	if (mainRf_handle)
		cusolverRfDestroy(mainRf_handle);
#endif
}

#ifdef STARPU_HAVE_LIBCUSOLVER
cusolverDnHandle_t starpu_cusolverDn_get_local_handle(void)
{
	if (!starpu_cuda_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cusolverDn_handles[workerid];
	else
		return mainDn_handle;
}

cusolverSpHandle_t starpu_cusolverSp_get_local_handle(void)
{
	if (!starpu_cuda_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cusolverSp_handles[workerid];
	else
		return mainSp_handle;
}

cusolverRfHandle_t starpu_cusolverRf_get_local_handle(void)
{
	if (!starpu_cuda_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cusolverRf_handles[workerid];
	else
		return mainRf_handle;
}
#endif
