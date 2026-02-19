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

#include <common/config.h>
#include <starpu.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_LIBHIPSOLVER
#include <hipsolver/hipsolver.h>

static hipsolverDnHandle_t hipsolverDn_handles[STARPU_NMAXWORKERS];
static hipsolverDnHandle_t mainDn_handle;
#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
static hipsolverSpHandle_t hipsolverSp_handles[STARPU_NMAXWORKERS];
static hipsolverSpHandle_t mainSp_handle;
#endif
#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
static hipsolverRfHandle_t hipsolverRf_handles[STARPU_NMAXWORKERS];
static hipsolverRfHandle_t mainRf_handle;
#endif

static void init_hipsolver_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	if (hipsolverDnCreate(&hipsolverDn_handles[starpu_worker_get_id_check()]) != HIPSOLVER_STATUS_SUCCESS)
		hipsolverDn_handles[starpu_worker_get_id_check()] = NULL;
	else
		hipsolverDnSetStream(hipsolverDn_handles[starpu_worker_get_id_check()], starpu_hip_get_local_stream());
#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
	if (hipsolverSpCreate(&hipsolverSp_handles[starpu_worker_get_id_check()]) != HIPSOLVER_STATUS_SUCCESS)
		hipsolverSp_handles[starpu_worker_get_id_check()] = NULL;
	else
		hipsolverSpSetStream(hipsolverSp_handles[starpu_worker_get_id_check()], starpu_hip_get_local_stream());
#endif
#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
	if (hipsolverRfCreate(&hipsolverRf_handles[starpu_worker_get_id_check()]) != HIPSOLVER_STATUS_SUCCESS)
		hipsolverRf_handles[starpu_worker_get_id_check()] = NULL;
	// Not available?
	//else
	//	hipsolverRfSetStream(hipsolverRf_handles[starpu_worker_get_id_check()], starpu_hip_get_local_stream());
#endif
}

static void shutdown_hipsolver_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	if (hipsolverDn_handles[starpu_worker_get_id_check()])
	{
		hipsolverDnDestroy(hipsolverDn_handles[starpu_worker_get_id_check()]);
		hipsolverDn_handles[starpu_worker_get_id_check()] = NULL;
	}
#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
	if (hipsolverSp_handles[starpu_worker_get_id_check()])
	{
		hipsolverSpDestroy(hipsolverSp_handles[starpu_worker_get_id_check()]);
		hipsolverSp_handles[starpu_worker_get_id_check()] = NULL;
	}
#endif
#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
	if (hipsolverRf_handles[starpu_worker_get_id_check()])
	{
		hipsolverRfDestroy(hipsolverRf_handles[starpu_worker_get_id_check()]);
		hipsolverRf_handles[starpu_worker_get_id_check()] = NULL;
	}
#endif
}
#endif

void starpu_hipsolver_init(void)
{
#ifdef STARPU_HAVE_LIBHIPSOLVER
	if (!starpu_hip_worker_get_count())
		return;
	starpu_execute_on_each_worker_ex(init_hipsolver_func, NULL, STARPU_HIP, "init_hipsolver");

	if (hipsolverDnCreate(&mainDn_handle) != HIPSOLVER_STATUS_SUCCESS)
		mainDn_handle = NULL;
#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
	if (hipsolverSpCreate(&mainSp_handle) != HIPSOLVER_STATUS_SUCCESS)
		mainSp_handle = NULL;
#endif
#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
	if (hipsolverRfCreate(&mainRf_handle) != HIPSOLVER_STATUS_SUCCESS)
		mainRf_handle = NULL;
#endif
#endif
}

void starpu_hipsolver_shutdown(void)
{
#ifdef STARPU_HAVE_LIBHIPSOLVER
	if (!starpu_hip_worker_get_count())
		return;
	starpu_execute_on_each_worker_ex(shutdown_hipsolver_func, NULL, STARPU_HIP, "shutdown_hipsolver");

	if (mainDn_handle)
	{
		hipsolverDnDestroy(mainDn_handle);
		mainDn_handle = NULL;
	}
#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
	if (mainSp_handle)
	{
		hipsolverSpDestroy(mainSp_handle);
		mainSp_handle = NULL;
	}
#endif
#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
	if (mainRf_handle)
	{
		hipsolverRfDestroy(mainRf_handle);
		mainRf_handle = NULL;
	}
#endif
#endif
}

#ifdef STARPU_HAVE_LIBHIPSOLVER
hipsolverDnHandle_t starpu_hipsolverDn_get_local_handle(void)
{
	if (!starpu_hip_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0 && hipsolverDn_handles[workerid])
		return hipsolverDn_handles[workerid];
	else
		return mainDn_handle;
}

#ifdef STARPU_HAVE_LIBHIPSOLVER_SP
hipsolverSpHandle_t starpu_hipsolverSp_get_local_handle(void)
{
	if (!starpu_hip_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0 && hipsolverSp_handles[workerid])
		return hipsolverSp_handles[workerid];
	else
		return mainSp_handle;
}
#endif

#ifdef STARPU_HAVE_LIBHIPSOLVER_RF
hipsolverRfHandle_t starpu_hipsolverRf_get_local_handle(void)
{
	if (!starpu_hip_worker_get_count())
		return NULL;
	int workerid = starpu_worker_get_id();
	if (workerid >= 0 && hipsolverRf_handles[workerid])
		return hipsolverRf_handles[workerid];
	else
		return mainRf_handle;
}
#endif

void starpu_hipsolver_report_error(const char *func, const char *file, int line, hipError_t error)
{
	const char *errormsg = hipGetErrorString(error);
	_STARPU_ERROR("oops in %s (%s:%d)... %d: %s \n", func, file, line, error, errormsg);
}

void starpu_hipsolver_report_status(const char *func, const char *file, int line, hipsolverStatus_t status)
{
        char *errormsg;
        switch (status)
        {
	case HIPSOLVER_STATUS_SUCCESS:
		errormsg = "HIPSOLVER_STATUS_SUCCESS";
		break;
	case HIPSOLVER_STATUS_NOT_INITIALIZED:
		errormsg = "HIPSOLVER_STATUS_NOT_INITIALIZED";
		break;
	case HIPSOLVER_STATUS_ALLOC_FAILED:
		errormsg = "HIPSOLVER_STATUS_ALLOC_FAILED";
		break;
	case HIPSOLVER_STATUS_INVALID_VALUE:
		errormsg = "HIPSOLVER_STATUS_INVALID_VALUE";
		break;
	case HIPSOLVER_STATUS_MAPPING_ERROR:
		errormsg = "HIPSOLVER_STATUS_MAPPING_ERROR";
		break;
	case HIPSOLVER_STATUS_EXECUTION_FAILED:
		errormsg = "HIPSOLVER_STATUS_EXECUTION_FAILED";
		break;
	case HIPSOLVER_STATUS_INTERNAL_ERROR:
		errormsg = "HIPSOLVER_STATUS_INTERNAL_ERROR";
		break;
	case HIPSOLVER_STATUS_NOT_SUPPORTED:
		errormsg = "HIPSOLVER_STATUS_NOT_SUPPORTED";
		break;
	case HIPSOLVER_STATUS_ARCH_MISMATCH:
		errormsg = "HIPSOLVER_STATUS_ARCH_MISMATCH";
		break;
	case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
		errormsg = "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
		break;
	case HIPSOLVER_STATUS_INVALID_ENUM:
		errormsg = "HIPSOLVER_STATUS_INVALID_ENUM";
		break;
	case HIPSOLVER_STATUS_UNKNOWN:
		errormsg = "HIPSOLVER_STATUS_UNKNOWN";
		break;
	default:
		errormsg = "unknown error";
		break;
        }
        _STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
        STARPU_ABORT();
}
#endif
