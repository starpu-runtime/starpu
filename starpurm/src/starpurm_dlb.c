/* StarPU --- Resource Management Layer.
 *
 * Copyright (C) 2017, 2018                              Inria
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

/* CPUSET routines */
#define _GNU_SOURCE
#include <sched.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <config.h>

#include <hwloc.h>
#ifdef HAVE_HWLOC_GLIBC_SCHED_H
#include <hwloc/glibc-sched.h>
#endif
#include <pthread.h>
#include <starpu.h>
#include <starpurm.h>
#include <starpurm_private.h>

#ifndef STARPURM_HAVE_DLB
#error "STARPU-RM DLB support not enabled"
#endif

#include <dlb_sp.h>
#include <dlb_errors.h>

/*
 * DLB interfacing
 */

static dlb_handler_t      dlb_handle;
static cpu_set_t          starpurm_process_mask;
static hwloc_cpuset_t     starpurm_process_cpuset;
static struct s_starpurm *_starpurm = NULL;
static pthread_mutex_t dlb_handle_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char * _dlb_error_str(int error_code)
{
	const char *s = NULL;
	switch (error_code)
	{
		case DLB_NOUPDT:
		s="DLB_NOUPDT";
		break;
		case DLB_NOTED:
		s="DLB_NOTED";
		break;
		case DLB_SUCCESS:
		s="DLB_SUCCESS";
		break;
		case DLB_ERR_UNKNOWN:
		s="DLB_ERR_UNKNOWN";
		break;
		case DLB_ERR_NOINIT:
		s="DLB_ERR_NOINIT";
		break;
		case DLB_ERR_INIT:
		s="DLB_ERR_INIT";
		break;
		case DLB_ERR_DISBLD:
		s="DLB_ERR_DISBLD";
		break;
		case DLB_ERR_NOSHMEM:
		s="DLB_ERR_NOSHMEM";
		break;
		case DLB_ERR_NOPROC:
		s="DLB_ERR_NOPROC";
		break;
		case DLB_ERR_PDIRTY:
		s="DLB_ERR_PDIRTY";
		break;
		case DLB_ERR_PERM:
		s="DLB_ERR_PERM";
		break;
		case DLB_ERR_TIMEOUT:
		s="DLB_ERR_TIMEOUT";
		break;
		case DLB_ERR_NOCBK:
		s="DLB_ERR_NOCBK";
		break;
		case DLB_ERR_NOENT:
		s="DLB_ERR_NOENT";
		break;
		case DLB_ERR_NOCOMP:
		s="DLB_ERR_NOCOMP";
		break;
		case DLB_ERR_REQST:
		s="DLB_ERR_REQST";
		break;
		case DLB_ERR_NOMEM:
		s="DLB_ERR_NOMEM";
		break;
		case DLB_ERR_NOPOL:
		s="DLB_ERR_NOPOL";
		break;

		default:
		s = "<unknown DLB error code>";
		break;
	}
	return s;
}

#define _dlb_check(s,r) do { if ((r) != DLB_SUCCESS) {fprintf(stderr, "%s:%u, %s - DLB call '%s' %s %d (%s)\n",__FILE__, __LINE__, __func__, (s), (r)>0?"returned warning code":"failed with error code", (r), _dlb_error_str((r))); assert(dlb_ret >= DLB_SUCCESS); }} while (0)

#if 0
/* unused for now */
static void _glibc_cpuset_to_hwloc_cpuset(const cpu_set_t *glibc_cpuset, hwloc_cpuset_t *hwloc_cpuset)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	int status = hwloc_cpuset_from_glibc_sched_affinity(rm->topology, *hwloc_cpuset, glibc_cpuset, sizeof(cpu_set_t));
	assert(status == 0);
}
#endif

static void _hwloc_cpuset_to_glibc_cpuset(const hwloc_cpuset_t hwloc_cpuset, cpu_set_t *glibc_cpuset)
{
	assert(_starpurm != NULL);
	assert(_starpurm->state != state_uninitialized);
	struct s_starpurm *rm = _starpurm;
	int status = hwloc_cpuset_to_glibc_sched_affinity(rm->topology, hwloc_cpuset, glibc_cpuset, sizeof(cpu_set_t));
	assert(status == 0);
}

int starpurm_dlb_notify_starpu_worker_mask_going_to_sleep(const hwloc_cpuset_t hwloc_workers_cpuset)
{
	int status = 0;
	pthread_mutex_lock(&dlb_handle_mutex);
	if (dlb_handle != NULL)
	{
		hwloc_cpuset_t hwloc_to_lend_cpuset = hwloc_bitmap_alloc();
		hwloc_cpuset_t hwloc_to_return_cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_zero(hwloc_to_lend_cpuset);
		hwloc_bitmap_zero(hwloc_to_return_cpuset);
		hwloc_bitmap_and(hwloc_to_lend_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
		hwloc_bitmap_andnot(hwloc_to_return_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
		if (!hwloc_bitmap_iszero(hwloc_to_lend_cpuset))
		{
			cpu_set_t glibc_to_lend_cpuset;
			CPU_ZERO(&glibc_to_lend_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_lend_cpuset, &glibc_to_lend_cpuset);
			DLB_LendCpuMask_sp(dlb_handle, &glibc_to_lend_cpuset);
		}
		if (!hwloc_bitmap_iszero(hwloc_to_return_cpuset))
		{
			cpu_set_t glibc_to_return_cpuset;
			CPU_ZERO(&glibc_to_return_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_return_cpuset, &glibc_to_return_cpuset);
			DLB_ReturnCpuMask_sp(dlb_handle, &glibc_to_return_cpuset);
		}
		hwloc_bitmap_free(hwloc_to_lend_cpuset);
		hwloc_bitmap_free(hwloc_to_return_cpuset);
		status = 1;
	}
	pthread_mutex_unlock(&dlb_handle_mutex);
	return status;
}

int starpurm_dlb_notify_starpu_worker_mask_waking_up(const hwloc_cpuset_t hwloc_workers_cpuset)
{
	int status = 0;
	pthread_mutex_lock(&dlb_handle_mutex);
	if (dlb_handle != NULL)
	{
		hwloc_cpuset_t hwloc_to_reclaim_cpuset = hwloc_bitmap_alloc();
		hwloc_cpuset_t hwloc_to_acquire_cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_zero(hwloc_to_reclaim_cpuset);
		hwloc_bitmap_zero(hwloc_to_acquire_cpuset);
		hwloc_bitmap_and(hwloc_to_reclaim_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
		hwloc_bitmap_andnot(hwloc_to_acquire_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
		if (!hwloc_bitmap_iszero(hwloc_to_reclaim_cpuset))
		{
			cpu_set_t glibc_to_reclaim_cpuset;
			CPU_ZERO(&glibc_to_reclaim_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_reclaim_cpuset, &glibc_to_reclaim_cpuset);
			DLB_ReclaimCpuMask_sp(dlb_handle, &glibc_to_reclaim_cpuset);
		}
		if (!hwloc_bitmap_iszero(hwloc_to_acquire_cpuset))
		{
			cpu_set_t glibc_to_acquire_cpuset;
			CPU_ZERO(&glibc_to_acquire_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_acquire_cpuset, &glibc_to_acquire_cpuset);
			DLB_AcquireCpuMask_sp(dlb_handle, &glibc_to_acquire_cpuset);
		}
		hwloc_bitmap_free(hwloc_to_reclaim_cpuset);
		hwloc_bitmap_free(hwloc_to_acquire_cpuset);
		status = 1;
	}
	pthread_mutex_unlock(&dlb_handle_mutex);
	return status;
}

#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
static void _dlb_callback_enable_cpu(int cpuid)
{
	starpurm_enqueue_event_cpu_unit_available(cpuid);
}

static void _dlb_callback_disable_cpu(int cpuid)
{
	/* nothing */
}
#endif

void starpurm_dlb_init(struct s_starpurm *rm)
{
	_starpurm = rm;

	CPU_ZERO(&starpurm_process_mask);
	starpurm_process_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_and(starpurm_process_cpuset, starpurm_process_cpuset, rm->initially_owned_cpuset_mask);
	_hwloc_cpuset_to_glibc_cpuset(starpurm_process_cpuset, &starpurm_process_mask);

	pthread_mutex_lock(&dlb_handle_mutex);
	dlb_handle = DLB_Init_sp(0, &starpurm_process_mask, "--policy=new --mode=async");

	/* cpu-based callbacks are mutually exclusive with mask-based callbacks,
	 * we only register cpu-based callbacks */
#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
	assert(DLB_CallbackSet_sp(dlb_handle, dlb_callback_disable_cpu, (dlb_callback_t)_dlb_callback_disable_cpu) == DLB_SUCCESS);
	assert(DLB_CallbackSet_sp(dlb_handle, dlb_callback_enable_cpu, (dlb_callback_t)_dlb_callback_enable_cpu) == DLB_SUCCESS);
#endif

	DLB_Enable_sp(dlb_handle);
	pthread_mutex_unlock(&dlb_handle_mutex);

}

void starpurm_dlb_exit(void)
{
	pthread_mutex_lock(&dlb_handle_mutex);
	dlb_handler_t dlb_handle_save = dlb_handle;
	dlb_handle = 0;
	pthread_mutex_unlock(&dlb_handle_mutex);

	/* lend every resources that StarPU may still have */
	DLB_Lend_sp(dlb_handle_save);
	DLB_Return_sp(dlb_handle_save);

	pthread_mutex_lock(&dlb_handle_mutex);
	DLB_Disable_sp(dlb_handle_save);
	DLB_Finalize_sp(dlb_handle_save);
	hwloc_bitmap_free(starpurm_process_cpuset);
	pthread_mutex_unlock(&dlb_handle_mutex);
}
