/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <common/config.h>

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
static int glibc_cpuid_to_unitid[CPU_SETSIZE];
static int *unitid_to_glibc_cpuid = NULL;

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

#define _dlb_check(s,r) do { if ((r) != DLB_SUCCESS) {fprintf(stderr, "%s:%d, %s - DLB call '%s' %s %d (%s)\n",__FILE__, __LINE__, __func__, (s), (r)>0?"returned warning code":"failed with error code", (r), _dlb_error_str((r))); assert(dlb_ret >= DLB_SUCCESS); }} while (0)

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
#ifdef STARPURM_DLB_VERBOSE
		{
			char * s_to_lend = NULL;
			char * s_to_return = NULL;
			hwloc_bitmap_asprintf(&s_to_lend, hwloc_to_lend_cpuset);
			hwloc_bitmap_asprintf(&s_to_return, hwloc_to_return_cpuset);
			fprintf(stderr, "%s: to_lend='%s', to_return='%s'\n", __func__, s_to_lend, s_to_return);
			free(s_to_lend);
			free(s_to_return);
		}
#endif
		if (!hwloc_bitmap_iszero(hwloc_to_lend_cpuset))
		{
			cpu_set_t glibc_to_lend_cpuset;
			CPU_ZERO(&glibc_to_lend_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_lend_cpuset, &glibc_to_lend_cpuset);
			int dlb_ret = DLB_LendCpuMask_sp(dlb_handle, &glibc_to_lend_cpuset);
			_dlb_check("DLB_LendCpuMask_sp", dlb_ret);
		}
		if (!hwloc_bitmap_iszero(hwloc_to_return_cpuset))
		{
			cpu_set_t glibc_to_return_cpuset;
			CPU_ZERO(&glibc_to_return_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_return_cpuset, &glibc_to_return_cpuset);
			/* Use DLB_Lend for returning borrowed units. DLB_Return seems to require that
			 * a reclaim has previously been emitted by the unit owning runtime system */
#if 0
			int dlb_ret = DLB_ReturnCpuMask_sp(dlb_handle, &glibc_to_return_cpuset);
			_dlb_check("DLB_ReturnCpuMask_sp", dlb_ret);
#else
			int dlb_ret = DLB_LendCpuMask_sp(dlb_handle, &glibc_to_return_cpuset);
			_dlb_check("DLB_LendCpuMask_sp", dlb_ret);
#endif
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
		hwloc_cpuset_t hwloc_to_borrow_cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_zero(hwloc_to_reclaim_cpuset);
		hwloc_bitmap_zero(hwloc_to_borrow_cpuset);
		hwloc_bitmap_and(hwloc_to_reclaim_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
		hwloc_bitmap_andnot(hwloc_to_borrow_cpuset, hwloc_workers_cpuset, starpurm_process_cpuset);
#ifdef STARPURM_DLB_VERBOSE
		{
			char * s_to_reclaim = NULL;
			char * s_to_borrow = NULL;
			hwloc_bitmap_asprintf(&s_to_reclaim, hwloc_to_reclaim_cpuset);
			hwloc_bitmap_asprintf(&s_to_borrow, hwloc_to_borrow_cpuset);
			fprintf(stderr, "%s: to_reclaim='%s', to_borrow='%s'\n", __func__, s_to_reclaim, s_to_borrow);
			free(s_to_reclaim);
			free(s_to_borrow);
		}
#endif
		if (!hwloc_bitmap_iszero(hwloc_to_reclaim_cpuset))
		{
			cpu_set_t glibc_to_reclaim_cpuset;
			CPU_ZERO(&glibc_to_reclaim_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_reclaim_cpuset, &glibc_to_reclaim_cpuset);
			int dlb_ret = DLB_ReclaimCpuMask_sp(dlb_handle, &glibc_to_reclaim_cpuset);
			_dlb_check("DLB_ReclaimCpuMask_sp", dlb_ret);
		}
		if (!hwloc_bitmap_iszero(hwloc_to_borrow_cpuset))
		{
			cpu_set_t glibc_to_borrow_cpuset;
			CPU_ZERO(&glibc_to_borrow_cpuset);
			_hwloc_cpuset_to_glibc_cpuset(hwloc_to_borrow_cpuset, &glibc_to_borrow_cpuset);
			int dlb_ret = DLB_BorrowCpuMask_sp(dlb_handle, &glibc_to_borrow_cpuset);
			_dlb_check("DLB_BorrowCpuMask_sp", dlb_ret);
		}
		hwloc_bitmap_free(hwloc_to_reclaim_cpuset);
		hwloc_bitmap_free(hwloc_to_borrow_cpuset);
		status = 1;
	}
	pthread_mutex_unlock(&dlb_handle_mutex);
	return status;
}

#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
#ifdef STARPURM_HAVE_DLB_CALLBACK_ARG
static void _dlb_callback_enable_cpu(int cpuid, void *arg)
#else
static void _dlb_callback_enable_cpu(int cpuid)
#endif
{
#ifdef STARPURM_HAVE_DLB_CALLBACK_ARG
	(void) arg;
#endif
	int unitid = glibc_cpuid_to_unitid[cpuid];
#ifdef STARPURM_DLB_VERBOSE
	fprintf(stderr, "%s: cpuid=%d, unitid=%d\n", __func__, cpuid, unitid);
#endif
	if (unitid != -1)
	{
		starpurm_enqueue_event_cpu_unit_available(unitid);
	}
}

#ifdef STARPURM_HAVE_DLB_CALLBACK_ARG
static void _dlb_callback_disable_cpu(int cpuid, void *arg)
#else
static void _dlb_callback_disable_cpu(int cpuid)
#endif
{
#ifdef STARPURM_HAVE_DLB_CALLBACK_ARG
	(void) arg;
#endif
	int unitid = glibc_cpuid_to_unitid[cpuid];
#ifdef STARPURM_DLB_VERBOSE
	fprintf(stderr, "%s: cpuid=%d, unitid=%d\n", __func__, cpuid, unitid);
#endif
	if (unitid != -1)
	{
		/* nothing */
	}
}
#endif

void starpurm_dlb_init(struct s_starpurm *rm)
{
	_starpurm = rm;

	{
		int unitid;
		int cpuid;
		unitid_to_glibc_cpuid = malloc(rm->nunits * sizeof(*unitid_to_glibc_cpuid));
		for (cpuid = 0; cpuid<CPU_SETSIZE; cpuid++)
		{
			glibc_cpuid_to_unitid[cpuid] = -1;
		}

		for (unitid = 0; unitid < rm->nunits; unitid++)
		{
			hwloc_cpuset_t unit_cpuset = starpurm_get_unit_cpuset(unitid);
			cpu_set_t unit_mask;
			CPU_ZERO(&unit_mask);
			_hwloc_cpuset_to_glibc_cpuset(unit_cpuset, &unit_mask);
			unitid_to_glibc_cpuid[unitid] = -1;
			for (cpuid = 0; cpuid<CPU_SETSIZE; cpuid++)
			{
				if (CPU_ISSET(cpuid, &unit_mask))
				{
					/* assume no overlap on units cpuid */
					assert(glibc_cpuid_to_unitid[cpuid] == -1);

					unitid_to_glibc_cpuid[unitid] = cpuid;
					glibc_cpuid_to_unitid[cpuid] = unitid;
					break;
				}
			}
#ifdef STARPURM_DLB_VERBOSE
			{
				char * s_unit = NULL;
				hwloc_bitmap_asprintf(&s_unit, unit_cpuset);
				fprintf(stderr, "%s: unitid=%d, cpuid=%d, unit hwloc cpuset=%s\n", __func__, unitid, cpuid, s_unit);
				free(s_unit);
			}
#endif
			hwloc_bitmap_free(unit_cpuset);
		}
	}

	CPU_ZERO(&starpurm_process_mask);
	starpurm_process_cpuset = hwloc_bitmap_dup(rm->selected_cpuset);
	hwloc_bitmap_and(starpurm_process_cpuset, starpurm_process_cpuset, rm->initially_owned_cpuset_mask);
	_hwloc_cpuset_to_glibc_cpuset(starpurm_process_cpuset, &starpurm_process_mask);
#ifdef STARPURM_DLB_VERBOSE
	{
		char * s_reachable = NULL;
		char * s_initially_owned = NULL;
		hwloc_bitmap_asprintf(&s_reachable, rm->selected_cpuset);
		hwloc_bitmap_asprintf(&s_initially_owned, starpurm_process_cpuset);
		fprintf(stderr, "%s: StarPU reachable units='%s', StarPU initially owned units='%s'\n", __func__, s_reachable, s_initially_owned);
		free(s_reachable);
		free(s_initially_owned);
	}
#endif

	pthread_mutex_lock(&dlb_handle_mutex);

	/* TODO: autodetect DLB policy according to DLB version */
#if 1
	dlb_handle = DLB_Init_sp(0, &starpurm_process_mask, "--lewi=yes --drom=no --mode=async");
#else
	dlb_handle = DLB_Init_sp(0, &starpurm_process_mask, "--policy=new --drom=no --mode=async");
#endif

	/* cpu-based callbacks are mutually exclusive with mask-based callbacks,
	 * we only register cpu-based callbacks */
	int dlb_ret;
#ifdef STARPURM_STARPU_HAVE_WORKER_CALLBACKS
#ifdef STARPURM_HAVE_DLB_CALLBACK_ARG
	dlb_ret = DLB_CallbackSet_sp(dlb_handle, dlb_callback_disable_cpu, (dlb_callback_t)_dlb_callback_disable_cpu, NULL);
	_dlb_check("DLB_CallbackSet_sp", dlb_ret);
	dlb_ret = DLB_CallbackSet_sp(dlb_handle, dlb_callback_enable_cpu, (dlb_callback_t)_dlb_callback_enable_cpu, NULL);
	_dlb_check("DLB_CallbackSet_sp", dlb_ret);
#else
	dlb_ret = DLB_CallbackSet_sp(dlb_handle, dlb_callback_disable_cpu, (dlb_callback_t)_dlb_callback_disable_cpu);
	_dlb_check("DLB_CallbackSet_sp", dlb_ret);
	dlb_ret = DLB_CallbackSet_sp(dlb_handle, dlb_callback_enable_cpu, (dlb_callback_t)_dlb_callback_enable_cpu);
	_dlb_check("DLB_CallbackSet_sp", dlb_ret);
#endif
#endif

	dlb_ret = DLB_Enable_sp(dlb_handle);
	_dlb_check("DLB_Enable_sp", dlb_ret);
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
	free(unitid_to_glibc_cpuid);
	pthread_mutex_unlock(&dlb_handle_mutex);
}
