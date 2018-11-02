/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017,2018                                Inria
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

/*
 * DLB interfacing
 */

static dlb_handler_t      dlb_handle;
static cpu_set_t          starpurm_process_mask;
static struct s_starpurm *_starpurm = NULL;
static pthread_mutex_t dlb_handle_mutex = PTHREAD_MUTEX_INITIALIZER;

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
		cpu_set_t glibc_workers_cpuset;
		CPU_ZERO(&glibc_workers_cpuset);
		_hwloc_cpuset_to_glibc_cpuset(hwloc_workers_cpuset, &glibc_workers_cpuset);
		DLB_LendCpuMask_sp(dlb_handle, &glibc_workers_cpuset);
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
		cpu_set_t glibc_workers_cpuset;
		CPU_ZERO(&glibc_workers_cpuset);
		_hwloc_cpuset_to_glibc_cpuset(hwloc_workers_cpuset, &glibc_workers_cpuset);
		DLB_ReclaimCpuMask_sp(dlb_handle, &glibc_workers_cpuset);
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
	_hwloc_cpuset_to_glibc_cpuset(rm->selected_cpuset, &starpurm_process_mask);

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

	pthread_mutex_lock(&dlb_handle_mutex);
	DLB_Disable_sp(dlb_handle_save);
	DLB_Finalize_sp(dlb_handle_save);
	pthread_mutex_unlock(&dlb_handle_mutex);
}
