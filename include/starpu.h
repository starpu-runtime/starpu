/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2017                                Inria
 * Copyright (C) 2009-2014,2016-2018                      Université de Bordeaux
 * Copyright (C) 2010-2015,2017,2019                      CNRS
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

#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>

#ifndef _MSC_VER
#include <stdint.h>
#else
#include <windows.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef UINT_PTR uintptr_t;
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef INT_PTR intptr_t;
#endif

#include <starpu_config.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
#include <starpu_opencl.h>
#endif

#include <starpu_thread.h>
#include <starpu_thread_util.h>
#include <starpu_util.h>
#include <starpu_data.h>
#include <starpu_disk.h>
#include <starpu_data_interfaces.h>
#include <starpu_data_filters.h>
#include <starpu_stdlib.h>
#include <starpu_task_bundle.h>
#include <starpu_task.h>
#include <starpu_worker.h>
#include <starpu_perfmodel.h>
#include <starpu_worker.h>
#ifndef BUILDING_STARPU
#include <starpu_task_list.h>
#endif
#include <starpu_task_util.h>
#include <starpu_sched_ctx.h>
#include <starpu_expert.h>
#include <starpu_rand.h>
#include <starpu_cuda.h>
#include <starpu_cublas.h>
#include <starpu_cusparse.h>
#include <starpu_bound.h>
#include <starpu_hash.h>
#include <starpu_profiling.h>
#include <starpu_top.h>
#include <starpu_fxt.h>
#include <starpu_driver.h>
#include <starpu_tree.h>
#include <starpu_openmp.h>
#include <starpu_simgrid_wrap.h>
#include <starpu_bitmap.h>
#include <starpu_clusters.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_conf
{
	int magic;

	const char *sched_policy_name;
	struct starpu_sched_policy *sched_policy;
	void (*sched_policy_init)(unsigned);

	int ncpus;
	int reserve_ncpus;
	int ncuda;
	int nopencl;
	int nmic;
	int nscc;
        int nmpi_ms;

	unsigned use_explicit_workers_bindid;
	unsigned workers_bindid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_cuda_gpuid;
	unsigned workers_cuda_gpuid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_opencl_gpuid;
	unsigned workers_opencl_gpuid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_mic_deviceid;
	unsigned workers_mic_deviceid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_scc_deviceid;
	unsigned workers_scc_deviceid[STARPU_NMAXWORKERS];

	unsigned use_explicit_workers_mpi_ms_deviceid;
	unsigned workers_mpi_ms_deviceid[STARPU_NMAXWORKERS];

	int bus_calibrate;
	int calibrate;

	int single_combined_worker;

	char *mic_sink_program_path;

	int disable_asynchronous_copy;
	int disable_asynchronous_cuda_copy;
	int disable_asynchronous_opencl_copy;
	int disable_asynchronous_mic_copy;
	int disable_asynchronous_mpi_ms_copy;

	unsigned *cuda_opengl_interoperability;
	unsigned n_cuda_opengl_interoperability;

	struct starpu_driver *not_launched_drivers;
	unsigned n_not_launched_drivers;

	unsigned trace_buffer_size;
	int global_sched_ctx_min_priority;
	int global_sched_ctx_max_priority;

#ifdef STARPU_WORKER_CALLBACKS
	void (*callback_worker_going_to_sleep)(unsigned workerid);
	void (*callback_worker_waking_up)(unsigned workerid);
#endif
};

int starpu_conf_init(struct starpu_conf *conf);

int starpu_init(struct starpu_conf *conf) STARPU_WARN_UNUSED_RESULT;
int starpu_initialize(struct starpu_conf *user_conf, int *argc, char ***argv);
int starpu_is_initialized(void);
void starpu_wait_initialized(void);

#define STARPU_THREAD_ACTIVE (1 << 0)
unsigned starpu_get_next_bindid(unsigned flags, unsigned *preferred, unsigned npreferred);
int starpu_bind_thread_on(int cpuid, unsigned flags, const char *name);

void starpu_pause(void);
void starpu_resume(void);

void starpu_shutdown(void);

void starpu_topology_print(FILE *f);

int starpu_asynchronous_copy_disabled(void);
int starpu_asynchronous_cuda_copy_disabled(void);
int starpu_asynchronous_opencl_copy_disabled(void);
int starpu_asynchronous_mic_copy_disabled(void);
int starpu_asynchronous_mpi_ms_copy_disabled(void);

void starpu_display_stats();

void starpu_get_version(int *major, int *minor, int *release);

#ifdef __cplusplus
}
#endif

#include "starpu_deprecated_api.h"

#endif /* __STARPU_H__ */
