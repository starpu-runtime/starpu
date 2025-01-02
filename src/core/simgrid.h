/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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

#ifndef __SIMGRID_H__
#define __SIMGRID_H__

/** @file */

/* Note: when changing something here, update the include list in configure.ac
 * in the part that tries to enable stdc++11 */
#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MSG_H
#include <simgrid/msg.h>
#elif defined(STARPU_HAVE_MSG_MSG_H)
#include <msg/msg.h>
#endif

#ifdef STARPU_HAVE_XBT_BASE_H
#include <xbt/base.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_VERSION_H
#include <simgrid/version.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_ZONE_H
#include <simgrid/zone.h>
#endif
#ifdef STARPU_HAVE_SIMGRID_HOST_H
#include <simgrid/host.h>
#endif
#if defined(HAVE_SIMGRID_SIMDAG_H) && (SIMGRID_VERSION >= 31300)
#include <simgrid/simdag.h>
#endif

#include <xbt/xbt_os_time.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_SIMGRID
#pragma GCC visibility push(hidden)

struct _starpu_pthread_args
{
	void *(*f)(void*);
	void *arg;
};

#if (SIMGRID_VERSION >= 32600)
typedef void _starpu_simgrid_main_ret;
#define _STARPU_SIMGRID_MAIN_RETURN do { } while (0)
#else
typedef int _starpu_simgrid_main_ret;
#define _STARPU_SIMGRID_MAIN_RETURN return 0
#endif
#if (SIMGRID_VERSION >= 31500) && (SIMGRID_VERSION != 31559)
typedef sg_link_t starpu_sg_link_t;
#else
typedef SD_link_t starpu_sg_link_t;
#endif
_starpu_simgrid_main_ret
_starpu_simgrid_thread_start(int argc, char *argv[]);

#define MAX_TSD 16

#define STARPU_MPI_AS_PREFIX "StarPU-MPI"
#define _starpu_simgrid_running_smpi() (getenv("SMPI_GLOBAL_SIZE") != NULL)

void _starpu_start_simgrid(int *argc, char **argv);

void _starpu_simgrid_init_early(int *argc, char ***argv);
void _starpu_simgrid_init(void);
void _starpu_simgrid_cpp_init(void);
void _starpu_simgrid_deinit(void);
void _starpu_simgrid_deinit_late(void);
void _starpu_simgrid_actor_setup(void);
void _starpu_simgrid_wait_tasks(int workerid);
struct _starpu_job;
void _starpu_simgrid_submit_job(int workerid, int sched_ctx_id, struct _starpu_job *job, struct starpu_perfmodel_arch* perf_arch, double length, double energy, unsigned *finished);
struct _starpu_data_request;
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_simgrid_wait_transfer_event(void *event);
int _starpu_simgrid_test_transfer_event(void *event);
void _starpu_simgrid_sync_gpus(void);
/** Return the number of hosts prefixed by PREFIX */
int _starpu_simgrid_get_nbhosts(const char *prefix);
unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid);
const char *_starpu_simgrid_get_devname(const char *prefix, unsigned devid);
starpu_sg_host_t _starpu_simgrid_get_host_by_name(const char *name);
starpu_sg_host_t _starpu_simgrid_get_memnode_host(unsigned node);
struct _starpu_worker;
starpu_sg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker);
void _starpu_simgrid_get_platform_path(int version, char *path, size_t maxlen);
#if defined(HAVE_SG_ZONE_GET_BY_NAME) || defined(sg_zone_get_by_name)
sg_netzone_t _starpu_simgrid_get_as_by_name(const char *name);
#else
msg_as_t _starpu_simgrid_get_as_by_name(const char *name);
#endif
#pragma weak starpu_mpi_world_rank
extern int starpu_mpi_world_rank(void);
#pragma weak _starpu_mpi_simgrid_init
int _starpu_mpi_simgrid_init(int argc, char *argv[]);

extern starpu_pthread_queue_t _starpu_simgrid_transfer_queue[STARPU_MAXNODES];
extern starpu_pthread_queue_t _starpu_simgrid_task_queue[STARPU_NMAXWORKERS];

#ifdef STARPU_HAVE_S4U_ON_TIME_ADVANCE_CB
extern starpu_pthread_mutex_t _starpu_simgrid_time_advance_mutex;
extern starpu_pthread_cond_t _starpu_simgrid_time_advance_cond;
#endif

#define _starpu_simgrid_cuda_malloc_cost() starpu_getenv_number_default("STARPU_SIMGRID_CUDA_MALLOC_COST", 1)
#define _starpu_simgrid_cuda_queue_cost() starpu_getenv_number_default("STARPU_SIMGRID_CUDA_QUEUE_COST", 1)
#define _starpu_simgrid_task_submit_cost() starpu_getenv_number_default("STARPU_SIMGRID_TASK_SUBMIT_COST", 1)
#define _starpu_simgrid_task_push_cost() starpu_getenv_number_default("STARPU_SIMGRID_TASK_PUSH_COST", 1)
#define _starpu_simgrid_fetching_input_cost() starpu_getenv_number_default("STARPU_SIMGRID_FETCHING_INPUT_COST", 1)
#define _starpu_simgrid_sched_cost() starpu_getenv_number_default("STARPU_SIMGRID_SCHED_COST", 0)

/** Called at initialization to count how many GPUs are interfering with each
 * bus */
void _starpu_simgrid_count_ngpus(void);

extern size_t _starpu_default_stack_size;
void _starpu_simgrid_set_stack_size(size_t stack_size);
void _starpu_simgrid_xbt_thread_create(const char *name, starpu_pthread_attr_t *attr, void_f_pvoid_t code,
				       void *param);

#define _SIMGRID_TIMER_BEGIN(cond)			\
	{		\
		xbt_os_timer_t __timer = NULL;		\
		if (cond) {		\
		  __timer = xbt_os_timer_new();		\
		  xbt_os_threadtimer_start(__timer);	\
		}
#define _SIMGRID_TIMER_END		\
		if (__timer) {		\
			xbt_os_threadtimer_stop(__timer);		\
			starpu_sleep(xbt_os_timer_elapsed(__timer));\
			xbt_os_timer_free(__timer);		\
		}	\
	}

#pragma GCC visibility pop

#else // !STARPU_SIMGRID
#define _SIMGRID_TIMER_BEGIN(cond) {
#define _SIMGRID_TIMER_END }
#endif

/** Experimental functions for OOC stochastic analysis */
/* disk <-> MAIN_RAM only */
#if defined(STARPU_SIMGRID) && 0
void _starpu_simgrid_data_new(size_t size);
void _starpu_simgrid_data_increase(size_t size);
void _starpu_simgrid_data_alloc(size_t size);
void _starpu_simgrid_data_free(size_t size);
void _starpu_simgrid_data_transfer(size_t size, unsigned src_node, unsigned dst_node);
#else
#define _starpu_simgrid_data_new(size) (void)0
#define _starpu_simgrid_data_increase(size) (void)0
#define _starpu_simgrid_data_alloc(size) (void)0
#define _starpu_simgrid_data_free(size) (void)0
#define _starpu_simgrid_data_transfer(size, src_node, dst_node) (void)0
#endif

#ifdef __cplusplus
}
#endif

#endif // __SIMGRID_H__
