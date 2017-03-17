/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2017  Université de Bordeaux
 * Copyright (C) 2016  INRIA
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

#ifdef STARPU_SIMGRID
#ifdef STARPU_HAVE_SIMGRID_MSG_H
#include <simgrid/msg.h>
#else
#include <msg/msg.h>
#endif

#include <datawizard/data_request.h>
#include <xbt/xbt_os_time.h>

struct _starpu_pthread_args
{
	void *(*f)(void*);
	void *arg;
};

#define MAX_TSD 16

#define STARPU_MPI_AS_PREFIX "StarPU-MPI"
#define _starpu_simgrid_running_smpi() (getenv("SMPI_GLOBAL_SIZE") != NULL)

void _starpu_simgrid_init_early(int *argc, char ***argv);
void _starpu_simgrid_init(void);
void _starpu_simgrid_deinit(void);
void _starpu_simgrid_wait_tasks(int workerid);
void _starpu_simgrid_submit_job(int workerid, struct _starpu_job *job, struct starpu_perfmodel_arch* perf_arch, double length, unsigned *finished);
int _starpu_simgrid_transfer(size_t size, unsigned src_node, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_simgrid_wait_transfer_event(union _starpu_async_channel_event *event);
int _starpu_simgrid_test_transfer_event(union _starpu_async_channel_event *event);
void _starpu_simgrid_sync_gpus(void);
/* Return the number of hosts prefixed by PREFIX */
int _starpu_simgrid_get_nbhosts(const char *prefix);
unsigned long long _starpu_simgrid_get_memsize(const char *prefix, unsigned devid);
msg_host_t _starpu_simgrid_get_host_by_name(const char *name);
msg_host_t _starpu_simgrid_get_memnode_host(unsigned node);
struct _starpu_worker;
msg_host_t _starpu_simgrid_get_host_by_worker(struct _starpu_worker *worker);
void _starpu_simgrid_get_platform_path(int version, char *path, size_t maxlen);
msg_as_t _starpu_simgrid_get_as_by_name(const char *name);
#pragma weak starpu_mpi_world_rank
extern int starpu_mpi_world_rank(void);
#pragma weak _starpu_mpi_simgrid_init
int _starpu_mpi_simgrid_init(int argc, char *argv[]);

starpu_pthread_queue_t _starpu_simgrid_transfer_queue[STARPU_MAXNODES];
starpu_pthread_queue_t _starpu_simgrid_task_queue[STARPU_NMAXWORKERS];

#define _starpu_simgrid_cuda_malloc_cost() starpu_get_env_number_default("STARPU_SIMGRID_CUDA_MALLOC_COST", 1)
#define _starpu_simgrid_queue_malloc_cost() starpu_get_env_number_default("STARPU_SIMGRID_QUEUE_MALLOC_COST", 1)
#define _starpu_simgrid_task_submit_cost() starpu_get_env_number_default("STARPU_SIMGRID_TASK_SUBMIT_COST", 1)
#define _starpu_simgrid_fetching_input_cost() starpu_get_env_number_default("STARPU_SIMGRID_FETCHING_INPUT_COST", 1)
#define _starpu_simgrid_sched_cost() starpu_get_env_number_default("STARPU_SIMGRID_SCHED_COST", 1)

/* Called at initialization to count how many GPUs are interfering with each
 * bus */
void _starpu_simgrid_count_ngpus(void);

void _starpu_simgrid_xbt_thread_create(const char *name, void_f_pvoid_t code,
				       void *param);

#define _SIMGRID_TIMER_BEGIN		\
	{		\
		xbt_os_timer_t __timer = NULL;		\
		if (_starpu_simgrid_sched_cost()) {		\
		  __timer = xbt_os_timer_new();		\
		  xbt_os_threadtimer_start(__timer);	\
		}
#define _SIMGRID_TIMER_END		\
		if (__timer) {		\
			xbt_os_threadtimer_stop(__timer);		\
			MSG_process_sleep(xbt_os_timer_elapsed(__timer));\
			xbt_os_timer_free(__timer);		\
		}	\
	}

#else // !STARPU_SIMGRID
#define _SIMGRID_TIMER_BEGIN {
#define _SIMGRID_TIMER_END }
#endif

#endif // __SIMGRID_H__
