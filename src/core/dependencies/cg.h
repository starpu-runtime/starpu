/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __CG_H__
#define __CG_H__

/** @file */

#include <starpu.h>
#include <common/config.h>

/** we do not necessarily want to allocate room for 256 dependencies, but we
   want to handle the few situation where there are a lot of dependencies as
   well */
#define STARPU_DYNAMIC_DEPS_SIZE	1

/* randomly choosen ! */
#ifndef STARPU_DYNAMIC_DEPS_SIZE
#define STARPU_NMAXDEPS	256
#endif

struct _starpu_job;

/** Completion Group list, records both the number of expected notifications
 * before the completion can start, and the list of successors when the
 * completion is finished. */
struct _starpu_cg_list
{
	/** Protects atomicity of the list and the terminated flag */
	struct _starpu_spinlock lock;

	/** Number of notifications to be waited for */
	unsigned ndeps; /* how many deps ? */
	unsigned ndeps_completed; /* how many deps are done ? */
#ifdef STARPU_DEBUG
	/** Array of the notifications, size ndeps */
	struct _starpu_cg **deps;
	/** Which ones have notified, size ndeps */
	char *done;
#endif

	/** Whether the completion is finished.
	 * For restartable/restarted tasks, only the first iteration is taken into account here.
	 */
	unsigned terminated;

	/** List of successors */
	unsigned nsuccs; /* how many successors ? */
#ifdef STARPU_DYNAMIC_DEPS_SIZE
	/** How many allocated items in succ */
	unsigned succ_list_size;
	struct _starpu_cg **succ;
#else
	struct _starpu_cg *succ[STARPU_NMAXDEPS];
#endif
};

enum _starpu_cg_type
{
	STARPU_CG_APPS=(1<<0),
	STARPU_CG_TAG=(1<<1),
	STARPU_CG_TASK=(1<<2)
};

/** Completion Group */
struct _starpu_cg
{
	/** number of tags depended on */
	unsigned ntags;
	/** number of remaining tags */
	unsigned remaining;

#ifdef STARPU_DEBUG
	unsigned ndeps;
	/** array of predecessors, size ndeps */
	void **deps;
	/** which ones have notified, size ndeps */
	char *done;
#endif

	enum _starpu_cg_type cg_type;

	union
	{
		/** STARPU_CG_TAG */
		struct _starpu_tag *tag;

		/** STARPU_CG_TASK */
		struct _starpu_job *job;

		/** STARPU_CG_APPS
		 * in case this completion group is related to an application,
		 * we have to explicitely wake the waiting thread instead of
		 * reschedule the corresponding task */
		struct
		{
			unsigned completed;
			starpu_pthread_mutex_t cg_mutex;
			starpu_pthread_cond_t cg_cond;
		} succ_apps;
	} succ;
};

typedef struct _starpu_notify_job_start_data _starpu_notify_job_start_data;

void _starpu_notify_dependencies(struct _starpu_job *j);
void _starpu_job_notify_start(struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch);
void _starpu_job_notify_ready_soon(struct _starpu_job *j, _starpu_notify_job_start_data *data);

void _starpu_cg_list_init(struct _starpu_cg_list *list);
void _starpu_cg_list_deinit(struct _starpu_cg_list *list);
int _starpu_add_successor_to_cg_list(struct _starpu_cg_list *successors, struct _starpu_cg *cg);
int _starpu_list_task_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, struct starpu_task *task_array[]);
int _starpu_list_task_scheduled_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, struct starpu_task *task_array[]);
int _starpu_list_tag_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, starpu_tag_t tag_array[]);
void _starpu_notify_cg(void *pred, struct _starpu_cg *cg);
void _starpu_notify_cg_list(void *pred, struct _starpu_cg_list *successors);
void _starpu_notify_job_start_cg_list(void *pred, struct _starpu_cg_list *successors, _starpu_notify_job_start_data *data);
void _starpu_notify_task_dependencies(struct _starpu_job *j);
void _starpu_notify_job_start_tasks(struct _starpu_job *j, _starpu_notify_job_start_data *data);

#endif // __CG_H__
