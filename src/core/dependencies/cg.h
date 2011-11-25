/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <common/config.h>

/* we do not necessarily want to allocate room for 256 dependencies, but we
   want to handle the few situation where there are a lot of dependencies as
   well */
#define STARPU_DYNAMIC_DEPS_SIZE	1

/* randomly choosen ! */
#ifndef STARPU_DYNAMIC_DEPS_SIZE
#define STARPU_NMAXDEPS	256
#endif

/* Completion Group list */
struct _starpu_cg_list {
	unsigned nsuccs; /* how many successors ? */
	unsigned ndeps; /* how many deps ? */
	unsigned ndeps_completed; /* how many deps are done ? */
#ifdef STARPU_DYNAMIC_DEPS_SIZE
	unsigned succ_list_size;
	struct _starpu_cg **succ;
#else
	struct _starpu_cg *succ[STARPU_NMAXDEPS];
#endif
};

#define STARPU_CG_APPS	(1<<0)
#define STARPU_CG_TAG	(1<<1)
#define STARPU_CG_TASK	(1<<2)

/* Completion Group */
struct _starpu_cg {
	unsigned ntags; /* number of tags depended on */
	unsigned remaining; /* number of remaining tags */

	unsigned cg_type; /* STARPU_CG_APPS or STARPU_CG_TAG or STARPU_CG_TASK */

	union {
		/* STARPU_CG_TAG */
		struct _starpu_tag *tag;

		/* STARPU_CG_TASK */
		struct starpu_job_s *job;

		/* STARPU_CG_APPS */
		/* in case this completion group is related to an application,
		 * we have to explicitely wake the waiting thread instead of
		 * reschedule the corresponding task */
		struct {
			unsigned completed;
			pthread_mutex_t cg_mutex;
			pthread_cond_t cg_cond;
		} succ_apps;
	} succ;
};

void _starpu_cg_list_init(struct _starpu_cg_list *list);
void _starpu_cg_list_deinit(struct _starpu_cg_list *list);
void _starpu_add_successor_to_cg_list(struct _starpu_cg_list *successors, struct _starpu_cg *cg);
void _starpu_notify_cg(struct _starpu_cg *cg);
void _starpu_notify_cg_list(struct _starpu_cg_list *successors);
void _starpu_notify_task_dependencies(struct starpu_job_s *j);

#endif // __CG_H__
