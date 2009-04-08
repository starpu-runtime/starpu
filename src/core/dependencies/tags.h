/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __TAGS_H__
#define __TAGS_H__

#include <stdint.h>
#include <starpu-mutex.h>
#include <core/jobs.h>

/* we do not necessarily want to allocate room for 256 dependencies, but we
   want to handle the few situation where there are a lot of dependencies as
   well */
#define DYNAMIC_DEPS_SIZE	1

/* randomly choosen ! */
#ifndef DYNAMIC_DEPS_SIZE
#define NMAXDEPS	256
#endif

#define TAG_SIZE        (sizeof(starpu_tag_t)*8)

typedef enum {
	DONE,
	READY,
	SCHEDULED,
	BLOCKED
} tag_state;

struct job_s;

struct tag_s {
	starpu_mutex lock;
	starpu_tag_t id; /* an identifier for the task */
	tag_state state;
	unsigned nsuccs; /* how many successors ? */
#ifdef DYNAMIC_DEPS_SIZE
	unsigned succ_list_size;
	struct _cg_t **succ;
#else
	struct _cg_t *succ[NMAXDEPS];
#endif
	struct job_s *job; /* which job is associated to the tag if any ? */

	unsigned is_assigned;
	unsigned is_submitted;
};

typedef struct _cg_t {
	unsigned ntags; /* number of remaining tags */
	struct tag_s *tag; /* which tags depends on that cg ?  */

	unsigned completed;

	/* in case this completion group is related to an application, we have
 	 * to explicitely wake the waiting thread instead of reschedule the
	 * corresponding task */
	unsigned used_by_apps;
	pthread_mutex_t cg_mutex;
	pthread_cond_t cg_cond;
} cg_t;

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);

void notify_dependencies(struct job_s *j);
void tag_declare(starpu_tag_t id, struct job_s *job);

unsigned submit_job_enforce_task_deps(struct job_s *j);

#endif // __TAGS_H__
