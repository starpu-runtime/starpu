/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <common/config.h>
#include <common/starpu-spinlock.h>
#include <core/dependencies/cg.h>

#define TAG_SIZE        (sizeof(starpu_tag_t)*8)

typedef enum {
	/* this tag is not declared by any task */
	INVALID_STATE,
	/* _starpu_tag_declare was called to associate the tag to a task */
	ASSOCIATED,
	/* some task dependencies are not fulfilled yet */
	BLOCKED,
	/* the task can be (or has been) submitted to the scheduler (all deps
 	 * fulfilled) */
	READY,
// useless ...
//	/* the task has been submitted to the scheduler */
//	SCHEDULED,
	/* the task has been performed */
	DONE
} tag_state;

struct starpu_job_s;

struct tag_s {
	starpu_spinlock_t lock;
	starpu_tag_t id; /* an identifier for the task */
	tag_state state;

	struct starpu_cg_list_s tag_successors;

	struct starpu_job_s *job; /* which job is associated to the tag if any ? */

	unsigned is_assigned;
	unsigned is_submitted;
};

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);

void _starpu_notify_dependencies(struct starpu_job_s *j);
void _starpu_tag_declare(starpu_tag_t id, struct starpu_job_s *job);
void _starpu_tag_set_ready(struct tag_s *tag);

unsigned submit_job_enforce_task_deps(struct starpu_job_s *j);

#endif // __TAGS_H__
