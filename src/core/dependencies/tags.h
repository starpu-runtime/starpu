/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __TAGS_H__
#define __TAGS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <common/starpu_spinlock.h>
#include <core/dependencies/cg.h>

#pragma GCC visibility push(hidden)

#define _STARPU_TAG_SIZE        (sizeof(starpu_tag_t)*8)

enum _starpu_tag_state
{
	/** this tag is not declared by any task */
	STARPU_INVALID_STATE,
	/** _starpu_tag_declare was called to associate the tag to a task */
	STARPU_ASSOCIATED,
	/** some task dependencies are not fulfilled yet */
	STARPU_BLOCKED,
	/** the task can be (or has been) submitted to the scheduler (all deps fulfilled) */
	STARPU_READY,
// useless ...
//	/** the task has been submitted to the scheduler */
//	STARPU_SCHEDULED,
	/** the task has been performed */
	STARPU_DONE
};

struct _starpu_job;

struct _starpu_tag
{
	/**
	   Lock for this structure. Locking order is in dependency order: a tag
	   * must not be locked before locking a tag it depends on */
	struct _starpu_spinlock lock;
	/** an identifier for the task */
	starpu_tag_t id;
	enum _starpu_tag_state state;

	struct _starpu_cg_list tag_successors;

	/** which job is associated to the tag if any ? */
	struct _starpu_job *job;

	unsigned is_assigned;
	unsigned is_submitted;
};

void _starpu_init_tags(void);

void _starpu_notify_tag_dependencies(struct _starpu_tag *tag);
void _starpu_notify_job_start_tag_dependencies(struct _starpu_tag *tag, _starpu_notify_job_start_data *data);

void _starpu_tag_declare(starpu_tag_t id, struct _starpu_job *job);

/** lock should be taken, and this releases it */
void _starpu_tag_set_ready(struct _starpu_tag *tag);

#pragma GCC visibility pop

#endif // __TAGS_H__
