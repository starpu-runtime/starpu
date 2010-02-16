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

#include <core/dependencies/htable.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>
#include <core/dependencies/data-concurrency.h>
#include <starpu.h>
#include <common/config.h>
#include <core/dependencies/tags.h>
#include <core/dependencies/cg.h>

void _starpu_notify_cg(cg_t *cg)
{
	STARPU_ASSERT(cg);
	unsigned remaining = STARPU_ATOMIC_ADD(&cg->remaining, -1);
	if (remaining == 0) {
		cg->remaining = cg->ntags;

		struct tag_s *tag;
		struct cg_list_s *tag_successors;

		/* the group is now completed */
		switch (cg->cg_type) {
			case CG_APPS:
				/* this is a cg for an application waiting on a set of
	 			 * tags, wake the thread */
				pthread_mutex_lock(&cg->succ.succ_apps.cg_mutex);
				cg->succ.succ_apps.completed = 1;
				pthread_cond_signal(&cg->succ.succ_apps.cg_cond);
				pthread_mutex_unlock(&cg->succ.succ_apps.cg_mutex);
				break;

			case CG_TAG:
				tag = cg->succ.tag;
				tag_successors = &tag->tag_successors;
	
				tag_successors->ndeps_completed++;
	
				if ((tag->state == BLOCKED) &&
					(tag_successors->ndeps == tag_successors->ndeps_completed)) {
					tag_successors->ndeps_completed = 0;
					_starpu_tag_set_ready(tag);
				}
				break;

			case CG_TASK:
				/* TODO */
				break;

			default:
				STARPU_ABORT();
		}
	}
}


