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

#include <starpu.h>
#include <common/config.h>
#include <core/jobs.h>
#include <core/dependencies/cg.h>
#include <core/dependencies/tags.h>

void _starpu_cg_list_init(struct cg_list_s *list)
{
	list->nsuccs = 0;
	list->ndeps = 0;
	list->ndeps_completed = 0;

#ifdef DYNAMIC_DEPS_SIZE
	/* this is a small initial default value ... may be changed */
	list->succ_list_size = 0;
	list->succ =
		realloc(NULL, list->succ_list_size*sizeof(struct cg_s *));
#endif
}

void _starpu_add_successor_to_cg_list(struct cg_list_s *successors, cg_t *cg)
{
	/* where should that cg should be put in the array ? */
	unsigned index = STARPU_ATOMIC_ADD(&successors->nsuccs, 1) - 1;

#ifdef DYNAMIC_DEPS_SIZE
	if (index >= successors->succ_list_size)
	{
		/* the successor list is too small */
		if (successors->succ_list_size > 0)
			successors->succ_list_size *= 2;
		else
			successors->succ_list_size = 4;

		/* NB: this is thread safe as the tag->lock is taken */
		successors->succ = realloc(successors->succ, 
			successors->succ_list_size*sizeof(struct cg_s *));
	}
#else
	STARPU_ASSERT(index < NMAXDEPS);
#endif
	successors->succ[index] = cg;
}

void _starpu_notify_cg(cg_t *cg)
{
	STARPU_ASSERT(cg);
	unsigned remaining = STARPU_ATOMIC_ADD(&cg->remaining, -1);
	if (remaining == 0) {
		cg->remaining = cg->ntags;

		struct tag_s *tag;
		struct cg_list_s *tag_successors, *job_successors;
		job_t j;

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
					/* reset the counter so that we can reuse the completion group */
					tag_successors->ndeps_completed = 0;
					_starpu_tag_set_ready(tag);
				}
				break;

			case CG_TASK:
				/* TODO */
				j = cg->succ.job;

				job_successors = &j->job_successors;

				unsigned ndeps_completed =
					STARPU_ATOMIC_ADD(&job_successors->ndeps_completed, 1);

				if (job_successors->ndeps == ndeps_completed)
				{
					/* reset the counter so that we can reuse the completion group */
					job_successors->ndeps_completed = 0;
					_starpu_enforce_deps_starting_from_data(j);
				}

				break;

			default:
				STARPU_ABORT();
		}
	}
}

void _starpu_notify_cg_list(struct cg_list_s *successors)
{
	unsigned nsuccs;
	unsigned succ;

	nsuccs = successors->nsuccs;

	for (succ = 0; succ < nsuccs; succ++)
	{
		struct cg_s *cg = successors->succ[succ];
		struct tag_s *cgtag = cg->succ.tag;

		unsigned cg_type = cg->cg_type;

		if (cg_type == CG_TAG)
			starpu_spin_lock(&cgtag->lock);

		_starpu_notify_cg(cg);
		if (cg_type == CG_APPS) {
			/* Remove the temporary ref to the cg */
			memmove(&successors->succ[succ], &successors->succ[succ+1], (nsuccs-(succ+1)) * sizeof(successors->succ[succ]));
			succ--;
			nsuccs--;
			successors->nsuccs--;
		}

		if (cg_type == CG_TAG)
			starpu_spin_unlock(&cgtag->lock);
	}
}
