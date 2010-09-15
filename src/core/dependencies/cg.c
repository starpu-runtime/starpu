/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <common/utils.h>
#include <core/jobs.h>
#include <core/dependencies/cg.h>
#include <core/dependencies/tags.h>

void _starpu_cg_list_init(struct starpu_cg_list_s *list)
{
	list->nsuccs = 0;
	list->ndeps = 0;
	list->ndeps_completed = 0;

#ifdef STARPU_DYNAMIC_DEPS_SIZE
	/* this is a small initial default value ... may be changed */
	list->succ_list_size = 0;
	list->succ =
		realloc(NULL, list->succ_list_size*sizeof(struct starpu_cg_s *));
#endif
}

void _starpu_cg_list_deinit(struct starpu_cg_list_s *list)
{
	unsigned id;
	for (id = 0; id < list->nsuccs; id++)
	{
		starpu_cg_t *cg = list->succ[id];

		/* We remove the reference on the completion group, and free it
		 * if there is no more reference. */		
		unsigned ntags = STARPU_ATOMIC_ADD(&cg->ntags, -1);
		if (ntags == 0)
			free(list->succ[id]);
	}

#ifdef STARPU_DYNAMIC_DEPS_SIZE
	free(list->succ);
#endif
}

void _starpu_add_successor_to_cg_list(struct starpu_cg_list_s *successors, starpu_cg_t *cg)
{
	STARPU_ASSERT(cg);

	/* where should that cg should be put in the array ? */
	unsigned index = STARPU_ATOMIC_ADD(&successors->nsuccs, 1) - 1;

#ifdef STARPU_DYNAMIC_DEPS_SIZE
	if (index >= successors->succ_list_size)
	{
		/* the successor list is too small */
		if (successors->succ_list_size > 0)
			successors->succ_list_size *= 2;
		else
			successors->succ_list_size = 4;

		/* NB: this is thread safe as the tag->lock is taken */
		successors->succ = realloc(successors->succ, 
			successors->succ_list_size*sizeof(struct starpu_cg_s *));
	}
#else
	STARPU_ASSERT(index < STARPU_NMAXDEPS);
#endif
	successors->succ[index] = cg;
}

void _starpu_notify_cg(starpu_cg_t *cg)
{
	STARPU_ASSERT(cg);
	unsigned remaining = STARPU_ATOMIC_ADD(&cg->remaining, -1);

	if (remaining == 0) {
		cg->remaining = cg->ntags;

		struct starpu_tag_s *tag;
		struct starpu_cg_list_s *tag_successors, *job_successors;
		starpu_job_t j;

		/* the group is now completed */
		switch (cg->cg_type) {
			case STARPU_CG_APPS:
				/* this is a cg for an application waiting on a set of
	 			 * tags, wake the thread */
				PTHREAD_MUTEX_LOCK(&cg->succ.succ_apps.cg_mutex);
				cg->succ.succ_apps.completed = 1;
				PTHREAD_COND_SIGNAL(&cg->succ.succ_apps.cg_cond);
				PTHREAD_MUTEX_UNLOCK(&cg->succ.succ_apps.cg_mutex);
				break;

			case STARPU_CG_TAG:
				tag = cg->succ.tag;
				tag_successors = &tag->tag_successors;
	
				tag_successors->ndeps_completed++;

#warning FIXME: who locks this?
				if ((tag->state == STARPU_BLOCKED) &&
					(tag_successors->ndeps == tag_successors->ndeps_completed)) {
					/* reset the counter so that we can reuse the completion group */
					tag_successors->ndeps_completed = 0;
					_starpu_tag_set_ready(tag);
				}
				break;

			case STARPU_CG_TASK:
				j = cg->succ.job;

				job_successors = &j->job_successors;

				unsigned ndeps_completed =
					STARPU_ATOMIC_ADD(&job_successors->ndeps_completed, 1);

				if (job_successors->ndeps == ndeps_completed)
				{
					/* Note that this also ensures that tag deps are
					 * fulfilled. This counter is reseted only when the
					 * dependencies are are all fulfilled) */
					_starpu_enforce_deps_and_schedule(j, 1);
				}

				break;

			default:
				STARPU_ABORT();
		}
	}
}

void _starpu_notify_cg_list(struct starpu_cg_list_s *successors)
{
	unsigned nsuccs;
	unsigned succ;

	nsuccs = successors->nsuccs;

	for (succ = 0; succ < nsuccs; succ++)
	{
		struct starpu_cg_s *cg = successors->succ[succ];
		STARPU_ASSERT(cg);

		struct starpu_tag_s *cgtag;

		unsigned cg_type = cg->cg_type;

		if (cg_type == STARPU_CG_TAG)
		{
			cgtag = cg->succ.tag;
			STARPU_ASSERT(cgtag);
			_starpu_spin_lock(&cgtag->lock);
		}

		if (cg_type == STARPU_CG_TASK)
		{
			starpu_job_t j = cg->succ.job;
			PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		}			

		_starpu_notify_cg(cg);

		if (cg_type == STARPU_CG_TASK)
		{
			starpu_job_t j = cg->succ.job;
			
			/* In case this task was immediately terminated, since
			 * _starpu_notify_cg_list already hold the sync_mutex
			 * lock, it is its reponsability to destroy the task if
			 * needed. */
			unsigned must_destroy_task = 0;
			struct starpu_task *task = j->task;

			if ((j->terminated > 0) && task->destroy && task->detach)
				must_destroy_task = 1;

			PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

			if (must_destroy_task)
				starpu_task_destroy(task);
		}			

		if (cg_type == STARPU_CG_APPS) {
			/* Remove the temporary ref to the cg */
			memmove(&successors->succ[succ], &successors->succ[succ+1], (nsuccs-(succ+1)) * sizeof(successors->succ[succ]));
			succ--;
			nsuccs--;
			successors->nsuccs--;
		}

		if (cg_type == STARPU_CG_TAG)
			_starpu_spin_unlock(&cgtag->lock);
	}
}
