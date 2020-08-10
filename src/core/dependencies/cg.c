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

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/jobs.h>
#include <core/task.h>
#include <core/dependencies/cg.h>
#include <core/dependencies/tags.h>

void _starpu_cg_list_init(struct _starpu_cg_list *list)
{
	_starpu_spin_init(&list->lock);
	list->ndeps = 0;
	list->ndeps_completed = 0;

	list->terminated = 0;

	list->nsuccs = 0;
#ifdef STARPU_DYNAMIC_DEPS_SIZE
	/* this is a small initial default value ... may be changed */
	list->succ_list_size = 0;
	list->succ = NULL;
#endif
}

void _starpu_cg_list_deinit(struct _starpu_cg_list *list)
{
	unsigned id;
	for (id = 0; id < list->nsuccs; id++)
	{
		struct _starpu_cg *cg = list->succ[id];

		/* We remove the reference on the completion group, and free it
		 * if there is no more reference. */
		unsigned ntags = STARPU_ATOMIC_ADD(&cg->ntags, -1);
		if (ntags == 0)
			free(list->succ[id]);
	}

#ifdef STARPU_DYNAMIC_DEPS_SIZE
	free(list->succ);
#endif
	_starpu_spin_destroy(&list->lock);
}

/* Returns whether the completion was already terminated, and caller should
 * thus immediately proceed. */
int _starpu_add_successor_to_cg_list(struct _starpu_cg_list *successors, struct _starpu_cg *cg)
{
	int ret;
	STARPU_ASSERT(cg);

	_starpu_spin_lock(&successors->lock);
	ret = successors->terminated;

	/* where should that cg should be put in the array ? */
	unsigned index = successors->nsuccs++;

#ifdef STARPU_DYNAMIC_DEPS_SIZE
	if (index >= successors->succ_list_size)
	{
		/* the successor list is too small */
		if (successors->succ_list_size > 0)
			successors->succ_list_size *= 2;
		else
			successors->succ_list_size = 4;

		_STARPU_REALLOC(successors->succ, successors->succ_list_size*sizeof(struct _starpu_cg *));
	}
#else
	STARPU_ASSERT(index < STARPU_NMAXDEPS);
#endif
	successors->succ[index] = cg;
	_starpu_spin_unlock(&successors->lock);

	return ret;
}

int _starpu_list_task_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, struct starpu_task *task_array[])
{
	unsigned i;
	unsigned n = 0;
	_starpu_spin_lock(&successors->lock);
	for (i = 0; i < successors->nsuccs; i++)
	{
		struct _starpu_cg *cg = successors->succ[i];
		if (cg->cg_type != STARPU_CG_TASK)
			continue;
		if (n < ndeps)
		{
			task_array[n] = cg->succ.job->task;
			n++;
		}
	}
	_starpu_spin_unlock(&successors->lock);
	return n;
}

int _starpu_list_task_scheduled_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, struct starpu_task *task_array[])
{
	unsigned i;
	unsigned n = 0;
	_starpu_spin_lock(&successors->lock);
	for (i = 0; i < successors->nsuccs; i++)
	{
		struct _starpu_cg *cg = successors->succ[i];
		if (cg->cg_type != STARPU_CG_TASK)
			continue;
		if (n < ndeps)
		{
			struct starpu_task *task = cg->succ.job->task;
			if (task->cl == NULL || task->cl->where == STARPU_NOWHERE
					|| task->execute_on_a_specific_worker)
				/* will not be scheduled */
				continue;
			task_array[n] = task;
			n++;
		}
	}
	_starpu_spin_unlock(&successors->lock);
	return n;
}

int _starpu_list_tag_successors_in_cg_list(struct _starpu_cg_list *successors, unsigned ndeps, starpu_tag_t tag_array[])
{
	unsigned i;
	unsigned n = 0;
	_starpu_spin_lock(&successors->lock);
	for (i = 0; i < successors->nsuccs; i++)
	{
		struct _starpu_cg *cg = successors->succ[i];
		if (cg->cg_type != STARPU_CG_TAG)
			continue;
		if (n < ndeps)
		{
			tag_array[n] = cg->succ.tag->id;
			n++;
		}
	}
	_starpu_spin_unlock(&successors->lock);
	return n;
}

void _starpu_notify_cg(struct _starpu_cg *cg)
{
	STARPU_ASSERT(cg);
	unsigned remaining = STARPU_ATOMIC_ADD(&cg->remaining, -1);
	ANNOTATE_HAPPENS_BEFORE(&cg->remaining);

	if (remaining == 0)
	{
		ANNOTATE_HAPPENS_AFTER(&cg->remaining);
		cg->remaining = cg->ntags;

		/* the group is now completed */
		switch (cg->cg_type)
		{
			case STARPU_CG_APPS:
			{
				/* this is a cg for an application waiting on a set of
				 * tags, wake the thread */
				STARPU_PTHREAD_MUTEX_LOCK(&cg->succ.succ_apps.cg_mutex);
				cg->succ.succ_apps.completed = 1;
				STARPU_PTHREAD_COND_SIGNAL(&cg->succ.succ_apps.cg_cond);
				STARPU_PTHREAD_MUTEX_UNLOCK(&cg->succ.succ_apps.cg_mutex);
				break;
			}

			case STARPU_CG_TAG:
			{
				struct _starpu_cg_list *tag_successors;
				struct _starpu_tag *tag;

				tag = cg->succ.tag;
				_starpu_spin_lock(&tag->lock);
				tag_successors = &tag->tag_successors;

				tag_successors->ndeps_completed++;

				/* Note: the tag is already locked by the
				 * caller. */
				if ((tag->state == STARPU_BLOCKED) &&
					(tag_successors->ndeps == tag_successors->ndeps_completed))
				{
					/* reset the counter so that we can reuse the completion group */
					tag_successors->ndeps_completed = 0;
					_starpu_tag_set_ready(tag);
				}
				_starpu_spin_unlock(&tag->lock);
				break;
			}

 		        case STARPU_CG_TASK:
			{
				struct _starpu_cg_list *job_successors;
				struct _starpu_job *j;

				j = cg->succ.job;

				STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);

				job_successors = &j->job_successors;

				unsigned ndeps_completed =
					STARPU_ATOMIC_ADD(&job_successors->ndeps_completed, 1);

				STARPU_ASSERT(job_successors->ndeps >= ndeps_completed);

				/* Need to atomically test submitted and check
				 * dependencies, since this is concurrent with
				 * _starpu_submit_job */
				if (j->submitted && job_successors->ndeps == ndeps_completed &&
					j->task->status == STARPU_TASK_BLOCKED_ON_TASK)
				{
					/* That task has already passed tag checks,
					 * do not do them again since the tag has been cleared! */
					_starpu_enforce_deps_starting_from_task(j);
				}
				else
					STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);


				break;
			}

			default:
				STARPU_ABORT();
		}
	}
}

/* Caller just has to promise that the list will not disappear.
 * _starpu_notify_cg_list protects the list itself.
 * No job lock should be held, since we might want to immediately call the callback of an empty task.
 */
void _starpu_notify_cg_list(struct _starpu_cg_list *successors)
{
	unsigned succ;

	_starpu_spin_lock(&successors->lock);
	/* Note: some thread might be concurrently adding other items */
	for (succ = 0; succ < successors->nsuccs; succ++)
	{
		struct _starpu_cg *cg = successors->succ[succ];
		STARPU_ASSERT(cg);
		unsigned cg_type = cg->cg_type;

		if (cg_type == STARPU_CG_APPS)
		{
			/* Remove the temporary ref to the cg */
			memmove(&successors->succ[succ], &successors->succ[succ+1], (successors->nsuccs-(succ+1)) * sizeof(successors->succ[succ]));
			succ--;
			successors->nsuccs--;
		}
		_starpu_spin_unlock(&successors->lock);
		_starpu_notify_cg(cg);
		_starpu_spin_lock(&successors->lock);
	}
	successors->terminated = 1;
	_starpu_spin_unlock(&successors->lock);
}
