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

#include <starpu.h>
#include <common/config.h>
#include <core/dependencies/tags.h>
#include <core/dependencies/htable.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>
#include <core/dependencies/data-concurrency.h>

static htbl_node_t *tag_htbl = NULL;
static pthread_rwlock_t tag_global_rwlock = PTHREAD_RWLOCK_INITIALIZER;

static cg_t *create_cg_apps(unsigned ntags)
{
	cg_t *cg = malloc(sizeof(cg_t));
	STARPU_ASSERT(cg);

	cg->ntags = ntags;
	cg->remaining = ntags;
	cg->cg_type = CG_APPS;

	cg->succ.succ_apps.completed = 0;
	pthread_mutex_init(&cg->succ.succ_apps.cg_mutex, NULL);
	pthread_cond_init(&cg->succ.succ_apps.cg_cond, NULL);

	return cg;
}


static cg_t *create_cg_tag(unsigned ntags, struct tag_s *tag)
{
	cg_t *cg = malloc(sizeof(cg_t));
	STARPU_ASSERT(cg);

	cg->ntags = ntags;
	cg->remaining = ntags;
	cg->cg_type = CG_TAG;

	cg->succ.tag = tag;
	tag->tag_successors.ndeps++;

	return cg;
}

static struct tag_s *_starpu_tag_init(starpu_tag_t id)
{
	struct tag_s *tag;
	tag = malloc(sizeof(struct tag_s));
	STARPU_ASSERT(tag);

	tag->job = NULL;
	tag->is_assigned = 0;
	tag->is_submitted = 0;

	tag->id = id;
	tag->state = INVALID_STATE;

	_starpu_cg_list_init(&tag->tag_successors);

	starpu_spin_init(&tag->lock);

	return tag;
}

void starpu_tag_remove(starpu_tag_t id)
{
	struct tag_s *tag;

	pthread_rwlock_wrlock(&tag_global_rwlock);

	tag = htbl_remove_tag(tag_htbl, id);

	pthread_rwlock_unlock(&tag_global_rwlock);

	if (tag) {
		starpu_spin_lock(&tag->lock);

		unsigned nsuccs = tag->tag_successors.nsuccs;
		unsigned succ;

		for (succ = 0; succ < nsuccs; succ++)
		{
			struct cg_s *cg = tag->tag_successors.succ[succ];

			unsigned ntags = STARPU_ATOMIC_ADD(&cg->ntags, -1);
			unsigned remaining __attribute__ ((unused)) = STARPU_ATOMIC_ADD(&cg->remaining, -1);

			if (!ntags && (cg->cg_type == CG_TAG))
				/* Last tag this cg depends on, cg becomes unreferenced */
				free(cg);
		}

#ifdef DYNAMIC_DEPS_SIZE
		free(tag->tag_successors.succ);
#endif

		starpu_spin_unlock(&tag->lock);
	}

	free(tag);
}

static struct tag_s *gettag_struct(starpu_tag_t id)
{
	pthread_rwlock_wrlock(&tag_global_rwlock);

	/* search if the tag is already declared or not */
	struct tag_s *tag;
	tag = htbl_search_tag(tag_htbl, id);

	if (tag == NULL) {
		/* the tag does not exist yet : create an entry */
		tag = _starpu_tag_init(id);

		void *old;
		old = htbl_insert_tag(&tag_htbl, id, tag);
		/* there was no such tag before */
		STARPU_ASSERT(old == NULL);
	}

	pthread_rwlock_unlock(&tag_global_rwlock);

	return tag;
}

/* lock should be taken */
void _starpu_tag_set_ready(struct tag_s *tag)
{
	/* mark this tag as ready to run */
	tag->state = READY;
	/* declare it to the scheduler ! */
	struct job_s *j = tag->job;

	/* In case the task job is going to be scheduled immediately, and if
	 * the task is "empty", calling push_task would directly try to enforce
	 * the dependencies of the task, and therefore it would try to grab the
	 * lock again, resulting in a deadlock. */
	starpu_spin_unlock(&tag->lock);

	/* enforce data dependencies */
	if (_starpu_submit_job_enforce_data_deps(j))
	{
		starpu_spin_lock(&tag->lock);
		return;
	}

	push_task(j);

	starpu_spin_lock(&tag->lock);
}

/* the lock must be taken ! */
static void _starpu_tag_add_succ(struct tag_s *tag, cg_t *cg)
{
	STARPU_ASSERT(tag);

	_starpu_add_successor_to_cg_list(&tag->tag_successors, cg);

	if (tag->state == DONE) {
		/* the tag was already completed sooner */
		_starpu_notify_cg(cg);
	}
}

static void _starpu_notify_tag_dependencies(struct tag_s *tag)
{
	unsigned nsuccs;
	unsigned succ;

	struct cg_list_s *tag_successors = &tag->tag_successors;

	starpu_spin_lock(&tag->lock);

	tag->state = DONE;

	TRACE_TASK_DONE(tag);

	nsuccs = tag_successors->nsuccs;

	for (succ = 0; succ < nsuccs; succ++)
	{
		struct cg_s *cg = tag_successors->succ[succ];
		struct tag_s *cgtag = cg->succ.tag;

		unsigned cg_type = cg->cg_type;

		if (cg_type == CG_TAG)
			starpu_spin_lock(&cgtag->lock);

		_starpu_notify_cg(cg);
		if (cg_type == CG_APPS) {
			/* Remove the temporary ref to the cg */
			memmove(&tag_successors->succ[succ], &tag_successors->succ[succ+1], (nsuccs-(succ+1)) * sizeof(tag_successors->succ[succ]));
			succ--;
			nsuccs--;
			tag_successors->nsuccs--;
		}

		if (cg_type == CG_TAG)
			starpu_spin_unlock(&cgtag->lock);
	}

	starpu_spin_unlock(&tag->lock);
}

void _starpu_notify_dependencies(struct job_s *j)
{
	STARPU_ASSERT(j);
	STARPU_ASSERT(j->task);
	
	/* in case there are dependencies, wake up the proper tasks */
	if (j->task->use_tag)
		_starpu_notify_tag_dependencies(j->tag);
}

void starpu_tag_notify_from_apps(starpu_tag_t id)
{
	struct tag_s *tag = gettag_struct(id);

	_starpu_notify_tag_dependencies(tag);
}

void _starpu_tag_declare(starpu_tag_t id, struct job_s *job)
{
	TRACE_CODELET_TAG(id, job);
	job->task->use_tag = 1;
	
	struct tag_s *tag= gettag_struct(id);
	tag->job = job;
	tag->is_assigned = 1;
	
	job->tag = tag;

	/* the tag is now associated to a job */
	starpu_spin_lock(&tag->lock);
	tag->state = ASSOCIATED;
	starpu_spin_unlock(&tag->lock);
}

void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array)
{
	unsigned i;

	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);

	starpu_spin_lock(&tag_child->lock);

	cg_t *cg = create_cg_tag(ndeps, tag_child);

	STARPU_ASSERT(ndeps != 0);
	
	for (i = 0; i < ndeps; i++)
	{
		starpu_tag_t dep_id = array[i];
		
		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		TRACE_CODELET_TAG_DEPS(id, dep_id);
		struct tag_s *tag_dep = gettag_struct(dep_id);
		starpu_spin_lock(&tag_dep->lock);
		_starpu_tag_add_succ(tag_dep, cg);
		starpu_spin_unlock(&tag_dep->lock);
	}

	starpu_spin_unlock(&tag_child->lock);
}

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...)
{
	unsigned i;
	
	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);

	starpu_spin_lock(&tag_child->lock);

	cg_t *cg = create_cg_tag(ndeps, tag_child);

	STARPU_ASSERT(ndeps != 0);
	
	va_list pa;
	va_start(pa, ndeps);
	for (i = 0; i < ndeps; i++)
	{
		starpu_tag_t dep_id;
		dep_id = va_arg(pa, starpu_tag_t);
	
		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		TRACE_CODELET_TAG_DEPS(id, dep_id);
		struct tag_s *tag_dep = gettag_struct(dep_id);
		starpu_spin_lock(&tag_dep->lock);
		_starpu_tag_add_succ(tag_dep, cg);
		starpu_spin_unlock(&tag_dep->lock);
	}
	va_end(pa);

	starpu_spin_unlock(&tag_child->lock);
}

/* this function may be called by the application (outside callbacks !) */
int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id)
{
	unsigned i;
	unsigned current;

	struct tag_s *tag_array[ntags];

	/* It is forbidden to block within callbacks or codelets */
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	/* only wait the tags that are not done yet */
	for (i = 0, current = 0; i < ntags; i++)
	{
		struct tag_s *tag = gettag_struct(id[i]);
		
		starpu_spin_lock(&tag->lock);

		if (tag->state == DONE)
		{
			/* that tag is done already */
			starpu_spin_unlock(&tag->lock);
		}
		else
		{
			tag_array[current] = tag;
			current++;
		}
	}

	if (current == 0)
	{
		/* all deps are already fulfilled */
		return 0;
	}
	
	/* there is at least one task that is not finished */
	cg_t *cg = create_cg_apps(current);

	for (i = 0; i < current; i++)
	{
		_starpu_tag_add_succ(tag_array[i], cg);
		starpu_spin_unlock(&tag_array[i]->lock);
	}

	pthread_mutex_lock(&cg->succ.succ_apps.cg_mutex);

	if (!cg->succ.succ_apps.completed)
		pthread_cond_wait(&cg->succ.succ_apps.cg_cond, &cg->succ.succ_apps.cg_mutex);

	pthread_mutex_unlock(&cg->succ.succ_apps.cg_mutex);

	pthread_mutex_destroy(&cg->succ.succ_apps.cg_mutex);
	pthread_cond_destroy(&cg->succ.succ_apps.cg_cond);

	free(cg);

	return 0;
}

int starpu_tag_wait(starpu_tag_t id)
{
	return starpu_tag_wait_array(1, &id);
}
