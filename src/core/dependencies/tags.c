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

#include <stdarg.h>
#include <stdlib.h>
#include <core/dependencies/tags.h>
#include <core/dependencies/htable.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>
#include <core/dependencies/data-concurrency.h>
#include <starpu.h>

static htbl_node_t *tag_htbl = NULL;
static pthread_spinlock_t tag_mutex;

void initialize_tag_mutex(void)
{
	pthread_spin_init(&tag_mutex, 0);
}

static cg_t *create_cg(unsigned ntags, struct tag_s *tag, unsigned is_apps_cg)
{
	cg_t *cg;

	cg = malloc(sizeof(cg_t));
	STARPU_ASSERT(cg);
	if (cg) {
		cg->ntags = ntags;
		cg->completed = 0;
		cg->used_by_apps = is_apps_cg;

		if (is_apps_cg)
		{
			pthread_mutex_init(&cg->cg_mutex, NULL);
			pthread_cond_init(&cg->cg_cond, NULL);
		}
		else
		{
			cg->tag = tag;
			tag->ndeps++;
		}
	}

	return cg;
}

static struct tag_s *tag_init(starpu_tag_t id)
{
	struct tag_s *tag;
	tag = malloc(sizeof(struct tag_s));
	STARPU_ASSERT(tag);

	tag->job = NULL;
	tag->is_assigned = 0;
	tag->is_submitted = 0;

	tag->id = id;
	tag->state = INVALID_STATE;
	tag->nsuccs = 0;
	tag->ndeps = 0;
	tag->ndeps_completed = 0;

#ifdef DYNAMIC_DEPS_SIZE
	/* this is a small initial default value ... may be changed */
	tag->succ_list_size = 4;
	tag->succ = realloc(NULL, tag->succ_list_size*sizeof(struct _cg_t *));
#endif

	pthread_spin_init(&tag->lock, 0);

	return tag;
}

void starpu_tag_remove(starpu_tag_t id)
{
	struct tag_s *tag;

	pthread_spin_lock(&tag_mutex);

	tag = htbl_remove_tag(tag_htbl, id);

	pthread_spin_unlock(&tag_mutex);

	pthread_spin_lock(&tag->lock);
	
#ifdef DYNAMIC_DEPS_SIZE
	if (tag)
		free(tag->succ);
#endif

	pthread_spin_unlock(&tag->lock);

	free(tag);
}

static struct tag_s *gettag_struct(starpu_tag_t id)
{
	pthread_spin_lock(&tag_mutex);

	/* search if the tag is already declared or not */
	struct tag_s *tag;
	tag = htbl_search_tag(tag_htbl, id);

	if (tag == NULL) {
		/* the tag does not exist yet : create an entry */
		tag = tag_init(id);

		void *old;
		old = htbl_insert_tag(&tag_htbl, id, tag);
		/* there was no such tag before */
		STARPU_ASSERT(old == NULL);
	}

	pthread_spin_unlock(&tag_mutex);

	return tag;
}

/* lock should be taken */
static void tag_set_ready(struct tag_s *tag)
{
	/* mark this tag as ready to run */
	tag->state = READY;
	/* declare it to the scheduler ! */
	struct job_s *j = tag->job;

#ifdef NO_DATA_RW_LOCK
	/* enforce data dependencies */
	if (submit_job_enforce_data_deps(j))
		return;
#endif

	push_task(j);
}

static void notify_cg(cg_t *cg)
{
	STARPU_ASSERT(cg);
	unsigned ntags = STARPU_ATOMIC_ADD(&cg->ntags, -1);
	if (ntags == 0) {
		/* the group is now completed */
		if (cg->used_by_apps)
		{
			/* this is a cg for an application waiting on a set of
 			 * tags, wake the thread */
			pthread_mutex_lock(&cg->cg_mutex);
			cg->completed = 1;
			pthread_cond_signal(&cg->cg_cond);
			pthread_mutex_unlock(&cg->cg_mutex);
		}
		else
		{
			struct tag_s *tag = cg->tag;
			tag->ndeps_completed++;

			if ((tag->state == BLOCKED) 
				&& (tag->ndeps == tag->ndeps_completed))
				tag_set_ready(cg->tag);

			free(cg);
		}
	}
}

/* the lock must be taken ! */
static void tag_add_succ(struct tag_s *tag, cg_t *cg)
{
	STARPU_ASSERT(tag);

	if (tag->state == DONE) {
		/* the tag was already completed sooner */
		notify_cg(cg);
	}
	else {
		/* where should that cg should be put in the array ? */
		unsigned index = STARPU_ATOMIC_ADD(&tag->nsuccs, 1) - 1;

#ifdef DYNAMIC_DEPS_SIZE
		if (index >= tag->succ_list_size)
		{
			/* the successor list is too small */
			tag->succ_list_size *= 2;

			/* NB: this is thread safe as the tag->lock is taken */
			tag->succ = realloc(tag->succ, 
				tag->succ_list_size*sizeof(struct _cg_t *));
		}
#else
		STARPU_ASSERT(index < NMAXDEPS);
#endif
		tag->succ[index] = cg;
	}

	pthread_spin_unlock(&tag->lock);
}

static void notify_tag_dependencies(struct tag_s *tag)
{
	unsigned nsuccs;
	unsigned succ;

	pthread_spin_lock(&tag->lock);

	tag->state = DONE;
	TRACE_TASK_DONE(tag->id);

	nsuccs = tag->nsuccs;

	for (succ = 0; succ < nsuccs; succ++)
	{
		struct _cg_t *cg = tag->succ[succ];
		unsigned used_by_apps = cg->used_by_apps;
		struct tag_s *cgtag = cg->tag;

		if (!used_by_apps)
			pthread_spin_lock(&cgtag->lock);

		notify_cg(cg);

		if (!used_by_apps)
			pthread_spin_unlock(&cgtag->lock);
	}

	pthread_spin_unlock(&tag->lock);
}

void notify_dependencies(struct job_s *j)
{
	STARPU_ASSERT(j);
	STARPU_ASSERT(j->task);
	
	/* in case there are dependencies, wake up the proper tasks */
	if (j->task->use_tag)
		notify_tag_dependencies(j->tag);
}

void starpu_tag_notify_from_apps(starpu_tag_t id)
{
	struct tag_s *tag = gettag_struct(id);

	notify_tag_dependencies(tag);
}

void tag_declare(starpu_tag_t id, struct job_s *job)
{
	TRACE_CODELET_TAG(id, job);
	job->task->use_tag = 1;
	
	struct tag_s *tag= gettag_struct(id);
	tag->job = job;
	tag->is_assigned = 1;
	
	job->tag = tag;

	/* the tag is now associated to a job */
	tag->state = ASSOCIATED;
}

void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array)
{
	unsigned i;

	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);

	pthread_spin_lock(&tag_child->lock);

	cg_t *cg = create_cg(ndeps, tag_child, 0);

	STARPU_ASSERT(ndeps != 0);
	
	for (i = 0; i < ndeps; i++)
	{
		starpu_tag_t dep_id = array[i];
		
		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		TRACE_CODELET_TAG_DEPS(id, dep_id);
		struct tag_s *tag_dep = gettag_struct(dep_id);
		pthread_spin_lock(&tag_dep->lock);
		tag_add_succ(tag_dep, cg);
		pthread_spin_unlock(&tag_dep->lock);
	}

	pthread_spin_unlock(&tag_child->lock);
}

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...)
{
	unsigned i;
	
	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);

	pthread_spin_lock(&tag_child->lock);

	cg_t *cg = create_cg(ndeps, tag_child, 0);

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
		pthread_spin_lock(&tag_dep->lock);
		tag_add_succ(tag_dep, cg);
		pthread_spin_unlock(&tag_dep->lock);
	}
	va_end(pa);

	pthread_spin_unlock(&tag_child->lock);
}

/* this function may be called by the application (outside callbacks !) */
void starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id)
{
	unsigned i;
	unsigned current;

	struct tag_s *tag_array[ntags];

	/* only wait the tags that are not done yet */
	for (i = 0, current = 0; i < ntags; i++)
	{
		struct tag_s *tag = gettag_struct(id[i]);
		
		pthread_spin_lock(&tag->lock);

		if (tag->state == DONE)
		{
			/* that tag is done already */
			pthread_spin_unlock(&tag->lock);
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
		return;
	}
	
	/* there is at least one task that is not finished */
	cg_t *cg = create_cg(current, NULL, 1);

	for (i = 0; i < current; i++)
	{
		tag_add_succ(tag_array[i], cg);
		pthread_spin_unlock(&tag_array[i]->lock);
	}

	pthread_mutex_lock(&cg->cg_mutex);

	if (!cg->completed)
		pthread_cond_wait(&cg->cg_cond, &cg->cg_mutex);

	pthread_mutex_unlock(&cg->cg_mutex);

	pthread_mutex_destroy(&cg->cg_mutex);
	pthread_cond_destroy(&cg->cg_cond);

	free(cg);
}

void starpu_tag_wait(starpu_tag_t id)
{
	starpu_tag_wait_array(1, &id);
}
