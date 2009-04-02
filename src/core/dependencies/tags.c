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
static starpu_mutex tag_mutex = {
	.taken = 0
};

cg_t *create_cg(unsigned ntags, struct tag_s *tag)
{
	cg_t *cg;

	cg = malloc(sizeof(cg_t));
	STARPU_ASSERT(cg);
	if (cg) {
		cg->ntags = ntags;
		cg->tag = tag;
	}

	return cg;
}

static struct tag_s *tag_init(starpu_tag_t id)
{
	struct tag_s *tag;
	tag = malloc(sizeof(struct tag_s));
	STARPU_ASSERT(tag);

	tag->id = id;
	tag->state = UNASSIGNED;
	tag->nsuccs = 0;

#ifdef DYNAMIC_DEPS_SIZE
	/* this is a small initial default value ... may be changed */
	tag->succ_list_size = 4;
	tag->succ = realloc(NULL, tag->succ_list_size*sizeof(struct _cg_t *));
#endif

	init_mutex(&tag->lock);

	tag->job = NULL;

	return tag;
}

void starpu_tag_remove(starpu_tag_t id)
{
	struct tag_s *tag;

	take_mutex(&tag_mutex);
	tag = htbl_remove_tag(tag_htbl, id);
	
#ifdef DYNAMIC_DEPS_SIZE
	if (tag)
		free(tag->succ);
#endif

	release_mutex(&tag_mutex);

	free(tag);
}

struct tag_s *gettag_struct(starpu_tag_t id)
{
	take_mutex(&tag_mutex);

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

	release_mutex(&tag_mutex);

	return tag;
}

void notify_cg(cg_t *cg)
{

	STARPU_ASSERT(cg);
	unsigned ntags = STARPU_ATOMIC_ADD(&cg->ntags, -1);
	if (ntags == 0) {
		/* the group is now completed */
		tag_set_ready(cg->tag);
		free(cg);
	}
}

void tag_add_succ(starpu_tag_t id, cg_t *cg)
{
	/* find out the associated structure */
	struct tag_s *tag = gettag_struct(id);
	STARPU_ASSERT(tag);

	take_mutex(&tag->lock);

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

	release_mutex(&tag->lock);
}

void notify_dependencies(struct job_s *j)
{
	struct tag_s *tag;
	unsigned nsuccs;
	unsigned succ;

	STARPU_ASSERT(j);
	
	if (j->task->use_tag) {
		/* in case there are dependencies, wake up the proper tasks */
		tag = j->tag;

		tag->state = DONE;
		TRACE_TASK_DONE(tag->id);

		nsuccs = tag->nsuccs;
		for (succ = 0; succ < nsuccs; succ++)
		{
			notify_cg(tag->succ[succ]);
		}
	}
}

void tag_declare(starpu_tag_t id, struct job_s *job)
{
	TRACE_CODELET_TAG(id, job);
	job->task->use_tag = 1;
	
	struct tag_s *tag= gettag_struct(id);
	tag->job = job;
	
	job->tag = tag;
}

void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array)
{
	unsigned i;

	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);
	cg_t *cg = create_cg(ndeps, tag_child);
	
	tag_child->state = BLOCKED;
	
	STARPU_ASSERT(ndeps != 0);
	
	for (i = 0; i < ndeps; i++)
	{
		starpu_tag_t dep_id = array[i];
		
		/* id depends on dep_id
		 * so cg should be among dep_id's successors*/
		TRACE_CODELET_TAG_DEPS(id, dep_id);
		tag_add_succ(dep_id, cg);
	}
}

void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...)
{
	unsigned i;
	
	/* create the associated completion group */
	struct tag_s *tag_child = gettag_struct(id);
	cg_t *cg = create_cg(ndeps, tag_child);
	
	tag_child->state = BLOCKED;
	
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
		tag_add_succ(dep_id, cg);
	}
	va_end(pa);
}

void tag_set_ready(struct tag_s *tag)
{
	/* mark this tag as ready to run */
	tag->state = READY;
	/* declare it to the scheduler ! */
	struct job_s *j = tag->job;

	/* perhaps the corresponding task was not declared yet */
	if (!j)
		return;

#ifdef NO_DATA_RW_LOCK
	/* enforce data dependencies */
	if (submit_job_enforce_data_deps(j))
		return;
#endif

	push_task(j);
}

/* This function is called when a new task is submitted to StarPU 
 * it returns 1 if the task deps are not fulfilled, 0 otherwise */
unsigned submit_job_enforce_task_deps(job_t j)
{
	struct tag_s *tag = j->tag;
	return (tag->state == BLOCKED);
}
