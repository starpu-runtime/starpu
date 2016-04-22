/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016  Université de Bordeaux
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

/*
 * This stores the task graph structure, to used by the schedulers which need
 * it.  We do not always enable it since it is costly.
 */

#include <starpu.h>
#include <core/jobs.h>
#include <common/graph.h>

/* Protects the whole task graph */
static starpu_pthread_rwlock_t graph_lock;

/* Whether we should enable recording the task graph */
int _starpu_graph_record;

/* This list contains all jobs without incoming dependency */
struct _starpu_job_list top;
/* This list contains all jobs without outgoing dependency */
struct _starpu_job_list bottom;
/* This list contains all jobs */
struct _starpu_job_list all;

void _starpu_graph_init(void)
{
	STARPU_PTHREAD_RWLOCK_INIT(&graph_lock, NULL);
	_starpu_job_list_init(&top);
	_starpu_job_list_init(&bottom);
	_starpu_job_list_init(&all);
}

static void __starpu_graph_foreach(void (*func)(void *data, struct _starpu_job *job), void *data)
{
	struct _starpu_job *job;

	for (job = _starpu_job_list_begin(&all, all);
	     job != _starpu_job_list_end(&all, all);
	     job = _starpu_job_list_next(&all, job, all))
		func(data, job);
}

/* Add a job to the graph */
void _starpu_graph_add_job(struct _starpu_job *job)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);

	/* It does not have any dependency yet, add to all lists */
	_starpu_job_list_push_back(&top, job, top);
	_starpu_job_list_push_back(&bottom, job, bottom);
	_starpu_job_list_push_back(&all, job, all);

	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

/* Add a job to an array of jobs */
static unsigned add_job(struct _starpu_job *job, struct _starpu_job ***jobs, unsigned *n_jobs, unsigned *alloc_jobs, unsigned **slot)
{
	unsigned ret;
	if (*n_jobs == *alloc_jobs)
	{
		if (*alloc_jobs)
			*alloc_jobs *= 2;
		else
			*alloc_jobs = 4;
		*jobs = realloc(*jobs, *alloc_jobs * sizeof(**jobs));
		if (slot)
			*slot = realloc(*slot, *alloc_jobs * sizeof(**slot));
	}
	ret = (*n_jobs)++;
	(*jobs)[ret] = job;
	return ret;
}

/* Add a dependency between jobs */
void _starpu_graph_add_job_dep(struct _starpu_job *job, struct _starpu_job *prev_job)
{
	unsigned rank_incoming, rank_outgoing;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);

	if (_starpu_job_list_queued(prev_job, bottom))
		/* Previous job is not at bottom any more */
		_starpu_job_list_erase(bottom, prev_job, bottom);

	if (_starpu_job_list_queued(job, top))
		/* Next job is not at top any more */
		_starpu_job_list_erase(top, job, top);

	rank_incoming = add_job(prev_job, &job->incoming, &job->n_incoming, &job->alloc_incoming, NULL);
	rank_outgoing = add_job(job, &prev_job->outgoing, &prev_job->n_outgoing, &prev_job->alloc_outgoing, &prev_job->outgoing_slot);
	prev_job->outgoing_slot[rank_outgoing] = rank_incoming;

	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

/* Drop a job, and thus its dependencies */
void _starpu_graph_drop_job(struct _starpu_job *job)
{
	unsigned i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);

	if (_starpu_job_list_queued(job, bottom))
		_starpu_job_list_erase(bottom, job, bottom);
	if (_starpu_job_list_queued(job, top))
		_starpu_job_list_erase(top, job, top);
	if (_starpu_job_list_queued(job, all))
		_starpu_job_list_erase(all, job, all);

	/* Drop ourself from the incoming part of the outgoing jobs */
	for (i = 0; i < job->n_outgoing; i++)
	{
		struct _starpu_job *next = job->outgoing[i];
		next->incoming[job->outgoing_slot[i]] = NULL;
	}
	job->n_outgoing = 0;
	free(job->outgoing);
	job->outgoing = NULL;
	free(job->outgoing_slot);
	job->outgoing_slot = NULL;
	job->alloc_outgoing = 0;
	job->n_incoming = 0;
	free(job->incoming);
	job->incoming = NULL;
	job->alloc_incoming = 0;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

static void _starpu_graph_set_n(void *data, struct _starpu_job *job)
{
	int value = (intptr_t) data;
	job->graph_n = value;
}

/* Call func for each vertex of the task graph, from bottom to top, in topological order */
static void _starpu_graph_compute_bottom_up(void (*func)(struct _starpu_job *next_job, struct _starpu_job *prev_job, void *data), void *data)
{
	struct _starpu_job *job, *job2;
	struct _starpu_job **current_set = NULL, **next_set = NULL, **swap_set;
	unsigned current_n, next_n, i, j;
	unsigned current_alloc = 0, next_alloc = 0, swap_alloc;

	/* Classical flow algorithm: start from bottom, and propagate depths to top */

	/* Set number of processed outgoing edges to 0 for each node */
	__starpu_graph_foreach(_starpu_graph_set_n, (void*) 0);

	/* Start with the bottom of the graph */
	current_n = 0;
	for (job = _starpu_job_list_begin(&bottom, bottom);
	     job != _starpu_job_list_end(&bottom, bottom);
	     job = _starpu_job_list_next(&bottom, job, bottom))
		add_job(job, &current_set, &current_n, &current_alloc, NULL);

	/* Now propagate to top as long as we have current nodes */
	while (current_n)
	{
		/* Next set is initially empty */
		next_n = 0;

		/* For each node in the current set */
		for (i = 0; i < current_n; i++)
		{
			job = current_set[i];
			/* For each parent of this job */
			for (j = 0; j < job->n_incoming; j++)
			{
				job2 = job->incoming[j];
				if (!job2)
					continue;
				job2->graph_n++;
				func(job, job2, data);

				if ((unsigned) job2->graph_n == job2->n_outgoing)
					/* All outgoing edges were processed, can now add to next set */
					add_job(job2, &next_set, &next_n, &next_alloc, NULL);
			}
		}

		/* Swap next set with current set */
		swap_set = next_set;
		swap_alloc = next_alloc;
		next_set = current_set;
		next_alloc = current_alloc;
		current_set = swap_set;
		current_alloc = swap_alloc;
		current_n = next_n;
	}
}

static void compute_depth(struct _starpu_job *next_job, struct _starpu_job *prev_job, void *data STARPU_ATTRIBUTE_UNUSED)
{
	if (prev_job->depth < next_job->depth + 1)
		prev_job->depth = next_job->depth + 1;
}

void _starpu_graph_compute_depths(void)
{
	struct _starpu_job *job;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);

	/* The bottom of the graph has depth 0 */
	for (job = _starpu_job_list_begin(&bottom, bottom);
	     job != _starpu_job_list_end(&bottom, bottom);
	     job = _starpu_job_list_next(&bottom, job, bottom))
		job->depth = 0;

	_starpu_graph_compute_bottom_up(compute_depth, NULL);

	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

void _starpu_graph_compute_descendants(void)
{
	struct _starpu_job *job, *job2, *job3;
	struct _starpu_job **current_set = NULL, **next_set = NULL, **swap_set;
	unsigned current_n, next_n, i, j;
	unsigned current_alloc = 0, next_alloc = 0, swap_alloc;
	unsigned descendants;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);

	/* Yes, this is O(|V|.(|V|+|E|)) :( */

	/* We could get O(|V|.|E|) by doing a topological sort first.
	 *
	 * |E| is usually O(|V|), though (bounded number of data dependencies,
	 * and we use synchronization tasks) */

	for (job = _starpu_job_list_begin(&all, all);
	     job != _starpu_job_list_end(&all, all);
	     job = _starpu_job_list_next(&all, job, all))
	{
		/* Mark all nodes as unseen */
		for (job2 = _starpu_job_list_begin(&all, all);
		     job2 != _starpu_job_list_end(&all, all);
		     job2 = _starpu_job_list_next(&all, job2, all))
			job2->graph_n = 0;

		/* Start with the node we want to compute the number of descendants of */
		current_n = 0;
		add_job(job, &current_set, &current_n, &current_alloc, NULL);
		job->graph_n = 1;

		descendants = 0;
		/* While we have descendants, count their descendants */
		while (current_n) {
			/* Next set is initially empty */
			next_n = 0;

			/* For each node in the current set */
			for (i = 0; i < current_n; i++)
			{
				job2 = current_set[i];
				/* For each child of this job2 */
				for (j = 0; j < job2->n_outgoing; j++)
				{
					job3 = job2->outgoing[j];
					if (!job3)
						continue;
					if (job3->graph_n)
						/* Already seen */
						continue;
					/* Add this node */
					job3->graph_n = 1;
					descendants++;
					add_job(job3, &next_set, &next_n, &next_alloc, NULL);
				}
			}
			/* Swap next set with current set */
			swap_set = next_set;
			swap_alloc = next_alloc;
			next_set = current_set;
			next_alloc = current_alloc;
			current_set = swap_set;
			current_alloc = swap_alloc;
			current_n = next_n;
		}
		job->descendants = descendants;
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

void _starpu_graph_foreach(void (*func)(void *data, struct _starpu_job *job), void *data)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);
	__starpu_graph_foreach(func, data);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}
