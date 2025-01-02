/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <datawizard/filters.h>
#include <datawizard/sort_data_handles.h>

/* To avoid deadlocks in case we have multiple tasks accessing the same piece
 * of data  (eg. task T1 needs A and B, and T2 needs B and A), we need to lock
 * them in order, so that we need a total order over data. We must also not
 * lock a child before its parent. */

static void find_data_path(struct _starpu_data_state *data, unsigned path[])
{
	unsigned depth = data->depth;
	struct _starpu_data_state *current = data;

	/* Compute the path from the root to the data */
	unsigned level; /* level is the distance between the node and the current node */
	for (level = 0; level < depth; level++)
	{
		path[depth - level - 1] = current->sibling_index;
		current = current->father_handle;
	}
}

static int _compar_data_paths(const unsigned pathA[], unsigned depthA,
				const unsigned pathB[], unsigned depthB)
{
	unsigned level;
	unsigned depth = STARPU_MIN(depthA, depthB);

	for (level = 0; level < depth; level++)
	{
		if (pathA[level] != pathB[level])
			return (pathA[level] < pathB[level])?-1:1;
	}

	/* If this is the same path */
	if (depthA == depthB)
		return 0;

	/* A is a subdata of B or B is a subdata of A, so the smallest one is
	 * the father of the other (we take this convention). */
	return (depthA < depthB)?-1:1;
}

/* A comparison function between two handles makes it possible to use qsort to
 * sort a list of handles */
static int _starpu_compar_handles(const struct _starpu_data_descr *descrA,
				  const struct _starpu_data_descr *descrB)
{
	starpu_data_handle_t dataA = descrA->handle;
	starpu_data_handle_t dataB = descrB->handle;

	/* Perhaps we have the same piece of data */
	if (dataA->root_handle == dataB->root_handle)
	{
		int Awrites = descrA->mode & STARPU_W;
		int Bwrites = descrB->mode & STARPU_W;
		int Areads = descrA->mode & STARPU_R;
		int Breads = descrB->mode & STARPU_R;

		/* Process write requests first, this is needed for proper
		 * locking, see _submit_job_access_data,
		 * _starpu_fetch_task_input, and _starpu_push_task_output  */

		if (Awrites && !Bwrites)
			/* Only A writes, take it first */
			return -1;
		if (!Awrites &&  Bwrites)
			/* Only B writes, take it first */
			return 1;
		/* Both A and B write */

		if (Areads && !Breads)
			/* Only A reads, take it first */
			return -1;
		if (!Areads &&  Breads)
			/* Only B reads, take it first */
			return 1;
		/* Both A and B read and write */

		/* Things get more complicated: we need to find the location of dataA
		 * and dataB within the tree. */
		unsigned dataA_path[dataA->depth];
		unsigned dataB_path[dataB->depth];

		find_data_path(dataA, dataA_path);
		find_data_path(dataB, dataB_path);

		return _compar_data_paths(dataA_path, dataA->depth, dataB_path, dataB->depth);
	}

	/* Put arbitered accesses after non-arbitered */
	if (dataA->arbiter && !(dataB->arbiter))
		return 1;
	if (dataB->arbiter && !(dataA->arbiter))
		return -1;
	if (dataA->arbiter != dataB->arbiter)
		/* Both are arbitered, sort by arbiter pointer order */
		return (dataA->arbiter < dataB->arbiter)?-1:1;
	/* If both are arbitered by the same arbiter (or they are both not
	 * arbitered), we'll sort them by handle */
	return (dataA->root_handle < dataB->root_handle)?-1:1;
}

int _starpu_handles_same_root(starpu_data_handle_t dataA, starpu_data_handle_t dataB)
{
	return dataA->root_handle == dataB->root_handle;
}

static int _starpu_compar_buffer_descr(const void *_descrA, const void *_descrB)
{
	const struct _starpu_data_descr *descrA = (const struct _starpu_data_descr *) _descrA;
	const struct _starpu_data_descr *descrB = (const struct _starpu_data_descr *) _descrB;

	return _starpu_compar_handles(descrA, descrB);
}

/* The descr array will be overwritten, so this must be a copy ! */
void _starpu_sort_task_handles(struct _starpu_data_descr descr[], unsigned nbuffers)
{
	qsort(descr, nbuffers, sizeof(descr[0]), _starpu_compar_buffer_descr);
}
