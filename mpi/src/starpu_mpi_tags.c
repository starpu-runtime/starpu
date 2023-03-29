/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

#include <mpi.h>

#include <common/utils.h>
#include <starpu_mpi_private.h>

/**
 * @brief Structure to store tags ranges
 *
 * List structure to manage the set of available tags.
 */
struct starpu_tags_range
{
	int64_t min;  /**< Minimal value in the range     */
	int64_t max;  /**< Maximal value in the range     */
	struct starpu_tags_range *next; /**< Pointer to the following range */
};

static struct starpu_tags_range *cst_first = NULL;

/**
 * @brief StarPU tag upper bound
 */
static int64_t _starpu_tags_ub = 0;

void _starpu_mpi_tags_init(void)
{
	if (!_starpu_tags_ub)
	{
		int ok = 0;
		void *tag_ub_p = NULL;

		starpu_mpi_comm_get_attr(MPI_COMM_WORLD, STARPU_MPI_TAG_UB, &tag_ub_p, &ok);
		_starpu_tags_ub = (uint64_t)((intptr_t)tag_ub_p);

		STARPU_ASSERT_MSG(ok, "Failed to get the STARPU_MPI_TAG_UB attribute\n");
	}
}

int64_t starpu_mpi_tags_allocate(int64_t nbtags)
{
	struct starpu_tags_range *new;
	struct starpu_tags_range *prev = NULL;
	struct starpu_tags_range *current = cst_first;
	int64_t min = 0;
	int64_t max = (current == NULL) ? _starpu_tags_ub : current->min;

	if (nbtags == 0)
	{
		return -1;
	}
	STARPU_ASSERT(_starpu_tags_ub != 0); /* StarPU tag must be initialized */

	while (((max - min) < nbtags) && (current != NULL))
	{
		min = current->max;
		prev = current;
		current = current->next;
		max = (current == NULL) ? _starpu_tags_ub : current->min;
	}

	if ((max - min) < nbtags)
	{
		_STARPU_ERROR("No space left in tags.\n" );
		return -1;
	}

	_STARPU_MALLOC(new, sizeof(struct starpu_tags_range));
	new->min = min;
	new->max = min + nbtags;
	new->next = current;
	if (prev == NULL)
	{
		cst_first = new;
	}
	else
	{
		STARPU_ASSERT(prev->next == current);
		prev->next = new;
	}

	_STARPU_MPI_DEBUG(0, "Allocates tag range %ld - %ld\n", min, min + nbtags);

	STARPU_ASSERT(cst_first != NULL);
	return new->min;
}

void starpu_mpi_tags_free(int64_t min)
{
	struct starpu_tags_range *prev = NULL;
	struct starpu_tags_range *current = cst_first;

	STARPU_ASSERT(cst_first != NULL); /* At least one range must be registered */

	while ((current != NULL) && (current->min < min))
	{
		prev = current;
		current = current->next;
	}

	if (current == NULL)
	{
		_STARPU_ERROR("Failed to release the tag range starting by %ld", min);
		return;
	}

	STARPU_ASSERT(current != NULL);
	STARPU_ASSERT(current->min == min);

	if (prev)
	{
		prev->next = current->next;
	}
	else
	{
		STARPU_ASSERT(current == cst_first);
		cst_first = current->next;
	}

	_STARPU_MPI_DEBUG(0, "Free tag range %ld - %ld\n", current->min, current->max);

	free(current);

	return;
}
