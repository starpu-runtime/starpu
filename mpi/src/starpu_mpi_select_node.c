/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Centre National de la Recherche Scientifique
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

#include <stdarg.h>
#include <mpi.h>

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_task_insert.h>
#include <datawizard/coherency.h>

static char *_default_policy = "node_with_most_R_data";

char *starpu_mpi_node_selection_get_default_policy()
{
	return _default_policy;
}

int starpu_mpi_node_selection_set_default_policy(char *policy)
{
	strcpy(_default_policy, policy);
	return 0;
}

int _starpu_mpi_select_node_with_most_R_data(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
{
	size_t *size_on_nodes;
	size_t max_size;
	int i;
	int xrank;

	(void)me;
	size_on_nodes = (size_t *)calloc(1, nb_nodes * sizeof(size_t));

	for(i= 0 ; i<nb_data ; i++)
	{
		starpu_data_handle_t data = descr[i].handle;
		enum starpu_data_access_mode mode = descr[i].mode;
		if (mode & STARPU_R)
		{
			int rank = starpu_data_get_rank(data);
			size_on_nodes[rank] += data->ops->get_size(data);
		}
	}

	max_size = 0;
	for(i=0 ; i<nb_nodes ; i++)
	{
		if (size_on_nodes[i] > max_size)
		{
			max_size = size_on_nodes[i];
			xrank = i;
		}
	}

	return xrank;
}

int _starpu_mpi_select_node(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data, char *policy)
{
	char *current_policy = policy ? policy : _default_policy;
	if (current_policy == NULL)
		STARPU_ABORT_MSG("Node selection policy MUST be defined\n");
	if (strcmp(current_policy, "node_with_most_R_data") == 0)
		return _starpu_mpi_select_node_with_most_R_data(me, nb_nodes, descr, nb_data);
	else
		STARPU_ABORT_MSG("Node selection policy <%s> unknown\n", current_policy);
}
