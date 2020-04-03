/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi.h>
#include <starpu_data.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_select_node.h>
#include <datawizard/coherency.h>

static int _current_policy = STARPU_MPI_NODE_SELECTION_MOST_R_DATA;
static int _last_predefined_policy = STARPU_MPI_NODE_SELECTION_MOST_R_DATA;
static starpu_mpi_select_node_policy_func_t _policies[_STARPU_MPI_NODE_SELECTION_MAX_POLICY];

int _starpu_mpi_select_node_with_most_data(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data);

void _starpu_mpi_select_node_init()
{
	int i;

	_policies[STARPU_MPI_NODE_SELECTION_MOST_R_DATA] = _starpu_mpi_select_node_with_most_data;
	for(i=_last_predefined_policy+1 ; i<_STARPU_MPI_NODE_SELECTION_MAX_POLICY ; i++)
		_policies[i] = NULL;
}

int starpu_mpi_node_selection_get_current_policy()
{
	return _current_policy;
}

int starpu_mpi_node_selection_set_current_policy(int policy)
{
	STARPU_ASSERT_MSG(_policies[policy] != NULL, "Policy %d invalid.\n", policy);
	_current_policy = policy;
	return 0;
}

int starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_policy_func_t policy_func)
{
	int i=_last_predefined_policy+1;
	// Look for a unregistered policy
	while(i<_STARPU_MPI_NODE_SELECTION_MAX_POLICY)
	{
		if (_policies[i] == NULL)
			break;
		i++;
	}
	STARPU_ASSERT_MSG(i<_STARPU_MPI_NODE_SELECTION_MAX_POLICY, "No unused policy available. Unregister existing policies before registering a new one.");
	_policies[i] = policy_func;
	return i;
}

int starpu_mpi_node_selection_unregister_policy(int policy)
{
	STARPU_ASSERT_MSG(policy > _last_predefined_policy, "Policy %d invalid. Only user-registered policies can be unregistered\n", policy);
	_policies[policy] = NULL;
	return 0;
}

int _starpu_mpi_select_node_with_most_data(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
{
	size_t *size_on_nodes;
	size_t max_size;
	int i;
	int xrank = 0;

	(void)me;
	_STARPU_MPI_CALLOC(size_on_nodes, nb_nodes, sizeof(size_t));

	for(i= 0 ; i<nb_data ; i++)
	{
		starpu_data_handle_t data = descr[i].handle;
		enum starpu_data_access_mode mode = descr[i].mode;
		int rank = starpu_data_get_rank(data);
		size_t size = data->ops->get_size(data);

		if (mode & STARPU_R)
			size_on_nodes[rank] += size;

		if (mode & STARPU_W)
			/* Would have to transfer it back */
			size_on_nodes[rank] += size;
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

	free(size_on_nodes);
	return xrank;
}

int _starpu_mpi_select_node(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data, int policy)
{
	int ppolicy = policy == STARPU_MPI_NODE_SELECTION_CURRENT_POLICY ? _current_policy : policy;
	STARPU_ASSERT_MSG(ppolicy < _STARPU_MPI_NODE_SELECTION_MAX_POLICY, "Invalid policy %d\n", ppolicy);
	STARPU_ASSERT_MSG(_policies[ppolicy], "Unregistered policy %d\n", ppolicy);
	starpu_mpi_select_node_policy_func_t func = _policies[ppolicy];
	return func(me, nb_nodes, descr, nb_data);
}
