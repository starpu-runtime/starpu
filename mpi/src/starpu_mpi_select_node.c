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

int _starpu_mpi_select_node(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
{
	size_t *size_on_nodes;
	size_t max_size;
	int i;
	int xrank;

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

	// We select the node which has the most data in R mode
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
