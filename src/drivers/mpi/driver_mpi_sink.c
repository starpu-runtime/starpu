/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "driver_mpi_sink.h"
#include "driver_mpi_source.h"
#include "driver_mpi_common.h"

void _starpu_mpi_sink_init(struct _starpu_mp_node *node)
{
	_starpu_mpi_common_mp_initialize_src_sink(node);

	_STARPU_MALLOC(node->thread_table, sizeof(starpu_pthread_t)*node->nb_cores);
	//TODO
}

void _starpu_mpi_sink_bind_thread(const struct _starpu_mp_node *mp_node, int coreid, int *core_table, int nb_core)
{
	//TODO
	(void)mp_node;
	(void)coreid;
	(void)core_table;
	(void)nb_core;
}
