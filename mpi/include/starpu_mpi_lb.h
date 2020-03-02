/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_LOAD_BALANCER_H__
#define __STARPU_MPI_LOAD_BALANCER_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_mpi_lb_conf
{
	void (*get_neighbors)(int **neighbor_ids, int *nneighbors);
	void (*get_data_unit_to_migrate)(starpu_data_handle_t **handle_unit, int *nhandles, int dst_node);
};

/* Inits the load balancer's environment with the load policy provided by the
 * user
 */
void starpu_mpi_lb_init(const char *lb_policy_name, struct starpu_mpi_lb_conf *);
void starpu_mpi_lb_shutdown();

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_LOAD_BALANCER_H__
