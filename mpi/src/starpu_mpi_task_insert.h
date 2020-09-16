/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MPI_TASK_INSERT_H__
#define __STARPU_MPI_TASK_INSERT_H__

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

int _starpu_mpi_find_executee_node(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int *do_execute, int *inconsistent_execute, int *xrank);
void _starpu_mpi_exchange_data_before_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int xrank, int do_execute, int prio, MPI_Comm comm);
int _starpu_mpi_task_postbuild_v(MPI_Comm comm, int xrank, int do_execute, struct starpu_data_descr *descrs, int nb_data, int prio);

#ifdef __cplusplus
}
#endif
#endif /* __STARPU_MPI_TASK_INSERT_H__ */
