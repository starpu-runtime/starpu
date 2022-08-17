/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef _STARPU_MPI_CHECKPOINT_PACKAGE_H
#define _STARPU_MPI_CHECKPOINT_PACKAGE_H

#include <starpu_mpi.h>
#include <common/list.h>
#include <starpu_mpi_private.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*TODO: This structure should be a hashtable accessible with these keys:
 *  CPid > CPinstance > Rank > tag */

LIST_TYPE(_starpu_mpi_checkpoint_data,
	int cp_id;
	int cp_inst;
	int rank;
	starpu_mpi_tag_t tag;
	int type;
	void* ptr;
	int count;
);

int checkpoint_package_init();
int checkpoint_package_shutdown();
int checkpoint_package_data_add(int cp_id, int cp_inst, int rank, starpu_mpi_tag_t tag, int type, void* ptr, int count);
int checkpoint_package_data_del(int cp_id, int cp_inst, int rank);

#ifdef __cplusplus
}
#endif

#endif //_STARPU_MPI_CHECKPOINT_PACKAGE_H
