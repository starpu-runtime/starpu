/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef FT_STARPU_STARPU_MPI_CHECKPOINT_H
#define FT_STARPU_STARPU_MPI_CHECKPOINT_H

#include <starpu_mpi.h>
#include <common/list.h>
#include <starpu_mpi_private.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern int _my_rank;

struct _starpu_mpi_cp_ack_msg
{
	int checkpoint_id;
	int checkpoint_instance;
};

struct _starpu_mpi_cp_info_msg
{
	int checkpoint_id;
	int checkpoint_instance;
	int validation:1;
	int discard:1;
};

struct _starpu_mpi_cp_ack_arg_cb
{
	int                           rank;
	starpu_data_handle_t          handle;
	starpu_data_handle_t          copy_handle;
	int type;
	int count;
	starpu_mpi_tag_t              tag;
	struct _starpu_mpi_cp_ack_msg msg;
	int checkpoint_instance_hint;
	int cache_flag;
};

struct _starpu_mpi_cp_discard_arg_cb
{
	int                            rank;
	struct _starpu_mpi_cp_info_msg msg;
};

void _ack_msg_recv_cb(void* _args);

#ifdef __cplusplus
}
#endif

#endif //FT_STARPU_STARPU_MPI_CHECKPOINT_H
