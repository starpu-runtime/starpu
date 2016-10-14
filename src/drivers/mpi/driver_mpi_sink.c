/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Mathieu Lirzin <mthl@openmailbox.org>
 * Copyright (C) 2016  Inria
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

void _starpu_mpi_sink_init(struct _starpu_mp_node *node)
{
    //TODO
}

void _starpu_mpi_sink_deinit(struct _starpu_mp_node *node)
{
    //TODO
}

void _starpu_mpi_sink_launch_workers(struct _starpu_mp_node *node)
{
    //TODO
}

//void _starpu_mpi_sink_send(const struct _starpu_mp_node *sink, void *msg,
//			   int len)
//{
//	int dst = STARPU_MP_SRC_NODE;
//	if (MPI_Send(msg, len, MPI_CHAR, dst, dst, MPI_COMM_WORLD))
//		STARPU_MP_COMMON_REPORT_ERROR(sink, errno);
//}
//
//void _starpu_mpi_sink_recv(const struct _starpu_mp_node *sink, void *msg,
//			   int len)
//{
//	int src = STARPU_MP_SRC_NODE;
//	if (MPI_Recv(msg, len, MPI_CHAR, src, sink->mp_connection.mpi_nodeid,
//		     MPI_COMM_WORLD, MPI_STATUS_IGNORE))
//		STARPU_MP_COMMON_REPORT_ERROR(sink, errno);
//}

