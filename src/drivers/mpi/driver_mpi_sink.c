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
#include "driver_mpi_common.h"

void _starpu_mpi_sink_init(struct _starpu_mp_node *node)
{
    _starpu_mpi_common_mp_initialize_src_sink(node);

    node->thread_table = malloc(sizeof(starpu_pthread_t)*node->nb_cores);
    //TODO
}

void _starpu_mpi_sink_deinit(struct _starpu_mp_node *node)
{
    free(node->thread_table);
    //TODO
}

void _starpu_mpi_sink_launch_workers(struct _starpu_mp_node *node)
{
    //TODO
    int i, ret;
    struct arg_sink_thread * arg;
    cpu_set_t cpuset;
    starpu_pthread_attr_t attr;
    starpu_pthread_t thread;

    for(i=0; i < node->nb_cores; i++)
    {
        //init the set
        CPU_ZERO(&cpuset);
        CPU_SET(i,&cpuset);

        ret = starpu_pthread_attr_init(&attr);
        STARPU_ASSERT(ret == 0);
        ret = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        STARPU_ASSERT(ret == 0);

        /*prepare the argument for the thread*/
        arg= malloc(sizeof(struct arg_sink_thread));
        arg->coreid = i;
        arg->node = node;

        ret = starpu_pthread_create(&thread, &attr, _starpu_sink_thread, arg);
        STARPU_ASSERT(ret == 0);
        ((starpu_pthread_t *)node->thread_table)[i] = thread;

    }
}

void _starpu_mpi_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, int coreid, int * core_table, int nb_core)
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

