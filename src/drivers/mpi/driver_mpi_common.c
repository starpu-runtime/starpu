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
#include <core/workers.h>
#include "driver_mpi_common.h"

#define DRIVER_MPI_MASTER_NODE_DEFAULT 0

static int mpi_initialized;
static int extern_initialized;
static int src_node_id;

static void _starpu_mpi_set_src_node_id()
{
	int node_id = starpu_get_env_number("STARPU_MPI_MASTER_NODE");

	if (node_id != -1)
	{
        int nb_proc, id_proc;
        MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
        MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

		if (node_id < nb_proc)
		{
			src_node_id = node_id;
			return;
		}
		else if (id_proc == DRIVER_MPI_MASTER_NODE_DEFAULT)
		{
			/* Only one node prints the error message. */
			fprintf(stderr, "The node you specify to be the master is "
					"greater than the total number of nodes.\n"
					"Taking node %d by default...\n", DRIVER_MPI_MASTER_NODE_DEFAULT);
		}
	}

	/* Node by default. */
	src_node_id = DRIVER_MPI_MASTER_NODE_DEFAULT;
}

int _starpu_mpi_common_mp_init()
{
    //Here we supposed the programmer has already called this function
    if (mpi_initialized)
        return 0;

    if (MPI_Initialized(&extern_initialized) != MPI_SUCCESS)
        STARPU_ABORT_MSG("Cannot check if MPI is initialized or not !");

    //Here MPI_Init or MPI_Init_thread is already called
    if (extern_initialized)
        return 1;

#if defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
    int required = MPI_THREAD_MULTIPLE;
#else
    int required = MPI_THREAD_FUNNELED;
#endif

    int thread_support;
	if (MPI_Init_thread(_starpu_get_argc(), _starpu_get_argv(), required, &thread_support) != MPI_SUCCESS)
        return 0;

    if (thread_support != required)
    {
        if (required == MPI_THREAD_MULTIPLE)
            fprintf(stderr, "MPI doesn't support MPI_THREAD_MULTIPLE option. MPI Master-Slave can have problems if multiple slaves are launched. \n");
        if (required == MPI_THREAD_FUNNELED)
            fprintf(stderr, "MPI doesn't support MPI_THREAD_FUNNELED option. Many errors can occur. \n");
    }

	mpi_initialized = 1;

    /* Find which node is the master */
    _starpu_mpi_set_src_node_id();

    return 1;
}

void _starpu_mpi_common_mp_deinit()
{
    if (!extern_initialized)
        MPI_Finalize();    

    mpi_initialized = 0;
}


int _starpu_mpi_common_is_src_node()
{   
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
    return id_proc == src_node_id;
} 

int _starpu_mpi_common_get_src_node()
{
    return src_node_id;
}

int _starpu_mpi_common_is_mp_initialized()
{
	return mpi_initialized;
}

/* common parts to initialize a source or a sink node */
void _starpu_mpi_common_mp_initialize_src_sink(struct _starpu_mp_node *node)
{
    struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;

    node->nb_cores = topology->nhwcpus;
}

int _starpu_mpi_common_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
    int res, tag;
    int flag = 0;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
        tag = mp_node->mp_connection.mpi_remote_nodeid;
    else
        tag = id_proc;
    
    res = MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot test if we received a message !");

    return flag;
}

/* SEND to source node */
void _starpu_mpi_common_send(const struct _starpu_mp_node *node, void *msg, int len)
{
    printf("envoi %d B to %d \n", len, node->mp_connection.mpi_remote_nodeid);
    int res, tag;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
        tag = node->mp_connection.mpi_remote_nodeid;
    else
        tag = id_proc;

    res = MPI_Send(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, tag, MPI_COMM_WORLD);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

/* RECV to source node */
void _starpu_mpi_common_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
    printf("recv %d B from %d \n", len, node->mp_connection.mpi_remote_nodeid);
    int res, tag;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
        tag = node->mp_connection.mpi_remote_nodeid;
    else
        tag = id_proc;

    res = MPI_Recv(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

/* SEND to any node */
void _starpu_mpi_common_send_to_device(const struct _starpu_mp_node *node, int dst_devid, void *msg, int len)
{   
    int res, tag;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
        tag = node->mp_connection.mpi_remote_nodeid;
    else
        tag = id_proc;

    res = MPI_Send(msg, len, MPI_BYTE, dst_devid, tag, MPI_COMM_WORLD);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

/* RECV to any node */
void _starpu_mpi_common_recv_from_device(const struct _starpu_mp_node *node, int src_devid, void *msg, int len)
{
    int res, tag;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
        tag = node->mp_connection.mpi_remote_nodeid;
    else
        tag = id_proc;

    res = MPI_Recv(msg, len, MPI_BYTE, src_devid, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}
