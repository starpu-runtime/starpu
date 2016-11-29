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
#include <core/perfmodel/perfmodel.h>
#include "driver_mpi_common.h"

#define NITER 32
#define SIZE_BANDWIDTH (1024*1024)

#define SYNC_TAG 44
#define ASYNC_TAG 45

#define DRIVER_MPI_MASTER_NODE_DEFAULT 0

static int mpi_initialized = 0;
static int extern_initialized = 0;
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
    //Here we supposed the programmer called two times starpu_init.
    if (mpi_initialized)
        return -ENODEV;

    mpi_initialized = 1;

    if (MPI_Initialized(&extern_initialized) != MPI_SUCCESS)
        STARPU_ABORT_MSG("Cannot check if MPI is initialized or not !");

    //Here MPI_Init or MPI_Init_thread is already called
    if (!extern_initialized)
    {

#if defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
        int required = MPI_THREAD_MULTIPLE;
#else
        int required = MPI_THREAD_FUNNELED;
#endif

            int thread_support;
            STARPU_ASSERT(MPI_Init_thread(_starpu_get_argc(), _starpu_get_argv(), required, &thread_support) == MPI_SUCCESS);

            if (thread_support != required)
            {
                if (required == MPI_THREAD_MULTIPLE)
                    fprintf(stderr, "MPI doesn't support MPI_THREAD_MULTIPLE option. MPI Master-Slave can have problems if multiple slaves are launched. \n");
                if (required == MPI_THREAD_FUNNELED)
                    fprintf(stderr, "MPI doesn't support MPI_THREAD_FUNNELED option. Many errors can occur. \n");
            }
        }
        
        /* Find which node is the master */
        _starpu_mpi_set_src_node_id();

        return 1;
    }

void _starpu_mpi_common_mp_deinit()
{
    if (!extern_initialized)
        MPI_Finalize();    
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
    int res, source;
    int flag = 0;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    if (id_proc == src_node_id)
    {
        /* Source has mp_node defined */
        source = mp_node->mp_connection.mpi_remote_nodeid;
    }
    else
    {
        /* Sink can have sink to sink message */
        source = MPI_ANY_SOURCE;
    }

    res = MPI_Iprobe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot test if we received a message !");

    return flag;
}

/* SEND to source node */
void _starpu_mpi_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
    int res;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    printf("envoi %d B to %d\n", len, node->mp_connection.mpi_remote_nodeid);

    if (event)
    {
        /* Asynchronous send */
        struct _starpu_async_channel * channel = event;
        channel->event.mpi_ms_event.finished = 0;
        channel->event.mpi_ms_event.is_sender = 1;
        res = MPI_Isend(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, ASYNC_TAG, MPI_COMM_WORLD, &channel->event.mpi_ms_event.request);
    } 
    else
    {
        /* Synchronous send */
        res = MPI_Send(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, SYNC_TAG, MPI_COMM_WORLD);
    }
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

void _starpu_mpi_common_mp_send(const struct _starpu_mp_node *node, void *msg, int len)
{
    _starpu_mpi_common_send(node, msg, len, NULL);
}


/* RECV to source node */
void _starpu_mpi_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
    int res;
    int id_proc;
    MPI_Status s;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    printf("recv %d B from %d in %p\n", len, node->mp_connection.mpi_remote_nodeid, msg);

    if (event)
    {
        /* Asynchronous recv */
        struct _starpu_async_channel * channel = event;
        channel->event.mpi_ms_event.finished = 0;
        channel->event.mpi_ms_event.is_sender = 0;
        res = MPI_Irecv(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, ASYNC_TAG, MPI_COMM_WORLD, &channel->event.mpi_ms_event.request);
    } 
    else
    {
        /* Synchronous recv */
        res = MPI_Recv(msg, len, MPI_BYTE, node->mp_connection.mpi_remote_nodeid, SYNC_TAG, MPI_COMM_WORLD, &s);
        int num_expected;
        MPI_Get_count(&s, MPI_BYTE, &num_expected);

        STARPU_ASSERT_MSG(num_expected == len, "MPI Master/Slave received a msg with a size of %d Bytes (expected %d Bytes) !", num_expected, len);
    }
    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

void _starpu_mpi_common_mp_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
    _starpu_mpi_common_recv(node, msg, len, NULL);
}

/* SEND to any node */
void _starpu_mpi_common_send_to_device(const struct _starpu_mp_node *node, int dst_devid, void *msg, int len, void * event)
{   
    int res;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    printf("send %d bytes from %d from %p\n", len, dst_devid, msg);

    if (event)
    {
        /* Asynchronous send */
        struct _starpu_async_channel * channel = event;
        channel->event.mpi_ms_event.finished = 0;
        channel->event.mpi_ms_event.is_sender = 1;
        res = MPI_Isend(msg, len, MPI_BYTE, dst_devid, ASYNC_TAG, MPI_COMM_WORLD, &channel->event.mpi_ms_event.request);
    } 
    else
    {
        /* Synchronous send */
        res = MPI_Send(msg, len, MPI_BYTE, dst_devid, SYNC_TAG, MPI_COMM_WORLD);
    }    

    STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

/* RECV to any node */
void _starpu_mpi_common_recv_from_device(const struct _starpu_mp_node *node, int src_devid, void *msg, int len, void * event)
{
    int res;
    int id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

    printf("nop recv %d bytes from %d\n", len, src_devid);

    if (event)
    {
        /* Asynchronous recv */
        struct _starpu_async_channel * channel = event;
        channel->event.mpi_ms_event.finished = 0;
        channel->event.mpi_ms_event.is_sender = 0;
        res = MPI_Irecv(msg, len, MPI_BYTE, src_devid, ASYNC_TAG, MPI_COMM_WORLD, &channel->event.mpi_ms_event.request);
    } 
    else
    {
        /* Synchronous recv */
        res = MPI_Recv(msg, len, MPI_BYTE, src_devid, SYNC_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
    }
}

/* - In MPI Master-Slave communications between host and device,
 * host is always considered as the sender and the device, the receiver.
 * - In device to device communications, the first ack received by host
 * is considered as the sender (but it cannot be, in fact, the sender)
 */
int _starpu_mpi_common_test_event(struct _starpu_async_channel * event)
{
    //if the event is not finished, maybe it's a host-device communication
    //or host has already finished its work
    if (!event->event.mpi_ms_event.finished)
    {
        int flag = 0;
        MPI_Test(&event->event.mpi_ms_event.request, &flag, MPI_STATUS_IGNORE);
        if (flag)
        {
            event->event.mpi_ms_event.finished = 1;
            if (event->event.mpi_ms_event.is_sender)
                event->starpu_mp_common_finished_sender = 1;
            else
                event->starpu_mp_common_finished_receiver = 1;
        }
    }

    return event->starpu_mp_common_finished_sender && event->starpu_mp_common_finished_receiver;
}




void _starpu_mpi_common_barrier(void)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

/* Compute bandwidth and latency between source and sink nodes
 * Source node has to have the entire set of times at the end
 */
void _starpu_mpi_common_measure_bandwidth_latency(double * bandwidth_htod, double * bandwidth_dtoh, double * latency_htod, double * latency_dtoh)
{
    int ret;
    unsigned iter;

    int nb_proc, id_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

    char * buf = malloc(SIZE_BANDWIDTH);
    memset(buf, 0, SIZE_BANDWIDTH);

    unsigned node;
    unsigned id = 0;
    for(node = 0; node < nb_proc; node++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        //Don't measure link master <-> master
        if(node == src_node_id)
            continue;

        if(_starpu_mpi_common_is_src_node())
        {
            double start, end;

            /* measure bandwidth host to device */
            start = starpu_timing_now();
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Send(buf, SIZE_BANDWIDTH, MPI_BYTE, node, node, MPI_COMM_WORLD); 
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }
            end = starpu_timing_now();
            bandwidth_htod[id] = (NITER*1000000)/(end - start);

            /* measure bandwidth device to host */
            start = starpu_timing_now();
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Recv(buf, SIZE_BANDWIDTH, MPI_BYTE, node, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }
            end = starpu_timing_now();
            bandwidth_dtoh[id] = (NITER*1000000)/(end - start);

            /* measure latency host to device */
            start = starpu_timing_now();
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Send(buf, 1, MPI_BYTE, node, node, MPI_COMM_WORLD); 
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Latency of MPI Master/Slave cannot be measured !");
            }
            end = starpu_timing_now();
            latency_htod[id] = (end - start)/NITER;

            /* measure latency device to host */
            start = starpu_timing_now();
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Recv(buf, 1, MPI_BYTE, node, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }
            end = starpu_timing_now();
            latency_dtoh[id] = (end - start)/NITER;

        }
        else if (node == id_proc) /* if we are the sink node evaluated */
        {
            /* measure bandwidth host to device */
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Recv(buf, SIZE_BANDWIDTH, MPI_BYTE, src_node_id, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }

            /* measure bandwidth device to host */
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Send(buf, SIZE_BANDWIDTH, MPI_BYTE, src_node_id, node, MPI_COMM_WORLD); 
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }

            /* measure latency host to device */
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Recv(buf, 1, MPI_BYTE, src_node_id, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
            }

            /* measure latency device to host */
            for (iter = 0; iter < NITER; iter++)
            {
                ret = MPI_Send(buf, 1, MPI_BYTE, src_node_id, node, MPI_COMM_WORLD); 
                STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Latency of MPI Master/Slave cannot be measured !");
            }
        }

        id++;
    }
    free(buf);
}
