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
#include <errno.h>

#include <starpu.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>


struct _starpu_mp_node *mpi_ms_nodes[STARPU_MAXMPIDEVS];

//static void _starpu_mpi_src_init_context(int devid)
//{
//    mpi_mp_nodes[devid] = _starpu_mp_common_node_create(STARPU_MPI_SOURCE, devid);
//}

static void _starpu_mpi_src_deinit_context(int devid)
{
    _starpu_mp_common_send_command(mpi_ms_nodes[devid], STARPU_EXIT, NULL, 0);

    _starpu_mp_common_node_destroy(mpi_ms_nodes[devid]);
}



void _starpu_mpi_source_init(struct _starpu_mp_node *node)
{
    //TODO
}

void _starpu_mpi_source_deinit(struct _starpu_mp_node *node)
{
    //TODO
}

unsigned _starpu_mpi_src_get_device_count()
{
    int nb_mpi_devices;

    if (!_starpu_mpi_common_is_mp_initialized())
        return 0;
    
    MPI_Comm_size(MPI_COMM_WORLD, &nb_mpi_devices);

    //Remove one for master
    nb_mpi_devices = nb_mpi_devices - 1;

    return nb_mpi_devices;

}

 void _starpu_mpi_exit_useless_node(int devid)
{   
    struct _starpu_mp_node *node = _starpu_mp_common_node_create(STARPU_MPI_SOURCE, devid);

    _starpu_mp_common_send_command(node, STARPU_EXIT, NULL, 0);

    _starpu_mp_common_node_destroy(node);
}  

void *_starpu_mpi_src_worker(void *arg)
{
    struct _starpu_worker_set *worker_set = arg;
    /* As all workers of a set share common data, we just use the first
     *       * one for intializing the following stuffs. */
    struct _starpu_worker *baseworker = &worker_set->workers[0];
    struct _starpu_machine_config *config = baseworker->config;
    unsigned baseworkerid = baseworker - config->workers;
    unsigned devid = baseworker->devid;
    unsigned i;

    /* unsigned memnode = baseworker->memory_node; */

    _starpu_driver_start(baseworker, _STARPU_FUT_MPI_KEY, 0);
#ifdef STARPU_USE_FXT             
    for (i = 1; i < worker_set->nworkers; i++)
        _starpu_worker_start(&worker_set->workers[i], _STARPU_FUT_MPI_KEY, 0);
#endif          

    // Current task for a thread managing a worker set has no sense.
    _starpu_set_current_task(NULL);

    for (i = 0; i < config->topology.nmpicores[devid]; i++)
    {
        struct _starpu_worker *worker = &config->workers[baseworkerid+i];
        snprintf(worker->name, sizeof(worker->name), "MPI_MS %d core %u", devid, i);
        snprintf(worker->short_name, sizeof(worker->short_name), "MPI_MS %d.%u", devid, i);
    }
    {
        char thread_name[16];
        snprintf(thread_name, sizeof(thread_name), "MPI_MS %d", devid);
        starpu_pthread_setname(thread_name);
    }

    for (i = 0; i < worker_set->nworkers; i++)
    {
        struct _starpu_worker *worker = &worker_set->workers[i];
        _STARPU_TRACE_WORKER_INIT_END(worker->workerid);
    }

    /* tell the main thread that this one is ready */
    STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
    baseworker->status = STATUS_UNKNOWN;
    worker_set->set_is_initialized = 1;
    STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
    STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

    _starpu_src_common_worker(worker_set, baseworkerid, mpi_ms_nodes[devid]);

    return NULL;

    
}


//void _starpu_mpi_source_send(const struct _starpu_mp_node *node, void *msg,
//			     int len)
//{
//	int dst = node->mp_connection.mpi_nodeid;
//	if (MPI_Send(msg, len, MPI_CHAR, dst, dst, MPI_COMM_WORLD))
//		STARPU_MP_COMMON_REPORT_ERROR(node, errno);
//}
//
//void _starpu_mpi_source_recv(const struct _starpu_mp_node *node, void *msg,
//			     int len)
//{
//	int src = node->mp_connection.mpi_nodeid;
//	if (MPI_Recv(msg, len, MPI_CHAR, src, STARPU_MP_SRC_NODE,
//		     MPI_COMM_WORLD, MPI_STATUS_IGNORE))
//		STARPU_MP_COMMON_REPORT_ERROR(node, errno);
//}
//
//int _starpu_mpi_copy_src_to_sink(void *src,
//				 unsigned src_node STARPU_ATTRIBUTE_UNUSED,
//				 void *dst, unsigned dst_node, size_t size)
//{
//	/* TODO */
//	return 0;
//}
//
//int _starpu_mpi_copy_sink_to_src(void *src, unsigned src_node, void *dst,
//				 unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
//				 size_t size)
//{
//	/* TODO */
//	return 0;
//}
//
//int _starpu_mpi_copy_sink_to_sink(void *src, unsigned src_node, void *dst,
//				  unsigned dst_node, size_t size)
//{
//	/* TODO */
//	return 0;
//}
//
//void (*_starpu_mpi_get_kernel_from_job(const struct _starpu_mp_node *node,
//				       struct _starpu_job *j))(void)
//{
//	/* TODO */
//	return NULL;
//}

