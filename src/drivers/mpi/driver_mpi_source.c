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

/* Mutex for concurrent access to the table.
 */
starpu_pthread_mutex_t htbl_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

/* Structure used by host to store informations about a kernel executable on
 * a MPI MS device : its name, and its address on each device.
 * If a kernel has been initialized, then a lookup has already been achieved and the
 * device knows how to call it, else the host still needs to do a lookup.
 */
struct _starpu_mpi_ms_kernel
{
	UT_hash_handle hh;
	char *name;
	starpu_mpi_ms_kernel_t func[STARPU_MAXMPIDEVS];
} *kernels;


/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
struct _starpu_mp_node *mpi_ms_nodes[STARPU_MAXMPIDEVS];

void _starpu_mpi_source_init(struct _starpu_mp_node *node)
{
    _starpu_mpi_common_mp_initialize_src_sink(node);
    //TODO
}

void _starpu_mpi_source_deinit(struct _starpu_mp_node *node)
{

}

int _starpu_mpi_ms_src_register_kernel(starpu_mpi_ms_func_symbol_t *symbol, const char *func_name)
{
	unsigned int func_name_size = (strlen(func_name) + 1) * sizeof(char);

	STARPU_PTHREAD_MUTEX_LOCK(&htbl_mutex);
	struct _starpu_mpi_ms_kernel *kernel;
	
	HASH_FIND_STR(kernels, func_name, kernel);

	if (kernel != NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		// Function already in the table.
		*symbol = kernel;
		return 0;
	}

	kernel = malloc(sizeof(*kernel));
	if (kernel == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		return -ENOMEM;
	}

	kernel->name = malloc(func_name_size);
	if (kernel->name == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		free(kernel);
		return -ENOMEM;
	}

	memcpy(kernel->name, func_name, func_name_size);

	HASH_ADD_STR(kernels, name, kernel);

	unsigned int nb_mpi_devices = _starpu_mpi_src_get_device_count();
	unsigned int i;
	for (i = 0; i < nb_mpi_devices; ++i)
		kernel->func[i] = NULL;

	STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);

	*symbol = kernel;

	return 0;
}


starpu_mpi_ms_kernel_t _starpu_mpi_ms_src_get_kernel(starpu_mpi_ms_func_symbol_t symbol)
{
	int workerid = starpu_worker_get_id();
	
	/* This function has to be called in the codelet only, by the thread
	 * which will handle the task */
	if (workerid < 0)
		return NULL;

	int devid = starpu_worker_get_devid(workerid);

	struct _starpu_mpi_ms_kernel *kernel = symbol;

	if (kernel->func[devid] == NULL)
	{
		struct _starpu_mp_node *node = mpi_ms_nodes[devid];
		int ret = _starpu_src_common_lookup(node, (void (**)(void))&kernel->func[devid], kernel->name);
		if (ret)
			return NULL;
	}

	return kernel->func[devid];
}

void(* _starpu_mpi_ms_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void)
{
	starpu_mpi_ms_kernel_t kernel = NULL;

	starpu_mpi_ms_func_t func = _starpu_task_get_mpi_ms_nth_implementation(j->task->cl, j->nimpl);
	if (func)
	{
		/* We execute the function contained in the codelet, it must return a
		 * pointer to the function to execute on the device, either specified
		 * directly by the user or by a call to starpu_mpi_ms_get_func().
		 */
		kernel = func();
	}
	else
	{
		/* If user dont define any starpu_mpi_ms_fun_t in cl->mpi_ms_func we try to use
		 * cpu_func_name.
		 */
		char *func_name = _starpu_task_get_cpu_name_nth_implementation(j->task->cl, j->nimpl);
		if (func_name)
		{
			starpu_mpi_ms_func_symbol_t symbol;

			_starpu_mpi_ms_src_register_kernel(&symbol, func_name);

			kernel = _starpu_mpi_ms_src_get_kernel(symbol);
		}
	}
	STARPU_ASSERT(kernel);

	return (void (*)(void))kernel;
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

