/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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


#include <errno.h>
#include <dlfcn.h>

#include <common/COISysInfo_common.h>

#include <starpu.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <datawizard/interfaces/data_interface.h>

#include "driver_mic_common.h"
#include "driver_mic_sink.h"

static int mic_core_to_thread[240];
/* Initialize the MIC sink, initializing connection to the source
 * and to the other devices (not implemented yet).
 */
void _starpu_mic_sink_init(struct _starpu_mp_node *node)
{
#ifdef __KNC__
	starpu_pthread_t self;
	cpu_set_t cpuset;
	/* We reserve one core for the communications */
	/*Bind on the first core*/
	self = starpu_pthread_self();
	CPU_ZERO(&cpuset);
	CPU_SET(0,&cpuset);
	pthread_setaffinity_np(self,sizeof(cpu_set_t),&cpuset);
#endif


	/* Initialize connection with the source */
	_starpu_mic_common_accept(&node->mp_connection.mic_endpoint,
					 STARPU_MIC_SOURCE_PORT_NUMBER);

	_starpu_mic_common_accept(&node->host_sink_dt_connection.mic_endpoint,
									 STARPU_MIC_SOURCE_DT_PORT_NUMBER);

	node->nb_cores = COISysGetHardwareThreadCount() - COISysGetHardwareThreadCount() / COISysGetCoreCount();
	_STARPU_MALLOC(node->thread_table, sizeof(starpu_pthread_t)*node->nb_cores);

#ifdef STARPU_DEVEL
#warning rather use hwloc
#endif
#ifdef __KNC__
	unsigned core,thread;
	/* Round-robin between cores. Take care of the odd numbering of threads on the KNC */
	for (core = 0; core < 60; core++)
		for (thread = 0; thread < 4; thread++)
			mic_core_to_thread[core + thread * 60] = core * 4 + thread + 1;
#elif defined(__KNF__)
#error need to check the numbering
#endif

	//_STARPU_MALLOC(node->sink_sink_dt_connections, node->nb_mp_sinks * sizeof(union _starpu_mp_connection));

	//for (i = 0; i < (unsigned int)node->devid; ++i)
	//	_starpu_mic_common_connect(&node->sink_sink_dt_connections[i].mic_endpoint,
	//								STARPU_TO_MIC_ID(i),
	//								NULL,
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(node->devid, i),
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(i, node->devid));

	//for (i = node->devid + 1; i < node->nb_mp_sinks; ++i)
	//	_starpu_mic_common_accept(&node->sink_sink_dt_connections[i].mic_endpoint,
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(node->devid, i));
}

/* Launch all workers on the mic
 */
void _starpu_mic_sink_launch_workers(struct _starpu_mp_node *node)
{
	int i;
	struct arg_sink_thread * arg;
	cpu_set_t cpuset;
	starpu_pthread_attr_t attr;
	starpu_pthread_t thread;

	/*for each core init the mutex, the task pointer and launch the thread */
	for(i=0; i<node->nb_cores; i++)
	{
		int ret;

		//init the set
		CPU_ZERO(&cpuset);
		CPU_SET(i,&cpuset);

		ret = starpu_pthread_attr_init(&attr);
		STARPU_ASSERT(ret == 0);
		ret = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
		STARPU_ASSERT(ret == 0);

		/*prepare the argument for the thread*/
		_STARPU_MALLOC(arg, sizeof(struct arg_sink_thread));
		arg->coreid = i;
		arg->node = node;

		STARPU_PTHREAD_CREATE(&thread, &attr, _starpu_sink_thread, arg);
		((starpu_pthread_t *)node->thread_table)[i] = thread;
	}
}

/* Deinitialize the MIC sink, close all the connections.
 */
void _starpu_mic_sink_deinit(struct _starpu_mp_node *node)
{

	int i;
	node->is_running = 0;
	for(i=0; i<node->nb_cores; i++)
	{
		sem_post(&node->sem_run_table[i]);
		STARPU_PTHREAD_JOIN(((starpu_pthread_t *)node->thread_table)[i],NULL);
	}

	free(node->thread_table);

	scif_close(node->host_sink_dt_connection.mic_endpoint);
	scif_close(node->mp_connection.mic_endpoint);

	//unsigned int i;

	//for (i = 0; i < node->nb_mp_sinks; ++i)
	//{
	//	if (i != (unsigned int)node->devid)
	//		scif_close(node->sink_sink_dt_connections[i].mic_endpoint);
	//}

	//free(node->sink_sink_dt_connections);

}

/* Report an error which occured when using a MIC device
 * and print this error in a human-readable style
 */
void _starpu_mic_sink_report_error(const char *func, const char *file, const int line, const int status)
{
	const char *errormsg = strerror(status);
	_STARPU_ERROR("SINK: oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

/* Allocate memory on the MIC.
 * Memory is register for remote direct access. */
void _starpu_mic_sink_allocate(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(size_t));

	void *addr = NULL;
	size_t size = *(size_t *)(arg);

	if (posix_memalign(&addr, STARPU_MIC_PAGE_SIZE, size) != 0)
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ERROR_ALLOCATE, NULL, 0);

#ifndef STARPU_DISABLE_ASYNCHRONOUS_MIC_COPY
	scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;
	size_t window_size = STARPU_MIC_GET_PAGE_SIZE_MULTIPLE(size);

	if (scif_register(epd, addr, window_size, (off_t)addr, SCIF_PROT_READ | SCIF_PROT_WRITE, SCIF_MAP_FIXED) < 0)
	{
		free(addr);
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ERROR_ALLOCATE, NULL, 0);
	}
#endif

	_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ANSWER_ALLOCATE, &addr, sizeof(addr));
}

/* Unregister and free memory. */
void _starpu_mic_sink_free(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mic_free_command));

	void *addr = ((struct _starpu_mic_free_command *)arg)->addr;

#ifndef STARPU_DISABLE_ASYNCHRONOUS_MIC_COPY
	scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;
	size_t size = ((struct _starpu_mic_free_command *)arg)->size;
	size_t window_size = STARPU_MIC_GET_PAGE_SIZE_MULTIPLE(size);

	scif_unregister(epd, (off_t)addr, window_size);
#endif
	free(addr);
}


/* bind the thread to a core
 */
#ifdef STARPU_DEVEL
#warning Use hwloc, the numbering is *really* odd on the MIC
#endif
void _starpu_mic_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, int coreid, int * core_table, int nb_core)
{
	cpu_set_t cpuset;
	int i;

  	//init the set
	CPU_ZERO(&cpuset);

	//adding the core to the set
	for(i=0;i<nb_core;i++)
		CPU_SET(mic_core_to_thread[core_table[i]],&cpuset);

	pthread_setaffinity_np(((starpu_pthread_t*)mp_node->thread_table)[coreid],sizeof(cpu_set_t),&cpuset);
}

void (*_starpu_mic_sink_lookup (const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED, char* func_name))(void)
{
	void *dl_handle = dlopen(NULL, RTLD_NOW);
	return dlsym(dl_handle, func_name);
}
