/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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

#include <common/COISysInfo_common.h>

#include <starpu.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <datawizard/interfaces/data_interface.h>

#include "driver_mic_common.h"
#include "driver_mic_sink.h"

#define HYPER_THREAD_NUMBER 4


/* Initialize the MIC sink, initializing connection to the source
 * and to the other devices (not implemented yet).
 */

void _starpu_mic_sink_init(struct _starpu_mp_node *node)
{
	//unsigned int i;
	
	/* Initialize connection with the source */
	_starpu_mic_common_accept(&node->mp_connection.mic_endpoint,
					 STARPU_MIC_SOURCE_PORT_NUMBER);

	_starpu_mic_common_accept(&node->host_sink_dt_connection.mic_endpoint,
									 STARPU_MIC_SOURCE_DT_PORT_NUMBER);

	//node->sink_sink_dt_connections = malloc(node->nb_mp_sinks * sizeof(union _starpu_mp_connection));

	//for (i = 0; i < (unsigned int)node->devid; ++i)
	//	_starpu_mic_common_connect(&node->sink_sink_dt_connections[i].mic_endpoint,
	//								STARPU_TO_MIC_ID(i),
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(node->devid, i),	
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(i, node->devid));

	//for (i = node->devid + 1; i < node->nb_mp_sinks; ++i)
	//	_starpu_mic_common_accept(&node->sink_sink_dt_connections[i].mic_endpoint,
	//								STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(node->devid, i));
}

/* Deinitialize the MIC sink, close all the connections.
 */

void _starpu_mic_sink_deinit(struct _starpu_mp_node *node)
{
	//unsigned int i;

	//for (i = 0; i < node->nb_mp_sinks; ++i)
	//{
	//	if (i != (unsigned int)node->devid)
	//		scif_close(node->sink_sink_dt_connections[i].mic_endpoint);
	//}

	//free(node->sink_sink_dt_connections);

	scif_close(node->host_sink_dt_connection.mic_endpoint);
	scif_close(node->mp_connection.mic_endpoint);
}

/* Report an error which occured when using a MIC device
 * and print this error in a human-readable style
 */

void _starpu_mic_sink_report_error(const char *func, const char *file, const int line, const int status)
{
	const char *errormsg = strerror(status);
	printf("SINK: oops in %s (%s:%u)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ASSERT(0);
}

/* Return the number of cores on the callee, a MIC device or Processor Xeon
 */
unsigned int _starpu_mic_sink_get_nb_core(void)
{
	return (unsigned int) COISysGetCoreCount();
}

/* Allocate memory on the MIC.
 * Memory is register for remote direct access. */
void _starpu_mic_sink_allocate(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(size_t));

	void *addr = NULL;
	size_t size = *(size_t *)(arg);
	
	if (posix_memalign(&addr, STARPU_MIC_PAGE_SIZE, size) != 0)
		_starpu_mp_common_send_command(mp_node, STARPU_ERROR_ALLOCATE, NULL, 0);

#ifndef STARPU_DISABLE_ASYNCHRONOUS_MIC_COPY
	scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;
	size_t window_size = STARPU_MIC_GET_PAGE_SIZE_MULTIPLE(size);

	if (scif_register(epd, addr, window_size, (off_t)addr, SCIF_PROT_READ | SCIF_PROT_WRITE, SCIF_MAP_FIXED) < 0)
	{
		free(addr);
		_starpu_mp_common_send_command(mp_node, STARPU_ERROR_ALLOCATE, NULL, 0);
	}
#endif
	
	_starpu_mp_common_send_command(mp_node, STARPU_ANSWER_ALLOCATE, &addr, sizeof(addr));
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
void _starpu_mic_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, cpu_set_t * cpuset, int coreid, pthread_t *thread)
{
  int j, ret;
  //init the set
  CPU_ZERO(cpuset);

  //adding the core to the set
  for(j=0;j<HYPER_THREAD_NUMBER;j++)
    CPU_SET(j+coreid*HYPER_THREAD_NUMBER,cpuset);
  
  //affect the thread to the core
  ret = pthread_setaffinity_np(*thread, sizeof(cpu_set_t), cpuset);
  STARPU_ASSERT(ret == 0);
}
