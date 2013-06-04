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

#include <stdlib.h>
#include <pthread.h>

#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <drivers/mic/driver_mic_common.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/mic/driver_mic_sink.h>
#include <drivers/scc/driver_scc_common.h>
#include <drivers/scc/driver_scc_source.h>
#include <drivers/scc/driver_scc_sink.h>

/* Allocate and initialize the sink structure, when the function returns
 * all the pointer of functions are linked to the right ones.
 */
struct _starpu_mp_node * __attribute__((malloc))
    _starpu_mp_common_node_create(enum _starpu_mp_node_kind node_kind,
				  int peer_id)
{
	struct _starpu_mp_node *node;

	node = (struct _starpu_mp_node *) malloc(sizeof(struct _starpu_mp_node));

	node->kind = node_kind;

	node->peer_id = peer_id;

	switch(node->kind)
	{
#ifdef STARPU_USE_MIC
		case STARPU_MIC_SOURCE:
			{
				node->nb_mp_sinks = starpu_mic_worker_get_count();
				node->devid = peer_id;

				node->init = _starpu_mic_src_init;
				node->deinit = _starpu_mic_src_deinit;
				node->report_error = _starpu_mic_src_report_scif_error;

				node->mp_send = _starpu_mic_common_send;
				node->mp_recv = _starpu_mic_common_recv;
				node->dt_send = _starpu_mic_common_dt_send;
				node->dt_recv = _starpu_mic_common_dt_recv;

				node->execute = NULL;
				node->nbcores = NULL;
				node->allocate = NULL;
				node->free = NULL;

				/* A source node is only working on one core,
				 * there is no need for this function */
				node->get_nb_core = NULL;
			}
			break;

		case STARPU_MIC_SINK:
			{
				node->devid = atoi(getenv("DEVID"));;
				node->nb_mp_sinks = atoi(getenv("NB_MIC"));

				node->init = _starpu_mic_sink_init;
				node->deinit = _starpu_mic_sink_deinit;
				node->report_error = _starpu_mic_sink_report_error;

				node->mp_send = _starpu_mic_common_send;
				node->mp_recv = _starpu_mic_common_recv;
				node->dt_send = _starpu_mic_common_dt_send;
				node->dt_recv = _starpu_mic_common_dt_recv;

				node->execute = _starpu_sink_common_execute;
				node->nbcores = _starpu_sink_nbcores;
				node->allocate = _starpu_mic_sink_allocate;
				node->free = _starpu_mic_sink_free;

				node->get_nb_core = _starpu_mic_sink_get_nb_core;
			}
			break;
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_SCC
		case STARPU_SCC_SOURCE:
			{
				node->init = _starpu_scc_src_init;
				node->deinit = NULL;
				node->report_error = _starpu_scc_common_report_rcce_error;

				node->mp_send = _starpu_scc_common_send;
				node->mp_recv = _starpu_scc_common_recv;
				node->dt_send = _starpu_scc_common_send;
				node->dt_recv = _starpu_scc_common_recv;
				node->dt_send_to_device = NULL;
				node->dt_recv_from_device = NULL;

				node->execute = NULL;
				node->allocate = NULL;
				node->free = NULL;

				node->get_nb_core = NULL;
			}
			break;

		case STARPU_SCC_SINK:
			{
				node->init = _starpu_scc_sink_init;
				node->deinit = _starpu_scc_sink_deinit;
				node->report_error = _starpu_scc_common_report_rcce_error;

				node->mp_send = _starpu_scc_common_send;
				node->mp_recv = _starpu_scc_common_recv;
				node->dt_send = _starpu_scc_common_send;
				node->dt_recv = _starpu_scc_common_recv;
				node->dt_send_to_device = _starpu_scc_sink_send_to_device;
				node->dt_recv_from_device = _starpu_scc_sink_recv_from_device;

				node->execute = _starpu_scc_sink_execute;
				node->allocate = _starpu_sink_common_allocate;
				node->free = _starpu_sink_common_free;

				node->get_nb_core = NULL;
			}
			break;
#endif /* STARPU_USE_SCC */

#ifdef STARPU_USE_MPI
		case STARPU_MPI_SOURCE:
			STARPU_ABORT();
			break;

		case STARPU_MPI_SINK:
			STARPU_ABORT();
			break;
#endif /* STARPU_USE_MPI */

		default:
			STARPU_ASSERT(0);
	}

	/* Let's allocate the buffer, we want it to be big enough to contain
	 * a command, an argument and the argument size */
	node->buffer = (void *) malloc(BUFFER_SIZE);

	if (node->init)
		node->init(node);

	return node;
}

/* Deinitialize the sink structure and release the structure */

void _starpu_mp_common_node_destroy(struct _starpu_mp_node *node)
{
	if (node->deinit)
		node->deinit(node);

	free(node->buffer);

	free(node);
}

/* Send COMMAND to RECIPIENT, along with ARG if ARG_SIZE is non-zero */

void _starpu_mp_common_send_command(const struct _starpu_mp_node *node,
				    const enum _starpu_mp_command command,
				    void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size <= BUFFER_SIZE);

	/* MIC and MPI sizes are given through a int */
	int command_size = sizeof(enum _starpu_mp_command);
	int arg_size_size = sizeof(int);

	/* Let's copy the data into the command line buffer */
	memcpy(node->buffer, &command, command_size);
	memcpy(node->buffer + command_size, &arg_size, arg_size_size);

	node->mp_send(node, node->buffer, command_size + arg_size_size);

	if (arg_size)
		node->mp_send(node, arg, arg_size);
}

/* Return the command received from SENDER. In case SENDER sent an argument
 * beside the command, an address to a copy of this argument is returns in arg.
 * There is no need to free this address as it's not allocated at this time.
 * However, the data pointed by arg shouldn't be relied on after a new call to
 * STARPU_MP_COMMON_RECV_COMMAND as it might corrupt it.
 */

enum _starpu_mp_command _starpu_mp_common_recv_command(const struct _starpu_mp_node *node,
						       void **arg, int *arg_size)
{
	enum _starpu_mp_command command;

	/* MIC and MPI sizes are given through a int */
	int command_size = sizeof(enum _starpu_mp_command);
	int arg_size_size = sizeof(int);

	node->mp_recv(node, node->buffer, command_size + arg_size_size);

	command = *((enum _starpu_mp_command *) node->buffer);
	*arg_size = *((int *) (node->buffer + command_size));

	/* If there is no argument (ie. arg_size == 0),
	 * let's return the command right now */
	if (!(*arg_size))
	{
		*arg = NULL;
		return command;
	}

	STARPU_ASSERT(*arg_size <= BUFFER_SIZE);

	node->mp_recv(node, node->buffer, *arg_size);

	*arg = node->buffer;

	return command;
}
