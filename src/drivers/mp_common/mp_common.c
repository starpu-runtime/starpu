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

#include <stdlib.h>
#include <pthread.h>

#include <datawizard/interfaces/data_interface.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <drivers/mic/driver_mic_common.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/mic/driver_mic_sink.h>
#include <drivers/scc/driver_scc_common.h>
#include <drivers/scc/driver_scc_source.h>
#include <drivers/scc/driver_scc_sink.h>

#include <common/list.h>

const char *_starpu_mp_common_command_to_string(const int command)
{
	switch(command)
	{
		case STARPU_EXIT:
			return "EXIT";
		case STARPU_EXECUTE:
			return "EXECUTE";
		case STARPU_ERROR_EXECUTE:
			return "ERROR_EXECUTE";
		case STARPU_LOOKUP:
			return "LOOKUP";
		case STARPU_ANSWER_LOOKUP:
			return "ANSWER_LOOKUP";
		case STARPU_ERROR_LOOKUP:
			return "ERROR_LOOKUP";
		case STARPU_ALLOCATE:
			return "ALLOCATE";
		case STARPU_ANSWER_ALLOCATE:
			return "ANSWER_ALLOCATE";
		case STARPU_ERROR_ALLOCATE:
			return "ERROR_ALLOCATE";
		case STARPU_FREE:
			return "FREE";
		case STARPU_RECV_FROM_HOST:
			return "RECV_FROM_HOST";
		case STARPU_SEND_TO_HOST:
			return "SEND_TO_HOST";
		case STARPU_RECV_FROM_SINK:
			return "RECV_FROM_SINK";
		case STARPU_SEND_TO_SINK:
			return "SEND_TO_SINK";
		case STARPU_TRANSFER_COMPLETE:
			return "TRANSFER_COMPLETE";
		case STARPU_SINK_NBCORES:
			return "SINK_NBCORES";
		case STARPU_ANSWER_SINK_NBCORES:
			return "ANSWER_SINK_NBCORES";
		case STARPU_EXECUTION_SUBMITTED:
			return "EXECUTION_SUBMITTED";
		case STARPU_EXECUTION_COMPLETED:
			return "EXECUTION_COMPLETED";
		case STARPU_PRE_EXECUTION:
			return "PRE_EXECUTION";
		case STARPU_SYNC_WORKERS:
			return "SYNC_WORKERS";
		default:
			return "<invalid command code>";
	}
}

const char *_starpu_mp_common_node_kind_to_string(const int kind)
{
	switch(kind)
	{
		case STARPU_MIC_SINK:
			return "MIC_SINK";
		case STARPU_MIC_SOURCE:
			return "MIC_SOURCE";
		case STARPU_SCC_SINK:
			return "SCC_SINK";
		case STARPU_SCC_SOURCE:
			return "SCC_SOURCE";
		case STARPU_MPI_SINK:
			return "MPI_SINK";
		case STARPU_MPI_SOURCE:
			return "MPI_SOURCE";
		default:
			return "<invalid command code>";
	}
}
/* Allocate and initialize the sink structure, when the function returns
 * all the pointer of functions are linked to the right ones.
 */
struct _starpu_mp_node * STARPU_ATTRIBUTE_MALLOC
_starpu_mp_common_node_create(enum _starpu_mp_node_kind node_kind,
			      int peer_id)
{
	struct _starpu_mp_node *node;

	_STARPU_MALLOC(node, sizeof(struct _starpu_mp_node));

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
		node->launch_workers= NULL;
		node->deinit = _starpu_mic_src_deinit;
		node->report_error = _starpu_mic_src_report_scif_error;

		node->mp_recv_is_ready = _starpu_mic_common_recv_is_ready;
		node->mp_send = _starpu_mic_common_send;
		node->mp_recv = _starpu_mic_common_recv;
		node->dt_send = _starpu_mic_common_dt_send;
		node->dt_recv = _starpu_mic_common_dt_recv;

		node->get_kernel_from_job =_starpu_mic_src_get_kernel_from_job;
		node->lookup = NULL;
		node->bind_thread = NULL;
		node->execute = NULL;
		node->allocate = NULL;
		node->free = NULL;
	}
	break;

	case STARPU_MIC_SINK:
	{
		node->devid = atoi(starpu_getenv("_STARPU_MIC_DEVID"));
		node->nb_mp_sinks = atoi(starpu_getenv("_STARPU_MIC_NB"));

		node->init = _starpu_mic_sink_init;
		node->launch_workers = _starpu_mic_sink_launch_workers;
		node->deinit = _starpu_mic_sink_deinit;
		node->report_error = _starpu_mic_sink_report_error;

		node->mp_recv_is_ready = _starpu_mic_common_recv_is_ready;
		node->mp_send = _starpu_mic_common_send;
		node->mp_recv = _starpu_mic_common_recv;
		node->dt_send = _starpu_mic_common_dt_send;
		node->dt_recv = _starpu_mic_common_dt_recv;

		node->get_kernel_from_job = NULL;
		node->lookup = _starpu_mic_sink_lookup;
		node->bind_thread = _starpu_mic_sink_bind_thread;
		node->execute = _starpu_sink_common_execute;
		node->allocate = _starpu_mic_sink_allocate;
		node->free = _starpu_mic_sink_free;

	}
	break;
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_SCC
	case STARPU_SCC_SOURCE:
	{
		node->init = _starpu_scc_src_init;
		node->deinit = NULL;
		node->report_error = _starpu_scc_common_report_rcce_error;
				
		node->mp_recv_is_ready = _starpu_scc_common_recv_is_ready;
		node->mp_send = _starpu_scc_common_send;
		node->mp_recv = _starpu_scc_common_recv;
		node->dt_send = _starpu_scc_common_send;
		node->dt_recv = _starpu_scc_common_recv;
		node->dt_send_to_device = NULL;
		node->dt_recv_from_device = NULL;

		node->get_kernel_from_job =_starpu_scc_src_get_kernel_from_job;
		node->lookup = NULL;
		node->bind_thread = NULL;
		node->execute = NULL;
		node->allocate = NULL;
		node->free = NULL;
	}
	break;

	case STARPU_SCC_SINK:
	{
		node->init = _starpu_scc_sink_init;
		node->launch_workers = _starpu_scc_sink_launch_workers;
		node->deinit = _starpu_scc_sink_deinit;
		node->report_error = _starpu_scc_common_report_rcce_error;

		node->mp_recv_is_ready = _starpu_scc_common_recv_is_ready;
		node->mp_send = _starpu_scc_common_send;
		node->mp_recv = _starpu_scc_common_recv;
		node->dt_send = _starpu_scc_common_send;
		node->dt_recv = _starpu_scc_common_recv;
		node->dt_send_to_device = _starpu_scc_sink_send_to_device;
		node->dt_recv_from_device = _starpu_scc_sink_recv_from_device;

		node->get_kernel_from_job = NULL;
		node->lookup = _starpu_scc_sink_lookup;
		node->bind_thread = _starpu_scc_sink_bind_thread;
		node->execute = _starpu_scc_sink_execute;
		node->allocate = _starpu_sink_common_allocate;
		node->free = _starpu_sink_common_free;
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
	_STARPU_MALLOC(node->buffer, BUFFER_SIZE);

	if (node->init)
		node->init(node);

	mp_message_list_init(&node->message_queue);
	STARPU_PTHREAD_MUTEX_INIT(&node->message_queue_mutex,NULL);

	/* If the node is a sink then we must initialize some field */
	if(node->kind == STARPU_MIC_SINK || node->kind == STARPU_SCC_SINK)
	{
		int i;
		node->is_running = 1;
		_STARPU_MALLOC(node->run_table, sizeof(struct mp_task *)*node->nb_cores);
		_STARPU_MALLOC(node->sem_run_table, sizeof(sem_t)*node->nb_cores);

		for(i=0; i<node->nb_cores; i++)
		{
			node->run_table[i] = NULL;
			sem_init(&node->sem_run_table[i],0,0);
		}
		mp_barrier_list_init(&node->barrier_list);
		STARPU_PTHREAD_MUTEX_INIT(&node->barrier_mutex,NULL);

		STARPU_PTHREAD_BARRIER_INIT(&node->init_completed_barrier, NULL, node->nb_cores+1);

		node->launch_workers(node);
	}

	return node;
}

/* Deinitialize the sink structure and release the structure */
void _starpu_mp_common_node_destroy(struct _starpu_mp_node *node)
{
	if (node->deinit)
		node->deinit(node);
		
	STARPU_PTHREAD_MUTEX_DESTROY(&node->message_queue_mutex);

	/* If the node is a sink then we must destroy some field */
	if(node->kind == STARPU_MIC_SINK || node->kind == STARPU_SCC_SINK)
	{
		int i;
		for(i=0; i<node->nb_cores; i++)
		{
			sem_destroy(&node->sem_run_table[i]);
		}

		free(node->run_table);
		free(node->sem_run_table);

		STARPU_PTHREAD_MUTEX_DESTROY(&node->barrier_mutex);
		STARPU_PTHREAD_BARRIER_DESTROY(&node->init_completed_barrier);
	}

	free(node->buffer);

	free(node);
}

/* Send COMMAND to RECIPIENT, along with ARG if ARG_SIZE is non-zero */
void _starpu_mp_common_send_command(const struct _starpu_mp_node *node,
				    const enum _starpu_mp_command command,
				    void *arg, int arg_size)
{
	STARPU_ASSERT_MSG(arg_size <= BUFFER_SIZE, "Too much data (%d) for the static MIC buffer (%d), increase BUFFER_SIZE perhaps?", arg_size, BUFFER_SIZE);

	/* MIC and MPI sizes are given through a int */
	int command_size = sizeof(enum _starpu_mp_command);
	int arg_size_size = sizeof(int);

	/* Let's copy the data into the command line buffer */
	memcpy(node->buffer, &command, command_size);
	memcpy((void*) ((uintptr_t)node->buffer + command_size), &arg_size, arg_size_size);

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
	*arg_size = *((int *) ((uintptr_t)node->buffer + command_size));

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
