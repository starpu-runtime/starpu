/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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

#include <datawizard/interfaces/data_interface.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_sink.h>
#include <drivers/tcpip/driver_tcpip_common.h>
#include <drivers/tcpip/driver_tcpip_source.h>
#include <drivers/tcpip/driver_tcpip_sink.h>

#include <common/list.h>

const char *_starpu_mp_common_command_to_string(const enum _starpu_mp_command command)
{
	switch(command)
	{
		/* Commands from master to slave */
		case STARPU_MP_COMMAND_EXIT:
			return "EXIT";
		case STARPU_MP_COMMAND_EXECUTE:
			return "EXECUTE";
		case STARPU_MP_COMMAND_EXECUTE_DETACHED:
			return "EXECUTE_DETACHED";
		case STARPU_MP_COMMAND_SINK_NBCORES:
			return "SINK_NBCORES";
		case STARPU_MP_COMMAND_LOOKUP:
			return "LOOKUP";
		case STARPU_MP_COMMAND_ALLOCATE:
			return "ALLOCATE";
		case STARPU_MP_COMMAND_FREE:
			return "FREE";
		case STARPU_MP_COMMAND_MAP:
			return "MAP";
		case STARPU_MP_COMMAND_UNMAP:
			return "UNMAP";
		case STARPU_MP_COMMAND_SYNC_WORKERS:
			return "SYNC_WORKERS";

		/* Note: synchronous send */
		case STARPU_MP_COMMAND_RECV_FROM_HOST:
			return "RECV_FROM_HOST";
		case STARPU_MP_COMMAND_SEND_TO_HOST:
			return "SEND_TO_HOST";
		case STARPU_MP_COMMAND_RECV_FROM_SINK:
			return "RECV_FROM_SINK";
		case STARPU_MP_COMMAND_SEND_TO_SINK:
			return "SEND_TO_SINK";

		/* Note: Asynchronous send */
		case STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC:
			return "RECV_FROM_HOST_ASYNC";
		case STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC:
			return "SEND_TO_HOST_ASYNC";
		case STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC:
			return "RECV_FROM_SINK_ASYNC";
		case STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC:
			return "SEND_TO_SINK_ASYNC";

		/* Synchronous answers from slave to master */
		case STARPU_MP_COMMAND_ERROR_EXECUTE:
			return "ERROR_EXECUTE";
		case STARPU_MP_COMMAND_ERROR_EXECUTE_DETACHED:
			return "ERROR_EXECUTE_DETACHED";
		case STARPU_MP_COMMAND_ANSWER_LOOKUP:
			return "ANSWER_LOOKUP";
		case STARPU_MP_COMMAND_ERROR_LOOKUP:
			return "ERROR_LOOKUP";
		case STARPU_MP_COMMAND_ANSWER_ALLOCATE:
			return "ANSWER_ALLOCATE";
		case STARPU_MP_COMMAND_ERROR_ALLOCATE:
			return "ERROR_ALLOCATE";
		case STARPU_MP_COMMAND_ANSWER_MAP:
			return "ANSWER_MAP";
		case STARPU_MP_COMMAND_ERROR_MAP:
			return "ERROR_MAP";
		case STARPU_MP_COMMAND_ANSWER_TRANSFER_COMPLETE:
			return "ANSWER_TRANSFER_COMPLETE";
		case STARPU_MP_COMMAND_ANSWER_SINK_NBCORES:
			return "ANSWER_SINK_NBCORES";
		case STARPU_MP_COMMAND_ANSWER_EXECUTION_SUBMITTED:
			return "ANSWER_EXECUTION_SUBMITTED";
		case STARPU_MP_COMMAND_ANSWER_EXECUTION_DETACHED_SUBMITTED:
			return "ANSWER_EXECUTION_DETACHED_SUBMITTED";

		/* Asynchronous notifications from slave to master */
		case STARPU_MP_COMMAND_NOTIF_RECV_FROM_HOST_ASYNC_COMPLETED:
			return "NOTIF_RECV_FROM_HOST_ASYNC_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_SEND_TO_HOST_ASYNC_COMPLETED:
			return "NOTIF_SEND_TO_HOST_ASYNC_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_RECV_FROM_SINK_ASYNC_COMPLETED:
			return "NOTIF_RECV_FROM_SINK_ASYNC_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_SEND_TO_SINK_ASYNC_COMPLETED:
			return "NOTIF_SEND_TO_SINK_ASYNC_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_EXECUTION_COMPLETED:
			return "NOTIF_EXECUTION_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_EXECUTION_DETACHED_COMPLETED:
			return "NOTIF_EXECUTION_DETACHED_COMPLETED";
		case STARPU_MP_COMMAND_NOTIF_PRE_EXECUTION:
			return "NOTIF_PRE_EXECUTION";

		default:
			return "<invalid command code>";
	}
}

const char *_starpu_mp_common_node_kind_to_string(const int kind)
{
	switch(kind)
	{
		case STARPU_NODE_MPI_SINK:
			return "MPI_SINK";
		case STARPU_NODE_MPI_SOURCE:
			return "MPI_SOURCE";
		case STARPU_NODE_TCPIP_SINK:
			return "TCPIP_SINK";
		case STARPU_NODE_TCPIP_SOURCE:
			return "TCPIP_SOURCE";
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
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		case STARPU_NODE_MPI_SOURCE:
		{
			/*
			  node->nb_mp_sinks =
			  node->devid =
			*/
			node->peer_id = (_starpu_mpi_common_get_src_node() <= peer_id ? peer_id+1 : peer_id);
			node->mp_connection.mpi_remote_nodeid = node->peer_id;

			node->init = _starpu_mpi_source_init;
			node->launch_workers = NULL;
			node->deinit = _starpu_mpi_source_deinit;
			/*     node->report_error = */

			node->mp_recv_is_ready = _starpu_mpi_common_recv_is_ready;
			node->mp_send = _starpu_mpi_common_mp_send;
			node->mp_recv = _starpu_mpi_common_mp_recv;
			node->nt_recv_is_ready = _starpu_mpi_common_notif_recv_is_ready;
			node->nt_send_is_ready = _starpu_mpi_common_notif_send_is_ready;
			node->mp_wait = NULL;
			node->mp_signal = NULL;
			node->nt_send = _starpu_mpi_common_nt_send;
			node->nt_recv = _starpu_mpi_common_nt_recv;
			node->dt_send = _starpu_mpi_common_send;
			node->dt_recv = _starpu_mpi_common_recv;
			node->dt_send_to_device = _starpu_mpi_common_send_to_device;
			node->dt_recv_from_device = _starpu_mpi_common_recv_from_device;

			node->get_kernel_from_job = _starpu_src_common_get_cpu_func_from_job;
			node->lookup = NULL;
			node->bind_thread = NULL;
			node->execute = NULL;
			node->allocate = NULL;
			node->free = NULL;
			node->map = NULL;
			node->unmap = NULL;
		}
		break;

		case STARPU_NODE_MPI_SINK:
		{
			/*
			  node->nb_mp_sinks =
			  node->devid =
			*/
			node->mp_connection.mpi_remote_nodeid = _starpu_mpi_common_get_src_node();

			node->init = _starpu_mpi_sink_init;
			node->launch_workers = _starpu_sink_launch_workers;
			node->deinit = _starpu_sink_deinit;
			/*    node->report_error =  */

			node->mp_recv_is_ready = _starpu_mpi_common_recv_is_ready;
			node->mp_send = _starpu_mpi_common_mp_send;
			node->mp_recv = _starpu_mpi_common_mp_recv;
			node->nt_recv_is_ready = _starpu_mpi_common_notif_recv_is_ready;
			node->nt_send_is_ready = _starpu_mpi_common_notif_send_is_ready;
			node->mp_wait = NULL;
			node->mp_signal = NULL;
			node->nt_send = _starpu_mpi_common_nt_send;
			node->nt_recv = _starpu_mpi_common_nt_recv;
			node->dt_send = _starpu_mpi_common_send;
			node->dt_recv = _starpu_mpi_common_recv;
			node->dt_send_to_device = _starpu_mpi_common_send_to_device;
			node->dt_recv_from_device = _starpu_mpi_common_recv_from_device;

			node->dt_test = _starpu_mpi_common_test_event;

			node->get_kernel_from_job = NULL;
			node->lookup = _starpu_sink_common_cpu_lookup;
			node->bind_thread = _starpu_mpi_sink_bind_thread;
			node->execute = _starpu_sink_common_execute;
			node->allocate = _starpu_sink_common_allocate;
			node->free = _starpu_sink_common_free;
			node->map = _starpu_sink_common_map;
			node->unmap = _starpu_sink_common_unmap;
		}
		break;
#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		case STARPU_NODE_TCPIP_SOURCE:
		{
			/*
			  node->nb_mp_sinks =
			  node->devid =
			*/
			node->peer_id = (0 <= peer_id ? peer_id+1 : peer_id);

			node->mp_connection.tcpip_mp_connection = &tcpip_sock[node->peer_id];

			node->init = _starpu_tcpip_source_init;
			node->launch_workers = NULL;
			node->deinit = _starpu_tcpip_source_deinit;
			/*     node->report_error = */

			node->mp_recv_is_ready = _starpu_tcpip_common_recv_is_ready;
			node->mp_send = _starpu_tcpip_common_mp_send;
			node->mp_recv = _starpu_tcpip_common_mp_recv;
			node->nt_recv_is_ready = _starpu_tcpip_common_notif_recv_is_ready;
			node->nt_send_is_ready = _starpu_tcpip_common_notif_send_is_ready;
			node->mp_wait = _starpu_tcpip_common_wait;
			node->mp_signal = _starpu_tcpip_common_signal;
			node->nt_send = _starpu_tcpip_common_nt_send;
			node->nt_recv = _starpu_tcpip_common_nt_recv;
			node->dt_send = _starpu_tcpip_common_send;
			node->dt_recv = _starpu_tcpip_common_recv;
			node->dt_send_to_device = _starpu_tcpip_common_send_to_device;
			node->dt_recv_from_device = _starpu_tcpip_common_recv_from_device;

			node->get_kernel_from_job = _starpu_src_common_get_cpu_func_from_job;
			node->lookup = NULL;
			node->bind_thread = NULL;
			node->execute = NULL;
			node->allocate = NULL;
			node->free = NULL;
			node->map = NULL;
			node->unmap = NULL;
		}
		break;

		case STARPU_NODE_TCPIP_SINK:
		{
			/*
			  node->nb_mp_sinks =
			  node->devid =
			*/
			node->mp_connection.tcpip_mp_connection = &tcpip_sock[0];

			node->init = _starpu_tcpip_sink_init;
			node->launch_workers = _starpu_sink_launch_workers;
			node->deinit = _starpu_sink_deinit;
			/*    node->report_error =  */

			node->mp_recv_is_ready = _starpu_tcpip_common_recv_is_ready;
			node->mp_send = _starpu_tcpip_common_mp_send;
			node->mp_recv = _starpu_tcpip_common_mp_recv;
			node->nt_recv_is_ready = _starpu_tcpip_common_notif_recv_is_ready;
			node->nt_send_is_ready = _starpu_tcpip_common_notif_send_is_ready;
			node->mp_wait = _starpu_tcpip_common_wait;
			node->mp_signal = _starpu_tcpip_common_signal;
			node->nt_send = _starpu_tcpip_common_nt_send;
			node->nt_recv = _starpu_tcpip_common_nt_recv;
			node->dt_send = _starpu_tcpip_common_send;
			node->dt_recv = _starpu_tcpip_common_recv;
			node->dt_send_to_device = _starpu_tcpip_common_send_to_device;
			node->dt_recv_from_device = _starpu_tcpip_common_recv_from_device;

			node->dt_test = _starpu_tcpip_common_test_event;

			node->get_kernel_from_job = NULL;
			node->lookup = _starpu_sink_common_cpu_lookup;
			node->bind_thread = _starpu_tcpip_sink_bind_thread;
			node->execute = _starpu_sink_common_execute;
			node->allocate = _starpu_sink_common_allocate;
			node->free = _starpu_sink_common_free;
			node->map = _starpu_sink_common_map;
			node->unmap = _starpu_sink_common_unmap;
		}
		break;
#endif /* STARPU_USE_TCPIP_MASTER_SLAVE */

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

	STARPU_PTHREAD_MUTEX_INIT(&node->connection_mutex, NULL);

	_starpu_mp_event_list_init(&node->event_list);
	_starpu_mp_event_list_init(&node->event_queue);

	/* If the node is a sink then we must initialize some field */
	if(node->kind == STARPU_NODE_MPI_SINK || node->kind == STARPU_NODE_TCPIP_SINK)
	{
		int i;
		STARPU_HG_DISABLE_CHECKING(node->is_running);
		node->is_running = 1;
		_STARPU_MALLOC(node->run_table, sizeof(struct mp_task *)*node->nb_cores);
		_STARPU_MALLOC(node->run_table_detached, sizeof(struct mp_task *)*node->nb_cores);
		_STARPU_MALLOC(node->sem_run_table, sizeof(sem_t)*node->nb_cores);

		for(i=0; i<node->nb_cores; i++)
		{
			node->run_table[i] = NULL;
			node->run_table_detached[i] = NULL;
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
	if(node->kind == STARPU_NODE_MPI_SINK || node->kind == STARPU_NODE_TCPIP_SINK)
	{
		int i;
		for(i=0; i<node->nb_cores; i++)
		{
			sem_destroy(&node->sem_run_table[i]);
		}

		free(node->run_table);
		free(node->run_table_detached);
		free(node->sem_run_table);

		STARPU_PTHREAD_MUTEX_DESTROY(&node->barrier_mutex);
		STARPU_PTHREAD_BARRIER_DESTROY(&node->init_completed_barrier);
	}

	free(node->buffer);
	free(node);
}

/* Send COMMAND to RECIPIENT, along with ARG if ARG_SIZE is non-zero */
static void __starpu_mp_common_send_command(const struct _starpu_mp_node *node, const enum _starpu_mp_command command, void *arg, int arg_size, int notif)
{
	STARPU_ASSERT_MSG(arg_size <= BUFFER_SIZE, "Too much data (%d) for the static buffer (%d), increase BUFFER_SIZE perhaps?", arg_size, BUFFER_SIZE);

	//printf("SEND %s: %d/%s - arg_size %d by %lu \n", notif?"NOTIF":"CMD", command, _starpu_mp_common_command_to_string(command), arg_size, starpu_pthread_self());

	/* MPI sizes are given through a int */
	int command_size = sizeof(enum _starpu_mp_command);
	int arg_size_size = sizeof(int);

	/* Let's copy the data into the command line buffer */
	memcpy(node->buffer, &command, command_size);
	memcpy((void*) ((uintptr_t)node->buffer + command_size), &arg_size, arg_size_size);

	if (!notif)
		node->mp_send(node, node->buffer, command_size + arg_size_size);
	else
		node->nt_send(node, node->buffer, command_size + arg_size_size);

	if (arg_size)
	{
		if (!notif)
			node->mp_send(node, arg, arg_size);
		else
			node->nt_send(node, arg, arg_size);
	}
}

/* Send COMMAND to RECIPIENT, along with ARG if ARG_SIZE is non-zero */
void _starpu_mp_common_send_command(const struct _starpu_mp_node *node, const enum _starpu_mp_command command, void *arg, int arg_size)
{
	__starpu_mp_common_send_command(node, command, arg, arg_size, 0);
}

/* Send NOTIF COMMAND to RECIPIENT, along with ARG if ARG_SIZE is non-zero */
void _starpu_nt_common_send_command(const struct _starpu_mp_node *node, const enum _starpu_mp_command command, void *arg, int arg_size)
{
	__starpu_mp_common_send_command(node, command, arg, arg_size, 1);
}

/* Return the command received from SENDER. In case SENDER sent an argument
 * beside the command, an address to a copy of this argument is returns in arg.
 * There is no need to free this address as it's not allocated at this time.
 * However, the data pointed by arg shouldn't be relied on after a new call to
 * STARPU_MP_COMMON_RECV_COMMAND as it might corrupt it.
 */
static enum _starpu_mp_command __starpu_mp_common_recv_command(const struct _starpu_mp_node *node, void **arg, int *arg_size, int notif)
{
	enum _starpu_mp_command command;

	/* MPI sizes are given through a int */
	int command_size = sizeof(enum _starpu_mp_command);
	int arg_size_size = sizeof(int);

	if (!notif)
		node->mp_recv(node, node->buffer, command_size + arg_size_size);
	else
		node->nt_recv(node, node->buffer, command_size + arg_size_size);

	command = *((enum _starpu_mp_command *) node->buffer);
	*arg_size = *((int *) ((uintptr_t)node->buffer + command_size));

	//printf("RECV %s : %d/%s - arg_size %d by %lu \n", notif?"NOTIF":"CMD", command, _starpu_mp_common_command_to_string(command), *arg_size, starpu_pthread_self());

	/* If there is no argument (ie. arg_size == 0),
	 * let's return the command right now */
	if (!(*arg_size))
	{
		*arg = NULL;
		return command;
	}

	STARPU_ASSERT(*arg_size <= BUFFER_SIZE);

	if (!notif)
		node->mp_recv(node, node->buffer, *arg_size);
	else
		node->nt_recv(node, node->buffer, *arg_size);

	*arg = node->buffer;

	return command;
}

/* Return the command received from SENDER*/
enum _starpu_mp_command _starpu_mp_common_recv_command(const struct _starpu_mp_node *node, void **arg, int *arg_size)
{
	return __starpu_mp_common_recv_command(node, arg, arg_size, 0);
}

/* Return the notif command received from SENDER*/
enum _starpu_mp_command _starpu_nt_common_recv_command(const struct _starpu_mp_node *node, void **arg, int *arg_size)
{
	return __starpu_mp_common_recv_command(node, arg, arg_size, 1);
}

void _starpu_sink_deinit(struct _starpu_mp_node *node)
{
	int i;
	node->is_running = 0;
	for(i=0; i<node->nb_cores; i++)
	{
		sem_post(&node->sem_run_table[i]);
		STARPU_PTHREAD_JOIN(((starpu_pthread_t *)node->thread_table)[i],NULL);
	}
	free(node->thread_table);
}

void _starpu_sink_launch_workers(struct _starpu_mp_node *node)
{
	//TODO
	int i;
	struct arg_sink_thread * arg;
	cpu_set_t cpuset;
	starpu_pthread_attr_t attr;
	starpu_pthread_t thread;

	for(i=0; i < node->nb_cores; i++)
	{
		int ret;

		ret = starpu_pthread_attr_init(&attr);
		STARPU_ASSERT(ret == 0);

#if defined(HAVE_PTHREAD_SETAFFINITY_NP) && defined(__linux__)
		//init the set
		CPU_ZERO(&cpuset);
		CPU_SET(i,&cpuset);

		int nobind = starpu_getenv_number("STARPU_WORKERS_NOBIND");

		if (nobind <= 0)
		{
			ret = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
			STARPU_ASSERT(ret == 0);
		}
#else
#warning no CPU binding support
#endif

		/*prepare the argument for the thread*/
		_STARPU_MALLOC(arg, sizeof(struct arg_sink_thread));
		arg->coreid = i;
		arg->node = node;

		STARPU_PTHREAD_CREATE(&thread, &attr, _starpu_sink_thread, arg);
		starpu_pthread_attr_destroy(&attr);
		((starpu_pthread_t *)node->thread_table)[i] = thread;

	}
}
