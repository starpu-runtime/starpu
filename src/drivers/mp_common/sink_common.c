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

#include <starpu.h>
#include <dlfcn.h>
#include <common/config.h>
#include <common/utils.h>
#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mp_common/sink_common.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <drivers/tcpip/driver_tcpip_common.h>
#include <datawizard/interfaces/data_interface.h>
#include <common/barrier.h>
#include <core/workers.h>
#include <common/barrier_counter.h>

#include "sink_common.h"

/* Return the sink kind of the running process, based on the value of the
 * STARPU_SINK environment variable.
 * If there is no valid value retrieved, return STARPU_INVALID_KIND
 */
static enum _starpu_mp_node_kind _starpu_sink_common_get_kind(void)
{
	/* Environment variable STARPU_SINK must be defined when running on sink
	 * side : let's use it to get the kind of node we're running on */
	char *node_kind = starpu_getenv("STARPU_SINK");
	STARPU_ASSERT(node_kind);

	if (!strcmp(node_kind, "STARPU_MPI_MS"))
		return STARPU_NODE_MPI_SINK;
	else if (!strcmp(node_kind, "STARPU_TCPIP_MS"))
		return STARPU_NODE_TCPIP_SINK;
	else
		return STARPU_NODE_INVALID_KIND;
}

/*
  Send to host the number of cores of the sink device
*/
static void _starpu_sink_common_get_nb_cores(struct _starpu_mp_node *node)
{
	// Process packet received from `_starpu_src_common_sink_cores'.
	_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_ANSWER_SINK_NBCORES, &node->nb_cores, sizeof(int));
}

/* Send to host the address of the function given in parameter
 */
static void _starpu_sink_common_lookup(const struct _starpu_mp_node *node, char *func_name)
{
	void (*func)(void);
	func = node->lookup(node,func_name);

	//_STARPU_DEBUG("Looked up %s, got %p\n", func_name, func);

	/* If we couldn't find the function, let's send an error to the host.
	 * The user probably made a mistake in the name */
	if (func)
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_ANSWER_LOOKUP, &func, sizeof(func));
	else
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_ERROR_LOOKUP, NULL, 0);
}

/* CPU version of sink lookup */
void (*_starpu_sink_common_cpu_lookup(const struct _starpu_mp_node * node STARPU_ATTRIBUTE_UNUSED, char* func_name))(void)
{
#ifdef RTLD_DEFAULT
	return dlsym(RTLD_DEFAULT, func_name);
#else
	void *dl_handle = dlopen(NULL, RTLD_NOW);
	return dlsym(dl_handle, func_name);
#endif
}

/* Allocate a memory space and send the address of this space to the host
 */
void _starpu_sink_common_allocate(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(size_t));

	void *addr;
	_STARPU_MALLOC(addr, *(size_t *)(arg));

	/* If the allocation fail, let's send an error to the host.
	 */
	if (addr)
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ANSWER_ALLOCATE, &addr, sizeof(addr));
	else
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ERROR_ALLOCATE, NULL, 0);
}

void _starpu_sink_common_free(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(void *));

	free(*(void **)(arg));
}

/* Map a memory space and send the address of this space to the host
 */
void _starpu_sink_common_map(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT((unsigned int)arg_size >= sizeof(struct _starpu_mp_transfer_map_command));

	struct _starpu_mp_transfer_map_command *map_cmd = (struct _starpu_mp_transfer_map_command *)arg;

	void *map_addr = _starpu_sink_map(map_cmd->fd_name, map_cmd->offset, map_cmd->size);

	/* If mapping fail, let's send an error to the host.
	 */
	if (map_addr)
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ANSWER_MAP, &map_addr, sizeof(map_addr));
	else
		_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ERROR_MAP, NULL, 0);
}

void _starpu_sink_common_unmap(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_unmap_command));

	struct _starpu_mp_transfer_unmap_command *unmap_cmd = (struct _starpu_mp_transfer_unmap_command *)arg;

	_starpu_sink_unmap(unmap_cmd->addr, unmap_cmd->size);
}

static void _starpu_sink_common_copy_from_host_sync(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	mp_node->dt_recv(mp_node, cmd->addr, cmd->size, NULL);
}

static void _starpu_sink_common_copy_from_host_async(struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	/* For asynchronous transfers, we store events to test them later when they are finished */
	struct _starpu_mp_event * sink_event = _starpu_mp_event_new();
	/* Save the command to send */
	sink_event->answer_cmd = STARPU_MP_COMMAND_NOTIF_RECV_FROM_HOST_ASYNC_COMPLETED;
	sink_event->remote_event = cmd->event;

	/* Set the sender (host) ready because we don't want to wait its ack */
	struct _starpu_async_channel * async_channel = &sink_event->event;
	async_channel->node_ops = NULL;
	async_channel->starpu_mp_common_finished_sender = -1;
	async_channel->starpu_mp_common_finished_receiver = 0;
	async_channel->polling_node_receiver = NULL;
	async_channel->polling_node_sender = NULL;

	mp_node->dt_recv(mp_node, cmd->addr, cmd->size, &sink_event->event);
	/* Push event on the list */
	_starpu_mp_event_list_push_back(&mp_node->event_list, sink_event);
}

static void _starpu_sink_common_copy_to_host_sync(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	/* Save values before sending command to prevent the overwriting */
	size_t size = cmd->size;
	void * addr = cmd->addr;

	_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_SEND_TO_HOST, NULL, 0);

	mp_node->dt_send(mp_node, addr, size, NULL);
}

static void _starpu_sink_common_copy_to_host_async(struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	/* For asynchronous transfers, we need to say dt_send that we are in async mode
	 * but we don't push event on list because we don't need to know if it's finished
	 */
	struct _starpu_mp_event * sink_event = _starpu_mp_event_new();
	/* Save the command to send */
	sink_event->answer_cmd = STARPU_MP_COMMAND_NOTIF_SEND_TO_HOST_ASYNC_COMPLETED;
	sink_event->remote_event = cmd->event;

	/* Set the receiver (host) ready because we don't want to wait its ack */
	struct _starpu_async_channel * async_channel = &sink_event->event;
	async_channel->node_ops = NULL;
	async_channel->starpu_mp_common_finished_sender = 0;
	async_channel->starpu_mp_common_finished_receiver = -1;
	async_channel->polling_node_receiver = NULL;
	async_channel->polling_node_sender = NULL;

	mp_node->dt_send(mp_node, cmd->addr, cmd->size, &sink_event->event);
	/* Push event on the list */
	_starpu_mp_event_list_push_back(&mp_node->event_list, sink_event);
}

static void _starpu_sink_common_copy_from_sink_sync(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == offsetof(struct _starpu_mp_transfer_command_to_device, end));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	mp_node->dt_recv_from_device(mp_node, cmd->devid, cmd->addr, cmd->size, NULL);
	_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ANSWER_TRANSFER_COMPLETE, NULL, 0);
}

static void _starpu_sink_common_copy_from_sink_async(struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == offsetof(struct _starpu_mp_transfer_command_to_device, end));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	/* For asynchronous transfers, we store events to test them later when they are finished
	*/
	struct _starpu_mp_event * sink_event = _starpu_mp_event_new();
	/* Save the command to send */
	sink_event->answer_cmd = STARPU_MP_COMMAND_NOTIF_RECV_FROM_SINK_ASYNC_COMPLETED;
	sink_event->remote_event = cmd->event;

	/* Set the sender ready because we don't want to wait its ack */
	struct _starpu_async_channel * async_channel = &sink_event->event;
	async_channel->node_ops = NULL;
	async_channel->starpu_mp_common_finished_sender = -1;
	async_channel->starpu_mp_common_finished_receiver = 0;
	async_channel->polling_node_receiver = NULL;
	async_channel->polling_node_sender = NULL;

	mp_node->dt_recv_from_device(mp_node, cmd->devid, cmd->addr, cmd->size, &sink_event->event);
	/* Push event on the list */
	_starpu_mp_event_list_push_back(&mp_node->event_list, sink_event);
}

static void _starpu_sink_common_copy_to_sink_sync(const struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == offsetof(struct _starpu_mp_transfer_command_to_device, end));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	mp_node->dt_send_to_device(mp_node, cmd->devid, cmd->addr, cmd->size, NULL);
}

static void _starpu_sink_common_copy_to_sink_async(struct _starpu_mp_node *mp_node, void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == offsetof(struct _starpu_mp_transfer_command_to_device, end));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	/* For asynchronous transfers, we need to say dt_send that we are in async mode
	 * but we don't push event on list because we don't need to know if it's finished
	 */
	struct _starpu_mp_event * sink_event = _starpu_mp_event_new();
	/* Save the command to send */
	sink_event->answer_cmd = STARPU_MP_COMMAND_NOTIF_SEND_TO_SINK_ASYNC_COMPLETED;
	sink_event->remote_event = cmd->event;

	/* Set the receiver ready because we don't want to wait its ack */
	struct _starpu_async_channel * async_channel = &sink_event->event;
	async_channel->node_ops = NULL;
	async_channel->starpu_mp_common_finished_sender = 0;
	async_channel->starpu_mp_common_finished_receiver = -1;
	async_channel->polling_node_receiver = NULL;
	async_channel->polling_node_sender = NULL;

	mp_node->dt_send_to_device(mp_node, cmd->devid, cmd->addr, cmd->size, &sink_event->event);

	/* Push event on the list */
	_starpu_mp_event_list_push_back(&mp_node->event_list, sink_event);
}

/* Receive workers and combined workers and store them into the struct config
 */
static void _starpu_sink_common_recv_workers(struct _starpu_mp_node * node, void *arg, int arg_size)
{
	/* Retrieve information from the message */
	STARPU_ASSERT(arg_size == (sizeof(int)*5));
	uintptr_t arg_ptr = (uintptr_t) arg;
	int i;

	int nworkers = *(int *)arg_ptr;
	arg_ptr += sizeof(nworkers);

	int worker_size = *(int *)arg_ptr;
	arg_ptr += sizeof(worker_size);

	int combined_worker_size = *(int *)arg_ptr;
	arg_ptr += sizeof(combined_worker_size);

	int baseworkerid = *(int *)arg_ptr;
	arg_ptr += sizeof(baseworkerid);

	/* Clear data we won't use */
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	for(i=0; i<config->topology.nworkers; i++)
	{
		free(config->workers[i].perf_arch.devices);
		config->workers[i].perf_arch.devices = NULL;
	}

	config->topology.nworkers = *(int *)arg_ptr;

	/* Retrieve workers */
	struct _starpu_worker * workers = &config->workers[baseworkerid];
	node->dt_recv(node,workers,worker_size, NULL);

	/* Update workers to have coherent field */
	for(i=0; i<nworkers; i++)
	{
		workers[i].config = config;
		STARPU_PTHREAD_MUTEX_INIT(&workers[i].mutex,NULL);
		STARPU_PTHREAD_MUTEX_DESTROY(&workers[i].mutex);

		STARPU_PTHREAD_COND_INIT(&workers[i].started_cond,NULL);
		STARPU_PTHREAD_COND_DESTROY(&workers[i].started_cond);

		STARPU_PTHREAD_COND_INIT(&workers[i].ready_cond,NULL);
		STARPU_PTHREAD_COND_DESTROY(&workers[i].ready_cond);

		STARPU_PTHREAD_MUTEX_INIT(&workers[i].sched_mutex,NULL);
		STARPU_PTHREAD_MUTEX_DESTROY(&workers[i].sched_mutex);

		STARPU_PTHREAD_COND_INIT(&workers[i].sched_cond,NULL);
		STARPU_PTHREAD_COND_DESTROY(&workers[i].sched_cond);

		workers[i].current_task = NULL;
		workers[i].set = NULL;
	}

	/* Retrieve combined workers */
	struct _starpu_combined_worker * combined_workers = config->combined_workers;
	node->dt_recv(node, combined_workers, combined_worker_size, NULL);

	node->baseworkerid = baseworkerid;
	STARPU_PTHREAD_BARRIER_WAIT(&node->init_completed_barrier);
}

/* Function looping on the sink, waiting for tasks to execute.
 * If the caller is the host, don't do anything.
 */
void _starpu_sink_common_worker(void)
{
	struct _starpu_mp_node *node = NULL;
	enum _starpu_mp_command command;
	int arg_size = 0;
	void *arg = NULL;
	int exit_starpu = 0;
	enum _starpu_mp_node_kind node_kind = _starpu_sink_common_get_kind();

	if (node_kind == STARPU_NODE_INVALID_KIND)
		_STARPU_ERROR("No valid sink kind retrieved, use the STARPU_SINK environment variable to specify this\n");

	/* Create and initialize the node */
	node = _starpu_mp_common_node_create(node_kind, -1);

	starpu_pthread_key_t worker_key;
	STARPU_PTHREAD_KEY_CREATE(&worker_key, NULL);

	while (!exit_starpu)
	{
		/* Wait send/recv is ready */
		if (node->mp_wait)
			node->mp_wait(node);

		/* If we have received a message */
		if(node->mp_recv_is_ready(node))
		{
			command = _starpu_mp_common_recv_command(node, &arg, &arg_size);
			switch(command)
			{
				case STARPU_MP_COMMAND_EXIT:
					exit_starpu = 1;
					break;
				case STARPU_MP_COMMAND_EXECUTE_DETACHED:
				case STARPU_MP_COMMAND_EXECUTE:
					node->execute(node, arg, arg_size);
					break;
				case STARPU_MP_COMMAND_SINK_NBCORES:
					_starpu_sink_common_get_nb_cores(node);
					break;
				case STARPU_MP_COMMAND_LOOKUP:
					_starpu_sink_common_lookup(node, (char *) arg);
					break;

				case STARPU_MP_COMMAND_ALLOCATE:
					node->allocate(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_FREE:
					node->free(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_MAP:
					node->map(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_UNMAP:
					node->unmap(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_RECV_FROM_HOST:
					_starpu_sink_common_copy_from_host_sync(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_SEND_TO_HOST:
					_starpu_sink_common_copy_to_host_sync(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_RECV_FROM_SINK:
					_starpu_sink_common_copy_from_sink_sync(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_SEND_TO_SINK:
					_starpu_sink_common_copy_to_sink_sync(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC:
					_starpu_sink_common_copy_from_host_async(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC:
					_starpu_sink_common_copy_to_host_async(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC:
					_starpu_sink_common_copy_from_sink_async(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC:
					_starpu_sink_common_copy_to_sink_async(node, arg, arg_size);
					break;

				case STARPU_MP_COMMAND_SYNC_WORKERS:
					_starpu_sink_common_recv_workers(node, arg, arg_size);
					break;
				default:
					_STARPU_MSG("Oops, command %x unrecognized\n", command);
			}
		}

		STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
		/* If the list is not empty and we can send a notification */
		while(!mp_message_list_empty(&node->message_queue) && node->nt_send_is_ready(node))
		{
			/* We pop a message and send it to the host */
			struct mp_message * message = mp_message_list_pop_back(&node->message_queue);
			STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);
			//_STARPU_DEBUG("telling host that we have finished the task %p sur %d.\n", task->kernel, task->coreid);
			STARPU_ASSERT(message->type >= STARPU_MP_COMMAND_NOTIF_FIRST && message->type <= STARPU_MP_COMMAND_NOTIF_LAST);
			_starpu_nt_common_send_command(node, message->type, message->buffer, message->size);
			free(message->buffer);
			mp_message_delete(message);
			STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);

		struct _starpu_mp_event * sink_event;
		struct _starpu_mp_event * sink_event_next;

		for (sink_event = _starpu_mp_event_list_begin(&node->event_list);
		     sink_event != _starpu_mp_event_list_end(&node->event_list);
		     sink_event = sink_event_next)
		{
			sink_event_next = _starpu_mp_event_list_next(sink_event);

			/*if event is completed move it into event queue*/
			if(node->dt_test(&sink_event->event))
			{
				_starpu_mp_event_list_erase(&node->event_list, sink_event);
				_starpu_mp_event_list_push_front(&node->event_queue, sink_event);
			}
		}

		/*if the list is not empty and we can send a notification*/
		while(!_starpu_mp_event_list_empty(&node->event_queue) && node->nt_send_is_ready(node))
		{
			struct _starpu_mp_event * sink_event_completed = _starpu_mp_event_list_pop_back(&node->event_queue);
			/* send ACK to host */
			STARPU_ASSERT(sink_event_completed->answer_cmd >= STARPU_MP_COMMAND_NOTIF_FIRST && sink_event_completed->answer_cmd <= STARPU_MP_COMMAND_NOTIF_LAST);
			_starpu_nt_common_send_command(node, sink_event_completed->answer_cmd, &sink_event_completed->remote_event, sizeof(sink_event_completed->remote_event));

			_starpu_mp_event_delete(sink_event_completed);
		}
	}

	STARPU_PTHREAD_KEY_DELETE(worker_key);

	/* Deinitialize the node and release it */
	_starpu_mp_common_node_destroy(node);

	starpu_perfmodel_free_sampling();
	_starpu_profiling_terminate();
	_starpu_perf_knob_exit();
	_starpu_perf_counter_exit();

	_starpu_destroy_machine_config(&_starpu_config, 1);

	free((char*) _starpu_config.conf.sched_policy_name);
	if (_starpu_config.conf.n_cuda_opengl_interoperability)
		free(_starpu_config.conf.cuda_opengl_interoperability);
	if (_starpu_config.conf.n_not_launched_drivers)
		free(_starpu_config.conf.not_launched_drivers);

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	_starpu_mpi_common_mp_deinit();
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	_starpu_tcpip_common_mp_deinit();
#endif

	exit(0);
}

/* Search for the mp_barrier correspondind to the specified combined worker
 * and create it if it doesn't exist
 */
static struct mp_barrier * _starpu_sink_common_get_barrier(struct _starpu_mp_node * node, int cb_workerid, int cb_workersize)
{
	struct mp_barrier * b = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&node->barrier_mutex);
	/* Search if the barrier already exist */
	for(b = mp_barrier_list_begin(&node->barrier_list);
	    b != mp_barrier_list_end(&node->barrier_list) && b->id != cb_workerid;
	    b = mp_barrier_list_next(b));

	/* If we found the barrier */
	if(b != NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->barrier_mutex);
		return b;
	}
	else
	{

		/* Else we create, initialize and add it to the list*/
		b = mp_barrier_new();
		b->id = cb_workerid;
		STARPU_PTHREAD_BARRIER_INIT(&b->before_work_barrier,NULL,cb_workersize);
		STARPU_PTHREAD_BARRIER_INIT(&b->after_work_barrier,NULL,cb_workersize);
		mp_barrier_list_push_back(&node->barrier_list,b);
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->barrier_mutex);
		return b;
	}
}

/* Erase for the mp_barrier correspondind to the specified combined worker
*/
static void _starpu_sink_common_erase_barrier(struct _starpu_mp_node * node, struct mp_barrier *barrier)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->barrier_mutex);
	mp_barrier_list_erase(&node->barrier_list,barrier);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->barrier_mutex);
}

/* Append the message given in parameter to the message list
 */
static void _starpu_sink_common_append_message(struct _starpu_mp_node *node, struct mp_message * message)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
	mp_message_list_push_front(&node->message_queue,message);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);
	/* Send the signal that message is in message_queue */
	if(node->mp_signal)
	{
		node->mp_signal(node);
	}
}

/* Append to the message list a "STARPU_PRE_EXECUTION" message
 */
static void _starpu_sink_common_pre_execution_message(struct _starpu_mp_node *node, struct mp_task *task)
{
	/* Init message to tell the sink that the execution has begun */
	struct mp_message * message = mp_message_new();
	message->type = STARPU_MP_COMMAND_NOTIF_PRE_EXECUTION;
	_STARPU_MALLOC(message->buffer, sizeof(int));
	*(int *) message->buffer = task->combined_workerid;
	message->size = sizeof(int);

	/* Append the message to the queue */
	_starpu_sink_common_append_message(node, message);
}

/* Append to the message list a "STARPU_EXECUTION_COMPLETED" message and cl_ret
 */
static void _starpu_sink_common_execution_completed_message(struct _starpu_mp_node *node, struct mp_task *task)
{
	/* Init message to tell the sink that the execution is completed */
	struct mp_message * message = mp_message_new();
	if (task->detached)
		message->type = STARPU_MP_COMMAND_NOTIF_EXECUTION_DETACHED_COMPLETED;
	else
		message->type = STARPU_MP_COMMAND_NOTIF_EXECUTION_COMPLETED;

	message->size = sizeof(int);

	/* If the user didn't give any cl_ret, there is no need to send it */
	 if (task->cl_ret)
	 {
		STARPU_ASSERT(task->cl_ret_size);
		message->size += task->cl_ret_size;
	 }

	_STARPU_MALLOC(message->buffer, message->size);

	*(int*) message->buffer = task->coreid;

	 if (task->cl_ret)
		memcpy(message->buffer+sizeof(int), task->cl_ret, task->cl_ret_size);

	/* Append the message to the queue */
	_starpu_sink_common_append_message(node, message);
}

/* Bind the thread which is running on the specified core to the combined worker */
static void _starpu_sink_common_bind_to_combined_worker(struct _starpu_mp_node *node, int coreid, struct _starpu_combined_worker * combined_worker)
{
	int i;
	int * bind_set;
	_STARPU_MALLOC(bind_set, sizeof(int)*combined_worker->worker_size);
	for(i=0;i<combined_worker->worker_size;i++)
		bind_set[i] = combined_worker->combined_workerid[i] - node->baseworkerid;
	node->bind_thread(node, coreid, bind_set, combined_worker->worker_size);
}

/* Get the current rank of the worker in the combined worker
 */
static int _starpu_sink_common_get_current_rank(int workerid, struct _starpu_combined_worker * combined_worker)
{
	int i;
	for(i=0; i<combined_worker->worker_size; i++)
		if(workerid == combined_worker->combined_workerid[i])
			return i;

	STARPU_ASSERT(0);
	return -1;
}

/* Execute the task
 */
static void _starpu_sink_common_execute_kernel(struct _starpu_mp_node *node, int coreid, struct _starpu_worker * worker, int detached)
{
	struct _starpu_combined_worker * combined_worker = NULL;
	struct mp_task* task;
	if (detached)
		task = node->run_table_detached[coreid];
	else
		task = node->run_table[coreid];

	/* If it's a parallel task */
	if(task->is_parallel_task)
	{
		combined_worker = _starpu_get_combined_worker_struct(task->combined_workerid);

		worker->current_rank = _starpu_sink_common_get_current_rank(worker->workerid, combined_worker);
		worker->combined_workerid = task->combined_workerid;
		worker->worker_size = combined_worker->worker_size;

		/* Synchronize with others threads of the combined worker*/
		STARPU_PTHREAD_BARRIER_WAIT(&task->mp_barrier->before_work_barrier);

		/* The first thread of the combined worker */
		if(worker->current_rank == 0)
		{
			/* tell the sink that the execution has begun */
			_starpu_sink_common_pre_execution_message(node,task);

			/* If the mode is FORKJOIN,
			 * the first thread binds himself
			 * on all core of the combined worker*/
			if(task->type == STARPU_FORKJOIN)
			{
				_starpu_sink_common_bind_to_combined_worker(node, coreid, combined_worker);
			}
		}
	}
	else
	{
		worker->current_rank = 0;
		worker->combined_workerid = 0;
		worker->worker_size = 1;
	}

	if(task->type != STARPU_FORKJOIN || worker->current_rank == 0)
	{
		if (_starpu_get_disable_kernels() <= 0)
		{
			struct starpu_task s_task;
			starpu_task_init(&s_task);

			/*copy cl_arg and cl_arg_size from mp_task into starpu_task*/
			s_task.cl_arg=task->cl_arg;
			s_task.cl_arg_size=task->cl_arg_size;

			_starpu_set_current_task(&s_task);
			/* execute the task */
			task->kernel(task->interfaces,task->cl_arg);
			_starpu_set_current_task(NULL);

			/*copy cl_ret and cl_ret_size from starpu_task into mp_task*/
			task->cl_ret=s_task.cl_ret;
			task->cl_ret_size=s_task.cl_ret_size;
		}
	}

	/* If it's a parallel task */
	if(task->is_parallel_task)
	{
		/* Synchronize with others threads of the combined worker*/
		STARPU_PTHREAD_BARRIER_WAIT(&task->mp_barrier->after_work_barrier);

		/* The first thread of the combined */
		if(worker->current_rank == 0)
		{
			/* Erase the barrier from the list */
			_starpu_sink_common_erase_barrier(node,task->mp_barrier);

			/* If the mode is FORKJOIN,
			 * the first thread rebinds himself on his own core */
			if(task->type == STARPU_FORKJOIN)
				node->bind_thread(node, coreid, &coreid, 1);

		}
	}

	if (detached)
		node->run_table_detached[coreid] = NULL;
	else
		node->run_table[coreid] = NULL;

	/* tell the sink that the execution is completed */
	_starpu_sink_common_execution_completed_message(node,task);

	/*free the task*/
	unsigned i;
	for (i = 0; i < task->nb_interfaces; i++)
	{
		struct starpu_data_interface_ops *ops = _starpu_data_interface_get_ops(task->ids[i]);
		if (ops->free_meta)
		{
			ops->free_meta(task->interfaces[i]);
		}
		free(task->interfaces[i]);
	}
	free(task->interfaces);
	free(task->ids);
	if (task->cl_arg != NULL)
		free(task->cl_arg);
	free(task);
}

/* The main function executed by the thread
 * thread_arg is a structure containing the information needed by the thread
 */
void* _starpu_sink_thread(void * thread_arg)
{
	/* Retrieve the information from the structure */
	struct _starpu_mp_node *node = ((struct arg_sink_thread *)thread_arg)->node;
	int coreid =((struct arg_sink_thread *)thread_arg)->coreid;
	/* free the structure */
	free(thread_arg);

	STARPU_PTHREAD_BARRIER_WAIT(&node->init_completed_barrier);

	struct _starpu_worker *worker = &_starpu_get_machine_config()->workers[node->baseworkerid + coreid];
	char *s;
	asprintf(&s, "slave %d core %d", node->baseworkerid, coreid);
	starpu_pthread_setname(s);
	free(s);

	node->bind_thread(node, coreid, &coreid, 1);

	_starpu_set_local_worker_key(worker);
	while(node->is_running)
	{
		/*Wait there is a task available */
		sem_wait(&node->sem_run_table[coreid]);

		STARPU_ASSERT((node->run_table_detached[coreid]!=NULL) || (node->run_table[coreid]!=NULL) || node->is_running==0);

		if (node->run_table_detached[coreid] != NULL)
			_starpu_sink_common_execute_kernel(node, coreid, worker, 1);
		else if (node->run_table[coreid] != NULL)
			_starpu_sink_common_execute_kernel(node, coreid, worker, 0);
		else
			STARPU_ASSERT(!node->is_running);

	}
	starpu_pthread_exit(NULL);
}

/* Add the task to the specific thread and wake him up
*/
static void _starpu_sink_common_execute_thread(struct _starpu_mp_node *node, struct mp_task *task)
{
	int detached = task->detached;
	/* Add the task to the specific thread */
	if (detached)
	{
		STARPU_ASSERT(!node->run_table_detached[task->coreid]);
		node->run_table_detached[task->coreid] = task;
	}
	else
	{
		STARPU_ASSERT(!node->run_table[task->coreid]);
		node->run_table[task->coreid] = task;
	}
	/* Unlock the mutex to wake up the thread which will execute the task */
	sem_post(&node->sem_run_table[task->coreid]);
}

/* Receive paquet from _starpu_src_common_execute_kernel in the form below :
 * [Function pointer on sink, number of interfaces, interfaces
 * (union _starpu_interface), cl_arg]
 * Then call the function given, passing as argument an array containing the
 * addresses of the received interfaces
 */

void _starpu_sink_common_execute(struct _starpu_mp_node *node, void *arg, int arg_size)
{
	unsigned i;

	uintptr_t arg_ptr = (uintptr_t) arg;
	struct mp_task *task;

	_STARPU_CALLOC(task, 1, sizeof(struct mp_task));
	task->kernel = *(void(**)(void **, void *)) arg_ptr;
	arg_ptr += sizeof(task->kernel);

	task->type = *(enum starpu_codelet_type *) arg_ptr;
	arg_ptr += sizeof(task->type);

	task->is_parallel_task = *(int *) arg_ptr;
	arg_ptr += sizeof(task->is_parallel_task);

	if(task->is_parallel_task)
	{
		task->combined_workerid= *(int *) arg_ptr;
		arg_ptr += sizeof(task->combined_workerid);

		task->mp_barrier = _starpu_sink_common_get_barrier(node,task->combined_workerid,_starpu_get_combined_worker_struct(task->combined_workerid)->worker_size);
	}

	task->coreid = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(task->coreid);

	task->nb_interfaces = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(task->nb_interfaces);

	task->detached = *(int *) arg_ptr;
	arg_ptr += sizeof(task->detached);

	_STARPU_MALLOC(task->interfaces, task->nb_interfaces * sizeof(*task->interfaces));
	_STARPU_MALLOC(task->ids, task->nb_interfaces * sizeof(*task->ids));

	/* The function needs an array pointing to each interface it needs
	 * during execution. The interface is first identified by its
	 * id, which will indicate if this is a basic interface or if
	 * it needs to be unpacked through unpack_meta
	 */
	for (i = 0; i < task->nb_interfaces; i++)
	{
		// first extract the interface id
		memcpy(&(task->ids[i]), (void *)arg_ptr, sizeof(task->ids[i]));
		arg_ptr += sizeof(task->ids[i]);

		// and then the interface
		struct starpu_data_interface_ops *ops = _starpu_data_interface_get_ops(task->ids[i]);
		if (ops->unpack_meta)
		{
			STARPU_ASSERT_MSG(ops->pack_meta, "unpack_meta defined without pack_meta for interface %d", task->ids[i]);
			starpu_ssize_t count;
			ops->unpack_meta(&task->interfaces[i], (void*) arg_ptr, &count);
			arg_ptr += count;
		}
		else
		{
			union _starpu_interface *interface;
			_STARPU_MALLOC(interface, sizeof(union _starpu_interface));
			memcpy(interface, (void*) arg_ptr, sizeof(union _starpu_interface));
			task->interfaces[i] = interface;
			arg_ptr += sizeof(union _starpu_interface);
		}
	}

	/* Was cl_arg sent ? */
	if (arg_size > arg_ptr - (uintptr_t) arg)
	{
		/* Copy cl_arg to prevent overwriting by an other task */
		unsigned cl_arg_size = arg_size - (arg_ptr - (uintptr_t) arg);
		_STARPU_MALLOC(task->cl_arg, cl_arg_size);
		memcpy(task->cl_arg, (void *) arg_ptr, cl_arg_size);
		task->cl_arg_size=cl_arg_size;
	}
	else
		task->cl_arg = NULL;

	//_STARPU_DEBUG("telling host that we have submitted the task %p.\n", task->kernel);
	if (task->detached)
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_ANSWER_EXECUTION_DETACHED_SUBMITTED, NULL, 0);
	else
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_ANSWER_EXECUTION_SUBMITTED, NULL, 0);

	//_STARPU_DEBUG("executing the task %p\n", task->kernel);
	_starpu_sink_common_execute_thread(node, task);
}
