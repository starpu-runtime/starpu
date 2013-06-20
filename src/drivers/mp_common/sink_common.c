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


#include <dlfcn.h>

#include <starpu.h>
#include <common/config.h>
#include <common/utils.h>
#include <drivers/mp_common/mp_common.h>
#include <datawizard/interfaces/data_interface.h>

#ifdef STARPU_USE_MIC
#include <common/COISysInfo_common.h>
#endif

#include "sink_common.h"

#include "task_fifo.h"

/* Return the sink kind of the running process, based on the value of the
 * STARPU_SINK environment variable.
 * If there is no valid value retrieved, return STARPU_INVALID_KIND
 */
static enum _starpu_mp_node_kind _starpu_sink_common_get_kind(void)
{
	/* Environment varible STARPU_SINK must be defined when running on sink
	 * side : let's use it to get the kind of node we're running on */
	char *node_kind = getenv("STARPU_SINK");
	STARPU_ASSERT(node_kind);

	if (!strcmp(node_kind, "STARPU_MIC"))
		return STARPU_MIC_SINK;
	else if (!strcmp(node_kind, "STARPU_SCC"))
		return STARPU_SCC_SINK;
	else if (!strcmp(node_kind, "STARPU_MPI"))
		return STARPU_MPI_SINK;
	else
		return STARPU_INVALID_KIND;
}

void
_starpu_sink_nbcores (const struct _starpu_mp_node *node)
{
	// Process packet received from `_starpu_src_common_sink_cores'.
	int nbcores = 1;

#ifdef STARPU_USE_MIC
	// XXX I currently only support MIC for now.
	if (STARPU_MIC_SINK == _starpu_sink_common_get_kind ())
		nbcores = COISysGetCoreCount();
#endif

	_starpu_mp_common_send_command (node, STARPU_ANSWER_SINK_NBCORES,
					&nbcores, sizeof (int));
}



static void _starpu_sink_common_lookup(const struct _starpu_mp_node *node,
				       char *func_name)
{
	void (*func)(void);
	void *dl_handle = dlopen(NULL, RTLD_NOW);
	func = dlsym(dl_handle, func_name);
	
	//_STARPU_DEBUG("Looked up %s, got %p\n", func_name, func);

	/* If we couldn't find the function, let's send an error to the host.
	 * The user probably made a mistake in the name */
	if (func)
		_starpu_mp_common_send_command(node, STARPU_ANSWER_LOOKUP,
					       &func, sizeof(func));
	else
		_starpu_mp_common_send_command(node, STARPU_ERROR_LOOKUP,
					       NULL, 0);
}

void _starpu_sink_common_allocate(const struct _starpu_mp_node *mp_node,
				  void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(size_t));

	void *addr = malloc(*(size_t *)(arg));

	/* If the allocation fail, let's send an error to the host.
	 */
	if (addr)
		_starpu_mp_common_send_command(mp_node, STARPU_ANSWER_ALLOCATE,
					       &addr, sizeof(addr));
	else
		_starpu_mp_common_send_command(mp_node, STARPU_ERROR_ALLOCATE,
					       NULL, 0);
}

void _starpu_sink_common_free(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED,
			      void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(void *));

	free(*(void **)(arg));
}

static void _starpu_sink_common_copy_from_host(const struct _starpu_mp_node *mp_node,
					       void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	mp_node->dt_recv(mp_node, cmd->addr, cmd->size);
}

static void _starpu_sink_common_copy_to_host(const struct _starpu_mp_node *mp_node,
					     void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command));

	struct _starpu_mp_transfer_command *cmd = (struct _starpu_mp_transfer_command *)arg;

	mp_node->dt_send(mp_node, cmd->addr, cmd->size);
}

static void _starpu_sink_common_copy_from_sink(const struct _starpu_mp_node *mp_node,
					       void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command_to_device));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	mp_node->dt_recv_from_device(mp_node, cmd->devid, cmd->addr, cmd->size);

	_starpu_mp_common_send_command(mp_node, STARPU_TRANSFER_COMPLETE, NULL, 0);
}

static void _starpu_sink_common_copy_to_sink(const struct _starpu_mp_node *mp_node,
					     void *arg, int arg_size)
{
	STARPU_ASSERT(arg_size == sizeof(struct _starpu_mp_transfer_command_to_device));

	struct _starpu_mp_transfer_command_to_device *cmd = (struct _starpu_mp_transfer_command_to_device *)arg;

	mp_node->dt_send_to_device(mp_node, cmd->devid, cmd->addr, cmd->size);
}

/* Function looping on the sink, waiting for tasks to execute.
 * If the caller is the host, don't do anything.
 */

void _starpu_sink_common_worker(void)
{
	struct _starpu_mp_node *node = NULL;
	enum _starpu_mp_command command = STARPU_EXIT;
	int arg_size = 0;
	void *arg = NULL;
	int exit_starpu = 0;
	enum _starpu_mp_node_kind node_kind = _starpu_sink_common_get_kind();

	if (node_kind == STARPU_INVALID_KIND)
		_STARPU_ERROR("No valid sink kind retrieved, use the"
			      "STARPU_SINK environment variable to specify"
			      "this\n");

	/* Create and initialize the node */
	node = _starpu_mp_common_node_create(node_kind, -1);
	
	while (!exit_starpu)
	{
		if(node->mp_recv_is_ready(node))
		{	
			command = _starpu_mp_common_recv_command(node, &arg, &arg_size);
			switch(command)
			{
			case STARPU_EXIT:
				exit_starpu = 1;
				break;
			case STARPU_EXECUTE:
				node->execute(node, arg, arg_size);
				break;
			case STARPU_SINK_NBCORES:
				node->nbcores (node);
				break;
			case STARPU_LOOKUP:
				_starpu_sink_common_lookup(node, (char *) arg);
				break;

			case STARPU_ALLOCATE:
				node->allocate(node, arg, arg_size);
				break;

			case STARPU_FREE:
				node->free(node, arg, arg_size);
				break;

			case STARPU_RECV_FROM_HOST:
				_starpu_sink_common_copy_from_host(node, arg, arg_size);
				break;

			case STARPU_SEND_TO_HOST:
				_starpu_sink_common_copy_to_host(node, arg, arg_size);
				break;

			case STARPU_RECV_FROM_SINK:
				_starpu_sink_common_copy_from_sink(node, arg, arg_size);
				break;

			case STARPU_SEND_TO_SINK:
				_starpu_sink_common_copy_to_sink(node, arg, arg_size);
				break;

			default:
				printf("Oops, command %x unrecognized\n", command);
			}
		}

		if(!task_fifo_is_empty(&(node->dead_queue)))
		{
			struct task * task = node->dead_queue.first;
			//_STARPU_DEBUG("telling host that we have finished the task %p sur %d.\n", task->kernel, task->coreid);
			_starpu_mp_common_send_command(task->node, STARPU_EXECUTION_COMPLETED,
						       &(task->coreid), sizeof(task->coreid));
			task_fifo_pop(&(node->dead_queue));
			free(task);
		}
	}

	/* Deinitialize the node and release it */
	_starpu_mp_common_node_destroy(node);

	exit(0);
}



static void* _starpu_sink_thread(void * thread_arg)
{
	struct task *arg = (struct task *)thread_arg;
  
	//execute the task
	arg->kernel(arg->interfaces,arg->cl_arg);

	//append the finished task to the dead queue
	task_fifo_append(&(arg->node->dead_queue),arg);
	pthread_exit(NULL);
}

static void _starpu_sink_execute_thread(struct task *arg)
{
	pthread_t thread;
	cpu_set_t cpuset;
	int ret;
  
	//create the tread
	ret = pthread_create(&thread, NULL, _starpu_sink_thread, arg);
	STARPU_ASSERT(ret == 0);
  
	//bind the thread on the core coreid
	arg->node->bind_thread(arg->node, &cpuset, arg->coreid, &thread);
}


/* Receive paquet from _starpu_src_common_execute_kernel in the form below :
 * [Function pointer on sink, number of interfaces, interfaces
 * (union _starpu_interface), cl_arg]
 * Then call the function given, passing as argument an array containing the
 * addresses of the received interfaces
 */

void _starpu_sink_common_execute(const struct _starpu_mp_node *node,
				 void *arg, int arg_size)
{
	unsigned id = 0;
	unsigned nb_interfaces;

	void *arg_ptr = arg;
	struct task *thread_arg = malloc(sizeof(struct task));
	
	thread_arg->node = node;

	thread_arg->kernel = *(void(**)(void **, void *)) arg_ptr;
	arg_ptr += sizeof(thread_arg->kernel);

	thread_arg->coreid = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(thread_arg->coreid);

	nb_interfaces = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(nb_interfaces);

	/* The function needs an array pointing to each interface it needs
	 * during execution. As in sink-side there is no mean to know which
	 * kind of interface to expect, the array is composed of unions of
	 * interfaces, thus we expect the same size anyway */
	for (id = 0; id < nb_interfaces; id++)
	{
		thread_arg->interfaces[id] = arg_ptr;
		arg_ptr += sizeof(union _starpu_interface);
	}

	/* Was cl_arg sent ? */
	if (arg_size > arg_ptr - arg)
		thread_arg->cl_arg = arg_ptr;
	else
		thread_arg->cl_arg = NULL;

	//_STARPU_DEBUG("telling host that we have submitted the task %p.\n", thread_arg->kernel);
	_starpu_mp_common_send_command(node, STARPU_EXECUTION_SUBMITTED,
				       NULL, 0);

	//_STARPU_DEBUG("executing the task %p\n", thread_arg->kernel);
	_starpu_sink_execute_thread(thread_arg);	
}
