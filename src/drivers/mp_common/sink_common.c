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
#include <common/barrier.h>
#ifdef STARPU_USE_MIC
#include <common/COISysInfo_common.h>
#endif

#include "sink_common.h"


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
     	_starpu_mp_common_send_command (node, STARPU_ANSWER_SINK_NBCORES,
					&node->nb_cores, sizeof (int));
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

		pthread_mutex_lock(&node->dead_queue_mutex);
		if(!mp_task_list_empty(node->dead_queue))
		{
			struct mp_task * task = mp_task_list_pop_back(node->dead_queue);
			pthread_mutex_unlock(&node->dead_queue_mutex);
			//_STARPU_DEBUG("telling host that we have finished the task %p sur %d.\n", task->kernel, task->coreid);
			_starpu_mp_common_send_command(node, STARPU_EXECUTION_COMPLETED,
						       &(task->coreid), sizeof(task->coreid));
			mp_task_delete(task);
		}
		else
		{
			pthread_mutex_unlock(&node->dead_queue_mutex);
		}
	}

	/* Deinitialize the node and release it */
	_starpu_mp_common_node_destroy(node);

	exit(0);
}




void* _starpu_sink_thread(void * thread_arg)
{

	struct mp_task **task = ((struct arg_sink_thread *)thread_arg)->task;
	struct mp_task * task_tmp;
	struct _starpu_mp_node *node = ((struct arg_sink_thread *)thread_arg)->node;
	pthread_mutex_t * mutex = ((struct arg_sink_thread *)thread_arg)->mutex;
	int coreid =((struct arg_sink_thread *)thread_arg)->coreid;
	free(thread_arg);
	while(1)
	{
		pthread_mutex_lock(mutex);
		if((*task) != NULL)
		{
			task_tmp = (*task);
			if(task_tmp->is_parallel_task)
			{
				_STARPU_DEBUG("BARRIER WAIT\n");
				STARPU_PTHREAD_BARRIER_WAIT(task_tmp->barrier);
				_STARPU_DEBUG("BARRIER JUMP\n");
			}
			if(task_tmp->type == STARPU_FORKJOIN && task_tmp->is_parallel_task)
				node->bind_thread(node, coreid, task_tmp->combined_worker, task_tmp->combined_worker_size);

			//execute the task
			task_tmp->kernel(task_tmp->interfaces,task_tmp->cl_arg);


			if(task_tmp->type == STARPU_FORKJOIN && task_tmp->is_parallel_task)
				node->bind_thread(node, coreid, &coreid, 1);

			(*task) = NULL;

			//append the finished task to the dead queue
			pthread_mutex_lock(&node->dead_queue_mutex);
			mp_task_list_push_front(node->dead_queue,task_tmp);
			pthread_mutex_unlock(&node->dead_queue_mutex);

		}
	}
	pthread_exit(NULL);
}

static void _starpu_sink_common_execute_thread(struct _starpu_mp_node *node, struct mp_task *task)
{
	//add the task to the spesific thread
	node->run_table[task->coreid] = task;
	//unlock the mutex
	pthread_mutex_unlock(&node->mutex_run_table[task->coreid]);
}

/**/
_starpu_pthread_barrier_t * _starpu_sink_common_get_barrier(struct _starpu_mp_node * node, int cb_workerid, int cb_workersize)
{
	struct mp_barrier * b = NULL;
	pthread_mutex_lock(&node->barrier_mutex);
	if(!mp_barrier_list_empty(node->barrier_list))
	{
		for(b = mp_barrier_list_begin(node->barrier_list); 
				b != mp_barrier_list_end(node->barrier_list) && b->id != cb_workerid; 
				b = mp_barrier_list_next(b));

		if(b->id == cb_workerid)
		{
			pthread_mutex_unlock(&node->barrier_mutex);
			return &b->barrier;
		}
	}
       	b = mp_barrier_new();
	b->id = cb_workerid;

	STARPU_PTHREAD_BARRIER_INIT(&b->barrier,NULL,cb_workersize);
	mp_barrier_list_push_back(node->barrier_list,b);
	pthread_mutex_unlock(&node->barrier_mutex);
	return &b->barrier;
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
	struct mp_task *task = malloc(sizeof(struct mp_task));
	
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

		task->combined_worker_size = *(int *) arg_ptr;
		arg_ptr += sizeof(task->combined_worker_size);
	
		for (id = 0; id < task->combined_worker_size; id++)
		{
			
			task->combined_worker[id] = *(int*) arg_ptr;
			arg_ptr += sizeof(task->combined_worker[id]);
		}
		
		task->barrier = _starpu_sink_common_get_barrier(node,task->combined_workerid,task->combined_worker_size);
	}

	task->coreid = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(task->coreid);

	nb_interfaces = *(unsigned *) arg_ptr;
	arg_ptr += sizeof(nb_interfaces);

	/* The function needs an array pointing to each interface it needs
	 * during execution. As in sink-side there is no mean to know which
	 * kind of interface to expect, the array is composed of unions of
	 * interfaces, thus we expect the same size anyway */
	for (id = 0; id < nb_interfaces; id++)
	{
		task->interfaces[id] = arg_ptr;
		arg_ptr += sizeof(union _starpu_interface);
	}

	/* Was cl_arg sent ? */
	if (arg_size > arg_ptr - arg)
		task->cl_arg = arg_ptr;
	else
		task->cl_arg = NULL;

	
	//_STARPU_DEBUG("telling host that we have submitted the task %p.\n", task->kernel);
	_starpu_mp_common_send_command(node, STARPU_EXECUTION_SUBMITTED,
				       NULL, 0);

	//_STARPU_DEBUG("executing the task %p\n", task->kernel);
	_starpu_sink_common_execute_thread(node, task);	
}
