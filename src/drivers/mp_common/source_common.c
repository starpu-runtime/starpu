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


#include <string.h>
#include <starpu.h>
#include <core/task.h>
#include <core/sched_policy.h>

#include <drivers/driver_common/driver_common.h>


#include <datawizard/coherency.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/interfaces/data_interface.h>
#include <drivers/mp_common/mp_common.h>

#if defined(STARPU_USE_MPI_MASTER_SLAVE) && !defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
struct starpu_save_thread_env
{
        struct starpu_task * current_task;
        struct _starpu_worker * current_worker;
        struct _starpu_worker_set * current_worker_set;
#ifdef STARPU_OPENMP
        struct starpu_omp_thread * current_omp_thread;
        struct starpu_omp_task * current_omp_task;
#endif
};

struct starpu_save_thread_env save_thread_env[STARPU_MAXMPIDEVS];
#endif

static unsigned mp_node_memory_node(struct _starpu_mp_node *node)
{
	return starpu_worker_get_memory_node(node->baseworkerid);
}

/* Finalize the execution of a task by a worker*/
static int _starpu_src_common_finalize_job (struct _starpu_job *j, struct _starpu_worker *worker)
{
	int profiling = starpu_profiling_status_get();
	_starpu_driver_end_job(worker, j, &worker->perf_arch, 0, profiling);

	int count = worker->current_rank;

	/* If it's a combined worker, we check if it's the last one of his combined */
	if(j->task_size > 1)
	{
		struct _starpu_combined_worker * cb_worker = _starpu_get_combined_worker_struct(worker->combined_workerid);
		(void) STARPU_ATOMIC_ADD(&j->after_work_busy_barrier, -1);

		STARPU_PTHREAD_MUTEX_LOCK(&cb_worker->count_mutex);
		count = cb_worker->count--;
		if(count == 0)
			cb_worker->count = cb_worker->worker_size - 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&cb_worker->count_mutex);
	}

	/* Finalize the execution */
	if(count == 0)
	{
		_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

		_starpu_push_task_output (j);

		_starpu_handle_job_termination(j);
	}
	return 0;
}

/* Complete the execution of the job */
static int _starpu_src_common_process_completed_job(struct _starpu_mp_node *node, struct _starpu_worker_set *workerset, void * arg, int arg_size, int stored)
{
	int coreid;

	STARPU_ASSERT(sizeof(coreid) == arg_size);

	coreid = *(int *) arg;

	struct _starpu_worker *worker = &workerset->workers[coreid];
	struct _starpu_job *j = _starpu_get_job_associated_to_task(worker->current_task);

	struct _starpu_worker * old_worker = _starpu_get_local_worker_key();

        /* if arg is not copied we release the mutex */
        if (!stored)
                STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);

	_starpu_set_local_worker_key(worker);
	_starpu_src_common_finalize_job (j, worker);
	_starpu_set_local_worker_key(old_worker);

	worker->current_task = NULL;

	return 0;
}

/* Tell the scheduler when the execution has begun */
static void _starpu_src_common_pre_exec(struct _starpu_mp_node *node, void * arg, int arg_size, int stored)
{
	int cb_workerid, i;
	STARPU_ASSERT(sizeof(cb_workerid) == arg_size);
	cb_workerid = *(int *) arg;
	struct _starpu_combined_worker *combined_worker = _starpu_get_combined_worker_struct(cb_workerid);

        /* if arg is not copied we release the mutex */
        if (!stored)
                STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);

	for(i=0; i < combined_worker->worker_size; i++)
	{
		struct _starpu_worker * worker = _starpu_get_worker_struct(combined_worker->combined_workerid[i]);
		_starpu_set_local_worker_key(worker);
		_starpu_sched_pre_exec_hook(worker->current_task);
	}
}

/* recv a message and handle asynchronous message
 * return 0 if the message has not been handle (it's certainly mean that it's a synchronous message)
 * return 1 if the message has been handle
 */
static int _starpu_src_common_handle_async(struct _starpu_mp_node *node, void * arg, int arg_size, enum _starpu_mp_command answer, int stored)
{
        struct _starpu_worker_set * worker_set = NULL;
        switch(answer)
        {
                case STARPU_MP_COMMAND_EXECUTION_COMPLETED:
		{
                        worker_set = _starpu_get_worker_struct(starpu_worker_get_id())->set;
                        _starpu_src_common_process_completed_job(node, worker_set, arg, arg_size, stored);
                        break;
		}
                case STARPU_MP_COMMAND_EXECUTION_DETACHED_COMPLETED:
		{
			_STARPU_ERROR("Detached execution completed should not arrive here... \n");
                        break;
		}
                case STARPU_MP_COMMAND_PRE_EXECUTION:
		{
                        _starpu_src_common_pre_exec(node, arg,arg_size, stored);
                        break;
		}
                case STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC_COMPLETED:
                case STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC_COMPLETED:
		{
			struct _starpu_async_channel * event = *((struct _starpu_async_channel **) arg);
			event->starpu_mp_common_finished_receiver--;
			if (!stored)
				STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
			break;
		}
                case STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC_COMPLETED:
                case STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC_COMPLETED:
		{
			struct _starpu_async_channel * event = *((struct _starpu_async_channel **) arg);
			event->starpu_mp_common_finished_sender--;
			if (!stored)
				STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
			break;
		}
		default:
                        return 0;
                        break;
        }
        return 1;
}

/* Handle all message which have been stored in the message_queue */
static void _starpu_src_common_handle_stored_async(struct _starpu_mp_node *node)
{
	int stopped_progress = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
	/* while the list is not empty */
	while(!mp_message_list_empty(&node->message_queue))
	{
		/* We pop a message and handle it */
		struct mp_message * message = mp_message_list_pop_back(&node->message_queue);
                /* Release mutex during handle */
		stopped_progress = 1;
		_STARPU_TRACE_END_PROGRESS(mp_node_memory_node(node));
                STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);
		_starpu_src_common_handle_async(node, message->buffer, message->size, message->type, 1);
		free(message->buffer);
		mp_message_delete(message);
                /* Take it again */
                STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);
	if (stopped_progress)
		_STARPU_TRACE_START_PROGRESS(mp_node_memory_node(node));
}

/* Store a message if is asynchronous
 * return 1 if the message has been stored
 * return 0 if the message is unknown or synchrone */
int _starpu_src_common_store_message(struct _starpu_mp_node *node, void * arg, int arg_size, enum _starpu_mp_command answer)
{
	switch(answer)
	{
		case STARPU_MP_COMMAND_EXECUTION_COMPLETED:
		case STARPU_MP_COMMAND_EXECUTION_DETACHED_COMPLETED:
		case STARPU_MP_COMMAND_PRE_EXECUTION:
		{
			struct mp_message *message = mp_message_new();
			message->type = answer;
			_STARPU_MALLOC(message->buffer, arg_size);
			memcpy(message->buffer, arg, arg_size);
			message->size = arg_size;

			STARPU_PTHREAD_MUTEX_LOCK(&node->message_queue_mutex);
			mp_message_list_push_front(&node->message_queue,message);
			STARPU_PTHREAD_MUTEX_UNLOCK(&node->message_queue_mutex);
			return 1;
		}
		/* For ASYNC commands don't store them, update event */
                case STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC_COMPLETED:
                case STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC_COMPLETED:
		{
                                struct _starpu_async_channel * event = *((struct _starpu_async_channel **) arg);
                                event->starpu_mp_common_finished_receiver--;
                                return 1;
		}
                case STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC_COMPLETED:
                case STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC_COMPLETED:
		{
                                struct _starpu_async_channel * event = *((struct _starpu_async_channel **) arg);
                                event->starpu_mp_common_finished_sender--;
                                return 1;
		}
                default:
                        return 0;
	}
}

/* Store all asynchronous messages and return when a synchronous message is received */
static enum _starpu_mp_command _starpu_src_common_wait_command_sync(struct _starpu_mp_node *node, void ** arg, int* arg_size)
{
	enum _starpu_mp_command answer;
	int is_sync = 0;
	while(!is_sync)
	{
		answer = _starpu_mp_common_recv_command(node, arg, arg_size);
		if(!_starpu_src_common_store_message(node,*arg,*arg_size,answer))
			is_sync=1;
	}
	return answer;
}

/* Handle a asynchrone message and return a error if a synchronous message is received */
static void _starpu_src_common_recv_async(struct _starpu_mp_node * node)
{
	enum _starpu_mp_command answer;
	void *arg;
	int arg_size;
	answer = _starpu_mp_common_recv_command(node, &arg, &arg_size);
	if(!_starpu_src_common_handle_async(node,arg,arg_size,answer, 0))
	{
		_STARPU_ERROR("incorrect command: unknown command or sync command");
	}
}

/* Handle all asynchrone message while a completed execution message from a specific worker has been receive */
 enum _starpu_mp_command _starpu_src_common_wait_completed_execution(struct _starpu_mp_node *node, int devid, void **arg, int * arg_size)
{
	enum _starpu_mp_command answer;

	int completed = 0;
	/*While the waited completed execution message has not been receive*/
	while(!completed)
	{
		answer = _starpu_mp_common_recv_command (node, arg, arg_size);

		if(answer == STARPU_MP_COMMAND_EXECUTION_DETACHED_COMPLETED)
		{
			int coreid;
			STARPU_ASSERT(sizeof(coreid) == *arg_size);
			coreid = *(int *) *arg;
			if(devid == coreid)
				completed = 1;
			else if(!_starpu_src_common_store_message(node, *arg, *arg_size, answer))
				/* We receive a unknown or asynchronous message  */
				STARPU_ASSERT(0);
		}
		else
		{
			if(!_starpu_src_common_store_message(node, *arg, *arg_size, answer))
				/* We receive a unknown or asynchronous message  */
				STARPU_ASSERT(0);
		}
	}
	return answer;
}

/* Send a request to the sink NODE for the number of cores on it. */
int _starpu_src_common_sink_nbcores (struct _starpu_mp_node *node, int *buf)
{
	enum _starpu_mp_command answer;
	void *arg;
	int arg_size = sizeof (int);

        STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);

	_starpu_mp_common_send_command (node, STARPU_MP_COMMAND_SINK_NBCORES, NULL, 0);

	answer = _starpu_mp_common_recv_command (node, &arg, &arg_size);

	STARPU_ASSERT(answer == STARPU_MP_COMMAND_ANSWER_SINK_NBCORES && arg_size == sizeof (int));

	memcpy (buf, arg, arg_size);

        STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);

	return 0;
}

/* Send a request to the sink linked to NODE for the pointer to the
 * function defined by FUNC_NAME.
 * In case of success, it returns 0 and FUNC_PTR contains the pointer ;
 * else it returns -ESPIPE if the function was not found.
 */
int _starpu_src_common_lookup(struct _starpu_mp_node *node, void (**func_ptr)(void), const char *func_name)
{
	enum _starpu_mp_command answer;
	void *arg;
	int arg_size;

	/* strlen ignore the terminating '\0' */
	arg_size = (strlen(func_name) + 1) * sizeof(char);

        STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);

	//_STARPU_DEBUG("Looking up %s\n", func_name);
	_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_LOOKUP, (void *) func_name,
			arg_size);

	answer = _starpu_src_common_wait_command_sync(node, (void **) &arg, &arg_size);

	if (answer == STARPU_MP_COMMAND_ERROR_LOOKUP)
	{
		_STARPU_DISP("Error looking up symbol %s\n", func_name);
                STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
		return -ESPIPE;
	}

	/* We have to be sure the device answered the right question and the
	 * answer has the right size */
	STARPU_ASSERT(answer == STARPU_MP_COMMAND_ANSWER_LOOKUP);
	STARPU_ASSERT(arg_size == sizeof(*func_ptr));

	memcpy(func_ptr, arg, arg_size);

        STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);

	//_STARPU_DEBUG("got %p\n", *func_ptr);

	return 0;
}

/* Send a message to the sink to execute a kernel.
 * The message sent has the form below :
 * [Function pointer on sink, number of interfaces, interfaces
 * (union _starpu_interface), cl_arg]
 */
/* Launch the execution of the function KERNEL points to on the sink linked
 * to NODE. Returns 0 in case of success, -EINVAL if kernel is an invalid
 * pointer.
 * Data interfaces in task are send to the sink.
 */
int _starpu_src_common_execute_kernel(struct _starpu_mp_node *node,
				      void (*kernel)(void), unsigned coreid,
				      enum starpu_codelet_type type,
				      int is_parallel_task, int cb_workerid,
				      starpu_data_handle_t *handles,
				      void **interfaces,
				      unsigned nb_interfaces,
				      void *cl_arg, size_t cl_arg_size, int detached)
{
	void *buffer, *arg =NULL;
	uintptr_t buffer_ptr;
	int buffer_size = 0, arg_size =0;
	unsigned i;

	buffer_size = sizeof(kernel) + sizeof(coreid) + sizeof(type) + sizeof(detached)
		+ sizeof(nb_interfaces) + nb_interfaces * sizeof(union _starpu_interface) + sizeof(is_parallel_task);

	/*if the task is parallel*/
	if(is_parallel_task)
	{
		buffer_size += sizeof(cb_workerid);
	}

	/* If the user didn't give any cl_arg, there is no need to send it */
	if (cl_arg)
	{
		STARPU_ASSERT(cl_arg_size);
		buffer_size += cl_arg_size;
	}

	/* We give to send_command a buffer we just allocated, which contains
	 * a pointer to the function (sink-side), core on which execute this
	 * function (sink-side), number of interfaces we send,
	 * an array of generic (union) interfaces and the value of cl_arg */
	_STARPU_MALLOC(buffer, buffer_size);
	buffer_ptr = (uintptr_t) buffer;

	*(void(**)(void)) buffer = kernel;
	buffer_ptr += sizeof(kernel);

	*(enum starpu_codelet_type *) buffer_ptr = type;
	buffer_ptr += sizeof(type);

	*(int *) buffer_ptr = is_parallel_task;
	buffer_ptr += sizeof(is_parallel_task);

	if(is_parallel_task)
	{
		*(int *) buffer_ptr = cb_workerid ;
		buffer_ptr += sizeof(cb_workerid);
	}

	*(unsigned *) buffer_ptr = coreid;
	buffer_ptr += sizeof(coreid);

	*(unsigned *) buffer_ptr = nb_interfaces;
	buffer_ptr += sizeof(nb_interfaces);

	*(int *) buffer_ptr = detached;
	buffer_ptr += sizeof(detached);

	/* Message-passing execution is a particular case as the codelet is
	 * executed on a sink with a different memory, whereas a codelet is
	 * executed on the host part for the other accelerators.
	 * Thus we need to send a copy of each interface on the MP device */

	for (i = 0; i < nb_interfaces; i++)
	{
		starpu_data_handle_t handle = handles[i];

		memcpy ((void*) buffer_ptr, interfaces[i], handle->ops->interface_size);
		/* The sink side has no mean to get the type of each
		 * interface, we use a union to make it generic and permit the
		 * sink to go through the array */
		buffer_ptr += sizeof(union _starpu_interface);
	}

	if (cl_arg)
		memcpy((void*) buffer_ptr, cl_arg, cl_arg_size);

        STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);

	if (detached)
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_EXECUTE_DETACHED, buffer, buffer_size);
	else
		_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_EXECUTE, buffer, buffer_size);

	enum _starpu_mp_command answer = _starpu_src_common_wait_command_sync(node, &arg, &arg_size);

        if (answer == STARPU_MP_COMMAND_ERROR_EXECUTE_DETACHED)
        {
                STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
                return -EINVAL;
        }

	if (detached)
		STARPU_ASSERT(answer == STARPU_MP_COMMAND_EXECUTION_DETACHED_SUBMITTED);
	else
		STARPU_ASSERT(answer == STARPU_MP_COMMAND_EXECUTION_SUBMITTED);

        STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);

	free(buffer);

	return 0;
}

/* Get the information and call the function to send to the sink a message to execute the task*/
static int _starpu_src_common_execute(struct _starpu_job *j, struct _starpu_worker *worker, struct _starpu_mp_node * node)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);

	void (*kernel)(void)  = node->get_kernel_from_job(node,j);

	_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);

	//_STARPU_DEBUG("\nworkerid:%d, rank:%d, type:%d,	cb_workerid:%d, task_size:%d\n\n",worker->devid,worker->current_rank,task->cl->type,j->combined_workerid,j->task_size);

	_starpu_src_common_execute_kernel(node, kernel, worker->subworkerid, task->cl->type,
					  (j->task_size > 1),
					  j->combined_workerid, STARPU_TASK_GET_HANDLES(task),
					  _STARPU_TASK_GET_INTERFACES(task), STARPU_TASK_GET_NBUFFERS(task),
					  task->cl_arg, task->cl_arg_size, 0);
	return 0;
}

/* Send a request to the sink linked to the MP_NODE to allocate SIZE bytes on
 * the sink.
 * In case of success, it returns 0 and *ADDR contains the address of the
 * allocated area ;
 * else it returns 1 if the allocation fail.
 */
int _starpu_src_common_allocate(struct _starpu_mp_node *mp_node, void **addr, size_t size)
{
	enum _starpu_mp_command answer;
	void *arg;
	int arg_size;

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

	_starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_ALLOCATE, &size,
			sizeof(size));

	answer = _starpu_src_common_wait_command_sync(mp_node, &arg, &arg_size);

        if (answer == STARPU_MP_COMMAND_ERROR_ALLOCATE)
        {
                STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);
                return 1;
        }

	STARPU_ASSERT(answer == STARPU_MP_COMMAND_ANSWER_ALLOCATE && arg_size == sizeof(*addr));

	memcpy(addr, arg, arg_size);

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

	return 0;
}

/* Send a request to the sink linked to the MP_NODE to deallocate the memory
 * area pointed by ADDR.
 */
void _starpu_src_common_free(struct _starpu_mp_node *mp_node, void *addr)
{
        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);
        _starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_FREE, &addr, sizeof(addr));
        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);
}

/* Send SIZE bytes pointed by SRC to DST on the sink linked to the MP_NODE with a
 * synchronous mode.
 */
int _starpu_src_common_copy_host_to_sink_sync(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size)
{
        struct _starpu_mp_transfer_command cmd = {size, dst, NULL};

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

        _starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_RECV_FROM_HOST, &cmd, sizeof(cmd));

        mp_node->dt_send(mp_node, src, size, NULL);

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

        return 0;
}

/* Send SIZE bytes pointed by SRC to DST on the sink linked to the MP_NODE with an
 * asynchronous mode.
 */
int _starpu_src_common_copy_host_to_sink_async(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size, void * event)
{
        struct _starpu_mp_transfer_command cmd = {size, dst, event};

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

        /* For asynchronous transfers, we save informations
         * to test is they are finished
         */
        struct _starpu_async_channel * async_channel = event;
        async_channel->polling_node_receiver = mp_node;

        _starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC, &cmd, sizeof(cmd));

        mp_node->dt_send(mp_node, src, size, event);

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

        return -EAGAIN;
}

/* Receive SIZE bytes pointed by SRC on the sink linked to the MP_NODE and store them in DST
 * with a synchronous mode.
 */
int _starpu_src_common_copy_sink_to_host_sync(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size)
{
        enum _starpu_mp_command answer;
        void *arg;
        int arg_size;
        struct _starpu_mp_transfer_command cmd = {size, src, NULL};

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

        _starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_SEND_TO_HOST, &cmd, sizeof(cmd));

        answer = _starpu_src_common_wait_command_sync(mp_node, &arg, &arg_size);

        STARPU_ASSERT(answer == STARPU_MP_COMMAND_SEND_TO_HOST);

        mp_node->dt_recv(mp_node, dst, size, NULL);

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

        return 0;
}

/* Receive SIZE bytes pointed by SRC on the sink linked to the MP_NODE and store them in DST
 * with an asynchronous mode.
 */
int _starpu_src_common_copy_sink_to_host_async(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size, void * event)
{
        struct _starpu_mp_transfer_command cmd = {size, src, event};

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

        /* For asynchronous transfers, we save informations
         * to test is they are finished
         */
        struct _starpu_async_channel * async_channel = event;
        async_channel->polling_node_sender = mp_node;

        _starpu_mp_common_send_command(mp_node, STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC, &cmd, sizeof(cmd));

        mp_node->dt_recv(mp_node, dst, size, event);

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

        return -EAGAIN;
}

/* Tell the sink linked to SRC_NODE to send SIZE bytes of data pointed by SRC
 * to the sink linked to DST_NODE. The latter store them in DST with a synchronous
 * mode.
 */
int _starpu_src_common_copy_sink_to_sink_sync(struct _starpu_mp_node *src_node, struct _starpu_mp_node *dst_node, void *src, void *dst, size_t size)
{
        enum _starpu_mp_command answer;
        void *arg;
        int arg_size;

        struct _starpu_mp_transfer_command_to_device cmd = {dst_node->peer_id, size, src, NULL};

        /* lock the node with the little peer_id first to prevent deadlock */
        if (src_node->peer_id > dst_node->peer_id)
        {
                STARPU_PTHREAD_MUTEX_LOCK(&dst_node->connection_mutex);
                STARPU_PTHREAD_MUTEX_LOCK(&src_node->connection_mutex);
        }
        else
        {
                STARPU_PTHREAD_MUTEX_LOCK(&src_node->connection_mutex);
                STARPU_PTHREAD_MUTEX_LOCK(&dst_node->connection_mutex);
        }

        /* Tell source to send data to dest. */
        _starpu_mp_common_send_command(src_node, STARPU_MP_COMMAND_SEND_TO_SINK, &cmd, sizeof(cmd));

        /* Release the source as fast as possible */
        STARPU_PTHREAD_MUTEX_UNLOCK(&src_node->connection_mutex);

        cmd.devid = src_node->peer_id;
        cmd.size = size;
        cmd.addr = dst;

        /* Tell dest to receive data from source. */
        _starpu_mp_common_send_command(dst_node, STARPU_MP_COMMAND_RECV_FROM_SINK, &cmd, sizeof(cmd));

        /* Wait for answer from dest to know wether transfer is finished. */
        answer = _starpu_src_common_wait_command_sync(dst_node, &arg, &arg_size);

        STARPU_ASSERT(answer == STARPU_MP_COMMAND_TRANSFER_COMPLETE);

        /* Release the receiver when we received the acknowlegment */
        STARPU_PTHREAD_MUTEX_UNLOCK(&dst_node->connection_mutex);

        return 0;
}

/* Tell the sink linked to SRC_NODE to send SIZE bytes of data pointed by SRC
 * to the sink linked to DST_NODE. The latter store them in DST with an asynchronous
 * mode.
 */
int _starpu_src_common_copy_sink_to_sink_async(struct _starpu_mp_node *src_node, struct _starpu_mp_node *dst_node, void *src, void *dst, size_t size, void * event)
{
        struct _starpu_mp_transfer_command_to_device cmd = {dst_node->peer_id, size, src, event};

        /* lock the node with the little peer_id first to prevent deadlock */
        if (src_node->peer_id > dst_node->peer_id)
        {
                STARPU_PTHREAD_MUTEX_LOCK(&dst_node->connection_mutex);
                STARPU_PTHREAD_MUTEX_LOCK(&src_node->connection_mutex);
        }
        else
        {
                STARPU_PTHREAD_MUTEX_LOCK(&src_node->connection_mutex);
                STARPU_PTHREAD_MUTEX_LOCK(&dst_node->connection_mutex);
        }

        /* For asynchronous transfers, we save informations
         * to test is they are finished
         */
        struct _starpu_async_channel * async_channel = event;
        async_channel->polling_node_sender = src_node;
        async_channel->polling_node_receiver = dst_node;
        /* Increase number of ack waited */
        async_channel->starpu_mp_common_finished_receiver++;
        async_channel->starpu_mp_common_finished_sender++;

        /* Tell source to send data to dest. */
        _starpu_mp_common_send_command(src_node, STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC, &cmd, sizeof(cmd));

        STARPU_PTHREAD_MUTEX_UNLOCK(&src_node->connection_mutex);

        cmd.devid = src_node->peer_id;
        cmd.size = size;
        cmd.addr = dst;

        /* Tell dest to receive data from source. */
        _starpu_mp_common_send_command(dst_node, STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC, &cmd, sizeof(cmd));

        STARPU_PTHREAD_MUTEX_UNLOCK(&dst_node->connection_mutex);

        return -EAGAIN;
}

/* 5 functions to determine the executable to run on the device (MIC, MPI).
 */
static void _starpu_src_common_cat_3(char *final, const size_t len, const char *first, const char *second, const char *third)
{
	snprintf(final, len, "%s%s%s", first, second, third);
}

static void _starpu_src_common_cat_2(char *final, const size_t len, const char *first, const char *second)
{
	_starpu_src_common_cat_3(final, len, first, second, "");
}

static void _starpu_src_common_dir_cat(char *final, const size_t len, const char *dir, const char *file)
{
	if (file[0] == '/')
		++file;

	size_t size = strlen(dir);
	if (dir[size - 1] == '/')
		_starpu_src_common_cat_2(final, len, dir, file);
	else
		_starpu_src_common_cat_3(final, len, dir, "/", file);
}

static int _starpu_src_common_test_suffixes(char *located_file_name, const size_t len, const char *base, const char **suffixes)
{
	unsigned int i;
	for (i = 0; suffixes[i] != NULL; ++i)
	{
		_starpu_src_common_cat_2(located_file_name, len, base, suffixes[i]);
		if (access(located_file_name, R_OK) == 0)
			return 0;
	}

	return 1;
}

int _starpu_src_common_locate_file(char *located_file_name, size_t len,
				   const char *env_file_name, const char *env_mic_path,
				   const char *config_file_name, const char *actual_file_name,
				   const char **suffixes)
{
	if (env_file_name != NULL)
	{
		if (access(env_file_name, R_OK) == 0)
		{
			strncpy(located_file_name, env_file_name, len);
			return 0;
		}
		else if(env_mic_path != NULL)
		{
			_starpu_src_common_dir_cat(located_file_name, len, env_mic_path, env_file_name);

			return access(located_file_name, R_OK);
		}
	}
	else if (config_file_name != NULL)
	{
		if (access(config_file_name, R_OK) == 0)
		{
			strncpy(located_file_name, config_file_name, len);
			return 0;
		}
		else if (env_mic_path != NULL)
		{
			_starpu_src_common_dir_cat(located_file_name, len, env_mic_path, config_file_name);

			return access(located_file_name, R_OK);
		}
	}
	else if (actual_file_name != NULL)
	{
		if (_starpu_src_common_test_suffixes(located_file_name, len, actual_file_name, suffixes) == 0)
			return 0;

		if (env_mic_path != NULL)
		{
			char actual_cpy[1024];
			strncpy(actual_cpy, actual_file_name, sizeof(actual_cpy));

			char *last =  strrchr(actual_cpy, '/');
			while (last != NULL)
			{
				char tmp[1024];

				_starpu_src_common_dir_cat(tmp, sizeof(tmp), env_mic_path, last);

				if (access(tmp, R_OK) == 0)
				{
					strncpy(located_file_name, tmp, len);
					return 0;
				}

				if (_starpu_src_common_test_suffixes(located_file_name, len, tmp, suffixes) == 0)
					return 0;

				*last = '\0';
				char *last_tmp = strrchr(actual_cpy, '/');
				*last = '/';
				last = last_tmp;
			}
		}
	}

	return 1;
}

#if defined(STARPU_USE_MPI_MASTER_SLAVE) && !defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
void _starpu_src_common_init_switch_env(unsigned this)
{
        save_thread_env[this].current_task = starpu_task_get_current();
        save_thread_env[this].current_worker = STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_key);
        save_thread_env[this].current_worker_set = STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_set_key);
#ifdef STARPU_OPENMP
        save_thread_env[this].current_omp_thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
        save_thread_env[this].current_omp_task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
#endif
}

static void _starpu_src_common_switch_env(unsigned old, unsigned new)
{
        save_thread_env[old].current_task = starpu_task_get_current();
        save_thread_env[old].current_worker = STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_key);
        save_thread_env[old].current_worker_set = STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_set_key);
#ifdef STARPU_OPENMP
        save_thread_env[old].current_omp_thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
        save_thread_env[old].current_omp_task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
#endif

        _starpu_set_current_task(save_thread_env[new].current_task);
        STARPU_PTHREAD_SETSPECIFIC(_starpu_worker_key, save_thread_env[new].current_worker);
        STARPU_PTHREAD_SETSPECIFIC(_starpu_worker_set_key, save_thread_env[new].current_worker_set);
#ifdef STARPU_OPENMP
        STARPU_PTHREAD_SETSPECIFIC(omp_thread_key, save_thread_env[new].current_omp_thread);
        STARPU_PTHREAD_SETSPECIFIC(omp_task_key, save_thread_env[new].current_omp_task);
#endif
}
#endif

/* Send workers to the sink node
 */
static void _starpu_src_common_send_workers(struct _starpu_mp_node * node, int baseworkerid, int nworkers)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	int worker_size = sizeof(struct _starpu_worker)*nworkers;
	int combined_worker_size = STARPU_NMAX_COMBINEDWORKERS*sizeof(struct _starpu_combined_worker);
	int msg[5];
	msg[0] = nworkers;
	msg[1] = worker_size;
	msg[2] = combined_worker_size;
	msg[3] = baseworkerid;
	msg[4] = starpu_worker_get_count();

        STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);

	/* tell the sink node that we will send him all workers */
	_starpu_mp_common_send_command(node, STARPU_MP_COMMAND_SYNC_WORKERS, &msg, sizeof(msg));

	/* Send all worker to the sink node */
	node->dt_send(node,&config->workers[baseworkerid],worker_size, NULL);

	/* Send all combined workers to the sink node */
	node->dt_send(node, &config->combined_workers,combined_worker_size, NULL);

        STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
}

static void _starpu_src_common_worker_internal_work(struct _starpu_worker_set * worker_set, struct _starpu_mp_node * mp_node, struct starpu_task **tasks, unsigned memnode)
{
        int res = 0;
	unsigned i;

        _starpu_may_pause();

#ifdef STARPU_SIMGRID
        starpu_pthread_wait_reset(&worker_set->workers[0].wait);
#endif

	/* Test if async transfers are completed */
	for (i = 0; i < worker_set->nworkers; i++)
	{
		struct starpu_task *task = worker_set->workers[i].task_transferring;
		/* We send all buffers to execute the task */
		if (task != NULL && worker_set->workers[i].nb_buffers_transferred == worker_set->workers[i].nb_buffers_totransfer)
		{
			STARPU_RMB();
			struct _starpu_job * j = _starpu_get_job_associated_to_task(task);

			_STARPU_TRACE_END_PROGRESS(memnode);
			_starpu_set_local_worker_key(&worker_set->workers[i]);
			_starpu_fetch_task_input_tail(task, j, &worker_set->workers[i]);
			_starpu_set_worker_status(&worker_set->workers[i], STATUS_UNKNOWN);
			/* Reset it */
			worker_set->workers[i].task_transferring = NULL;

			/* Execute the task */
			res =  _starpu_src_common_execute(j, &worker_set->workers[i], mp_node);
			switch (res)
			{
				case 0:
					/* The task task has been launched with no error */
					break;
				case -EAGAIN:
					_STARPU_DISP("ouch, this MP worker could not actually run task %p, putting it back...\n", tasks[i]);
					_starpu_push_task_to_workers(worker_set->workers[i].task_transferring);
					STARPU_ABORT();
					continue;
					break;
				default:
					STARPU_ASSERT(0);
			}

			_STARPU_TRACE_START_PROGRESS(memnode);
		}
	}

        res |= __starpu_datawizard_progress(1, 1);

        /* Handle message which have been store */
        _starpu_src_common_handle_stored_async(mp_node);

        STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);

	unsigned stopped_progress = 0;
        /* poll the device for completed jobs.*/
        while(mp_node->mp_recv_is_ready(mp_node))
        {
		stopped_progress = 1;
		_STARPU_TRACE_END_PROGRESS(mp_node_memory_node(mp_node));
                _starpu_src_common_recv_async(mp_node);
                /* Mutex is unlock in _starpu_src_common_recv_async */
                STARPU_PTHREAD_MUTEX_LOCK(&mp_node->connection_mutex);
        }
	if (stopped_progress)
		_STARPU_TRACE_START_PROGRESS(mp_node_memory_node(mp_node));

        STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->connection_mutex);

        /* get task for each worker*/
        res |= _starpu_get_multi_worker_task(worker_set->workers, tasks, worker_set->nworkers, memnode);

#ifdef STARPU_SIMGRID
        if (!res)
                starpu_pthread_wait_wait(&worker_set->workers[0].wait);
#endif

	/*if at least one worker have pop a task*/
	if(res != 0)
	{
		for(i=0; i<worker_set->nworkers; i++)
		{
			if(tasks[i] != NULL)
			{
				struct _starpu_worker *worker = &worker_set->workers[i];
				_STARPU_TRACE_END_PROGRESS(worker->memory_node);
				_starpu_set_local_worker_key(worker);
				int ret = _starpu_fetch_task_input(tasks[i], _starpu_get_job_associated_to_task(tasks[i]), 1);
				STARPU_ASSERT(!ret);
				_STARPU_TRACE_START_PROGRESS(worker->memory_node);
			}
		}

		/* Handle message which have been store */
		_starpu_src_common_handle_stored_async(mp_node);
	}
}

#if defined(STARPU_USE_MPI_MASTER_SLAVE) && !defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
/* Function looping on the source node */
void _starpu_src_common_workers_set(struct _starpu_worker_set * worker_set, int ndevices, struct _starpu_mp_node ** mp_node)
{
        unsigned memnode[ndevices];
        unsigned offsetmemnode[ndevices];
        memset(offsetmemnode, 0, ndevices*sizeof(unsigned));

        int device;
        int nbworkers = 0;
        for (device = 0; device < ndevices; device++)
        {
                memnode[device] = worker_set[device].workers[0].memory_node;
                nbworkers += worker_set[device].nworkers;
                if (device != 0)
                        offsetmemnode[device] += offsetmemnode[device-1];
                if (device != ndevices -1)
                        offsetmemnode[device+1] += worker_set[device].nworkers;
        }

        struct starpu_task **tasks;
        _STARPU_MALLOC(tasks, sizeof(struct starpu_task *)*nbworkers);

        for (device = 0; device < ndevices; device++)
        {
                struct _starpu_worker *baseworker = &worker_set[device].workers[0];
                struct _starpu_machine_config *config = baseworker->config;
                unsigned baseworkerid = baseworker - config->workers;
                _starpu_src_common_send_workers(mp_node[device], baseworkerid, worker_set[device].nworkers);
        	_STARPU_TRACE_START_PROGRESS(memnode[device]);
        }

        /*main loop*/
        while (_starpu_machine_is_running())
        {
                for (device = 0; device < ndevices ; device++)
                {
                        _starpu_src_common_switch_env(((device-1)+ndevices)%ndevices, device);
                        _starpu_src_common_worker_internal_work(&worker_set[device], mp_node[device], tasks+offsetmemnode[device], memnode[device]);
                }
        }
        free(tasks);

        for (device = 0; device < ndevices; device++)
	{
        	_STARPU_TRACE_END_PROGRESS(memnode[device]);
                _starpu_handle_all_pending_node_data_requests(memnode[device]);
	}

        /* In case there remains some memory that was automatically
         * allocated by StarPU, we release it now. Note that data
         * coherency is not maintained anymore at that point ! */
        for (device = 0; device < ndevices; device++)
                _starpu_free_all_automatically_allocated_buffers(memnode[device]);

}
#endif

/* Function looping on the source node */
void _starpu_src_common_worker(struct _starpu_worker_set * worker_set, unsigned baseworkerid, struct _starpu_mp_node * mp_node)
{
        unsigned memnode = worker_set->workers[0].memory_node;
        struct starpu_task **tasks;

        _STARPU_MALLOC(tasks, sizeof(struct starpu_task *)*worker_set->nworkers);

        _starpu_src_common_send_workers(mp_node, baseworkerid, worker_set->nworkers);

        _STARPU_TRACE_START_PROGRESS(memnode);
        /*main loop*/
        while (_starpu_machine_is_running())
        {
                _starpu_src_common_worker_internal_work(worker_set, mp_node, tasks, memnode);
        }
        free(tasks);

        _STARPU_TRACE_END_PROGRESS(memnode);

        _starpu_handle_all_pending_node_data_requests(memnode);

        /* In case there remains some memory that was automatically
         * allocated by StarPU, we release it now. Note that data
         * coherency is not maintained anymore at that point ! */
        _starpu_free_all_automatically_allocated_buffers(memnode);
}
