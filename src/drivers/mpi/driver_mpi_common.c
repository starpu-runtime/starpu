/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/mpi/driver_mpi_common.h>

#define NITER 32
#define SIZE_BANDWIDTH (1024*1024)

#define DRIVER_MPI_MASTER_NODE_DEFAULT 0

static int mpi_initialized = 0;
static int extern_initialized = 0;
static int src_node_id;

int _starpu_mpi_common_multiple_thread;

/* (For a given datawizard we may have several starpu_interface_copy calls) */
LIST_TYPE(_starpu_mpi_ms_event_request,
	MPI_Request request;
);

struct _starpu_mpi_ms_async_event
{
	int is_sender;
	struct _starpu_mpi_ms_event_request_list * requests;
};

static inline struct _starpu_mpi_ms_async_event *_starpu_mpi_ms_async_event(union _starpu_async_channel_event *_event)
{
	struct _starpu_mpi_ms_async_event *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (void *) _event;
	return event;
}

/* This lets the user decide which MPI rank is to be the master. Usually it's just rank 0 */
static void _starpu_mpi_set_src_node_id()
{
	int node_id = starpu_getenv_number("STARPU_MPI_MASTER_NODE");

	if (node_id != -1)
	{
		int nb_proc, id_proc;
		MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
		MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

		if (node_id < nb_proc)
		{
			src_node_id = node_id;
			return;
		}
		else if (id_proc == DRIVER_MPI_MASTER_NODE_DEFAULT)
		{
			/* Only one node prints the error message. */
			_STARPU_MSG("The node (%d) you specify to be the master is "
				    "greater than the total number of nodes (%d). "
				    "StarPU will use node %d.\n", node_id, nb_proc, DRIVER_MPI_MASTER_NODE_DEFAULT);
		}
	}

	/* Node by default. */
	src_node_id = DRIVER_MPI_MASTER_NODE_DEFAULT;
}

int _starpu_mpi_common_mp_init()
{
	//Here we supposed the programmer called two times starpu_init.
	if (mpi_initialized)
		return -ENODEV;

	mpi_initialized = 1;

	_starpu_mpi_common_multiple_thread = starpu_getenv_number_default("STARPU_MPI_MS_MULTIPLE_THREAD", 0);

	if (MPI_Initialized(&extern_initialized) != MPI_SUCCESS)
		STARPU_ABORT_MSG("Cannot check if MPI is initialized or not !");

	//Here MPI_Init or MPI_Init_thread is already called
	if (!extern_initialized)
	{

		int required = _starpu_mpi_common_multiple_thread ? MPI_THREAD_MULTIPLE : MPI_THREAD_FUNNELED;

		int thread_support;
		if (MPI_Init_thread(_starpu_get_argc(), _starpu_get_argv(), required, &thread_support) != MPI_SUCCESS)
		{
			STARPU_ABORT_MSG("Cannot Initialize MPI !");
		}

		if (thread_support != required)
		{
			if (required == MPI_THREAD_MULTIPLE)
				_STARPU_DISP("MPI doesn't support MPI_THREAD_MULTIPLE option. MPI Master-Slave can have problems if multiple slaves are launched. \n");
			if (required == MPI_THREAD_FUNNELED)
				_STARPU_DISP("MPI doesn't support MPI_THREAD_FUNNELED option. Many errors can occur. \n");
		}
	}

	/* Find which node is the master */
	_starpu_mpi_set_src_node_id();

	/* In MPI case we look at the rank to know if we are a sink */
	if (!_starpu_mpi_common_is_src_node())
		setenv("STARPU_SINK", "STARPU_MPI_MS", 1);

	return 1;
}

void _starpu_mpi_common_mp_deinit()
{
	if (!extern_initialized)
		MPI_Finalize();
}

int _starpu_mpi_common_is_src_node()
{
	int id_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
	return id_proc == src_node_id;
}

int _starpu_mpi_common_get_src_node()
{
	return src_node_id;
}

int _starpu_mpi_common_is_mp_initialized()
{
	return mpi_initialized;
}

/* common parts to initialize a source or a sink node */
void _starpu_mpi_common_mp_initialize_src_sink(struct _starpu_mp_node *node)
{
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;

	int nmpicores = starpu_getenv_number("STARPU_NMPIMSTHREADS");
	if (nmpicores == -1)
	{
		int nhyperthreads = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];
		node->nb_cores = topology->nusedpus / nhyperthreads;
	}
	else
		node->nb_cores = nmpicores;
}

int _starpu_mpi_common_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
	int res, source;
	int flag = 0;
	int id_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

	if (id_proc == src_node_id)
	{
		/* Source has mp_node defined */
		source = mp_node->mp_connection.mpi_remote_nodeid;
	}
	else
	{
		/* Sink can have sink to sink message */
		source = MPI_ANY_SOURCE;
	}

	res = MPI_Iprobe(source, SYNC_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot test if we received a message !");

	return flag;
}

int _starpu_mpi_common_notif_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
	int res, source;
	int flag = 0;
	int id_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);

	if (id_proc == src_node_id)
	{
		/* Source has mp_node defined */
		source = mp_node->mp_connection.mpi_remote_nodeid;
	}
	else
	{
		/* Sink can have sink to sink message */
		source = MPI_ANY_SOURCE;
	}

	res = MPI_Iprobe(source, NOTIF_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot test if we received a message !");

	return flag;
}

int _starpu_mpi_common_notif_send_is_ready(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED)
{
	return 1;
}

static void __starpu_mpi_common_send_to_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int dst_devid, void *msg, int len, void * event, int notif);
static void __starpu_mpi_common_recv_from_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int src_devid, void *msg, int len, void * event, int notif);

/* SEND to source node */
static void __starpu_mpi_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif)
{
	//_STARPU_MSG("envoi %d B to %d\n", len, node->mp_connection.mpi_remote_nodeid);
	__starpu_mpi_common_send_to_device(node, node->mp_connection.mpi_remote_nodeid, msg, len, event, notif);
}

void _starpu_mpi_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
	__starpu_mpi_common_send(node, msg, len, event, 0);
}

void _starpu_mpi_common_mp_send(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_mpi_common_send(node, msg, len, NULL, 0);
}

void _starpu_mpi_common_nt_send(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_mpi_common_send(node, msg, len, NULL, 1);
}

/* RECV to source node */
static void __starpu_mpi_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif)
{
	//_STARPU_MSG("recv %d B from %d in %p\n", len, node->mp_connection.mpi_remote_nodeid, msg);
	__starpu_mpi_common_recv_from_device(node, node->mp_connection.mpi_remote_nodeid, msg, len, event, notif);
}

void _starpu_mpi_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
	__starpu_mpi_common_recv(node, msg, len, event, 0);
}

void _starpu_mpi_common_mp_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_mpi_common_recv(node, msg, len, NULL, 0);
}

void _starpu_mpi_common_nt_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_mpi_common_recv(node, msg, len, NULL, 1);
}

/* SEND to any node */
static void __starpu_mpi_common_send_to_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int dst_devid, void *msg, int len, void * event, int notif)
{
	int res;

	//_STARPU_MSG("S_to_D send %d bytes from %d from %p\n", len, dst_devid, msg);

	if (event)
	{
		/* Asynchronous send */
		struct _starpu_async_channel * channel = event;
		struct _starpu_mpi_ms_async_event *mpi_ms_event = _starpu_mpi_ms_async_event(&channel->event);
		mpi_ms_event->is_sender = 1;

		/* call by sink, we need to initialize some parts, for host it's done in data_request.c */
		if (channel->node_ops == NULL)
			mpi_ms_event->requests = NULL;

		/* Initialize the list */
		if (mpi_ms_event->requests == NULL)
			mpi_ms_event->requests = _starpu_mpi_ms_event_request_list_new();

		struct _starpu_mpi_ms_event_request * req = _starpu_mpi_ms_event_request_new();

		res = MPI_Isend(msg, len, MPI_BYTE, dst_devid, ASYNC_TAG, MPI_COMM_WORLD, &req->request);

		channel->starpu_mp_common_finished_receiver++;
		channel->starpu_mp_common_finished_sender++;

		_starpu_mpi_ms_event_request_list_push_back(mpi_ms_event->requests, req);
	}
	else
	{
		/* Synchronous send */
		/* Send commands */
		if (!notif)
			res = MPI_Send(msg, len, MPI_BYTE, dst_devid, SYNC_TAG, MPI_COMM_WORLD);
		/* Send notifications */
		else
			res = MPI_Send(msg, len, MPI_BYTE, dst_devid, NOTIF_TAG, MPI_COMM_WORLD);
	}

	STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
}

void _starpu_mpi_common_send_to_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int dst_devid, void *msg, int len, void * event)
{
	__starpu_mpi_common_send_to_device(node, dst_devid, msg, len, event, 0);
}

/* RECV to any node */
static void __starpu_mpi_common_recv_from_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int src_devid, void *msg, int len, void * event, int notif)
{
	int res;

	//_STARPU_MSG("R_to_D nop recv %d bytes from %d\n", len, src_devid);

	if (event)
	{
		/* Asynchronous recv */
		struct _starpu_async_channel * channel = event;
		struct _starpu_mpi_ms_async_event *mpi_ms_event = _starpu_mpi_ms_async_event(&channel->event);
		mpi_ms_event->is_sender = 0;

		/* call by sink, we need to initialize some parts, for host it's done in data_request.c */
		if (channel->node_ops == NULL)
			mpi_ms_event->requests = NULL;

		/* Initialize the list */
		if (mpi_ms_event->requests == NULL)
			mpi_ms_event->requests = _starpu_mpi_ms_event_request_list_new();

		struct _starpu_mpi_ms_event_request * req = _starpu_mpi_ms_event_request_new();

		res = MPI_Irecv(msg, len, MPI_BYTE, src_devid, ASYNC_TAG, MPI_COMM_WORLD, &req->request);
		STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot Ireceive a msg with a size of %d Bytes !", len);

		channel->starpu_mp_common_finished_receiver++;
		channel->starpu_mp_common_finished_sender++;

		_starpu_mpi_ms_event_request_list_push_back(mpi_ms_event->requests, req);
	}
	else
	{
		/* Synchronous recv */
		MPI_Status s;
		/* Send commands */
		if (!notif)
			res = MPI_Recv(msg, len, MPI_BYTE, src_devid, SYNC_TAG, MPI_COMM_WORLD, &s);
		else
			res = MPI_Recv(msg, len, MPI_BYTE, src_devid, NOTIF_TAG, MPI_COMM_WORLD, &s);

		int num_expected;
		MPI_Get_count(&s, MPI_BYTE, &num_expected);

		STARPU_ASSERT_MSG(num_expected == len, "MPI Master/Slave received a msg with a size of %d Bytes (expected %d Bytes) !", num_expected, len);
		STARPU_ASSERT_MSG(res == MPI_SUCCESS, "MPI Master/Slave cannot receive a msg with a size of %d Bytes !", len);
	}
}

void _starpu_mpi_common_recv_from_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int src_devid, void *msg, int len, void * event)
{
	__starpu_mpi_common_recv_from_device(node, src_devid, msg, len, event, 0);
}

static void _starpu_mpi_common_polling_node(struct _starpu_mp_node * node)
{
	/* poll the asynchronous messages.*/
	if (node != NULL)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);
		while(node->nt_recv_is_ready(node))
		{
			enum _starpu_mp_command answer;
			void *arg;
			int arg_size;
			answer = _starpu_nt_common_recv_command(node, &arg, &arg_size);
			if(!_starpu_src_common_store_message(node,arg,arg_size,answer))
			{
				_STARPU_ERROR("incorrect command: unknown command or sync command");
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
	}
}

/* - In device to device communications, the first ack received by host
 * is considered as the sender (but it cannot be, in fact, the sender)
 */
unsigned int _starpu_mpi_common_test_event(struct _starpu_async_channel * event)
{
	struct _starpu_mpi_ms_async_event *mpi_ms_event = _starpu_mpi_ms_async_event(&event->event);
	if (mpi_ms_event->requests != NULL && !_starpu_mpi_ms_event_request_list_empty(mpi_ms_event->requests))
	{
		struct _starpu_mpi_ms_event_request * req = _starpu_mpi_ms_event_request_list_begin(mpi_ms_event->requests);
		struct _starpu_mpi_ms_event_request * req_next;

		while (req != _starpu_mpi_ms_event_request_list_end(mpi_ms_event->requests))
		{
			req_next = _starpu_mpi_ms_event_request_list_next(req);

			int flag = 0;
			MPI_Test(&req->request, &flag, MPI_STATUS_IGNORE);
			if (flag)
			{
				_starpu_mpi_ms_event_request_list_erase(mpi_ms_event->requests, req);
				_starpu_mpi_ms_event_request_delete(req);

				if (mpi_ms_event->is_sender)
					event->starpu_mp_common_finished_sender--;
				else
					event->starpu_mp_common_finished_receiver--;

			}
			req = req_next;
		}

		/* When the list is empty, we finished to wait each request */
		if (_starpu_mpi_ms_event_request_list_empty(mpi_ms_event->requests))
		{
			/* Destroy the list */
			_starpu_mpi_ms_event_request_list_delete(mpi_ms_event->requests);
			mpi_ms_event->requests = NULL;
		}
	}

	_starpu_mpi_common_polling_node(event->polling_node_sender);
	_starpu_mpi_common_polling_node(event->polling_node_receiver);

	return !event->starpu_mp_common_finished_sender && !event->starpu_mp_common_finished_receiver;
}

/* - In device to device communications, the first ack received by host
 * is considered as the sender (but it cannot be, in fact, the sender)
 */
void _starpu_mpi_common_wait_request_completion(struct _starpu_async_channel * event)
{
	struct _starpu_mpi_ms_async_event *mpi_ms_event = _starpu_mpi_ms_async_event(&event->event);
	if (mpi_ms_event->requests != NULL && !_starpu_mpi_ms_event_request_list_empty(mpi_ms_event->requests))
	{
		struct _starpu_mpi_ms_event_request * req = _starpu_mpi_ms_event_request_list_begin(mpi_ms_event->requests);
		struct _starpu_mpi_ms_event_request * req_next;

		while (req != _starpu_mpi_ms_event_request_list_end(mpi_ms_event->requests))
		{
			req_next = _starpu_mpi_ms_event_request_list_next(req);

			MPI_Wait(&req->request, MPI_STATUS_IGNORE);
			_starpu_mpi_ms_event_request_list_erase(mpi_ms_event->requests, req);

			_starpu_mpi_ms_event_request_delete(req);
			req = req_next;

			if (mpi_ms_event->is_sender)
				event->starpu_mp_common_finished_sender--;
			else
				event->starpu_mp_common_finished_receiver--;

		}

		STARPU_ASSERT_MSG(_starpu_mpi_ms_event_request_list_empty(mpi_ms_event->requests), "MPI Request list is not empty after a wait_event !");

		/* Destroy the list */
		_starpu_mpi_ms_event_request_list_delete(mpi_ms_event->requests);
		mpi_ms_event->requests = NULL;
	}

	//incoming ack from devices
	while(event->starpu_mp_common_finished_sender > 0 || event->starpu_mp_common_finished_receiver > 0)
	{
		_starpu_mpi_common_polling_node(event->polling_node_sender);
		_starpu_mpi_common_polling_node(event->polling_node_receiver);
	}
}

void _starpu_mpi_common_barrier(void)
{
	int ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier failed");
}

/* Compute bandwidth and latency between source and sink nodes
 * Source node has to have the entire set of times at the end
 */
void _starpu_mpi_common_measure_bandwidth_latency(double timing_dtod[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS], double latency_dtod[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS])
{
	int ret;
	unsigned iter;

	int nb_proc, id_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
	MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);

	char * buf;
	_STARPU_MALLOC(buf, SIZE_BANDWIDTH);
	memset(buf, 0, SIZE_BANDWIDTH);

	int sender, receiver;
	for(sender = 0; sender < nb_proc; sender++)
	{
		for(receiver = 0; receiver < nb_proc; receiver++)
		{
			//Node can't be a sender and a receiver
			if(sender == receiver)
				continue;

			if (src_node_id == id_proc)
				_STARPU_DISP("measuring from %d to %d\n", sender, receiver);

			ret = MPI_Barrier(MPI_COMM_WORLD);
			STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier failed");

			if(id_proc == sender)
			{
				double start, end;

				/* measure bandwidth sender to receiver */
				start = starpu_timing_now();
				for (iter = 0; iter < NITER; iter++)
				{
					ret = MPI_Send(buf, SIZE_BANDWIDTH, MPI_BYTE, receiver, 42, MPI_COMM_WORLD);
					STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
				}
				end = starpu_timing_now();
				timing_dtod[sender][receiver] = (end - start)/NITER/SIZE_BANDWIDTH;

				/* measure latency sender to receiver */
				start = starpu_timing_now();
				for (iter = 0; iter < NITER; iter++)
				{
					ret = MPI_Send(buf, 1, MPI_BYTE, receiver, 42, MPI_COMM_WORLD);
					STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Latency of MPI Master/Slave cannot be measured !");
				}
				end = starpu_timing_now();
				latency_dtod[sender][receiver] = (end - start)/NITER;
			}

			if (id_proc == receiver)
			{
				/* measure bandwidth sender to receiver*/
				for (iter = 0; iter < NITER; iter++)
				{
					ret = MPI_Recv(buf, SIZE_BANDWIDTH, MPI_BYTE, sender, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
				}

				/* measure latency sender to receiver */
				for (iter = 0; iter < NITER; iter++)
				{
					ret = MPI_Recv(buf, 1, MPI_BYTE, sender, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "Bandwidth of MPI Master/Slave cannot be measured !");
				}
			}
		}

		/* When a sender finished its work, it has to send its results to the master */

		/* Sender doesn't need to send to itself its data */
		if (sender == src_node_id)
			goto print;

		/* if we are the sender, we send the data */
		if (sender == id_proc)
		{
			MPI_Send(timing_dtod[sender], STARPU_MAXMPIDEVS, MPI_DOUBLE, src_node_id, 42, MPI_COMM_WORLD);
			MPI_Send(latency_dtod[sender], STARPU_MAXMPIDEVS, MPI_DOUBLE, src_node_id, 42, MPI_COMM_WORLD);
		}

		/* the master node receives the data */
		if (src_node_id == id_proc)
		{
			MPI_Recv(timing_dtod[sender], STARPU_MAXMPIDEVS, MPI_DOUBLE, sender, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(latency_dtod[sender], STARPU_MAXMPIDEVS, MPI_DOUBLE, sender, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}


print:
		if (src_node_id == id_proc)
		{
			for(receiver = 0; receiver < nb_proc; receiver++)
			{
				if(sender == receiver)
					continue;

				_STARPU_DISP("BANDWIDTH %d -> %d %.0fMB/s %.2fus\n", sender, receiver, 1/timing_dtod[sender][receiver], latency_dtod[sender][receiver]);
			}
		}
	}
	free(buf);
}

