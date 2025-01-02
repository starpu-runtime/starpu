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

#ifndef __MP_COMMON_H__
#define __MP_COMMON_H__

/** @file */

#include <semaphore.h>

#include <starpu.h>
#include <common/config.h>
#include <common/list.h>
#include <common/barrier.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/copy_driver.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_USE_MP

#define BUFFER_SIZE 65536

#define STARPU_MP_SRC_NODE 0
#define STARPU_MP_SINK_NODE(a) ((a) + 1)

#define STARPU_MP_COMMON_REPORT_ERROR(node, status)			\
	(node)->report_error(__starpu_func__, __FILE__, __LINE__, (status))
enum _starpu_mp_command
{
	/* Commands from master to slave */

	STARPU_MP_COMMAND_EXIT,
	STARPU_MP_COMMAND_EXECUTE,
	STARPU_MP_COMMAND_EXECUTE_DETACHED,
	STARPU_MP_COMMAND_SINK_NBCORES,
	STARPU_MP_COMMAND_LOOKUP,
	STARPU_MP_COMMAND_ALLOCATE,
	STARPU_MP_COMMAND_FREE,
	STARPU_MP_COMMAND_MAP,
	STARPU_MP_COMMAND_UNMAP,
	STARPU_MP_COMMAND_SYNC_WORKERS,

	/* Note: synchronous send */
	STARPU_MP_COMMAND_RECV_FROM_HOST,
	STARPU_MP_COMMAND_SEND_TO_HOST,
	STARPU_MP_COMMAND_RECV_FROM_SINK,
	STARPU_MP_COMMAND_SEND_TO_SINK,

	/* Note: Asynchronous send */
	STARPU_MP_COMMAND_RECV_FROM_HOST_ASYNC,
	STARPU_MP_COMMAND_SEND_TO_HOST_ASYNC,
	STARPU_MP_COMMAND_RECV_FROM_SINK_ASYNC,
	STARPU_MP_COMMAND_SEND_TO_SINK_ASYNC,

	/* Synchronous answers from slave to master */
	STARPU_MP_COMMAND_ERROR_EXECUTE,
	STARPU_MP_COMMAND_ERROR_EXECUTE_DETACHED,
	STARPU_MP_COMMAND_ANSWER_LOOKUP,
	STARPU_MP_COMMAND_ERROR_LOOKUP,
	STARPU_MP_COMMAND_ANSWER_ALLOCATE,
	STARPU_MP_COMMAND_ERROR_ALLOCATE,
	STARPU_MP_COMMAND_ANSWER_MAP,
	STARPU_MP_COMMAND_ERROR_MAP,
	STARPU_MP_COMMAND_ANSWER_TRANSFER_COMPLETE,
	STARPU_MP_COMMAND_ANSWER_SINK_NBCORES,
	STARPU_MP_COMMAND_ANSWER_EXECUTION_SUBMITTED,
	STARPU_MP_COMMAND_ANSWER_EXECUTION_DETACHED_SUBMITTED,

	/* Asynchronous notifications from slave to master */
	STARPU_MP_COMMAND_NOTIF_RECV_FROM_HOST_ASYNC_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_SEND_TO_HOST_ASYNC_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_RECV_FROM_SINK_ASYNC_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_SEND_TO_SINK_ASYNC_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_EXECUTION_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_EXECUTION_DETACHED_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_PRE_EXECUTION,

	STARPU_MP_COMMAND_NOTIF_FIRST = STARPU_MP_COMMAND_NOTIF_RECV_FROM_HOST_ASYNC_COMPLETED,
	STARPU_MP_COMMAND_NOTIF_LAST = STARPU_MP_COMMAND_NOTIF_PRE_EXECUTION,
};

const char *_starpu_mp_common_command_to_string(const enum _starpu_mp_command command);

enum _starpu_mp_node_kind
{
	STARPU_NODE_MPI_SINK,
	STARPU_NODE_MPI_SOURCE,
	STARPU_NODE_TCPIP_SINK,
	STARPU_NODE_TCPIP_SOURCE,
	STARPU_NODE_INVALID_KIND
};

const char *_starpu_mp_common_node_kind_to_string(const int kind);

union _starpu_mp_connection
{
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	int mpi_remote_nodeid;
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	struct _starpu_tcpip_socket *tcpip_mp_connection;
#endif
};

struct _starpu_mp_transfer_command
{
	size_t size;
	void *addr;
	void *event;
};

struct _starpu_mp_transfer_command_to_device
{
	size_t size;
	void *addr;
	void *event;
	int devid;
	char end[];	/* Keep last to compute non-padded size */
};

struct _starpu_mp_transfer_map_command
{
	size_t offset;
	size_t size;
	char fd_name[];
};

struct _starpu_mp_transfer_unmap_command
{
	uintptr_t addr;
	size_t size;
};

LIST_TYPE(mp_barrier,
		int id;
		starpu_pthread_barrier_t before_work_barrier;
		starpu_pthread_barrier_t after_work_barrier;
	 );

LIST_TYPE(mp_message,
		enum _starpu_mp_command type;
		char *buffer;
		int size;
	 );

struct mp_task
{
	void (*kernel)(void **, void *);
	enum starpu_data_interface_id *ids;
	void **interfaces;
	unsigned nb_interfaces;
	void *cl_arg;
	unsigned cl_arg_size;
	void *cl_ret;
	unsigned cl_ret_size;
	unsigned coreid;
	enum starpu_codelet_type type;
	int is_parallel_task;
	int combined_workerid;
	int detached;
	struct mp_barrier* mp_barrier;
};

LIST_TYPE(_starpu_mp_event,
		struct _starpu_async_channel event;
		void * remote_event;
		enum _starpu_mp_command answer_cmd;
);


/** Message-passing working node, whether source
 * or sink */
struct _starpu_mp_node
{
	enum _starpu_mp_node_kind kind;

	int baseworkerid;

	/*the number of core on the device
	 * Must be initialized during init function*/
	int nb_cores;

	/*Is starpu running*/
	int is_running;

	/** Buffer used for data transfers, allocated
	 * during node initialization.
	 * Size : BUFFER_SIZE */
	void *buffer;

	/** For sink : -1.
	 * For host : index of the sink = devid.
	 */
	int peer_id;

	/** Connection used for command passing between the host thread and the
	 * sink it controls */
	union _starpu_mp_connection mp_connection;

	/** Mutex to protect the interleaving of communications when using one thread per node,
	 * for instance, when a thread transfers piece of data and an other wants to use
	 * a sink_to_sink communication */
	starpu_pthread_mutex_t connection_mutex;

	/** This list contains events
	 * about asynchronous request
	 */
	struct _starpu_mp_event_list event_list;
	/*list where threads add events to send to the source node */
	struct _starpu_mp_event_list event_queue;

	/** */
	starpu_pthread_barrier_t init_completed_barrier;

	/** table to store pointer of the thread workers*/
	void* thread_table;

	/*list where threads add messages to send to the source node */
	struct mp_message_list message_queue;
	starpu_pthread_mutex_t message_queue_mutex;

	/*list of barrier for combined worker*/
	struct mp_barrier_list barrier_list;
	starpu_pthread_mutex_t barrier_mutex;

	/*table where worker comme pick task*/
	struct mp_task ** run_table;
	struct mp_task ** run_table_detached;
	sem_t * sem_run_table;

	/** Node general functions */
	void (*init)		(struct _starpu_mp_node *node);
	void (*launch_workers)	(struct _starpu_mp_node *node);
	void (*deinit)		(struct _starpu_mp_node *node);
	void (*report_error)	(const char *, const char *, const int, const int);

	/** Message passing */
	int (*mp_recv_is_ready) (const struct _starpu_mp_node *);
	void (*mp_send)		(const struct _starpu_mp_node *, void *, int);
	void (*mp_recv)		(const struct _starpu_mp_node *, void *, int);

	/** Notifications */
	int (*nt_recv_is_ready) (const struct _starpu_mp_node *);
	int (*nt_send_is_ready) (const struct _starpu_mp_node *);
	void (*nt_send)		(const struct _starpu_mp_node *, void *, int);
	void (*nt_recv)		(const struct _starpu_mp_node *, void *, int);

	/*signal*/
	void (*mp_wait)		   (struct _starpu_mp_node *);
	void (*mp_signal)	   (const struct _starpu_mp_node *);

	/** Data transfers */
	void (*dt_send)		    (const struct _starpu_mp_node *, void *, int, void *);
	void (*dt_recv)		    (const struct _starpu_mp_node *, void *, int, void *);
	void (*dt_send_to_device)   (const struct _starpu_mp_node *, int, void *, int, void *);
	void (*dt_recv_from_device) (const struct _starpu_mp_node *, int, void *, int, void *);

	/** Test async transfers */
	unsigned int (*dt_test) (struct _starpu_async_channel *);

	void (*(*get_kernel_from_job)	(const struct _starpu_mp_node *,struct _starpu_job *))(void);
	void (*(*lookup)		(const struct _starpu_mp_node *, char*))(void);
	void (*bind_thread)		(const struct _starpu_mp_node *, int,int *,int);
	void (*execute)			(struct _starpu_mp_node *, void *, int);
	void (*allocate)		(const struct _starpu_mp_node *, void *, int);
	void (*free)			(const struct _starpu_mp_node *, void *, int);
	void (*map)			(const struct _starpu_mp_node *, void *, int);
	void (*unmap)			(const struct _starpu_mp_node *, void *, int);
};

struct _starpu_mp_node * _starpu_mp_common_node_create(enum _starpu_mp_node_kind node_kind, int peer_devid) STARPU_ATTRIBUTE_MALLOC;

void _starpu_mp_common_node_destroy(struct _starpu_mp_node *node);

void _starpu_mp_common_send_command(const struct _starpu_mp_node *node,
				    const enum _starpu_mp_command command,
				    void *arg, int arg_size);

void _starpu_nt_common_send_command(const struct _starpu_mp_node *node,
				    const enum _starpu_mp_command command,
				    void *arg, int arg_size);

enum _starpu_mp_command _starpu_mp_common_recv_command(const struct _starpu_mp_node *node, void **arg, int *arg_size);

enum _starpu_mp_command _starpu_nt_common_recv_command(const struct _starpu_mp_node *node, void **arg, int *arg_size);

void _starpu_sink_deinit(struct _starpu_mp_node *node);
void _starpu_sink_launch_workers(struct _starpu_mp_node *node);

#endif /* STARPU_USE_MP */

#pragma GCC visibility pop

#endif /* __MP_COMMON_H__ */
