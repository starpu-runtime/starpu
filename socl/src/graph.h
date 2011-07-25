/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

#ifndef SOCL_GRAPH_H
#define SOCL_GRAPH_H

#include "socl.h"

typedef struct graph_node_t * graph_node;

struct graph_node_t {
	int 		id; 		/* Kind of node */
	graph_node 	next; 		/* Linked-list of nodes... */
	cl_uint 	num_events;	/* Number of dependencies */
	cl_event * 	events;		/* Dependencies */
	cl_event  	event;		/* Event for this node */
};

void graph_init(void);
void graph_destroy(void);
void graph_node_init(graph_node node);
void graph_store(void * node);
void graph_free(void * node);

#define NODE_ENQUEUE_KERNEL 1


typedef struct node_enqueue_kernel_t {
	struct graph_node_t node;

	char 		 is_task; /* Set if clEnqueueTask is used */
	cl_command_queue cq;
	cl_kernel        kernel;
	cl_uint          work_dim;
	const size_t *   global_work_offset;
	const size_t *   global_work_size;
	const size_t *   local_work_size;
	cl_uint 	 num_args;
	size_t *	 arg_sizes;
	enum kernel_arg_type * arg_types;
	void **		 args;
} * node_enqueue_kernel;

node_enqueue_kernel graph_create_enqueue_kernel(char is_task,
		cl_command_queue cq,
		cl_kernel        kernel,
		cl_uint          work_dim,
		const size_t *   global_work_offset,
		const size_t *   global_work_size,
		const size_t *   local_work_size,
		cl_uint          num_events,
		const cl_event * events,
		cl_uint		 num_args,
		size_t *	 arg_sizes,
		enum kernel_arg_type * arg_types,
		void **		args);

cl_int graph_play_enqueue_kernel(node_enqueue_kernel n);

#endif /* SOCL_GRAPH_H */
