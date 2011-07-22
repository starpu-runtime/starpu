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

#include "socl.h"
#include "graph.h"
#include "event.h"

static pthread_spinlock_t graph_lock;
static graph_node graph_nodes = NULL;


/**
 * Initialize graph structure
 */
void graph_init(void) {
	pthread_spin_init(&graph_lock, PTHREAD_PROCESS_PRIVATE);
}

/**
 * Release graph structure
 */
void graph_destroy(void) {
	pthread_spin_destroy(&graph_lock);
}

/**
 * Initialize a graph node
 */
void graph_node_init(graph_node node) {
	node->id = -1;
	node->next = NULL;
}

/**
 * Store a node in the graph
 */
void graph_store(void * node) {
	pthread_spin_lock(&graph_lock);

	graph_node n = (graph_node)node;
	n->next = graph_nodes;
	graph_nodes = n;

	pthread_spin_unlock(&graph_lock);
}



/**
 * Duplicate a memory area into a fresh allocated buffer
 */
static void * memdupa(const void *p, size_t size) {
	void * s = malloc(size);
	memcpy(s,p,size);
	return s;
}

#define memdup(p, size) ((typeof(p))memdupa(p,size))
#define nullOrDup(name,size) s->name = (name == NULL ? NULL : memdup(name,size))
#define dup(name) s->name = name


node_enqueue_kernel graph_create_enqueue_kernel(char is_task,
		cl_command_queue cq,
		cl_kernel        kernel,
		cl_uint          work_dim,
		const size_t *   global_work_offset,
		const size_t *   global_work_size,
		const size_t *   local_work_size,
		cl_uint          num_events,
		const cl_event * events,
		cl_event *       event,
		cl_uint 		num_args,
		size_t *		arg_sizes,
		enum kernel_arg_type * arg_types,
		void **		args)
{
	node_enqueue_kernel s = malloc(sizeof(struct node_enqueue_kernel_t));
	graph_node_init(&s->node);
	s->node.id = NODE_ENQUEUE_KERNEL;

	dup(is_task);
	dup(cq);
	dup(kernel);
	dup(work_dim);
	nullOrDup(global_work_offset, work_dim*sizeof(size_t));
	nullOrDup(global_work_size, work_dim*sizeof(size_t));
	nullOrDup(local_work_size, work_dim*sizeof(size_t));
	dup(num_events);
	nullOrDup(events, num_events * sizeof(cl_event));
	dup(num_args);
	nullOrDup(arg_sizes, num_args * sizeof(size_t));
	nullOrDup(arg_types, num_args * sizeof(enum kernel_arg_type));
	nullOrDup(args, num_args * sizeof(void*));

	
	if (event != NULL) {
		*event = event_create();
		s->event = event;
	}
	else {
		s->event = NULL;
	}

	return s;
}

#undef nullOrDup
#undef memdup
#undef dup
