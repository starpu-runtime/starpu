/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __RECURSIVE_PERFMODEL_H__
#define __RECURSIVE_PERFMODEL_H__

#include <starpu.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Performance_Model Performance Model
   @{
*/

struct starpu_task;

struct _starpu_recursive_perfmodel_graph_node_list;

/**
 * Structure that represents a task of a subgraph
*/
struct _starpu_recursive_perfmodel_graph_node
{
	struct starpu_codelet *cl;
	size_t data_size;
	size_t parent_datasize;
	uint32_t footprint;
	uint32_t index;
	unsigned level;
	unsigned nb_sucs;
	uint64_t scheduling_data; // a data used for internal scheduling simulation
	struct _starpu_recursive_perfmodel_graph_node_list *node_successors;
};

struct _starpu_recursive_perfmodel_graph_node_list
{
	struct _starpu_recursive_perfmodel_graph_node *node;
	struct _starpu_recursive_perfmodel_graph_node_list *next_node;
};

struct split_description
{
	char split_scheme[10000];
	unsigned first_subtask_index[10000];
	char scheduling[10000];
	unsigned mean_cpu_used; // the mean number of cpu used during the execution
	unsigned nsubtasks_cpu;
	unsigned nsubtasks_gpu;
	double min_occupancy_gpu; // the amount of work needed on GPU to start execution on CPU
	double part_exec_gpu; // the sum of times of tasks executed on GPU divided by the GPU time of the sequential implementation : the win obtained for this task.
	double general_time;
	double sequential_gpu_time;
};

/**
 * Structure that represents for a codelet, a footprint and a data splitting scheme the corresponding subgraph
*/
struct _starpu_recursive_perfmodel_subgraph
{
	struct starpu_codelet *cl;
	uint32_t footprint;
	size_t data_size;
	const char *name;
	unsigned subgraph_initialisation_is_finished;
	unsigned created_during_execution;
	unsigned nb_subtasks;
	void *subgraph_initialisation_data;
	struct _starpu_recursive_perfmodel_graph_node_list *nodes;
	struct _starpu_recursive_perfmodel_graph_node_list *last_node;
	struct split_description *splittings;
	struct split_description *best_split;
	unsigned one_split_exist;
	starpu_pthread_mutex_t subgraph_mutex;
	unsigned one_already_split;
};

struct starpu_recursive_perfmodel_subgraph_list
{
	struct _starpu_recursive_perfmodel_subgraph *graph;
	struct starpu_recursive_perfmodel_subgraph_list *next;
};

void _starpu_recursive_perfmodel_init();

/**
   Return the subgraph associated to the given task, according to its footprint
*/
struct _starpu_recursive_perfmodel_subgraph *_starpu_recursive_perfmodel_get_subgraph_from_task(struct starpu_task *task);

/**
   Create an empty subgraph associated to the given task, according to its footprint
*/
struct _starpu_recursive_perfmodel_subgraph *_starpu_recursive_perfmodel_create_subgraph_from_task(struct starpu_task *task);

/**
   Add the subtask to the subgraph associated to the given task
*/
void _starpu_recursive_perfmodel_add_subtask_to_subgraph(struct _starpu_recursive_perfmodel_subgraph *parent_subgraph, struct starpu_task *subtask);

void _starpu_recursive_perfmodel_dump_created_subgraphs();
void _starpu_recursive_perfmodel_record_codelet(struct starpu_codelet *cl);
void _starpu_recursive_perfmodel_get_best_schedule_alap(struct starpu_task *task, struct _starpu_recursive_perfmodel_subgraph *graph, double *best_time, double *ncuda_mean_used, double *ncpu_mean_used);

#ifdef __cplusplus
}
#endif

#endif /* __RECURSIVE_PERFMODEL_H__ */
