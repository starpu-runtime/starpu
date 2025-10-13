/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/workers.h>
#include <core/jobs.h>
#include <core/task.h>
#include <stdio.h>
#include <core/perfmodel/recursive_perfmodel.h>

#ifdef STARPU_RECURSIVE_TASKS

static struct starpu_recursive_perfmodel_subgraph_list *all_subgraphs = NULL;
static starpu_pthread_mutex_t subgraph_lock;

void _starpu_recursive_perfmodel_init()
{
	STARPU_PTHREAD_MUTEX_INIT(&subgraph_lock, NULL);
}

static struct _starpu_recursive_perfmodel_subgraph *_starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(struct starpu_perfmodel *perfmodel, uint32_t footprint)
{
	struct starpu_recursive_perfmodel_subgraph_list *node_lst = perfmodel->recursive_graphs;
	while (node_lst != NULL)
	{
		if (node_lst->graph->footprint == footprint)
		{
			return node_lst->graph;
		}
		node_lst = node_lst->next;
	}
	return NULL;
}

struct _starpu_recursive_perfmodel_subgraph *_starpu_recursive_perfmodel_get_subgraph_from_task(struct starpu_task *task)
{
	struct _starpu_recursive_perfmodel_subgraph *subgraph = _starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(task->cl->model, starpu_task_data_footprint(task));
	if (subgraph != NULL)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&subgraph->subgraph_mutex);
		task->cl->model->footprint_per_level[_starpu_task_get_level(task)] = starpu_task_data_footprint(task);
		if (subgraph != NULL && _starpu_task_get_level(task) == 0 && subgraph->subgraph_initialisation_is_finished && subgraph->splittings == NULL && starpu_worker_get_count_by_type(STARPU_CUDA_WORKER))
		{
//			fprintf(stderr, "Gen Dag %s\n", task->name);
			double best_time, ncuda_mean_used, ncpu_mean_used;
			_starpu_recursive_perfmodel_get_best_schedule_alap(task, subgraph, &best_time, &ncuda_mean_used, &ncpu_mean_used);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&subgraph->subgraph_mutex);
	}
	return subgraph;
}

static struct starpu_perfmodel_arch *arch_cuda = NULL;
double starpu_codelet_expected_length_by_level(struct starpu_codelet *cl, int level)
{
	if (cl->model == NULL)
		return -1.;
	struct starpu_perfmodel *perfmodel = cl->model;
	uint32_t footprint = perfmodel->footprint_per_level[level];

	if (arch_cuda == NULL && starpu_cuda_worker_get_count() > 0)
		arch_cuda = starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), STARPU_NMAX_SCHED_CTXS);
	double cuda_time = arch_cuda ? (cl->cuda_funcs[0] != NULL) ? starpu_perfmodel_history_based_expected_perf(perfmodel, arch_cuda, footprint) : -1.:  -1.;

	if (cuda_time < 0)
	{
		struct _starpu_recursive_perfmodel_subgraph *subgraph = _starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(perfmodel, footprint);
		if (subgraph && subgraph->splittings)
			cuda_time = subgraph->splittings->general_time;
	}

	return cuda_time;
}

#define NMAX_READ_ACCESSORS 16
struct _subgraph_initialisation_data
{
	starpu_data_handle_t handle;
	struct _starpu_recursive_perfmodel_graph_node *r_accessors[NMAX_READ_ACCESSORS];
	unsigned n_r_accessors;
	struct _starpu_recursive_perfmodel_graph_node *w_accessor;
	struct _subgraph_initialisation_data *next;
};

struct _starpu_codelet_list
{
	struct starpu_codelet *cl;
	struct _starpu_codelet_list *next;
	unsigned model_is_loaded;
};
static struct _starpu_codelet_list *registered_codelet;

/**
 * Create an empty subgraph associated to the given task, according to its footprint
 * This function is supposed to be protected by the perfmodel lock state->model_rwlock
 */
struct _starpu_recursive_perfmodel_subgraph *_starpu_recursive_perfmodel_create_subgraph_from_task(struct starpu_task *task)
{
	if (!task->cl || ! task->cl->model)
		return NULL;
	struct _starpu_recursive_perfmodel_subgraph *subgraph;
	starpu_malloc((void**)&subgraph, sizeof(*subgraph));
	memset(subgraph, 0, sizeof(*subgraph));
	subgraph->cl = task->cl;
	subgraph->subgraph_initialisation_is_finished = 0;
	subgraph->footprint = starpu_task_data_footprint(task);
	subgraph->nodes = NULL;
	subgraph->last_node = NULL;
	subgraph->created_during_execution = 1;
	subgraph->nb_subtasks = 0;
	subgraph->splittings = NULL;
	subgraph->subgraph_initialisation_data = NULL;
	STARPU_PTHREAD_MUTEX_INIT0(&subgraph->subgraph_mutex, NULL);
	subgraph->name = task->cl->name != NULL ? task->cl->name : task->name;
	struct starpu_perfmodel *model = task->cl->model;
	subgraph->data_size = _starpu_job_get_data_size(model, NULL, 0, _starpu_get_job_associated_to_task(task));

	struct starpu_recursive_perfmodel_subgraph_list *node_lst;
	starpu_malloc((void**) &node_lst, sizeof(*node_lst));
	node_lst->graph = subgraph;
	node_lst->next = model->recursive_graphs;
	model->recursive_graphs = node_lst;

	STARPU_PTHREAD_MUTEX_LOCK(&subgraph_lock);
	struct starpu_recursive_perfmodel_subgraph_list *global_lst;
	starpu_malloc((void**) &global_lst, sizeof(*global_lst));
	global_lst->graph = subgraph;
	global_lst->next = all_subgraphs;
	all_subgraphs = global_lst;
	STARPU_PTHREAD_MUTEX_UNLOCK(&subgraph_lock);
	return subgraph;
}

static struct _subgraph_initialisation_data *__starpu_get_subgraph_data_from_handle(struct _starpu_recursive_perfmodel_subgraph *graph, starpu_data_handle_t handle, struct _subgraph_initialisation_data *cur_data)
{
	if (cur_data == NULL)
	{
		struct _subgraph_initialisation_data *new_data;
		starpu_malloc((void**) &new_data, sizeof(*new_data));
		new_data->handle = handle;
		new_data->next = graph->subgraph_initialisation_data;
		new_data->w_accessor = NULL;
		new_data->n_r_accessors = 0;
		graph->subgraph_initialisation_data = new_data;
		return new_data;
	}
	if (cur_data->handle == handle)
	{
		return cur_data;
	}
	return __starpu_get_subgraph_data_from_handle(graph, handle, cur_data->next);
}

static struct _subgraph_initialisation_data *_starpu_get_subgraph_data_from_handle(struct _starpu_recursive_perfmodel_subgraph *graph, starpu_data_handle_t handle)
{
	return __starpu_get_subgraph_data_from_handle(graph, handle, graph->subgraph_initialisation_data);
}

static void _starpu_recursive_perfmodel_graph_node_add_successors(struct _starpu_recursive_perfmodel_graph_node *pred, struct _starpu_recursive_perfmodel_graph_node *suc)
{
	struct _starpu_recursive_perfmodel_graph_node_list *lst = pred->node_successors;
	while (lst != NULL)
	{
		if (lst->node == suc)
			return;
		lst = lst->next_node;
	}
	struct _starpu_recursive_perfmodel_graph_node_list *new_list;
	starpu_malloc((void**) &new_list, sizeof(*new_list));
	new_list->node = suc;
	new_list->next_node = pred->node_successors;
	pred->node_successors = new_list;
	pred->nb_sucs ++;
}

#if 0
static void _starpu_recursive_perfmodel_print_node_list_debug(struct _starpu_recursive_perfmodel_graph_node_list *nodes)
{
	STARPU_PTHREAD_MUTEX_LOCK(&subgraph_lock);
	fprintf(stderr, "NODES : \n");
	struct _starpu_recursive_perfmodel_graph_node_list *tmp = nodes;
	while(tmp)
	{
		fprintf(stderr, "%s(%p) : %u\n", tmp->node->cl->model->symbol, tmp->node, tmp->node->index);
		tmp = tmp->next_node;
	}
	fprintf(stderr, "\nEDGES :\n");
	tmp = nodes;
	while(tmp)
	{
		struct _starpu_recursive_perfmodel_graph_node *cur = tmp->node;
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = cur->node_successors;
		while(sucs)
		{
			fprintf(stderr, "(%s (%p): %u) -> (%s(%p) : %u)\n", cur->cl->model->symbol, cur, cur->index, sucs->node->cl->model->symbol, sucs->node, sucs->node->index);
			sucs = sucs->next_node;
		}
		tmp = tmp->next_node;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&subgraph_lock);
}
#endif

/**
   Add the subtask to the subgraph associated to the given task
*/
void _starpu_recursive_perfmodel_add_subtask_to_subgraph(struct _starpu_recursive_perfmodel_subgraph *parent_subgraph, struct starpu_task *subtask)
{
	if (subtask->cl == NULL)
		return;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(subtask);
	unsigned index;

	struct _starpu_recursive_perfmodel_graph_node *cur_node;
	struct starpu_perfmodel *model = subtask->cl->model;
	starpu_malloc((void**) &cur_node, sizeof(*cur_node));
	cur_node->footprint = starpu_task_data_footprint(subtask);
	cur_node->data_size = _starpu_job_get_data_size(model, NULL, 0, _starpu_get_job_associated_to_task(subtask));
	cur_node->cl = subtask->cl;
	cur_node->nb_sucs = 0;
	cur_node->node_successors = NULL;
	cur_node->index = parent_subgraph->nb_subtasks;

	struct _starpu_recursive_perfmodel_graph_node_list *new_node_list;
	starpu_malloc((void**) &new_node_list, sizeof(*new_node_list));
	new_node_list->node = cur_node;
	new_node_list->next_node = NULL;
	if (parent_subgraph->last_node == NULL)
	{
		parent_subgraph->nodes = new_node_list;
	}
	else
	{
		parent_subgraph->last_node->next_node = new_node_list;
	}
	parent_subgraph->last_node = new_node_list;
	parent_subgraph->nb_subtasks++;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(subtask, index);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(subtask, index);
		struct _subgraph_initialisation_data *corresponding_data = _starpu_get_subgraph_data_from_handle(parent_subgraph, handle);
		if (mode & STARPU_W)
		{
				// We add dependencies from each reader and writer
				unsigned suc_ind;
				for (suc_ind = 0; suc_ind < corresponding_data->n_r_accessors; suc_ind++)
				{
					struct _starpu_recursive_perfmodel_graph_node *reader = corresponding_data->r_accessors[suc_ind];
					_starpu_recursive_perfmodel_graph_node_add_successors(reader, cur_node);
				}
				struct _starpu_recursive_perfmodel_graph_node *writer = corresponding_data->w_accessor;
				if (writer != NULL)
				{
					_starpu_recursive_perfmodel_graph_node_add_successors(writer, cur_node);
				}
				corresponding_data->n_r_accessors = 0;
				corresponding_data->w_accessor = cur_node;
		}
		else if (mode & STARPU_R)
		{
			STARPU_ASSERT(corresponding_data->n_r_accessors < NMAX_READ_ACCESSORS-1);
			struct _starpu_recursive_perfmodel_graph_node *writer = corresponding_data->w_accessor;
			if (writer != NULL)
			{
				_starpu_recursive_perfmodel_graph_node_add_successors(writer, cur_node);
			}
			corresponding_data->r_accessors[corresponding_data->n_r_accessors] = cur_node;
			corresponding_data->n_r_accessors ++;
		}
	}
}

static void _starpu_recursive_perfmodel_dump_graph(struct _starpu_recursive_perfmodel_subgraph *graph)
{
	char filename[256];
	snprintf(filename, 255, "graph_%s_%u_%lu.graph", graph->cl->model->symbol, graph->footprint, graph->data_size);
	FILE *f = fopen(filename, "w");

	// We first dump all nodes
	fprintf(f, ">>>NODES :\n");
	struct _starpu_recursive_perfmodel_graph_node_list *lst_nodes = graph->nodes;
	while(lst_nodes != NULL)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = lst_nodes->node;
		if (node->cl->model == &starpu_perfmodel_nop)
			break;
		const char *ptr = node->cl->model->symbol;
		while(*ptr != '\0')
		{
			// We do not want '_' or '\n' on symbol because it becames impossible to read after
			STARPU_ASSERT(*ptr != '_' && *ptr != '\n');
			ptr ++;
		}
		fprintf(f, "%s_%u_%u_%lu\n", node->cl->model->symbol, node->footprint, node->index, node->data_size);
		lst_nodes = lst_nodes->next_node;
	}
	fprintf(f, ">>>EDGES :\n");

	lst_nodes = graph->nodes;
	while(lst_nodes != NULL)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = lst_nodes->node;
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = node->node_successors;
		while(sucs != NULL)
		{
			struct _starpu_recursive_perfmodel_graph_node *suc = sucs->node;
			fprintf(f, "%s_%u_%u_%lu -> %s_%u_%u_%lu\n", node->cl->model->symbol, node->footprint, node->index, node->data_size, suc->cl->model->symbol, suc->footprint, suc->index, suc->data_size);
			sucs = sucs->next_node;
		}
		lst_nodes = lst_nodes->next_node;
	}
	fclose(f);

	// We also dump general infos
	snprintf(filename, 255, "graph_%s.gen", graph->cl->model->symbol);
	f = fopen(filename, "a");
	fprintf(f, "%u\t%lu\n", graph->footprint, graph->data_size);
	fclose(f);
}

void _starpu_recursive_perfmodel_dump_created_subgraphs()
{
	struct starpu_recursive_perfmodel_subgraph_list *graphs = all_subgraphs;
	while(graphs != NULL)
	{
		if (graphs->graph->created_during_execution)
		{
			_starpu_recursive_perfmodel_dump_graph(graphs->graph);
		}
		graphs = graphs->next;
	}
}

// Returns the number of chars read
static unsigned __read_one_symbol_footprint_datasize(const char *string, char *symbol, uint32_t *footprint, uint32_t *index, size_t *datasize)
{
	unsigned i = 0;
	while(string[i] != '_')
	{
		symbol[i] = string[i];
		i++;
	}
	symbol[i] = '\0';
	i++;
	char *endptr;
	*footprint = strtoul(string + i, &endptr, 10);
	i += (endptr - string - i + 1); // +1 because we jump '_'
	*index = strtoul(string + i, &endptr, 10);
	i += (endptr - string - i + 1); // +1 because we jump '_
	*datasize = strtoul(string + i, &endptr, 10);
	i += (endptr - string - i);
	return i;
}

static int _starpu_recursive_perfmodel_read_one_subgraph_from_disk(struct starpu_codelet *codelet, struct _starpu_recursive_perfmodel_subgraph *subgraph, uint32_t footprint, size_t data_size)
{
	subgraph->cl = codelet;
	subgraph->footprint = footprint;
	subgraph->nodes = NULL;
	subgraph->created_during_execution = 0;
	subgraph->subgraph_initialisation_data = NULL;
	subgraph->name = codelet->name;
	subgraph->data_size = data_size;
	subgraph->nb_subtasks = 0;
	subgraph->splittings = NULL;
	subgraph->subgraph_initialisation_is_finished=0;
	STARPU_PTHREAD_MUTEX_INIT0(&subgraph->subgraph_mutex, NULL);

	struct starpu_perfmodel *model = codelet->model;
	struct starpu_recursive_perfmodel_subgraph_list *node_lst;
	starpu_malloc((void**) &node_lst, sizeof(*node_lst));
	node_lst->graph = subgraph;
	node_lst->next = model->recursive_graphs;
	model->recursive_graphs = node_lst;


	struct starpu_recursive_perfmodel_subgraph_list *global_lst;
	starpu_malloc((void**)&global_lst, sizeof(*global_lst));
	global_lst->graph = subgraph;
	global_lst->next = all_subgraphs;
	all_subgraphs = global_lst;

	// Structure is initialized
	// Now we read the graph file to rebuild the graph
	char filename[256];
	snprintf(filename, 255, "graph_%s_%u_%lu.graph", model->symbol, footprint, data_size);
	FILE *f = fopen(filename, "r");
	STARPU_ASSERT_MSG(f, "Error: corrupted general graph file\n");

	// We first build all nodes
	char string[512];
	fgets(string, 256, f);
	STARPU_ASSERT_MSG(strcmp(string, ">>>NODES :\n") == 0, "Corrupted file %s\n", filename);
	struct _starpu_recursive_perfmodel_graph_node_list *lst_nodes = NULL, *end_lst = NULL;
	while(1)
	{
		struct _starpu_recursive_perfmodel_graph_node *node;
		starpu_malloc((void**) &node, sizeof(*node));
		char symbol[256];
		fgets(string, 512, f);
		if (strcmp(string, ">>>EDGES :\n") == 0)
		{
			starpu_free(node);
			break;
		}
		subgraph->nb_subtasks++;
		/* Else we have a symbol to read */
		__read_one_symbol_footprint_datasize(string, symbol, &node->footprint, &node->index, &node->data_size);
//		fprintf(stderr, "Read one %s\n", string);
		// We have the symbol, now we can recover it from the registered_codelet
		struct starpu_codelet *good_cl = NULL;
		struct _starpu_codelet_list *lst_cl = registered_codelet;
		while(lst_cl != NULL && !good_cl)
		{
			if (strcmp(lst_cl->cl->model->symbol, symbol) == 0)
			{
				good_cl = lst_cl->cl;
			}
			lst_cl = lst_cl->next;
		}
		if (!good_cl)
		{
			return 0;
		}


		node->cl = good_cl;
		node->nb_sucs = 0;
		node->node_successors = NULL;
		struct _starpu_recursive_perfmodel_graph_node_list *new_lst_node;
		starpu_malloc((void**)&new_lst_node, sizeof(*new_lst_node));
		new_lst_node->node = node;
		new_lst_node->next_node = NULL;
		if (lst_nodes == NULL)
		{
			lst_nodes = new_lst_node;
		}
		else
		{
			end_lst->next_node = new_lst_node;
		}
		end_lst = new_lst_node;
	}
	subgraph->nodes = lst_nodes;

	while(1)
	{
		char symbol0[256], symbol1[256];
		uint32_t footprint0, footprint1;
		uint32_t index0, index1;
		size_t datasize0, datasize1;
		if (fgets(string, 512, f) == NULL)
		{
			break; // EOF
		}
		unsigned read = __read_one_symbol_footprint_datasize(string, symbol0, &footprint0, &index0, &datasize0);
		__read_one_symbol_footprint_datasize(string + read + 4, symbol1, &footprint1, &index1, &datasize1);
		struct _starpu_recursive_perfmodel_graph_node *pred = NULL, *suc = NULL;
		lst_nodes = subgraph->nodes;
		while(lst_nodes != NULL && (!pred || !suc))
		{
			struct _starpu_recursive_perfmodel_graph_node *node = lst_nodes->node;
			if (node->index == index0)
			{
				pred = node;
			}
			if (node->index == index1)
			{
				suc = node;
			}
			lst_nodes = lst_nodes->next_node;
		}
		STARPU_ASSERT_MSG(suc && pred, "Corrupted graph file\n");
		_starpu_recursive_perfmodel_graph_node_add_successors(pred, suc);
	}
	fclose(f);

	subgraph->subgraph_initialisation_is_finished = 1;
	return 1;
}

static void _starpu_recursive_perfmodel_free_graph(struct _starpu_recursive_perfmodel_graph_node_list *nodes)
{
	while (nodes)
	{
		// we have to free all nodes of successors and then all nodes
		struct _starpu_recursive_perfmodel_graph_node *node = nodes->node;
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = node->node_successors;
		while(sucs)
		{
				struct _starpu_recursive_perfmodel_graph_node_list *tmp = sucs->next_node;
				starpu_free(sucs);
				sucs = tmp;
		}
		struct _starpu_recursive_perfmodel_graph_node_list *tmp = nodes->next_node;
		starpu_free(nodes->node);
		starpu_free(nodes);
		nodes = tmp;
	}
}

#if 0
static void _starpu_recursive_perfmodel_invert_graph(struct _starpu_recursive_perfmodel_graph_node_list *nodes_to_invert, struct _starpu_recursive_perfmodel_graph_node_list **inverted_nodes)
{
	struct _starpu_recursive_perfmodel_graph_node_list *antigraph = NULL, *tmp_antigraph = NULL;
	struct _starpu_recursive_perfmodel_graph_node_list *nodes_ptr = nodes_to_invert;
	while (nodes_ptr)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = nodes_ptr->node;
		struct _starpu_recursive_perfmodel_graph_node *antinode;
		starpu_malloc((void**) &antinode, sizeof(*antinode));
		antinode->cl = node->cl;
		antinode->data_size = node->data_size;
		antinode->footprint = node->footprint;
		antinode->index = node->index;
		antinode->nb_sucs = 0;
		antinode->node_successors = NULL;

		struct _starpu_recursive_perfmodel_graph_node_list *new_node;
		starpu_malloc((void**)&new_node, sizeof(*new_node));
		new_node->node = antinode;
		new_node->next_node = NULL;
		if (tmp_antigraph)
		{
			tmp_antigraph->next_node = new_node;
			tmp_antigraph = new_node;
		}
		else
		{
			tmp_antigraph = new_node;
			antigraph = new_node;
		}

		nodes_ptr = nodes_ptr->next_node;
	}
	// Now consider all edges
	nodes_ptr = nodes_to_invert;
	tmp_antigraph = antigraph;
	while (nodes_ptr)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = nodes_ptr->node;
		struct _starpu_recursive_perfmodel_graph_node *antinode = tmp_antigraph->node;
		STARPU_ASSERT(node->index == antinode->index);
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = node->node_successors;
		while(sucs)
		{
			struct _starpu_recursive_perfmodel_graph_node *suc = sucs->node;
			// We recover the corresponding antinode
			struct _starpu_recursive_perfmodel_graph_node_list *tmp_antinodes = antigraph;
			struct _starpu_recursive_perfmodel_graph_node *pred_antinode = NULL;
			while (tmp_antinodes && !pred_antinode)
			{
				if (tmp_antinodes->node->index == suc->index)
				{
					pred_antinode = tmp_antinodes->node;
				}
				tmp_antinodes = tmp_antinodes->next_node;
			}
			STARPU_ASSERT(pred_antinode);
			_starpu_recursive_perfmodel_graph_node_add_successors(pred_antinode, antinode);
			sucs = sucs->next_node;
		}
		nodes_ptr = nodes_ptr->next_node;
		tmp_antigraph = tmp_antigraph->next_node;
	}
	*inverted_nodes = antigraph;
}
#endif

static struct _starpu_recursive_perfmodel_graph_node_list *_starpu_recursive_perfmodel_copy_all_nodes(struct _starpu_recursive_perfmodel_graph_node_list *nodes_to_copy)
{
	struct _starpu_recursive_perfmodel_graph_node_list *copy = NULL, *tmp_copy = NULL;
	struct _starpu_recursive_perfmodel_graph_node_list *nodes_ptr = nodes_to_copy;
	while (nodes_ptr)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = nodes_ptr->node;
		struct _starpu_recursive_perfmodel_graph_node *node_cp;
		starpu_malloc((void**) &node_cp, sizeof(*node_cp));
		node_cp->cl = node->cl;
		node_cp->data_size = node->data_size;
		node_cp->parent_datasize = node->parent_datasize;
		node_cp->footprint = node->footprint;
		node_cp->index = node->index;
		node_cp->nb_sucs = 0;
		node_cp->node_successors = NULL;

		struct _starpu_recursive_perfmodel_graph_node_list *new_node;
		starpu_malloc((void**)&new_node, sizeof(*new_node));
		new_node->node = node_cp;
		new_node->next_node = NULL;
		if (tmp_copy)
		{
			tmp_copy->next_node = new_node;
			tmp_copy = new_node;
		}
		else
		{
			tmp_copy = new_node;
			copy = new_node;
		}
		nodes_ptr = nodes_ptr->next_node;
	}
	// Now consider all edges
	nodes_ptr = nodes_to_copy;
	tmp_copy = copy;
	while (nodes_ptr)
	{
		struct _starpu_recursive_perfmodel_graph_node *node = nodes_ptr->node;
		struct _starpu_recursive_perfmodel_graph_node *node_copy = tmp_copy->node;
		STARPU_ASSERT(node->index == node_copy->index);
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = node->node_successors;
		while(sucs)
		{
			struct _starpu_recursive_perfmodel_graph_node *suc = sucs->node;
			// We recover the corresponding node
			struct _starpu_recursive_perfmodel_graph_node_list *ttmp_copy = copy;
			struct _starpu_recursive_perfmodel_graph_node *suc_copy = NULL;
			while (ttmp_copy && !suc_copy)
			{
				if (ttmp_copy->node->index == suc->index)
				{
					suc_copy = ttmp_copy->node;
				}
				ttmp_copy = ttmp_copy->next_node;
			}
			STARPU_ASSERT(suc_copy);
			_starpu_recursive_perfmodel_graph_node_add_successors(node_copy, suc_copy);
			sucs = sucs->next_node;
		}
		nodes_ptr = nodes_ptr->next_node;
		tmp_copy = tmp_copy->next_node;
	}
	return copy;
}

/**
 * Transforms into subgraph if exists and returns it if the transformation has been made
*/
static struct _starpu_recursive_perfmodel_graph_node_list *_starpu_recursive_perfmodel_turn_task_into_subgraph(struct _starpu_recursive_perfmodel_graph_node_list *graph, struct _starpu_recursive_perfmodel_graph_node *node_to_split, struct _starpu_recursive_perfmodel_graph_node_list **next_node_after_split)
{
	struct _starpu_recursive_perfmodel_graph_node_list *tmp_graph = graph;
	int node_exist_on_graph = 0;
	while(tmp_graph && !node_exist_on_graph)
	{
		if (tmp_graph->node == node_to_split)
		{
			node_exist_on_graph = 1;
		}
		tmp_graph = tmp_graph->next_node;
	}
	STARPU_ASSERT_MSG(node_exist_on_graph, "We can only turn into subgraph a task that is on the graph\n");
	struct _starpu_recursive_perfmodel_subgraph *subgraph = _starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(node_to_split->cl->model, node_to_split->footprint);
	if (!subgraph || !subgraph->subgraph_initialisation_is_finished)
	{
		return NULL;
	}
	// we can copy graph now and returns it at the end
	struct _starpu_recursive_perfmodel_graph_node_list *new_graph = _starpu_recursive_perfmodel_copy_all_nodes(graph);
	struct _starpu_recursive_perfmodel_graph_node *node_to_split_cp = NULL;
	struct _starpu_recursive_perfmodel_graph_node_list *tmp_new_graph = new_graph;
	while(tmp_new_graph && !node_to_split_cp)
	{
		if (tmp_new_graph->node->index == node_to_split->index)
		{
			node_to_split_cp = tmp_new_graph->node;
			*next_node_after_split = tmp_new_graph->next_node;
		}
		tmp_new_graph = tmp_new_graph->next_node;
	}
	STARPU_ASSERT(node_to_split_cp);
	struct _starpu_recursive_perfmodel_graph_node_list *subgraph_nodes = _starpu_recursive_perfmodel_copy_all_nodes(subgraph->nodes);
	struct _starpu_recursive_perfmodel_graph_node_list *tmp_subgraph = subgraph_nodes;
	unsigned nb_tasks = 0;
	while(tmp_subgraph)
	{
		tmp_subgraph->node->parent_datasize = node_to_split->data_size;
		tmp_subgraph->node->index = nb_tasks++;
		tmp_subgraph = tmp_subgraph->next_node;
	}

	unsigned *has_preds;
	_STARPU_CALLOC(has_preds, nb_tasks, sizeof(unsigned));
	tmp_subgraph = subgraph_nodes;

	while(tmp_subgraph)
	{
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = tmp_subgraph->node->node_successors;
		while(sucs)
		{
			has_preds[sucs->node->index] = 1;
			sucs = sucs->next_node;
		}
		tmp_subgraph = tmp_subgraph->next_node;
	}
	tmp_new_graph = new_graph;
	// Now, we have to find all predecessors on graph of node_to_split, and add as successors all nodes of subgraph_nodes without predecessors
	while(tmp_new_graph)
	{
		struct _starpu_recursive_perfmodel_graph_node *cur_node = tmp_new_graph->node;
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = cur_node->node_successors;
		int to_split_is_successor = 0;
		while(sucs && !to_split_is_successor)
		{
			if (sucs->node == node_to_split_cp)
			{
				to_split_is_successor = 1;
			}
			sucs = sucs->next_node;
		}
		if (to_split_is_successor)
		{
			tmp_subgraph = subgraph_nodes;
			while(tmp_subgraph)
			{
				if (!has_preds[tmp_subgraph->node->index])
				{
					_starpu_recursive_perfmodel_graph_node_add_successors(cur_node, tmp_subgraph->node);
				}
				tmp_subgraph = tmp_subgraph->next_node;
			}
		}
		tmp_new_graph = tmp_new_graph->next_node;
	}
	// and now, we add each successor of node_to_split to all node without successors of th subgraph
	tmp_subgraph = subgraph_nodes;
	while(tmp_subgraph)
	{
		if (tmp_subgraph->node->nb_sucs == 0)
		{
			struct _starpu_recursive_perfmodel_graph_node_list *sucs = node_to_split_cp->node_successors;
			while(sucs)
			{
				_starpu_recursive_perfmodel_graph_node_add_successors(tmp_subgraph->node, sucs->node);
				sucs = sucs->next_node;
			}
		}
		tmp_subgraph = tmp_subgraph->next_node;
	}
	free(has_preds);

	/* We finally remove node_to_split of the new graph */
	while (new_graph && new_graph->node == node_to_split_cp)
	{
		struct _starpu_recursive_perfmodel_graph_node_list *to_rm = new_graph;
		new_graph = new_graph->next_node;
		starpu_free(to_rm);
	}
	tmp_new_graph = new_graph;
	while (tmp_new_graph != NULL)
	{
		if (tmp_new_graph->next_node != NULL && tmp_new_graph->next_node->node == node_to_split_cp)
		{ // remove next_node
			struct _starpu_recursive_perfmodel_graph_node_list *to_rm = tmp_new_graph->next_node;
			tmp_new_graph->next_node = tmp_new_graph->next_node->next_node;
			starpu_free(to_rm);
		}
		else
		{ // looking at current successors to try to remove next_node
			struct _starpu_recursive_perfmodel_graph_node_list *sucs = tmp_new_graph->node->node_successors;
			while(sucs && sucs->node == node_to_split_cp)
			{
				struct _starpu_recursive_perfmodel_graph_node_list *to_rm = sucs;
				sucs = sucs->next_node;
				starpu_free(to_rm);
			}
			while(sucs != NULL)
			{
				if (sucs->next_node != NULL && sucs->next_node->node == node_to_split_cp)
				{
					struct _starpu_recursive_perfmodel_graph_node_list *to_rm = sucs->next_node;
					sucs->next_node = to_rm->next_node;
					starpu_free(to_rm);
				}
				else
				{
					sucs = sucs->next_node;
				}
			}
			tmp_new_graph = tmp_new_graph->next_node;
		}
	}
	if (!*next_node_after_split)
	{
		*next_node_after_split = subgraph_nodes;
	}
	/* We now add subgraph at the end of new_graph */
	if (new_graph == NULL)
	{
		new_graph = subgraph_nodes;
	}
	else
	{
		tmp_new_graph = new_graph;
		while (tmp_new_graph->next_node != NULL)
		{
			tmp_new_graph = tmp_new_graph->next_node;
		}
		tmp_new_graph->next_node = subgraph_nodes;
	}
	/* And finally we re-index the nodes */
	nb_tasks = 0;
	tmp_new_graph = new_graph;
	while (tmp_new_graph)
	{
		tmp_new_graph->node->index = nb_tasks++;
		tmp_new_graph = tmp_new_graph->next_node;
	}
	return new_graph;
}

static int _starpu_recursive_perfmodel_schedule_graph_asap(struct _starpu_recursive_perfmodel_graph_node_list *nodes_to_schedule, double *best_time, unsigned ncpu_can_use, unsigned ncuda_can_use, double *ncpu_mean_used, double *ncuda_mean_used, double idle_cpu, double idle_gpu, int print_sched, double *minimal_cpu_time_to_add_task_on_cpu, double *sequential_gpu_time, enum starpu_worker_archtype **scheduling, double maximal_task_length_on_cpu, double minimal_task_length_on_cuda, unsigned *nsubtasks_cpu)
{
	struct starpu_perfmodel_arch *arch_cpu = ncpu_can_use > 0 ? starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), STARPU_NMAX_SCHED_CTXS) : NULL,
															 *arch_gpu = ncuda_can_use > 0 ? starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), STARPU_NMAX_SCHED_CTXS) : NULL;
	STARPU_ASSERT_MSG(arch_cpu || arch_gpu, "We need at least one PU to make scheduling\n");
	/* First step is finding the number of tasks to schedule */
	unsigned nb_tasks = 0;
	(*nsubtasks_cpu) = 0;
	struct _starpu_recursive_perfmodel_graph_node_list *tmp_nodes = nodes_to_schedule;
	while(tmp_nodes)
	{
		nb_tasks ++;
		tmp_nodes->node->scheduling_data = 0; // could be schedule anywhere
		tmp_nodes = tmp_nodes->next_node;
	}
	if (print_sched)
		fprintf(stderr, "\n\n");

	/* Second step is init the number of predecessors for each task */
	unsigned *nb_preds_for_each;
	_STARPU_CALLOC(*scheduling, nb_tasks, sizeof(*scheduling));
	_STARPU_CALLOC(nb_preds_for_each, nb_tasks, sizeof(unsigned));
	tmp_nodes = nodes_to_schedule;
	while(tmp_nodes)
	{
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = tmp_nodes->node->node_successors;
		while(sucs)
		{
			nb_preds_for_each[sucs->node->index] ++;
			sucs = sucs->next_node;
		}
		tmp_nodes = tmp_nodes->next_node;
	}
	/* Now we can start the schedule */
	unsigned nb_tasks_scheduled = 0;
	double *min_start_for_each; /* The minimal start, that is updated: each time a task is scheduled, we update the min start for each successors */
	double *min_idle_time_cpus; /* During the execution, the moment a CPU become idle */
	double *min_idle_time_gpus; /* During the execution, the moment a CPU become idle */
	unsigned index;
	_STARPU_CALLOC(min_start_for_each, nb_tasks, sizeof(double));
	_STARPU_CALLOC(min_idle_time_cpus, ncpu_can_use, sizeof(double));
	_STARPU_CALLOC(min_idle_time_gpus, ncuda_can_use, sizeof(double));
	for (index = 0; index < ncpu_can_use; index++)
	{
		min_idle_time_cpus[index] = idle_cpu;
	}
	for (index = 0; index < ncuda_can_use; index++)
	{
		min_idle_time_gpus[index] = idle_gpu;
	}

	double max_end = 0., max_end_cpu = 0.;
	double total_cuda_time = 0., total_cpu_time = 0.;
	double minimal_cpu_time_task_on_gpu = -1;
	unsigned max_used = 0;
	while (nb_tasks_scheduled < nb_tasks)
	{
		// First, we recover a node to schedule
		struct _starpu_recursive_perfmodel_graph_node *to_sched = NULL;
		tmp_nodes = nodes_to_schedule;
		double min_start = 0.;
		while(tmp_nodes)
		{
			if (nb_preds_for_each[tmp_nodes->node->index] == 0 && (to_sched == NULL || min_start > min_start_for_each[tmp_nodes->node->index]))
			{
				to_sched = tmp_nodes->node;
				min_start = min_start_for_each[to_sched->index];
			}
			tmp_nodes = tmp_nodes->next_node;
		}
		nb_preds_for_each[to_sched->index] = 1; // we not reconsider it next time
		// We thus schedule to_sched
		min_start = min_start_for_each[to_sched->index];
		// Take the first idle CPU that have the min idle moment higher to min_start
		unsigned best_cpu_index = 0, cpu_index;
		for(cpu_index=0; cpu_index < ncpu_can_use; cpu_index++)
		{
			if (min_idle_time_cpus[cpu_index] < min_idle_time_cpus[best_cpu_index] && (min_idle_time_cpus[cpu_index] >= min_start || min_idle_time_cpus[best_cpu_index] > min_start))
			{
				best_cpu_index = cpu_index;
			}
		}
		unsigned best_cuda_index = 0, cuda_index;
		for(cuda_index=0; cuda_index < ncuda_can_use; cuda_index++)
		{
			if (min_idle_time_gpus[cuda_index] < min_idle_time_gpus[best_cuda_index] && (min_idle_time_gpus[cuda_index] >= min_start || min_idle_time_gpus[best_cuda_index] > min_start))
				best_cuda_index = cuda_index;
		}
		double cuda_time = arch_gpu ? (to_sched->cl->cuda_funcs[0] != NULL) ? starpu_perfmodel_history_based_expected_perf(to_sched->cl->model, arch_gpu, to_sched->footprint) : -1.:  -1.;
		double cpu_time = arch_cpu ? 1.4*starpu_perfmodel_history_based_expected_perf(to_sched->cl->model, arch_cpu, to_sched->footprint)*1 : -1.;
		double real_cuda_time = cuda_time, real_cpu_time = cpu_time;
		if (cuda_time > minimal_task_length_on_cuda && (to_sched->scheduling_data & ((1<<2))))
		{
			// Mandatory to schedule this node on CUDA
			real_cpu_time = -1.;
		}
		if (cuda_time > minimal_task_length_on_cuda && to_sched->scheduling_data & (1<<1))
		{ // This node has a predecessor on CPU : we add to gpu_time transfer of he whole data
//			real_cuda_time += starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), to_sched->data_size);
//			real_cuda_time += starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), to_sched->data_size);
//				fprintf(stderr, "transfer takes %lf\n", starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), to_sched->data_size));
		}
		if (cpu_time > 0. && !(to_sched->scheduling_data & (1<<1)) && minimal_task_length_on_cuda > 0.)
		{
			// Nothing has been decided. We should add at CPU time something since previous task (from another task) were probably on cuda device
			real_cpu_time += starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), to_sched->parent_datasize);
			real_cpu_time += starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), to_sched->data_size);
		}
		double end_cpu = (min_idle_time_cpus[best_cpu_index] > min_start ? min_idle_time_cpus[best_cpu_index] : min_start) + real_cpu_time;
		double end_gpu = (min_idle_time_gpus[best_cuda_index] > min_start ? min_idle_time_gpus[best_cuda_index] : min_start) + real_cuda_time;
		double end;
		if (real_cpu_time >= 0. && (end_cpu <= end_gpu || real_cuda_time < 0. || cuda_time < minimal_task_length_on_cuda))
		{
			// Scheduling this node on CPU
			if (print_sched)
				fprintf(stderr, "node %s (%u) in CPU %u at %lf, and takes %lf\n", to_sched->cl->name, to_sched->index, best_cpu_index, min_idle_time_cpus[best_cpu_index] > min_start ? min_idle_time_cpus[best_cpu_index] : min_start, cpu_time);
			min_idle_time_cpus[best_cpu_index] = end_cpu;
			end = end_cpu;
			total_cpu_time += cpu_time;
			(*scheduling)[to_sched->index] = STARPU_CPU_WORKER;
			if (!(to_sched->scheduling_data & (1<<1)))
				to_sched->scheduling_data += (1 << 1); // CPU
			(*nsubtasks_cpu) ++;
			if (end > max_end_cpu)
				max_end_cpu = end;
			if (best_cpu_index+1 > max_used)
				max_used = best_cpu_index+1;
		}
		else
		{
			// Scheduling this node on CUDA
			if (print_sched)
				fprintf(stderr, "node %s (%u) in CUDA %u at %lf, and takes %lf\n", to_sched->cl->name, to_sched->index, best_cuda_index, min_idle_time_gpus[best_cuda_index] > min_start ? min_idle_time_gpus[best_cuda_index] : min_start, cuda_time);
			min_idle_time_gpus[best_cuda_index] = end_gpu;
			end = end_gpu;
			total_cuda_time += cuda_time;
			(*scheduling)[to_sched->index] = STARPU_CUDA_WORKER;
			if ( cpu_time > 0. && (minimal_cpu_time_task_on_gpu < 0. || cpu_time < minimal_cpu_time_task_on_gpu))
			{
				minimal_cpu_time_task_on_gpu = cpu_time;
			}
			if (!(to_sched->scheduling_data & (1<<2)))
				to_sched->scheduling_data += (1 << 2); // GPU
		}
		struct _starpu_recursive_perfmodel_graph_node_list *sucs = to_sched->node_successors;
		while(sucs)
		{
			if ((to_sched->scheduling_data & (1<<2)) && !(sucs->node->scheduling_data & (1<<2)))
			{
				if (print_sched)
					fprintf(stderr, "Enforce %s %u to be on cuda\n", sucs->node->cl->name, sucs->node->index);
				sucs->node->scheduling_data += (1<<2);
			}
			else if ((to_sched->scheduling_data & (1<<1) ) && !(sucs->node->scheduling_data & (1<<1)) )
			{
				if (print_sched)
					fprintf(stderr, "%s %u can be on cpu\n", sucs->node->cl->name, sucs->node->index);
				sucs->node->scheduling_data += (1<<1);
			}

			min_start_for_each[sucs->node->index] = end;
			nb_preds_for_each[sucs->node->index] --;
			sucs = sucs->next_node;
		}
		if (end > max_end)
		{
			max_end = end;
		}
		nb_tasks_scheduled ++;
	}
	*best_time = max_end;
	*ncpu_mean_used =  /*(double)max_used ;// */max_end_cpu > 0. ? total_cpu_time / max_end_cpu: 0.;
	*ncuda_mean_used = total_cuda_time / max_end;
	free(min_start_for_each);
	free(min_idle_time_gpus);
	free(min_idle_time_cpus);
	*minimal_cpu_time_to_add_task_on_cpu = minimal_cpu_time_task_on_gpu > 0. && (minimal_cpu_time_task_on_gpu < maximal_task_length_on_cpu && maximal_task_length_on_cpu > 0) ? minimal_cpu_time_task_on_gpu : -1.;
	*sequential_gpu_time = total_cuda_time;
	return 1;
}

#define SPLIT_INDEX(cpu, part) ((cpu)*10+part)

static void __get_best_time_rec_version_from_nodes(struct starpu_task *task_to_sched, struct _starpu_recursive_perfmodel_graph_node_list *nodes_to_schedule, unsigned index, char *split_scheme, unsigned *first_subtask_index, unsigned first_index_free, struct _starpu_recursive_perfmodel_graph_node_list *first_node_we_can_split, struct split_description *split_description)
{
	// split description[i][j] represents the best split in term of completion time for a mean usage of i cpus that represents a sequential use of GPU betweed j*10 and (j+1)*10 % of the sequential execution of the task
	if (first_node_we_can_split == NULL)
	{
		// We try the scheduling
		split_scheme[index] = '\0';
		unsigned nsubtasks_cpu = 0;
		enum starpu_worker_archtype *scheduling;
		double time, minimal_cpu, sequential_gpu;
		double ncpu_mean_used, ncuda_mean_used;
		_starpu_recursive_perfmodel_schedule_graph_asap(nodes_to_schedule, &time, starpu_cpu_worker_get_count(), starpu_cuda_worker_get_count(), &ncpu_mean_used, &ncuda_mean_used, 0., 0., 0, &minimal_cpu, &sequential_gpu, &scheduling, -1, 0., &nsubtasks_cpu);
//		fprintf(stderr, "For task %s, for scheme %s, we can finish the execution at %lf, using a mean of %d cores during the execution\n", task_to_sched->name, split_scheme, time, ncpu_mean_used);
		if ( split_description->general_time < 0. || time < split_description->general_time)
		{
			strcpy(split_description->split_scheme, split_scheme);
			memcpy(split_description->first_subtask_index, first_subtask_index, sizeof(unsigned)*10000);
			split_description->mean_cpu_used = (int) ncpu_mean_used;
			split_description->min_occupancy_gpu = 0.;
			split_description->part_exec_gpu = 100;
			split_description->general_time = time;
			split_description->nsubtasks_cpu = nsubtasks_cpu;
			split_description->nsubtasks_gpu = 0;
			split_description->sequential_gpu_time = sequential_gpu;
			unsigned split_index;
			struct _starpu_recursive_perfmodel_graph_node_list *tmp_nodes =  nodes_to_schedule;
			for (split_index=0; split_index < index; split_index++)
			{
				if (split_scheme[split_index] == '1')
				{
					//fprintf(stderr, "Index %u is split : CPU\n", split_index);
					split_description->scheduling[split_index] = '0';
				}
				else
				{
					//fprintf(stderr, "Index %u correspond to a %s, on %s\n", split_index, tmp_nodes->node->cl->name, scheduling[tmp_nodes->node->index]== STARPU_CPU_WORKER ?  "CPU": "GPU");
					split_description->scheduling[split_index] = scheduling[tmp_nodes->node->index] == STARPU_CPU_WORKER ? '0' : '1';
					if (scheduling[tmp_nodes->node->index] == STARPU_CUDA_WORKER)
						split_description->nsubtasks_gpu ++;
					tmp_nodes = tmp_nodes->next_node;
				}
			}
			STARPU_ASSERT(!tmp_nodes);
			split_description->scheduling[split_index] = '\0';
		}
		return;
	}
	// Else, we make recursive calls
	struct _starpu_recursive_perfmodel_subgraph *subgraph = _starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(first_node_we_can_split->node->cl->model, first_node_we_can_split->node->footprint);
	if (!subgraph || !subgraph->subgraph_initialisation_is_finished)
	{
		// Node cannot be split, pass it
		split_scheme[index]='0'; //no split possbility
		__get_best_time_rec_version_from_nodes(task_to_sched, nodes_to_schedule, index+1, split_scheme, first_subtask_index, first_index_free, first_node_we_can_split->next_node, split_description);
		return;
	}
	struct _starpu_recursive_perfmodel_graph_node_list *next_node_on_new_graph = NULL;
	struct _starpu_recursive_perfmodel_graph_node_list *split = _starpu_recursive_perfmodel_turn_task_into_subgraph(nodes_to_schedule, first_node_we_can_split->node, &next_node_on_new_graph);
	STARPU_ASSERT(split);
	// We can split it, thus two possibility : either we split it, or not
	split_scheme[index] = '1';
	first_subtask_index[index] = first_index_free;
	__get_best_time_rec_version_from_nodes(task_to_sched, split, index+1, split_scheme, first_subtask_index, first_index_free + subgraph->nb_subtasks, next_node_on_new_graph, split_description);

	split_scheme[index] = '0';
	__get_best_time_rec_version_from_nodes(task_to_sched, nodes_to_schedule, index+1, split_scheme, first_subtask_index, first_index_free, first_node_we_can_split->next_node, split_description);
	_starpu_recursive_perfmodel_free_graph(split);
}

static struct split_description * get_best_time_rec_version_from_nodes(struct starpu_task *task_to_sched)
{
	char *split_scheme;
	unsigned *first_subtask_index;
	struct split_description *best_dec;
	struct _starpu_recursive_perfmodel_graph_node_list lst;
	struct _starpu_recursive_perfmodel_graph_node node =
	{
		task_to_sched->cl, _starpu_job_get_data_size(task_to_sched->cl->model, NULL, 0, _starpu_get_job_associated_to_task(task_to_sched)), 0, starpu_task_data_footprint(task_to_sched), 0, 0, 0, 2, NULL
	};

	_STARPU_CALLOC(split_scheme, 10000, sizeof(char));
	_STARPU_CALLOC(first_subtask_index, 10000, sizeof(unsigned));
	lst.node = &node;
	lst.next_node = NULL;
	starpu_malloc((void**)&best_dec, sizeof(*best_dec));
	memset(best_dec, 0, sizeof(*best_dec));
	best_dec->general_time = -1.;
	__get_best_time_rec_version_from_nodes(task_to_sched, &lst, 0, split_scheme, first_subtask_index, 1, &lst, best_dec);
	return best_dec;
}


// task_to_sched is the subgraph we want to schedule
// nodes_to_schedule corresponds to the current list of nodes we wwould like to decide if it is preferable to split them or not
// index is the index of the corresponding node we look on the split_scheme
// split_scheme[i] is 1 if the ith task is split, 0 else
// first_subtask_index[i] is the index on split_scheme and first_subtask_index of the first subtask of task i, if task i is split
// first_index_free is the index of split_scheme on which we positionate the split scheme of the next task we split
// first_node_we_can_split is the next node to consider to split on nodes_to_schedule

// The coherency between split_scheme and nodes_to_schedule is ensured by the function turn_ask_into_subgraph: all the graph is positioned at the end of nodes_to_schedule
// Thus split_scheme[0] is the main_task, and subtasks will be positionate after it
// split_shceme[1] will be the first subtask, if split its subtasks will be positionate directly after the graph of level 1, and its split_scheme will be after the split_shceme of level 1
// split_scheme[2] will be the second subtask, if split, and if 1 is also split, its subtasks will be positionate after the graph of level 1 and after the subgraph of 1, because first_index_free will be after
static void __get_best_rec_version_from_nodes(struct starpu_task *task_to_sched, struct _starpu_recursive_perfmodel_graph_node_list *nodes_to_schedule, unsigned index, char *split_scheme, unsigned *first_subtask_index, unsigned first_index_free, struct _starpu_recursive_perfmodel_graph_node_list *first_node_we_can_split, double *ncpu_mean_used, double *ncuda_mean_used, double *best_time, double sequential_gpu_time, struct split_description *split_description)
{
	// split description[i][j] represents the best split in term of completion time for a mean usage of i cpus that represents a sequential use of GPU betweed j*10 and (j+1)*10 % of the sequential execution of the task
	if (first_node_we_can_split == NULL)
	{
		// We try the scheduling
		double minimal_cpu = 1., sequential_gpu;
		split_scheme[index] = '\0';
		sequential_gpu = 1.;
		double time_gpu_used = 0.;
		unsigned nsubtasks_cpu = 0;
		enum starpu_worker_archtype *scheduling;
		double tcuda_lv0 = starpu_task_worker_expected_length(task_to_sched, starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), 0, 0);
		double maximal_time = 40*tcuda_lv0;
		while (minimal_cpu > 0. && time_gpu_used < maximal_time)
		{
			time_gpu_used += 10;
			_starpu_recursive_perfmodel_schedule_graph_asap(/*antigraph*/ nodes_to_schedule, best_time, starpu_cpu_worker_get_count(), 1 /* Only one gpu that can be used in complement */, ncpu_mean_used, ncuda_mean_used, 3000., time_gpu_used, /*strcmp(split_scheme, "10101011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000") == */0, &minimal_cpu, &sequential_gpu, &scheduling, maximal_time, 700., &nsubtasks_cpu);
	//		fprintf(stderr, "Task %s %s take time %lf, min_time is %lf\n", task_to_sched->name, split_scheme, *best_time, minimal_cpu);

			//size_t size = _starpu_job_get_data_size(task_to_sched->cl->model, NULL, 0, _starpu_get_job_associated_to_task(task_to_sched));
//			(*best_time) += starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), size);
			int part_exec_gpu = sequential_gpu/sequential_gpu_time*10;
			int cpu_used = (int) *ncpu_mean_used;

//			fprintf(stderr, "For task %s, for scheme %s, by starting to use GPU at %lf, we can finish the execution at %lf, using a mean of %d cores during the execution, that represents a portion executed on GPU of approximatively %d percent. Add a task on CPU takes %lf, with seq gpu = %lf, while sum_gpu = %lf, transfer is %lf\n", task_to_sched->name, split_scheme, time_gpu_used, *best_time, cpu_used, part_exec_gpu, minimal_cpu, tcuda_lv0, sequential_gpu, starpu_transfer_predict(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), starpu_worker_get_by_type(STARPU_CPU_WORKER, 0), size));
			if ((*best_time) < maximal_time && part_exec_gpu < 10 && (split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].general_time < 0. || split_description[SPLIT_INDEX(cpu_used,part_exec_gpu)].general_time > *best_time ))
			{
				strcpy(split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].split_scheme, split_scheme);
				memcpy(split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].first_subtask_index, first_subtask_index, sizeof(unsigned)*10000);
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].mean_cpu_used = cpu_used;
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].min_occupancy_gpu = time_gpu_used;
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].part_exec_gpu = sequential_gpu/sequential_gpu_time*100;
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].general_time = *best_time;
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].nsubtasks_cpu = nsubtasks_cpu;
				unsigned split_index;
				struct _starpu_recursive_perfmodel_graph_node_list *tmp_nodes =  nodes_to_schedule;
				for (split_index=0; split_index < index; split_index++)
				{
					if (split_scheme[split_index] == '1')
					{
						split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].scheduling[split_index] = '0';
					}
					else
					{
						split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].scheduling[split_index] = scheduling[tmp_nodes->node->index] == STARPU_CPU_WORKER ? '0' : '1';
						tmp_nodes = tmp_nodes->next_node;
					}
				}
				STARPU_ASSERT(!tmp_nodes);
				split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].scheduling[split_index] = '\0';
		//		fprintf(stderr, "For task %s, for scheme %s, by starting to use GPU at %lf, we can finish the execution at %lf, using a mean of %u cores during the execution, that represents a portion executed on GPU of approximatively %lf percent, scheduling is %s\n", task_to_sched->name, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].split_scheme, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].min_occupancy_gpu, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].general_time, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].mean_cpu_used, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].part_exec_gpu, split_description[SPLIT_INDEX(cpu_used, part_exec_gpu)].scheduling);
			}
			time_gpu_used += minimal_cpu;
			free(scheduling);
		}

		return; // no other possibility
	}
	// Else, we make recursive calls
	struct _starpu_recursive_perfmodel_subgraph *subgraph = _starpu_recursive_perfmodel_get_subgraph_from_model_and_footprint(first_node_we_can_split->node->cl->model, first_node_we_can_split->node->footprint);
	if (!subgraph || !subgraph->subgraph_initialisation_is_finished)
	{
		// Node cannot be split, pass it
		split_scheme[index]='0'; //no split possbility
		__get_best_rec_version_from_nodes(task_to_sched, nodes_to_schedule, index+1, split_scheme, first_subtask_index, first_index_free, first_node_we_can_split->next_node, ncpu_mean_used, ncuda_mean_used, best_time, sequential_gpu_time, split_description);
		return;
	}

	struct _starpu_recursive_perfmodel_graph_node_list *next_node_on_new_graph = NULL;
	struct _starpu_recursive_perfmodel_graph_node_list *split = _starpu_recursive_perfmodel_turn_task_into_subgraph(nodes_to_schedule, first_node_we_can_split->node, &next_node_on_new_graph);
	STARPU_ASSERT(split);
	// We can split it, thus two possibility : either we split it, or not
	double best_time_split;
	split_scheme[index] = '1';
	first_subtask_index[index] = first_index_free;
	__get_best_rec_version_from_nodes(task_to_sched, split, index+1, split_scheme, first_subtask_index, first_index_free + subgraph->nb_subtasks, next_node_on_new_graph, ncpu_mean_used, ncuda_mean_used, &best_time_split, sequential_gpu_time, split_description);

	split_scheme[index] = '0';
	double best_time_not_split;
	__get_best_rec_version_from_nodes(task_to_sched, nodes_to_schedule, index+1, split_scheme, first_subtask_index, first_index_free, first_node_we_can_split->next_node, ncpu_mean_used, ncuda_mean_used, &best_time_not_split, sequential_gpu_time, split_description);
	_starpu_recursive_perfmodel_free_graph(split);

	if (best_time_split > best_time_not_split)
	{
		*best_time = best_time_not_split;
	}
	else
	{
		*best_time = best_time_split;
	}
}

static void get_best_rec_version_from_nodes(struct starpu_task *task_to_sched, double *ncpu_mean_used, double *ncuda_mean_used, double *best_time, double sequential_gpu_time, struct split_description *split_description)
{
	char *split_scheme;
	unsigned *first_subtask_index;
	struct _starpu_recursive_perfmodel_graph_node_list lst;
	struct _starpu_recursive_perfmodel_graph_node node =
	{
		task_to_sched->cl, _starpu_job_get_data_size(task_to_sched->cl->model, NULL, 0, _starpu_get_job_associated_to_task(task_to_sched)), 0, starpu_task_data_footprint(task_to_sched), 0, 0, 0, 2, NULL
	};

	_STARPU_CALLOC(split_scheme, 10000, sizeof(char));
	_STARPU_CALLOC(first_subtask_index, 10000, sizeof(unsigned));
	lst.node = &node;
	lst.next_node = NULL;
	__get_best_rec_version_from_nodes(task_to_sched, &lst, 0, split_scheme, first_subtask_index, 1, &lst, ncpu_mean_used, ncuda_mean_used,  best_time, sequential_gpu_time, split_description);
}

void _starpu_recursive_perfmodel_get_best_schedule_alap(struct starpu_task *task, struct _starpu_recursive_perfmodel_subgraph *graph, double *best_time, double *ncuda_mean_used, double *ncpu_mean_used)
{
	unsigned nb_cpus = starpu_cpu_worker_get_count(), nb_cuda = starpu_cuda_worker_get_count();
	struct starpu_perfmodel_arch *arch_gpu = nb_cuda > 0 ? starpu_worker_get_perf_archtype(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), STARPU_NMAX_SCHED_CTXS) : NULL;

	if (graph == NULL || !graph->subgraph_initialisation_is_finished)
	{
		return;
	}
	if (graph->splittings == NULL)
	{ // We wll generate all splittings

		if (nb_cuda == 0 || !starpu_worker_can_execute_task(starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0), task, 0))
		{
			graph->splittings = get_best_time_rec_version_from_nodes(task);
			//fprintf(stderr, "For task %s, scheme %s that startto use GPU at %lf, and finish the execution at %lf, using a mean of %u cores during the execution, that represents a portion executed on GPU of approximatively %lf percents, scheduled as %s has %u subtasks on CPU ; time %lf on GPU\n", task->name, graph->splittings->split_scheme, graph->splittings->min_occupancy_gpu, graph->splittings->general_time, graph->splittings->mean_cpu_used, graph->splittings->part_exec_gpu, graph->splittings->scheduling, graph->splittings->nsubtasks_cpu, graph->splittings->sequential_gpu_time);
			return;
		}

		starpu_malloc((void**) &graph->splittings, 10*(nb_cpus+1)*sizeof(*graph->splittings));
		for (unsigned i=0; i < nb_cpus+1; i++)
		{
			for (unsigned j=0; j < 10; j++)
			{
				memset(&graph->splittings[SPLIT_INDEX(i, j)], 0, sizeof(graph->splittings[SPLIT_INDEX(i, j)]));
				graph->splittings[SPLIT_INDEX(i,j)].general_time = -1.;
			}

		}
		double t_cuda_lv0 = starpu_perfmodel_history_based_expected_perf(graph->cl->model, arch_gpu, graph->footprint);

		// Invert graph
		get_best_rec_version_from_nodes(task, ncpu_mean_used, ncuda_mean_used, best_time, t_cuda_lv0, graph->splittings);
		*best_time = -1.;
		for (unsigned i = 0; i < nb_cpus; i++)
		{
			for (unsigned j=0; j <= 9; j++)
			{
				for (unsigned ii = 0; ii <= i; ii++)
				{
					for (unsigned jj=0; jj < j && graph->splittings[SPLIT_INDEX(i, j)].general_time > 0.; jj++)
					{
						if (graph->splittings[SPLIT_INDEX(ii,jj)].general_time > 0. && graph->splittings[SPLIT_INDEX(ii,jj)].general_time < graph->splittings[SPLIT_INDEX(i,j)].general_time &&
								jj < j) // deuxieme partie pour dire que la part d'occupation est plus grande pour i,j que pour ii,jj et donc pas d'utilité de garder i,j. Cette partie inutile car assurer par le for
						{
	//						fprintf(stderr, "For task %s, scheme %s that startto use GPU at %lf, and finish the execution at %lf, using a mean of %u cores during the execution, that represents a portion executed on GPU of approximatively %lf percents is suppressed by scheme %s, finish at %lf, using %u cores and a portition of %lf percents\n", graph->cl->model->symbol, graph->splittings[SPLIT_INDEX(i, j)].split_scheme, graph->splittings[SPLIT_INDEX(i, j)].min_occupancy_gpu, graph->splittings[SPLIT_INDEX(i, j)].general_time, graph->splittings[SPLIT_INDEX(i, j)].mean_cpu_used, graph->splittings[SPLIT_INDEX(i, j)].part_exec_gpu, graph->splittings[SPLIT_INDEX(ii, jj)].split_scheme, graph->splittings[SPLIT_INDEX(ii, jj)].general_time, graph->splittings[SPLIT_INDEX(ii, jj)].mean_cpu_used, graph->splittings[SPLIT_INDEX(ii, jj)].part_exec_gpu);
							graph->splittings[SPLIT_INDEX(i,j)].general_time = -1.;
						}
					}
				}

				if (graph->splittings[SPLIT_INDEX(i, j)].general_time > 0. && (*best_time < 0. || graph->splittings[SPLIT_INDEX(i,j)].general_time < *best_time))
				{
					graph->one_split_exist = 1;
					*best_time = graph->splittings[SPLIT_INDEX(i,j)].general_time;
					//fprintf(stderr, "For task %s, for scheme %s, by starting to use GPU at %lf, we can finish the execution at %lf, using a mean of %u cores during the execution, that represents a portion executed on GPU of approximatively %lf \%\n", graph->cl->model->symbol, graph->splittings[i][j].split_scheme, graph->splittings[i][j].min_occupancy_gpu, graph->splittings[i][j].general_time, graph->splittings[i][j].mean_cpu_used, graph->splittings[i][j].part_exec_gpu);
				}
			}
		}
		if (*best_time < 0. || *best_time > t_cuda_lv0 )
			*best_time = t_cuda_lv0;
		//fprintf(stderr, "tcuda lv0 for %s(%lu) is %lf\n", graph->cl->model->symbol, graph->data_size, t_cuda_lv0);
		graph->best_split = get_best_time_rec_version_from_nodes(task);
//		fprintf(stderr, "For task %s, scheme %s that finish at %lf is the best, with scheduling %s, compared to time normal %lf\n", graph->cl->model->symbol, graph->best_split->split_scheme, graph->best_split->general_time, graph->best_split->scheduling, t_cuda_lv0);
		/*// An ALAP schedule is an ASAP schedule for inverted graph
		_starpu_recursive_perfmodel_invert_graph(all_nodes, &antigraph_lv1);
		_starpu_recursive_perfmodel_schedule_graph_asaa(antigraph_lv1, best_time, nb_cpus, nb_cuda, ncpu_mean_used, ncuda_mean_used, 0., time_gpu_used);

		fprintf(stderr, "trec lv1 for %s(%lu) is %lf. Mean cuda : %lf, mean cpu : %lf, nb_subtasks = %u\n", graph->cl->model->symbol, graph->data_size, *best_time, *ncuda_mean_used, *ncpu_mean_used, graph->nb_subtasks);

		struct _starpu_recursive_perfmodel_graph_node_list *node_to_split = all_nodes->node;
		struct _starpu_recursive_perfmodel_graph_node_list *split = _starpu_recursive_perfmodel_turn_task_into_subgraph(all_nodes, all_nodes->node);
		while()
		{
			fprintf(stderr, "Node split : %s(%p)\n", all_nodes->node->cl->model->symbol, all_nodes->node);
			struct _starpu_recursive_perfmodel_graph_node_list *antigraph_split = NULL;
			_starpu_recursive_perfmodel_invert_graph(split, &antigraph_split);
			*/
		//_starpu_recursive_perfmodel_schedule_graph_asap(antigraph_split, best_time, nb_cpus, 1 /* Only one cpu that can be used in complement */, ncpu_mean_used, ncuda_mean_used, 0., 3*t_cuda_lv0 /* Used in complement of CPU*/);
			//fprintf(stderr, "trec split for %s(%lu) is %lf. Mean cuda : %lf, mean cpu : %lf\n", graph->cl->model->symbol, graph->data_size, *best_time, *ncuda_mean_used, *ncpu_mean_used);
			//_starpu_recursive_perfmodel_free_graph(antigraph_split);
//		}
//		_starpu_recursive_perfmodel_free_graph(antigraph_lv1);
		char c[128];
		sprintf(c, "%s_%u.recursive", graph->cl->model->symbol, graph->footprint);
		FILE *f = fopen(c, "w");
		fprintf(f, "split scheme - occ gpu - end - ncpu - part_gpu - scheduling\n");
		for (unsigned i = 0; i < nb_cpus; i++)
		{
			for (unsigned j=0; j < 9; j++)
			{
				if (graph->splittings[SPLIT_INDEX(i,j)].general_time > 0.)
				{
						fprintf(f, "%s -  %lf - %lf - %u - %lf - %s\n\n", graph->splittings[SPLIT_INDEX(i, j)].split_scheme, graph->splittings[SPLIT_INDEX(i, j)].min_occupancy_gpu, graph->splittings[SPLIT_INDEX(i, j)].general_time, graph->splittings[SPLIT_INDEX(i, j)].mean_cpu_used, graph->splittings[SPLIT_INDEX(i, j)].part_exec_gpu, graph->splittings[SPLIT_INDEX(i, j)].scheduling);
					}
			}
		}
		fclose(f);
	}
}

static int _starpu_recursive_perfmodel_read_subgraphs_codelet_from_disk(struct starpu_codelet *codelet)
{
	if (codelet->model == NULL)
		return 1;
	struct starpu_perfmodel *model = codelet->model;
	char filename[256];
	snprintf(filename, 255, "graph_%s.gen", model->symbol);
	FILE *f = fopen(filename, "r");
	if (f != NULL)
	{
		// The file exists
		while(!feof(f))
		{
			uint32_t footprint;
			size_t data_size;
			fscanf(f, "%u\t%lu\n", &footprint, &data_size);

			struct _starpu_recursive_perfmodel_subgraph *subgraph;
			starpu_malloc((void**)&subgraph, sizeof(*subgraph));
			memset((void*)subgraph, 0, sizeof(*subgraph));
			if(_starpu_recursive_perfmodel_read_one_subgraph_from_disk(codelet, subgraph, footprint, data_size) == 0)
			{
				fclose(f);
				return 0;
			}
		}
		fclose(f);
	}
	return 1;
}

void starpu_recursive_perfmodel_read_all_subgraphs_from_disk()
{
	struct _starpu_codelet_list *lst = registered_codelet;
	while(lst)
	{
		if (!lst->model_is_loaded)
			lst->model_is_loaded = _starpu_recursive_perfmodel_read_subgraphs_codelet_from_disk(lst->cl);
		lst = lst->next;
	}
}

void _starpu_recursive_perfmodel_record_codelet(struct starpu_codelet *cl)
{
	STARPU_PTHREAD_MUTEX_LOCK(&subgraph_lock);
	if (cl->model == NULL || cl->model->is_recorded_recursive)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&subgraph_lock);
		return;
	}
	cl->model->is_recorded_recursive = 1;

	struct _starpu_codelet_list *lst;
	starpu_malloc((void**)&lst, sizeof(*lst));
	memset(lst, 0, sizeof(*lst));
	lst->cl = cl;
	lst->next = registered_codelet;
	registered_codelet = lst;
	starpu_recursive_perfmodel_read_all_subgraphs_from_disk();
	STARPU_PTHREAD_MUTEX_UNLOCK(&subgraph_lock);
}

#endif
