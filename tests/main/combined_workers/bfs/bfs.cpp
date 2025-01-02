/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <starpu.h>
#include "common.h"
#include "timer.h"

#define NB_ITERATION 10


extern void omp_bfs_func(void *buffers[], void *_args);

void Usage(int argc, char**argv)
{
	fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
}

void read_file(char *input_f, unsigned int *nb_nodes, unsigned int *nb_edges,
	       Node **origin_graph_nodes, bool **origin_graph_mask,
	       bool **origin_updating_graph_mask, bool **origin_graph_visited,
	       int **origin_graph_edges, int **origin_cost)
{
	FILE *fp;
	int source = 0;

	printf("Reading File\n");

	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		exit(1);
	}

	fscanf(fp, "%u", nb_nodes);

	// allocate host memory
	*origin_graph_nodes = (Node *) malloc(sizeof(Node) * (*nb_nodes));
	*origin_graph_mask = (bool *) malloc(sizeof(bool) * (*nb_nodes));
	*origin_updating_graph_mask = (bool *) malloc(sizeof(bool) * (*nb_nodes));
	*origin_graph_visited = (bool *) malloc(sizeof(bool) * (*nb_nodes));

	int start, edgeno;
	// initialize the memory
	for( unsigned int i = 0; i < *nb_nodes; i++)
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		(*origin_graph_nodes)[i].starting = start;
		(*origin_graph_nodes)[i].no_of_edges = edgeno;
		(*origin_graph_mask)[i]=false;
		(*origin_updating_graph_mask)[i]=false;
		(*origin_graph_visited)[i]=false;
	}

	//read the source node from the file
	fscanf(fp, "%d", &source);
	source=0;

	//set the source node as true in the mask
	(*origin_graph_mask)[source]=true;
	(*origin_graph_visited)[source]=true;

	fscanf(fp, "%u", nb_edges);

	int id, cost;
	*origin_graph_edges = (int*) malloc(sizeof(int) * (*nb_edges));
	for(unsigned int i=0; i < *nb_edges ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		(*origin_graph_edges)[i] = id;
	}


	// allocate mem for the result on host side
	*origin_cost = (int*) malloc( sizeof(int)* (*nb_nodes));
	for(unsigned int i = 0; i < (*nb_nodes); i++)
		(*origin_cost)[i]=-1;
	(*origin_cost)[source]=0;

	fclose(fp);
}

//extern void omp_bfs_func(Node* h_graph_nodes, int* h_graph_edges, bool *h_graph_mask, bool *h_updating_graph_mask, bool *h_graph_visited, int* h_cost, int nb_nodes, int nb_edges);
//extern void cuda_bfs_func(Node* h_graph_nodes, int* h_graph_edges, bool *h_graph_mask, bool *h_updating_graph_mask, bool *h_graph_visited, int* h_cost, int nb_nodes, int nb_edges);
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	int ret;
	char *input_f;
	Timer timer;

	unsigned int nb_nodes = 0, nb_edges = 0;

	Node *origin_graph_nodes, *graph_nodes;
	bool *origin_graph_mask, *graph_mask;
	bool *origin_updating_graph_mask, *updating_graph_mask;
	bool *origin_graph_visited, *graph_visited;
	int *origin_graph_edges, *graph_edges;
	int *origin_cost, *cost;

	static struct starpu_perfmodel bfs_model;
	static struct starpu_codelet bfs_cl;

	bfs_model.type = STARPU_HISTORY_BASED;
	bfs_model.symbol = "omp_bfs";

	bfs_cl.modes[0] = STARPU_R;
	bfs_cl.modes[1] = STARPU_R;
	bfs_cl.modes[2] = STARPU_RW;
	bfs_cl.modes[3] = STARPU_RW;
	bfs_cl.modes[4] = STARPU_RW;
	bfs_cl.modes[5] = STARPU_RW;
	bfs_cl.where = STARPU_CPU;
	bfs_cl.type = STARPU_FORKJOIN;
	bfs_cl.max_parallelism = INT_MAX;
	bfs_cl.cpu_funcs[0] = omp_bfs_func;
	bfs_cl.nbuffers = 6;
	bfs_cl.model = &bfs_model;

	starpu_data_handle_t graph_nodes_handle;
	starpu_data_handle_t graph_edges_handle;
	starpu_data_handle_t graph_mask_handle;
	starpu_data_handle_t updating_graph_mask_handle;
	starpu_data_handle_t graph_visited_handle;
	starpu_data_handle_t cost_handle;

	if(argc != 2)
	{
		Usage(argc, argv);
		exit(1);
	}

	input_f = argv[1];
	read_file(input_f, &nb_nodes, &nb_edges, &origin_graph_nodes,
		  &origin_graph_mask, &origin_updating_graph_mask,
		  &origin_graph_visited, &origin_graph_edges, &origin_cost);

	graph_nodes = (Node *) calloc(nb_nodes, sizeof(Node));
	graph_mask = (bool *) calloc(nb_nodes, sizeof(bool));
	updating_graph_mask = (bool *) calloc(nb_nodes, sizeof(bool));
	graph_visited = (bool *) calloc(nb_nodes, sizeof(bool));
	graph_edges = (int *) calloc(nb_edges, sizeof(int));
	cost = (int *) calloc(nb_nodes, sizeof(int));

	memcpy(graph_nodes, origin_graph_nodes, nb_nodes*sizeof(Node));
	memcpy(graph_edges, origin_graph_edges, nb_edges*sizeof(int));

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_vector_data_register(&graph_nodes_handle, STARPU_MAIN_RAM,
				    (uintptr_t) graph_nodes, nb_nodes,
				    sizeof(graph_nodes[0] ));
	starpu_vector_data_register(&graph_edges_handle, STARPU_MAIN_RAM,
				    (uintptr_t)graph_edges, nb_edges,
				    sizeof(graph_edges[0]));
	starpu_vector_data_register(&graph_mask_handle, STARPU_MAIN_RAM,
				    (uintptr_t)graph_mask, nb_nodes,
				    sizeof(graph_mask[0] ));
	starpu_vector_data_register(&updating_graph_mask_handle, STARPU_MAIN_RAM,
				    (uintptr_t)updating_graph_mask,
				    nb_nodes,
				    sizeof(updating_graph_mask[0]));
	starpu_vector_data_register(&graph_visited_handle, STARPU_MAIN_RAM,
				    (uintptr_t)graph_visited, nb_nodes,
				    sizeof(graph_visited[0]));
	starpu_vector_data_register(&cost_handle, STARPU_MAIN_RAM, (uintptr_t)cost,
				    nb_nodes, sizeof(cost[0]));

	for(int it=0; it < NB_ITERATION; it++)
	{
		starpu_data_acquire(graph_mask_handle, STARPU_W);
		starpu_data_acquire(updating_graph_mask_handle, STARPU_W);
		starpu_data_acquire(graph_visited_handle, STARPU_W);
		starpu_data_acquire(cost_handle, STARPU_W);

		memcpy(graph_mask, origin_graph_mask, nb_nodes * sizeof(bool));
		memcpy(updating_graph_mask, origin_updating_graph_mask, nb_nodes * sizeof(bool));
		memcpy(graph_visited, origin_graph_visited, nb_nodes * sizeof(bool));
		memcpy(cost, origin_cost, nb_nodes * sizeof(int));

		starpu_data_release(graph_mask_handle);
		starpu_data_release(updating_graph_mask_handle);
		starpu_data_release(graph_visited_handle);
		starpu_data_release(cost_handle);

		struct starpu_task *task = starpu_task_create();

		task->cl = &bfs_cl;

		task->handles[0] = graph_nodes_handle;
		task->handles[1] = graph_edges_handle;
		task->handles[2] = graph_mask_handle;
		task->handles[3] = updating_graph_mask_handle;
		task->handles[4] = graph_visited_handle;
		task->handles[5] = cost_handle;

		task->synchronous = 1;

		printf("Start traversing the tree\n");

		timer.start();

		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

		timer.stop();
	}

	starpu_data_unregister(graph_nodes_handle);
	starpu_data_unregister(graph_edges_handle);
	starpu_data_unregister(graph_mask_handle);
	starpu_data_unregister(updating_graph_mask_handle);
	starpu_data_unregister(graph_visited_handle);
	starpu_data_unregister(cost_handle);

	starpu_shutdown();

	printf("File: %s, Avergae Time: %f, Total time: %f\n", input_f,
	       timer.getAverageTime(), timer.getTotalTime());

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(unsigned int i=0;i<nb_nodes;i++)
		fprintf(fpo,"%u) cost:%d\n", i, cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free(graph_nodes);
	free(graph_edges);
	free(graph_mask);
	free(updating_graph_mask);
	free(graph_visited);
	free(cost);
	free(origin_graph_nodes);
	free(origin_graph_edges);
	free(origin_graph_mask);
	free(origin_updating_graph_mask);
	free(origin_graph_visited);
	free(origin_cost);

}
