#include "../common.h"
#include <starpu.h>
#include <omp.h>

#include <stdio.h>

void omp_bfs_func(void *buffers[], void *_args)
{
	Node* graph_nodes = (Node *) STARPU_VECTOR_GET_PTR(buffers[0]);
	int no_of_nodes = STARPU_VECTOR_GET_NX(buffers[0]);
	int* graph_edges = (int *) STARPU_VECTOR_GET_PTR(buffers[1]);
	bool *graph_mask = (bool *) STARPU_VECTOR_GET_PTR(buffers[2]);
	bool *updating_graph_mask = (bool *) STARPU_VECTOR_GET_PTR(buffers[3]);
	bool *graph_visited = (bool *) STARPU_VECTOR_GET_PTR(buffers[4]);
	int* cost = (int *) STARPU_VECTOR_GET_PTR(buffers[5]);
	int k=0;
    
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

#ifdef OPEN
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
#endif 
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (graph_mask[tid] == true)
			{ 
				graph_mask[tid]=false;
				for(int i=graph_nodes[tid].starting; i<(graph_nodes[tid].no_of_edges + graph_nodes[tid].starting); i++)
				{
					int id = graph_edges[i];
					if(!graph_visited[id])
						{
						cost[id]=cost[tid]+1;
						updating_graph_mask[id]=true;
						}
				}
			}
		}

  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (updating_graph_mask[tid] == true){
			graph_mask[tid]=true;
			graph_visited[tid]=true;
			stop=true;
			updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);
	
	printf("Kernel Executed %d times, threads: %d\n",k, starpu_combined_worker_get_size());
	//printf("graph_edges = %d, %d, %d\n",graph_edges[0], graph_edges[1], graph_edges[2]);
	//printf("graph_mask = %d, %d, %d\n",graph_mask[0], graph_mask[1], graph_mask[2]);
	//printf("updating_graph_mask = %d, %d, %d\n",updating_graph_mask[0], updating_graph_mask[1], updating_graph_mask[2]);
	//printf("graph_visited = %d, %d, %d\n",graph_visited[0], graph_visited[1], graph_visited[2]);
	//printf("Cost = %d, %d, %d\n",cost[0], cost[1], cost[2]);
}
