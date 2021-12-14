/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2020       Maxime Gonthier
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

/* Building a maximum spanning tree from the set of tasks using Prim's algorithm.
 * The task processing order is then the order in which tasks are added to the tree.
 * hMETIS can be used before executing MST on each package with the environemment variable HMETIS=1.
 */

//~ #define PRINT

#include <schedulers/HFP.h> /* Headers containing struct and function nedded */

//~ int hmetis;

static int mst_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct HFP_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	component->can_pull(component);
	return 0;
}

struct starpu_task_list mst(struct starpu_task_list task_list, int number_task, starpu_ssize_t GPU_RAM_M)
{
	struct starpu_task_list SIGMA;
	starpu_task_list_init(&SIGMA);
	int i = 0; 
	int count = 0;
	int j = 0;
	int v = 0;
	int i_bis = 0;
	int j_bis = 0;
	int tab_runner = 0;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;
	int matrice_adjacence[number_task][number_task]; 
	for (i = 0; i < number_task; i++) 
	{ 
		for (j = 0; j < number_task; j++) 
		{ 
			matrice_adjacence[i][j] = 0; 
		} 
	} 
	temp_task_1  = starpu_task_list_begin(&task_list);
	temp_task_2  = starpu_task_list_begin(&task_list);
	temp_task_2  = starpu_task_list_next(temp_task_2);
	/* Building the adjacency matrix */
	for (i = 0; i < number_task; i++) 
	{
		for (j = i + 1; j < number_task; j++) 
		{
			for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++) 
			{
				for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++) 
				{
					if (STARPU_TASK_GET_HANDLE(temp_task_1, i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2, j_bis)) 
					{ 
						matrice_adjacence[i][j]++; 
					}
				}
			}
			temp_task_2  = starpu_task_list_next(temp_task_2);
		}
		temp_task_1  = starpu_task_list_next(temp_task_1);
		temp_task_2 = temp_task_1;
		if (i + 1 != number_task) 
		{ 
			temp_task_2  = starpu_task_list_next(temp_task_2); 
		}
	}	
				
	/* Printing the adjacency matrix */
	#ifdef PRINT
		printf("Matrice d'adjacence :\n"); 
		for (i = 0; i < number_task; i++) 
		{ 
			for (j = 0; j < number_task; j++) 
			{ 
				printf("%d ",matrice_adjacence[i][j]); 
			} 
			printf("\n"); 
		}
	#endif
	
	/* Struct of packages to have one task by package and thus being able to number each task.
	 * We need to number them to recognize them later on. */
	struct my_list *temp_sub_list = malloc(sizeof(*temp_sub_list));
	struct paquets *temp_paquets = malloc(sizeof(*temp_paquets));
	starpu_task_list_init(&temp_sub_list->sub_list);
	temp_sub_list->next = NULL;
	temp_paquets->temp_pointer_1 = temp_sub_list;
	temp_paquets->first_link = temp_paquets->temp_pointer_1;		
	int do_not_add_more = 0;
	while (!starpu_task_list_empty(&task_list)) 
	{	
		starpu_task_list_push_back(&temp_paquets->temp_pointer_1->sub_list, starpu_task_list_pop_front(&task_list));
		temp_paquets->temp_pointer_1->index_package = do_not_add_more;
		if (do_not_add_more != number_task - 1) 
		{ 
			HFP_insertion(temp_paquets); 
		}
		do_not_add_more++;
	}
	temp_paquets->first_link = temp_paquets->temp_pointer_1;
			
	/* Start of the MST algorithm */		
	// Key values used to pick minimum weight edge in cut
	int key[number_task];
	// To represent set of vertices included in MST
	bool mstSet[number_task];
	int tab_SIGMA[number_task];
	// Initialize all keys as 0
	for (i = 0; i < number_task; i++) 
	{ 
		key[i] = 0, mstSet[i] = false; 
	}

	// Always include first 1st vertex in MST.
	// Make key 0 so that this vertex is picked as first vertex.
	key[0] = 1;
	for (count = 0; count < number_task - 1; count++) 
	{
		// Pick the minimum key vertex from the
		// set of vertices not yet included in MST
		int max = -1, max_index = 0;
		for (v = 0; v < number_task; v++)
			if (mstSet[v] == false && key[v] > max)
				max = key[v], max_index = v;
		
		int u = max_index;
		// Add the picked vertex to the MST Set	
		mstSet[u] = true;
		tab_SIGMA[tab_runner] = u;
		tab_runner++;
		
		// Update key value and parent index of
		// the adjacent vertices of the picked vertex.
		// Consider only those vertices which are not
		// yet included in MST
		for (v = 0; v < number_task; v++)
			// matrice_adjacence[u][v] is non zero only for adjacent vertices of m
			// mstSet[v] is false for vertices not yet included in MST
			// Update the key only if graph[u][v] is greater than key[v]
			if (matrice_adjacence[u][v] && mstSet[v] == false && matrice_adjacence[u][v] > key[v])
				key[v] = matrice_adjacence[u][v];
	}
	/* End of the MST algorithm */
	
	/* Put last vertex in SIGMA */	
	for (i = 0; i < number_task; i++) 
	{
		if (mstSet[i] == false) 
		{
			tab_SIGMA[number_task - 1] = i;
		}
	}	
			
	#ifdef PRINT
		printf("tab_SIGMA[i] : "); 
		for (i = 0; i < number_task; i++) 
		{ 
			printf("%d ",tab_SIGMA[i]); 
		} 
		printf("\n"); 
	#endif
	
	i = 0;
	
	/* Filling our task list */
	temp_paquets->temp_pointer_1 = temp_paquets->first_link;
	while (i != number_task) 
	{
		if (tab_SIGMA[i] == temp_paquets->temp_pointer_1->index_package) 
		{
			starpu_task_list_push_back(&SIGMA, starpu_task_list_pop_front(&temp_paquets->temp_pointer_1->sub_list));
			i++;
			temp_paquets->temp_pointer_1 = temp_paquets->first_link;
		}
		else 
		{
			temp_paquets->temp_pointer_1 = temp_paquets->temp_pointer_1->next;
		}
	}
				
	//Belady
	//~ if (starpu_get_env_number_default("BELADY",0) == 1) 
	//~ {
		//~ get_ordre_utilisation_donnee_mst(data, NB_TOTAL_DONNEES);
	//~ }
				
	return SIGMA;
}

static struct starpu_task *mst_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct HFP_sched_data *data = component->data;
	int i = 0;
	struct starpu_task *task = NULL; 
	
	if (do_schedule_done == true)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	
	/* If one or more task have been refused */
	data->p->temp_pointer_1 = data->p->first_link;
	if (data->p->temp_pointer_1->next != NULL) 
	{ 
		for (i = 0; i < Ngpu; i++) 
		{
			if (to == component->children[i]) 
			{
				break;
			}
			else 
			{
				data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
			}
		}
	}
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list)) 
	{
		task = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		#ifdef PRINT 
			printf("Task %p is getting out of pull_task from fifo refused list on gpu %p\n",task, to); 
		#endif
		return task;
	}	
	/* If the linked list is empty, we can pull more tasks */
	if (is_empty(data->p->first_link) == true) 
	{
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			return NULL;
	}
	else 
	{
		task = get_task_to_return(component, to, data->p, Ngpu);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		#ifdef PRINT 
			printf("Task %p is getting out of pull_task from gpu %p\n", task, to); 
		#endif
		return task;
	}
}
return NULL;
}

static int mst_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	int i = 0;
	struct HFP_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		data->p->temp_pointer_1 = data->p->first_link;
		int nb_gpu = get_number_GPU();
		if (data->p->temp_pointer_1->next == NULL) 
		{ 
			starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task); 
		}
		else 
		{
			for (i = 0; i < nb_gpu; i++) 
			{
				if (to == component->children[i]) 
				{
					break;
				}
				else
				{
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				}
			}
			starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		/* Can I uncomment this part ? */
		//~ {
			//~ if (didwork)
				//~ fprintf(stderr, "pushed some tasks to %p\n", to);
			//~ else
				//~ fprintf(stderr, "I didn't have anything for %p\n", to);
		//~ }
	}

	/* There is room now */
	return didwork || starpu_sched_component_can_push(component, to);
}

static int mst_can_pull(struct starpu_sched_component * component)
{
	return starpu_sched_component_can_pull(component);
}

static void mst_do_schedule(struct starpu_sched_component *component)
{
	int i = 0;
	struct starpu_task_list temp_task_list;
	starpu_task_list_init(&temp_task_list);
	int NB_TOTAL_DONNEES = 0;
	struct HFP_sched_data *data = component->data;
	struct starpu_task *task = NULL;
	NT = 0;
 	int number_of_package_to_build = get_number_GPU(); 
	GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));
	
	/* If the linked list is empty, we can pull more tasks */
	if (is_empty(data->p->first_link) == true) 
	{
		if (!starpu_task_list_empty(&data->sched_list)) 
		{
			appli = starpu_task_get_name(starpu_task_list_begin(&data->sched_list));
			if (hmetis != 0) 
			{
				hmetis_scheduling(data->p, &data->sched_list, number_of_package_to_build, GPU_RAM_M);
				
				/* Apply mst on each package */
				data->p->temp_pointer_1 = data->p->first_link;
				for (i = 0; i < number_of_package_to_build; i++) 
				{
					data->p->temp_pointer_1->sub_list = mst(data->p->temp_pointer_1->sub_list, data->p->temp_pointer_1->nb_task_in_sub_list, GPU_RAM_M);
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				}
				do_schedule_done = true;
				return;
			}
			
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list)) 
			{				
				task = starpu_task_list_pop_front(&data->sched_list);
				NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(task);
				NT++;
				#ifdef PRINT
					printf("%p\n",task); 
				#endif
				starpu_task_list_push_back(&temp_task_list, task);
			} 		
			#ifdef PRINT
				printf("%d task(s) have been pulled\n", NT); 
			#endif
			//~ task = starpu_task_list_begin(&data->popped_task_list);
			//~ printf("tache %p\n", task);
			/* Apply mst on the task list */
			//~ data->p->temp_pointer_1->sub_list = mst(data->popped_task_list, NT, GPU_RAM_M);			
			data->p->temp_pointer_1->sub_list = mst(temp_task_list, NT, GPU_RAM_M);			
		
		do_schedule_done = true;
		}
	}
}

struct starpu_sched_component *starpu_sched_component_mst_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	hmetis = starpu_get_env_number_default("HMETIS", 0);
	
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "mst");
	
	struct HFP_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	struct paquets *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));
	
	index_current_popped_task = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_all_gpu = 0;
	index_current_popped_task_all_gpu_prefetch = 0;
	
	do_schedule_done = false;
	Ngpu = get_number_GPU();
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	//~ starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);
 
	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
	data->p = paquets_data;
	data->p->temp_pointer_1->nb_task_in_sub_list = 0;
	data->p->temp_pointer_1->expected_time_pulled_out = 0;
	
	component->data = data;
	component->do_schedule = mst_do_schedule;
	component->push_task = mst_push_task;
	component->pull_task = mst_pull_task;
	component->can_push = mst_can_push;
	component->can_pull = mst_can_pull;

	return component;
}

static void initialize_mst_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_mst_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_mst_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

void get_current_tasks_mst(struct starpu_task *task, unsigned sci)
{
	task_currently_treated = task;
	starpu_sched_component_worker_pre_exec_hook(task,sci);
}

struct starpu_sched_policy _starpu_sched_mst_policy =
{
	.init_sched = initialize_mst_center_policy,
	.deinit_sched = deinitialize_mst_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = get_current_tasks_mst,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "mst",
	.policy_description = "Maximum Spanning Tree",
	.worker_type = STARPU_WORKER_LIST,
};
