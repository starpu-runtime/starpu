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

/* Maximum Spanning tree
 */

#include <starpu_data_maxime.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <starpu.h>
#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include "core/task.h"
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include "starpu_stdlib.h"
#include "common/list.h"
#define PRINTF /* O or 1 */
#define BELADY /* O or 1 */

int NT;

/* Structure used to acces the struct my_list. There are also task's list */
struct mst_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	struct starpu_task_list SIGMA; /* order in which task will go out */
	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

struct my_list
{
	int index;
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct my_list *next;	
};

/* Put a link at the beginning of the linked list */
void insertion_mst(struct mst_sched_data *a)
{
    struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
    new->next = a->temp_pointer_1;
    a->temp_pointer_1 = new;
}

/* Pushing the tasks */		
static int mst_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct mst_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* Donne l'ordre d'utilisation des données ainsi que la liste de l'ensemble des différentes données */
static void get_ordre_utilisation_donnee_mst(struct mst_sched_data *a, int NB_TOTAL_DONNEES)
{
	FILE *f = fopen("Output_maxime/ordre_utilisation_donnees.txt","w");
	FILE *f_2 = fopen("Output_maxime/ordre_traitement_taches.txt","w");
	struct starpu_task *task = NULL; 
	int i = 0; int j = 0; int k = 0;
	
	total_nb_data = NB_TOTAL_DONNEES; 
	task = starpu_task_list_begin(&a->SIGMA);
	data_use_order = malloc(NB_TOTAL_DONNEES*sizeof(STARPU_TASK_GET_HANDLE(task,0)));
	task_position_in_data_use_order = malloc(NT*sizeof(int));
	
	for (task = starpu_task_list_begin(&a->SIGMA); task != starpu_task_list_end(&a->SIGMA); task = starpu_task_list_next(task)) {
		fprintf(f_2,"%p\n",task);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			data_use_order[k] = STARPU_TASK_GET_HANDLE(task,i);
			k++;
			fprintf(f,"%p\n",STARPU_TASK_GET_HANDLE(task,i));
		}
		if (j != 0) { task_position_in_data_use_order[j] = STARPU_TASK_GET_NBUFFERS(task) + task_position_in_data_use_order[j - 1]; }
		else { task_position_in_data_use_order[j] = STARPU_TASK_GET_NBUFFERS(task); }
		j++;
	}
	index_task_currently_treated = 0;
	fclose(f);
	fclose(f_2);
}

/* The function that sort the tasks in packages */
static struct starpu_task *mst_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	int i, j, i_bis, j_bis, count, tab_runner = 0;
	int NB_TOTAL_DONNEES = 0;
	struct mst_sched_data *data = component->data;

	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;
	//~ struct starpu_task *task2 = NULL;
	
	NT = 0;
 		
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->list_if_fifo_full)) {
		task1 = starpu_task_list_pop_back(&data->list_if_fifo_full); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		//~ printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	/* If the linked list is empty, we can pull more tasks */
	//OLD
	//~ if (starpu_task_list_empty(&data->popped_task_list)) {
	//SIGMA
	if (starpu_task_list_empty(&data->SIGMA)) {
		if (!starpu_task_list_empty(&data->sched_list)) {
			time_t start, end; time(&start); 
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list)) {				
				task1 = starpu_task_list_pop_front(&data->sched_list);
				NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(task1);
				NT++;
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("%p\n",task1); }
				//~ printf("%p\n",task1);
				starpu_task_list_push_back(&data->popped_task_list,task1);
				//~ data->id = NT;
			} 		
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("%d task(s) have been pulled\n",NT); }
			
			int matrice_adjacence[NT][NT]; for (i = 0; i < NT; i++) { for (j = 0; j < NT; j++) { matrice_adjacence[i][j] = 0; } } 
			temp_task_1  = starpu_task_list_begin(&data->popped_task_list);
			temp_task_2  = starpu_task_list_begin(&data->popped_task_list);
			temp_task_2  = starpu_task_list_next(temp_task_2);
			for (i = 0; i < NT; i++) {
				for (j = i + 1; j < NT; j++) {
					for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++) {
						for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++) {
							if (STARPU_TASK_GET_HANDLE(temp_task_1,i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2,j_bis)) { matrice_adjacence[i][j]++; }
						}
					}
					temp_task_2  = starpu_task_list_next(temp_task_2);
				}
				temp_task_1  = starpu_task_list_next(temp_task_1);
				temp_task_2 = temp_task_1;
				if (i + 1 != NT) { temp_task_2  = starpu_task_list_next(temp_task_2); }
			}				
			/* Affichage de la matrice d'adjacence */
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Matrice d'adjacence :\n"); for (i = 0; i < NT; i++) { for (j = 0; j < NT; j++) { printf("%d ",matrice_adjacence[i][j]); } printf("\n"); } }
			
			//NEW
			int do_not_add_more = 0;
			while (!starpu_task_list_empty(&data->popped_task_list)) {	
				starpu_task_list_push_back(&data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&data->popped_task_list));
				data->temp_pointer_1->index = do_not_add_more;
				if (do_not_add_more != NT-1) { insertion_mst(data); }
				do_not_add_more++;
			}
			data->first_link = data->temp_pointer_1;
			
				
				// Array to store constructed MST
				int parent[NT];
				// Key values used to pick minimum weight edge in cut
				int key[NT];
				// To represent set of vertices included in MST
				bool mstSet[NT];
				int tab_SIGMA[NT];
				//~ const char* tab_SIGMA[NT];

				// Initialize all keys as 0
				for (int i = 0; i < NT; i++) { 
					key[i] = 0, mstSet[i] = false; }

				// Always include first 1st vertex in MST.
				// Make key 0 so that this vertex is picked as first vertex.
				key[0] = 1;
				parent[0] = INT_MAX; // First node is always root of MST
				
				for (count = 0; count < NT - 1; count++) {
					// Pick the minimum key vertex from the
					// set of vertices not yet included in MST
					int max = -1, max_index;

					for (int v = 0; v < NT; v++)
						if (mstSet[v] == false && key[v] > max)
							max = key[v], max_index = v;
										
					int u = max_index;

					// Add the picked vertex to the MST Set	
					mstSet[u] = true;
					//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("u = %d\n",u); }
					//~ temp_task_1  = starpu_task_list_begin(&data->popped_task_list); for (i = 0; i < u; i++) { temp_task_1  = starpu_task_list_next(temp_task_1); }
					if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Add %p to sigma\n",temp_task_1); }
					//~ return temp_task_1;
					//~ starpu_task_list_push_back(&data->SIGMA,temp_task_1);
					//~ tab_SIGMA[tab_runner] = starpu_task_get_name(temp_task_1);
					//~ printf("dans tab_sigma %p\n",tab_SIGMA[tab_runner]);
					tab_SIGMA[tab_runner] = u;
					tab_runner++;

					// Update key value and parent index of
					// the adjacent vertices of the picked vertex.
					// Consider only those vertices which are not
					// yet included in MST
					for (int v = 0; v < NT; v++)
						// matrice_adjacence[u][v] is non zero only for adjacent vertices of m
						// mstSet[v] is false for vertices not yet included in MST
						// Update the key only if graph[u][v] is greater than key[v]
						if (matrice_adjacence[u][v] && mstSet[v] == false && matrice_adjacence[u][v] > key[v])
							parent[v] = u, key[v] = matrice_adjacence[u][v];
				}
					
				/* On met le dernier sommet dans sigma */
				for (i = 0; i < NT; i++) {
					if (mstSet[i] == false) {
						//~ temp_task_1  = starpu_task_list_begin(&data->popped_task_list); for (i_bis = 0; i_bis < i; i_bis++) { temp_task_1  = starpu_task_list_next(temp_task_1); }
						if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Dernier sommet: add %p to sigma\n",temp_task_1); }
						//~ starpu_task_list_push_back(&data->SIGMA,temp_task_1);
						//~ tab_SIGMA[NT - 1] = starpu_task_get_name(temp_task_1);
						tab_SIGMA[NT - 1] = i;
						//~ mstSet[i] = NT-1;
					}
				}
				//~ printf("mstSet:	"); for (i = 0; i < NT; i++) { printf("%d ",i); }
				
				//~ printf("tab_SIGMA[i] : "); for (i = 0; i < NT; i++) { printf("%d ",tab_SIGMA[i]); } printf("\n");
				i = 0;
				data->temp_pointer_1 = data->first_link;
				while (i != NT) {
					//~ temp_task_1  = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list);
					if (tab_SIGMA[i] == data->temp_pointer_1->index) {
					//~ if (strcmp(char_SIGMA[i],starpu_task_get_name(temp_task_1) == 0)) {
						starpu_task_list_push_back(&data->SIGMA,starpu_task_list_pop_front(&data->temp_pointer_1->sub_list));
						i++;
						data->temp_pointer_1 = data->first_link;
						//~ printf("ok3\n");
					}
					else { 
						//~ starpu_task_list_push_back(&data->popped_task_list,temp_task_1);
						data->temp_pointer_1 = data->temp_pointer_1->next;
					}
				}
				
				//Belady
				if (starpu_get_env_number_default("BELADY",0) == 1) {
					get_ordre_utilisation_donnee_mst(data, NB_TOTAL_DONNEES);
				}
				
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("\nFin de MST\n"); }


			
			time(&end); int time_taken = end - start; if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Temps d'exec : %d secondes\n",time_taken); }
			FILE *f_time = fopen("Output_maxime/Execution_time_raw.txt","a");
			fprintf(f_time,"%d\n",time_taken);
			fclose(f_time);
			
			//OLD
			//~ task1 = starpu_task_list_pop_front(&data->popped_task_list);
			//Avec SIGMA
			task1 = starpu_task_list_pop_front(&data->SIGMA);
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			//~ printf("Task %p is getting out of pull_task\n",task1);
			return task1;
		}
		else {
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			//~ printf("Task %p is getting out of pull_task\n",task1);
			return task1; 
		}
	}
	else { 
		//OLD
		//~ task1 = starpu_task_list_pop_front(&data->popped_task_list);
		//SIGMA
		task1 = starpu_task_list_pop_front(&data->SIGMA);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		//~ printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	//~ return task1;
}

static int mst_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct mst_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, task %p got refused\n", task); }
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->list_if_fifo_full, task);
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
	struct mst_sched_data *data = component->data;
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_mst_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "mst");
	
	struct mst_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&data->SIGMA);
	starpu_task_list_init(&my_data->sub_list);
 
	my_data->next = NULL;
	data->temp_pointer_1 = my_data;
	
	component->data = data;
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
	index_task_currently_treated++;	
	starpu_sched_component_worker_pre_exec_hook(task,sci);
}

//~ /* Almost Belady while tasks are being executed */
//~ starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch)
//~ {
	//~ int donnee_utilise_dans_le_plus_longtemps = 0; int distance_donnee_utilise_dans_le_plus_longtemps = 0;
	//~ int k = 0; int nb_data_next_task = 0; int i = 0; int j = 0;
	//~ unsigned nb_data_on_node = 0; /* Number of data loaded on memory. Needed to init the tab containing data on node */
	//~ int is_allocated;
	//~ if (task_currently_treated != NULL) {
		
		//~ //New memory read
		//~ starpu_data_handle_t *data_on_node;
		//~ starpu_data_get_node_data(node, &data_on_node, &nb_data_on_node);
		
		//~ //Because I started at 1 and not 0
		//~ int used_index_task_currently_treated = index_task_currently_treated - 1;
		
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("La tâche en cours est %p, index numéro %d, position %d dans le tableau d'ordre des données\n",task_currently_treated, used_index_task_currently_treated, task_position_in_data_use_order[used_index_task_currently_treated]); }
		
		//~ if (task_position_in_data_use_order[index_task_currently_treated] != total_nb_data) {
			//~ nb_data_next_task = task_position_in_data_use_order[used_index_task_currently_treated] - task_position_in_data_use_order[used_index_task_currently_treated - 1];

			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("nb data next :%d\n",nb_data_next_task);
			//~ printf("Données de la tâche en cours : ");
			//~ for (i = 0; i < nb_data_next_task; i++) {
				//~ printf("%p ",data_use_order[task_position_in_data_use_order[used_index_task_currently_treated] - i - 1]); } printf ("\n"); 
			//~ }
			
			//~ for (i = 0; i < nb_data_next_task; i++) {	
				//~ /* On regarde si la donnée est pas déjà sur M par hasard */
				//~ starpu_data_query_status(data_use_order[task_position_in_data_use_order[used_index_task_currently_treated] - i - 1], node, &is_allocated, NULL, NULL);
				//~ if (is_allocated && i == 1000) {
					//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("La donnée %p est déjà sur M\n",data_use_order[task_position_in_data_use_order[used_index_task_currently_treated] - i - 1]); }
				//~ }
				//~ else {
						//~ int *prochaine_utilisation_donnee;
						//~ prochaine_utilisation_donnee = malloc(nb_data_on_node*sizeof(int));
						
						//~ for (j = 0; j < nb_data_on_node; j++) { prochaine_utilisation_donnee[j] = INT_MAX; }
						//~ //Care if a task is never use again and is on node, we must evict it
						//~ for (j = 0; j < nb_data_on_node; j++) { 
							//~ if (starpu_data_can_evict(data_on_node[j], node, is_prefetch)) {
										//~ for (k = task_position_in_data_use_order[used_index_task_currently_treated]; k < total_nb_data; k++) {
											//~ if (data_on_node[j] == data_use_order[k]) {
												//~ prochaine_utilisation_donnee[j] = k;
												//~ break;
											//~ }
										//~ }
							//~ }
							//~ else { prochaine_utilisation_donnee[j] = -1; }
						//~ }
						
					
					//~ distance_donnee_utilise_dans_le_plus_longtemps = -1;
					//~ for (j = 0; j < nb_data_on_node; j++) {
						//~ if (prochaine_utilisation_donnee[j] > distance_donnee_utilise_dans_le_plus_longtemps) {
								//~ donnee_utilise_dans_le_plus_longtemps = j;
								//~ distance_donnee_utilise_dans_le_plus_longtemps = prochaine_utilisation_donnee[j]; 
						//~ }
					//~ }
					//~ if (distance_donnee_utilise_dans_le_plus_longtemps == -1) {
						//~ free(data_on_node); 
						//~ free(prochaine_utilisation_donnee);
						//~ return STARPU_DATA_NO_VICTIM;  
					//~ }
					
					//~ starpu_data_handle_t returned_handle = data_on_node[donnee_utilise_dans_le_plus_longtemps];
					//~ free(data_on_node);
					//~ free(prochaine_utilisation_donnee);
					//~ return returned_handle;
													
				//~ }
			//~ }
	//~ }
	//~ else {
		 //~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On est sur la dernière tâche il faudrait sortir la\n"); } 
		//~ free(data_on_node);
		//~ return NULL;
		 //~ }
	//~ } 
	//~ else { 
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("task current = null\n"); } 
		//~ }
	//~ return STARPU_DATA_NO_VICTIM;
//~ }

struct starpu_sched_policy _starpu_sched_mst_policy =
{
	.init_sched = initialize_mst_center_policy,
	.deinit_sched = deinitialize_mst_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = get_current_tasks_mst,
	//~ .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "mst",
	.policy_description = "Maximum Spanning Tree",
	.worker_type = STARPU_WORKER_LIST,
};

//~ .init_sched = initialize_HFP_center_policy,
	//~ .deinit_sched = deinitialize_HFP_center_policy,
	//~ .add_workers = starpu_sched_tree_add_workers,
	//~ .remove_workers = starpu_sched_tree_remove_workers,
	//~ .push_task = starpu_sched_tree_push_task,
	//~ .pop_task = starpu_sched_tree_pop_task,
	//~ .pre_exec_hook = get_current_tasks,
	//~ .post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	//~ .pop_every_task = NULL,
	//~ .policy_name = "HFP",
	//~ .policy_description = "Affinity aware task ordering",
	//~ .worker_type = STARPU_WORKER_LIST,
