/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

/* Ordonnanceur de base sous contrainte mémoire */


#include <starpu.h>
#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
//#include <starpu_task_list.h>
#include "core/task.h"
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include "starpu_stdlib.h"
#define NTASKS 5

#define MEMORY_AFFINITY

struct basic_sched_data *variable_globale;

struct child_data {
	double expected_start;
	double predicted;
	double predicted_transfer;
	double expected_end;
	unsigned child;
};


static int compar(const void *_a, const void *_b)
{
	const struct child_data *a = _a;
	const struct child_data *b = _b;
	if (a->expected_end < b->expected_end)
		return -1;
	if (a->expected_end == b->expected_end)
		return 0;
	return 1;
}

struct basic_sched_data
{
	struct starpu_task_list tache_pop;
	struct starpu_task_list list_if_fifo_full;
	
	//Ma liste
	struct my_list *head;
	struct my_list *head_2;
	struct my_list *head_3;
	struct my_list *first_link;
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

struct my_list
{
	int package_nb_data;
	starpu_data_handle_t * package_data;
	struct starpu_task_list sub_list;
	struct my_list *next;
	struct my_list *previous;
};

void insertion(struct basic_sched_data *a)
{
    /* Création du nouvel élément */
    struct my_list *nouveau = malloc(sizeof(*nouveau));
  
	starpu_task_list_init(&nouveau->sub_list);
    /* Insertion de l'élément au début de la liste */
    nouveau->next = a->head;
    a->head = nouveau;
}

void insertion_fin(struct basic_sched_data *a)
{
    /* Création du nouvel élément */
    struct my_list *nouveau = malloc(sizeof(*nouveau));
  
	starpu_task_list_init(&nouveau->sub_list);
    /* Insertion de l'élément à la fin de la liste */
    nouveau->next = NULL;
   while (a->head->next != NULL) { a->head = a->head->next; }
   a->head->next = nouveau;
   //~ a->head->next->package_data = malloc(3*sizeof(a->head->package_data[0]));
   printf("ok\n");
}


// Efface le premier élément de la liste tels que numeroCompte==val
// Retourne le nouveau pointeur sur le début de la liste
struct basic_sched_data* delete_link(struct basic_sched_data* a)
{
	while (a->first_link != NULL && a->first_link->package_nb_data == 0) {
		a->head = a->first_link;
		a->first_link = a->first_link->next;
		free(a->head);
	}
	if (a->first_link != NULL) {
		a->head_2 = a->first_link;
		a->head_3 = a->first_link->next;
		while (a->head_3 != NULL) {
			while (a->head_3 != NULL && a->head_3->package_nb_data == 0) {
				a->head = a->head_3;
				a->head_3 = a->head_3->next;
				a->head_2->next = a->head_3;
				free(a->head);
			}
			if (a->head_3 != NULL) {
				a->head_2 = a->head_3;
				a->head_3 = a->head_3->next;
			}
		}
	}
	return a->first_link;
}
	
int intComparator ( const void * first, const void * second ) {
  return ( *(int*)first - *(int*)second );
}
	
		
static int basic_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct basic_sched_data *data = component->data;
		fprintf(stderr, "Pushing task %p\n", task);

        STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	starpu_task_list_push_front(&data->sched_list, task);

	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	/* Tell below that they can now pull */
	component->can_pull(component);
	//~ printf("Push OK!\n");
	return 0;
}

static struct starpu_task *basic_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct basic_sched_data *data = component->data;	
	int i = 0; int j = 0; int nb_pop = 0; int temp_nb_pop = 0; int tab_runner = 0; int max_donnees_commune = 0; int k = 0; int nb_data_commun = 0; int nb_tasks_in_linked_list = 0;
	int je_suis_ou = 0;
	int do_not_add_more = 0;	
	int link_index = 0;
	int nb_duplicate_data = 0;
	int max_value_common_data_matrix = 0;
	int index_temp_task_1 = 0; int index_temp_task_2 = 0;
	int index_head_1 = 0; int index_head_2 = 0;
	int i_bis = 0; int j_bis = 0;
	starpu_ssize_t GPU_RAM = 0;
	STARPU_ASSERT(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component));
	GPU_RAM = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))))/2;
	
//Doesn't work for me	
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_task_list_empty(&data->sched_list))
		return NULL;
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_2 = NULL;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_3 = NULL;
	struct starpu_task *temp_task_4 = NULL;
	
	//If the list is not empty, it means that we have task to get out of pull before pulling more tasks
	//If we use a linked list we need to go to the next one and verify it's not equal to NULL
	//Else we can pull tasks

//Verif au cas où une tache est passé dans le oops et a été refusé
if (!starpu_task_list_empty(&data->list_if_fifo_full)) {
		task1 = starpu_task_list_pop_back(&data->list_if_fifo_full); 
		//~ printf("La tâche %p a été refusé, je la fais sortir de nouveau du pull_task\n",task1);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return task1;
	}

	//Si le next est null et que la liste est vide, alors on peut pull à nouveau des tâches
	if ((data->head->next == NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		if (!starpu_task_list_empty(&data->sched_list)) {
			while (!starpu_task_list_empty(&data->sched_list)) {
				//Pulling all tasks and counting them
				task1 = starpu_task_list_pop_front(&data->sched_list);
				nb_pop++;
				starpu_task_list_push_back(&data->tache_pop,task1);
			} 
		
			printf("%d task(s) have been pulled\n",nb_pop);
			
			//Version avec des paquets de tâches et des comparaisons de paquets ------------------------------------------------------------------------------------------------------
			temp_task_3  = starpu_task_list_begin(&data->tache_pop);
			data->head->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_3)*sizeof(data->head->package_data[0]));
			
			//Here I put each data of each task in a package (a linked list). One task == one link
			do_not_add_more = nb_pop - 1;
			for (temp_task_3  = starpu_task_list_begin(&data->tache_pop); temp_task_3 != starpu_task_list_end(&data->tache_pop); temp_task_3  = temp_task_4) {
				temp_task_4 = starpu_task_list_next(temp_task_3);
				temp_task_3 = starpu_task_list_pop_front(&data->tache_pop);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(temp_task_3); i++) {
					data->head->package_data[i] = STARPU_TASK_GET_HANDLE(temp_task_3,i);
				}
				data->head->package_nb_data = STARPU_TASK_GET_NBUFFERS(temp_task_3);
				starpu_task_list_push_back(&data->head->sub_list,temp_task_3);
				
				if(do_not_add_more != 0) { insertion(data); data->head->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_3)*sizeof(data->head->package_data[0])); }
				do_not_add_more--;
			}
			
			data->first_link = data->head;
			
			//Code to print all the data of all the packages ---------------------------------------------
			//~ while (data->head != NULL) {
				//~ for (i = 0; i < 3; i++) {
					//~ printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->head->package_data[i],temp_task_3  = starpu_task_list_begin(&data->head->sub_list),link_index);
				//~ }
				//~ link_index++;
				//~ data->head = data->head->next;
			//~ } printf("NULL\n");
			//~ data->head = data->first_link;
			//--------------------------------------------------------------------------------------------
			
			int matrice_donnees_commune[nb_pop][nb_pop]; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}

			data->head_2 = data->first_link;
			index_head_2++;
			
			//Filling the common data matrix
			for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
				for (data->head_2 = data->head->next; data->head_2 != NULL; data->head_2 = data->head_2->next) {
					for (i = 0; i < data->head_2->package_nb_data; i++) {
						for (j = 0; j < data->head_2->package_nb_data; j++) {
							//~ printf("Je compare la donnée %p du paquet %d et la donnée %p du paquet %d\n",data->head->package_data[i],index_head_1,data->head_2->package_data[j],index_head_2);
							if (((data->head->package_data[i] == data->head_2->package_data[j])) &&  ( (starpu_data_get_size(data->head->package_data[i]) + starpu_data_get_size(data->head_2->package_data[j])) < GPU_RAM )) {
								matrice_donnees_commune[index_head_1][index_head_2] += (starpu_data_get_size(data->head->package_data[i]) + starpu_data_get_size(data->head_2->package_data[j]));
								//~ printf("Données communes entre %p et %p\n",data->head->package_data[i],data->head_2->package_data[j]);
							} 
						}
					} index_head_2++;
				} index_head_1++; index_head_2 = index_head_1 + 1;
			}
			
			//Here is code to print the common data matrix  ----------------
			//~ printf("Common data matrix : \n");
			//~ for (i = 0; i < nb_pop; i++) {
				//~ for (j = 0; j < nb_pop; j++) {
					//~ // //~ printf (" %zd ",matrice_donnees_commune[i][j]/100000);
					//~ if (matrice_donnees_commune[i][j] != 0) {printf(" 1 ");} else {printf(" 0 ");}			
				//~ }
				//~ printf("\n");
			//~ }
			//--------------------------------------------------------------
			
			//Getting the number of package that have data in commons
			for (i = 0; i < nb_pop; i++) {
				for (j = 0; j < nb_pop; j++) {
					if (matrice_donnees_commune[i][j] != 0) { nb_data_commun++; }
				}
			}
			//~ printf("Nb data en commun : %d\n",nb_data_commun);
			
			//Getting back to the beginning of the linked list
			data->head = data->first_link;
			data->head_2 = data->first_link;

			//Getting W_max
			for (i_bis =0; i_bis < nb_pop; i_bis++) { for (j_bis = 0; j_bis < nb_pop; j_bis++) { 
				if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } } }
				//~ printf("Le max de poids de data communes est %zd\n",max_value_common_data_matrix);
				
			//MArche que pour 1 tache par 12 toache pour le moment
			for (i = 0; i < nb_pop; i++) {
				data->head_2 = data->head;
				data->head_2 = data->head_2->next;
				for (j = i + 1; j< nb_pop; j++) {
					
					if ((matrice_donnees_commune[i][j] == max_value_common_data_matrix) && (max_value_common_data_matrix != 0) && (max_value_common_data_matrix < GPU_RAM)) {
						//Interdit i et j de faire des regroupements par la suite
						for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; }
						for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; }
						nb_data_commun--;
						//Si il a pas déjà été bougé
						//Marche que si il y a une seule tâche
						if (!starpu_task_list_empty(&data->head_2->sub_list)) { 
							//Push la tâche
							starpu_task_list_push_back(&data->head->sub_list,starpu_task_list_pop_front(&data->head_2->sub_list)); 
							//Merge les données
							//A FAIRE SUPPR LES DOUBLONS DE DONN2ES
							//Met dans un tableau a part
							starpu_data_handle_t *temp_data_tab_1 = malloc((data->head->package_nb_data + data->head_2->package_nb_data) * sizeof(data->head->package_data[0]));
							//Tri des tab de data pour pouvoir supprimer les doublons en O(n)
							//~ qsort(data->head->package_data,data->head->package_nb_data + data->head_2->package_nb_data,sizeof(data->head->package_data[0]),intComparator);
							//~ qsort(data->head_2->package_data,data->head->package_nb_data + data->head_2->package_nb_data,sizeof(data->head_2->package_data[0]),intComparator);
							for (i_bis = 0; i_bis < data->head->package_nb_data; i_bis++) {
								temp_data_tab_1[i_bis] = data->head->package_data[i_bis];
							}
							for (i_bis = 0; i_bis < data->head_2->package_nb_data; i_bis++) {
								temp_data_tab_1[i_bis + data->head->package_nb_data] = data->head_2->package_data[i_bis];
							}
							qsort(temp_data_tab_1,data->head->package_nb_data + data->head_2->package_nb_data,sizeof(temp_data_tab_1[0]),intComparator);
							
							//~ //Code to print the tab of data --------
							//~ printf("Tab trié avant -------------------\n");
							//~ for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data); i_bis++) {
								//~ printf("%d\n",temp_data_tab_1[i_bis]);
							//~ }
							//~ printf("----------------------------\n");
							//~ //-------------------------------------
							
							//On vire les doublons
							for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data); i_bis++) {
								if (temp_data_tab_1[i_bis] == temp_data_tab_1[i_bis + 1]) {
									temp_data_tab_1[i_bis] = 0;
									nb_duplicate_data++;
								}
							}
								
							//Puis on met tout dans head
							data->head->package_data = malloc((data->head->package_nb_data + data->head_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
							j_bis = 0;
							for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data); i_bis++)
							{
								if (temp_data_tab_1[i_bis] != 0) { data->head->package_data[j_bis] = temp_data_tab_1[i_bis]; j_bis++; }
							}
							
							//~ //Code to print the tab of data --------
							//~ printf("Tab trié après -------------------\n");
							//~ for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data - nb_duplicate_data); i_bis++) {
								//~ printf("%d\n",data->head->package_data[i_bis]);
							//~ }
							//~ printf("----------------------------\n");
							//-------------------------------------
							nb_duplicate_data = 0;
							
							//Met le nb de data de head_2 à 0
							data->head->package_nb_data += data->head_2->package_nb_data;
							data->head_2->package_nb_data = 0;
							
						}	
					}
					data->head_2 = data->head_2->next;
				} 
				//~ if (nb_tasks_in_linked_list == nb_pop) { break; }
				if (nb_data_commun > 1) { //a verif ce if hein
					data->head = data->head->next;
				} 
			}
			
			//Supprimer les maillons vide
			data->head = data->first_link;
			data->head = delete_link(data);
			
			//Code to print everything ------------------------------------------------------------------
			//~ while (data->head != NULL) {
				//~ for (i = 0; i < data->head->package_nb_data; i++) {
					//~ printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->head->package_data[i],temp_task_3  = starpu_task_list_begin(&data->head->sub_list),link_index);
				//~ }
				//~ link_index++;
				//~ data->head = data->head->next;
			//~ } printf("NULL\n");
			//~ printf("Il y a %d paquets\n",link_index);
			//--------------------------------------------------------------------------------------------
			data->head = data->first_link;
			
			task1 = starpu_task_list_pop_front(&data->head->sub_list);
	}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	} //Else de if ((data->head->next == NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
	if (!starpu_task_list_empty(&data->head->sub_list)) {
		task1 = starpu_task_list_pop_front(&data->head->sub_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	if ((data->head->next != NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		//The list is empty and it's not the last one, so we go on the next link
		data->head = data->head->next;
		while (starpu_task_list_empty(&data->head->sub_list)) { data->head = data->head->next; }
			task1 = starpu_task_list_pop_front(&data->head->sub_list); 
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			printf("Task %p is getting out of pull_task from starpu_task_list_empty(&data->head->sub_list)\n",task1);
			return task1;
	}
	if ((data->head->next == NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		printf("On est pas censé entrer dans ce if\n");
	}			
}

static int basic_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct basic_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//~ fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task);
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->list_if_fifo_full, task);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		//A décommenté une fois le code fini
		//~ {
			//~ if (didwork)
				//~ fprintf(stderr, "pushed some tasks to %p\n", to);
			//~ else
				//~ fprintf(stderr, "I didn't have anything for %p\n", to);
		//~ }
	}

	/* There is room now */
	//~ printf("Can push OK!\n");
	return didwork || starpu_sched_component_can_push(component, to);
}

static int basic_can_pull(struct starpu_sched_component * component)
{
	struct basic_sched_data *data = component->data;

	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_basic_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "basic");
	
	struct basic_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	variable_globale = data;
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->tache_pop);
	starpu_task_list_init(&my_data->sub_list);
 
	my_data->next = NULL;
	data->head = my_data;
	
	component->data = data;
	component->push_task = basic_push_task;
	component->pull_task = basic_pull_task;
	component->can_push = basic_can_push;
	component->can_pull = basic_can_pull;

	//~ printf("Create OK!\n");
	return component;
}

static void initialize_basic_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_basic_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
	//~ printf("Initialize OK!\n");

	//~ variable_globale = *data;
}

static void deinitialize_basic_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_basic_sched_policy =
{
	.init_sched = initialize_basic_center_policy,
	.deinit_sched = deinitialize_basic_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "basic-sched",
	.policy_description = "sched de base pour tester",
	.worker_type = STARPU_WORKER_LIST,
};

