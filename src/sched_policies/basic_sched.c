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
#define NTASKS 5

#define MEMORY_AFFINITY

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

//~ struct basic_sched_data *next;
//~ struct starpu_task_list sub_list *next;

//Dummy
struct basic_sched_data
{
	struct starpu_task_list tache_pop;
	
	//Ma liste
	struct my_list *head;
	struct my_list *first_link;
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

struct my_list
{
	struct starpu_task_list sub_list;
	struct my_list *next;
};

void insertion(struct basic_sched_data *a)
//~ void insertion(struct basic_sched_data *a, struct starpu_task_list b)
{
    /* Création du nouvel élément */
    struct my_list *nouveau = malloc(sizeof(*nouveau));
  
    //~ nouveau->sub_list = b;
	starpu_task_list_init(&nouveau->sub_list);
    /* Insertion de l'élément au début de la liste */
    nouveau->next = a->head;
    a->head = nouveau;
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
	//~ printf("Pull breakpoint 1\n");
	struct basic_sched_data *data = component->data;	
	int i = 0; int j = 0; int nb_pop = 0; int temp_nb_pop = 0; int tab_runner = 0; int max_donnees_commune = 0; int k = 0; int nb_data_commun = 0; int nb_tasks_in_linked_list = 0;
	
	int taille_tache = 0;
	
//Doesn't work for me	
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_task_list_empty(&data->sched_list))
		return NULL;
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task1 = NULL;
	
	//If the list is not empty, it means that we have task to get out of pull before pulling more tasks
	//If we use a linked list we need to go to the next one and verify it's not equal to NULL
	//Else we can pull tasks
	
	//Version 1 liste---------------------------------------------------------------------------------
	//~ if (starpu_task_list_empty(&data->tache_pop)) {
	//------------------------------------------------------------------------------------------------

	//Version liste chainée --------------------------------------------------------------------------
	//Si le next est null et que la liste est vide, alors on peut pull à nouveau des tâches
	if ((data->head->next == NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		//Je dois passer au premier elem de la liste ici A FAIRE JSP ?
	//------------------------------------------------------------------------------------------------
		//~ printf("Pull breakpoint 2\n");
		if (!starpu_task_list_empty(&data->sched_list)) {
			while (!starpu_task_list_empty(&data->sched_list)) {
				//Pulling all tasks and counting them
				task1 = starpu_task_list_pop_back(&data->sched_list);
				//Getting the "size" of a task for later, it's bad here cause it's in a while loop
				taille_tache = STARPU_TASK_GET_NBUFFERS(task1);
				//~ printf("Pulling task %p\n",task1);
				nb_pop++;
				//True line
				starpu_task_list_push_back(&data->tache_pop,task1);
				
				//Test
				//~ starpu_task_list_push_back(&data_liste->sub_list,task1);
				//~ &data_liste->next;
			} printf("%d task(s) have been pulled\n",nb_pop);
			//~ if (nb_pop > 2) {
			if (nb_pop > 0) {
				//Filling a tab with every handles of every tasks
				struct starpu_task *task_tab [nb_pop];
				int *handles[nb_pop*taille_tache]; for (i = 0; i < nb_pop*taille_tache; i++) { handles[i] = 0; }
				for (i = 0; i < nb_pop; i++) {
					task1 = starpu_task_list_pop_front(&data->tache_pop);
					handles[tab_runner] = STARPU_TASK_GET_HANDLE(task1, 0);
					handles[tab_runner + 1] = STARPU_TASK_GET_HANDLE(task1, 1);
					handles[tab_runner + 2] = STARPU_TASK_GET_HANDLE(task1, 2);
					tab_runner += taille_tache;
					starpu_task_list_push_back(&data->tache_pop,task1);
				}
				
				//Here is code to print the handles tab ------------------------
				//~ printf("Handles tab : \n");
				//~ for (i = 0; i < nb_pop*taille_tache; i++) {
					//~ printf("%p\n",handles[i]);
				//~ }
				//--------------------------------------------------------------
				
				tab_runner = 0;
				int *matrice_donnees_commune[nb_pop][nb_pop]; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
				for (i = 0; i < nb_pop - 1; i++) {
					//Compare handles of every task and put the result in a matrix
					for (tab_runner = i+1; tab_runner < nb_pop; tab_runner++) {
						if (handles[i*taille_tache] == handles[tab_runner*taille_tache]) { matrice_donnees_commune[i][tab_runner] += 1; }
						if (handles[i*taille_tache + 1] == handles[tab_runner*taille_tache + 1]) { matrice_donnees_commune[i][tab_runner] += 1; }
						if (handles[i*taille_tache + 2] == handles[tab_runner*taille_tache + 2]) { matrice_donnees_commune[i][tab_runner] += 1; }
					}				
				}
				
				//Here is code to print the common data matrix  ----------------
				printf("Common data matrix : \n");
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						printf (" %d ",matrice_donnees_commune[i][j]);
					}
					printf("\n");
				}
				//--------------------------------------------------------------
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						if (matrice_donnees_commune[i][j] == 4) { nb_data_commun++; }
					}
				}
				printf("Nb data en commun : %d\n",nb_data_commun);
				
				//A FAIRE : Pour l'instant vu que les comparaisons ne trouvait que 1 point commun entre les taches, je fonctionne comme ca. 
				//A FAIRE : Plus tard il y aura plus que 1 point commun, il faudra donc comparer et voir qui a le plus de points commun
				
				//Version avec une seule liste et un tableau -----------------------------------------------------------------------------
				//Using a temp tab to reorder tasks
				//Here we put every task in the tab
				//~ for (i = 0; i < nb_pop; i++) {
					//~ task_tab[i] = starpu_task_list_pop_front(&data->tache_pop);
				//~ }
				//~ //Here, if a tab has 0 in it, it means that a linked task got put in the tab so we have to put this one too next to it
				//~ for (i = 0; i < nb_pop; i++) {
					//~ if (task_tab[i] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[i]); task_tab[i] = 0; }
					//~ for (j = i + 1; j< nb_pop; j++) {
						//~ if (matrice_donnees_commune[i][j] == 4) {
							//~ printf ("Data in common\n");
							//~ if (task_tab[j] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[j]); task_tab[j] = 0; }
						//~ }
					//~ }
				//~ }
				//------------------------------------------------------------------------------------------------------------------------
				
				//Version avec la liste chainée ------------------------------------------------------------------------------------------
				//Here we put every task in the tab
				for (i = 0; i < nb_pop; i++) {
					task_tab[i] = starpu_task_list_pop_front(&data->tache_pop);
				}
				//Here, if a tab has 0 in it, it means that a linked task got put in the tab so we have to put this one too next to it
	
				for (i = 0; i < nb_pop; i++) {
					if (task_tab[i] != 0) { starpu_task_list_push_back(&data->head->sub_list,task_tab[i]); nb_tasks_in_linked_list++; task_tab[i] = 0; }
					for (j = i + 1; j< nb_pop; j++) {
						if (matrice_donnees_commune[i][j] == 4) {
							//~ printf ("Data in common\n");
							nb_data_commun--;
							if (task_tab[j] != 0) { starpu_task_list_push_back(&data->head->sub_list,task_tab[j]); nb_tasks_in_linked_list++; task_tab[j] = 0; }
							if (nb_tasks_in_linked_list == nb_pop) { break; }
						}
					} if (nb_tasks_in_linked_list == nb_pop) { break; }
					if (nb_data_commun > 1) {
					insertion(data);
					printf("Pull breakpoint 3\n"); 
					} 
				}
				printf("Il y a %d tâches qui ont été mises dans la liste chainée\n",nb_tasks_in_linked_list);

				//Pas utile normalement, ca sert juste a voir si toutes les taches ont bien été mise dans la liste, qu'il reste rien dans me tableau
				//~ for (i = 0; i < nb_pop; i++) {
					//~ if (task_tab[i] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[i]); task_tab[i] = 0; }
				//~ }
				
				//Version 1 liste ----------------------------------------
				//~ task1 = starpu_task_list_pop_front(&data->tache_pop);
				
				//Version liste chainée ----------------------------------
				printf("Pull breakpoint 4\n");
				task1 = starpu_task_list_pop_front(&data->head->sub_list);
				printf("Pull breakpoint 5\n");
			}
			//Else here means that we have only 2 task or less, so no need to compare the handles A FAIRE
			//A voir si il faut le garder ou pas dans le cas des liste chainées
			else {
				//~ printf("Pull breakpoint 5\n");
				task1 = starpu_task_list_pop_front(&data->tache_pop);
			}
		}
		//~ printf("Pull breakpoint 6\n");
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ printf("Pull breakpoint 7\n");
		printf("Task %p is getting out of pull_task\n",task1);
		//~ printf("Pull OK!\n");
		return task1;
	}
	
	//Version 1 liste -----------------------------------------------------------------------------------------------------------------------
	//~ task1 = starpu_task_list_pop_front(&data->tache_pop);
	//~ STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	//~ return task1;
	//---------------------------------------------------------------------------------------------------------------------------------------
	
	//Version liste chainée -----------------------------------------------------------------------------------------------------------------
	if ((data->head->next != NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		//la liste est vide on passe au prochain maillon, qui n'est pas NULL car on a check plus haut
		data->head = data->head->next;
		task1 = starpu_task_list_pop_front(&data->head->sub_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ printf("Pull breakpoint 5\n");
		printf("Task %p is getting out of pull_task\n",task1);
		//~ printf("Pull OK!\n");
		return task1;
	}
	else {
		task1 = starpu_task_list_pop_front(&data->head->sub_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ printf("Pull breakpoint 6\n");
		printf("Task %p is getting out of pull_task\n",task1);
		//~ printf("Pull OK!\n");
		return task1;
	}
	//---------------------------------------------------------------------------------------------------------------------------------------
		
		
}

static int basic_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct basic_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//if (data->verbose)
			fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task);
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->sched_list, task);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		//if (data->verbose)
		{
			if (didwork)
				fprintf(stderr, "pushed some tasks to %p\n", to);
			else
				fprintf(stderr, "I didn't have anything for %p\n", to);
		}
	}

	/* There is room now */
	//~ printf("Can push OK!\n");
	return didwork || starpu_sched_component_can_push(component, to);
}

static int basic_can_pull(struct starpu_sched_component * component)
{
	struct basic_sched_data *data = component->data;


	//~ printf("Can pull OK!\n");
	return starpu_sched_component_can_pull(component);
}
//Fin dummy

struct starpu_sched_component *starpu_sched_component_basic_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "basic");
	
	struct basic_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	//~ my_data = malloc(sizeof(*my_data));
	
	//~ printf("Create breakpoint 1\n");
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	//~ printf("Create breakpoint 2\n");
	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->tache_pop);
	//~ printf("Create breakpoint 3\n");
	starpu_task_list_init(&my_data->sub_list);
	//~ printf("Create breakpoint 4\n");
	
	//La je dois initialiser ma liste chainée
	//~ my_data->sub_list = 0;
 
	my_data->next = NULL;
	data->head = my_data;
	
	//~ data->head = my_data;
	//~ data->first_link = my_data;
	//~ data->head->next = NULL;
	//~ printf("Create breakpoint 5\n");
	
	component->data = data;
	component->push_task = basic_push_task;
	component->pull_task = basic_pull_task;
	component->can_push = basic_can_push;
	component->can_pull = basic_can_pull;

	printf("Create OK!\n");
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
	printf("Initialize OK!\n");
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

