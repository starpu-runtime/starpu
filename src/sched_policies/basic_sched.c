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

/* Ordonnanceur de base */

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


//Dummy
struct basic_sched_data
{
	struct starpu_task_list tache_pop;
	struct starpu_task_list sub_list;
	
	//~ struct basic_sched_data *next;
	struct starpu_task_list *next;
	
	struct starpu_task_list first;
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

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

	return 0;
}

static struct starpu_task *basic_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct basic_sched_data *data = component->data;	
	int i = 0; int j = 0; int nb_pop = 0; int temp_nb_pop = 0; int tab_runner = 0; int max_donnees_commune = 0; int k = 0;
	
	int taille_tache = 0;
	
//Doesn't work for me	
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_task_list_empty(&data->sched_list))
		return NULL;
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task1 = NULL;
	//struct starpu_task *task2 = NULL;
	//struct starpu_task *task_temp_1 = NULL;
	//struct starpu_task *task_temp_2 = NULL;
	
	//If the list is not empty, it means that we have task to get out of pull before pulling more tasks
	//If we use a linked list we need to go to the next one and verify it's not equal to NULL
	//Else we can pull tasks
	
	//Version 1 liste---------------------------------------------------------------------------------
	//~ if (starpu_task_list_empty(&data->tache_pop)) {
	//------------------------------------------------------------------------------------------------
	
	//Version liste chainée --------------------------------------------------------------------------
	//Si le next est null et que la liste est vide, alors on peut pull à nouveau des tâches
	if ((data->next == NULL) && (starpu_task_list_empty(&data->sub_list))) {
		data->sub_list = data->first;
	//------------------------------------------------------------------------------------------------
	
		if (!starpu_task_list_empty(&data->sched_list)) {
			while (!starpu_task_list_empty(&data->sched_list)) {
				//Pulling all tasks and counting them
				task1 = starpu_task_list_pop_back(&data->sched_list);
				//Getting the "size" of a task for later, it's bad here cause it's in a while loop
				taille_tache = STARPU_TASK_GET_NBUFFERS(task1);
				printf("taille = %d\n",taille_tache);
				printf("Pulling task %p\n",task1);
				nb_pop++;
				printf("%d task(s) have been poped \n",nb_pop);
				
				//True line
				starpu_task_list_push_back(&data->tache_pop,task1);
				
				//Test
				//~ starpu_task_list_push_back(&data_liste->sub_list,task1);
				//~ &data_liste->next;
			}
			if (nb_pop > 2) {
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
				printf("Handles tab : \n");
				for (i = 0; i < nb_pop*taille_tache; i++) {
					printf("%p\n",handles[i]);
				}
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
				printf("Data matrix complete : \n");
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						printf (" %d ",matrice_donnees_commune[i][j]);
					}
					printf("\n");
				}
				
				//Pour l'instant vu que les comparaisons ne trouvait que 1 point commun entre les taches, je fonctionne comme ca. 
				//Plus tard il y aura plus que 1 point commun, il faudra donc comparer et voir qui a le plus de points commun
				
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
				printf("test 1 ok\n");
				//Here, if a tab has 0 in it, it means that a linked task got put in the tab so we have to put this one too next to it
				for (i = 0; i < nb_pop; i++) {
					if (task_tab[i] != 0) { starpu_task_list_push_back(&data->sub_list,task_tab[i]); task_tab[i] = 0; }
					printf("test 2 ok\n");
					for (j = i + 1; j< nb_pop; j++) {
						if (matrice_donnees_commune[i][j] == 4) {
							printf ("Data in common\n");
							if (task_tab[j] != 0) { starpu_task_list_push_back(&data->sub_list,task_tab[j]); task_tab[j] = 0; }
						}
					}
					//Here we go on the next list of our linked list of task
					data->sub_list = data->next;
				}
				//Now tasks are in sub_list
				//Need to get back to the beggining of the list without erasing everything ???
				//data->sub_list = data_controle->first;
				//-------------------------------------------------------------------------------------------------------------------------
				
				
				//Pas utile normalement, ca sert juste a voir si toutes les taches ont bien été mise dans la liste, qu'il reste rien dans me tableau
				//~ for (i = 0; i < nb_pop; i++) {
					//~ if (task_tab[i] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[i]); task_tab[i] = 0; }
				//~ }
				
				//Version 1 liste
				//~ task1 = starpu_task_list_pop_front(&data->tache_pop);
				
				//Version liste chainée
				task1 = starpu_task_list_pop_front(&data->sub_list);
			}
			//Else here means that we have only 2 task or less, so no need to compare the handles
			//A voir si il faut le garder ou pas dans le cas des liste chainées
			else {
				task1 = starpu_task_list_pop_front(&data->tache_pop);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	//Version 1 liste -----------------------------------------------------------------------------------------------------------------------
	//~ task1 = starpu_task_list_pop_front(&data->tache_pop);
	//~ STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	//~ return task1;
	//---------------------------------------------------------------------------------------------------------------------------------------
	
	//Version liste chainée -----------------------------------------------------------------------------------------------------------------
	if (starpu_task_list_empty(&data->sub_list)) {
		//la liste est vide on passe au prochain maillon, qui n'est pas NULL car on a check plus haut
		data->sub_list = data->next;
		task1 = starpu_task_list_pop_front(&data->sub_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	else {
		task1 = starpu_task_list_pop_front(&data->sub_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	//---------------------------------------------------------------------------------------------------------------------------------------
		
		
}

static int basic_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct basic_sched_data *data = component->data;
	int didwork = 0;

	//if (data->verbose)
		fprintf(stderr, "tells me I can push to him\n");

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
	return didwork || starpu_sched_component_can_push(component, to);
}

static int basic_can_pull(struct starpu_sched_component * component)
{
	struct basic_sched_data *data = component->data;

	//if (data->verbose)
		fprintf(stderr,"telling below they can pull\n");

	return starpu_sched_component_can_pull(component);
}
//Fin dummy

struct starpu_sched_component *starpu_sched_component_basic_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "basic");
	
	struct basic_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	//data_controle = init_liste_tache(component);
	
	//Init de la liste chainée
	data->next = NULL;
	data->first = data->sub_list;
	
	//~ data_liste->next = NULL;
	//~ data_controle->first = data_liste;
	//~ data_liste->sub_list = data_liste;
	

	//_starpu_prio_deque_init(&data->prio);
	//STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	//data->mct_data = mct_data;
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->tache_pop);
	starpu_task_list_init(&data->sub_list);
	starpu_task_list_init(&data->first);
	
	component->data = data;
	component->push_task = basic_push_task;
	component->pull_task = basic_pull_task;
	component->can_push = basic_can_push;
	component->can_pull = basic_can_pull;

	//~ struct basic_sched_data *data = malloc(sizeof(*data));
	//~ STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	//~ starpu_task_list_init(&data->sched_list);
	//~ data->verbose = params->verbose;
	//Fifo
	// struct starpu_sched_component *component = starpu_sched_component_create(tree, "fifo");
	// struct _starpu_fifo_data *data;
	// _STARPU_MALLOC(data, sizeof(*data));
	// data->fifo = _starpu_create_fifo();
	// STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	// //Fin fifo
	// component->data = data;
	// component->push_task = fifo_push_task;
	// component->pull_task = fifo_pull_task;
	// component->can_push = fifo_can_push;
	// component->can_pull = fifo_can_pull;
	//~ component->deinit_data = basic_deinit_data;

	return component;
}

static void initialize_basic_center_policy(unsigned sched_ctx_id)
{
	printf("Debut initialize\n");
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_basic_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);

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

