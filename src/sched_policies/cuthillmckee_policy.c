/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* CM
 */

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
#include <sched_policies/sched_visu.h>

#define REVERSE /* O or 1 */
static int reverse;
static bool do_schedule_done_cm = false;

/* Structure used to acces the struct my_list_cm. There are also task's list */
struct cuthillmckee_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	struct starpu_task_list SIGMA; /* order in which task will go out */
	/* All the pointer use to navigate through the linked list */
	struct my_list_cm *temp_pointer_1;
	struct my_list_cm *first_link; /* Pointer that we will use to point on the first link of the linked list */
	//~ int id;
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

struct my_list_cm
{
	int index;
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct my_list_cm *next;
};

/* Put a link at the beginning of the linked list */
static void insertion_cuthillmckee(struct cuthillmckee_sched_data *a)
{
	struct my_list_cm *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
	new->next = a->temp_pointer_1;
	a->temp_pointer_1 = new;
}

/* Pushing the tasks */
static int cuthillmckee_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct cuthillmckee_sched_data *data = component->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *cuthillmckee_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	_STARPU_SCHED_PRINT("Début de cuthillmckee_pull_task\n");

	(void)to;
	struct cuthillmckee_sched_data  *data = component->data;
	struct starpu_task *task1 = NULL;
	if (do_schedule_done_cm == true)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

		/* If one or more task have been refused */
		if (!starpu_task_list_empty(&data->list_if_fifo_full))
		{
			task1 = starpu_task_list_pop_back(&data->list_if_fifo_full);
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			return task1;
		}
		/* If the linked list is empty, we can pull more tasks */
		if (starpu_task_list_empty(&data->SIGMA))
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			return NULL;
		}
		else
		{
			task1 = starpu_task_list_pop_front(&data->SIGMA);

			_sched_visu_print_data_to_load_prefetch(task1, starpu_worker_get_id() + 1, 1);

			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			_STARPU_SCHED_PRINT("Task %p is getting out of pull_task\n", task1);
			return task1;
		}
	}
	return NULL;
}

static int cuthillmckee_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct cuthillmckee_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
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

static int cuthillmckee_can_pull(struct starpu_sched_component * component)
{
	return starpu_sched_component_can_pull(component);
}

static void cuthillmckee_do_schedule(struct starpu_sched_component *component)
{
	int i, j, tab_runner, tab_runner_bis, nb_voisins = 0;
	int poids_aretes_min = INT_MAX; int indice_poids_aretes_min = INT_MAX;
	struct cuthillmckee_sched_data *data = component->data;
	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;

	int NT = 0;

	/* If the linked list is empty, we can pull more tasks */
	if (starpu_task_list_empty(&data->SIGMA))
	{
		if (!starpu_task_list_empty(&data->sched_list))
		{
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list))
			{
				task1 = starpu_task_list_pop_front(&data->sched_list);
				NT++;
				starpu_task_list_push_back(&data->popped_task_list,task1);
			}

			//~ int matrice_adjacence[NT][NT]; for (i = 0; i < NT; i++) { for (j = 0; j < NT; j++) { matrice_adjacence[i][j] = 0; } }
			long int matrice_adjacence[NT][NT];
			for (i = 0; i < NT; i++)
			{
				for (j = 0; j < NT; j++)
				{
					matrice_adjacence[i][j] = 0;
				}
			}
			temp_task_1 = starpu_task_list_begin(&data->popped_task_list);
			temp_task_2 = starpu_task_list_begin(&data->popped_task_list);
			for (i = 0; i < NT; i++)
			{
				for (j = 0; j < NT; j++)
				{
					if (i != j)
					{
						unsigned i_bis;
						for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++)
						{
							unsigned j_bis;
							for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++)
							{
								if (STARPU_TASK_GET_HANDLE(temp_task_1, i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2, j_bis))
								{
									matrice_adjacence[i][j]++;
								}
								/* Pour adapter a des tailles heterogènes ici il faut juste faire : et declrer poids des arretes et matrice adja en long int */
								if (STARPU_TASK_GET_HANDLE(temp_task_1,i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2,j_bis))
								{
									matrice_adjacence[i][j] += starpu_data_get_size(STARPU_TASK_GET_HANDLE(temp_task_1, i_bis));
								}
							}
						}
					}
					temp_task_2  = starpu_task_list_next(temp_task_2);
				}
				temp_task_1  = starpu_task_list_next(temp_task_1);
				temp_task_2  = starpu_task_list_begin(&data->popped_task_list);
			}
			/* Affichage de la matrice d'adjacence */
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Matrice d'adjacence :\n"); for (i = 0; i < NT; i++) { for (j = 0; j < NT; j++) { printf("%d ",matrice_adjacence[i][j]); } printf("\n"); } }

			//NEW
			int do_not_add_more = 0;
			while (!starpu_task_list_empty(&data->popped_task_list))
			{
				starpu_task_list_push_back(&data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&data->popped_task_list));
				data->temp_pointer_1->index = do_not_add_more;
				if (do_not_add_more != NT-1)
				{
					insertion_cuthillmckee(data);
				}
				do_not_add_more++;
			}
			data->first_link = data->temp_pointer_1;

			//~ int poids_aretes[NT];
			long int poids_aretes[NT];
			int tab_SIGMA[NT];	for (i = 0; i < NT; i++) { tab_SIGMA[i] = -1; }

			/* Calcul du poids des arêtes de chaque sommet */
			for (i = 0; i < NT; i++)
			{
				poids_aretes[i] = 0;
				for (j = 0; j < NT; j++)
				{
					if (matrice_adjacence[i][j] != 0)
					{
						poids_aretes[i] += matrice_adjacence[i][j];
					}
				}
			}
			tab_runner = 0;
			tab_runner_bis = 0;
			while (tab_runner < NT)
			{
				//~ for (i = 0; i < NT; i++) { if (poids_aretes[i] != -1) { poids_aretes[i] = 0; } }
				/* Si tab_SIGMA est vide ou qu'on a déjà exploré tous ses sommets on prend le sommet dont le poids des arêtes est le plus faible, sinon on passe au sommet de tab_SIGMA suivant (cas de graphe non connexe en fait) */
				//OLD
				if (tab_SIGMA[tab_runner] == -1)
				{
					//NEW
					//~ if (tab_SIGMA[tab_runner] == -1 && tab_runner != NT) {
					/* Recherche du sommet dont le poids des arêtes est minimal */
					poids_aretes_min = INT_MAX; indice_poids_aretes_min = INT_MAX;
					for (i = 0; i < NT; i++)
					{
						if (poids_aretes_min > poids_aretes[i] && poids_aretes[i] != - 1)
						{
							poids_aretes_min = poids_aretes[i]; indice_poids_aretes_min = i;
						}
					}

					//~ tab_SIGMA[tab_runner_bis] = indice_poids_aretes_min;
					//~ temp_task_1  = starpu_task_list_begin(&data->popped_task_list); for (i = 0; i < indice_poids_aretes_min; i++) { temp_task_1  = starpu_task_list_next(temp_task_1); }
					//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Add %p to sigma\n",temp_task_1); }
					tab_SIGMA[tab_runner_bis] = indice_poids_aretes_min;
					//~ printf("ok1\n");

					//~ strcpy(char_SIGMA[tab_runner_bis],starpu_task_get_name(temp_task_1));
					//~ strcpy(char_SIGMA[tab_runner_bis],"oui");
					//~ printf("ok2\n");

					tab_runner_bis++;
					poids_aretes[indice_poids_aretes_min] = -1;
				}
				else
				{
					/* On étudie les sommets de tab_SIGMA non explorés */
					while (tab_runner < NT && tab_SIGMA[tab_runner] != -1)
					{
						//~ tab_runner++;
						//~ for (j = 0; j < NT; j++) { printf("%d ",matrice_adjacence[tab_SIGMA[tab_runner]][j]); } printf("\n");
						/* Pour chaque voisins, on les mets dans tab_SIGMA si ils n'y sont pas déjà, par poids des arêtes croissant */
						/* Recherche du nb de voisins pour la boucle suivante */
						for (j = 0; j < NT; j++)
						{
							if (matrice_adjacence[tab_SIGMA[tab_runner]][j] != 0 && matrice_adjacence[tab_SIGMA[tab_runner]][j] != -1 && poids_aretes[j] != -1)
							{
								nb_voisins++;
							}
						}

						if (nb_voisins == 0)
						{
							tab_runner++;
						}
						else
						{
							int i_bis;
							for (i_bis = 0; i_bis < nb_voisins; i_bis++)
							{
								//~ poids_aretes_min_bis = INT_MAX; indice_poids_aretes_min_bis = INT_MAX;
								//~ if (poids_aretes[i_bis] != -1) {
								/* Recherche du min du poids des arêtes */
								poids_aretes_min = INT_MAX; indice_poids_aretes_min = INT_MAX;
								for (j = 0; j < NT; j++)
								{
									if (poids_aretes_min > matrice_adjacence[tab_SIGMA[tab_runner]][j] && poids_aretes[j] != -1 && matrice_adjacence[tab_SIGMA[tab_runner]][j] != 0)
									{
										poids_aretes_min = matrice_adjacence[tab_SIGMA[tab_runner]][j]; indice_poids_aretes_min = j;
									}
								}

								/* Ajout à tab_SIGMA */
								tab_SIGMA[tab_runner_bis] = indice_poids_aretes_min;
								tab_runner_bis++;
								/* On supprime ce sommet de la liste des possibilité et on recommence à chercher le max parmi les voisins */
								poids_aretes[indice_poids_aretes_min] = -1;
							}
							nb_voisins = 0;
							tab_runner++;
						}
					}
				}
			}

			/* I put the task in order in SIGMA */
			_sched_visu_print_vector(tab_SIGMA, NT, "tab_SIGMA[i] : ");

			if (reverse == 1)
			{
				int tab_SIGMA_2[NT];
				for (i = NT - 1, j = 0; i >= 0; i--, j++)
					tab_SIGMA_2[j] = tab_SIGMA[i];
				for (i = 0; i < NT; i++)
					tab_SIGMA[i] = tab_SIGMA_2[i];
			}

			i = 0;
			data->temp_pointer_1 = data->first_link;

			while (i != NT)
			{
				//~ temp_task_1  = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list);
				if (tab_SIGMA[i] == data->temp_pointer_1->index)
				{
					//~ if (strcmp(char_SIGMA[i],starpu_task_get_name(temp_task_1) == 0)) {
					starpu_task_list_push_back(&data->SIGMA,starpu_task_list_pop_front(&data->temp_pointer_1->sub_list));
					i++;
					data->temp_pointer_1 = data->first_link;
				}
				else
				{
					data->temp_pointer_1 = data->temp_pointer_1->next;
				}
			}
			do_schedule_done_cm = true;
		}
	}
}

struct starpu_sched_component *starpu_sched_component_cuthillmckee_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	reverse = starpu_get_env_number_default("REVERSE", 0);
	_starpu_visu_init();

	//~ srandom(time(0)); /* For the random selection in ALGO 4 */
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "cuthillmckee");

	struct cuthillmckee_sched_data *data;
	struct my_list_cm *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));

	do_schedule_done_cm = false;

	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&data->SIGMA);

	starpu_task_list_init(&my_data->sub_list);
	my_data->next = NULL;
	data->temp_pointer_1 = my_data;

	component->data = data;
	component->do_schedule = cuthillmckee_do_schedule;
	component->push_task = cuthillmckee_push_task;
	component->pull_task = cuthillmckee_pull_task;
	component->can_push = cuthillmckee_can_push;
	component->can_pull = cuthillmckee_can_pull;

	return component;
}

static void initialize_cuthillmckee_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_cuthillmckee_create, NULL,
							   STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
							   STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
							   STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_cuthillmckee_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_cuthillmckee_policy =
{
	.init_sched = initialize_cuthillmckee_center_policy,
	.deinit_sched = deinitialize_cuthillmckee_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = _sched_visu_get_data_to_load,
	.pre_exec_hook = _sched_visu_get_current_tasks,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.policy_name = "cuthillmckee",
	.policy_description = "cuthillmckee algorithm",
	.worker_type = STARPU_WORKER_LIST,
};
