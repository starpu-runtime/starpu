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
#define ORDER_U /* O or 1 */

/* Structure used to acces the struct my_list. There are also task's list */
struct HFP_sched_data
{
	int ALGO_USED_READER;
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	
	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *temp_pointer_2;
	struct my_list *temp_pointer_3;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

/* Structure used to store all the variable we need and the tasks of each package. Each link is a package */
struct my_list
{
	int package_nb_data; 
	int nb_task_in_sub_list;
	int index_package; /* Used to write in Data_coordinates.txt and keep track of the initial index of the package */
	starpu_data_handle_t * package_data; /* List of all the data in the packages. We don't put two times the duplicates */
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct my_list *next;
	/* The separator of the last state of the current package */
	int split_last_ij;
};

/* Empty a task's list. We use this for the lists last_package */
void HFP_empty_list(struct starpu_task_list *a)
{
	struct starpu_task *task = NULL;
	for (task  = starpu_task_list_begin(a); task != starpu_task_list_end(a); task = starpu_task_list_next(task)) {
		starpu_task_list_erase(a,task);
	}
}

/* Put a link at the beginning of the linked list */
void HFP_insertion(struct HFP_sched_data *a)
{
    struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
    new->next = a->temp_pointer_1;
    a->temp_pointer_1 = new;
}

/* Delete all the empty packages */
struct my_list* HFP_delete_link(struct HFP_sched_data* a)
{
	while (a->first_link != NULL && a->first_link->package_nb_data == 0) {
		a->temp_pointer_1 = a->first_link;
		a->first_link = a->first_link->next;
		free(a->temp_pointer_1);
	}
	if (a->first_link != NULL) {
		a->temp_pointer_2 = a->first_link;
		a->temp_pointer_3 = a->first_link->next;
		while (a->temp_pointer_3 != NULL) {
			while (a->temp_pointer_3 != NULL && a->temp_pointer_3->package_nb_data == 0) {
				a->temp_pointer_1 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
				a->temp_pointer_2->next = a->temp_pointer_3;
				free(a->temp_pointer_1);
			}
			if (a->temp_pointer_3 != NULL) {
				a->temp_pointer_2 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
			}
		}
	}
	return a->first_link;
}

struct my_list* HFP_reverse_sub_list(struct my_list *a) 
{
	struct starpu_task_list b;
	starpu_task_list_init(&b);
	while (!starpu_task_list_empty(&a->sub_list)) {				
		starpu_task_list_push_front(&b,starpu_task_list_pop_front(&a->sub_list));
	}
	while (!starpu_task_list_empty(&b)) {				
		starpu_task_list_push_back(&a->sub_list,starpu_task_list_pop_front(&b));
	}
	return a; 	
}

int get_common_data_last_package(struct my_list*I, struct my_list*J, int evaluation_I, int evaluation_J, bool IJ_inferieur_GPU_RAM, starpu_ssize_t GPU_RAM_M) 
{
	int split_ij = 0;
	//~ printf("I a %d taches et %d données\n",I->nb_task_in_sub_list,I->package_nb_data);
	//~ printf("J a %d taches et %d données\n",J->nb_task_in_sub_list,J->package_nb_data);
	//~ printf("Split de last ij de I = %d\n",I->split_last_ij);
	//~ printf("Split de last ij de J = %d\n",J->split_last_ij);
	/* evaluation: 0 = tout, 1 = début, 2 = fin */
	struct starpu_task *task = NULL; bool insertion_ok = false;										
	bool donnee_deja_presente = false; int j = 0; int i = 0;
	int common_data_last_package = 0; long int poids_tache_en_cours = 0; long int poids = 0;
	int index_tab_donnee_I = 0; int index_tab_donnee_J = 0; int parcours_liste = 0; int i_bis = 0;
	
	starpu_data_handle_t * donnee_J = malloc((J->package_nb_data) * sizeof(J->package_data[0]));
	for (i = 0; i < J->package_nb_data; i++) { donnee_J[i] = NULL; }
	starpu_data_handle_t * donnee_I = malloc((I->package_nb_data) * sizeof(I->package_data[0]));
	
	if (evaluation_I == 0) {
		for (i = 0; i < I->package_nb_data; i++) {
			donnee_I[i] = I->package_data[i];
		}
		index_tab_donnee_I = I->package_nb_data;
	}
	else if (evaluation_I == 1 && IJ_inferieur_GPU_RAM == false) {
		poids = 0; insertion_ok = false;
		task = starpu_task_list_begin(&I->sub_list);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		while(1) {
			task = starpu_task_list_next(task);
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i)); 				
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M) {
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
					if (tab_tache_en_cours[i] != NULL) { donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i]; 
						index_tab_donnee_I++;
						insertion_ok = true;
					}											
				} 
				if (insertion_ok == true) { poids += poids_tache_en_cours; }
				insertion_ok = false;
			}
			else { break; }											
		}
		printf("Poids a la fin pour i1 : %li\n",poids);	
	}
	else if (evaluation_I == 2 && IJ_inferieur_GPU_RAM == false) { 
		poids = 0;
		i_bis = 1; insertion_ok = false;
		task = starpu_task_list_begin(&I->sub_list);
		while(starpu_task_list_next(task) != NULL) { 
			task = starpu_task_list_next(task);
			//~ printf("%p\n",task);
		}
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		while(1) {
			i_bis++;
			task = starpu_task_list_begin(&I->sub_list);
			for (parcours_liste = I->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--) {
				task = starpu_task_list_next(task);
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i)); 				
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M) {
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
					if (tab_tache_en_cours[i] != NULL) { donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i]; 
						index_tab_donnee_I++;
						insertion_ok = true;
					}											
				} 
				if (insertion_ok == true) { poids += poids_tache_en_cours; }
				insertion_ok = false;
			}
			else { break; }											
		}
		printf("Poids a la fin pour i2 : %li\n",poids);	
	}
	else if (IJ_inferieur_GPU_RAM == true) {
		if (evaluation_I == 0) { printf("Error evaluation de I alors que I et J <= GPU_RAM\n"); exit(0); }
		if (evaluation_I == 2) { split_ij = I->nb_task_in_sub_list - I->split_last_ij + 1; } else { split_ij = I->split_last_ij + 1; }
		task = starpu_task_list_begin(&I->sub_list);
		if (evaluation_I == 2) { 
			while(starpu_task_list_next(task) != NULL) { 
				task = starpu_task_list_next(task);
			}
		}
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task,i);
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		for (i_bis = 2; i_bis < split_ij; i_bis++) {
			if (evaluation_I == 2) { 
				task = starpu_task_list_begin(&I->sub_list);
				for (parcours_liste = I->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--) {
					task = starpu_task_list_next(task);
				}
			}
			else { task = starpu_task_list_next(task); }	
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				if (tab_tache_en_cours[i] != NULL) { donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i]; 
					index_tab_donnee_I++;
				}											
			} 
		} 
	}
	
	if (evaluation_J == 0) {
		for (i = 0; i < J->package_nb_data; i++) {
			donnee_J[i] = J->package_data[i];
		}
		index_tab_donnee_J = J->package_nb_data;
	}
	else if (evaluation_J == 1 && IJ_inferieur_GPU_RAM == false) {
		poids = 0; insertion_ok = false;
		task = starpu_task_list_begin(&J->sub_list);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		while(1) {
			task = starpu_task_list_next(task);
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i)); 				
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M) {
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
					if (tab_tache_en_cours[i] != NULL) { donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i]; 
						index_tab_donnee_J++;
						insertion_ok = true;
					}											
				} 
				if (insertion_ok == true) { poids += poids_tache_en_cours; }
				insertion_ok = false;
			}
			else { break; }											
		}	
		printf("Poids a la fin pour j1 : %li\n",poids);	
	}
	else if (evaluation_J == 2 && IJ_inferieur_GPU_RAM == false) {
		poids = 0;
		i_bis = 1; insertion_ok = false;
		/* Se placer sur la dernière tâche du paquet J */
		task = starpu_task_list_begin(&J->sub_list);
		//~ printf("%p\n",task);
		while(starpu_task_list_next(task) != NULL) { 
			task = starpu_task_list_next(task);
		}
		//~ printf("-----\nJe suis sur la tâche %p\n",task);	
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		while(1) {
			i_bis++;
			task = starpu_task_list_begin(&J->sub_list);
			for (parcours_liste = J->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--) {
				task = starpu_task_list_next(task);
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i)); 				
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M) {
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
					if (tab_tache_en_cours[i] != NULL) { donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i]; 
						index_tab_donnee_J++;
						insertion_ok = true;
					}											
				} 
				if (insertion_ok == true) { poids += poids_tache_en_cours; }
				insertion_ok = false;
			}
			else { break; }											
		}
		printf("Poids a la fin pour j2 : %li\n",poids);		
	}
	else if (IJ_inferieur_GPU_RAM == true) {
		if (evaluation_J == 0) { printf("Error evaluation de J alors que I et J <= GPU_RAM\n"); exit(0); }
		if (evaluation_J == 2) { split_ij = J->nb_task_in_sub_list - J->split_last_ij + 1; } else { split_ij = J->split_last_ij + 1; }
		task = starpu_task_list_begin(&J->sub_list);
		if (evaluation_J == 2) { 
			while(starpu_task_list_next(task) != NULL) { 
				task = starpu_task_list_next(task);
			}
		}
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		for (i_bis = 2; i_bis < split_ij; i_bis++) {
			if (evaluation_J == 2) { 
				task = starpu_task_list_begin(&J->sub_list);
				for (parcours_liste = J->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--) {
					task = starpu_task_list_next(task);
				}
			}
			else { task = starpu_task_list_next(task); }	
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) { tab_tache_en_cours[i] = NULL; }
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++) {
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j]) {
						donnee_deja_presente = true;
						break; 
					}																									
				}												
				if (donnee_deja_presente == false) { 
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i); 
				}
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				if (tab_tache_en_cours[i] != NULL) { donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i]; 
					index_tab_donnee_J++;
				}											
			} 
		} 
	}
	//~ printf("I a un index de %d et %d données\n",index_tab_donnee_I,I->package_nb_data);		
	//~ printf("J a un index de %d et %d données\n",index_tab_donnee_J,J->package_nb_data);
	printf("Données de I:"); for (i = 0; i < index_tab_donnee_I; i++) { printf(" %p",donnee_I[i]); }
	printf("\n");
	printf("Données de J:"); for (i = 0; i < index_tab_donnee_J; i++) { printf(" %p",donnee_J[i]); }
	printf("\n");
	for (i = 0; i < index_tab_donnee_I; i++) {
		for (j = 0; j < index_tab_donnee_J; j++) {
			if (donnee_I[i] == donnee_J[j]) {
				common_data_last_package++;
				printf("%p\n",donnee_I[i]);
				break;
			}
		}
	}
	return common_data_last_package;
}

/* Give a color for each package. Written in the file Data_coordinates.txt */
static void rgb(int num, int *r, int *g, int *b)
{
    int i = 0;

    if (num < 7) {
	num ++;
	*r = num & 1 ? 255 : 0;
	*g = num & 2 ? 255 : 0;
	*b = num & 4 ? 255 : 0;
	return;
    }

    num -= 7;

    *r = 0; *g = 0; *b = 0;
    for (i = 0; i < 8; i++) {
        *r = *r << 1 | ((num & 1) >> 0);
        *g = *g << 1 | ((num & 2) >> 1);
        *b = *b << 1 | ((num & 4) >> 2);
        num >>= 3;
    }
}

/* Comparator used to sort the data of a packages to erase the duplicate in O(n) */
int HFP_pointeurComparator ( const void * first, const void * second ) {
  return ( *(int*)first - *(int*)second );
}

/* Pushing the tasks */		
static int HFP_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct HFP_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *HFP_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	int common_data_last_package_i2_j = 0; int common_data_last_package_i1_j = 0; int index_tab_donnee_i1 = 0; int common_data_last_package_i_j1 = 0; int common_data_last_package_i_j2 = 0;
	struct HFP_sched_data *data = component->data;
	
	/* Variables */
	/* Variables used to calculate, navigate through a loop or other things */
	int i = 0; int j = 0; int tab_runner = 0; int do_not_add_more = 0; int index_head_1 = 0; int index_head_2 = 0; int i_bis = 0; int j_bis = 0; double number_tasks = 0; int random_value = 0;
	/* double mean_task_by_packages = 0; */ double temp_moyenne = 0; double temp_variance = 0; double temp_ecart_type = 0;  double moyenne = 0; double ecart_type = 0;
	int min_nb_task_in_sub_list = 0; int nb_min_task_packages = 0; int temp_nb_min_task_packages = 0; int red = 0; int green = 0; int blue = 0; int temp_i_bis = 0;
	struct starpu_task *task1 = NULL; struct starpu_task *temp_task_1 = NULL; struct starpu_task *temp_task_2 = NULL;
	int Nb_package_forbidden = 0; int Nb_package = 0;
		 
	int nb_pop = 0; /* Variable used to track the number of tasks that have been popped */
	int nb_common_data = 0; /* Track the number of packages that have data in commons with other packages */
	int link_index = 0; /* Track the number of packages */
	int nb_duplicate_data = 0; /* Used to store the number of duplicate data between two packages */
	long int weight_two_packages; /* Used to store the weight the merging of two packages would be. It is then used to see if it's inferior to the size of the RAM of the GPU */
	long int max_value_common_data_matrix = 0; /* Store the maximum weight of the commons data between two packages for all the tasks */
	int nb_of_loop = 0; /* Number of iteration of the while loop */
	int packaging_impossible = 0; /* We use this to stop the while loop and thus stop the packaging. 0 = false, 1 = true */
	int temp_tab_coordinates[2]; /* Tab to store the coordinate of a data */
	int bool_data_common = 0; /* ""boolean"" used to check if two packages have data in commons whe we merge them */
	int GPU_limit_switch = 1; /* On 1 it means we use the size of the GPU limit. It is usefull for algorithm 3 that remove this limit at the end of it execution */	
	int nb_grouping_available = 0; /* Used in algorithm 4 to track the number of package a ackage can merge with and then choose a random one */
	/* List used to store tasks in sub package and then compare them to apply order-U */
	struct starpu_task_list sub_package_1_i; /* Used for order U to store the tasks of the sub package 1 of i */
	struct starpu_task_list sub_package_2_i;
	struct starpu_task_list sub_package_1_j;
	struct starpu_task_list sub_package_2_j;
	starpu_task_list_init(&sub_package_1_i);
	starpu_task_list_init(&sub_package_2_i);
	starpu_task_list_init(&sub_package_1_j);
	starpu_task_list_init(&sub_package_2_j);
	/* Variable used to store the common data weight beetween two sub packages of packages i and j before merging */
	long int common_data_last_package_i1_j1 = 0; /* Variables used to compare the affinity between sub package 1i and 1j, 1i and 2j etc... */
	long int common_data_last_package_i1_j2 = 0; 
	long int common_data_last_package_i2_j1 = 0; 
	long int common_data_last_package_i2_j2 = 0; 
	long int max_common_data_last_package = 0;
	long int weight_package_i = 0; /* Used for ORDER_U too */
	long int weight_package_j = 0;
	long int temp_weight = 0;
	int nb_task_until_B_1_i = 0; /* int to separate the last two sub packages of the package i */
	int nb_task_until_B_2_i = 0;
	int nb_task_until_B_1_j = 0;
	int nb_task_until_B_2_j = 0;
		
	/* Here we calculate the size of the RAM of the GPU. We allow our packages to have half of this size */
	starpu_ssize_t GPU_RAM_M = 0;
	STARPU_ASSERT(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component));
	GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));
	
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->list_if_fifo_full)) {
		task1 = starpu_task_list_pop_back(&data->list_if_fifo_full); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return task1;
	}

	/* If the linked list is empty, we can pull more tasks */
	if ((data->temp_pointer_1->next == NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) {
		if (!starpu_task_list_empty(&data->sched_list)) {
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list)) {				
				task1 = starpu_task_list_pop_front(&data->sched_list);
				nb_pop++;
				starpu_task_list_push_back(&data->popped_task_list,task1);
			} 		
			number_tasks = nb_pop;
					
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("%d task(s) have been pulled\n",nb_pop); }
			
			temp_task_1  = starpu_task_list_begin(&data->popped_task_list);
			data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->temp_pointer_1->package_data[0]));
			
			/* One task == one link in the linked list */
			do_not_add_more = nb_pop - 1;
			for (temp_task_1  = starpu_task_list_begin(&data->popped_task_list); temp_task_1 != starpu_task_list_end(&data->popped_task_list); temp_task_1  = temp_task_2) {
				temp_task_2 = starpu_task_list_next(temp_task_1);
				temp_task_1 = starpu_task_list_pop_front(&data->popped_task_list);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(temp_task_1); i++) {
					data->temp_pointer_1->package_data[i] = STARPU_TASK_GET_HANDLE(temp_task_1,i);
				}
				data->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(temp_task_1);
				/* We sort our datas in the packages */
				qsort(data->temp_pointer_1->package_data,data->temp_pointer_1->package_nb_data,sizeof(data->temp_pointer_1->package_data[0]),HFP_pointeurComparator);
				/* Pushing the task and the number of the package in the package*/
				starpu_task_list_push_back(&data->temp_pointer_1->sub_list,temp_task_1);
				data->temp_pointer_1->index_package = link_index;
				/* Initialization of the lists last_packages */
				data->temp_pointer_1->split_last_ij = 0;
				
				link_index++;
				data->temp_pointer_1->nb_task_in_sub_list ++;
				
				if(do_not_add_more != 0) { HFP_insertion(data); data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->temp_pointer_1->package_data[0])); }
				do_not_add_more--;
			}
			data->first_link = data->temp_pointer_1;				
			
			/* Code to print all the data of all the packages */
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) {
				//~ link_index = 0;
				//~ printf("Initialement on a : \n");
				//~ while (data->temp_pointer_1 != NULL) {
					//~ for (i = 0; i < 3; i++) {
						//~ printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->temp_pointer_1->package_data[i],temp_task_1  = starpu_task_list_begin(&data->temp_pointer_1->sub_list),link_index);
					//~ }
					//~ link_index++;
					//~ data->temp_pointer_1 = data->temp_pointer_1->next;
				//~ } printf("NULL\n");
				//~ data->temp_pointer_1 = data->first_link;
			//~ }
			
			data->temp_pointer_2 = data->first_link;
			index_head_2++;
			
			/* THE while loop. Stop when no more packaging are possible */
			while (packaging_impossible == 0) {
				/* algo 3's goto */
				algo3:
				nb_of_loop++;
				packaging_impossible = 1;
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("############# Itération numéro : %d #############\n",nb_of_loop); }
								
				/* Variables we need to reinitialize for a new iteration */
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link; index_head_1 = 0; index_head_2 = 1; link_index = 0; tab_runner = 0; nb_min_task_packages = 0;
				min_nb_task_in_sub_list = 0; nb_common_data = 0; weight_two_packages = 0; max_value_common_data_matrix = 0; long int matrice_donnees_commune[nb_pop][nb_pop];
				min_nb_task_in_sub_list = data->temp_pointer_1->nb_task_in_sub_list; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
				
				/* For algorithm Algo 4 we need a symmetric matrix and the minimal packages */
				 
					/* First we get the number of packages that have the minimal number of tasks */
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list > data->temp_pointer_1->nb_task_in_sub_list) { min_nb_task_in_sub_list = data->temp_pointer_1->nb_task_in_sub_list; } }
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list == data->temp_pointer_1->nb_task_in_sub_list) { nb_min_task_packages++; } }
					if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list); }
					/* Then we create the common data matrix */
					printf("nb pop = %d\n",nb_pop);
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						for (data->temp_pointer_2 = data->temp_pointer_1->next; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
							for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
								for (j = 0; j < data->temp_pointer_2->package_nb_data; j++) {
									if ((data->temp_pointer_1->package_data[i] == data->temp_pointer_2->package_data[j])) {
										matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->temp_pointer_1->package_data[i]);
										matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(data->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->temp_pointer_1->package_data[i]);
									} } } index_head_2++; } index_head_1++; index_head_2 = index_head_1 + 1; }
				/* Code to print the common data matrix */
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Common data matrix : \n"); for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { printf (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); }}
				
				/* Getting the number of package that have data in commons */
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						if (matrice_donnees_commune[i][j] != 0) { nb_common_data++; } } }
				
				/* Getting back to the beginning of the linked list */
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link;
				
					/* ALGO 4' ou 5 */
					i_bis = 0; j_bis = 0; 
					temp_nb_min_task_packages = nb_min_task_packages;
				debut_while:
					data->temp_pointer_1 = data->first_link;
					data->temp_pointer_2 = data->first_link;
					max_value_common_data_matrix = 0;
					if (GPU_limit_switch == 1) {
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (data->temp_pointer_2 = data->first_link; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
										if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM_M)) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							data->temp_pointer_1=data->temp_pointer_1->next;
							j_bis = 0; }
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link;
				}
				/* Else, we are using algo 5, so we don't check the max weight */
				else {
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (data->temp_pointer_2 = data->first_link; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
										if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); } } 
									if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							data->temp_pointer_1=data->temp_pointer_1->next;
							j_bis = 0; }
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link;
				}		
				i_bis = 0; j_bis = 0; i = 0; j = 0;
				for (i = 0; i < nb_pop; i++) {
					if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) {
						for (j = 0; j < nb_pop; j++) {
							weight_two_packages = 0;  weight_package_i = 0;  weight_package_j = 0;
							for (i_bis = 0; i_bis < data->temp_pointer_1->package_nb_data; i_bis++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i_bis]); } weight_package_i = weight_two_packages;
							for (i_bis = 0; i_bis < data->temp_pointer_2->package_nb_data; i_bis++) { bool_data_common = 0;
								for (j_bis = 0; j_bis < data->temp_pointer_1->package_nb_data; j_bis++) { if (data->temp_pointer_2->package_data[i_bis] == data->temp_pointer_1->package_data[j_bis]) { bool_data_common = 1; } }
								if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i_bis]); } 
								weight_package_j += starpu_data_get_size(data->temp_pointer_2->package_data[i_bis]); }							
							if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j && max_value_common_data_matrix != 0) {
								if ((weight_two_packages <= GPU_RAM_M) || (GPU_limit_switch == 0)) {
								/* Merge */
								packaging_impossible = 0;
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
								
								if (data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list) { temp_nb_min_task_packages--; }
								
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;}
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;}
								nb_common_data--;
								
								if (starpu_get_env_number_default("ORDER_U",0) == 1) {
									printf("I a %d taches et %d données\n",data->temp_pointer_1->nb_task_in_sub_list,data->temp_pointer_1->package_nb_data);
									printf("J a %d taches et %d données\n",data->temp_pointer_2->nb_task_in_sub_list,data->temp_pointer_2->package_nb_data);
									printf("Split de last ij de I = %d\n",data->temp_pointer_1->split_last_ij);
									printf("Split de last ij de J = %d\n",data->temp_pointer_2->split_last_ij); 
									printf("Poids paquet i : %li / Poids paquet j : %li / M : %li\n",weight_package_i,weight_package_j,GPU_RAM_M);
									if (data->temp_pointer_1->nb_task_in_sub_list == 1 && data->temp_pointer_2->nb_task_in_sub_list == 1) {
										printf("I = 1 et J = 1\n"); 
									}
									else if (weight_package_i > GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
										printf("I > PU_RAM et J <= PU_RAM\n");
										common_data_last_package_i1_j = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 1, 0, false,GPU_RAM_M);					
										common_data_last_package_i2_j = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 2, 0, false,GPU_RAM_M);					
										printf("\ni1j = %d / i2j = %d\n",common_data_last_package_i1_j,common_data_last_package_i2_j);
										if (common_data_last_package_i1_j > common_data_last_package_i2_j) {
											printf("SWITCH PAQUET I\n");
											//~ interReverseLL(pointeur_paquet_1);
										}
										else { printf("Pas de switch\n"); }
									}
									else if (weight_package_i <= GPU_RAM_M && weight_package_j > GPU_RAM_M) {
										printf("I <= PU_RAM et J > PU_RAM\n");
										common_data_last_package_i_j1 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 0, 1, false,GPU_RAM_M);					
										common_data_last_package_i_j2 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 0, 2, false,GPU_RAM_M);					
										printf("\nij1 = %d / ij2 = %d\n",common_data_last_package_i_j1,common_data_last_package_i_j2);
										if (common_data_last_package_i_j2 > common_data_last_package_i_j1) {
											printf("SWITCH PAQUET J\n");
											//~ interReverseLL(pointeur_paquet_1);
										}
										else { printf("Pas de switch\n"); }
									}
									else {
										if (weight_package_i > GPU_RAM_M && weight_package_j > GPU_RAM_M) {
											printf("I > PU_RAM et J > PU_RAM\n");
											common_data_last_package_i1_j1 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 1, 1, false,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 1, 2, false,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 2, 1, false,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 2, 2, false,GPU_RAM_M);
										}
										else if (weight_package_i <= GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
											printf("I <= PU_RAM et J <= PU_RAM\n");
											printf("Tâches de I\n");
											temp_task_1 = starpu_task_list_begin(&data->temp_pointer_1->sub_list);
											printf("%p:",temp_task_1);
											printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));
											while (starpu_task_list_next(temp_task_1) != NULL) { temp_task_1 = starpu_task_list_next(temp_task_1); printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));}
											printf("Tâches de J\n");
											temp_task_1 = starpu_task_list_begin(&data->temp_pointer_2->sub_list);
											printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));
											while (starpu_task_list_next(temp_task_1) != NULL) { temp_task_1 = starpu_task_list_next(temp_task_1); printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));}
											
											//~ printf("I a %d taches et %d données\n",data->temp_pointer_1->nb_task_in_sub_list,data->temp_pointer_1->package_nb_data);
											//~ printf("J a %d taches et %d données\n",data->temp_pointer_2->nb_task_in_sub_list,data->temp_pointer_2->package_nb_data);
											//~ printf("Split de last ij de I = %d\n",data->temp_pointer_1->split_last_ij);
											//~ printf("Split de last ij de J = %d\n",data->temp_pointer_2->split_last_ij);
											common_data_last_package_i1_j1 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 1, 1, true,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 1, 2, true,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 2, 1, true,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(data->temp_pointer_1, data->temp_pointer_2, 2, 2, true,GPU_RAM_M);
										}
										else { printf("Erreur dans ordre U, aucun cas choisi\n"); exit(0); }
										printf("i1j1 = %d / i1j2 = %d / i2j1 = %d / i2j2 = %d\n",common_data_last_package_i1_j1,common_data_last_package_i1_j2,common_data_last_package_i2_j1,common_data_last_package_i2_j2);
										max_common_data_last_package = common_data_last_package_i2_j1;
										if (max_common_data_last_package < common_data_last_package_i1_j1) { max_common_data_last_package = common_data_last_package_i1_j1; }
										if (max_common_data_last_package < common_data_last_package_i1_j2) { max_common_data_last_package = common_data_last_package_i1_j2; }
										if (max_common_data_last_package < common_data_last_package_i2_j2) { max_common_data_last_package = common_data_last_package_i2_j2; }
										if (max_common_data_last_package == common_data_last_package_i2_j1) {
											printf("Pas de switch\n");
										}								
										else if (max_common_data_last_package == common_data_last_package_i1_j2) {
											printf("SWITCH PAQUET I ET J\n");	
											//~ HFP_reverse_sub_list(data->temp_pointer_1);															
											//~ HFP_reverse_sub_list(data->temp_pointer_2);															
										}
										else if (max_common_data_last_package == common_data_last_package_i2_j2) {
											printf("Tâches du paquet %d:\n",data->temp_pointer_2->index_package);
											for (temp_task_1  = starpu_task_list_begin(&data->temp_pointer_2->sub_list); temp_task_1 != starpu_task_list_end(&data->temp_pointer_2->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												printf("%p / ",temp_task_1); } 
											printf("SWITCH PAQUET J\n");
											data->temp_pointer_2 = HFP_reverse_sub_list(data->temp_pointer_2);	
											printf("Tâches du paquet %d:\n",data->temp_pointer_2->index_package);
											for (temp_task_1  = starpu_task_list_begin(&data->temp_pointer_2->sub_list); temp_task_1 != starpu_task_list_end(&data->temp_pointer_2->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												printf("%p /",temp_task_1); }								
										}
										else { /* max_common_data_last_package == common_data_last_package_i1_j1 */
											printf("SWITCH PAQUET I\n");
											//~ HFP_reverse_sub_list(data->temp_pointer_1);								
										}		
									}							
									printf("Fin de l'ordre U sans doublons\n");
								}
								
								data->temp_pointer_1->split_last_ij = data->temp_pointer_1->nb_task_in_sub_list;
								while (!starpu_task_list_empty(&data->temp_pointer_2->sub_list)) {
								starpu_task_list_push_back(&data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&data->temp_pointer_2->sub_list)); 
								data->temp_pointer_1->nb_task_in_sub_list ++; }
								i_bis = 0; j_bis = 0; tab_runner = 0;
								starpu_data_handle_t *temp_data_tab = malloc((data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data) * sizeof(data->temp_pointer_1->package_data[0]));
								while (i_bis < data->temp_pointer_1->package_nb_data && j_bis < data->temp_pointer_2->package_nb_data) {
									if (data->temp_pointer_1->package_data[i_bis] <= data->temp_pointer_2->package_data[j_bis]) {
										temp_data_tab[tab_runner] = data->temp_pointer_1->package_data[i_bis];
										i_bis++; }
									else {
										temp_data_tab[tab_runner] = data->temp_pointer_2->package_data[j_bis];
										j_bis++; }
									tab_runner++;
								}
								while (i_bis < data->temp_pointer_1->package_nb_data) { temp_data_tab[tab_runner] = data->temp_pointer_1->package_data[i_bis]; i_bis++; tab_runner++; }
								while (j_bis < data->temp_pointer_2->package_nb_data) { temp_data_tab[tab_runner] = data->temp_pointer_2->package_data[j_bis]; j_bis++; tab_runner++; }
								for (i_bis = 0; i_bis < (data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1]) {
										temp_data_tab[i_bis] = 0;
										nb_duplicate_data++; } }
								data->temp_pointer_1->package_data = malloc((data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
								j_bis = 0;
								for (i_bis = 0; i_bis < (data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] != 0) { data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++; } }
								data->temp_pointer_1->package_nb_data = data->temp_pointer_2->package_nb_data + data->temp_pointer_1->package_nb_data - nb_duplicate_data;
								data->temp_pointer_2->package_nb_data = 0;
								nb_duplicate_data = 0;
								data->temp_pointer_2->nb_task_in_sub_list = 0;
							temp_nb_min_task_packages--;
							if (temp_nb_min_task_packages > 1) {
								goto debut_while; 
							}
							else { j = nb_pop; i = nb_pop; }
							} }
							data->temp_pointer_2=data->temp_pointer_2->next;
						}
					}
					data->temp_pointer_1=data->temp_pointer_1->next; data->temp_pointer_2=data->first_link;
				}			
				
								
			
				
				data->temp_pointer_1 = data->first_link;
				data->temp_pointer_1 = HFP_delete_link(data);
				tab_runner = 0;
					/* Code to get the coordinates of each data in the order in wich tasks get out of pull_task */
					while (data->temp_pointer_1 != NULL) {
						//~ for (temp_task_1 = starpu_task_list_begin(&data->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
							//~ starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(temp_task_1,2),2,temp_tab_coordinates);
							//~ coordinate_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = number_tasks - data->temp_pointer_1->index_package - 1;
							//~ coordinate_order_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = tab_runner;
							//~ tab_runner++;	
							//~ temp_tab_coordinates[0] = 0; temp_tab_coordinates[1] = 0;
						//~ }			
						//~ temp_moyenne += data->temp_pointer_1->nb_task_in_sub_list;
						link_index++;
						data->temp_pointer_1 = data->temp_pointer_1->next;
					} 
			 
			/* Else we are using algorithm 3 */					
			if (link_index == 1) {  goto end_algo3; }
				
			for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
			/* Reset nb_pop for the matrix initialisation */
			nb_pop = link_index;
			/* If we have only one package we don't have to do more packages */			
			if (nb_pop == 1) {  packaging_impossible = 1; }
			
		} /* End of while (packaging_impossible == 0) { */
		/* We are in algorithm 3, we remove the size limit of a package */
		GPU_limit_switch = 0; goto algo3;
		
		end_algo3:
				
		data->temp_pointer_1 = data->first_link;	
		
		/* Code to printf everything */
		if (starpu_get_env_number_default("PRINTF",0) == 1) { 
			link_index = 0;
			long int total_weight = 0;
			printf("A la fin du regroupement des tâches utilisant l'algo %d on obtient : \n",data->ALGO_USED_READER);
			while (data->temp_pointer_1 != NULL) { link_index++; data->temp_pointer_1 = data->temp_pointer_1->next;				
				} data->temp_pointer_1 = data->first_link;
			printf("On a fais %d tour(s) de la boucle while et on a fais %d paquet(s)\n",nb_of_loop,link_index);
			printf("-----\n");
			link_index = 0;	
			while (data->temp_pointer_1 != NULL) {
				printf("Le paquet %d contient %d tâche(s) et %d données\n",link_index,data->temp_pointer_1->nb_task_in_sub_list,data->temp_pointer_1->package_nb_data);
				for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
					total_weight+= starpu_data_get_size(data->temp_pointer_1->package_data[i]);
				}
				printf("Le poids des données du paquet %d est : %li\n",link_index,total_weight);
				total_weight = 0;
				link_index++;
				data->temp_pointer_1 = data->temp_pointer_1->next;
				printf("-----\n");
			}
			data->temp_pointer_1 = data->first_link;
			temp_task_1  = starpu_task_list_begin(&data->temp_pointer_1->sub_list);
			data->temp_pointer_1 = data->first_link;
		}
		
		/* We pop the first task of the first package */
		task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list);
		}
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			/* We remove the first list of the first link of the linked list */
			if (task1 != NULL) { 
				/* Lines like this under and at the beggining of this function are for printing the tasks getting out */
				//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			}
			return task1;
	} /* Else of if ((data->temp_pointer_1->next == NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) { */
	if (!starpu_task_list_empty(&data->temp_pointer_1->sub_list)) {
		task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		return task1;
	}
	if ((data->temp_pointer_1->next != NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) {
		/* The list is empty and it's not the last one, so we go on the next link */
		data->temp_pointer_1 = data->temp_pointer_1->next;
		while (starpu_task_list_empty(&data->temp_pointer_1->sub_list)) { data->temp_pointer_1 = data->temp_pointer_1->next; }
			task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list); 
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task from starpu_task_list_empty(&data->temp_pointer_1->sub_list)\n",task1); }
			return task1;
	}		
}

static int HFP_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct HFP_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task); }
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

static int HFP_can_pull(struct starpu_sched_component * component)
{
	struct HFP_sched_data *data = component->data;
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_HFP_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	srandom(time(0)); /* For the random selection in ALGO 4 */
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "HFP");
	
	struct HFP_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	data->ALGO_USED_READER = starpu_get_env_number_default("ALGO_USED",1);
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
 
	my_data->next = NULL;
	data->temp_pointer_1 = my_data;
	
	component->data = data;
	component->push_task = HFP_push_task;
	component->pull_task = HFP_pull_task;
	component->can_push = HFP_can_push;
	component->can_pull = HFP_can_pull;

	return component;
}

static void initialize_HFP_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_HFP_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_HFP_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_HFP_policy =
{
	.init_sched = initialize_HFP_center_policy,
	.deinit_sched = deinitialize_HFP_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "HFP",
	.policy_description = "Affinity aware task ordering",
	.worker_type = STARPU_WORKER_LIST,
};
