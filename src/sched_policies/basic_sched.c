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

/* This code re-order the tasks in packages before sending them to a scheduler. 
 * The goal is to minimize the number of packages created.
 * The heuristic used is to regroup task that maximize the weight of the common data.
 * There are 4 different algorithms.
 */

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
#define ALGO_USED /* 1,2,3 or 4. Add ALGO_4_RANDOM=1 and ALGO_USED=4 to use the algorithm 4. Add ALGO_4_RANDOM=0 and ALGO_USED=4 or just ALGO_USED=4 for the algorithm 4' */
#define ALGO_4_RANDOM /* 0 or 1 */
#define PRINTF /* O or 1 */
#define HILBERT /* O or 1 */

/* Structure used to acces the struct my_list. There are also task's list */
struct basic_sched_data
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
	/* The task's list of the last state of the current package */
	struct starpu_task_list last_package_1; 
	struct starpu_task_list last_package_2; 
};

/* Empty a task's list. We use this for the lists last_package */
void empty_list(struct starpu_task_list *a)
{
	struct starpu_task *task = NULL;
	while (!starpu_task_list_empty(a)) {				
		task = starpu_task_list_pop_front(a);
	}
}

/* Put a link at the beginning of the linked list */
void insertion(struct basic_sched_data *a)
{
    struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
    new->next = a->temp_pointer_1;
    a->temp_pointer_1 = new;
}

/* Delete all the empty packages */
struct basic_sched_data* delete_link(struct basic_sched_data* a)
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

/* Give a color for each package. Written in the file Data_coordinates.txt */
static void rgb(int num, int *r, int *g, int *b)
{
    int *p[3] = {r, g, b};
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
int pointeurComparator ( const void * first, const void * second ) {
  return ( *(int*)first - *(int*)second );
}

/* Pushing the tasks */		
static int basic_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct basic_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *basic_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct basic_sched_data *data = component->data;
	
	/* Variables */
	/* Variables used to calculate, navigate through a loop or other things */
	int i = 0; int j = 0; int tab_runner = 0; int do_not_add_more = 0; int index_head_1 = 0; int index_head_2 = 0; int i_bis = 0; int j_bis = 0; double number_tasks = 0; int temp_number_task = 0; int random_value = 0;
	double mean_task_by_packages = 0; double temp_moyenne = 0; double temp_variance = 0; double temp_ecart_type = 0; long cursor_position = 0; int packing_time = 0; double moyenne = 0; double ecart_type = 0;
	int min_nb_task_in_sub_list = 0; int nb_min_task_packages = 0; int temp_nb_min_task_packages = 0; int *red = 0; int *green = 0; int *blue = 0; int temp_i_bis = 0;
	struct starpu_task *task1 = NULL; struct starpu_task *temp_task_1 = NULL; struct starpu_task *temp_task_2 = NULL; starpu_data_handle_t data_0_0_in_C = NULL;
	long int common_data_last_package_i1_j0 = 0; long int common_data_last_package_i1_j1 = 0;
	
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
		
	/* Here we calculate the size of the RAM of the GPU. We allow our packages to have half of this size */
	starpu_ssize_t GPU_RAM = 0;
	STARPU_ASSERT(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component));
	GPU_RAM = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))))/2;
	
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
			
			/* Putting all the coordinates of all the datas in a tabular */
			i = 0; int *tab_coordinates = malloc((nb_pop*3+1)*sizeof(tab_coordinates[0])); tab_coordinates[0] = nb_pop*3+1; i++;
			for (temp_task_1  = starpu_task_list_begin(&data->popped_task_list); temp_task_1 != starpu_task_list_end(&data->popped_task_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
					starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(temp_task_1,2),2,temp_tab_coordinates);
					if ((temp_tab_coordinates[0] == 0) && (temp_tab_coordinates[1] == 0)) { data_0_0_in_C = STARPU_TASK_GET_HANDLE(temp_task_1,2); }
			}
					
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
				qsort(data->temp_pointer_1->package_data,data->temp_pointer_1->package_nb_data,sizeof(data->temp_pointer_1->package_data[0]),pointeurComparator);
				/* Pushing the task and the number of the package in the package*/
				starpu_task_list_push_back(&data->temp_pointer_1->sub_list,temp_task_1);
				data->temp_pointer_1->index_package = link_index;
				/* Initialization of the lists last_packages */
				starpu_task_list_push_back(&data->temp_pointer_1->last_package_1,temp_task_1);
				starpu_task_list_push_back(&data->temp_pointer_1->last_package_2,temp_task_1);
				
				link_index++;
				data->temp_pointer_1->nb_task_in_sub_list ++;
				
				if(do_not_add_more != 0) { insertion(data); data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->temp_pointer_1->package_data[0])); }
				do_not_add_more--;
			}
			data->first_link = data->temp_pointer_1;				
			
			/* Code to print all the data of all the packages */
			if (starpu_get_env_number_default("PRINTF",0) == 1) {
				link_index = 0;
				printf("Initialement on a : \n");
				while (data->temp_pointer_1 != NULL) {
					for (i = 0; i < 3; i++) {
						printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->temp_pointer_1->package_data[i],temp_task_1  = starpu_task_list_begin(&data->temp_pointer_1->sub_list),link_index);
					}
					link_index++;
					data->temp_pointer_1 = data->temp_pointer_1->next;
				} printf("NULL\n");
				data->temp_pointer_1 = data->first_link;
			}
			
			data->temp_pointer_2 = data->first_link;
			index_head_2++;
			
			/* Output files */
			FILE * fcoordinate; /* Coordinates at each iteration */
			fcoordinate = fopen("Data_coordinates.txt", "w+");
			fprintf(fcoordinate,"\\begin{figure}[H]");
			FILE * fcoordinate_order; /* Order in wich the task go out */
			fcoordinate_order = fopen("Data_coordinates_order.txt", "w+");
			fprintf(fcoordinate_order,"\\begin{figure}[H]");
			FILE * variance_ecart_type;
			variance_ecart_type = fopen("variance_ecart_type.txt", "w+");
			FILE * mean_task_by_loop;
			mean_task_by_loop = fopen("mean_task_by_loop.txt", "w+");
			FILE * mean_ecart_type_finaux; /* Mean tasks by packages and standart deviation at the end of the execution only */
			mean_ecart_type_finaux = fopen("mean_ecart_type_finaux.txt", "a+");
			
			/* Matrix used to store all the common data weights between packages */
			int coordinate_visualization_matrix_size = sqrt(number_tasks);
			int coordinate_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			int coordinate_order_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) {
				for (j_bis = 0; j_bis < sqrt(number_tasks); j_bis++) {
					coordinate_visualization_matrix[j_bis][i_bis] = NULL;
					coordinate_order_visualization_matrix[j_bis][i_bis] = NULL;
				}
			}
			
			/* THE while loop. Stop when no more packaging are possible */
			while (packaging_impossible == 0) {
				/* algo 3's goto */
				algo3:
				nb_of_loop++;
				packaging_impossible = 1;
								
				/* Variables we need to reinitialize for a new iteration */
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link; index_head_1 = 0; index_head_2 = 1; link_index = 0; tab_runner = 0; nb_min_task_packages = 0;
				min_nb_task_in_sub_list = 0; nb_common_data = 0; weight_two_packages = 0; max_value_common_data_matrix = 0; long int matrice_donnees_commune[nb_pop][nb_pop];
				min_nb_task_in_sub_list = data->temp_pointer_1->nb_task_in_sub_list; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
				
				/* For algorithm 4 we need a symmetric matrix */
				if (data->ALGO_USED_READER == 4) { 
					/* First we get the number of packages that have the minimal number of tasks */
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list > data->temp_pointer_1->nb_task_in_sub_list) { min_nb_task_in_sub_list = data->temp_pointer_1->nb_task_in_sub_list; } }
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list == data->temp_pointer_1->nb_task_in_sub_list) { nb_min_task_packages++; } }
					if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list); }
					/* Then we create the common data matrix */
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						for (data->temp_pointer_2 = data->temp_pointer_1->next; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
							for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
								for (j = 0; j < data->temp_pointer_2->package_nb_data; j++) {
									if ((data->temp_pointer_1->package_data[i] == data->temp_pointer_2->package_data[j])) {
										matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->temp_pointer_1->package_data[i]);
										matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(data->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->temp_pointer_1->package_data[i]);
									} } } index_head_2++; } index_head_1++; index_head_2 = index_head_1 + 1; }
				}
				/* We are not in ALGO 4 we don't need a symetric matrix */
				else {
					/* Filling the common data matrix */
					for (data->temp_pointer_1 = data->first_link; data->temp_pointer_1 != NULL; data->temp_pointer_1 = data->temp_pointer_1->next) {
						for (data->temp_pointer_2 = data->temp_pointer_1->next; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
							for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
								for (j = 0; j < data->temp_pointer_2->package_nb_data; j++) {
									if ((data->temp_pointer_1->package_data[i] == data->temp_pointer_2->package_data[j])) {
										matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->temp_pointer_1->package_data[i]);
									}
								} 
							} index_head_2++; 
						} index_head_1++; index_head_2 = index_head_1 + 1;
					}
				}
				
				/* Code to print the common data matrix */
				if (starpu_get_env_number_default("PRINTF",0) == 1) { 
					printf("Common data matrix : \n"); for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { printf (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); }
				}
				
				/* Getting the number of package that have data in commons */
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						if (matrice_donnees_commune[i][j] != 0) { nb_common_data++; } } }
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Nb data en commun : %d\n",nb_common_data); }
				
				/* Getting back to the beginning of the linked list */
				data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link;
				
				long int tab_max_value_common_data_matrix [nb_min_task_packages]; /* Tab to store the max possible weight of each packages that have a minimal size */
				int indice_paquet_a_regrouper [nb_min_task_packages]; /* Store the index of the packages wich regroup with the maximum weight */
				if (data->ALGO_USED_READER == 4) {
					
					/* ALGO 4 */
					if (starpu_get_env_number_default("ALGO_4_RANDOM",0) == 1) {
					
				i_bis = 0; j_bis = 0; 
					temp_nb_min_task_packages = nb_min_task_packages;
				debut_while_1:
			
					data->temp_pointer_1 = data->first_link;
					data->temp_pointer_2 = data->first_link;
					max_value_common_data_matrix = 0;
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { /* If we are on a package of minimal size */
							for (data->temp_pointer_2 = data->first_link; data->temp_pointer_2 != NULL; data->temp_pointer_2 = data->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
										if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM)) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } } 
							data->temp_pointer_1=data->temp_pointer_1->next;
							j_bis = 0; }
					data->temp_pointer_1 = data->first_link;
					data->temp_pointer_2 = data->first_link;
					nb_grouping_available = 0; i_bis = 0; j_bis = 0;
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						temp_i_bis = i_bis;
						if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (j_bis = 0; j_bis < nb_pop; j_bis++) {
									
									weight_two_packages = 0;
									for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
										if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix == matrice_donnees_commune[temp_i_bis][j_bis]) && i_bis != j_bis && (weight_two_packages <= GPU_RAM)) { 
										nb_grouping_available++; i_bis = nb_pop; } 
							data->temp_pointer_2=data->temp_pointer_2->next; } } 
							data->temp_pointer_1=data->temp_pointer_1->next; data->temp_pointer_2=data->first_link; }
														
			if (nb_grouping_available != 0) {				
				int *tab_grouping_available = malloc(nb_grouping_available*sizeof(int)); for (i = 0; i < nb_grouping_available; i++) { tab_grouping_available[i] = 0; }
				data->temp_pointer_1 = data->first_link;
				data->temp_pointer_2 = data->first_link;
					tab_runner = 0; i_bis = 0; j_bis = 0;
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						temp_i_bis = i_bis;
						if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (j_bis = 0; j_bis < nb_pop; j_bis++) {						
									weight_two_packages = 0;
									for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
										if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix == matrice_donnees_commune[temp_i_bis][j_bis]) && i_bis != j_bis && (weight_two_packages <= GPU_RAM)) { 
										tab_grouping_available[tab_runner] = j_bis; tab_runner++; i_bis = nb_pop;} 
							data->temp_pointer_2=data->temp_pointer_2->next; } } 
							data->temp_pointer_1=data->temp_pointer_1->next; data->temp_pointer_2=data->first_link; }
							
							
				random_value = random()%nb_grouping_available;

				data->temp_pointer_1 = data->first_link;
				data->temp_pointer_2 = data->first_link;		
				i_bis = 0; j_bis = 0; i = 0; j = 0;
				
				
				for (i = 0; i < tab_grouping_available[random_value]; i++) {
					data->temp_pointer_1 = data->temp_pointer_1->next;
				}
						for (j = 0; j < nb_pop; j++) {
							if (data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list && max_value_common_data_matrix == matrice_donnees_commune[i][j] && i != j) {
								//Merge
								packaging_impossible = 0;
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
								
								if (data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list) { temp_nb_min_task_packages--; }
								
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;}
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;}
								nb_common_data--;
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
								goto debut_while_1; 
							}
							else { j = nb_pop; i = nb_pop; }					
					}
				data->temp_pointer_2 = data->temp_pointer_2->next; }		
							}
														 														
			} else {
					/* ALGO 4' */
					i_bis = 0; j_bis = 0; 
					temp_nb_min_task_packages = nb_min_task_packages;
				debut_while:
					data->temp_pointer_1 = data->first_link;
					data->temp_pointer_2 = data->first_link;
					max_value_common_data_matrix = 0;
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
									if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM)) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							data->temp_pointer_1=data->temp_pointer_1->next;
							j_bis = 0; }

				data->temp_pointer_1 = data->first_link;
				data->temp_pointer_2 = data->first_link;		
				i_bis = 0; j_bis = 0; i = 0; j = 0;
				for (i = 0; i < nb_pop; i++) {
					if (data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) {
						for (j = 0; j < nb_pop; j++) {
							weight_two_packages = 0;
							for (i_bis = 0; i_bis < data->temp_pointer_1->package_nb_data; i_bis++) { weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i_bis]); }
							for (i_bis = 0; i_bis < data->temp_pointer_2->package_nb_data; i_bis++) { bool_data_common = 0;
								for (j_bis = 0; j_bis < data->temp_pointer_1->package_nb_data; j_bis++) { if (data->temp_pointer_2->package_data[i_bis] == data->temp_pointer_1->package_data[j_bis]) { bool_data_common = 1; } }
								if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i_bis]); } }							
							//boucler sur le 0 et reinit i et j du coup pour les prochains. Interdire celui qu'on vient de faire aussi	
							if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j && weight_two_packages <= GPU_RAM) {
								//Merge
								packaging_impossible = 0;
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
								
								if (data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list) { temp_nb_min_task_packages--; }
								
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;}
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;}
								nb_common_data--;
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
							}
							data->temp_pointer_2=data->temp_pointer_2->next;
						}
					}
					data->temp_pointer_1=data->temp_pointer_1->next; data->temp_pointer_2=data->first_link;
				}			
				}
				}
				/* ALGO 1, 2 and 3 */
				else {
					if (GPU_limit_switch == 1) {
					/* Getting W_max. W_max get the max common data only if the merge of the two packages without the duplicates data would weight less than GPU_RAM */
					for (i_bis =0; i_bis < nb_pop; i_bis++) { 
						data->temp_pointer_2 = data->temp_pointer_1;
						data->temp_pointer_2 = data->temp_pointer_2->next;
						for (j_bis = i_bis+1; j_bis < nb_pop; j_bis++) {
							weight_two_packages = 0;
							for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
								weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i]);
							}
							for (i = 0; i < data->temp_pointer_2->package_nb_data; i++) {
								bool_data_common = 0;
								for (j = 0; j < data->temp_pointer_1->package_nb_data; j++) {
									if (data->temp_pointer_2->package_data[i] == data->temp_pointer_1->package_data[j])
									{
										bool_data_common = 1;
									}
								}
								if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i]); }
							}

							if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM)) { 
								max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
							} weight_two_packages = 0;
							data->temp_pointer_2 = data->temp_pointer_2->next;
						} data->temp_pointer_1 = data->temp_pointer_1->next;
					}
					/* Getting back to the beginning of the linked list */
					data->temp_pointer_1 = data->first_link; data->temp_pointer_2 = data->first_link;
				}
				/* Else, we are using algo 3, so we don't check the max weight */
				else {
					for (i_bis =0; i_bis < nb_pop; i_bis++) { 
						for (j_bis = i_bis+1; j_bis < nb_pop; j_bis++) {
							if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { 
								max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
							}
						}
					}
				}
				
				/* Merge of the packages and verification that the weight would be inferior to GPU_MAX */
				for (i = 0; i < nb_pop; i++) {
					data->temp_pointer_2 = data->temp_pointer_1;
					data->temp_pointer_2 = data->temp_pointer_2->next;
					for (j = i + 1; j< nb_pop; j++) {
						
						weight_two_packages = 0;
						for (i_bis = 0; i_bis < data->temp_pointer_1->package_nb_data; i_bis++) {
							weight_two_packages += starpu_data_get_size(data->temp_pointer_1->package_data[i_bis]);
						}
						for (i_bis = 0; i_bis < data->temp_pointer_2->package_nb_data; i_bis++) {
							bool_data_common = 0;
							for (j_bis = 0; j_bis < data->temp_pointer_1->package_nb_data; j_bis++) {
								if (data->temp_pointer_2->package_data[i_bis] == data->temp_pointer_1->package_data[j_bis])
								{
									bool_data_common = 1;
								}
							}
							if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->temp_pointer_2->package_data[i_bis]); }
						}
												
						if ((((matrice_donnees_commune[i][j] == max_value_common_data_matrix) && (max_value_common_data_matrix != 0)) || (GPU_limit_switch == 0)) && data->ALGO_USED_READER != 4) 
						{
							
							if ( (weight_two_packages > GPU_RAM) && (GPU_limit_switch == 1) ) { 
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On dépasse GPU_RAM!\n"); }
							}
							else {
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
							packaging_impossible = 0;
							
							/* Forbid i and j to do merge in the remaining of this iteration */
							for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; }
							for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; }
							nb_common_data--;
							
							if (starpu_get_env_number_default("HILBERT",0) == 1) {
								for (temp_task_1 = starpu_task_list_begin(&data->temp_pointer_1->last_package_2); temp_task_1 != starpu_task_list_end(&data->temp_pointer_1->last_package_2); temp_task_1 = starpu_task_list_next(temp_task_1)) {
									for (temp_task_2 = starpu_task_list_begin(&data->temp_pointer_2->last_package_1); temp_task_2 != starpu_task_list_end(&data->temp_pointer_2->last_package_1); temp_task_2 = starpu_task_list_next(temp_task_2)) {
										for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++) {
											for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++) {
												if (STARPU_TASK_GET_HANDLE(temp_task_1,i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2,j_bis)) {
													//C'est le poids qu'il faut faire ici
													common_data_last_package_i1_j0++;
												}
											}
										}
									}
									for (temp_task_2 = starpu_task_list_begin(&data->temp_pointer_2->last_package_2); temp_task_2 != starpu_task_list_end(&data->temp_pointer_2->last_package_2); temp_task_2 = starpu_task_list_next(temp_task_2)) {
										for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++) {
											for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++) {
												if (STARPU_TASK_GET_HANDLE(temp_task_1,i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2,j_bis)) {
													//C'est le poids qu'il faut faire ici
													common_data_last_package_i1_j1++;
												}
											}
										}
									}
								}
								//~ empty_list(&data->temp_pointer_1->last_package_1);
								//~ empty_list(&data->temp_pointer_1->last_package_2);
								printf("Il y a pour j0 : %d et pour j1 : %d\n",common_data_last_package_i1_j0,common_data_last_package_i1_j1);
								if (common_data_last_package_i1_j0 < common_data_last_package_i1_j1) {
									printf("Il faut switch!\n");
								}
							}
							
							/* Merging the tasks's list */
							while (!starpu_task_list_empty(&data->temp_pointer_2->sub_list)) { 
								starpu_task_list_push_back(&data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&data->temp_pointer_2->sub_list)); 
								data->temp_pointer_1->nb_task_in_sub_list ++;
							}

								i_bis = 0; j_bis = 0;
								tab_runner = 0;
								starpu_data_handle_t *temp_data_tab = malloc((data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data) * sizeof(data->temp_pointer_1->package_data[0]));
								/* Merge the two tabs containing data */
								while (i_bis < data->temp_pointer_1->package_nb_data && j_bis < data->temp_pointer_2->package_nb_data) {
									if (data->temp_pointer_1->package_data[i_bis] <= data->temp_pointer_2->package_data[j_bis]) {
										temp_data_tab[tab_runner] = data->temp_pointer_1->package_data[i_bis];
										i_bis++;
									}
									else
									{
										temp_data_tab[tab_runner] = data->temp_pointer_2->package_data[j_bis];
										j_bis++;
									}
									tab_runner++;
								}
								/* Copy the remaining data(s) of the tabs (if there are any) */
								while (i_bis < data->temp_pointer_1->package_nb_data) { temp_data_tab[tab_runner] = data->temp_pointer_1->package_data[i_bis]; i_bis++; tab_runner++; }
								while (j_bis < data->temp_pointer_2->package_nb_data) { temp_data_tab[tab_runner] = data->temp_pointer_2->package_data[j_bis]; j_bis++; tab_runner++; }
								
								/* We remove the duplicate data(s) */
								for (i_bis = 0; i_bis < (data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1]) {
										temp_data_tab[i_bis] = 0;
										nb_duplicate_data++;
									}
								}
								/* Then we put the temp_tab in temp_pointer_1 */
								data->temp_pointer_1->package_data = malloc((data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
								j_bis = 0;
								for (i_bis = 0; i_bis < (data->temp_pointer_1->package_nb_data + data->temp_pointer_2->package_nb_data); i_bis++)
								{
									if (temp_data_tab[i_bis] != 0) { data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++; }
								}
								
								/* We update the number of data of each package. It's important because we use delete_link(data) to delete all the packages with 0 data */
								data->temp_pointer_1->package_nb_data = data->temp_pointer_2->package_nb_data + data->temp_pointer_1->package_nb_data - nb_duplicate_data;
								data->temp_pointer_2->package_nb_data = 0;
								nb_duplicate_data = 0;
								data->temp_pointer_2->nb_task_in_sub_list = 0;
								
								if (data->ALGO_USED_READER == 2) { 
									goto algo_2; 
								}
						} }
						data->temp_pointer_2 = data->temp_pointer_2->next;
					} 
					if (nb_common_data > 1) {
						data->temp_pointer_1 = data->temp_pointer_1->next;
					} 
				}
				/* End of else ALGO_USED=4 */
				}
				
				/* goto for algo 2 */
				algo_2:
				
				data->temp_pointer_1 = data->first_link;
				data->temp_pointer_1 = delete_link(data);
				tab_runner = 0;
				if (data->ALGO_USED_READER != 3) {
					if (packaging_impossible != 1) {
					while (data->temp_pointer_1 != NULL) {
						/* Code to print the coordinates and the data of each package iteration by iteration */
						cursor_position = ftell(fcoordinate);
						for (i = 0; i < data->temp_pointer_1->package_nb_data; i++) {
							starpu_data_get_coordinates_array(data->temp_pointer_1->package_data[i],2,temp_tab_coordinates);
							if (((temp_tab_coordinates[0]) != 0) || ((temp_tab_coordinates[1]) !=0 ) || ((data_0_0_in_C == data->temp_pointer_1->package_data[i])))  {
								coordinate_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = number_tasks - data->temp_pointer_1->index_package - 1;
								coordinate_order_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = tab_runner;
								tab_runner++;						
								temp_tab_coordinates[0] = 0; temp_tab_coordinates[1] = 0;
							}
						}
						for (temp_task_1  = starpu_task_list_begin(&data->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
							if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("La tâche %p est dans le paquet numéro %d\n",temp_task_1,link_index); }
							temp_number_task ++;
						}
						temp_moyenne += data->temp_pointer_1->nb_task_in_sub_list;
						temp_number_task = 0;
						link_index++;
						data->temp_pointer_1 = data->temp_pointer_1->next;
					} 
					
					//Code to write in a file the coordinates ---------------------------------------------------------------------------
					fprintf(fcoordinate,"\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{ c | c | c | c| c| c| c| c| c| c}"); 
					fprintf(fcoordinate_order,"\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{ c | c | c | c| c| c| c| c|c|c}"); 
					fprintf(fcoordinate,"\n");
					fprintf(fcoordinate_order,"\n");
					for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) { 
						for (j_bis = 0; j_bis < sqrt(number_tasks) - 1; j_bis++) {	
							/* Code to color the tabs in Data_coordinates.txt */
							if (coordinate_visualization_matrix[j_bis][i_bis] == 0) { red = 255; green = 255; blue = 255; }
							else if (coordinate_visualization_matrix[j_bis][i_bis] == 6) { red = 70; green = 130; blue = 180; }
							else { 
								rgb(coordinate_visualization_matrix[j_bis][i_bis], &red, &green, &blue); 
							}
							fprintf(fcoordinate,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, coordinate_visualization_matrix[j_bis][i_bis]);
							fprintf(fcoordinate_order,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, coordinate_order_visualization_matrix[j_bis][i_bis]);	
						}
						/* The last tab is out of the loop because we don't printf "&" */
						if (coordinate_visualization_matrix[j_bis][i_bis] == 0) { red = 255; green = 255; blue = 255; }
						else if (coordinate_visualization_matrix[j_bis][i_bis] == 6) { red = 70; green = 130; blue = 180; }
						else { 
							rgb(coordinate_visualization_matrix[j_bis][i_bis], &red, &green, &blue); 
						}
						fprintf(fcoordinate,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,coordinate_visualization_matrix[j_bis][i_bis]); 
						fprintf(fcoordinate_order,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,coordinate_order_visualization_matrix[j_bis][i_bis]); 
						fprintf(fcoordinate," \\\\"); fprintf(fcoordinate,"\\hline");
						fprintf(fcoordinate_order," \\\\"); fprintf(fcoordinate_order,"\\hline");
					}
					if (nb_of_loop > 1 && nb_of_loop%2 == 0) { 
						fprintf(fcoordinate, "\\end{tabular} \\caption{Itération %d} \\label{fig:sub-third} \\end{subfigure} \\\\",nb_of_loop); 
						fprintf(fcoordinate_order, "\\end{tabular} \\caption{Itération %d} \\label{fig:sub-third} \\end{subfigure} \\\\",nb_of_loop); 
					}
					else { 
						fprintf(fcoordinate, "\\end{tabular} \\caption{Itération %d} \\label{fig:sub-third} \\end{subfigure}",nb_of_loop); 
						fprintf(fcoordinate_order, "\\end{tabular} \\caption{Itération %d} \\label{fig:sub-third} \\end{subfigure}",nb_of_loop); 
					}
					fprintf(fcoordinate,"\n");
					fprintf(fcoordinate_order,"\n");
					for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) {
						for (j_bis = 0; j_bis < sqrt(number_tasks); j_bis++) {
							coordinate_visualization_matrix[j_bis][i_bis] = NULL;
						}
					}
					// ----------------------------------------------------------------------------------------------------------------
					
					/* Code to printf and fprintf the mean number of tasks in each package and the standart deviation */
					temp_moyenne = temp_moyenne/(link_index);
					moyenne = temp_moyenne;
					data->temp_pointer_1 = data->first_link; temp_variance = 0;
					while (data->temp_pointer_1 != NULL) {
						temp_variance += (data->temp_pointer_1->nb_task_in_sub_list - moyenne)*(data->temp_pointer_1->nb_task_in_sub_list - moyenne);
						data->temp_pointer_1 = data->temp_pointer_1->next;
					}
					temp_variance = temp_variance/link_index;
					data->temp_pointer_1 = data->first_link;					
					temp_ecart_type = sqrt(temp_variance);
					ecart_type = temp_ecart_type;					
					if (starpu_get_env_number_default("PRINTF",0) == 1) { 
						printf("La variance du nb de taches par paquets est %f\n",temp_variance);
						printf("L'ecart type du nb de taches par paquets est %f\n",temp_ecart_type);
						printf("A la fin du tour numéro %d du while on a %d paquets\n\n",nb_of_loop,link_index);
						printf("Fin du tour numéro %d du while!\n\n",nb_of_loop);
					}
					
					temp_moyenne = 0; temp_variance = 0; temp_ecart_type = 0;				
					fprintf(mean_task_by_loop,"%d	%d	%f	%f	\n",nb_of_loop,link_index,moyenne,ecart_type);
				}
				}
				/* Else we are using algorithm 3 */
				else { while (data->temp_pointer_1 != NULL) { link_index++; data->temp_pointer_1 = data->temp_pointer_1->next; } }
									
				
			for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
			//Reset de nb_pop!
			nb_pop = link_index;
			/* If we have only one package we don't have to do more packages */			
			if (nb_pop == 1) { break; }

		} // Fin du while (packaging_impossible == 0) {
		fprintf(mean_ecart_type_finaux,"%f	%f\n",moyenne,ecart_type);
		if (data->ALGO_USED_READER != 3) { 
			fprintf(fcoordinate,"\\caption{ALGO 1 / BW 500 / CUDA MEM 500 / RANDOM TASK ORDER 0 / MATRICE 10x10} \\label{fig:fig} \\end{figure}");
			fprintf(fcoordinate_order,"\\caption{ALGO 1 / BW 500 / CUDA MEM 500 / RANDOM TASK ORDER 0 / MATRICE 10x10} \\label{fig:fig} \\end{figure}");
			fclose(fcoordinate);
			fclose(fcoordinate_order);
			fclose(variance_ecart_type);
			fclose(mean_task_by_loop);
			fclose(mean_ecart_type_finaux);
		}
		
		/* We are in algorithm 3, we remove the size limit of a package */
		if ((data->ALGO_USED_READER == 3) && (link_index > 1)) { GPU_limit_switch = 0; goto algo3; }	
		
		data->temp_pointer_1 = data->first_link;	
		
		/* Code to printf everything */
		if (starpu_get_env_number_default("PRINTF",0) == 1) { 
			link_index = 0;
			long int total_weight = 0;
			double task_duration_info = 0;
			double bandwith_info = starpu_transfer_bandwidth(STARPU_MAIN_RAM, starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx)));
			printf("A la fin du regroupement des tâches utilisant l'algo %d on obtient : \n",data->ALGO_USED_READER);
			while (data->temp_pointer_1 != NULL) { link_index++; data->temp_pointer_1 = data->temp_pointer_1->next; } data->temp_pointer_1 = data->first_link;
			printf("On a fais %d tour(s) de la boucle while et on a fais %d paquet(s)\n",nb_of_loop,link_index);
			printf("-----\n");
			link_index = 0;	
			while (data->temp_pointer_1 != NULL) {
				printf("Le paquet %d contient %d tâche(s) et %d données\n",link_index,data->temp_pointer_1->nb_task_in_sub_list,data->temp_pointer_1->package_nb_data);
				for (temp_task_1  = starpu_task_list_begin(&data->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
					printf("%p\n",temp_task_1);
				}
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
			task_duration_info = starpu_task_worker_expected_length(temp_task_1, 0, component->tree->sched_ctx_id,0);
			data->temp_pointer_1 = data->first_link;
			packing_time = clock();
			printf("Temps d'empaquetage = %d ms\n", packing_time);
		}
		
		/* We pop the first task of the first package */
		task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list);
		}
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			/* We remove the first list of the first link of the linked list */
			if (task1 != NULL) { 
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			}
			return task1;
	} /* Else of if ((data->temp_pointer_1->next == NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) { */
	if (!starpu_task_list_empty(&data->temp_pointer_1->sub_list)) {
		task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		return task1;
	}
	if ((data->temp_pointer_1->next != NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) {
		/* The list is empty and it's not the last one, so we go on the next link */
		data->temp_pointer_1 = data->temp_pointer_1->next;
		while (starpu_task_list_empty(&data->temp_pointer_1->sub_list)) { data->temp_pointer_1 = data->temp_pointer_1->next; }
			task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list); 
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task from starpu_task_list_empty(&data->temp_pointer_1->sub_list)\n",task1); }
			return task1;
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
		if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task); }
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
	return didwork || starpu_sched_component_can_push(component, to);
}

static int basic_can_pull(struct starpu_sched_component * component)
{
	struct basic_sched_data *data = component->data;
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_basic_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	srandom(time(0)); /* For the random selection in ALGO 4 */
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "basic");
	
	struct basic_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	data->ALGO_USED_READER = starpu_get_env_number_default("ALGO_USED",1);
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->last_package_1);
	starpu_task_list_init(&my_data->last_package_2);
 
	my_data->next = NULL;
	data->temp_pointer_1 = my_data;
	
	component->data = data;
	component->push_task = basic_push_task;
	component->pull_task = basic_pull_task;
	component->can_push = basic_can_push;
	component->can_pull = basic_can_pull;

	return component;
}

static void initialize_basic_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_basic_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
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
