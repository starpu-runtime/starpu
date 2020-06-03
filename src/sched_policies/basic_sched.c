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
#define MEMORY_AFFINITY
#define ALGO_USED
//la variable d environemment c est la je crois avec l'option

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
	int ALGO_USED_READER;
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
	long int max_weight_common_data;
	int package_nb_data;
	int nb_task_in_sub_list;
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
}

// Delete all the link where package_nb_data equals 0
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
	
int pointeurComparator ( const void * first, const void * second ) {
  return ( *(int*)first - *(int*)second );
}
		
static int basic_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct basic_sched_data *data = component->data;
		//~ fprintf(stderr, "Pushing task %p\n", task);

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
	int i = 0; int j = 0; int nb_pop = 0; int temp_nb_pop = 0; int tab_runner = 0; int max_donnees_commune = 0; int k = 0; int nb_data_commun = 0; int nb_tasks_in_linked_list = 0;
	int do_not_add_more = 0; int link_index = 0; int nb_duplicate_data = 0; long int weight_two_packages; long int max_value_common_data_matrix = 0; int index_head_1 = 0; int index_head_2 = 0;
	int i_bis = 0; int j_bis = 0; int nb_of_loop = 0; 
	int packaging_impossible = 0; //0 = false / 1 = true
	int temp_tab_coordinates[2];
	int bool_data_common = 0;
	int GPU_limit_switch = 1; //On 1 it means we use the limit
	double number_tasks = 0;
	int temp_number_task = 0;
	double mean_task_by_packages = 0;
	double temp_moyenne = 0;
	double temp_variance = 0;
	double temp_ecart_type = 0;
	long cursor_position = 0;
	int packing_time = 0;
	double moyenne = 0; double ecart_type = 0;
	int min_nb_task_in_sub_list = 0;
	int nb_min_task_packages = 0;
	
	// Fichiers de sortie -------------------------
	//~ FILE * variance_ecart_type;
	//~ variance_ecart_type = fopen("variance_ecart_type.txt", "a+");
	//~ FILE * Nb_package_by_loop;
	//~ Nb_package_by_loop = fopen("Nb_package_by_loop.txt", "a+");
	//~ FILE * Mean_task_by_loop;
	//~ Mean_task_by_loop = fopen("Mean_task_by_loop.txt", "a+");
	//~ FILE * fcoordinate;
	//~ fcoordinate = fopen("Data_coordinates.txt", "w+");
	//---------------------------------------------
	
	//Here we calculate the size of the RAM of the GPU. We allow our packages to have half of this size
	starpu_ssize_t GPU_RAM = 0;
	STARPU_ASSERT(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component));
	GPU_RAM = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))))/2;
	//~ printf("GPU_RAM/2 : %d\n",GPU_RAM);
	//~ GPU_RAM/2 = 262.144.000 pour cuda mem 500
	//~ GPU_RAM/2 = 131.072.000 pour cuda mem 250
	
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_3 = NULL;
	struct starpu_task *temp_task_4 = NULL;
	starpu_data_handle_t data_0_0_in_C = NULL;

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
				//~ printf("%p\n",task1);
				nb_pop++;
				starpu_task_list_push_back(&data->tache_pop,task1);
			} 
			
			number_tasks = nb_pop;
			i = 0;
			int *tab_coordinates = malloc((nb_pop*3+1)*sizeof(tab_coordinates[0]));
			tab_coordinates[0] = nb_pop*3+1; i++;
			for (temp_task_3  = starpu_task_list_begin(&data->tache_pop); temp_task_3 != starpu_task_list_end(&data->tache_pop); temp_task_3  = starpu_task_list_next(temp_task_3)) {
					//Seul l'indice 2 du handle correspond a la matrice C
					starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(temp_task_3,2),2,temp_tab_coordinates);
					if ((temp_tab_coordinates[0] == 0) && (temp_tab_coordinates[1] == 0)) { data_0_0_in_C = STARPU_TASK_GET_HANDLE(temp_task_3,2); }
					//~ printf("Les coordonnées de la donnée %p de la tâche %p sont : x = %d / y = %d\n",STARPU_TASK_GET_HANDLE(temp_task_3,2),temp_task_3,temp_tab_coordinates[0],temp_tab_coordinates[1]);
					//~ tab_coordinates[i] = STARPU_TASK_GET_HANDLE(temp_task_3,2); i++; tab_coordinates[i] = temp_tab_coordinates[0]; i++; tab_coordinates[i] = temp_tab_coordinates[1]; i++;
			}
			
			//~ for (i = 1; i < nb_pop*3+1; i+=3) { printf("dans le tab à %p on a : %d et %d\n", tab_coordinates[i],tab_coordinates[i+1],tab_coordinates[i+2]); }
			
			//~ printf("%d task(s) have been pulled\n",nb_pop);
			//~ printf("Avec l'exemple de base on a %d données initialement\n",nb_pop*3);
			
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
				//We sort our data tab
				qsort(data->head->package_data,data->head->package_nb_data,sizeof(data->head->package_data[0]),pointeurComparator);
				starpu_task_list_push_back(&data->head->sub_list,temp_task_3);
				data->head->nb_task_in_sub_list ++;
				
				if(do_not_add_more != 0) { insertion(data); data->head->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_3)*sizeof(data->head->package_data[0])); }
				do_not_add_more--;
			}
			data->first_link = data->head;
			
			//Code to print all the data of all the packages ---------------------------------------------
			//~ printf("Initialement on a : \n");
			//~ while (data->head != NULL) {
				//~ for (i = 0; i < 3; i++) {
					//~ printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->head->package_data[i],temp_task_3  = starpu_task_list_begin(&data->head->sub_list),link_index);
				//~ }
				//~ link_index++;
				//~ data->head = data->head->next;
			//~ } printf("NULL\n");
			//~ data->head = data->first_link;
			//--------------------------------------------------------------------------------------------
			
			

			data->head_2 = data->first_link;
			index_head_2++;
			
//~ while (link_index != 1) {
			
			//Fichier pour visualisation des coordonnées
			FILE * fcoordinate;
			fcoordinate = fopen("Data_coordinates.txt", "w+");
			FILE * variance_ecart_type;
			variance_ecart_type = fopen("variance_ecart_type.txt", "a+");
			//Matrice pour visualisation des coordonnées
			int coordinate_visualization_matrix_size = sqrt(number_tasks);
			int coordinate_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) {
					for (j_bis = 0; j_bis < sqrt(number_tasks); j_bis++) {
						coordinate_visualization_matrix[j_bis][i_bis] = NULL;
					}
				}
			
			while (packaging_impossible == 0) {
				algo3:
				//~ printf("max value comme data vaut au debut %d\n",max_value_common_data_matrix);
				nb_of_loop++;
				packaging_impossible = 1;
			
			//Reinit indispendable
			data->head = data->first_link;
			data->head_2 = data->first_link;
			index_head_1 = 0;
			index_head_2 = 1;
			link_index = 0;
			tab_runner = 0;
			nb_min_task_packages = 0;
			min_nb_task_in_sub_list = 0;
			nb_data_commun = 0;
			weight_two_packages = 0;
			max_value_common_data_matrix = 0;
			long int matrice_donnees_commune[nb_pop][nb_pop];
			min_nb_task_in_sub_list = data->head->nb_task_in_sub_list;
			for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
			
			//ALGO 4 on fais une matrice symétrique
			if (data->ALGO_USED_READER == 4) { 
				//Je récupère le nombre de tâche minimal d'un paquet
				for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
					if (min_nb_task_in_sub_list < data->head->nb_task_in_sub_list) { min_nb_task_in_sub_list = data->head->nb_task_in_sub_list; } }
				for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
					if (min_nb_task_in_sub_list == data->head->nb_task_in_sub_list) { nb_min_task_packages++; } }
				printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list);
				for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
					for (data->head_2 = data->head->next; data->head_2 != NULL; data->head_2 = data->head_2->next) {
						for (i = 0; i < data->head->package_nb_data; i++) {
							for (j = 0; j < data->head_2->package_nb_data; j++) {
								if ((data->head->package_data[i] == data->head_2->package_data[j])) {
									matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->head_2->package_data[j]) + starpu_data_get_size(data->head->package_data[i]);
									matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(data->head_2->package_data[j]) + starpu_data_get_size(data->head->package_data[i]);
								} } } index_head_2++; } index_head_1++; index_head_2 = index_head_1 + 1; }
			}
			//On est pas dans le cas de l'algo 4, pas besoin de faire une matrice symétrique
			else {
				//Filling the common data matrix
				for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
					for (data->head_2 = data->head->next; data->head_2 != NULL; data->head_2 = data->head_2->next) {
						for (i = 0; i < data->head->package_nb_data; i++) {
							for (j = 0; j < data->head_2->package_nb_data; j++) {
								if ((data->head->package_data[i] == data->head_2->package_data[j])) {
									matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->head_2->package_data[j]) + starpu_data_get_size(data->head->package_data[i]);
								}
							} 
						} index_head_2++; 
					} index_head_1++; index_head_2 = index_head_1 + 1;
				}
			}
			
			//Code to print the common data matrix  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			printf("Common data matrix : \n"); for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { printf (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); }
			//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			
			//Getting the number of package that have data in commons
			for (i = 0; i < nb_pop; i++) {
				for (j = 0; j < nb_pop; j++) {
					if (matrice_donnees_commune[i][j] != 0) { nb_data_commun++; } } }
			//~ printf("Nb data en commun : %d\n",nb_data_commun);
			
			//Getting back to the beginning of the linked list
			data->head = data->first_link;
			data->head_2 = data->first_link;
			
			//ALGO 4
			long int tab_max_value_common_data_matrix [nb_min_task_packages];
			if (data->ALGO_USED_READER == 4) {
				i_bis = 0; j_bis = 0; 
				for (i = 0; i < nb_min_task_packages; i++) { tab_max_value_common_data_matrix[i] = 0; }
				for (data->head = data->first_link; data->head != NULL; data->head = data->head->next) {
					if (data->head->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
						for (data->head_2 = data->first_link; data->head_2 != NULL; data->head_2 = data->head_2->next) {
							//a tester ca
							if (i_bis != j_bis) {
								weight_two_packages = 0;
								for (i = 0; i < data->head->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->head->package_data[i]); }
								for (i = 0; i < data->head_2->package_nb_data; i++) {
									bool_data_common = 0;
									for (j = 0; j < data->head->package_nb_data; j++) {
									if (data->head_2->package_data[i] == data->head->package_data[j]) { bool_data_common = 1; } }
									if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->head_2->package_data[i]); } }
								if((tab_max_value_common_data_matrix[tab_runner] < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM)) { 
									tab_max_value_common_data_matrix[tab_runner] = matrice_donnees_commune[i_bis][j_bis]; } weight_two_packages = 0;
						} j_bis++; }} tab_runner++; i_bis++; j_bis = 0; } 
			qsort(tab_max_value_common_data_matrix,nb_min_task_packages,sizeof(tab_max_value_common_data_matrix[0]),pointeurComparator);
			for (i = 0; i < nb_min_task_packages; i++) { printf("%d de tab_max_value_common_data_matrix = %li\n",i,tab_max_value_common_data_matrix[i]); }
			data->head = data->first_link;
			data->head_2 = data->first_link;
			}
			else {
				if (GPU_limit_switch == 1) {
				//Getting W_max. W_max get the max common data ONLY IF THE MERGE OF THE TWO PACKAGES WITHOUT THE DUPLICATE WOULD BE INFERIOR TO GPU_RAM
				for (i_bis =0; i_bis < nb_pop; i_bis++) { 
					data->head_2 = data->head;
					data->head_2 = data->head_2->next;
					for (j_bis = i_bis+1; j_bis < nb_pop; j_bis++) {
						weight_two_packages = 0;
						//On somme d'abord les poids de tout le paquet 1
						for (i = 0; i < data->head->package_nb_data; i++) {
							weight_two_packages += starpu_data_get_size(data->head->package_data[i]);
						}
						//ensuite on somme le paquet 2 mais uniquement les données qui ne sont pas dans le paquet 1
						for (i = 0; i < data->head_2->package_nb_data; i++) {
							bool_data_common = 0;
							for (j = 0; j < data->head->package_nb_data; j++) {
								if (data->head_2->package_data[i] == data->head->package_data[j])
								{
									bool_data_common = 1;
								}
							}
							if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->head_2->package_data[i]); }
						}
									
						//~ for (i = 0; i < data->head->package_nb_data; i++) {
							//~ weight_two_packages += starpu_data_get_size(data->head->package_data[i]);
						//~ }
						//~ for (i = 0; i < data->head_2->package_nb_data; i++) {
							//~ weight_two_packages += starpu_data_get_size(data->head_2->package_data[i]);
						//~ } 
						//~ printf("Le poids des deux paquets avant le test %d et %d serait de : %li\n",i_bis,j_bis,weight_two_packages);
						if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM)) { 
							max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
							//~ printf("MAJ de max value : %d, avec un poids possible de %li pour les paquets %d et %d\n",max_value_common_data_matrix,weight_two_packages,i_bis,j_bis);
						} weight_two_packages = 0;
						data->head_2 = data->head_2->next;
					} data->head = data->head->next;
				}
				//~ printf("max value comme data vaut à la fin %d\n",max_value_common_data_matrix);
				//Getting back to the beginning of the linked list
				data->head = data->first_link;
				data->head_2 = data->first_link;
			}
			//Else, we are using algo 3, so we don't check the max weight
			else {
				printf("algo 3 bp 1\n");
				for (i_bis =0; i_bis < nb_pop; i_bis++) { 
					for (j_bis = i_bis+1; j_bis < nb_pop; j_bis++) {
						if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { 
							max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
						}
					}
				}
			}
		}
		//ALGO 4 : Merge les paquets min + dans l'ordre du tableau trié
			
			
				
			//Merge des paquets et test de taille < GPU_MAX
			for (i = 0; i < nb_pop; i++) {
				data->head_2 = data->head;
				data->head_2 = data->head_2->next;
				for (j = i + 1; j< nb_pop; j++) {
					//~ printf("GPU_RAM vaut : %zd\n",GPU_RAM);
					
					//Pas utile du coup normalement avec la modif plus haut de max value common data
					//~ weight_two_packages = 0;
					//~ //Calcul du poids des données des deux paquets que l'on veut merge
					//~ for (i_bis = 0; i_bis < data->head->package_nb_data; i_bis++) {
						//~ weight_two_packages += starpu_data_get_size(data->head->package_data[i_bis]);
					//~ }
					//~ for (i_bis = 0; i_bis < data->head_2->package_nb_data; i_bis++) {
						//~ weight_two_packages += starpu_data_get_size(data->head_2->package_data[i_bis]);
					//~ }
					
					weight_two_packages = 0;
					//On somme d'abord les poids de tout le paquet 1
					for (i_bis = 0; i_bis < data->head->package_nb_data; i_bis++) {
						weight_two_packages += starpu_data_get_size(data->head->package_data[i_bis]);
					}
					//ensuite on somme le paquet 2 mais uniquement les données qui ne sont pas dans le paquet 1
					for (i_bis = 0; i_bis < data->head_2->package_nb_data; i_bis++) {
						bool_data_common = 0;
						for (j_bis = 0; j_bis < data->head->package_nb_data; j_bis++) {
							if (data->head_2->package_data[i_bis] == data->head->package_data[j_bis])
							{
								bool_data_common = 1;
							}
						}
						if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->head_2->package_data[i_bis]); }
					}
					
					//~ printf("Le poids des paquets %d et %d serait de : %li\n",i,j,weight_two_packages);
					
					if ( ((((matrice_donnees_commune[i][j] == max_value_common_data_matrix) && (max_value_common_data_matrix != 0)) || (GPU_limit_switch == 0)) && data->ALGO_USED_READER != 4) 
					||  ( data->ALGO_USED_READER == 4 && data->head->nb_task_in_sub_list == min_nb_task_in_sub_list && matrice_donnees_commune[i][j] == tab_max_value_common_data_matrix[i]) ) {
						
						if ( (weight_two_packages > GPU_RAM) && (GPU_limit_switch == 1) ) { 
						//~ if (GPU_RAM == 0) { 
							printf("On dépasse GPU_RAM!\n"); 
						}
						else {
							printf("On va merge le paquet %d et le paquet %d\n",i,j);
						//Dis que on a reussi a faire un paquet
						packaging_impossible = 0;
						//Si on utilise l'algo 4 on veut arrêter cette boucle pour ne pas regrouper plus que 1 fois le paquet courant
						
						//Interdit i et j de faire des regroupements par la suite
						for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; }
						for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; }
						nb_data_commun--;

						//Marche que si il y a une seule tâche
						//~ if (!starpu_task_list_empty(&data->head_2->sub_list)) { 
						while (!starpu_task_list_empty(&data->head_2->sub_list)) { 
							//Push la tâche dans la liste de head_1
							starpu_task_list_push_back(&data->head->sub_list,starpu_task_list_pop_front(&data->head_2->sub_list)); 
							data->head->nb_task_in_sub_list ++;
						}

							i_bis = 0; j_bis = 0;
							tab_runner = 0;
							starpu_data_handle_t *temp_data_tab = malloc((data->head->package_nb_data + data->head_2->package_nb_data) * sizeof(data->head->package_data[0]));
							//Merge the two tabs containing datas
							while (i_bis < data->head->package_nb_data && j_bis < data->head_2->package_nb_data) {
								if (data->head->package_data[i_bis] <= data->head_2->package_data[j_bis]) {
									temp_data_tab[tab_runner] = data->head->package_data[i_bis];
									i_bis++;
								}
								else
								{
									temp_data_tab[tab_runner] = data->head_2->package_data[j_bis];
									j_bis++;
								}
								tab_runner++;
							}
							//Copy the remaining data(s) of the tabs (if there are any)
							while (i_bis < data->head->package_nb_data) { temp_data_tab[tab_runner] = data->head->package_data[i_bis]; i_bis++; tab_runner++; }
							while (j_bis < data->head_2->package_nb_data) { temp_data_tab[tab_runner] = data->head_2->package_data[j_bis]; j_bis++; tab_runner++; }
							
							//We remove the duplicate data(s)
							for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data); i_bis++) {
								if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1]) {
									temp_data_tab[i_bis] = 0;
									nb_duplicate_data++;
								}
							}
							//Then we put the temp_tab in head
							data->head->package_data = malloc((data->head->package_nb_data + data->head_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
							j_bis = 0;
							for (i_bis = 0; i_bis < (data->head->package_nb_data + data->head_2->package_nb_data); i_bis++)
							{
								if (temp_data_tab[i_bis] != 0) { data->head->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++; }
							}
							
							//We update the number of data of each package. It's important because we use delete_link(data) to delete all the packages with 0 data
							data->head->package_nb_data = data->head_2->package_nb_data + data->head->package_nb_data - nb_duplicate_data;
							data->head_2->package_nb_data = 0;
							nb_duplicate_data = 0;
							//~ data->head->nb_task_in_sub_list++;
							data->head_2->nb_task_in_sub_list = 0;
							
							//Goto pour l'algo 2
							if (data->ALGO_USED_READER == 2) { 
								//~ printf("algo 2\n"); 
								goto algo_2; 
							}
					} }
					data->head_2 = data->head_2->next;
				} 
				if (nb_data_commun > 1) {
					data->head = data->head->next;
				} 
			}
			
			//goto pour algo 2
			algo_2:
			
			//Supprimer les maillons vide
			data->head = data->first_link;
			data->head = delete_link(data);
			if (data->ALGO_USED_READER != 3) {
				fprintf(fcoordinate,"\nCoordonnées de l'itération n°%d\n",nb_of_loop);
				while (data->head != NULL) {
					//Code to print the coordinante and the data of each package ieration by iteration -----------------------
					cursor_position = ftell(fcoordinate);
					for (i = 0; i < data->head->package_nb_data; i++) {
						starpu_data_get_coordinates_array(data->head->package_data[i],2,temp_tab_coordinates);
						if (((temp_tab_coordinates[0]) != 0) || ((temp_tab_coordinates[1]) !=0 ) || ((data_0_0_in_C == data->head->package_data[i])))  {
							coordinate_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = link_index;
							//Re init du tab des coordonnées
							temp_tab_coordinates[0] = 0; temp_tab_coordinates[1] = 0;
						}
					}
					//--------------------------------------------------------------------------------------------------------
					for (temp_task_3  = starpu_task_list_begin(&data->head->sub_list); temp_task_3 != starpu_task_list_end(&data->head->sub_list); temp_task_3  = starpu_task_list_next(temp_task_3)) {
						//~ printf("La tâche %p est dans le paquet numéro %d\n",temp_task_3,link_index);
						temp_number_task ++;
					}
					temp_moyenne += temp_number_task;
					temp_variance += temp_number_task*temp_number_task;
					temp_number_task = 0;
					//Compte le nombre de paquets, permet d'arrêter l'algo 3 ou les autres algo si on arrive a 1 paquet
					link_index++;
					data->head = data->head->next;
					//~ printf("-----------------------------------------------\n");
				} 
				
				//Code to write in a file the coordinates ---------------------------------------------------------------------------
				fprintf(fcoordinate,"\\begin{tabular}{ c | c | c }"); 
				for (i_bis = 0; i_bis < sqrt(number_tasks)*3 + 6; i_bis++) { 
				
					} fprintf(fcoordinate,"\n");
				for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) { 
					for (j_bis = 0; j_bis < sqrt(number_tasks) - 1; j_bis++) {
						fprintf(fcoordinate,"%d &",coordinate_visualization_matrix[j_bis][i_bis]);
						
					} fprintf(fcoordinate,"%d",coordinate_visualization_matrix[j_bis][i_bis]); 
					fprintf(fcoordinate," \\\\"); fprintf(fcoordinate,"\n \\hline");
					fprintf(fcoordinate,"\n"); 
					for (j_bis = 0; j_bis < sqrt(number_tasks)*3 + 2; j_bis++) { 
						} 
				}
				fprintf(fcoordinate, "\\end{tabular}");

				for (i_bis = 0; i_bis < sqrt(number_tasks)*3 + 6; i_bis++) { 
					} fprintf(fcoordinate,"\n");
				for (i_bis = 0; i_bis < sqrt(number_tasks); i_bis++) {
					for (j_bis = 0; j_bis < sqrt(number_tasks); j_bis++) {
						coordinate_visualization_matrix[j_bis][i_bis] = NULL;
					}
				}
				// ----------------------------------------------------------------------------------------------------------------
				
				//~ //Code to print moyenne variance ecart type nombre de tâches par paquet -------
				temp_moyenne = temp_moyenne/link_index;
				moyenne = temp_moyenne;
				//~ printf("La moyenne du nb de taches par paquets est %f\n",temp_moyenne);
				//~ fprintf(variance_ecart_type,"%d	",nb_of_loop);
				temp_variance = (temp_variance/link_index) - (temp_moyenne*temp_moyenne);
				//~ printf("La variance du nb de taches par paquets est %f\n",temp_variance);
				//~ fprintf(variance_ecart_type,"%f",temp_variance);
				temp_ecart_type = sqrt(temp_variance);
				ecart_type = temp_ecart_type;
				//~ printf("L'ecart type du nb de taches par paquets est %f\n",temp_ecart_type);
				//~ fprintf(variance_ecart_type,"%f\n",temp_ecart_type);
				//~ printf("A la fin du tour numéro %d du while on a %d paquets\n\n",nb_of_loop,link_index);
				//~ printf("Fin du tour numéro %d du while!\n\n",nb_of_loop);
				temp_moyenne = 0; temp_variance = 0; temp_ecart_type = 0;
				//~ data->head = data->first_link;
				
				//~ //-----------------------------------------------------------------------------
				
				//Code to fprintf the number of packages per itération ---
				//~ fprintf(Nb_package_by_loop,"%d	%d\n",nb_of_loop,link_index);
				// ------------------------------------------------------
				
				//Code to fprintf the Mean_task by packages per itération ---
				//~ fprintf(Mean_task_by_loop,"%d	%d	%f	%f	%f	%f\n",nb_of_loop,link_index,temp_moyenne,temp_ecart_type,temp_moyenne-temp_ecart_type,temp_moyenne+temp_ecart_type);
				//Code for ecart type
				//~ fprintf(Mean_task_by_loop,"%f	\n",temp_ecart_type);
				//~ temp_moyenne = 0; temp_variance = 0; temp_ecart_type = 0;
				// ------------------------------------------------------
			}
			//Si on est dans l'algo 3
			else { while (data->head != NULL) { link_index++; data->head = data->head->next; } }
				
			
			
		//Reset de la matrice
		for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
		//Reset de nb_pop!
		nb_pop = link_index;
		
		if (nb_pop == 1) { break; }
		//~ printf("packaging impossible vaut %d\n",packaging_impossible);
		
		//~ link_index = 0;
		//~ printf("\n");
		//~ printf("----------------------------------\n");
		//~ while (data->head != NULL) {
			//~ for (i = 0; i < data->head->package_nb_data; i++) {
					//~ starpu_data_get_coordinates_array(data->head->package_data[i],2,temp_tab_coordinates);
					//~ if (((temp_tab_coordinates[0]) != 0) || ((temp_tab_coordinates[1]) !=0 ) || ((data_0_0_in_C == data->head->package_data[i])))  {
							//~ printf("Les coordonnées de la donnée %p du paquet %d sont : x = %d / y = %d\n",data->head->package_data[i],link_index,temp_tab_coordinates[0],temp_tab_coordinates[1]);
					//~ }
			//~ }	
			//~ data->head = data->head->next; link_index++; printf("---\n");
		//~ }
		//~ data->head = data->first_link;
		//~ printf("Fin du while\n");
		} // Fin du while (packaging_impossible == 0) {
		
		if (data->ALGO_USED_READER != 3) { 
			//~ fprintf(variance_ecart_type,"%f",moyenne);
			//~ fprintf(variance_ecart_type,"	%f\n",ecart_type);
			fclose(fcoordinate);
			//~ fclose(variance_ecart_type);
		}
		
		//Si on est dans l'algo 3 on retire la limite GPU-RAM et on refait un tour de while
		if ((data->ALGO_USED_READER == 3) && (link_index > 1)) { GPU_limit_switch = 0; printf("goto\n"); goto algo3; }
		//Sinon on arrête ici l'exécution
		else { }
		
		
		data->head = data->first_link;	
		
		//Code to print everything ----
		//~ link_index = 0;
		//~ long int total_weight = 0;
		//~ double task_duration_info = 0;
		//~ double bandwith_info = starpu_transfer_bandwidth(STARPU_MAIN_RAM, starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx)));
		//~ printf("A la fin du regroupement des tâches utilisant l'algo %d on obtient : \n",data->ALGO_USED_READER);
		//~ while (data->head != NULL) { link_index++; data->head = data->head->next; } data->head = data->first_link;
		//~ printf("On a fais %d tour(s) de la boucle while et on a fais %d paquet(s)\n",nb_of_loop,link_index);
		//~ printf("-----\n");
		//~ link_index = 0;	
		//~ while (data->head != NULL) {
			//~ printf("Le paquet %d contient %d tâche(s) et %d données\n",link_index,data->head->nb_task_in_sub_list,data->head->package_nb_data);
			//~ for (temp_task_3  = starpu_task_list_begin(&data->head->sub_list); temp_task_3 != starpu_task_list_end(&data->head->sub_list); temp_task_3  = starpu_task_list_next(temp_task_3)) {
				//~ printf("%p\n",temp_task_3);
			//~ }
			//~ for (i = 0; i < data->head->package_nb_data; i++) {
				//~ total_weight+= starpu_data_get_size(data->head->package_data[i]);
			//~ }
			//~ printf("Le poids des données du paquet %d est : %li\n",link_index,total_weight);
				//~ for (i = 0; i < data->head->package_nb_data; i++) {
					//~ starpu_data_get_coordinates_array(data->head->package_data[i],2,temp_tab_coordinates);
					//~ if (((temp_tab_coordinates[0]) != 0) || ((temp_tab_coordinates[1]) !=0 ) || ((data_0_0_in_C == data->head->package_data[i])))  {
						//~ printf("Les coordonnées de la donnée %p sont : x = %d / y = %d\n",data->head->package_data[i],temp_tab_coordinates[0],temp_tab_coordinates[1]);
					//~ }
				//~ }
			//~ total_weight = 0;
			//~ link_index++;
			//~ data->head = data->head->next;
			//~ printf("-----\n");
		//~ }
		//~ data->head = data->first_link;
		//~ printf("\n");
		//~ printf("Info de la bande passante : %f\n",bandwith_info);
		//~ temp_task_3  = starpu_task_list_begin(&data->head->sub_list);
		//~ task_duration_info = starpu_task_worker_expected_length(temp_task_3, 0, component->tree->sched_ctx_id,0);
		//~ printf("La tâche %p a durée %f\n",temp_task_3,task_duration_info);
		//~ printf("\n\n");
		//~ data->head = data->first_link;
		// ---------------------------
		
		//Code to fprintf in packages_data.txt ------------------------------------------------------------------------------------------------------------------------------------
		//data file we output with stat about the number of packages etc...
		//~ FILE * data_output;
		//~ data_output = fopen("packages_data.txt", "a+");
		//~ link_index = 0;	
		//~ int weight_all_packages = 0;
		//~ int number_data_without_duplicate = 0;
		//~ while (data->head != NULL) {
			//~ link_index++;
			//~ number_data_without_duplicate += data->head->package_nb_data;
			//~ for (i = 0; i < data->head->package_nb_data; i++) {
				//~ weight_all_packages += starpu_data_get_size(data->head->package_data[i]);
			//~ }
			//~ data->head = data->head->next;
		//~ }
		//~ double matrix_size = 960*sqrt(number_tasks);
		//~ mean_task_by_packages = number_tasks/link_index;
		//~ double mean_data_by_packages = number_data_without_duplicate/link_index;
		//~ double mean_weight_packages = weight_all_packages/link_index;
		//~ //Le *3 est du brute force, il faudra le changer si l'app devient générique
		//~ fprintf(data_output, "%f	%f	%f	%d	%d	%f %f %f\n",matrix_size,number_tasks,number_tasks*3,number_data_without_duplicate,link_index,mean_task_by_packages,mean_data_by_packages,mean_weight_packages);
		//~ data->head = data->first_link;
		// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		//~ data->head = data->first_link;
		//~ packing_time = clock();
		//~ printf("Temps d'empaquetage = %d ms\n", packing_time);
		
			task1 = starpu_task_list_pop_front(&data->head->sub_list);
	}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (task1 != NULL) { 
			//~ printf("Task %p is getting out of pull_task\n",task1); 
		}
		return task1;
	} //Else de if ((data->head->next == NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
	if (!starpu_task_list_empty(&data->head->sub_list)) {
		task1 = starpu_task_list_pop_front(&data->head->sub_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
	if ((data->head->next != NULL) && (starpu_task_list_empty(&data->head->sub_list))) {
		//The list is empty and it's not the last one, so we go on the next link
		data->head = data->head->next;
		while (starpu_task_list_empty(&data->head->sub_list)) { data->head = data->head->next; }
			task1 = starpu_task_list_pop_front(&data->head->sub_list); 
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ printf("Task %p is getting out of pull_task from starpu_task_list_empty(&data->head->sub_list)\n",task1);
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
	data->ALGO_USED_READER = starpu_get_env_number_default("ALGO_USED",1);
	
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
	//~ variable_globale = *data;
}

static void deinitialize_basic_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	//~ fclose(variance_ecart_type);
	//~ fcloseNb_package_by_loop;
	//~ FILE * Mean_task_by_loop;
	//~ fclose(fcoordinate);
	//~ total_time = clock();
	//~ printf("Temps total = %d ms\n", total_time);
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
