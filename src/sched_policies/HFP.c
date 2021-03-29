/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_data_maxime.h> //Nécessaire pour Belady
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
#include <stdio.h>
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include "starpu_stdlib.h"
#include "common/list.h"
#include <assert.h>
#define PRINTF /* O or 1 */
#define ORDER_U /* O or 1 */
#define BELADY /* O or 1 */
#define MULTIGPU /* 0 : on ne fais rien, 1 : on construit |GPU| paquets et on attribue chaque paquet à un GPU au hasard, 2 : pareil que 1 + load balance, 3 : pareil que 2 + HFP sur chaque paquet, 4 : pareil que 2 mais avec expected time a la place du nb de données, 5 pareil que 4 + HFP sur chaque paquet */
#define MODULAR_HEFT_HFP_MODE /* 0 we don't use heft, 1 we use starpu_prefetch_task_input_on_node_prio, 2 we use starpu_prefetch_task_input_on_node_prio. Put it at 1 or 2 if you use modular-heft-HFP, else it will crash. the 0 is just here so we don't do prefetch when we use regular HFP. */
#define HMETIS /* 0 we don't use hMETIS, 1 we use it to form |GPU| package, 2 same as 1 but we then apply HFP on each package */
#define READY /* 0 we don't use ready in initialize_HFP_center_policy, 1 we do */

static int NT;
static int N;
static double EXPECTED_TIME;

/* Structure used to acces the struct my_list. There are also task's list */
struct HFP_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	//~ struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	
	/* All the pointer use to navigate through the linked list */
	//~ struct my_list *temp_pointer_1;
	//~ struct my_list *temp_pointer_2;
	//~ struct my_list *temp_pointer_3;
	//~ struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	struct paquets *p;
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
     	
     //~ int NP; /* Number of packages */
};

/* Structure used to store all the variable we need and the tasks of each package. Each link is a package */
struct my_list
{
	int package_nb_data; 
	int nb_task_in_sub_list;
	int index_package; /* Used to write in Data_coordinates.txt and keep track of the initial index of the package */
	starpu_data_handle_t * package_data; /* List of all the data in the packages. We don't put two times the duplicates */
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */
	struct my_list *next;
	int split_last_ij; /* The separator of the last state of the current package */
	//~ starpu_data_handle_t * data_use_order; /* Order in which data will be loaded. used for Belady */
	int total_nb_data_package;
	double expected_time;
	double expected_time_pulled_out;
};

struct paquets
{		
	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *temp_pointer_2;
	struct my_list *temp_pointer_3;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */     	
    int NP; /* Number of packages */
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
void HFP_insertion(struct paquets *a)
{
    struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
    new->next = a->temp_pointer_1;
    new->nb_task_in_sub_list = 0;
    new->expected_time_pulled_out = 0;
    a->temp_pointer_1 = new;
}

/* Put a link at the beginning of the linked list */
void insertion_use_order(struct gpu_list *a)
{
    struct use_order *new = malloc(sizeof(*new));
    new->next_gpu = a->pointer;
    a->pointer = new;
}

/* Delete all the empty packages */
struct my_list* HFP_delete_link(struct paquets* a)
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
    int i = 0;
    if (num < 7) {
		num ++;
		*r = num & 1 ? 255 : 0;
		*g = num & 2 ? 255 : 0;
		*b = num & 4 ? 255 : 0;
		return;
    }
    num -= 7; *r = 0; *g = 0; *b = 0;
    for (i = 0; i < 8; i++) {
        *r = *r << 1 | ((num & 1) >> 0);
        *g = *g << 1 | ((num & 2) >> 1);
        *b = *b << 1 | ((num & 4) >> 2);
        num >>= 3;
    }
}

void init_visualisation_tache_matrice_format_tex()
{
	FILE * fcoordinate; /* Coordinates at each iteration */
	fcoordinate = fopen("Output_maxime/Data_coordinates.tex", "w");
	fprintf(fcoordinate,"\\documentclass{article}\\usepackage{color}\\usepackage{fullpage}\\usepackage{colortbl}\\usepackage{caption}\\usepackage{subcaption}\\usepackage{float}\\usepackage{graphics}\n\n\\begin{document}\n\n\\begin{figure}[H]");
	FILE * fcoordinate_order; /* Order in wich the task go out */
	FILE * fcoordinate_order_last; /* Order in wich the task go out */
	fcoordinate_order = fopen("Output_maxime/Data_coordinates_order.tex", "w");
	fcoordinate_order_last = fopen("Output_maxime/Data_coordinates_order_last.tex", "w");
	fprintf(fcoordinate_order,"\\documentclass{article}\\usepackage{color}\\usepackage{fullpage}\\usepackage{colortbl}\\usepackage{caption}\\usepackage{subcaption}\\usepackage{float}\\usepackage{graphics}\n\n\\begin{document}\n\n\\begin{figure}[H]");
	fprintf(fcoordinate_order_last,"\\documentclass{article}\\usepackage{color}\\usepackage{fullpage}\\usepackage{colortbl}\\usepackage{caption}\\usepackage{subcaption}\\usepackage{float}\\usepackage{graphics}\n\n\\begin{document}\n\n\\begin{figure}[H]");
	fclose(fcoordinate);
	fclose(fcoordinate_order);
	fclose(fcoordinate_order_last);
}

void end_visualisation_tache_matrice_format_tex()
{
	FILE * fcoordinate = fopen("Output_maxime/Data_coordinates.tex", "a");
	FILE * fcoordinate_order = fopen("Output_maxime/Data_coordinates_order.tex", "a");
	FILE * fcoordinate_order_last = fopen("Output_maxime/Data_coordinates_order_last.tex", "a");
	fprintf(fcoordinate,"\\caption{HFP packing}\\end{figure}\n\n\\end{document}");
	fprintf(fcoordinate_order,"\\caption{Task's processing order}\\end{figure}\n\n\\end{document}");
	fprintf(fcoordinate_order_last,"\\caption{Task's processing order}\\end{figure}\n\n\\end{document}");
	fclose(fcoordinate);
	fclose(fcoordinate_order);
	fclose(fcoordinate_order_last);
}

void visualisation_tache_matrice_format_tex(int tab_paquet[][N], int tab_order[][N], int nb_of_loop, int link_index)
{
	int i, j, red, green, blue = 0;
	FILE * fcoordinate = fopen("Output_maxime/Data_coordinates.tex", "a");
	FILE * fcoordinate_order = fopen("Output_maxime/Data_coordinates_order.tex", "a");
	fprintf(fcoordinate,"\n\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{|");
	fprintf(fcoordinate_order,"\n\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{|"); 
	for (i = 0; i < N - 1; i++) {
		fprintf(fcoordinate,"c|");
		fprintf(fcoordinate_order,"c|");
	}
	fprintf(fcoordinate,"c|}\n\\hline");
	fprintf(fcoordinate_order,"c|}\n\\hline");
	for (i = 0; i < N; i++) { 
		for (j = 0; j < N - 1; j++) {
			if (tab_paquet[j][i] == 0) { red = 255; green = 255; blue = 255; }
			else if (tab_paquet[j][i] == 6) { red = 70; green = 130; blue = 180; }
			else { rgb(tab_paquet[j][i], &red, &green, &blue); }
			fprintf(fcoordinate,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, tab_paquet[j][i]);
			fprintf(fcoordinate_order,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, tab_order[j][i]);
		}
		if (tab_paquet[j][i] == 0) { red = 255; green = 255; blue = 255; }
		else if (tab_paquet[j][i] == 6) { red = 70; green = 130; blue = 180; }
		else { rgb(tab_paquet[j][i], &red, &green, &blue); }
		fprintf(fcoordinate,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,tab_paquet[j][i]); 
		fprintf(fcoordinate_order,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,tab_order[j][i]); 
		fprintf(fcoordinate," \\\\"); fprintf(fcoordinate,"\\hline");
		fprintf(fcoordinate_order," \\\\"); fprintf(fcoordinate_order,"\\hline");
	}
	if (nb_of_loop > 1 && nb_of_loop%2 == 0) { 
		fprintf(fcoordinate, "\\end{tabular} \\caption{Iteration %d} \\end{subfigure} \\\\",nb_of_loop); 
		fprintf(fcoordinate_order, "\\end{tabular} \\caption{Iteration %d} \\end{subfigure} \\\\",nb_of_loop);
		if (nb_of_loop == 10) { 
			fprintf(fcoordinate,"\\end{figure}\\begin{figure}[H]\\ContinuedFloat");
			fprintf(fcoordinate_order,"\\end{figure}\\begin{figure}[H]\\ContinuedFloat");
		}
	}
	else { 
		fprintf(fcoordinate, "\\end{tabular} \\caption{Iteration %d} \\end{subfigure}",nb_of_loop); 
		fprintf(fcoordinate_order, "\\end{tabular} \\caption{Iteration %d} \\end{subfigure}",nb_of_loop); 
	}
	fprintf(fcoordinate,"\n");
	fprintf(fcoordinate_order,"\n");
	fclose(fcoordinate);
	fclose(fcoordinate_order);
	if (link_index == 1) { 
		FILE * fcoordinate_order_last = fopen("Output_maxime/Data_coordinates_order_last.tex", "a");
		fprintf(fcoordinate_order_last,"\n\\centering\\begin{tabular}{|"); 
	for (i = 0; i < N - 1; i++) {
		fprintf(fcoordinate_order_last,"c|");
	}
	fprintf(fcoordinate_order_last,"c|}\n\\hline");
	for (i = 0; i < N; i++) { 
		for (j = 0; j < N - 1; j++) {
			if (tab_paquet[j][i] == 0) { red = 255; green = 255; blue = 255; }
			else if (tab_paquet[j][i] == 6) { red = 70; green = 130; blue = 180; }
			else { rgb(tab_paquet[j][i], &red, &green, &blue); }
			fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, tab_order[j][i]);
		}
		if (tab_paquet[j][i] == 0) { red = 255; green = 255; blue = 255; }
		else if (tab_paquet[j][i] == 6) { red = 70; green = 130; blue = 180; }
		else { rgb(tab_paquet[j][i], &red, &green, &blue); }
		fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,tab_order[j][i]); 
		fprintf(fcoordinate_order_last," \\\\"); fprintf(fcoordinate_order_last,"\\hline");
	}
		fprintf(fcoordinate_order_last, "\\end{tabular} \\caption{Iteration %d}",nb_of_loop); 
	fprintf(fcoordinate_order_last,"\n");
		fclose(fcoordinate_order_last);
	}
}

struct my_list* HFP_reverse_sub_list(struct my_list *a) 
{
	struct starpu_task_list b;
	starpu_task_list_init(&b);
	while (!starpu_task_list_empty(&a->sub_list)) {				
		starpu_task_list_push_front(&b,starpu_task_list_pop_front(&a->sub_list)); }
	while (!starpu_task_list_empty(&b)) {				
		starpu_task_list_push_back(&a->sub_list,starpu_task_list_pop_front(&b)); }
	return a; 	
}

/* Get weight of all different data. Only works if you have only one package */
//~ static void get_weight_all_different_data(struct my_list *a, starpu_ssize_t GPU_RAM_M)
//~ {
	//~ printf("Taille de la matrice : %ld\n",GPU_RAM_M);
	//~ long int weight_all_different_data = 0;
	//~ for (int i = 0; i < a->package_nb_data; i++) {
		//~ weight_all_different_data += starpu_data_get_size(a->package_data[i]); 
	//~ }
	//~ printf("Poids de toutes les données différentes : %li\n",weight_all_different_data);
//~ }

/* Takes a task list and return the total number of data that will be used.
 * It means that it is the sum of the number of data for each task of the list.
 */
int get_total_number_data_task_list(struct starpu_task_list a) 
{
	int total_nb_data_list = 0;
	for (struct starpu_task *task = starpu_task_list_begin(&a); task != starpu_task_list_end(&a); task = starpu_task_list_next(task)) 
	{
		total_nb_data_list +=  STARPU_TASK_GET_NBUFFERS(task);
	}
	return total_nb_data_list;
}

struct gpu_list *gpu_data;
struct use_order *use_order_data;;

/* Donne l'ordre d'utilisation des données ainsi que la liste de l'ensemble des différentes données */
static void get_ordre_utilisation_donnee(struct paquets* a, int NB_TOTAL_DONNEES, int nb_gpu)
{
	int k = 0;
	//~ struct gpu_list *gpu_data = malloc(sizeof(*gpu_data));
	//~ struct use_order *use_order_data = malloc(sizeof(*use_order_data));
	gpu_data = malloc(sizeof(*gpu_data));
	use_order_data = malloc(sizeof(*use_order_data));
	use_order_data->next_gpu = NULL;
	gpu_data->pointer = use_order_data;
	gpu_data->first_gpu = gpu_data->pointer;
	
	/* ces deux fichiers sont juste utile pour le débuggage, on pourra les suppr plus tard */
	FILE *f = fopen("Output_maxime/ordre_utilisation_donnees.txt","w");
	FILE *f_2 = fopen("Output_maxime/ordre_traitement_taches.txt","w");
	//~ struct starpu_task *task = NULL; 
	a->temp_pointer_1 = a->first_link;
	//~ index_task_currently_treated = 0; /* Initialized to 0 here. We need it in get_current_task. Could be initialized elsewhere though */
	while (a->temp_pointer_1 != NULL) 
	{
		use_order_data->total_nb_data = get_total_number_data_task_list(a->temp_pointer_1->sub_list);
		use_order_data->data_list = malloc(use_order_data->total_nb_data*sizeof(a->temp_pointer_1->package_data[0]));
		use_order_data->last_position_in_data_use_order = 0;
		for (struct starpu_task *task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task)) 
		{
			fprintf(f_2,"%p\n",task);
			for (int i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
				use_order_data->data_list[k] = STARPU_TASK_GET_HANDLE(task,i);
				k++;
				fprintf(f,"%p\n",STARPU_TASK_GET_HANDLE(task,i));
				printf("Donnée de %p : %p\n",task,STARPU_TASK_GET_HANDLE(task,i));
			}
			//~ if (j != 0) { task_position_in_data_use_order[j] = STARPU_TASK_GET_NBUFFERS(task) + task_position_in_data_use_order[j - 1]; }
			//~ else { task_position_in_data_use_order[j] = STARPU_TASK_GET_NBUFFERS(task); }
			//~ j++;
		}
		k = 0;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		insertion_use_order(gpu_data);
		//~ use_order_data = use_order_data->next_gpu;
		fprintf(f_2,"-------------\n");
		fprintf(f,"-------------\n");
	}
	fclose(f);
	fclose(f_2);
}

int get_common_data_last_package(struct my_list*I, struct my_list*J, int evaluation_I, int evaluation_J, bool IJ_inferieur_GPU_RAM, starpu_ssize_t GPU_RAM_M) 
{
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("get common data\n"); }
	int split_ij = 0;
	//~ printf("dans get I a %d taches et %d données\n",I->nb_task_in_sub_list,I->package_nb_data);
	//~ printf("dans get J a %d taches et %d données\n",J->nb_task_in_sub_list,J->package_nb_data);
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
		//~ printf("Poids a la fin pour i1 : %li\n",poids);	
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
		//~ printf("Poids a la fin pour i2 : %li\n",poids);	
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
		//~ printf("Poids a la fin pour j1 : %li\n",poids);	
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
		//~ printf("Poids a la fin pour j2 : %li\n",poids);		
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
	//~ printf("Données de I:"); for (i = 0; i < index_tab_donnee_I; i++) { printf(" %p",donnee_I[i]); }
	//~ printf("\n");
	//~ printf("Données de J:"); for (i = 0; i < index_tab_donnee_J; i++) { printf(" %p",donnee_J[i]); }
	//~ printf("\n");
	for (i = 0; i < index_tab_donnee_I; i++) {
		for (j = 0; j < index_tab_donnee_J; j++) {
			if (donnee_I[i] == donnee_J[j]) {
				common_data_last_package++;
				//~ printf("%p\n",donnee_I[i]);
				break;
			}
		}
	}
	return common_data_last_package;
}

/* Comparator used to sort the data of a packages to erase the duplicate in O(n) */
int HFP_pointeurComparator ( const void * first, const void * second ) {
  return ( *(int*)first - *(int*)second );
}

void print_effective_order_in_file (struct starpu_task *task)
{
	//~ int i = 0;
	//~ for (i = 0; i < component->nchildren; i++) {
		//~ if (to == component->children[i]) {
			//~ break;
		//~ }
	//~ }
	//~ char str[2];
	//~ sprintf(str, "%d", i);
	//~ int size = strlen("Output_maxime/Task_order_effective_.txt") + strlen(str);
	//~ char *path = (char *)malloc(size);
	//~ strcpy(path, "Output_maxime/Task_order_effective_");
	//~ strcat(path, str);
	//~ strcat(path, ".txt");
	FILE *f = fopen("Output_maxime/Task_order_effective.txt", "a");
	fprintf(f, "%p\n",task);
	fclose(f);
}

/* Called in HFP_pull_task when we need to return a task. It is used when we have multiple GPUs
 * In case of modular-heft-HFP, it needs to do a round robin on the task it returned. So we use expected_time_pulled_out, 
 * an element of struct my_list in order to track which package pulled out the least expected task time. So heft can can
 * better divide tasks between GPUs */
static struct starpu_task *get_task_to_return(struct starpu_sched_component *component, struct starpu_sched_component *to, struct paquets* a, int nb_gpu)
{
	a->temp_pointer_1 = a->first_link; 
	int i = 0; struct starpu_task *task; double min_expected_time_pulled_out = 0; int package_min_expected_time_pulled_out = 0;
	if (starpu_get_env_number_default("MULTIGPU",0) == 0 && starpu_get_env_number_default("HMETIS",0) == 0)
	{
		task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
		return task;
	}
	else { 	
		if (starpu_get_env_number_default("MODULAR_HEFT_HFP_MODE",0) != 0)
		{
			package_min_expected_time_pulled_out = 0;
			min_expected_time_pulled_out = a->temp_pointer_1->expected_time_pulled_out;
			a->temp_pointer_1 = a->temp_pointer_1->next;
			for (i = 1; i < nb_gpu; i++) {
				/* We also need to check that the package is not empty */
				if (a->temp_pointer_1->expected_time_pulled_out < min_expected_time_pulled_out && !starpu_task_list_empty(&a->temp_pointer_1->sub_list)) {
					min_expected_time_pulled_out = a->temp_pointer_1->expected_time_pulled_out;
					package_min_expected_time_pulled_out = i;
				}
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_1 = a->first_link; 
			for (i = 0; i < package_min_expected_time_pulled_out; i++) {
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
			a->temp_pointer_1->expected_time_pulled_out += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0); 
			return task;
		}
		else
		{
			//~ printf("Il y a %u GPUs\n", component->nchildren);
			//~ printf("Children 1 : %p / ", component->children[0]);
			//~ printf("Children 2 : %p / ", component->children[1]);
			//~ printf("Children 3 : %p / ", component->children[2]);
			//~ printf("to : %p\n", to);	
			for (i = 0; i < nb_gpu; i++) {
				if (to == component->children[i]) {
					break;
				}
				else {
					a->temp_pointer_1 = a->temp_pointer_1->next;
				}
			}
			if (!starpu_task_list_empty(&a->temp_pointer_1->sub_list)) { 
				task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list); 
				//~ if (to == component->children[0]) { printf("GPU 1\n");  }
				//~ if (to == component->children[1]) { printf("GPU 2\n");  }
				//~ if (to == component->children[2]) { printf("GPU 3\n");  }
				return task;
			}
			else
			{ 
				return NULL; 
			}
		}
	}
}

/* Printing each package and its content */
void print_packages_in_terminal (struct paquets *a, int nb_of_loop) {
	int link_index = 0;
	struct starpu_task *task;
	//~ long int total_weight = 0;
	a->temp_pointer_1 = a->first_link;
	while (a->temp_pointer_1 != NULL) 
	{ 
		link_index++; a->temp_pointer_1 = a->temp_pointer_1->next;				
	} 
	a->temp_pointer_1 = a->first_link;
			printf("-----\nOn a fais %d tour(s) de la boucle while et on a fais %d paquet(s)\n",nb_of_loop,link_index);
			printf("-----\n");
			link_index = 0;	
			while (a->temp_pointer_1 != NULL) {
				printf("Le paquet %d contient %d tâche(s) et %d données, expected time = %f\n",link_index,a->temp_pointer_1->nb_task_in_sub_list,a->temp_pointer_1->package_nb_data,a->temp_pointer_1->expected_time);
				//~ for (i = 0; i < a->temp_pointer_1->package_nb_data; i++) {
					//~ total_weight+= starpu_data_get_size(a->temp_pointer_1->package_data[i]);
				//~ }
				for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task)) {
					printf("%p\n",task);
				}
				//~ printf("Le poids des données du paquet %d est : %li\n",link_index,total_weight);
				//~ total_weight = 0;
				link_index++;
				a->temp_pointer_1 = a->temp_pointer_1->next;
				printf("-----\n");
			}
			a->temp_pointer_1 = a->first_link;
}

/* Giving prefetch for each task to modular-heft-HFP */
void prefetch_each_task(struct paquets *a, struct starpu_sched_component *component)
{
	struct starpu_task *task;
	a->temp_pointer_1 = a->first_link;
	
	while (a->temp_pointer_1 != NULL) {
		for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task)) {
			if (starpu_get_env_number_default("MODULAR_HEFT_HFP_MODE",0) == 1)
			{  
				starpu_prefetch_task_input_on_node_prio(task, starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx)), 0);
				printf("prefetch of %p\n", task);
			}
			else if (starpu_get_env_number_default("MODULAR_HEFT_HFP_MODE",0) == 2)
			{  
				starpu_idle_prefetch_task_input_on_node_prio(task, starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx)), 0);
			}
			else
			{
				printf("Wrong environement variable MODULAR_HEFT_HFP_MODE\n"); exit(0);
			}
		}
		a->temp_pointer_1 = a->temp_pointer_1->next;
	}
}

/* Pushing the tasks */		
static int HFP_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Push task\n"); }
	struct HFP_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* Need an empty data paquets_data to build packages
 * Output a task list ordered. Son it's HFP if we have only one package at the end
 * Used for now to reorder task inside a package after load balancing
 * Can be used as main HFP like in pull task later
 * Things commented are things to print matrix or things like that TODO : fix it if we want to print in this function.
 */
struct starpu_task_list hierarchical_fair_packing (struct starpu_task_list task_list, int number_task, starpu_ssize_t GPU_RAM_M) {
	 
	struct paquets *paquets_data = malloc(sizeof(*paquets_data));
	struct my_list *my_data = malloc(sizeof(*my_data));
	starpu_task_list_init(&my_data->sub_list);
	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
		
	int number_of_package_to_build = 1; /* 1 because we want to order just a sub list. To change with an args if we use this elsewhere. */
	struct starpu_task_list non_connexe;
	starpu_task_list_init(&non_connexe);
	int nb_duplicate_data = 0; long int weight_two_packages; /* Used to store the weight the merging of two packages would be. It is then used to see if it's inferior to the size of the RAM of the GPU */
	long int max_value_common_data_matrix = 0; /* Store the maximum weight of the commons data between two packages for all the tasks */
	long int common_data_last_package_i1_j1 = 0; /* Variables used to compare the affinity between sub package 1i and 1j, 1i and 2j etc... */
	long int common_data_last_package_i1_j2 = 0; long int common_data_last_package_i2_j1 = 0; 
	long int common_data_last_package_i2_j2 = 0; long int max_common_data_last_package = 0;
	long int weight_package_i = 0; /* Used for ORDER_U too */
	long int weight_package_j = 0; int i = 0;
	int bool_data_common = 0; int GPU_limit_switch = 1; int temp_nb_min_task_packages = 0; int i_bis = 0; int j_bis = 0; int j = 0; int tab_runner = 0; int index_head_1 = 0; int index_head_2 = 0; int common_data_last_package_i2_j = 0; int common_data_last_package_i1_j = 0; int common_data_last_package_i_j1 = 0; int common_data_last_package_i_j2 = 0;
	int min_nb_task_in_sub_list = 0; int nb_min_task_packages = 0;
	struct starpu_task *task; int nb_of_loop = 0; int packaging_impossible = 0; int link_index = 0; int NB_TOTAL_DONNEES = 0;
	task  = starpu_task_list_begin(&task_list);
	paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
	struct starpu_task *temp_task;
			
			task  = starpu_task_list_begin(&task_list);
			paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
			/* One task == one link in the linked list */
			int do_not_add_more = number_task - 1;
			for (task = starpu_task_list_begin(&task_list); task != starpu_task_list_end(&task_list); task = temp_task) {
				temp_task = starpu_task_list_next(task);
				task = starpu_task_list_pop_front(&task_list);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) {
					paquets_data->temp_pointer_1->package_data[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
				paquets_data->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(task);
				NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(task);
				/* We sort our datas in the packages */
				qsort(paquets_data->temp_pointer_1->package_data,paquets_data->temp_pointer_1->package_nb_data,sizeof(paquets_data->temp_pointer_1->package_data[0]),HFP_pointeurComparator);
				/* Pushing the task and the number of the package in the package*/
				starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list,task);
				paquets_data->temp_pointer_1->index_package = link_index;
				/* Initialization of the lists last_packages */
				paquets_data->temp_pointer_1->split_last_ij = 0;
				paquets_data->temp_pointer_1->total_nb_data_package = STARPU_TASK_GET_NBUFFERS(task);
				link_index++;
				paquets_data->temp_pointer_1->nb_task_in_sub_list=1;
				
				if(do_not_add_more != 0) { 
					HFP_insertion(paquets_data); paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0])); 
				}
				do_not_add_more--;
			}
			paquets_data->first_link = paquets_data->temp_pointer_1;
			paquets_data->temp_pointer_2 = paquets_data->first_link;
			index_head_2++;
			
			/* Matrix used to store all the common data weights between packages
			int coordinate_visualization_matrix_size = N;
			int coordinate_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			int coordinate_order_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			for (i_bis = 0; i_bis < N; i_bis++) {
				for (j_bis = 0; j_bis < N; j_bis++) {
					coordinate_visualization_matrix[j_bis][i_bis] = 0;
					coordinate_order_visualization_matrix[j_bis][i_bis] = 0;
				}
			} */
			
			/* if (starpu_get_env_number_default("PRINTF",0) == 1) { init_visualisation_tache_matrice_format_tex(); } */
			/* THE while loop. Stop when no more packaging are possible */
			while (packaging_impossible == 0) {
				beggining_while_packaging_impossible:
				nb_of_loop++;
				packaging_impossible = 1;
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("############# Itération numéro : %d #############\n",nb_of_loop); }
								
				/* Variables we need to reinitialize for a new iteration */
				paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link; index_head_1 = 0; index_head_2 = 1; link_index = 0; tab_runner = 0; nb_min_task_packages = 0;
				min_nb_task_in_sub_list = 0; weight_two_packages = 0; max_value_common_data_matrix = 0; long int matrice_donnees_commune[number_task][number_task];
				min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list; for (i = 0; i < number_task; i++) { for (j = 0; j < number_task; j++) { matrice_donnees_commune[i][j] = 0; }}
								 
					/* First we get the number of packages that have the minimal number of tasks */
					for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list > paquets_data->temp_pointer_1->nb_task_in_sub_list) { min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list; } }
					for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list == paquets_data->temp_pointer_1->nb_task_in_sub_list) { nb_min_task_packages++; } }
					if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list); }
					/* Then we create the common data matrix */
					for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next) {
						for (paquets_data->temp_pointer_2 = paquets_data->temp_pointer_1->next; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next) {
							for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++) {
								for (j = 0; j < paquets_data->temp_pointer_2->package_nb_data; j++) {
									if ((paquets_data->temp_pointer_1->package_data[i] == paquets_data->temp_pointer_2->package_data[j])) {
										matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]) + starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
										matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]) + starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
									} } } index_head_2++; } index_head_1++; index_head_2 = index_head_1 + 1; }
				/* Code to print the common data matrix */
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Common data matrix : \n"); for (i = 0; i < number_task; i++) { for (j = 0; j < number_task; j++) { printf (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); }}
				
				/* Getting back to the beginning of the linked list */
				paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;
				
					i_bis = 0; j_bis = 0; 
					temp_nb_min_task_packages = nb_min_task_packages;
				debut_while:
					paquets_data->temp_pointer_1 = paquets_data->first_link;
					paquets_data->temp_pointer_2 = paquets_data->first_link;
					max_value_common_data_matrix = 0;
					if (GPU_limit_switch == 1) {
					for (i_bis = 0; i_bis < number_task; i_bis++) {
						if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < paquets_data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < paquets_data->temp_pointer_1->package_nb_data; j++) {
										if (paquets_data->temp_pointer_2->package_data[i] == paquets_data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM_M)) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							paquets_data->temp_pointer_1=paquets_data->temp_pointer_1->next;
							j_bis = 0; }
				paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;
				}
				/* Else, we are using algo 5, so we don't check the max weight */
				else {
					for (i_bis = 0; i_bis < number_task; i_bis++) {
						if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < paquets_data->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < paquets_data->temp_pointer_1->package_nb_data; j++) {
										if (paquets_data->temp_pointer_2->package_data[i] == paquets_data->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i]); } } 
									if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							paquets_data->temp_pointer_1=paquets_data->temp_pointer_1->next;
							j_bis = 0; }
				paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;
				}	
				if (max_value_common_data_matrix == 0) { 
					/* It means that P_i share no data with others, so we put it in the end of the list
					 * For this we use a separate list that we merge at the end
					 * We will put this list at the end of the rest of the packages */
					if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("graphe non connexe\n"); }
					while (!starpu_task_list_empty(&paquets_data->temp_pointer_1->sub_list)) { 
						starpu_task_list_push_back(&non_connexe,starpu_task_list_pop_front(&paquets_data->temp_pointer_1->sub_list));
					}
					paquets_data->temp_pointer_1->package_nb_data = 0;
				}
				else {
				i_bis = 0; j_bis = 0; i = 0; j = 0;
				for (i = 0; i < number_task; i++) {
					if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) {
						for (j = 0; j < number_task; j++) {
							weight_two_packages = 0;  weight_package_i = 0;  weight_package_j = 0;
							for (i_bis = 0; i_bis < paquets_data->temp_pointer_1->package_nb_data; i_bis++) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i_bis]); } weight_package_i = weight_two_packages;
							for (i_bis = 0; i_bis < paquets_data->temp_pointer_2->package_nb_data; i_bis++) { bool_data_common = 0;
								for (j_bis = 0; j_bis < paquets_data->temp_pointer_1->package_nb_data; j_bis++) { if (paquets_data->temp_pointer_2->package_data[i_bis] == paquets_data->temp_pointer_1->package_data[j_bis]) { bool_data_common = 1; } }
								if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i_bis]); } 
								weight_package_j += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i_bis]); }							
							if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j && max_value_common_data_matrix != 0) {
								if ((weight_two_packages <= GPU_RAM_M) || (GPU_limit_switch == 0)) {
								/* Merge */
								packaging_impossible = 0;
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
								
								paquets_data->NP--;
								
								if (paquets_data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list) { temp_nb_min_task_packages--; }
								
								for (j_bis = 0; j_bis < number_task; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;}
								for (j_bis = 0; j_bis < number_task; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;}
								
								if (starpu_get_env_number_default("ORDER_U",0) == 1) {
									if (paquets_data->temp_pointer_1->nb_task_in_sub_list == 1 && paquets_data->temp_pointer_2->nb_task_in_sub_list == 1) {
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I = 1 et J = 1\n"); }
									}
									else if (weight_package_i > GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I > PU_RAM et J <= PU_RAM\n"); }
										common_data_last_package_i1_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 0, false,GPU_RAM_M);					
										common_data_last_package_i2_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 0, false,GPU_RAM_M);					
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("\ni1j = %d / i2j = %d\n",common_data_last_package_i1_j,common_data_last_package_i2_j); }
										if (common_data_last_package_i1_j > common_data_last_package_i2_j) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I\n"); }
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
										}
									}
									else if (weight_package_i <= GPU_RAM_M && weight_package_j > GPU_RAM_M) {
										common_data_last_package_i_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 1, false,GPU_RAM_M);					
										common_data_last_package_i_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 2, false,GPU_RAM_M);					
										if (common_data_last_package_i_j2 > common_data_last_package_i_j1) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET J\n"); }
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
									}
									else {
										if (weight_package_i > GPU_RAM_M && weight_package_j > GPU_RAM_M) {
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, false,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, false,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, false,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, false,GPU_RAM_M);
										}
										else if (weight_package_i <= GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I <= PU_RAM et J <= PU_RAM\n"); }
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, true,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, true,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, true,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, true,GPU_RAM_M);
										}
										else { printf("Erreur dans ordre U, aucun cas choisi\n"); exit(0); }
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("i1j1 = %ld / i1j2 = %ld / i2j1 = %ld / i2j2 = %ld\n",common_data_last_package_i1_j1,common_data_last_package_i1_j2,common_data_last_package_i2_j1,common_data_last_package_i2_j2); }
										max_common_data_last_package = common_data_last_package_i2_j1;
										if (max_common_data_last_package < common_data_last_package_i1_j1) { max_common_data_last_package = common_data_last_package_i1_j1; }
										if (max_common_data_last_package < common_data_last_package_i1_j2) { max_common_data_last_package = common_data_last_package_i1_j2; }
										if (max_common_data_last_package < common_data_last_package_i2_j2) { max_common_data_last_package = common_data_last_package_i2_j2; }
										if (max_common_data_last_package == common_data_last_package_i2_j1) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Pas de switch\n"); }
										}								
										else if (max_common_data_last_package == common_data_last_package_i1_j2) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I ET J\n");	}
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);									
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
										else if (max_common_data_last_package == common_data_last_package_i2_j2) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET J\n"); }
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);	
										}
										else { /* max_common_data_last_package == common_data_last_package_i1_j1 */
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I\n"); }
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);									
										}		
									}							
									//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Fin de l'ordre U sans doublons\n"); }
								}
								
								paquets_data->temp_pointer_1->split_last_ij = paquets_data->temp_pointer_1->nb_task_in_sub_list;
								while (!starpu_task_list_empty(&paquets_data->temp_pointer_2->sub_list)) {
								starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&paquets_data->temp_pointer_2->sub_list)); 
								paquets_data->temp_pointer_1->nb_task_in_sub_list ++; }
								i_bis = 0; j_bis = 0; tab_runner = 0;
								starpu_data_handle_t *temp_data_tab = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data) * sizeof(paquets_data->temp_pointer_1->package_data[0]));
								while (i_bis < paquets_data->temp_pointer_1->package_nb_data && j_bis < paquets_data->temp_pointer_2->package_nb_data) {
									if (paquets_data->temp_pointer_1->package_data[i_bis] <= paquets_data->temp_pointer_2->package_data[j_bis]) {
										temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
										i_bis++; }
									else {
										temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
										j_bis++; }
									tab_runner++;
								}
								while (i_bis < paquets_data->temp_pointer_1->package_nb_data) { temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis]; i_bis++; tab_runner++; }
								while (j_bis < paquets_data->temp_pointer_2->package_nb_data) { temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis]; j_bis++; tab_runner++; }
								for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1]) {
										temp_data_tab[i_bis] = 0;
										nb_duplicate_data++; } }
								paquets_data->temp_pointer_1->package_data = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
								j_bis = 0;
								for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] != 0) { paquets_data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++; } }
								paquets_data->temp_pointer_1->package_nb_data = paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data - nb_duplicate_data;
								
								paquets_data->temp_pointer_1->total_nb_data_package += paquets_data->temp_pointer_2->total_nb_data_package;
								paquets_data->temp_pointer_1->expected_time += paquets_data->temp_pointer_2->expected_time;
								
								paquets_data->temp_pointer_2->package_nb_data = 0;
								nb_duplicate_data = 0;
								paquets_data->temp_pointer_2->nb_task_in_sub_list = 0;
							temp_nb_min_task_packages--;
							if(paquets_data->NP == number_of_package_to_build) { goto break_merging_1; }
							if (temp_nb_min_task_packages > 1) {
								goto debut_while; 
							}
							else { j = number_task; i = number_task; }
							} }
							paquets_data->temp_pointer_2=paquets_data->temp_pointer_2->next;
						}
					}
					paquets_data->temp_pointer_1=paquets_data->temp_pointer_1->next; paquets_data->temp_pointer_2=paquets_data->first_link;
				}
				}			
				
				break_merging_1:
				
				paquets_data->temp_pointer_1 = paquets_data->first_link;
				paquets_data->temp_pointer_1 = HFP_delete_link(paquets_data);
				tab_runner = 0;
				
				/* Code to get the coordinates of each data in the order in wich tasks get out of pull_task */
					while (paquets_data->temp_pointer_1 != NULL) {
					/* if ((strcmp(appli,"starpu_sgemm_gemm") == 0) && (starpu_get_env_number_default("PRINTF",0) == 1)) {
						for (task = starpu_task_list_begin(&paquets_data->temp_pointer_1->sub_list); task != starpu_task_list_end(&paquets_data->temp_pointer_1->sub_list); task  = starpu_task_list_next(task)) {
							starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task,2),2,temp_tab_coordinates);
							coordinate_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = NT - paquets_data->temp_pointer_1->index_package - 1;
							coordinate_order_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = tab_runner;
							tab_runner++;	
							temp_tab_coordinates[0] = 0; temp_tab_coordinates[1] = 0;
						}
					}		*/	
						link_index++;
						paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next;
					} 
					/* if (starpu_get_env_number_default("PRINTF",0) == 1) { visualisation_tache_matrice_format_tex(coordinate_visualization_matrix,coordinate_order_visualization_matrix,nb_of_loop,link_index); } */
			 
			/* Checking if we have the right number of packages. if MULTIGPU is equal to 0 we want only one package. if it is equal to 1 we want |GPU| packages */
			if (link_index == number_of_package_to_build) { goto end_while_packaging_impossible; }
				
			for (i = 0; i < number_task; i++) { for (j = 0; j < number_task; j++) { matrice_donnees_commune[i][j] = 0; }}
			/* Reset number_task for the matrix initialisation */
			number_task = link_index;
			/* If we have only one package we don't have to do more packages */			
			if (number_task == 1) {  packaging_impossible = 1; }
		} /* End of while (packaging_impossible == 0) { */
		/* We are in algorithm 3, we remove the size limit of a package */
		GPU_limit_switch = 0; goto beggining_while_packaging_impossible;
		
		end_while_packaging_impossible:
		
		/* Add tasks or packages that were not connexe */
		while(!starpu_task_list_empty(&non_connexe)) {
			starpu_task_list_push_back(&paquets_data->first_link->sub_list, starpu_task_list_pop_front(&non_connexe));
			paquets_data->first_link->nb_task_in_sub_list++;
		}
				
		return paquets_data->first_link->sub_list;
}

/* Check if our struct is empty */
bool is_empty(struct my_list* a)
{
	if (a == NULL) { return true; }
	if (!starpu_task_list_empty(&a->sub_list)) {
			return false;
		}
	while (a->next != NULL) {
		 a = a->next;
		if (!starpu_task_list_empty(&a->sub_list)) {
			return false;
		}
	}
	return true;
}

/* Push back in a package a task
 * Used in load_balance
 * Does not manage to migrate data of the task too
 */
void merge_task_and_package (struct my_list *package, struct starpu_task *task)
{
	int i = 0; int j = 0; int tab_runner = 0; int nb_duplicate_data = 0;
	package->nb_task_in_sub_list++; 
	starpu_data_handle_t *temp_data_tab = malloc((package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task))*sizeof(package->package_data[0]));
	while (i < package->package_nb_data && j < STARPU_TASK_GET_NBUFFERS(task)) {
		if (package->package_data[i] <= STARPU_TASK_GET_HANDLE(task,j)) {
			temp_data_tab[tab_runner] = package->package_data[i];
			i++; }
		else {
			temp_data_tab[tab_runner] = STARPU_TASK_GET_HANDLE(task,j);
			j++; }
			tab_runner++;
	}
	while (i < package->package_nb_data) { temp_data_tab[tab_runner] = package->package_data[i]; i++; tab_runner++; }
	while (j < STARPU_TASK_GET_NBUFFERS(task)) { temp_data_tab[tab_runner] = STARPU_TASK_GET_HANDLE(task,j); j++; tab_runner++; }
	for (i = 0; i < (package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task)); i++) {
		if (temp_data_tab[i] == temp_data_tab[i + 1]) {
			temp_data_tab[i] = 0;
			nb_duplicate_data++; } }
	package->package_data = malloc((package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task) - nb_duplicate_data) * sizeof(starpu_data_handle_t));
	j = 0;
	for (i = 0; i < (package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task)); i++) {
		if (temp_data_tab[i] != 0) { package->package_data[j] = temp_data_tab[i]; j++; } }
	package->package_nb_data = STARPU_TASK_GET_NBUFFERS(task) + package->package_nb_data - nb_duplicate_data;
	package->total_nb_data_package += STARPU_TASK_GET_NBUFFERS(task);	
	package->expected_time += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);	
	starpu_task_list_push_back(&package->sub_list, task); 						
}

//~ printf("%f\n", starpu_task_expected_length(task1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0));
/* Equilibrates package in order to have packages with the exact same expected task time
 * Called in HFP_pull_task once all packages are done 
 */
void load_balance_expected_time (struct paquets *a, int number_gpu)
{
	struct starpu_task *task;
	double ite = 0; int i = 0;
	int package_with_min_expected_time, package_with_max_expected_time;
	double min_expected_time, max_expected_time, expected_time_to_steal = 0;
	bool load_balance_needed = true;
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("A package should have %f time +/- 5 percent\n", EXPECTED_TIME/number_gpu); }
	/* Selecting the smallest and biggest package */
	while (load_balance_needed == true) { 
		a->temp_pointer_1 = a->first_link;
		min_expected_time = a->temp_pointer_1->expected_time;
		max_expected_time = a->temp_pointer_1->expected_time;
		package_with_min_expected_time = 0;
		package_with_max_expected_time = 0;
		i = 0;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		while (a->temp_pointer_1 != NULL) {
			i++;
			if (min_expected_time > a->temp_pointer_1->expected_time) {
				min_expected_time = a->temp_pointer_1->expected_time;
				package_with_min_expected_time = i;
			}
			if (max_expected_time < a->temp_pointer_1->expected_time) {
				max_expected_time = a->temp_pointer_1->expected_time;
				package_with_max_expected_time = i;
			}
			a->temp_pointer_1 = a->temp_pointer_1->next;
		}
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("min et max : %f et %f\n",min_expected_time, max_expected_time); }
		//~ exit(0);
		/* Stealing as much task from the last tasks of the biggest packages */
		//~ if (package_with_min_expected_time == package_with_max_expected_time || min_expected_time >=  max_expected_time - ((5*max_expected_time)/100)) {
		if (package_with_min_expected_time == package_with_max_expected_time) {
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("All packages have the same expected time\n"); }
			load_balance_needed = false;
		}
		else {
			/* Getting on the right packages */
			a->temp_pointer_1 = a->first_link;
			for (i = 0; i < package_with_min_expected_time; i++) {
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_2 = a->first_link;
			for (i = 0; i < package_with_max_expected_time; i++) {
				a->temp_pointer_2 = a->temp_pointer_2->next;
			}
			if (a->temp_pointer_2->expected_time - ((EXPECTED_TIME/number_gpu) - a->temp_pointer_1->expected_time) >= EXPECTED_TIME/number_gpu) {
				//~ printf("if\n");
				expected_time_to_steal = (EXPECTED_TIME/number_gpu) - a->temp_pointer_1->expected_time;
			}
			else {
				//~ printf("else\n");
				expected_time_to_steal = a->temp_pointer_2->expected_time - EXPECTED_TIME/number_gpu;
			}
			task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
			if (starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0) > expected_time_to_steal)
			{ 
				//~ printf("task et expected : %f, %f\n",starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0), expected_time_to_steal);
				starpu_task_list_push_back(&a->temp_pointer_2->sub_list, task);
				break; 
			}
			starpu_task_list_push_back(&a->temp_pointer_2->sub_list, task);
			//~ printf("EXPECTED_TIME/number_gpu : %f\n", EXPECTED_TIME/number_gpu);
			//~ printf("EXPECTED_TIME : %f\n", EXPECTED_TIME);
			//~ printf("expected time to steal : %f\n", expected_time_to_steal);
			//~ exit(0);
			ite = 0;
			while (ite < expected_time_to_steal) {
				task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
				ite += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
				//~ printf("ite = %f\n", ite);
				a->temp_pointer_2->expected_time = a->temp_pointer_2->expected_time - starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
				merge_task_and_package(a->temp_pointer_1, task);
				a->temp_pointer_2->nb_task_in_sub_list--;		
			}
			//~ printf("expected time du gros paquet = %f\n", a->temp_pointer_2->expected_time);
			//~ printf("expected time du petit paquet = %f\n", a->temp_pointer_1->expected_time);
			//~ exit(0);
		}
	}
}

/* Equilibrates package in order to have packages with the exact same number of tasks +/-1 task 
 * Called in HFP_pull_task once all packages are done 
 */
void load_balance (struct paquets *a, int number_gpu)
{
	int min_number_task_in_package, package_with_min_number_task, i, max_number_task_in_package, package_with_max_number_task, number_task_to_steal = 0;
	bool load_balance_needed = true;
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("A package should have %d or %d tasks\n", NT/number_gpu, NT/number_gpu+1); }
	/* Selecting the smallest and biggest package */
	while (load_balance_needed == true) { 
		a->temp_pointer_1 = a->first_link;
		min_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
		max_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
		package_with_min_number_task = 0;
		package_with_max_number_task = 0;
		i = 0;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		while (a->temp_pointer_1 != NULL) {
			i++;
			if (min_number_task_in_package > a->temp_pointer_1->nb_task_in_sub_list) {
				min_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
				package_with_min_number_task = i;
			}
			if (max_number_task_in_package < a->temp_pointer_1->nb_task_in_sub_list) {
				max_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
				package_with_max_number_task = i;
			}
			a->temp_pointer_1 = a->temp_pointer_1->next;
		}
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("min et max : %d et %d\n",min_number_task_in_package, max_number_task_in_package); }
		/* Stealing as much task from the last tasks of the biggest packages */
		if (package_with_min_number_task == package_with_max_number_task || min_number_task_in_package ==  max_number_task_in_package-1) {
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("All packages have the same number of tasks +/- 1 task\n"); }
			load_balance_needed = false;
		}
		else {
			a->temp_pointer_1 = a->first_link;
			for (i = 0; i < package_with_min_number_task; i++) {
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_2 = a->first_link;
			for (i = 0; i < package_with_max_number_task; i++) {
				a->temp_pointer_2 = a->temp_pointer_2->next;
			}
			printf("NT = %d, NT/number gpu = %d\n", NT, NT/number_gpu);
			if ((NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list == 0) { number_task_to_steal = 1; }
			else if (a->temp_pointer_2->nb_task_in_sub_list - ((NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list) >= NT/number_gpu) {
				number_task_to_steal = (NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list;
			}
			else {
				number_task_to_steal = a->temp_pointer_2->nb_task_in_sub_list - NT/number_gpu;
				printf ("NT/number = %d pointer 2 task = %d\n", NT/number_gpu, a->temp_pointer_2->nb_task_in_sub_list);
			}
			printf("to steal = %d\n",number_task_to_steal);
			for (i = 0; i < number_task_to_steal; i++) {
				merge_task_and_package(a->temp_pointer_1, starpu_task_list_pop_back(&a->temp_pointer_2->sub_list));
				a->temp_pointer_2->nb_task_in_sub_list--;
			}
		}
	}
}

void print_order_in_file_hfp (struct paquets *p)
{
	p->temp_pointer_1 = p->first_link;
	FILE *f = fopen("Output_maxime/Task_order_HFP.txt", "w");
	struct starpu_task *task;
	while (p->temp_pointer_1 != NULL) 
	{
		for (task = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task != starpu_task_list_end(&p->temp_pointer_1->sub_list); task = starpu_task_list_next(task)) 
		{
			fprintf(f, "%p\n",task);
		}
		fprintf(f, "End\n");
		p->temp_pointer_1 = p->temp_pointer_1->next;
	}
	fclose(f);
}

void hmetis(int nb_package_to_build, struct paquets *p, struct starpu_task_list *l, int nb_gpu, starpu_ssize_t GPU_RAM_M) 
{
	FILE *f = fopen("Output_maxime/input_hMETIS.txt", "w+");
	//TODO a corriger
	fprintf(f,"blanckk\n");
	NT = 0;
	int i = 0; struct starpu_task *task_1; struct starpu_task *task_2; struct starpu_task *task_3; int NT = 0; bool first_write_on_line = true; bool already_counted = false;
	int index_task_1 = 1; int index_task_2 = 0; int number_hyperedge = 0; int j = 0; int k = 0; int m = 0;
	for (task_1 = starpu_task_list_begin(l); task_1 != starpu_task_list_end(l); task_1 = starpu_task_list_next(task_1))
	{
		printf("Tâche 1 %p\n", task_1);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task_1); i++) 
		{
			task_3 = starpu_task_list_begin(l);
			already_counted = false;
			//~ printf("index task 1 = %d\n", index_task_1);
			for (k = 1; k < index_task_1; k++) 
			{
				for (m = 0; m < STARPU_TASK_GET_NBUFFERS(task_3); m++)
				{
					if (STARPU_TASK_GET_HANDLE(task_1, i) == STARPU_TASK_GET_HANDLE(task_3, m))
					{
						already_counted = true;
						break;
					}
				}
				if (already_counted == true)
				{
					break;
				}
				task_3 = starpu_task_list_next(task_3);
			}
			if (already_counted == false) 
			{	
				first_write_on_line = true;
				index_task_2 = index_task_1 + 1;
				for (task_2 = starpu_task_list_next(task_1); task_2 != starpu_task_list_end(l); task_2 = starpu_task_list_next(task_2))
				{
					//~ printf("Tâche 2 %p\n", task_2);
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task_2); j++)
					{
						if (STARPU_TASK_GET_HANDLE(task_1, i) == STARPU_TASK_GET_HANDLE(task_2, j))
						{
							if (first_write_on_line == true) 
							{
								first_write_on_line = false;			
								fprintf(f, "%d %d", index_task_1, index_task_2);
								number_hyperedge++;
							}
							else 
							{
								fprintf(f, " %d", index_task_2);
							}
						}
					}
					index_task_2++;
				}
				if (first_write_on_line == false) { fprintf(f, "\n"); }
			}
		}
		index_task_1++;
		NT++;
	}
	/* Printing expected time of each task */
	for (task_1 = starpu_task_list_begin(l); task_1 != starpu_task_list_end(l); task_1 = starpu_task_list_next(task_1))
	{
		fprintf(f, "%f\n", starpu_task_expected_length(task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0));			
	}
	/* Printing informations for hMETIS on the first line */
	rewind(f);
	fprintf(f, "%d %d 10\n", number_hyperedge, NT); /* Number of hyperedges, number of task, 10 for weighted vertices but non weighted */ 
	fclose(f);	
	//TODO : remplacer le 3 par nb_gpu ici
	int cr = system("../these_gonthier_maxime/hMETIS/hmetis-1.5-linux/shmetis Output_maxime/input_hMETIS.txt 3 2");
	if (cr != 0) 
	{
        printf("Impossible de lancer la commande\n");
        exit(0);
    }
    starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
    for (i = 1; i < nb_gpu; i++)
    {
		HFP_insertion(p);
		starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
		printf("new link\n");
	}
	p->first_link = p->temp_pointer_1;
	char str[2];
	sprintf(str, "%d", nb_gpu);
	int size = strlen("Output_maxime/input_hMETIS.txt.part.") + strlen(str);
	char *path2 = (char *)malloc(size);
	strcpy(path2, "Output_maxime/input_hMETIS.txt.part.");
	strcat(path2, str);
	FILE *f_2 = fopen(path2, "r");
	int number; int error;
	for (i = 0; i < NT; i++) 
	{
		error = fscanf(f_2, "%d", &number);
		if (error == 0) 
		{
			printf("error fscanf in hMETIS\n"); exit(0);
		}
		printf("%d\n", number);
		p->temp_pointer_1 = p->first_link;
		for (j = 0; j < number; j++) 
		{
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
		starpu_task_list_push_back(&p->temp_pointer_1->sub_list, starpu_task_list_pop_front(l));
		p->temp_pointer_1->nb_task_in_sub_list++;
	}
	fclose(f_2);
	print_packages_in_terminal(p, 0);
	/* Apply HFP on each package if we have the right option */
	if (starpu_get_env_number_default("HMETIS",0) == 2)
	{
		p->temp_pointer_1 = p->first_link;
		for (i = 0; i < nb_gpu; i++) 
		{
			hierarchical_fair_packing(p->temp_pointer_1->sub_list, p->temp_pointer_1->nb_task_in_sub_list, GPU_RAM_M);
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
	}
}

int get_number_GPU()
{
	int return_value = 0;
	if (starpu_get_env_number_default("MULTIGPU",0) != 0) 
	{
		unsigned nnodes = starpu_memory_nodes_get_count();
		for (int i = 0; i < nnodes; i++)
		{
			if (starpu_node_get_kind(i) == STARPU_CUDA_RAM)
			{
				return_value++;
			} 
		}
	}
	else { return_value = 1; }
	return return_value;
}

/* The function that sort the tasks in packages */
static struct starpu_task *HFP_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Pull task\n"); }
	struct HFP_sched_data *data = component->data;

	/* Variables used to calculate, navigate through a loop or other things */
	int i = 0; int j = 0; int tab_runner = 0; int do_not_add_more = 0; int index_head_1 = 0; int index_head_2 = 0; int i_bis = 0; int j_bis = 0; int common_data_last_package_i2_j = 0; int common_data_last_package_i1_j = 0; int common_data_last_package_i_j1 = 0; int common_data_last_package_i_j2 = 0; int NB_TOTAL_DONNEES = 0;
	int min_nb_task_in_sub_list = 0; int nb_min_task_packages = 0; int temp_nb_min_task_packages = 0;
	struct starpu_task *task1 = NULL; struct starpu_task *temp_task_1 = NULL; struct starpu_task *temp_task_2 = NULL;	 
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
	int number_of_package_to_build = 0;
	
	/* Getting the number of GPUs */
	number_of_package_to_build = get_number_GPU(); 
	
	/* Here we calculate the size of the RAM of the GPU. We allow our packages to have half of this size */
	starpu_ssize_t GPU_RAM_M = 0;
	//~ STARPU_ASSERT(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component)); /* If we have only one GPU uncomment this */
	GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Max size of a package = %ld\n",GPU_RAM_M); }
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	/* If one or more task have been refused */
	data->p->temp_pointer_1 = data->p->first_link;
	if (data->p->temp_pointer_1->next != NULL) { 
		for (i = 0; i < number_of_package_to_build; i++) {
			if (to == component->children[i]) {
				break;
			}
			else {
				data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
			}
		}
	}
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list)) {
		task1 = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task from fifo refused list on gpu %p\n",task1, to); }
		return task1;
	}	
	/* If the linked list is empty, we can pull more tasks */
	if (is_empty(data->p->first_link) == true) {
		if (!starpu_task_list_empty(&data->sched_list)) { /* Si la liste initiale (sched_list) n'est pas vide, ce sont des tâches non traitées */
			time_t start, end; time(&start);
			EXPECTED_TIME = 0;
			
			if (starpu_get_env_number_default("HMETIS",0) != 0) 
			{
				hmetis(number_of_package_to_build, data->p, &data->sched_list, number_of_package_to_build, GPU_RAM_M);
				task1 = get_task_to_return(component, to, data->p, number_of_package_to_build);
				STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task from hmetis on gpu %p\n",task1, to); }
				return task1;
			}
			//~ else
			//~ {
			
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list)) {
				task1 = starpu_task_list_pop_front(&data->sched_list);
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Tâche %p\n",task1); }
				//~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task1); i++) {
					//~ printf("%p\n",STARPU_TASK_GET_HANDLE(task1,i));
				//~ }
				if (starpu_get_env_number_default("MULTIGPU",0) != 0) { EXPECTED_TIME += starpu_task_expected_length(task1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);	}					
				nb_pop++;
				starpu_task_list_push_back(&data->popped_task_list,task1);
			} 	
			NT = nb_pop;
			N = sqrt(NT);
			data->p->NP = NT;
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("NT = %d\n",NT); }
				
			temp_task_1  = starpu_task_list_begin(&data->popped_task_list);
			const char* appli = starpu_task_get_name(temp_task_1);	
			//~ printf("appli: %s\n",appli);
			data->p->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->p->temp_pointer_1->package_data[0]));
			
			/* One task == one link in the linked list */
			do_not_add_more = nb_pop - 1;
			for (temp_task_1  = starpu_task_list_begin(&data->popped_task_list); temp_task_1 != starpu_task_list_end(&data->popped_task_list); temp_task_1  = temp_task_2) {
				temp_task_2 = starpu_task_list_next(temp_task_1);
				temp_task_1 = starpu_task_list_pop_front(&data->popped_task_list);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(temp_task_1); i++) {
					data->p->temp_pointer_1->package_data[i] = STARPU_TASK_GET_HANDLE(temp_task_1,i);
				}
				if (starpu_get_env_number_default("MULTIGPU",0) != 0) { data->p->temp_pointer_1->expected_time = starpu_task_expected_length(temp_task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0); }
				data->p->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(temp_task_1);
				NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(temp_task_1);
				/* We sort our datas in the packages */
				qsort(data->p->temp_pointer_1->package_data,data->p->temp_pointer_1->package_nb_data,sizeof(data->p->temp_pointer_1->package_data[0]),HFP_pointeurComparator);
				/* Pushing the task and the number of the package in the package*/
				starpu_task_list_push_back(&data->p->temp_pointer_1->sub_list,temp_task_1);
				data->p->temp_pointer_1->index_package = link_index;
				/* Initialization of the lists last_packages */
				data->p->temp_pointer_1->split_last_ij = 0;
				
				data->p->temp_pointer_1->total_nb_data_package = STARPU_TASK_GET_NBUFFERS(temp_task_1);
				
				link_index++;
				//~ data->p->temp_pointer_1->nb_task_in_sub_list ++;
				data->p->temp_pointer_1->nb_task_in_sub_list=1;
				
				if(do_not_add_more != 0) { HFP_insertion(data->p); data->p->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->p->temp_pointer_1->package_data[0])); }
				do_not_add_more--;
			}
			data->p->first_link = data->p->temp_pointer_1;				
			
			/* Code to print all the data of all the packages */
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) {
				//~ link_index = 0;
				//~ printf("Initialement on a : \n");
				//~ while (data->p->temp_pointer_1 != NULL) {
					//~ for (i = 0; i < 3; i++) {
						//~ printf("La donnée %p est dans la tâche %p du paquet numéro %d\n",data->p->temp_pointer_1->package_data[i],temp_task_1  = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list),link_index);
					//~ }
					//~ link_index++;
					//~ data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				//~ } printf("NULL\n");
				//~ data->p->temp_pointer_1 = data->p->first_link;
			//~ }
			
			data->p->temp_pointer_2 = data->p->first_link;
			index_head_2++;
			
			/* Matrix used to store all the common data weights between packages */
			int coordinate_visualization_matrix_size = N;
			int coordinate_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			int coordinate_order_visualization_matrix[coordinate_visualization_matrix_size][coordinate_visualization_matrix_size];
			for (i_bis = 0; i_bis < N; i_bis++) {
				for (j_bis = 0; j_bis < N; j_bis++) {
					coordinate_visualization_matrix[j_bis][i_bis] = 0;
					coordinate_order_visualization_matrix[j_bis][i_bis] = 0;
				}
			}
			
			if (starpu_get_env_number_default("PRINTF",0) == 1) { init_visualisation_tache_matrice_format_tex(); }
			
			/* THE while loop. Stop when no more packaging are possible */
			while (packaging_impossible == 0) {
				/* algo 3's goto */
				algo3:
				nb_of_loop++;
				packaging_impossible = 1;
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("############# Itération numéro : %d #############\n",nb_of_loop); }
								
				/* Variables we need to reinitialize for a new iteration */
				data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_2 = data->p->first_link; index_head_1 = 0; index_head_2 = 1; link_index = 0; tab_runner = 0; nb_min_task_packages = 0;
				min_nb_task_in_sub_list = 0; nb_common_data = 0; weight_two_packages = 0; max_value_common_data_matrix = 0; long int matrice_donnees_commune[nb_pop][nb_pop];
				min_nb_task_in_sub_list = data->p->temp_pointer_1->nb_task_in_sub_list; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
				
				/* For algorithm Algo 4 we need a symmetric matrix and the minimal packages */
				 
					/* First we get the number of packages that have the minimal number of tasks */
					for (data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_1 != NULL; data->p->temp_pointer_1 = data->p->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list > data->p->temp_pointer_1->nb_task_in_sub_list) { min_nb_task_in_sub_list = data->p->temp_pointer_1->nb_task_in_sub_list; } }
					for (data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_1 != NULL; data->p->temp_pointer_1 = data->p->temp_pointer_1->next) {
						if (min_nb_task_in_sub_list == data->p->temp_pointer_1->nb_task_in_sub_list) { nb_min_task_packages++; } }
					//~ if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list); }
					/* Then we create the common data matrix */
					//~ printf("nb pop = %d\n",nb_pop);
					for (data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_1 != NULL; data->p->temp_pointer_1 = data->p->temp_pointer_1->next) {
						for (data->p->temp_pointer_2 = data->p->temp_pointer_1->next; data->p->temp_pointer_2 != NULL; data->p->temp_pointer_2 = data->p->temp_pointer_2->next) {
							for (i = 0; i < data->p->temp_pointer_1->package_nb_data; i++) {
								for (j = 0; j < data->p->temp_pointer_2->package_nb_data; j++) {
									if ((data->p->temp_pointer_1->package_data[i] == data->p->temp_pointer_2->package_data[j])) {
										matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(data->p->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->p->temp_pointer_1->package_data[i]);
										matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(data->p->temp_pointer_2->package_data[j]) + starpu_data_get_size(data->p->temp_pointer_1->package_data[i]);
									} } } index_head_2++; } index_head_1++; index_head_2 = index_head_1 + 1; }
				/* Code to print the common data matrix */
				//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Common data matrix : \n"); for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { printf (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); }}
				
				/* Getting the number of package that have data in commons */
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						if (matrice_donnees_commune[i][j] != 0) { nb_common_data++; } } }
				
				/* Getting back to the beginning of the linked list */
				data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_2 = data->p->first_link;
				
					/* ALGO 4' ou 5 */
					i_bis = 0; j_bis = 0; 
					temp_nb_min_task_packages = nb_min_task_packages;
				debut_while:
					data->p->temp_pointer_1 = data->p->first_link;
					data->p->temp_pointer_2 = data->p->first_link;
					max_value_common_data_matrix = 0;
					if (GPU_limit_switch == 1) {
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						if (data->p->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (data->p->temp_pointer_2 = data->p->first_link; data->p->temp_pointer_2 != NULL; data->p->temp_pointer_2 = data->p->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < data->p->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->p->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->p->temp_pointer_1->package_nb_data; j++) {
										if (data->p->temp_pointer_2->package_data[i] == data->p->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_2->package_data[i]); } } 
									if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= GPU_RAM_M)) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							data->p->temp_pointer_1=data->p->temp_pointer_1->next;
							j_bis = 0; }
				data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_2 = data->p->first_link;
				}
				/* Else, we are using algo 5, so we don't check the max weight */
				else {
					for (i_bis = 0; i_bis < nb_pop; i_bis++) {
						if (data->p->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) { //Si on est sur un paquet de taille minimale
							for (data->p->temp_pointer_2 = data->p->first_link; data->p->temp_pointer_2 != NULL; data->p->temp_pointer_2 = data->p->temp_pointer_2->next) {
								if (i_bis != j_bis) {
									weight_two_packages = 0;
									for (i = 0; i < data->p->temp_pointer_1->package_nb_data; i++) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_1->package_data[i]); } 
									for (i = 0; i < data->p->temp_pointer_2->package_nb_data; i++) {
										bool_data_common = 0;
										for (j = 0; j < data->p->temp_pointer_1->package_nb_data; j++) {
										if (data->p->temp_pointer_2->package_data[i] == data->p->temp_pointer_1->package_data[j]) { bool_data_common = 1; } }
										if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_2->package_data[i]); } } 
									if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) { 
										max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis]; } 
							} j_bis++; } tab_runner++; } 
							data->p->temp_pointer_1=data->p->temp_pointer_1->next;
							j_bis = 0; }
				data->p->temp_pointer_1 = data->p->first_link; data->p->temp_pointer_2 = data->p->first_link;
				}		
				i_bis = 0; j_bis = 0; i = 0; j = 0;
				for (i = 0; i < nb_pop; i++) {
					if (data->p->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list) {
						for (j = 0; j < nb_pop; j++) {
							weight_two_packages = 0;  weight_package_i = 0;  weight_package_j = 0;
							for (i_bis = 0; i_bis < data->p->temp_pointer_1->package_nb_data; i_bis++) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_1->package_data[i_bis]); } weight_package_i = weight_two_packages;
							for (i_bis = 0; i_bis < data->p->temp_pointer_2->package_nb_data; i_bis++) { bool_data_common = 0;
								for (j_bis = 0; j_bis < data->p->temp_pointer_1->package_nb_data; j_bis++) { if (data->p->temp_pointer_2->package_data[i_bis] == data->p->temp_pointer_1->package_data[j_bis]) { bool_data_common = 1; } }
								if (bool_data_common != 1) { weight_two_packages += starpu_data_get_size(data->p->temp_pointer_2->package_data[i_bis]); } 
								weight_package_j += starpu_data_get_size(data->p->temp_pointer_2->package_data[i_bis]); }							
							if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j && max_value_common_data_matrix != 0) {
								if ((weight_two_packages <= GPU_RAM_M) || (GPU_limit_switch == 0)) {
								/* Merge */
								packaging_impossible = 0;
								if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On va merge le paquet %d et le paquet %d\n",i,j); }
								
								data->p->NP--;
								
								if (data->p->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list) { temp_nb_min_task_packages--; }
								
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;}
								for (j_bis = 0; j_bis < nb_pop; j_bis++) { matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;}
								nb_common_data--;
								
								if (starpu_get_env_number_default("ORDER_U",0) == 1) {
									//~ printf("I a %d taches et %d données\n",data->p->temp_pointer_1->nb_task_in_sub_list,data->p->temp_pointer_1->package_nb_data);
									//~ printf("J a %d taches et %d données\n",data->p->temp_pointer_2->nb_task_in_sub_list,data->p->temp_pointer_2->package_nb_data);
									//~ printf("Split de last ij de I = %d\n",data->p->temp_pointer_1->split_last_ij);
									//~ printf("Split de last ij de J = %d\n",data->p->temp_pointer_2->split_last_ij); 
									//~ printf("Poids paquet i : %li / Poids paquet j : %li / M : %li\n",weight_package_i,weight_package_j,GPU_RAM_M);
									if (data->p->temp_pointer_1->nb_task_in_sub_list == 1 && data->p->temp_pointer_2->nb_task_in_sub_list == 1) {
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I = 1 et J = 1\n"); }
									}
									else if (weight_package_i > GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I > PU_RAM et J <= PU_RAM\n"); }
										common_data_last_package_i1_j = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 1, 0, false,GPU_RAM_M);					
										common_data_last_package_i2_j = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 2, 0, false,GPU_RAM_M);					
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("\ni1j = %d / i2j = %d\n",common_data_last_package_i1_j,common_data_last_package_i2_j); }
										if (common_data_last_package_i1_j > common_data_last_package_i2_j) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I\n"); }
											data->p->temp_pointer_1 = HFP_reverse_sub_list(data->p->temp_pointer_1);
										}
										else { 
											//~ printf("Pas de switch\n"); 
											}
									}
									else if (weight_package_i <= GPU_RAM_M && weight_package_j > GPU_RAM_M) {
										//~ printf("I <= PU_RAM et J > PU_RAM\n");
										common_data_last_package_i_j1 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 0, 1, false,GPU_RAM_M);					
										common_data_last_package_i_j2 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 0, 2, false,GPU_RAM_M);					
										//~ printf("\nij1 = %d / ij2 = %d\n",common_data_last_package_i_j1,common_data_last_package_i_j2);
										if (common_data_last_package_i_j2 > common_data_last_package_i_j1) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET J\n"); }
											data->p->temp_pointer_2 = HFP_reverse_sub_list(data->p->temp_pointer_2);
										}
										else { 
											//~ printf("Pas de switch\n"); 
											}
									}
									else {
										if (weight_package_i > GPU_RAM_M && weight_package_j > GPU_RAM_M) {
											//~ printf("I > PU_RAM et J > PU_RAM\n");
											common_data_last_package_i1_j1 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 1, 1, false,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 1, 2, false,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 2, 1, false,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 2, 2, false,GPU_RAM_M);
										}
										else if (weight_package_i <= GPU_RAM_M && weight_package_j <= GPU_RAM_M) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("I <= PU_RAM et J <= PU_RAM\n"); }
											//~ printf("Tâches de I\n");
											//~ temp_task_1 = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list);
											//~ printf("%p:",temp_task_1);
											//~ printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));
											//~ while (starpu_task_list_next(temp_task_1) != NULL) { temp_task_1 = starpu_task_list_next(temp_task_1); printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));}
											//~ printf("Tâches de J\n");
											//~ temp_task_1 = starpu_task_list_begin(&data->p->temp_pointer_2->sub_list);
											//~ printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));
											//~ while (starpu_task_list_next(temp_task_1) != NULL) { temp_task_1 = starpu_task_list_next(temp_task_1); printf("%p:",temp_task_1); printf(" %p %p %p\n",STARPU_TASK_GET_HANDLE(temp_task_1,0),STARPU_TASK_GET_HANDLE(temp_task_1,1),STARPU_TASK_GET_HANDLE(temp_task_1,2));}
											
											//~ printf("I a %d taches et %d données\n",data->p->temp_pointer_1->nb_task_in_sub_list,data->p->temp_pointer_1->package_nb_data);
											//~ printf("J a %d taches et %d données\n",data->p->temp_pointer_2->nb_task_in_sub_list,data->p->temp_pointer_2->package_nb_data);
											//~ printf("Split de last ij de I = %d\n",data->p->temp_pointer_1->split_last_ij);
											//~ printf("Split de last ij de J = %d\n",data->p->temp_pointer_2->split_last_ij);
											common_data_last_package_i1_j1 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 1, 1, true,GPU_RAM_M);					
											common_data_last_package_i1_j2 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 1, 2, true,GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 2, 1, true,GPU_RAM_M);					
											common_data_last_package_i2_j2 = get_common_data_last_package(data->p->temp_pointer_1, data->p->temp_pointer_2, 2, 2, true,GPU_RAM_M);
										}
										else { printf("Erreur dans ordre U, aucun cas choisi\n"); exit(0); }
										//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("i1j1 = %ld / i1j2 = %ld / i2j1 = %ld / i2j2 = %ld\n",common_data_last_package_i1_j1,common_data_last_package_i1_j2,common_data_last_package_i2_j1,common_data_last_package_i2_j2); }
										max_common_data_last_package = common_data_last_package_i2_j1;
										if (max_common_data_last_package < common_data_last_package_i1_j1) { max_common_data_last_package = common_data_last_package_i1_j1; }
										if (max_common_data_last_package < common_data_last_package_i1_j2) { max_common_data_last_package = common_data_last_package_i1_j2; }
										if (max_common_data_last_package < common_data_last_package_i2_j2) { max_common_data_last_package = common_data_last_package_i2_j2; }
										if (max_common_data_last_package == common_data_last_package_i2_j1) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Pas de switch\n"); }
										}								
										else if (max_common_data_last_package == common_data_last_package_i1_j2) {
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I ET J\n");	}
											data->p->temp_pointer_1 = HFP_reverse_sub_list(data->p->temp_pointer_1);									
											data->p->temp_pointer_2 = HFP_reverse_sub_list(data->p->temp_pointer_2);
										}
										else if (max_common_data_last_package == common_data_last_package_i2_j2) {
											//~ printf("Tâches du paquet %d:\n",data->p->temp_pointer_2->index_package);
											//~ for (temp_task_1  = starpu_task_list_begin(&data->p->temp_pointer_2->sub_list); temp_task_1 != starpu_task_list_end(&data->p->temp_pointer_2->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												//~ printf("%p / ",temp_task_1); } 
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET J\n"); }
											data->p->temp_pointer_2 = HFP_reverse_sub_list(data->p->temp_pointer_2);	
											//~ printf("Tâches du paquet %d:\n",data->p->temp_pointer_2->index_package);
											//~ for (temp_task_1  = starpu_task_list_begin(&data->p->temp_pointer_2->sub_list); temp_task_1 != starpu_task_list_end(&data->p->temp_pointer_2->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												//~ printf("%p /",temp_task_1); }								
										}
										else { /* max_common_data_last_package == common_data_last_package_i1_j1 */
											//~ printf("Tâches du paquet %d:\n",data->p->temp_pointer_1->index_package);
											//~ for (temp_task_1  = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												//~ printf("%p / ",temp_task_1); } 
											//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("SWITCH PAQUET I\n"); }
											//~ printf("Tâches du paquet %d:\n",data->p->temp_pointer_1->index_package);
											//~ for (temp_task_1  = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
												//~ printf("%p / ",temp_task_1); } 
											data->p->temp_pointer_1 = HFP_reverse_sub_list(data->p->temp_pointer_1);									
										}		
									}							
									//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Fin de l'ordre U sans doublons\n"); }
								}
								
								data->p->temp_pointer_1->split_last_ij = data->p->temp_pointer_1->nb_task_in_sub_list;
								while (!starpu_task_list_empty(&data->p->temp_pointer_2->sub_list)) {
								starpu_task_list_push_back(&data->p->temp_pointer_1->sub_list,starpu_task_list_pop_front(&data->p->temp_pointer_2->sub_list)); 
								data->p->temp_pointer_1->nb_task_in_sub_list ++; }
								i_bis = 0; j_bis = 0; tab_runner = 0;
								starpu_data_handle_t *temp_data_tab = malloc((data->p->temp_pointer_1->package_nb_data + data->p->temp_pointer_2->package_nb_data) * sizeof(data->p->temp_pointer_1->package_data[0]));
								while (i_bis < data->p->temp_pointer_1->package_nb_data && j_bis < data->p->temp_pointer_2->package_nb_data) {
									if (data->p->temp_pointer_1->package_data[i_bis] <= data->p->temp_pointer_2->package_data[j_bis]) {
										temp_data_tab[tab_runner] = data->p->temp_pointer_1->package_data[i_bis];
										i_bis++; }
									else {
										temp_data_tab[tab_runner] = data->p->temp_pointer_2->package_data[j_bis];
										j_bis++; }
									tab_runner++;
								}
								while (i_bis < data->p->temp_pointer_1->package_nb_data) { temp_data_tab[tab_runner] = data->p->temp_pointer_1->package_data[i_bis]; i_bis++; tab_runner++; }
								while (j_bis < data->p->temp_pointer_2->package_nb_data) { temp_data_tab[tab_runner] = data->p->temp_pointer_2->package_data[j_bis]; j_bis++; tab_runner++; }
								for (i_bis = 0; i_bis < (data->p->temp_pointer_1->package_nb_data + data->p->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1]) {
										temp_data_tab[i_bis] = 0;
										nb_duplicate_data++; } }
								data->p->temp_pointer_1->package_data = malloc((data->p->temp_pointer_1->package_nb_data + data->p->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
								j_bis = 0;
								for (i_bis = 0; i_bis < (data->p->temp_pointer_1->package_nb_data + data->p->temp_pointer_2->package_nb_data); i_bis++) {
									if (temp_data_tab[i_bis] != 0) { data->p->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++; } }
								data->p->temp_pointer_1->package_nb_data = data->p->temp_pointer_2->package_nb_data + data->p->temp_pointer_1->package_nb_data - nb_duplicate_data;
								
								data->p->temp_pointer_1->total_nb_data_package += data->p->temp_pointer_2->total_nb_data_package;
								data->p->temp_pointer_1->expected_time += data->p->temp_pointer_2->expected_time;
								
								data->p->temp_pointer_2->package_nb_data = 0;
								nb_duplicate_data = 0;
								data->p->temp_pointer_2->nb_task_in_sub_list = 0;
							temp_nb_min_task_packages--;
							if(data->p->NP == number_of_package_to_build) { goto break_merging; }
							if (temp_nb_min_task_packages > 1) {
								goto debut_while; 
							}
							else { j = nb_pop; i = nb_pop; }
							} }
							data->p->temp_pointer_2=data->p->temp_pointer_2->next;
						}
					}
					data->p->temp_pointer_1=data->p->temp_pointer_1->next; data->p->temp_pointer_2=data->p->first_link;
				}			
				
				break_merging:
				
				data->p->temp_pointer_1 = data->p->first_link;
				data->p->temp_pointer_1 = HFP_delete_link(data->p);
				tab_runner = 0;
				
				/* Code to get the coordinates of each data in the order in wich tasks get out of pull_task */
					while (data->p->temp_pointer_1 != NULL) {
					if ((strcmp(appli,"starpu_sgemm_gemm") == 0) && (starpu_get_env_number_default("PRINTF",0) == 1)) {
						for (temp_task_1 = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); temp_task_1 != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); temp_task_1  = starpu_task_list_next(temp_task_1)) {
							starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(temp_task_1,2),2,temp_tab_coordinates);
							coordinate_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = NT - data->p->temp_pointer_1->index_package - 1;
							coordinate_order_visualization_matrix[temp_tab_coordinates[0]][temp_tab_coordinates[1]] = tab_runner;
							tab_runner++;	
							temp_tab_coordinates[0] = 0; temp_tab_coordinates[1] = 0;
						}
					}			
						link_index++;
						data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
					} 
					if (starpu_get_env_number_default("PRINTF",0) == 1) { visualisation_tache_matrice_format_tex(coordinate_visualization_matrix,coordinate_order_visualization_matrix,nb_of_loop,link_index); }
			 
			/* Checking if we have the right number of packages. if MULTIGPU is equal to 0 we want only one package. if it is equal to 1 we want |GPU| packages */
			if (link_index == number_of_package_to_build) { goto end_algo3; }
				
			for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
			/* Reset nb_pop for the matrix initialisation */
			nb_pop = link_index;
			/* If we have only one package we don't have to do more packages */			
			if (nb_pop == 1) {  packaging_impossible = 1; }
		} /* End of while (packaging_impossible == 0) { */
		/* We are in algorithm 3, we remove the size limit of a package */
		GPU_limit_switch = 0; goto algo3;
		
		end_algo3:
				
		data->p->temp_pointer_1 = data->p->first_link;	
		
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("After first execution of HFP we have ---\n"); print_packages_in_terminal(data->p, nb_of_loop); }
		
		/* Task stealing. Only in cases of multigpu */
		if (starpu_get_env_number_default("MULTIGPU",0) == 2 || starpu_get_env_number_default("MULTIGPU",0) == 3) {
			load_balance(data->p, number_of_package_to_build);
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("After load balance we have ---\n"); print_packages_in_terminal(data->p, nb_of_loop); }
		}
		/* Task stealing with expected time */
		if (starpu_get_env_number_default("MULTIGPU",0) == 4 || starpu_get_env_number_default("MULTIGPU",0) == 5) {
			load_balance_expected_time(data->p, number_of_package_to_build);
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("After load balance we have ---\n"); print_packages_in_terminal(data->p, nb_of_loop); }
		}
		/* Re-apply HFP on each package. 
		 * Once task stealing is done we need to re-apply HFP. For this I use an other instance of HFP_sched_data.
		 * It is in another function, if it work we can also put the packing above in it.
		 * Only with MULTIGPU = 2 because if we don't do load balance there is no point in re-applying HFP.
		 */
		 if (starpu_get_env_number_default("MULTIGPU",0) == 3 || starpu_get_env_number_default("MULTIGPU",0) == 5) {
			 data->p->temp_pointer_1 = data->p->first_link;
			 while (data->p->temp_pointer_1 != NULL) { 
				data->p->temp_pointer_1->sub_list = hierarchical_fair_packing(data->p->temp_pointer_1->sub_list, data->p->temp_pointer_1->nb_task_in_sub_list, GPU_RAM_M);
				data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
			}
			 if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("After execution of HFP on each package we have ---\n"); print_packages_in_terminal(data->p, nb_of_loop); }
		 }
		
		if (starpu_get_env_number_default("PRINTF",0) == 1) { end_visualisation_tache_matrice_format_tex(); }
		
		/* Belady */
		if (starpu_get_env_number_default("BELADY",0) == 1) {
			get_ordre_utilisation_donnee(data->p, NB_TOTAL_DONNEES, number_of_package_to_build);
		}
		
		/* If you want to get the sum of weight of all different data. Only works if you have only one package */
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { get_weight_all_different_data(data->p->first_link, GPU_RAM_M); }
		
		/* We prefetch data for each task for modular-heft-HFP */
		if (starpu_get_env_number_default("MODULAR_HEFT_HFP_MODE",0) != 0) {
			prefetch_each_task(data->p, component);
		}
		
		time(&end); int time_taken = end - start; if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Temps d'exec : %d secondes\n",time_taken); }
		FILE *f_time = fopen("Output_maxime/Execution_time_raw.txt","a");
		fprintf(f_time,"%d\n",time_taken);
		fclose(f_time);
		
		//~ end_hfp:
		
		/* Printing in a file the order produced by HFP. If we use modular-heft-HFP, we can compare this order with the one done by modular-heft */
		if (starpu_get_env_number_default("PRINTF",0) == 1)
		{
			printf("printing order in file\n");
			print_order_in_file_hfp(data->p);
			FILE *f = fopen("Output_maxime/Task_order_effective.txt", "w"); /* Just to empty it before */
			fclose(f);
		}
	
		/* We pop the first task of the first package. We look at the current GPU if we are in multi GPU in order to assign the first task of the corresponding package */
		task1 = get_task_to_return(component, to, data->p, number_of_package_to_build);
		}
		/* Else de if (!starpu_task_list_empty(&data->sched_list)), il faut donc return une tâche */
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			if (starpu_get_env_number_default("PRINTF",0) == 1 && task1 != NULL) { printf("Task %p is getting out of pull_task from gpu %p\n",task1,to); }
			return task1; /* Renvoie nil içi */
	} /* End of if ((data->p->temp_pointer_1->next == NULL) && (starpu_task_list_empty(&data->p->temp_pointer_1->sub_list))) { */
		task1 = get_task_to_return(component, to, data->p, number_of_package_to_build);
		if (starpu_get_env_number_default("PRINTF",0) == 1 && task1 != NULL) { printf("Task %p is getting out of pull_task from gpu %p\n",task1,to); }
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return task1;
	printf("Ah return NULL :(\n");
	return NULL;		
}

static int HFP_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Début HFP_can_push\n"); }
	struct HFP_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task); }
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		//~ starpu_task_list_push_back(&data->list_if_fifo_full, task);
		data->p->temp_pointer_1 = data->p->first_link;
		int nb_gpu = get_number_GPU();
		if (data->p->temp_pointer_1->next == NULL) { starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task); }
		else {
			//A corriger. En fait il faut push back dans une fifo a part puis pop back dans cette fifo dans pull task
			//Ici le pb c'est si plusieurs taches se font refusé
			for (int i = 0; i < nb_gpu; i++) {
				if (to == component->children[i]) {
					break;
				}
				else {
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				}
			}
			starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
		}
		//~ task1 = get_task_to_return(component, to, data->p); 
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task from fifo on gpu %p\n", task1,
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		/* Can I uncomment this part ? */
		//~ {
			//~ if (didwork)
				//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "pushed some tasks to %p\n", to); }
			//~ else
				//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "I didn't have anything for %p\n", to); }
		//~ }
	}

	/* There is room now */
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Fin de can push\n"); }
	return didwork || starpu_sched_component_can_push(component, to);
}

static int HFP_can_pull(struct starpu_sched_component * component)
{
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Can pull\n"); }
	//~ struct HFP_sched_data *data = component->data;
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_HFP_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Create\n"); }
	srandom(time(0)); /* If we need a random selection */
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "HFP");
	
	struct HFP_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	struct paquets *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	//~ starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);
 
	//~ my_data->next = NULL;
	//~ data->temp_pointer_1 = my_data;
	
	//~ struct my_list *my_data = malloc(sizeof(*my_data));
	//~ struct paquets *paquets_data = malloc(sizeof(*paquets_data));
	//~ starpu_task_list_init(&my_data->sub_list);
	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
	data->p = paquets_data;
	data->p->temp_pointer_1->nb_task_in_sub_list = 0;
	data->p->temp_pointer_1->expected_time_pulled_out = 0;
	
	component->data = data;
	component->push_task = HFP_push_task;
	component->pull_task = HFP_pull_task;
	component->can_push = HFP_can_push;
	component->can_pull = HFP_can_pull;
	
	return component;
}

static void initialize_HFP_center_policy(unsigned sched_ctx_id)
{	
	if (starpu_get_env_number_default("READY", 1) == 1) 
	{ 
		starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_HFP_create, NULL,
				STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
				STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
				STARPU_SCHED_SIMPLE_FIFOS_BELOW |
				STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY | /* ready of dmdar plugged into HFP */
				STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
				STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
	}
	else
	{
		starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_HFP_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
	}
}

static void deinitialize_HFP_center_policy(unsigned sched_ctx_id)
{
	//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Deinitialize\n"); }
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

void get_current_tasks_heft(struct starpu_task *task, unsigned sci)
{
	if (starpu_get_env_number_default("PRINTF",0) == 1) { print_effective_order_in_file(task); }
	starpu_sched_component_worker_pre_exec_hook(task,sci);
}

void get_current_tasks(struct starpu_task *task, unsigned sci)
{
	task_currently_treated = task;
	starpu_sched_component_worker_pre_exec_hook(task,sci);
}

/* Almost Belady while tasks are being executed 
 * TODO : corriger belady en cas de multi gpu
 */
starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch)
{
	if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Début de Belady\n"); }
	int donnee_utilise_dans_le_plus_longtemps = 0; int distance_donnee_utilise_dans_le_plus_longtemps = 0;
	int k = 0; int j = 0; int i = 0;
	unsigned nb_data_on_node = 0; /* Number of data loaded on memory. Needed to init the tab containing data on node */
	int is_allocated;
		
	/* Getting on the sub_list corresponding to the current GPU */
	int current_gpu = starpu_memory_node_get_devid(node);
	assert(starpu_node_get_kind(node) == 2); /* 2 = STARPU_CUDA_RAM */
	//~ struct gpu_list *gpu_data;
	//~ struct use_order *use_order_data = gpu_data->first_gpu;
	//~ gpu_data->use_order_data = gpu_data->first_gpu;
	
	printf("%p\n",use_order_data->data_list[0]);
	printf("current gpu = %d\n",current_gpu);
	for (i = 0; i < current_gpu; i++) 
	{ 
		printf("ok\n");
		use_order_data = use_order_data->next_gpu;
		printf("ok2\n");
	}
	printf("%p\n",use_order_data->data_list[0]);
	//TODO mettre un use_order_data->last data += le nb de doné de la tache courante; à la fin
	
	if (task_currently_treated != NULL && task_currently_treated->cl != NULL) {
		starpu_data_handle_t *data_on_node;
		starpu_data_get_node_data(node, &data_on_node, &nb_data_on_node);
		
		//~ //Because I started at 1 and not 0
		//~ int used_index_task_currently_treated = index_task_currently_treated - 1;
		
		if (starpu_get_env_number_default("PRINTF",0) == 1)
			printf("La tâche en cours est %p\n",task_currently_treated);
		
		printf("Donnés de la tache %p en cours : %p %p et %p\n",task_currently_treated,STARPU_TASK_GET_HANDLE(task_currently_treated,0),STARPU_TASK_GET_HANDLE(task_currently_treated,1),STARPU_TASK_GET_HANDLE(task_currently_treated,2));
		int nb = STARPU_TASK_GET_NBUFFERS(task_currently_treated);
		printf("Nb de données de la tâche : %d\n",nb);
		
		//~ //A CHANGER
		//~ if (task_position_in_data_use_order[index_task_currently_treated] != summed_nb_data_each_gpu[current_gpu]) {
		if (STARPU_TASK_GET_HANDLE(task_currently_treated,1) != use_order_data->data_list[use_order_data->total_nb_data]) {
			//~ nb_data_next_task = task_position_in_data_use_order[used_index_task_currently_treated] - task_position_in_data_use_order[used_index_task_currently_treated - 1];
			
			//~ /* pas les bonnesdonnées la mais dans le fichier ca a l'air bon 
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("nb data next :%d\n",nb_data_next_task);
			//~ printf("Données de la tâche en cours : ");
			//~ for (i = 0; i < nb_data_next_task; i++) {
				//~ printf("%p ",data_use_order[task_position_in_data_use_order[used_index_task_currently_treated] - i - 1]); } printf ("\n"); 
			//~ } */		
			
			for (i = 0; i < nb_data_on_node; i++) { 
				if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Data on node : %p\n",data_on_node[i]); }
			}

			//~ //tenir compte avec variable globale de l'index des données de la tache en cours dans le tableau. Pour
			//~ //cela regarder le num de gpu, et faire ndonnnégpu1 + ndonnégpu2 + donné courante de la tache courante
			
			for (i = 0; i < nb; i++) {	
				/* On regarde si la donnée est pas déjà sur M par hasard */
				starpu_data_query_status(STARPU_TASK_GET_HANDLE(task_currently_treated,i), node, &is_allocated, NULL, NULL);
				//~ /* //~ if (is_allocated && i == 1000) { */
				//~ if (is_allocated && i == 1000) { /* pk 1000 la ? a tester */
				if (is_allocated) { /* pk 1000 la ? a tester */
					if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("La donnée %p est déjà sur M\n",STARPU_TASK_GET_HANDLE(task_currently_treated,i)); }
				}
				else {
						int *prochaine_utilisation_donnee;
						prochaine_utilisation_donnee = malloc(nb_data_on_node*sizeof(int));
						for (j = 0; j < nb_data_on_node; j++) { prochaine_utilisation_donnee[j] = INT_MAX; }
						//Care, if a task is never use again and is on node, we must evict it
						for (j = 0; j < nb_data_on_node; j++) { 
							if (starpu_data_can_evict(data_on_node[j], node, is_prefetch)) {
								//N'est pas utilisé par la suite
								/* modifier le 11111 j'ai juste mis ca la pour que ca compile */
								//~ for (k = summed_nb_data_each_gpu[current_gpu] - 11111; k < summed_nb_data_each_gpu[current_gpu]; k++) {
								for (k = use_order_data->last_position_in_data_use_order; k < use_order_data->total_nb_data; k++) 
								{
									if (data_on_node[j] == use_order_data->data_list[k]) {
										prochaine_utilisation_donnee[j] = k;
										break;
									}
								}
							}
							else { prochaine_utilisation_donnee[j] = -1; }
						}
											
					distance_donnee_utilise_dans_le_plus_longtemps = -1;
					for (j = 0; j < nb_data_on_node; j++) {
						if (prochaine_utilisation_donnee[j] > distance_donnee_utilise_dans_le_plus_longtemps) {
								donnee_utilise_dans_le_plus_longtemps = j;
								distance_donnee_utilise_dans_le_plus_longtemps = prochaine_utilisation_donnee[j]; 
						}
					}
					if (distance_donnee_utilise_dans_le_plus_longtemps == -1) {
						free(data_on_node); 
						free(prochaine_utilisation_donnee);
						return STARPU_DATA_NO_VICTIM;  
					}
					
					starpu_data_handle_t returned_handle = data_on_node[donnee_utilise_dans_le_plus_longtemps];
					free(data_on_node);
					free(prochaine_utilisation_donnee);
					return returned_handle;
													
				}
			}
	}
	else 
	{
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("On est sur la dernière tâche il faudrait sortir la\n"); } 
		free(data_on_node);
		return NULL;
		/* return STARPU_DATA_NO_VICTIM; */
	} 
	}
	else 
	{ 
		if (starpu_get_env_number_default("PRINTF",0) == 1) {  printf("task current = null\n"); }
	} 
	return STARPU_DATA_NO_VICTIM;
}

struct starpu_sched_policy _starpu_sched_HFP_policy =
{
	.init_sched = initialize_HFP_center_policy,
	.deinit_sched = deinitialize_HFP_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = get_current_tasks, /* Getting current task for Belady later on */
	//~ .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "HFP",
	.policy_description = "Affinity aware task ordering",
	.worker_type = STARPU_WORKER_LIST,
};

static void initialize_heft_hfp_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_schedulers(sched_ctx_id, 1, (starpu_sched_component_create_t) starpu_sched_component_mct_create, NULL,
			STARPU_SCHED_SIMPLE_PRE_DECISION,
			(starpu_sched_component_create_t) starpu_sched_component_HFP_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_PERFMODEL |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL);
}

struct starpu_sched_policy _starpu_sched_modular_heft_HFP_policy =
{
	.init_sched = initialize_heft_hfp_policy,
	.deinit_sched = starpu_sched_tree_deinitialize,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = get_current_tasks_heft, /* Getting current task for printing diff later on */
	//~ .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-heft-HFP",
	.policy_description = "heft modular policy",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};
