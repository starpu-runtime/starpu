/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023	Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_sched_component.h>
#include <datawizard/memory_nodes.h>
#include <sched_policies/helper_mct.h>
#include <sched_policies/darts.h>
#include <sched_policies/HFP.h>
#include <sched_policies/sched_visu.h>

static int order_u;
static int multigpu;
static int modular_heft_hfp_mode;
static int task_stealing;
static int interlacing;
static int faster_first_iteration;
static starpu_pthread_mutex_t HFP_mutex;
static int belady;
static int hmetis_n;
static int number_task_out;

int _starpu_HFP_hmetis;
static int sparse_matrix;

static int _nb_gpus;
const char* _starpu_HFP_appli;
int _starpu_HFP_NT;
int _starpu_HFP_N;
static double EXPECTED_TIME;
starpu_ssize_t _starpu_HFP_GPU_RAM_M;
bool _starpu_HFP_do_schedule_done;

/* Variables used for Belady in xgemm.c and HFP.c and mst.c */
struct starpu_sched_policy _starpu_sched_HFP_policy;
struct starpu_sched_policy _starpu_sched_modular_heft_HFP_policy;

/* Other environmment variable you should use with HFP:
 * STARPU_NTASKS_THRESHOLD=30 ou 10 si on veut moins entrer dans victim_selector peut_être
 * STARPU_MINIMUM_CLEAN_BUFFERS=0
 * STARPU_TARGET_CLEAN_BUFFERS=0
 * STARPU_CUDA_PIPELINE=4
 * STARPU_NCPU=0
 * STARPU_NOPENCL=0
 * RANDOM_TASK_ORDER
 * RECURSIVE_MATRIX_LAYOUT
 * RANDOM_DATA_ACCESS
 * STARPU_SCHED_READY=1
 * STARPU_CUDA_PIPELINE=5
 * STARPU_NTASKS_THRESHOLD=10
 */

static int _get_number_GPU()
{
	int return_value = starpu_memory_nodes_get_count_by_kind(STARPU_CUDA_RAM);

	if (return_value == 0) /* We are not using GPUs so we are in an out-of-core case using CPUs. Need to return 1. If I want to deal with GPUs AND CPUs we need to adpt this function to return NGPU + 1 */
	{
		return 1;
	}

	return return_value;
}

void _starpu_visu_init()
{
	_nb_gpus = _get_number_GPU();
	_sched_visu_init(_nb_gpus);

	_starpu_HFP_do_schedule_done = false;
}

/* Used only for visualisation of non-HFP schedulers in python */
void _starpu_HFP_initialize_global_variable(struct starpu_task *task)
{
	_starpu_HFP_N = _print_n;
	_nb_gpus = _get_number_GPU();

	/* Getting the total number of tasks */
	if(_print3d == 0) /* 2D */
	{
		_starpu_HFP_NT = _starpu_HFP_N*_starpu_HFP_N;
	}
	else if(_print3d == 1) /* Z = 4 here */
	{
		_starpu_HFP_NT = _starpu_HFP_N*_starpu_HFP_N*4;
	}
	else if(_print3d == 2) /* It means that Z = N */
	{
		_starpu_HFP_NT = _starpu_HFP_N*_starpu_HFP_N*_starpu_HFP_N;
	}

	_starpu_HFP_appli = starpu_task_get_name(task);
}

/* Put a link at the beginning of the linked list */
void _starpu_HFP_insertion(struct _starpu_HFP_paquets *a)
{
	struct _starpu_HFP_my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
	new->next = a->temp_pointer_1;
	new->nb_task_in_sub_list = 0;
	new->expected_time_pulled_out = 0;
	new->expected_time = 0;
	new->expected_package_computation_time = 0;
	new->data_weight = 0;
	new->data_to_evict_next = NULL;
	starpu_task_list_init(&new->refused_fifo_list);
	a->temp_pointer_1 = new;
}

/* Delete all the empty packages */
static struct _starpu_HFP_my_list* HFP_delete_link(struct _starpu_HFP_paquets* a)
{
	while (a->first_link != NULL && a->first_link->package_nb_data == 0)
	{
		a->temp_pointer_1 = a->first_link;
		a->first_link = a->first_link->next;
		free(a->temp_pointer_1);
	}
	if (a->first_link != NULL)
	{
		a->temp_pointer_2 = a->first_link;
		a->temp_pointer_3 = a->first_link->next;
		while (a->temp_pointer_3 != NULL)
		{
			while (a->temp_pointer_3 != NULL && a->temp_pointer_3->package_nb_data == 0)
			{
				a->temp_pointer_1 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
				a->temp_pointer_2->next = a->temp_pointer_3;
				free(a->temp_pointer_1);
			}
			if (a->temp_pointer_3 != NULL)
			{
				a->temp_pointer_2 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
			}
		}
	}
	return a->first_link;
}

/* Reverse the order of task in a package for order U */
static struct _starpu_HFP_my_list* HFP_reverse_sub_list(struct _starpu_HFP_my_list *a)
{
	struct starpu_task_list b;
	starpu_task_list_init(&b);
	while (!starpu_task_list_empty(&a->sub_list))
	{
		starpu_task_list_push_front(&b,starpu_task_list_pop_front(&a->sub_list));
	}
	while (!starpu_task_list_empty(&b))
	{
		starpu_task_list_push_back(&a->sub_list,starpu_task_list_pop_front(&b));
	}
	return a;
}

/* Print for each GPU the order of processing of each data */
//static void print_next_use_each_data(struct _starpu_HFP_paquets* a)
//{
//	a->temp_pointer_1 = a->first_link;
//	struct starpu_task *task = NULL;
//	int current_gpu = 0;
//	struct _starpu_HFP_next_use_by_gpu *c = _starpu_HFP_next_use_by_gpu_new();
//	while (a->temp_pointer_1 != NULL)
//	{
//		printf("Pour le GPU %d.\n", current_gpu);
//		for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
//		{
//			printf("Task %p :", task);
//			unsigned i;
//			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
//			{
//				printf(" %p", STARPU_TASK_GET_HANDLE(task, i));
//				struct _starpu_HFP_next_use *b = STARPU_TASK_GET_HANDLE(task, i)->sched_data;
//				for (c = _starpu_HFP_next_use_by_gpu_list_begin(b->next_use_tab[current_gpu]); c != _starpu_HFP_next_use_by_gpu_list_end(b->next_use_tab[current_gpu]); c = _starpu_HFP_next_use_by_gpu_list_next(c))
//				{
//					printf("->%d", c->value_next_use);
//				}
//				printf(" |");
//			}
//			printf("\n-----\n");
//		}
//		a->temp_pointer_1 = a->temp_pointer_1->next;
//		current_gpu++;
//	}
//}

/* TODO a suppr */
//~ struct timeval time_start_getorderbelady;
//~ struct timeval time_end_getorderbelady;
//~ long long time_total_getorderbelady = 0;

/* Utile pour printing mais surtout pour l'itération 1 plus rapide */
static int iteration;

/* Read the tasks's order and each time it se a data, it add a value of it's next use in the task list.
 * Then in the post_exec_hook we pop the value of the handles of the task processed. In belady we just look at these value
 * for each data on node and evict the one with the furthest first value.
 * TODO : A noer/dire que si ready modifie l'ordre et bien les pop de valuers dans le post exec hook
 * ne sont plus exacts. mis bon cela ne devrait pas trop impacter les performances. */
static void get_ordre_utilisation_donnee(struct _starpu_HFP_paquets* a, int nb_gpu)
{
	//~ gettimeofday(&time_start_getorderbelady, NULL);

	struct starpu_task *task = NULL;
	a->temp_pointer_1 = a->first_link;
	int current_gpu = 0;
	int j = 0;
	int compteur = 0;
	struct _starpu_HFP_next_use *b = NULL;

	while (a->temp_pointer_1 != NULL)
	{
		for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
		{
			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				compteur++;
				struct _starpu_HFP_next_use_by_gpu *c = _starpu_HFP_next_use_by_gpu_new();
				c->value_next_use = compteur;
				if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL) /* If it's empty I create the list in the handle */
				{
					/* J'initialise à vide la liste pour chaque case du tableau */
					b = malloc(sizeof(*b));
					b->next_use_tab = malloc(sizeof(*b->next_use_tab));
					for (j = 0; j < nb_gpu; j++)
					{
						b->next_use_tab[j] = _starpu_HFP_next_use_by_gpu_list_new();
					}
					_starpu_HFP_next_use_by_gpu_list_push_back(b->next_use_tab[current_gpu], c);
					STARPU_TASK_GET_HANDLE(task, i)->sched_data = b;
				}
				else /* Else I just add a new int */
				{
					b = STARPU_TASK_GET_HANDLE(task, i)->sched_data;
					_starpu_HFP_next_use_by_gpu_list_push_back(b->next_use_tab[current_gpu], c);
					STARPU_TASK_GET_HANDLE(task, i)->sched_data = b;
				}
			}
		}
		current_gpu++;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		compteur = 0;
	}
	//~ gettimeofday(&time_end_getorderbelady, NULL);
	//~ time_total_getorderbelady += (time_end_getorderbelady.tv_sec - time_start_getorderbelady.tv_sec)*1000000LL + time_end_getorderbelady.tv_usec - time_start_getorderbelady.tv_usec;
}

/* TODO a suppr */
//~ struct timeval time_start_getcommondataorderu;
//~ struct timeval time_end_getcommondataorderu;
//~ long long time_total_getcommondataorderu = 0;

/* For order U. Return the number of common data of each sub package when merging I and J */
static int get_common_data_last_package(struct _starpu_HFP_my_list *I, struct _starpu_HFP_my_list *J, int evaluation_I, int evaluation_J, bool IJ_inferieur_GPU_RAM, starpu_ssize_t GPU_RAM_M)
{
	//~ gettimeofday(&time_start_getcommondataorderu, NULL);

	int split_ij = 0;
	/* evaluation: 0 = tout, 1 = début, 2 = fin */
	struct starpu_task *task = NULL;
	bool insertion_ok = false;
	bool donnee_deja_presente = false;
	int j = 0;
	int common_data_last_package = 0;
	long int poids_tache_en_cours = 0;
	long int poids = 0;
	int index_tab_donnee_I = 0;
	int index_tab_donnee_J = 0;
	int parcours_liste = 0;
	int i_bis = 0;
	starpu_data_handle_t * donnee_I = NULL;
	starpu_data_handle_t * donnee_J = NULL;

	if (strcmp(_starpu_HFP_appli, "chol_model_11") == 0)
	{
		donnee_J = calloc((J->package_nb_data*1.5), sizeof(J->package_data[0]));
		donnee_I = malloc((I->package_nb_data*1.5) * sizeof(I->package_data[0]));
	}
	else
	{
		donnee_J = calloc((J->package_nb_data), sizeof(J->package_data[0]));
		donnee_I = malloc((I->package_nb_data) * sizeof(I->package_data[0]));
	}

	if (evaluation_I == 0)
	{
		int i;
		for (i = 0; i < I->package_nb_data; i++)
		{
			donnee_I[i] = I->package_data[i];
		}
		index_tab_donnee_I = I->package_nb_data;
	}
	else if (evaluation_I == 1 && IJ_inferieur_GPU_RAM == false)
	{
		poids = 0; insertion_ok = false;
		task = starpu_task_list_begin(&I->sub_list);
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task, i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, i));
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		while(1)
		{
			task = starpu_task_list_next(task);
			if (task == NULL)
			{
				break;
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, i));
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task, i);
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M)
			{
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					if (tab_tache_en_cours[i] != NULL)
					{
						donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i];
						index_tab_donnee_I++;
						insertion_ok = true;
					}
				}
				if (insertion_ok == true)
				{
					poids += poids_tache_en_cours;
				}
				insertion_ok = false;
			}
			else
			{
				break;
			}
		}
	}
	else if (evaluation_I == 2 && IJ_inferieur_GPU_RAM == false)
	{
		poids = 0;
		i_bis = 1; insertion_ok = false;
		task = starpu_task_list_begin(&I->sub_list);
		while(starpu_task_list_next(task) != NULL)
		{
			task = starpu_task_list_next(task);
		}
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		while(1)
		{
			i_bis++;
			task = starpu_task_list_begin(&I->sub_list);
			for (parcours_liste = I->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--)
			{
				task = starpu_task_list_next(task);
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M)
			{
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					if (tab_tache_en_cours[i] != NULL)
					{
						donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i];
						index_tab_donnee_I++;
						insertion_ok = true;
					}
				}
				if (insertion_ok == true)
				{
					poids += poids_tache_en_cours;
				}
				insertion_ok = false;
			}
			else
			{
				break;
			}
		}
	}
	else if (IJ_inferieur_GPU_RAM == true)
	{
		if (evaluation_I == 0)
		{
			printf("Error evaluation de I alors que I et J <= GPU_RAM\n");
			exit(0);
		}
		if (evaluation_I == 2)
		{
			split_ij = I->nb_task_in_sub_list - I->split_last_ij + 1;
		}
		else
		{
			split_ij = I->split_last_ij + 1;
		}
		task = starpu_task_list_begin(&I->sub_list);
		if (evaluation_I == 2)
		{
			while(starpu_task_list_next(task) != NULL)
			{
				task = starpu_task_list_next(task);
			}
		}
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_I[i] = STARPU_TASK_GET_HANDLE(task,i);
		}
		index_tab_donnee_I = STARPU_TASK_GET_NBUFFERS(task);
		for (i_bis = 2; i_bis < split_ij; i_bis++)
		{
			if (evaluation_I == 2)
			{
				task = starpu_task_list_begin(&I->sub_list);
				for (parcours_liste = I->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--)
				{
					task = starpu_task_list_next(task);
				}
			}
			else
			{
				task = starpu_task_list_next(task);
			}
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(I->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < I->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_I[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (tab_tache_en_cours[i] != NULL)
				{
					donnee_I[index_tab_donnee_I] = tab_tache_en_cours[i];
					index_tab_donnee_I++;
				}
			}
		}
	}

	if (evaluation_J == 0)
	{
		int i;
		for (i = 0; i < J->package_nb_data; i++)
		{
			donnee_J[i] = J->package_data[i];
		}
		index_tab_donnee_J = J->package_nb_data;
	}
	else if (evaluation_J == 1 && IJ_inferieur_GPU_RAM == false)
	{
		poids = 0;
		insertion_ok = false;
		task = starpu_task_list_begin(&J->sub_list);
		unsigned int i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		while(1)
		{
			task = starpu_task_list_next(task);
			if (task == NULL)
			{
				break;
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M)
			{
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					if (tab_tache_en_cours[i] != NULL)
					{
						donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i];
						index_tab_donnee_J++;
						insertion_ok = true;
					}
				}
				if (insertion_ok == true)
				{
					poids += poids_tache_en_cours;
				}
				insertion_ok = false;
			}
			else
			{
				break;
			}
		}
	}
	else if (evaluation_J == 2 && IJ_inferieur_GPU_RAM == false)
	{
		poids = 0;
		i_bis = 1; insertion_ok = false;
		/* Se placer sur la dernière tâche du paquet J */
		task = starpu_task_list_begin(&J->sub_list);
		while(starpu_task_list_next(task) != NULL)
		{
			task = starpu_task_list_next(task);
		}
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
			poids += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		while(1)
		{
			i_bis++;
			task = starpu_task_list_begin(&J->sub_list);
			for (parcours_liste = J->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--)
			{
				task = starpu_task_list_next(task);
			}
			poids_tache_en_cours = 0;
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					poids_tache_en_cours += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
			}
			if (poids + poids_tache_en_cours <= GPU_RAM_M)
			{
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					if (tab_tache_en_cours[i] != NULL)
					{
						donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i];
						index_tab_donnee_J++;
						insertion_ok = true;
					}
				}
				if (insertion_ok == true)
				{
					poids += poids_tache_en_cours;
				}
				insertion_ok = false;
			}
			else
			{
				break;
			}
		}
	}
	else if (IJ_inferieur_GPU_RAM == true)
	{
		if (evaluation_J == 0)
		{
			printf("Error evaluation de J alors que I et J <= GPU_RAM\n");
			exit(0);
		}
		if (evaluation_J == 2)
		{
			split_ij = J->nb_task_in_sub_list - J->split_last_ij + 1;
		}
		else
		{
			split_ij = J->split_last_ij + 1;
		}
		task = starpu_task_list_begin(&J->sub_list);
		if (evaluation_J == 2)
		{
			while(starpu_task_list_next(task) != NULL)
			{
				task = starpu_task_list_next(task);
			}
		}
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			donnee_J[i] = STARPU_TASK_GET_HANDLE(task,i);
		}
		index_tab_donnee_J = STARPU_TASK_GET_NBUFFERS(task);
		for (i_bis = 2; i_bis < split_ij; i_bis++)
		{
			if (evaluation_J == 2)
			{
				task = starpu_task_list_begin(&J->sub_list);
				for (parcours_liste = J->nb_task_in_sub_list - i_bis; parcours_liste > 0; parcours_liste--)
				{
					task = starpu_task_list_next(task);
				}
			}
			else
			{
				task = starpu_task_list_next(task);
			}
			starpu_data_handle_t * tab_tache_en_cours = malloc((STARPU_TASK_GET_NBUFFERS(task)) * sizeof(J->package_data[0]));
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				tab_tache_en_cours[i] = NULL;
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				donnee_deja_presente = false;
				for (j = 0; j < J->package_nb_data; j++)
				{
					if (STARPU_TASK_GET_HANDLE(task,i) == donnee_J[j])
					{
						donnee_deja_presente = true;
						break;
					}
				}
				if (donnee_deja_presente == false)
				{
					tab_tache_en_cours[i] = STARPU_TASK_GET_HANDLE(task,i);
				}
			}
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (tab_tache_en_cours[i] != NULL)
				{
					donnee_J[index_tab_donnee_J] = tab_tache_en_cours[i];
					index_tab_donnee_J++;
				}
			}
		}
	}
	int i;
	for (i = 0; i < index_tab_donnee_I; i++)
	{
		for (j = 0; j < index_tab_donnee_J; j++)
		{
			if (donnee_I[i] == donnee_J[j])
			{
				common_data_last_package++;
				break;
			}
		}
	}

	//~ gettimeofday(&time_end_getcommondataorderu, NULL);
	//~ time_total_getcommondataorderu += (time_end_getcommondataorderu.tv_sec - time_start_getcommondataorderu.tv_sec)*1000000LL + time_end_getcommondataorderu.tv_usec - time_start_getcommondataorderu.tv_usec;

	return common_data_last_package;
}

/* Comparator used to sort the data of a packages to erase the duplicate in O(n) */
static int HFP_pointeurComparator(const void *first, const void *second)
{
	return (*(int*)first - *(int*)second);
}

//TODO : ne fonctionne plus en 3D car le fichier dans le quel j'écrit je met x y z gpu mainteannt et non x y gpu en 3D
//static void visualisation_tache_matrice_format_tex(char *algo)
//{
//	printf("debut visualisation, %d\n", _starpu_HFP_N);
//	int i, j, red, green, blue, x, y, gpu, k, ZN;
//	int processing_order[_nb_gpus]; /* One for each GPU */
//	for (i = 0; i < _nb_gpus; i++) { processing_order[i] = 0; }
//	int size = strlen("Output_maxime/Data_coordinates_order_last_.tex") + strlen(algo) + 1;
//	char *path = (char *)malloc(size);
//	strcpy(path, "Output_maxime/Data_coordinates_order_last_");
//	strcat(path, algo);
//	strcat(path, ".tex");
//
//	printf("coord : %s\n", path);
//
//	FILE * fcoordinate_order_last = fopen(path, "w");
//	size = strlen("Output_maxime/Data_coordinates_order_last_.txt") + strlen(algo) + 1;
//	path = (char *)malloc(size);
//	strcpy(path, "Output_maxime/Data_coordinates_order_last_");
//	strcat(path, algo);
//	strcat(path, ".txt");
//
//	printf("input : %s\n", path);
//
//	FILE *f_input = fopen(path, "r");
//	fprintf(fcoordinate_order_last,"\\documentclass{article}\\usepackage{color}\\usepackage{fullpage}\\usepackage{colortbl}\\usepackage{caption}\\usepackage{subcaption}\\usepackage{float}\\usepackage{graphics}\n\n\\begin{document}\n\n\\begin{figure}[H]");
//	i = 0; k = 0;
//	if (_print3d != 0) /* Printing a 3D matrix, we print 4 tabular because we have nblocksz 4 */
//	{
//		if (_print3d == 2)
//		{
//			ZN = _starpu_HFP_N;
//		}
//		else
//		{
//			ZN = 4;
//		}
//
//		printf("ZN = %d\n", ZN);
//
//		int tab_order_1[ZN][_starpu_HFP_N][_starpu_HFP_N];
//		for (k = 0; k < ZN; k++)
//		{
//			for (i = 0; i < _starpu_HFP_N; i++)
//			{
//				for (j = 0; j < _starpu_HFP_N; j++)
//				{
//					tab_order_1[k][i][j] = -1;
//				}
//			}
//		}
//		int tab_gpu_1[ZN][_starpu_HFP_N][_starpu_HFP_N];
//		for (k = 0; k < ZN; k++)
//		{
//			for (i = 0; i < _starpu_HFP_N; i++)
//			{
//				for (j = 0; j < _starpu_HFP_N; j++)
//				{
//					tab_gpu_1[k][i][j] = -1;
//				}
//			}
//		}
//		i = 0;
//		if (f_input != NULL && fcoordinate_order_last != NULL)
//		{
//			while (!feof (f_input))
//			{
//				if (fscanf(f_input, "%d	%d	%d", &x, &y, &gpu) != 3)
//				{
//					//~ perror("error fscanf in visualisation_tache_matrice_format_tex\n"); exit(EXIT_FAILURE);
//				}
//				if (tab_order_1[0][x][y] == -1)
//				{
//					tab_order_1[0][x][y] = processing_order[gpu];
//					processing_order[gpu]++;
//				}
//				else if (tab_order_1[1][x][y] == -1)
//				{
//					tab_order_1[1][x][y] = processing_order[gpu];
//					processing_order[gpu]++;
//				}
//				else if (tab_order_1[2][x][y] == -1)
//				{
//					tab_order_1[2][x][y] = processing_order[gpu];
//					processing_order[gpu]++;
//				}
//				else
//				{
//					tab_order_1[3][x][y] = processing_order[gpu];
//					processing_order[gpu]++;
//				}
//				if (tab_gpu_1[0][x][y] == -1)
//				{
//					tab_gpu_1[0][x][y] = gpu;
//				}
//				else if (tab_gpu_1[1][x][y] == -1)
//				{
//					tab_gpu_1[1][x][y] = gpu;
//				}
//				else if (tab_gpu_1[2][x][y] == -1)
//				{
//					tab_gpu_1[2][x][y] = gpu;
//				}
//				else
//				{
//					tab_gpu_1[3][x][y] = gpu;
//				}
//				i++;
//			}
//		}
//		else
//		{
//			perror("Impossible d'ouvrir au moins 1 fichier dans visualisation_tache_matrice_format_tex() dans if 3D\n");
//			exit(EXIT_FAILURE);
//		}
//		printf("ok\n");
//		tab_order_1[3][x][y] = tab_order_1[3][x][y] - 1;
//		for (k = 0; k < ZN; k++)
//		{
//			fprintf(fcoordinate_order_last, "\n\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{|");
//			for (i = 0; i < _starpu_HFP_N - 1; i++)
//			{
//				fprintf(fcoordinate_order_last,"c|");
//			}
//			fprintf(fcoordinate_order_last,"c|}\n\\hline");
//			for (i = 0; i < _starpu_HFP_N; i++)
//			{
//				for (j = 0; j < _starpu_HFP_N - 1; j++)
//				{
//					if (tab_gpu_1[k][j][i] == 0)
//					{
//						red = 255; green = 255; blue = 255;
//					}
//					else if (tab_gpu_1[k][j][i] == 6)
//					{
//						red = 70; green = 130; blue = 180;
//					}
//					else
//					{
//						rgb(tab_gpu_1[k][j][i], &red, &green, &blue);
//					}
//					fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, tab_order_1[k][j][i]);
//				}
//				if (tab_gpu_1[k][j][i] == 0)
//				{
//					red = 255; green = 255; blue = 255;
//				}
//				else if (tab_gpu_1[k][j][i] == 6)
//				{
//					red = 70; green = 130; blue = 180;
//				}
//				else
//				{
//					rgb(tab_gpu_1[k][j][i], &red, &green, &blue);
//				}
//				fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,tab_order_1[k][j][i]);
//				fprintf(fcoordinate_order_last," \\\\"); fprintf(fcoordinate_order_last,"\\hline");
//			}
//			fprintf(fcoordinate_order_last, "\\end{tabular}\\caption{Z = %d}\\end{subfigure}\n", k + 1);
//		}
//		fprintf(fcoordinate_order_last, "\n\\caption{Task's processing order on a 3D matrix}\\end{figure}\n\n\\end{document}");
//	}
//	else /* Printing a 2D matrix so only one matrix */
//	{
//		fprintf(fcoordinate_order_last, "\n\\centering\\begin{tabular}{|");
//		for (i = 0; i < _starpu_HFP_N - 1; i++)
//		{
//			fprintf(fcoordinate_order_last,"c|");
//		}
//		i = 0;
//		fprintf(fcoordinate_order_last,"c|}\n\\hline");
//		int tab_order[_starpu_HFP_N][_starpu_HFP_N];
//		int tab_gpu[_starpu_HFP_N][_starpu_HFP_N];
//		if (f_input != NULL && fcoordinate_order_last != NULL)
//		{
//			printf("reading, N = %d, NT = %d\n", _starpu_HFP_N, _starpu_HFP_NT);
//			while (!feof (f_input))
//			{
//				if (fscanf(f_input, "%d	%d	%d", &x, &y, &gpu) != 3)
//				{
//					//~ perror("error fscanf in visualisation_tache_matrice_format_tex_HEFT\n"); exit(EXIT_FAILURE);
//				}
//				tab_order[x][y] = processing_order[gpu];
//				processing_order[gpu]++;
//				tab_gpu[x][y] = gpu;
//				i++;
//			}
//		}
//		else
//		{
//			perror("Impossible d'ouvrir au moins 1 fichier dans visualisation_tache_matrice_format_tex()\n"); exit(EXIT_FAILURE);
//		}
//		tab_order[x][y] = tab_order[x][y] - 1;
//		for (i = 0; i < _starpu_HFP_N; i++)
//		{
//			for (j = 0; j < _starpu_HFP_N - 1; j++)
//			{
//				if (tab_gpu[j][i] == 0)
//				{
//					red = 255; green = 255; blue = 255;
//				}
//				else if (tab_gpu[j][i] == 6)
//				{
//					red = 70; green = 130; blue = 180;
//				}
//				else
//				{
//					rgb(tab_gpu[j][i], &red, &green, &blue);
//				}
//				fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d&", red,green,blue, tab_order[j][i]);
//			}
//			if (tab_gpu[j][i] == 0)
//			{
//				red = 255; green = 255; blue = 255;
//			}
//			else if (tab_gpu[j][i] == 6)
//			{
//				red = 70; green = 130; blue = 180;
//			}
//			else
//			{
//				rgb(tab_gpu[j][i], &red, &green, &blue);
//			}
//			fprintf(fcoordinate_order_last,"\\cellcolor[RGB]{%d,%d,%d}%d",red,green,blue,tab_order[j][i]);
//			fprintf(fcoordinate_order_last," \\\\"); fprintf(fcoordinate_order_last,"\\hline");
//		}
//		fprintf(fcoordinate_order_last, "\\end{tabular}\n\\caption{Task's processing order}\\end{figure}\n\n\\end{document}");
//	}
//	fclose(fcoordinate_order_last);
//	fclose(f_input);
//	printf("fin visualisation\n");
//}

/* To print data order and number of data to load, only for 2D */
//static void visualisation_tache_matrice_format_tex_with_data_2D()
//{
//	printf("Début de visualisation_tache_matrice_format_tex_with_data_2D()\n");
//	int i, j, red, green, blue, x, y, gpu, data_to_load, tikz_index;
//	int processing_order[_nb_gpus]; for (i = 0; i < _nb_gpus; i++) { processing_order[i] = 0; }
//	FILE * f_input_data_to_load = fopen("Output_maxime/Data_to_load_SCHEDULER.txt", "r");
//	FILE * f_input_data_coordinate = fopen("Output_maxime/Data_coordinates_order_last_SCHEDULER.txt", "r");
//	FILE * f_output = fopen("Output_maxime/visualisation_matrice_2D.tex", "w");
//
//	fprintf(f_output,"\\documentclass{article}\\usepackage{colortbl,tikz,float,caption}\\makeatletter\\tikzset{hatch distance/.store in=\\hatchdistance,hatch distance=5pt,hatch thickness/.store in=\\hatchthickness,hatch thickness=5pt}\\pgfdeclarepatternformonly[\\hatchdistance,\\hatchthickness]{north east hatch}{\\pgfqpoint{-1pt}{-1pt}}{\\pgfqpoint{\\hatchdistance}{\\hatchdistance}}{\\pgfpoint{\\hatchdistance-1pt}{\\hatchdistance-1pt}}{\\pgfsetcolor{\\tikz@pattern@color}\\pgfsetlinewidth{\\hatchthickness}\\pgfpathmoveto{\\pgfqpoint{0pt}{0pt}}\\pgfpathlineto{\\pgfqpoint{\\hatchdistance}{\\hatchdistance}}\\pgfusepath{stroke}}\\makeatother\\usetikzlibrary{calc,shadings,patterns,tikzmark}\\newcommand\\HatchedCell[5][0pt]{\\begin{tikzpicture}[overlay,remember picture]\\path ($(pic cs:#2)!0.5!(pic cs:#3)$)coordinate(aux1)(pic cs:#4)coordinate(aux2);\\fill[#5]($(aux1)+(-0.23*0.075\\textwidth,1.9ex)$)rectangle($(aux1 |- aux2)+(0.23*0.075\\textwidth,-#1*\\baselineskip-.8ex)$);\\end{tikzpicture}}\n\n\\begin{document}\n\n\\begin{figure}[H]\\centering\\begin{tabular}{|");
//	for (i = 0; i < _starpu_HFP_N - 1; i++)
//	{
//		fprintf(f_output,"c|");
//	}
//	fprintf(f_output,"c|}\n\\hline");
//	i = 0;
//	int tab_order[_starpu_HFP_N][_starpu_HFP_N];
//	int tab_gpu[_starpu_HFP_N][_starpu_HFP_N];
//	int tab_data_to_load[_starpu_HFP_N][_starpu_HFP_N];
//	while (!feof (f_input_data_coordinate))
//	{
//		if (fscanf(f_input_data_coordinate, "%d	%d	%d", &x, &y, &gpu) != 3)
//		{
//		}
//		tab_order[x][y] = processing_order[gpu];
//		processing_order[gpu]++;
//		tab_gpu[x][y] = gpu;
//		if (fscanf(f_input_data_to_load, "%d %d %d", &x, &y, &data_to_load) != 3)
//		{
//		}
//		tab_data_to_load[x][y] = data_to_load;
//		//~ printf("Dans visu, x = %d, y = %d, data to load = %d\n", x, y, data_to_load);
//		i++;
//	}
//
//	tikz_index = 1;
//	/* Because eof is one line too far */
//	tab_order[x][y] = tab_order[x][y] - 1;
//	processing_order[gpu] = processing_order[gpu] - 1;
//	//~ tab_data_to_load[x][y] = tab_data_to_load[x][y] - 1;
//	//~ printf("%d pour x = %d, y = %d\n", tab_data_to_load[x][y], x, y);
//	for (i = 0; i < _starpu_HFP_N; i++)
//	{
//		for (j = 0; j < _starpu_HFP_N - 1; j++)
//		{
//			//~ if (tab_gpu[j][i] == 0) { red = 255; green = 255; blue = 255; }
//			//~ else if (tab_gpu[j][i] == 6) { red = 70; green = 130; blue = 180; }
//			//~ else
//			//~ {
//				rgb_gradient(tab_gpu[j][i], tab_order[j][i], processing_order[tab_gpu[j][i]], &red, &green, &blue);
//			//~ }
//			if (tab_data_to_load[j][i] == 1)
//			{
//				fprintf(f_output,"\\tikzmark{start%d}\\cellcolor[RGB]{%d,%d,%d}\\tikzmark{middle%d}\\tikzmark{end%d}\\HatchedCell{start%d}{middle%d}{end%d}{pattern color=black!100,pattern=north east hatch,hatch distance=4mm,hatch thickness=.3pt}&", tikz_index, red, green, blue, tikz_index, tikz_index, tikz_index, tikz_index, tikz_index);
//				tikz_index++;
//			}
//			else if (tab_data_to_load[j][i] == 2)
//			{
//				fprintf(f_output,"\\tikzmark{start%d}\\cellcolor[RGB]{%d,%d,%d}\\tikzmark{middle%d}\\tikzmark{end%d}\\HatchedCell{start%d}{middle%d}{end%d}{pattern color=black!100,pattern=north east hatch,hatch distance=2mm,hatch thickness=.3pt}&", tikz_index, red, green, blue, tikz_index, tikz_index, tikz_index, tikz_index, tikz_index);
//				tikz_index++;
//			}
//			else
//			{
//				fprintf(f_output,"\\cellcolor[RGB]{%d,%d,%d}&", red, green, blue);
//			}
//		}
//			rgb_gradient(tab_gpu[j][i], tab_order[j][i], processing_order[tab_gpu[j][i]], &red, &green, &blue);
//		if (tab_data_to_load[j][i] == 1)
//		{
//			fprintf(f_output,"\\tikzmark{start%d}\\cellcolor[RGB]{%d,%d,%d}\\tikzmark{middle%d}\\tikzmark{end%d}\\HatchedCell{start%d}{middle%d}{end%d}{pattern color=black!100,pattern=north east hatch,hatch distance=4mm,hatch thickness=.3pt}", tikz_index, red, green, blue, tikz_index, tikz_index, tikz_index, tikz_index, tikz_index);
//			tikz_index++;
//		}
//		else if (tab_data_to_load[j][i] == 2)
//		{
//			fprintf(f_output,"\\tikzmark{start%d}\\cellcolor[RGB]{%d,%d,%d}\\tikzmark{middle%d}\\tikzmark{end%d}\\HatchedCell{start%d}{middle%d}{end%d}{pattern color=black!100,pattern=north east hatch,hatch distance=2mm,hatch thickness=.3pt}", tikz_index, red, green, blue, tikz_index, tikz_index, tikz_index, tikz_index, tikz_index);
//			tikz_index++;
//		}
//		else
//		{
//			fprintf(f_output,"\\cellcolor[RGB]{%d,%d,%d}", red, green, blue);
//		}
//		fprintf(f_output," \\\\\\hline");
//	}
//	fprintf(f_output, "\\end{tabular}\n\\caption{2D matrix visualization}\\end{figure}\n\n\\end{document}");
//	fclose(f_output);
//	fclose(f_input_data_to_load);
//	fclose(f_input_data_coordinate);
//
//	printf("Fin de visualisation_tache_matrice_format_tex_with_data_2D()\n");
//}

/* For multi gpu with expected package time (MULTIGPU == 6).
 * Which is different than expected time used in our experiments. */
static struct _starpu_HFP_data_on_node *init_data_list(starpu_data_handle_t d)
{
	struct _starpu_HFP_data_on_node *liste = malloc(sizeof(*liste));
	struct _starpu_HFP_handle *element = malloc(sizeof(*element));

	if (liste == NULL || element == NULL)
	{
		exit(EXIT_FAILURE);
	}

	liste->memory_used = starpu_data_get_size(d);
	element->h = d;
	element->last_use = 0;
	element->next = NULL;
	liste->first_data = element;
	return liste;
}

/* For gemm that has C tile put in won't use if they are never used again */
static bool is_it_a_C_tile_data_never_used_again(starpu_data_handle_t h, int i, struct starpu_task_list *l, struct starpu_task *current_task)
{
	struct starpu_task *task = NULL;
	if (i == 2)
	{
		/* Getting on the right data/right task */
		for (task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
		{
			if (current_task == task)
			{
				break;
			}
		}
		for (task = starpu_task_list_next(task); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
		{
			if (h == STARPU_TASK_GET_HANDLE(task, 2))
			{
				return false;
			}
		}
		return true;
	}
	else
	{
		return false;
	}
}

/* Encore une fois seulement pour MULTIGPU == 6 */
static void insertion_data_on_node(struct _starpu_HFP_data_on_node *liste, starpu_data_handle_t nvNombre, int use_order, int i, struct starpu_task_list *l, struct starpu_task *current_task)
{
	struct _starpu_HFP_handle *nouveau = malloc(sizeof(*nouveau));
	if (liste == NULL || nouveau == NULL)
	{
		perror("List in void insertion_data_on_node is NULL\n");
		exit(EXIT_FAILURE);
	}
	liste->memory_used += starpu_data_get_size(nvNombre);
	nouveau->h = nvNombre;
	nouveau->next = liste->first_data;
	if (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") == 0)
	{
		if (is_it_a_C_tile_data_never_used_again(nouveau->h, i, l, current_task) == true)
		{
			nouveau->last_use = -1;
		}
		else
		{
			nouveau->last_use = use_order;
		}
	}
	else
	{
		nouveau->last_use = use_order;
	}
	liste->first_data = nouveau;
}

//static void afficher_data_on_node(struct _starpu_HFP_my_list *liste)
//{
//	if (liste == NULL)
//	{
//		exit(EXIT_FAILURE);
//	}
//
//	struct _starpu_HFP_handle *actuel = liste->pointer_node->first_data;
//
//	printf("Memory used = %ld | Expected time = %f / ", liste->pointer_node->memory_used, liste->expected_package_computation_time);
//	while (actuel != NULL)
//	{
//		printf("%p | %d -> ", actuel->h, actuel->last_use);
//		actuel = actuel->next;
//	}
//	printf("NULL\n");
//}

/* Search a data on the linked list of data */
static bool SearchTheData(struct _starpu_HFP_data_on_node *pNode, starpu_data_handle_t iElement, int use_order)
{
	pNode->pointer_data_list = pNode->first_data;
	while (pNode->pointer_data_list != NULL)
	{
		if(pNode->pointer_data_list->h == iElement)
		{
			if (use_order != -2) { pNode->pointer_data_list->last_use = use_order; }
			return true;
		}
		else
		{
			pNode->pointer_data_list = pNode->pointer_data_list->next;
		}
	}
	return false;
}

/* Replace the least recently used data on memory with the new one.
 * But we need to look that it's not a data used by current task too!
 * We remove first data from C if we are in a gemm application..0
 * Again it's only for MULTI GPU == 6 which we don't use.
 */
static void replace_least_recently_used_data(struct _starpu_HFP_data_on_node *a, starpu_data_handle_t data_to_load, int use_order, struct starpu_task *current_task, struct starpu_task_list *l, int index_handle)
{
	bool data_currently_used = false;
	int least_recent_use = INT_MAX;
	for (a->pointer_data_list = a->first_data; a->pointer_data_list != NULL; a->pointer_data_list = a->pointer_data_list->next)
	{
		data_currently_used = false;
		if (least_recent_use > a->pointer_data_list->last_use)
		{
			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(current_task); i++)
			{
				if (STARPU_TASK_GET_HANDLE(current_task, i) == a->pointer_data_list->h)
				{
					data_currently_used = true;
					break;
				}
			}
			if (data_currently_used == false)
			{
				least_recent_use = a->pointer_data_list->last_use;
			}
		}
	}
	for (a->pointer_data_list = a->first_data; a->pointer_data_list != NULL; a->pointer_data_list = a->pointer_data_list->next)
	{
		if (least_recent_use == a->pointer_data_list->last_use)
		{
			//~ printf("Données utilisé il y a le plus longtemps : %p | %d\n", a->pointer_data_list->h, a->pointer_data_list->last_use);
			a->pointer_data_list->h = data_to_load;
			if (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") == 0)
			{
				if (is_it_a_C_tile_data_never_used_again(a->pointer_data_list->h, index_handle, l, current_task) == true)
				{
					a->pointer_data_list->last_use = -1;
				}
				else
				{
					a->pointer_data_list->last_use = use_order;
				}
			}
			else
			{
				a->pointer_data_list->last_use = use_order;
			}
			break;
		}
	}
}

/* Push back in a package a task
 * Used in load_balance
 * Does not manage to migrate data of the task too
 */
static void merge_task_and_package(struct _starpu_HFP_my_list *package, struct starpu_task *task)
{
	int tab_runner = 0; int nb_duplicate_data = 0;
	package->nb_task_in_sub_list++;
	starpu_data_handle_t *temp_data_tab = malloc((package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task))*sizeof(package->package_data[0]));
	int i=0;
	unsigned j=0;
	while (i < package->package_nb_data && j < STARPU_TASK_GET_NBUFFERS(task))
	{
		if (package->package_data[i] <= STARPU_TASK_GET_HANDLE(task,j))
		{
			temp_data_tab[tab_runner] = package->package_data[i];
			i++;
		}
		else
		{
			temp_data_tab[tab_runner] = STARPU_TASK_GET_HANDLE(task,j);
			j++;
		}
		tab_runner++;
	}
	while (i < package->package_nb_data)
	{
		temp_data_tab[tab_runner] = package->package_data[i];
		i++;
		tab_runner++;
	}
	while (j < STARPU_TASK_GET_NBUFFERS(task))
	{
		temp_data_tab[tab_runner] = STARPU_TASK_GET_HANDLE(task,j);
		j++;
		tab_runner++;
	}
	for (i = 0; i < (package->package_nb_data + (int)STARPU_TASK_GET_NBUFFERS(task)); i++)
	{
		if (temp_data_tab[i] == temp_data_tab[i + 1])
		{
			temp_data_tab[i] = 0;
			nb_duplicate_data++;
		}
	}
	package->package_data = malloc((package->package_nb_data + STARPU_TASK_GET_NBUFFERS(task) - nb_duplicate_data) * sizeof(starpu_data_handle_t));
	j = 0;
	for (i = 0; i < (package->package_nb_data + (int)STARPU_TASK_GET_NBUFFERS(task)); i++)
	{
		if (temp_data_tab[i] != 0)
		{
			package->package_data[j] = temp_data_tab[i]; j++;
		}
	}
	package->package_nb_data = STARPU_TASK_GET_NBUFFERS(task) + package->package_nb_data - nb_duplicate_data;
	package->expected_time += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
	starpu_task_list_push_back(&package->sub_list, task);
}

/* Return expected time of the list of task + fill a struct of data on the node,
 * so we can more easily simulate adding, removing task in a list,
 * without re-calculating everything.
 */
static void get_expected_package_computation_time(struct _starpu_HFP_my_list *l, starpu_ssize_t GPU_RAM)
{
	if (l->nb_task_in_sub_list < 1)
	{
		l->expected_package_computation_time = 0;
		return;
	}
	int use_order = 1;
	struct starpu_task *task;
	struct starpu_task *next_task;
	double time_to_add = 0;

	task = starpu_task_list_begin(&l->sub_list);
	/* Init linked list of data in this package */
	l->pointer_node = init_data_list(STARPU_TASK_GET_HANDLE(task, 0));
	l->expected_package_computation_time = starpu_transfer_predict(0, 1, starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, 0)));
	/* Put the remaining data on simulated memory */
	unsigned i;
	for (i = 1; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		insertion_data_on_node(l->pointer_node, STARPU_TASK_GET_HANDLE(task, i), use_order, i, &l->sub_list, task);
		l->expected_package_computation_time += starpu_transfer_predict(0, 1, starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, i)));
		use_order++;
	}
	for (next_task = starpu_task_list_next(task); next_task != starpu_task_list_end(&l->sub_list); next_task = starpu_task_list_next(next_task))
	{
		time_to_add = 0;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(next_task); i++)
		{
			if (SearchTheData(l->pointer_node, STARPU_TASK_GET_HANDLE(next_task, i), use_order) == false)
			{
				if (l->pointer_node->memory_used + starpu_transfer_predict(0, 1, starpu_data_get_size(STARPU_TASK_GET_HANDLE(next_task, i))) <= GPU_RAM)
				{
					insertion_data_on_node(l->pointer_node, STARPU_TASK_GET_HANDLE(next_task, i), use_order, i, &l->sub_list, task);
					use_order++;
					time_to_add += starpu_transfer_predict(0, 1, starpu_data_get_size(STARPU_TASK_GET_HANDLE(next_task, i)));
				}
				else
				{
					/* Need to evict a data and replace it */
					replace_least_recently_used_data(l->pointer_node, STARPU_TASK_GET_HANDLE(next_task, i), use_order, task, &l->sub_list, i);
					use_order++;
					time_to_add += starpu_transfer_predict(0, 1, starpu_data_get_size(STARPU_TASK_GET_HANDLE(next_task, i)));
				}
			}
			else
			{
				/* A data already on memory will be used, need to increment use_order */
				use_order++;
			}
		}
		/* Who cost more time ? Task T_{i-1} or data load from T_{i} */
		if (time_to_add > starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0))
		{
			l->expected_package_computation_time += time_to_add;
		}
		else
		{
			l->expected_package_computation_time += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
		}
		task = starpu_task_list_next(task);
	}
	l->expected_package_computation_time += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
}

/* Equilibrates package in order to have packages with the same expected computation time,
 * including transfers and computation/transfers overlap.
 * Called in HFP_pull_task once all packages are done.
 * It is called when MULTIGPU = 6 or 7.
 */
static void load_balance_expected_package_computation_time(struct _starpu_HFP_paquets *p, starpu_ssize_t GPU_RAM)
{
	//~ if (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") && strcmp(_starpu_HFP_appli, "random_set_of_task") != 0)
	//~ {
		/* What is different mainly is with the task of C that is in won't use for LRU with gemms once it used.
		 * We do something in replace_least_recently_used_data that maybe we can't do in cholesky or random graphs? */
		//~ perror("load_balance_expected_package_computation_time not implemented yet for non-gemm applications\n"); exit(EXIT_FAILURE);
	//~ }
	struct starpu_task *task;
	task = starpu_task_list_begin(&p->temp_pointer_1->sub_list);
	p->temp_pointer_1 = p->first_link;
	while (p->temp_pointer_1 != NULL)
	{
		get_expected_package_computation_time(p->temp_pointer_1, GPU_RAM);
		p->temp_pointer_1 = p->temp_pointer_1->next;
	}

	int package_with_min_expected_time, package_with_max_expected_time;
	int last_package_with_min_expected_time = 0;
	int last_package_with_max_expected_time = 0;
	double min_expected_time, max_expected_time;
	bool load_balance_needed = true;
	//~ int percentage = 1; /* percentage of difference between packages */
	/* Selecting the smallest and biggest package */
	while (load_balance_needed == true)
	{
		p->temp_pointer_1 = p->first_link;
		min_expected_time = p->temp_pointer_1->expected_package_computation_time;
		max_expected_time = p->temp_pointer_1->expected_package_computation_time;
		package_with_min_expected_time = 0;
		package_with_max_expected_time = 0;
		int i = 0;
		p->temp_pointer_1 = p->temp_pointer_1->next;
		while (p->temp_pointer_1 != NULL)
		{
			i++;
			if (min_expected_time > p->temp_pointer_1->expected_package_computation_time)
			{
				min_expected_time = p->temp_pointer_1->expected_package_computation_time;
				package_with_min_expected_time = i;
			}
			if (max_expected_time < p->temp_pointer_1->expected_package_computation_time)
			{
				max_expected_time = p->temp_pointer_1->expected_package_computation_time;
				package_with_max_expected_time = i;
			}
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
		//~ if (_print_in_terminal == 1) { printf("min et max : %f et %f, paquets %d et %d\n",min_expected_time, max_expected_time, package_with_min_expected_time, package_with_max_expected_time); }

		/* To avoid looping indefinitely */
		if (last_package_with_min_expected_time == package_with_max_expected_time && last_package_with_max_expected_time == package_with_min_expected_time)
		{
			break;
		}

		/* Stealing as much task from the last tasks of the biggest packages */
		/* Getting on the right packages */
		p->temp_pointer_1 = p->first_link;
		for (i = 0; i < package_with_min_expected_time; i++)
		{
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
		p->temp_pointer_2 = p->first_link;
		for (i = 0; i < package_with_max_expected_time; i++)
		{
			p->temp_pointer_2 = p->temp_pointer_2->next;
		}
		while (p->temp_pointer_1->expected_package_computation_time >= p->temp_pointer_2->expected_package_computation_time - ((p->temp_pointer_2->expected_package_computation_time*max_expected_time)/100))
		{
			task = starpu_task_list_pop_back(&p->temp_pointer_2->sub_list);
			p->temp_pointer_2->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
			merge_task_and_package(p->temp_pointer_1, task);
			p->temp_pointer_2->nb_task_in_sub_list--;
			free(p->temp_pointer_1->pointer_node);
			free(p->temp_pointer_2->pointer_node);
			get_expected_package_computation_time(p->temp_pointer_1, GPU_RAM);
			get_expected_package_computation_time(p->temp_pointer_2, GPU_RAM);
			if ( p->temp_pointer_1->expected_package_computation_time >= p->temp_pointer_2->expected_package_computation_time)
			{
				break;
			}
		}
		last_package_with_min_expected_time = package_with_min_expected_time;
		last_package_with_max_expected_time = package_with_max_expected_time;
	}
}

/* Called in HFP_pull_task. Cut in half the package and interlace task from end of left part and beginning of right part.
 * This way we alternate with task sharing data (the middle of the package) then end with task sharing few data (extremities).
 * This is only called if environemment value INTERLACING is set te something else than 1.
 * Example: 0 1 2 3 4 5 6 7 8 9 10 -> 5 6 4 7 3 8 2 9 1 10 0 */
static void interlacing_task_list(struct _starpu_HFP_paquets *a)
{
	a->temp_pointer_1 = a->first_link;
	int middle = 0;
	int i = 0;
	struct starpu_task_list sub_list_left;
	starpu_task_list_init(&sub_list_left);
	struct starpu_task_list sub_list_right;
	starpu_task_list_init(&sub_list_right);

	while (a->temp_pointer_1 != NULL)
	{
		middle = a->temp_pointer_1->nb_task_in_sub_list/2;
		if (a->temp_pointer_1->nb_task_in_sub_list%2 == 1)
		{
			/* So the biggest package is the one on the left, the one with which I start. */
			middle++;
		}
		/* Filling two sub_list, right and left */
		for (i = 0; i < middle; i++)
		{
			starpu_task_list_push_back(&sub_list_left, starpu_task_list_pop_front(&a->temp_pointer_1->sub_list));
		}
		for (i = middle; i < a->temp_pointer_1->nb_task_in_sub_list; i++)
		{
			starpu_task_list_push_back(&sub_list_right, starpu_task_list_pop_front(&a->temp_pointer_1->sub_list));
		}
		/* Re-filling the package alterning left and right */
		for (i = 0; i < a->temp_pointer_1->nb_task_in_sub_list; i++)
		{
			if (i%2 == 0)
			{
				starpu_task_list_push_back(&a->temp_pointer_1->sub_list, starpu_task_list_pop_back(&sub_list_left));
			}
			else
			{
				starpu_task_list_push_back(&a->temp_pointer_1->sub_list, starpu_task_list_pop_front(&sub_list_right));
			}
		}
		a->temp_pointer_1 = a->temp_pointer_1->next;
	}
}

/* TODO : a supprimer une fois les mesures du temps terminées */
//~ struct timeval time_start_gettasktoreturn;
//~ struct timeval time_end_gettasktoreturn;
//~ long long time_total_gettasktoreturn = 0;

/* Called in HFP_pull_task when we need to return a task. It is used when we have multiple GPUs
 * In case of modular-heft-HFP, it needs to do a round robin on the task it returned. So we use expected_time_pulled_out,
 * an element of struct my_list in order to track which package pulled out the least expected task time. So heft can can
 * better divide tasks between GPUs */
struct starpu_task *_starpu_HFP_get_task_to_return(struct starpu_sched_component *component, struct starpu_sched_component *to, struct _starpu_HFP_paquets* a, int nb_gpu)
{
	//~ gettimeofday(&time_start_gettasktoreturn, NULL);
	int max_task_time = 0;
	int index_package_max_task_time = 0;
	a->temp_pointer_1 = a->first_link;
	int i = 0; struct starpu_task *task; double min_expected_time_pulled_out = 0; int package_min_expected_time_pulled_out = 0;
	/* If there is only one big package */
	if (multigpu == 0 && _starpu_HFP_hmetis == 0 && _nb_gpus == 1)
	{
		//~ printf("Will pop front\n");
		task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
		_sched_visu_print_data_to_load_prefetch(task, starpu_worker_get_id() + 1, 0);

		//~ gettimeofday(&time_end_gettasktoreturn, NULL);
		//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;
		return task;
	}
	else
	{
		/* If we use modular heft I look at the expected time pulled out of each package to alternate between packages */
		if (modular_heft_hfp_mode != 0)
		{
			package_min_expected_time_pulled_out = 0;
			min_expected_time_pulled_out = DBL_MAX;
			for (i = 0; i < nb_gpu; i++)
			{
				/* We also need to check that the package is not empty */
				if (a->temp_pointer_1->expected_time_pulled_out < min_expected_time_pulled_out && !starpu_task_list_empty(&a->temp_pointer_1->sub_list))
				{
					min_expected_time_pulled_out = a->temp_pointer_1->expected_time_pulled_out;
					package_min_expected_time_pulled_out = i;
				}
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_1 = a->first_link;
			for (i = 0; i < package_min_expected_time_pulled_out; i++)
			{
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
			a->temp_pointer_1->expected_time_pulled_out += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);

				//~ gettimeofday(&time_end_gettasktoreturn, NULL);
	//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;


			return task;
		}
		else
		{
			/* We are using HFP */
			for (i = 0; i < nb_gpu; i++)
			{
				if (to == component->children[i])
				{
					break;
				}
				else
				{
					a->temp_pointer_1 = a->temp_pointer_1->next;
				}
			}
			if (!starpu_task_list_empty(&a->temp_pointer_1->sub_list))
			{
				//~ printf("Will pop front\n");
				task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
				a->temp_pointer_1->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
				a->temp_pointer_1->nb_task_in_sub_list--;
				_sched_visu_print_data_to_load_prefetch(task, starpu_worker_get_id() + 1, 0);

				//~ gettimeofday(&time_end_gettasktoreturn, NULL);
				//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;
				//~ printf("Return %p\n", task);
				return task;
			}
			else
			{
				/* Our current gpu's package is empty, we want to steal! */
				if (task_stealing == 1)
				{
					/* Stealing from package with the most tasks time duration.
					 * temp_pointer_2 = biggest package, temp_pointer_1 = empty package that will steal from temp_pointer_2. */
					a->temp_pointer_2 = a->first_link;
					i = 0;
					max_task_time = a->temp_pointer_2->expected_time;
					index_package_max_task_time = 0;
					while (a->temp_pointer_2->next != NULL)
					{
						a->temp_pointer_2 = a->temp_pointer_2->next;
						i++;
						if (max_task_time < a->temp_pointer_2->expected_time)
						{
							max_task_time = a->temp_pointer_2->expected_time;
							index_package_max_task_time = i;
						}
					}
					if (max_task_time != 0)
					{
						a->temp_pointer_2 = a->first_link;
						for (i = 0; i < index_package_max_task_time; i++)
						{
							a->temp_pointer_2 = a->temp_pointer_2->next;
						}
						task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
						a->temp_pointer_2->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
						a->temp_pointer_2->nb_task_in_sub_list--;
						_sched_visu_print_data_to_load_prefetch(task, starpu_worker_get_id() + 1, 0);

						//~ gettimeofday(&time_end_gettasktoreturn, NULL);
						//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;

						return task;
					}
					else
					{
						return NULL;
					}
				}
				else if (task_stealing == 2 || task_stealing == 3)
				{
					/* Stealing from package with the most expected package time */
					a->temp_pointer_2 = a->first_link;
					while (a->temp_pointer_2 != NULL)
					{
						get_expected_package_computation_time(a->temp_pointer_2, _starpu_HFP_GPU_RAM_M);
						a->temp_pointer_2 = a->temp_pointer_2->next;
					}
					i = 0;
					a->temp_pointer_2 = a->first_link;
					double max_package_time = a->temp_pointer_2->expected_package_computation_time;
					index_package_max_task_time = 0;
					while (a->temp_pointer_2->next != NULL)
					{
						a->temp_pointer_2 = a->temp_pointer_2->next;
						i++;
						if (max_package_time < a->temp_pointer_2->expected_package_computation_time)
						{
							max_package_time = a->temp_pointer_2->expected_package_computation_time;
							index_package_max_task_time = i;
						}
					}
					if (max_package_time != 0)
					{
						a->temp_pointer_2 = a->first_link;
						for (i = 0; i < index_package_max_task_time; i++)
						{
							a->temp_pointer_2 = a->temp_pointer_2->next;
						}
							if (task_stealing == 3)
							{
								/* We steal half of the package in terms of task duration */
								while (a->temp_pointer_1->expected_time < a->temp_pointer_2->expected_time/2)
								{
									/* We steal from the end */
									task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
									a->temp_pointer_2->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
									a->temp_pointer_2->nb_task_in_sub_list--;
									starpu_task_list_push_front(&a->temp_pointer_1->sub_list, task);
									a->temp_pointer_1->expected_time += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
									a->temp_pointer_1->nb_task_in_sub_list++;
								}
								get_expected_package_computation_time(a->temp_pointer_2, _starpu_HFP_GPU_RAM_M);
								task = starpu_task_list_pop_front(&a->temp_pointer_1->sub_list);
								a->temp_pointer_1->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
								a->temp_pointer_1->nb_task_in_sub_list--;
								get_expected_package_computation_time(a->temp_pointer_1, _starpu_HFP_GPU_RAM_M);
							}
							else
							{
								/* We only steal one task */
								task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
								a->temp_pointer_2->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
								a->temp_pointer_2->nb_task_in_sub_list--;
								get_expected_package_computation_time(a->temp_pointer_2, _starpu_HFP_GPU_RAM_M);
							}
							_sched_visu_print_data_to_load_prefetch(task, starpu_worker_get_id() + 1, 0);

							//~ gettimeofday(&time_end_gettasktoreturn, NULL);
							//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;
							return task;
					}
					else
					{
						/* Nothing to steal */
						//~ gettimeofday(&time_end_gettasktoreturn, NULL);
						//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;
						return NULL;
					}
				}
				else
				{
					/* We don't use task stealing */
					//~ gettimeofday(&time_end_gettasktoreturn, NULL);
					//~ time_total_gettasktoreturn += (time_end_gettasktoreturn.tv_sec - time_start_gettasktoreturn.tv_sec)*1000000LL + time_end_gettasktoreturn.tv_usec - time_start_gettasktoreturn.tv_usec;
					return NULL;
				}
			}
		}
	}
}

/* Giving prefetch for each task to modular-heft-HFP */
static void prefetch_each_task(struct _starpu_HFP_paquets *a, struct starpu_sched_component *component)
{
	struct starpu_task *task;
	int i = 0;
	a->temp_pointer_1 = a->first_link;

	while (a->temp_pointer_1 != NULL)
	{
		for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
		{
			/* Putting in workerid the information of the chosen gpu HFP. Then in helper_mct, we can use this information to influence the expected time */
			task->workerid = i;
			if (modular_heft_hfp_mode == 1)
			{
				starpu_prefetch_task_input_on_node_prio(task, starpu_worker_get_memory_node(starpu_bitmap_first(&component->children[0]->children[i]->workers_in_ctx)), 0);
			}
			else if (modular_heft_hfp_mode == 2)
			{
				starpu_idle_prefetch_task_input_on_node_prio(task, starpu_worker_get_memory_node(starpu_bitmap_first(&component->children[0]->children[i]->workers_in_ctx)), 0);
			}
			else
			{
				printf("Wrong environement variable MODULAR_HEFT_HFP_MODE\n");
				exit(0);
			}
		}
		a->temp_pointer_1 = a->temp_pointer_1->next; printf("next\n");
		i++;
	}
}

//static int get_max_value_common_data_matrix(struct _starpu_HFP_paquets *p, int GPU_limit_switch, int number_task, int min_nb_task_in_sub_list, long int matrice_donnees_commune[][number_task])
//{
//	struct _starpu_HFP_my_list *l1 = p->first_link;
//	struct _starpu_HFP_my_list *l2 = p->first_link;
//	int i_bis = 0;
//	int j_bis = 0;
//
//	int max_value_common_data_matrix = 0;
//	for (i_bis = 0; i_bis < number_task; i_bis++)
//	{
//		if (l1->nb_task_in_sub_list == min_nb_task_in_sub_list)
//		{
//			for (l2 = p->first_link; l2 != NULL; l2 = l2->next)
//			{
//				if (i_bis != j_bis)
//				{
//					//~ if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis])
//					if (max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis] && ((GPU_limit_switch == 0) || (GPU_limit_switch == 1 && (l1->data_weight + l2->data_weight - matrice_donnees_commune[i_bis][j_bis]))))
//					{
//						max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
//					}
//				}
//				j_bis++;
//			}
//		}
//		l1 = l1->next;
//		j_bis = 0;
//	}
//	return max_value_common_data_matrix;
//}

/* Pushing the tasks */
static int HFP_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct _starpu_HFP_sched_data *data = component->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* TODO : a supprimer une fois les mesures du temps terminées */
//~ struct timeval time_start_scheduling;
//~ struct timeval time_end_scheduling;
//~ long long time_total_scheduling = 0;
//~ struct timeval time_start_find_min_size;
//~ struct timeval time_end_find_min_size;
//~ long long time_total_find_min_size = 0;
//~ struct timeval time_start_init_packages;
//~ struct timeval time_end_init_packages;
//~ long long time_total_init_packages = 0;
//~ struct timeval time_start_fill_matrix_common_data_plus_get_max;
//~ struct timeval time_end_fill_matrix_common_data_plus_get_max;
//~ long long time_total_fill_matrix_common_data_plus_get_max = 0;
//~ struct timeval time_start_order_u_total;
//~ struct timeval time_end_order_u_total;
//~ long long time_total_order_u_total = 0;
//~ struct timeval time_start_reset_init_start_while_loop;
//~ struct timeval time_end_reset_init_start_while_loop;
//~ long long time_total_reset_init_start_while_loop = 0;
//~ struct timeval time_start_merge;
//~ struct timeval time_end_merge;
//~ long long time_total_merge = 0;
//~ struct timeval time_start_iteration_i;
//~ struct timeval time_end_iteration_i;
//~ long long time_total_iteration_i = 0;

/* Need an empty data paquets_data to build packages
 * Output a task list ordered. So it's HFP if we have only one package at the end
 * Used for now to reorder task inside a package after load balancing
 * Can be used as main HFP like in pull task later
 * Things commented are things to print matrix or things like that.
 */
static struct _starpu_HFP_paquets* hierarchical_fair_packing(struct starpu_task_list *task_list, int number_task, int number_of_package_to_build)
{
	struct _starpu_HFP_paquets *paquets_data = malloc(sizeof(*paquets_data));
	struct _starpu_HFP_my_list *my_data = malloc(sizeof(*my_data));
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);
	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
	struct starpu_task_list non_connexe;
	starpu_task_list_init(&non_connexe);
	int nb_duplicate_data = 0; /* Used to store the weight the merging of two packages would be. It is then used to see if it's inferior to the size of the RAM of the GPU */
	long int max_value_common_data_matrix = 0; /* Store the maximum weight of the commons data between two packages for all the tasks */
	long int common_data_last_package_i1_j1 = 0; /* Variables used to compare the affinity between sub package 1i and 1j, 1i and 2j etc... */
	long int common_data_last_package_i1_j2 = 0; long int common_data_last_package_i2_j1 = 0;
	long int common_data_last_package_i2_j2 = 0; long int max_common_data_last_package = 0;
	long int weight_package_i = 0; /* Used for ORDER_U too */
	long int weight_package_j = 0;
	int GPU_limit_switch = 1; int tab_runner = 0;
	int index_head_1 = 0;
	int index_head_2 = 0;
	int common_data_last_package_i2_j = 0;
	int common_data_last_package_i1_j = 0;
	int common_data_last_package_i_j1 = 0;
	int common_data_last_package_i_j2 = 0;
	int min_nb_task_in_sub_list = 0;
	struct starpu_task *task; int nb_of_loop = 0;
	int packaging_impossible = 0;
	int n_duplicate_cho = 0;

	/* One task == one link in the linked list */
	int do_not_add_more = number_task - 1;

	while (!starpu_task_list_empty(task_list))
	{
		task = starpu_task_list_pop_front(task_list);
		_sched_visu_print_data_for_task(task, "Task %p using data(s)");

		paquets_data->temp_pointer_1->expected_time = starpu_task_expected_length(task, starpu_worker_get_perf_archtype(0, 0), 0);
		paquets_data->temp_pointer_1->data_weight = 0;
		paquets_data->temp_pointer_1->data_to_evict_next = NULL; /* Mise à NULL de data to evict next pour eviter les pb en réel sur grid5k */

		/* Si on est sur Cholesky je vire les doublons de données au sein d'une tâche */
		if (strcmp(_starpu_HFP_appli, "chol_model_11") == 0)
		{
			n_duplicate_cho = 0;

			/* Getting the number of duplicate and filling a temp_tab */
			starpu_data_handle_t *temp_data_tab_cho = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task) - 1; i++)
			{
				if (STARPU_TASK_GET_HANDLE(task, i) == STARPU_TASK_GET_HANDLE(task, i + 1))
				{
					n_duplicate_cho++;
					temp_data_tab_cho[i] = NULL;
				}
				else
				{
					temp_data_tab_cho[i] = STARPU_TASK_GET_HANDLE(task, i);
				}
			}
			temp_data_tab_cho[i] = STARPU_TASK_GET_HANDLE(task, i);
			paquets_data->temp_pointer_1->package_data = malloc((STARPU_TASK_GET_NBUFFERS(task) - n_duplicate_cho)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
			int j = 0;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (temp_data_tab_cho[i] != NULL)
				{
					paquets_data->temp_pointer_1->package_data[j] = temp_data_tab_cho[i];
					j++;
				}
			}
			paquets_data->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(task) - n_duplicate_cho;
		}
		else
		{
			paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));

			/* Mise à NULL de data to evict next pour eviter les pb en réel sur grid5k */
			paquets_data->temp_pointer_1->data_to_evict_next = NULL;

			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				paquets_data->temp_pointer_1->package_data[i] = STARPU_TASK_GET_HANDLE(task,i);
				paquets_data->temp_pointer_1->data_weight += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task,i));
			}
			paquets_data->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(task);
		}

		/* We sort our data in the packages */
		qsort(paquets_data->temp_pointer_1->package_data, paquets_data->temp_pointer_1->package_nb_data, sizeof(paquets_data->temp_pointer_1->package_data[0]), HFP_pointeurComparator);

		/* Pushing the task and the number of the package in the package */
		starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list, task);
		/* Initialization of the lists last_packages */
		paquets_data->temp_pointer_1->split_last_ij = 0;
		paquets_data->temp_pointer_1->nb_task_in_sub_list = 1;

		if(do_not_add_more != 0)
		{
			_starpu_HFP_insertion(paquets_data);

			/*TODO utile ??? */
			//~ paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
		}
		do_not_add_more--;
	}
	paquets_data->first_link = paquets_data->temp_pointer_1;
	paquets_data->temp_pointer_2 = paquets_data->first_link;
	index_head_2++;
	paquets_data->NP = _starpu_HFP_NT;

	/* THE while loop. Stop when no more packaging are possible */
	while (packaging_impossible == 0)
	{
	beginning_while_packaging_impossible:
		nb_of_loop++;
		packaging_impossible = 1;

		/* Then we create the common data matrix */
		long int matrice_donnees_commune[number_task][number_task];
		int i;
		for (i = 0; i < number_task; i++)
		{
			int j;
			for (j = 0; j < number_task; j++)
			{
				matrice_donnees_commune[i][j] = 0;
			}
		}

		/* Faster first iteration by grouping together tasks that share at least one data. Doesn't look
		 * further after one task have been found */
		//~ if (nb_of_loop == 1 && strcmp(_starpu_HFP_appli, "chol_model_11") != 0 && faster_first_iteration == 1)
		if (nb_of_loop == 1 && faster_first_iteration == 1)
		{
			packaging_impossible = 0;
			index_head_1 = 0;
			index_head_2 = 0;
			for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
			{
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == 1)
				{
					for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
					{
						if (index_head_1 != index_head_2 && paquets_data->temp_pointer_2->nb_task_in_sub_list == 1)
						{
							for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++)
							{
								int j;
								for (j = 0; j < paquets_data->temp_pointer_2->package_nb_data; j++)
								{
									if (paquets_data->temp_pointer_1->package_data[i] == paquets_data->temp_pointer_2->package_data[j])
									{
										/* Merge */
										//~ printf("On va merge le paquet %d et le paquet %d dans nb of loop == 1.\n", index_head_1, index_head_2);
										paquets_data->NP--;

										paquets_data->temp_pointer_1->split_last_ij = paquets_data->temp_pointer_1->nb_task_in_sub_list;

										/* Fusion des listes de tâches */
										while (!starpu_task_list_empty(&paquets_data->temp_pointer_2->sub_list))
										{
											starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list, starpu_task_list_pop_front(&paquets_data->temp_pointer_2->sub_list));
										}
										paquets_data->temp_pointer_1->nb_task_in_sub_list += paquets_data->temp_pointer_2->nb_task_in_sub_list;

										int i_bis = 0;
										int j_bis = 0;
										tab_runner = 0;
										nb_duplicate_data = 0;
										/* Fusion des tableaux de données */
										//~ _print_in_terminal ("malloc de %d.\n", paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data);

										starpu_data_handle_t *temp_data_tab = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data) * sizeof(paquets_data->temp_pointer_1->package_data[0]));
										while (i_bis < paquets_data->temp_pointer_1->package_nb_data && j_bis < paquets_data->temp_pointer_2->package_nb_data)
										{
											if (paquets_data->temp_pointer_1->package_data[i_bis] == paquets_data->temp_pointer_2->package_data[j_bis])
											{
												temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
												temp_data_tab[tab_runner + 1] = paquets_data->temp_pointer_2->package_data[j_bis];
												i_bis++;
												j_bis++;
												tab_runner++;
												nb_duplicate_data++;
											}
											else if (paquets_data->temp_pointer_1->package_data[i_bis] < paquets_data->temp_pointer_2->package_data[j_bis])
											{
												temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
												i_bis++;
											}
											else
											{
												temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
												j_bis++;
											}
											tab_runner++;
										}
										/* Remplissage en vidant les données restantes du paquet I ou J */
										while (i_bis < paquets_data->temp_pointer_1->package_nb_data)
										{
											temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
											i_bis++;
											tab_runner++;
										}
										while (j_bis < paquets_data->temp_pointer_2->package_nb_data)
										{
											temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
											j_bis++;
											tab_runner++;
										}
										/* Remplissage du tableau de données en ignorant les doublons */
										paquets_data->temp_pointer_1->data_weight = 0;
										//~ print_in_terminal ("malloc de %d.\n", paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data - nb_duplicate_data);
										paquets_data->temp_pointer_1->package_data = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
										j_bis = 0;
										for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++)
										{
											paquets_data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis];

											paquets_data->temp_pointer_1->data_weight += starpu_data_get_size(temp_data_tab[i_bis]);

											if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1])
											{
												i_bis++;
											}
											j_bis++;
										}

										/* Fusion du nombre de données et du temps prévu */
										paquets_data->temp_pointer_1->package_nb_data = paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data - nb_duplicate_data;
										paquets_data->temp_pointer_1->expected_time += paquets_data->temp_pointer_2->expected_time;

										/* Il faut le mettre à 0 pour le suppr ensuite dans HFP_delete_link */
										paquets_data->temp_pointer_2->package_nb_data = 0;
										paquets_data->temp_pointer_2->nb_task_in_sub_list = 0;

										//~ for (i_bis = 0; i_bis < paquets_data->temp_pointer_1->package_nb_data; i_bis++)
										//~ {
											//~ printf("%p ", paquets_data->temp_pointer_1->package_data[i_bis]);
										//~ }
										//~ printf("\n");

										goto start_loop_1;

									}
								}
							}
						}
						index_head_2++;
					}
				}
				start_loop_1:
				index_head_1++;
				index_head_2 = 0;
			}
			goto break_merging_1;
		}

		//~ gettimeofday(&time_start_reset_init_start_while_loop, NULL);

		/* Variables we need to reinitialize for a new iteration */
		paquets_data->temp_pointer_1 = paquets_data->first_link;
		paquets_data->temp_pointer_2 = paquets_data->first_link;
		index_head_1 = 0;
		index_head_2 = 0;
		tab_runner = 0;
		//~ nb_min_task_packages = 0;
		min_nb_task_in_sub_list = 0;
		max_value_common_data_matrix = 0;
		min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list;

		//~ gettimeofday(&time_end_reset_init_start_while_loop, NULL);
		//~ time_total_reset_init_start_while_loop += (time_end_reset_init_start_while_loop.tv_sec - time_start_reset_init_start_while_loop.tv_sec)*1000000LL + time_end_reset_init_start_while_loop.tv_usec - time_start_reset_init_start_while_loop.tv_usec;
		/* First we get the number of packages that have the minimal number of tasks */

		//~ gettimeofday(&time_start_find_min_size, NULL);

		for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		{
			if (min_nb_task_in_sub_list > paquets_data->temp_pointer_1->nb_task_in_sub_list)
			{
				min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list;
			}
		}
		//~ gettimeofday(&time_end_find_min_size, NULL);
		//~ time_total_find_min_size += (time_end_find_min_size.tv_sec - time_start_find_min_size.tv_sec)*1000000LL + time_end_find_min_size.tv_usec - time_start_find_min_size.tv_usec;

		//~ for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		//~ {
			//~ if (min_nb_task_in_sub_list == paquets_data->temp_pointer_1->nb_task_in_sub_list)
			//~ {
				//~ nb_min_task_packages++;
			//~ }
		//~ }
		//~ if (_print_in_terminal == 1) {  printf("Il y a %d paquets de taille minimale %d tâche(s)\n", nb_min_task_packages, min_nb_task_in_sub_list); }

		/* Remplissage de la matrice + obtention du max du poids */
		/* Ancienne version quadratique */
		//~ for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		//~ {
			//~ if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
			//~ {
				//~ for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
				//~ {
					//~ if (index_head_1 != index_head_2)
					//~ {
						//~ for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++)
						//~ {
							//~ for (j = 0; j < paquets_data->temp_pointer_2->package_nb_data; j++)
							//~ {
								//~ printf("On compare %p et %p.\n", paquets_data->temp_pointer_1->package_data[i], paquets_data->temp_pointer_2->package_data[j]);
								//~ if ((paquets_data->temp_pointer_1->package_data[i] == paquets_data->temp_pointer_2->package_data[j]))
								//~ {
									//~ matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]);
								//~ }
							//~ }
						//~ }
						//~ if (max_value_common_data_matrix < matrice_donnees_commune[index_head_1][index_head_2] && (GPU_limit_switch == 0 || (GPU_limit_switch == 1 && (paquets_data->temp_pointer_1->data_weight + paquets_data->temp_pointer_2->data_weight - matrice_donnees_commune[index_head_1][index_head_2]) <= GPU_RAM_M)))
						//~ {
							//~ /* Sinon on met la valeur */
							//~ max_value_common_data_matrix = matrice_donnees_commune[index_head_1][index_head_2];
						//~ }
					//~ }
					//~ index_head_2++;
				//~ }
			//~ }
			//~ index_head_1++;
			//~ index_head_2 = 0;
		//~ }

		/* Nouvelle version linéaire */

		//~ gettimeofday(&time_start_fill_matrix_common_data_plus_get_max, NULL);

		for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		{
			if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
			{
				for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
				{
					if (index_head_1 != index_head_2)
					{
						i = 0;
						int j = 0;
						while (i < paquets_data->temp_pointer_1->package_nb_data && j < paquets_data->temp_pointer_2->package_nb_data)
						{
							if (paquets_data->temp_pointer_1->package_data[i] == paquets_data->temp_pointer_2->package_data[j])
							{
								matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]);
								i++;
								j++;
							}
							else if (paquets_data->temp_pointer_1->package_data[i] > paquets_data->temp_pointer_2->package_data[j])
							{
								j++;
							}
							else if (paquets_data->temp_pointer_1->package_data[i] < paquets_data->temp_pointer_2->package_data[j])
							{
								i++;
							}
						}
						if (max_value_common_data_matrix < matrice_donnees_commune[index_head_1][index_head_2] && (GPU_limit_switch == 0 || (GPU_limit_switch == 1 && (paquets_data->temp_pointer_1->data_weight + paquets_data->temp_pointer_2->data_weight - matrice_donnees_commune[index_head_1][index_head_2]) <= _starpu_HFP_GPU_RAM_M)))
						{
							max_value_common_data_matrix = matrice_donnees_commune[index_head_1][index_head_2];
						}
					}
					index_head_2++;
				}
			}
			index_head_1++;
			index_head_2 = 0;
		}

		//~ gettimeofday(&time_end_fill_matrix_common_data_plus_get_max, NULL);
		//~ time_total_fill_matrix_common_data_plus_get_max += (time_end_fill_matrix_common_data_plus_get_max.tv_sec - time_start_fill_matrix_common_data_plus_get_max.tv_sec)*1000000LL + time_end_fill_matrix_common_data_plus_get_max.tv_usec - time_start_fill_matrix_common_data_plus_get_max.tv_usec;

		/* Code to print the common data matrix */
		//~ if (_print_in_terminal == 1) { printf("Common data matrix : \n"); for (i = 0; i < number_task; i++) { for (j = 0; j < number_task; j++) { printf(" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); } }

		/* Ne fonctionne que en mono GPU pour les matrices sparses :/. */
		if (max_value_common_data_matrix == 0 && GPU_limit_switch == 0)
		{
			/* It means that P_i share no data with others, so we put it in the end of the list
			 * For this we use a separate list that we merge at the end
			 * We will put this list at the end of the rest of the packages */
			//~ if (_print_in_terminal == 1) { printf("graphe non connexe\n"); printf("%d paquets.\n", paquets_data->NP); }
			paquets_data->temp_pointer_1 = paquets_data->first_link;
			for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
			{
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
				{
					while (!starpu_task_list_empty(&paquets_data->temp_pointer_1->sub_list))
					{
						starpu_task_list_push_back(&non_connexe, starpu_task_list_pop_front(&paquets_data->temp_pointer_1->sub_list));
					}
					paquets_data->temp_pointer_1->package_nb_data = 0;
					paquets_data->NP--;
					//~ printf("--\n");
				}
			}

			/* Il ne faut pas supprimer le dernier paquet qu'il nous reste evidemment. */
			if (paquets_data->NP < _nb_gpus)
			{
				goto end_while_packaging_impossible;
			}
			HFP_delete_link(paquets_data);
			number_task = paquets_data->NP;
			goto beginning_while_packaging_impossible;
		}
		else if (max_value_common_data_matrix == 0)
		{
			GPU_limit_switch = 0;
			goto beginning_while_packaging_impossible;
		}
		else /* Searching the package that get max and merge them */
		{
			paquets_data->temp_pointer_1 = paquets_data->first_link;
			paquets_data->temp_pointer_2 = paquets_data->first_link;
			for (i = 0; i < number_task; i++)
			{
				//~ printf("i = %d, number task = %d.\n", i, number_task);
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
				{
					int j;
					for (j = 0; j < number_task; j++)
					{
						if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j)
						{
							/* Merge */
							packaging_impossible = 0;
							//~ printf("On va merge le paquet %d et le paquet %d. Ils ont %ld en commun. Ils ont %d et %d tâches.\n", i, j, max_value_common_data_matrix, paquets_data->temp_pointer_1->nb_task_in_sub_list, paquets_data->temp_pointer_2->nb_task_in_sub_list);

							paquets_data->NP--;

							//~ gettimeofday(&time_start_order_u_total, NULL);

							if (order_u == 1)
							{
								//~ printf("Début U\n");
								weight_package_i = paquets_data->temp_pointer_1->data_weight;
								weight_package_j = paquets_data->temp_pointer_2->data_weight;
								if (paquets_data->temp_pointer_1->nb_task_in_sub_list != 1 && paquets_data->temp_pointer_2->nb_task_in_sub_list != 1)
								{
									if (weight_package_i > _starpu_HFP_GPU_RAM_M && weight_package_j <= _starpu_HFP_GPU_RAM_M)
									{
										common_data_last_package_i1_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 0, false,_starpu_HFP_GPU_RAM_M);
										common_data_last_package_i2_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 0, false,_starpu_HFP_GPU_RAM_M);
										if (common_data_last_package_i1_j > common_data_last_package_i2_j)
										{
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
										}
									}
									else if (weight_package_i <= _starpu_HFP_GPU_RAM_M && weight_package_j > _starpu_HFP_GPU_RAM_M)
									{
										common_data_last_package_i_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 1, false, _starpu_HFP_GPU_RAM_M);
										common_data_last_package_i_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 2, false, _starpu_HFP_GPU_RAM_M);
										if (common_data_last_package_i_j2 > common_data_last_package_i_j1)
										{
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
									}
									else
									{
										if (weight_package_i > _starpu_HFP_GPU_RAM_M && weight_package_j > _starpu_HFP_GPU_RAM_M)
										{
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, false,_starpu_HFP_GPU_RAM_M);
										}
										else if (weight_package_i <= _starpu_HFP_GPU_RAM_M && weight_package_j <= _starpu_HFP_GPU_RAM_M)
										{
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, true,_starpu_HFP_GPU_RAM_M);
										}
										else
										{
											printf("Erreur dans ordre U, aucun cas choisi\n"); fflush(stdout);
											exit(0);
										}
										max_common_data_last_package = common_data_last_package_i2_j1;
										if (max_common_data_last_package < common_data_last_package_i1_j1) { max_common_data_last_package = common_data_last_package_i1_j1; }
										if (max_common_data_last_package < common_data_last_package_i1_j2) { max_common_data_last_package = common_data_last_package_i1_j2; }
										if (max_common_data_last_package < common_data_last_package_i2_j2) { max_common_data_last_package = common_data_last_package_i2_j2; }
										if (max_common_data_last_package == common_data_last_package_i1_j2)
										{
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
										else if (max_common_data_last_package == common_data_last_package_i2_j2)
										{
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
										else if (max_common_data_last_package == common_data_last_package_i1_j1)
										{
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
										}
									}
								}
							}
							//~ printf("Fin U\n");
							//~ gettimeofday(&time_end_order_u_total, NULL);
							//~ time_total_order_u_total += (time_end_order_u_total.tv_sec - time_start_order_u_total.tv_sec)*1000000LL + time_end_order_u_total.tv_usec - time_start_order_u_total.tv_usec;

							//~ gettimeofday(&time_start_merge, NULL);

							paquets_data->temp_pointer_1->data_weight = paquets_data->temp_pointer_1->data_weight + paquets_data->temp_pointer_2->data_weight - matrice_donnees_commune[i][j];

							/* Mise à 0 pour ne pas re-merge ces tableaux */
							int j_bis;
							for (j_bis = 0; j_bis < number_task; j_bis++)
							{
								matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;
							}
							for (j_bis = 0; j_bis < number_task; j_bis++)
							{
								matrice_donnees_commune[j][j_bis] = 0; matrice_donnees_commune[j_bis][j] = 0;
							}

							paquets_data->temp_pointer_1->split_last_ij = paquets_data->temp_pointer_1->nb_task_in_sub_list;

							/* Fusion des listes de tâches */
							paquets_data->temp_pointer_1->nb_task_in_sub_list += paquets_data->temp_pointer_2->nb_task_in_sub_list;
							while (!starpu_task_list_empty(&paquets_data->temp_pointer_2->sub_list))
							{
								starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list, starpu_task_list_pop_front(&paquets_data->temp_pointer_2->sub_list));
							}

							int i_bis = 0;
							j_bis = 0;
							tab_runner = 0;
							nb_duplicate_data = 0;

							/* Fusion des tableaux de données */
							starpu_data_handle_t *temp_data_tab = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data) * sizeof(paquets_data->temp_pointer_1->package_data[0]));
							while (i_bis < paquets_data->temp_pointer_1->package_nb_data && j_bis < paquets_data->temp_pointer_2->package_nb_data)
							{
								if (paquets_data->temp_pointer_1->package_data[i_bis] == paquets_data->temp_pointer_2->package_data[j_bis])
								{
									temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
									temp_data_tab[tab_runner + 1] = paquets_data->temp_pointer_2->package_data[j_bis];
									i_bis++;
									j_bis++;
									tab_runner++;
									nb_duplicate_data++;
								}
								else if (paquets_data->temp_pointer_1->package_data[i_bis] < paquets_data->temp_pointer_2->package_data[j_bis])
								{
									temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
									i_bis++;
								}
								else
								{
									temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
									j_bis++;
								}
								tab_runner++;
							}
							/* Remplissage en vidant les données restantes du paquet I ou J */
							while (i_bis < paquets_data->temp_pointer_1->package_nb_data)
							{
								temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
								i_bis++;
								tab_runner++;
							}
							while (j_bis < paquets_data->temp_pointer_2->package_nb_data)
							{
								temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
								j_bis++;
								tab_runner++;
							}
							//~ printf("Nb duplicate data = %d.\n", nb_duplicate_data);

							//~ for (i_bis = 0; i_bis < paquets_data->temp_pointer_1->package_nb_data; i_bis++)
							//~ {
								//~ printf("%p ", paquets_data->temp_pointer_1->package_data[i_bis]);
							//~ }
							//~ printf("\n");
							//~ for (i_bis = 0; i_bis < paquets_data->temp_pointer_2->package_nb_data; i_bis++)
							//~ {
								//~ printf("%p ", paquets_data->temp_pointer_2->package_data[i_bis]);
							//~ }
							//~ printf("\n");

							/* Remplissage du tableau de données en ignorant les doublons */
							//~ printf("malloc dans le vrai de %d.\n", paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data);
							paquets_data->temp_pointer_1->package_data = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
							//~ paquets_data->temp_pointer_1->package_data = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(paquets_data->temp_pointer_2->package_data[0]));
							//~ printf("Apres le malloc.\n"); fflush(stdout);
							j_bis = 0;
							for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++)
							{
								//~ printf("getting %p.\n", temp_data_tab[i_bis]);
								paquets_data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis];
								if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1])
								{
									i_bis++;
								}
								j_bis++;
							}
							//~ printf("Avant fusion des chiffres.\n");
							/* Fusion du nombre de données et du temps prévu */
							paquets_data->temp_pointer_1->package_nb_data = paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data - nb_duplicate_data;
							paquets_data->temp_pointer_1->expected_time += paquets_data->temp_pointer_2->expected_time;

							/* Il faut le mettre à 0 pour le suppr ensuite dans HFP_delete_link */
							paquets_data->temp_pointer_2->package_nb_data = 0;

							//~ nb_duplicate_data = 0;

							//~ gettimeofday(&time_end_merge, NULL);
							//~ time_total_merge += (time_end_merge.tv_sec - time_start_merge.tv_sec)*1000000LL + time_end_merge.tv_usec - time_start_merge.tv_usec;
							if(paquets_data->NP == number_of_package_to_build) { goto break_merging_1; }
							//~ printf("Fin du merge.\n");
							break;
						}
						paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next;
					}
				}
				paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next;
				paquets_data->temp_pointer_2 = paquets_data->first_link;
			}
		}

		break_merging_1:
		//~ printf("break merging.\n");
		paquets_data->temp_pointer_1 = HFP_delete_link(paquets_data);
		//~ printf("After delete %d.\n", paquets_data->NP);
		/* Checking if we have the right number of packages. if MULTIGPU is equal to 0 we want only one package. if it is equal to 1 we want |GPU| packages */
		if (paquets_data->NP == number_of_package_to_build)
		{
			goto end_while_packaging_impossible;
		}
		else if (paquets_data->NP == 1) /* If we have only one package we don't have to do more packages */
		{
			goto end_while_packaging_impossible;
		}
		else /* Reset number of packages for the matrix initialisation */
		{
			number_task = paquets_data->NP;
			//~ printf("la1.\n");
		}

		//~ if ((iteration == 3 && starpu_get_env_number_default("PRINT_TIME", 0) == 1) || starpu_get_env_number_default("PRINT_TIME", 0) == 2)
		//~ {
			//~ printf("la2.\n");
			//~ gettimeofday(&time_end_iteration_i, NULL);
			//~ time_total_iteration_i = (time_end_iteration_i.tv_sec - time_start_iteration_i.tv_sec)*1000000LL + time_end_iteration_i.tv_usec - time_start_iteration_i.tv_usec;
			//~ FILE *f = fopen("Output_maxime/HFP_iteration_time.txt", "a");
			//~ fprintf(f, "%d	%lld\n", nb_of_loop, time_total_iteration_i);
			//~ fclose(f);
		//~ }
	} /* End of while (packaging_impossible == 0) { */

	end_while_packaging_impossible:
	//~ if ((iteration == 3 && starpu_get_env_number_default("PRINT_TIME", 0) == 1) || starpu_get_env_number_default("PRINT_TIME", 0) == 2)
	//~ {
		//~ gettimeofday(&time_end_iteration_i, NULL);
		//~ time_total_iteration_i = (time_end_iteration_i.tv_sec - time_start_iteration_i.tv_sec)*1000000LL + time_end_iteration_i.tv_usec - time_start_iteration_i.tv_usec;
		//~ FILE *f = fopen("Output_maxime/HFP_iteration_time.txt", "a");
		//~ fprintf(f, "%d	%lld\n", nb_of_loop, time_total_iteration_i);
		//~ fclose(f);
	//~ }

	/* Add tasks or packages that were not connexe */
	while(!starpu_task_list_empty(&non_connexe))
	{
		starpu_task_list_push_back(&paquets_data->first_link->sub_list, starpu_task_list_pop_front(&non_connexe));
		paquets_data->first_link->nb_task_in_sub_list++;
	}
	//~ gettimeofday(&time_end_scheduling, NULL);
	//~ time_total_scheduling += (time_end_scheduling.tv_sec - time_start_scheduling.tv_sec)*1000000LL + time_end_scheduling.tv_usec - time_start_scheduling.tv_usec;

	return paquets_data;
}

/* TODO : attention ne fonctinne pas car non corrigé par rapport aux corrections ci dessus (la complexité, le fait
 * de ne pas répéter le get_max_value_common_data_matrix, la première itration simplifié et le calcul des intersections
 * pour la matrice en temps linéaire
 */
static struct starpu_task_list hierarchical_fair_packing_one_task_list (struct starpu_task_list task_list, int number_task)
{
	struct _starpu_HFP_paquets *paquets_data = malloc(sizeof(*paquets_data));
	struct _starpu_HFP_my_list *my_data = malloc(sizeof(*my_data));
	starpu_task_list_init(&my_data->sub_list);
	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;

	int number_of_package_to_build = 1;
	struct starpu_task_list non_connexe;
	starpu_task_list_init(&non_connexe);
	int nb_duplicate_data = 0; long int weight_two_packages; /* Used to store the weight the merging of two packages would be. It is then used to see if it's inferior to the size of the RAM of the GPU */
	long int max_value_common_data_matrix = 0; /* Store the maximum weight of the commons data between two packages for all the tasks */
	long int common_data_last_package_i1_j1 = 0; /* Variables used to compare the affinity between sub package 1i and 1j, 1i and 2j etc... */
	long int common_data_last_package_i1_j2 = 0; long int common_data_last_package_i2_j1 = 0;
	long int common_data_last_package_i2_j2 = 0; long int max_common_data_last_package = 0;
	long int weight_package_i = 0; /* Used for ORDER_U too */
	long int weight_package_j = 0; int i = 0;
	int bool_data_common = 0; int GPU_limit_switch = 1; int temp_nb_min_task_packages = 0; int i_bis = 0; int j_bis = 0; int tab_runner = 0; int index_head_1 = 0; int index_head_2 = 0; int common_data_last_package_i2_j = 0; int common_data_last_package_i1_j = 0; int common_data_last_package_i_j1 = 0; int common_data_last_package_i_j2 = 0;
	int min_nb_task_in_sub_list = 0; int nb_min_task_packages = 0;
	struct starpu_task *task; int nb_of_loop = 0; int packaging_impossible = 0; int link_index = 0; int NB_TOTAL_DONNEES = 0;
	task  = starpu_task_list_begin(&task_list);
	paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
	struct starpu_task *temp_task;

	//~ task  = starpu_task_list_begin(&task_list);
	//~ paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
	/* One task == one link in the linked list */
	int do_not_add_more = number_task - 1;
	for (task = starpu_task_list_begin(&task_list); task != starpu_task_list_end(&task_list); task = temp_task)
	{
		temp_task = starpu_task_list_next(task);
		task = starpu_task_list_pop_front(&task_list);

		paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));

		unsigned j;
		for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
		{
			paquets_data->temp_pointer_1->package_data[j] = STARPU_TASK_GET_HANDLE(task, j);
		}
		paquets_data->temp_pointer_1->package_nb_data = STARPU_TASK_GET_NBUFFERS(task);
		NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(task);
		/* We sort our data in the packages */
		qsort(paquets_data->temp_pointer_1->package_data,paquets_data->temp_pointer_1->package_nb_data,sizeof(paquets_data->temp_pointer_1->package_data[0]),HFP_pointeurComparator);
		/* Pushing the task and the number of the package in the package*/
		starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list,task);
		/* Initialization of the lists last_packages */
		paquets_data->temp_pointer_1->split_last_ij = 0;
		link_index++;
		paquets_data->temp_pointer_1->nb_task_in_sub_list=1;

		if(do_not_add_more != 0)
		{
			_starpu_HFP_insertion(paquets_data);
			paquets_data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(paquets_data->temp_pointer_1->package_data[0]));
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

	/* if (_print_in_terminal == 1) { init_visualisation_tache_matrice_format_tex(); } */
	/* THE while loop. Stop when no more packaging are possible */
	while (packaging_impossible == 0)
	{
	beginning_while_packaging_impossible:
		nb_of_loop++;
		packaging_impossible = 1;
		//~ if (_print_in_terminal == 1) { printf("############# Itération numéro : %d #############\n",nb_of_loop); }

		/* Variables we need to reinitialize for a new iteration */
		paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link; index_head_1 = 0; index_head_2 = 1; link_index = 0; tab_runner = 0; nb_min_task_packages = 0;
		min_nb_task_in_sub_list = 0; weight_two_packages = 0; max_value_common_data_matrix = 0; long int matrice_donnees_commune[number_task][number_task];
		min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list;
		for (i = 0; i < number_task; i++)
		{
			int j;
			for (j = 0; j < number_task; j++)
			{
				matrice_donnees_commune[i][j] = 0;
			}
		}

		/* First we get the number of packages that have the minimal number of tasks */
		for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		{
			if (min_nb_task_in_sub_list > paquets_data->temp_pointer_1->nb_task_in_sub_list)
			{
				min_nb_task_in_sub_list = paquets_data->temp_pointer_1->nb_task_in_sub_list;
			}
		}
		for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		{
			if (min_nb_task_in_sub_list == paquets_data->temp_pointer_1->nb_task_in_sub_list)
			{
				nb_min_task_packages++;
			}
		}
		//~ if (_print_in_terminal == 1) {  printf("Il y a %d paquets de taille minimale %d tâches\n",nb_min_task_packages,min_nb_task_in_sub_list); }
		/* Then we create the common data matrix */
		for (paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_1 != NULL; paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next)
		{
			for (paquets_data->temp_pointer_2 = paquets_data->temp_pointer_1->next; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
			{
				for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++)
				{
					int j;
					for (j = 0; j < paquets_data->temp_pointer_2->package_nb_data; j++)
					{
						if ((paquets_data->temp_pointer_1->package_data[i] == paquets_data->temp_pointer_2->package_data[j]))
						{
							matrice_donnees_commune[index_head_1][index_head_2] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]) + starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
							matrice_donnees_commune[index_head_2][index_head_1] += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[j]) + starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
						}
					}
				}
				index_head_2++;
			}
			index_head_1++;
			index_head_2 = index_head_1 + 1;
		}
		/* Code to print the common data matrix */
		//~ if (_print_in_terminal == 1) { printf("Common data matrix : \n"); for (i = 0; i < number_task; i++) { for (j = 0; j < number_task; j++) { print_in_terminal (" %3li ",matrice_donnees_commune[i][j]); } printf("\n"); printf("---------\n"); }}

		/* Getting back to the beginning of the linked list */
		paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;

		i_bis = 0; j_bis = 0;
		temp_nb_min_task_packages = nb_min_task_packages;
	debut_while:
		paquets_data->temp_pointer_1 = paquets_data->first_link;
		paquets_data->temp_pointer_2 = paquets_data->first_link;
		max_value_common_data_matrix = 0;
		if (GPU_limit_switch == 1)
		{
			for (i_bis = 0; i_bis < number_task; i_bis++)
			{
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
				{
					//Si on est sur un paquet de taille minimale
					//~ printf("Sur le paquet minimal %d de %d data\n", i_bis, paquets_data->temp_pointer_1->package_nb_data);
					for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
					{
						//~ if (i_bis != j_bis && matrice_donnees_commune[i_bis][j_bis] != 0) {
						if (i_bis != j_bis)
						{
							//~ printf("Sur le paquet %d de %d data\n", j_bis, paquets_data->temp_pointer_2->package_nb_data);
							weight_two_packages = 0;
							for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++)
							{
								weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
							}
							for (i = 0; i < paquets_data->temp_pointer_2->package_nb_data; i++)
							{
								bool_data_common = 0;
								int j;
								for (j = 0; j < paquets_data->temp_pointer_1->package_nb_data; j++)
								{
									if (paquets_data->temp_pointer_2->package_data[i] == paquets_data->temp_pointer_1->package_data[j])
									{
										bool_data_common = 1;
									}
								}
								if (bool_data_common != 1)
								{
									weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i]);
								}
							}
							if((max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis]) && (weight_two_packages <= _starpu_HFP_GPU_RAM_M))
							{
								max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
							}
						}
						j_bis++;
					}
					tab_runner++;
				}
				paquets_data->temp_pointer_1=paquets_data->temp_pointer_1->next;
				j_bis = 0;
			}
			paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;
		}
		/* Else, we are using algo 5, so we don't check the max weight */
		else
		{
			for (i_bis = 0; i_bis < number_task; i_bis++)
			{
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
				{ //Si on est sur un paquet de taille minimale
					for (paquets_data->temp_pointer_2 = paquets_data->first_link; paquets_data->temp_pointer_2 != NULL; paquets_data->temp_pointer_2 = paquets_data->temp_pointer_2->next)
					{
						if (i_bis != j_bis)
						{
							weight_two_packages = 0;
							for (i = 0; i < paquets_data->temp_pointer_1->package_nb_data; i++)
							{
								weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i]);
							}
							for (i = 0; i < paquets_data->temp_pointer_2->package_nb_data; i++)
							{
								bool_data_common = 0;
								int j;
								for (j = 0; j < paquets_data->temp_pointer_1->package_nb_data; j++)
								{
									if (paquets_data->temp_pointer_2->package_data[i] == paquets_data->temp_pointer_1->package_data[j])
									{
										bool_data_common = 1;
									}
								}
								if (bool_data_common != 1)
								{
									weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i]);
								}
							}
							if(max_value_common_data_matrix < matrice_donnees_commune[i_bis][j_bis])
							{
								max_value_common_data_matrix = matrice_donnees_commune[i_bis][j_bis];
							}
						}
						j_bis++;
					}
					tab_runner++;
				}
				paquets_data->temp_pointer_1=paquets_data->temp_pointer_1->next;
				j_bis = 0;
			}
			paquets_data->temp_pointer_1 = paquets_data->first_link; paquets_data->temp_pointer_2 = paquets_data->first_link;
		}
		//~ printf("la, max value = %ld, limit switch = %d\n", max_value_common_data_matrix, GPU_limit_switch);
		if (max_value_common_data_matrix == 0 && GPU_limit_switch == 0)
		{
			/* It means that P_i share no data with others, so we put it in the end of the list
			 * For this we use a separate list that we merge at the end
			 * We will put this list at the end of the rest of the packages */
			//~ if (_print_in_terminal == 1) { printf("graphe non connexe\n"); }
			while (paquets_data->temp_pointer_1->nb_task_in_sub_list != min_nb_task_in_sub_list)
			{
				paquets_data->temp_pointer_1 = paquets_data->temp_pointer_1->next;
			}
			while (!starpu_task_list_empty(&paquets_data->temp_pointer_1->sub_list))
			{
				starpu_task_list_push_back(&non_connexe,starpu_task_list_pop_front(&paquets_data->temp_pointer_1->sub_list));
			}
			paquets_data->temp_pointer_1->package_nb_data = 0;
			paquets_data->NP--;
		}

		//~ if (max_value_common_data_matrix == 0 && GPU_limit_switch == 0) {
		//~ /* It means that P_i share no data with others, so we put it in the end of the list
		//~ * For this we use a separate list that we merge at the end
		//~ * We will put this list at the end of the rest of the packages */
		//~ if (_print_in_terminal == 1) { printf("Graphe non connexe\n"); }
		//~ while (data->p->temp_pointer_1->nb_task_in_sub_list != min_nb_task_in_sub_list)
		//~ {
		//~ data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
		//~ }
		//~ while (!starpu_task_list_empty(&data->p->temp_pointer_1->sub_list)) {
		//~ starpu_task_list_push_back(&non_connexe, starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list));
		//~ }
		//~ data->p->temp_pointer_1->package_nb_data = 0;
		//~ data->p->NP--;
		//~ }
		else
		{
			i_bis = 0; j_bis = 0;
			for (i = 0; i < number_task; i++)
			{
				if (paquets_data->temp_pointer_1->nb_task_in_sub_list == min_nb_task_in_sub_list)
				{
					int j;
					for (j = 0; j < number_task; j++)
					{
						weight_two_packages = 0;  weight_package_i = 0;  weight_package_j = 0;
						for (i_bis = 0; i_bis < paquets_data->temp_pointer_1->package_nb_data; i_bis++)
						{
							weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_1->package_data[i_bis]);
						}
						weight_package_i = weight_two_packages;
						for (i_bis = 0; i_bis < paquets_data->temp_pointer_2->package_nb_data; i_bis++)
						{
							bool_data_common = 0;
							for (j_bis = 0; j_bis < paquets_data->temp_pointer_1->package_nb_data; j_bis++)
							{
								if (paquets_data->temp_pointer_2->package_data[i_bis] == paquets_data->temp_pointer_1->package_data[j_bis])
								{
									bool_data_common = 1;
								}
							}
							if (bool_data_common != 1)
							{
								weight_two_packages += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i_bis]);
							}
							weight_package_j += starpu_data_get_size(paquets_data->temp_pointer_2->package_data[i_bis]);
						}
						if (matrice_donnees_commune[i][j] == max_value_common_data_matrix && i != j && max_value_common_data_matrix != 0)
						{
							if ((weight_two_packages <= _starpu_HFP_GPU_RAM_M) || (GPU_limit_switch == 0))
							{
								/* Merge */
								packaging_impossible = 0;
								//~ if (_print_in_terminal == 1) { printf("On va merge le paquet %d et le paquet %d. Ils ont %ld en commun. Ils ont %d et %d tâches.\n", i, j, max_value_common_data_matrix, paquets_data->temp_pointer_1->nb_task_in_sub_list, paquets_data->temp_pointer_2->nb_task_in_sub_list); }

								paquets_data->NP--;

								if (paquets_data->temp_pointer_2->nb_task_in_sub_list == min_nb_task_in_sub_list)
								{
									temp_nb_min_task_packages--;
								}

								for (j_bis = 0; j_bis < number_task; j_bis++)
								{
									matrice_donnees_commune[i][j_bis] = 0; matrice_donnees_commune[j_bis][i] = 0;
								}
								for (j_bis = 0; j_bis < number_task; j_bis++)
								{
									matrice_donnees_commune[j][j_bis] = 0;
									matrice_donnees_commune[j_bis][j] = 0;
								}

								if (order_u == 1)
								{
									if (paquets_data->temp_pointer_1->nb_task_in_sub_list == 1 && paquets_data->temp_pointer_2->nb_task_in_sub_list == 1)
									{
										//~ if (_print_in_terminal == 1) { printf("I = 1 et J = 1\n"); }
									}
									else if (weight_package_i > _starpu_HFP_GPU_RAM_M && weight_package_j <= _starpu_HFP_GPU_RAM_M)
									{
										common_data_last_package_i1_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 0, false,_starpu_HFP_GPU_RAM_M);
										common_data_last_package_i2_j = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 0, false,_starpu_HFP_GPU_RAM_M);
										if (common_data_last_package_i1_j > common_data_last_package_i2_j)
										{
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
										}
									}
									else if (weight_package_i <= _starpu_HFP_GPU_RAM_M && weight_package_j > _starpu_HFP_GPU_RAM_M)
									{
										common_data_last_package_i_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 1, false,_starpu_HFP_GPU_RAM_M);
										common_data_last_package_i_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 0, 2, false,_starpu_HFP_GPU_RAM_M);
										if (common_data_last_package_i_j2 > common_data_last_package_i_j1)
										{
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
									}
									else
									{
										if (weight_package_i > _starpu_HFP_GPU_RAM_M && weight_package_j > _starpu_HFP_GPU_RAM_M)
										{
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, false,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, false,_starpu_HFP_GPU_RAM_M);
										}
										else if (weight_package_i <= _starpu_HFP_GPU_RAM_M && weight_package_j <= _starpu_HFP_GPU_RAM_M)
										{
											common_data_last_package_i1_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 1, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i1_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 1, 2, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j1 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 1, true,_starpu_HFP_GPU_RAM_M);
											common_data_last_package_i2_j2 = get_common_data_last_package(paquets_data->temp_pointer_1, paquets_data->temp_pointer_2, 2, 2, true,_starpu_HFP_GPU_RAM_M);
										}
										else
										{
											printf("Erreur dans ordre U, aucun cas choisi\n");
											exit(0);
										}
										max_common_data_last_package = common_data_last_package_i2_j1;
										if (max_common_data_last_package < common_data_last_package_i1_j1)
										{
											max_common_data_last_package = common_data_last_package_i1_j1;
										}
										if (max_common_data_last_package < common_data_last_package_i1_j2)
										{
											max_common_data_last_package = common_data_last_package_i1_j2;
										}
										if (max_common_data_last_package < common_data_last_package_i2_j2)
										{
											max_common_data_last_package = common_data_last_package_i2_j2;
										}
										if (max_common_data_last_package == common_data_last_package_i2_j1)
										{
											//~ if (_print_in_terminal == 1) { printf("Pas de switch\n"); }
										}
										else if (max_common_data_last_package == common_data_last_package_i1_j2)
										{
											//~ if (_print_in_terminal == 1) { printf("SWITCH PAQUET I ET J\n");	}
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
										else if (max_common_data_last_package == common_data_last_package_i2_j2)
										{
											//~ if (_print_in_terminal == 1) { printf("SWITCH PAQUET J\n"); }
											paquets_data->temp_pointer_2 = HFP_reverse_sub_list(paquets_data->temp_pointer_2);
										}
										else
										{ /* max_common_data_last_package == common_data_last_package_i1_j1 */
											//~ if (_print_in_terminal == 1) { printf("SWITCH PAQUET I\n"); }
											paquets_data->temp_pointer_1 = HFP_reverse_sub_list(paquets_data->temp_pointer_1);
										}
									}
									//~ if (_print_in_terminal == 1) { printf("Fin de l'ordre U sans doublons\n"); }
								}

								paquets_data->temp_pointer_1->split_last_ij = paquets_data->temp_pointer_1->nb_task_in_sub_list;
								while (!starpu_task_list_empty(&paquets_data->temp_pointer_2->sub_list))
								{
									starpu_task_list_push_back(&paquets_data->temp_pointer_1->sub_list,starpu_task_list_pop_front(&paquets_data->temp_pointer_2->sub_list));
									paquets_data->temp_pointer_1->nb_task_in_sub_list ++;
								}
								i_bis = 0; j_bis = 0; tab_runner = 0;
								starpu_data_handle_t *temp_data_tab = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data) * sizeof(paquets_data->temp_pointer_1->package_data[0]));
								while (i_bis < paquets_data->temp_pointer_1->package_nb_data && j_bis < paquets_data->temp_pointer_2->package_nb_data)
								{
									if (paquets_data->temp_pointer_1->package_data[i_bis] <= paquets_data->temp_pointer_2->package_data[j_bis])
									{
										temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis];
										i_bis++;
									}
									else
									{
										temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis];
										j_bis++;
									}
									tab_runner++;
								}
								while (i_bis < paquets_data->temp_pointer_1->package_nb_data)
								{
									temp_data_tab[tab_runner] = paquets_data->temp_pointer_1->package_data[i_bis]; i_bis++; tab_runner++;
								}
								while (j_bis < paquets_data->temp_pointer_2->package_nb_data)
								{
									temp_data_tab[tab_runner] = paquets_data->temp_pointer_2->package_data[j_bis]; j_bis++; tab_runner++;
								}
								for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++)
								{
									if (temp_data_tab[i_bis] == temp_data_tab[i_bis + 1])
									{
										temp_data_tab[i_bis] = 0;
										nb_duplicate_data++;
									}
								}
								paquets_data->temp_pointer_1->package_data = malloc((paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data - nb_duplicate_data) * sizeof(starpu_data_handle_t));
								j_bis = 0;
								for (i_bis = 0; i_bis < (paquets_data->temp_pointer_1->package_nb_data + paquets_data->temp_pointer_2->package_nb_data); i_bis++)
								{
									if (temp_data_tab[i_bis] != 0)
									{
										paquets_data->temp_pointer_1->package_data[j_bis] = temp_data_tab[i_bis]; j_bis++;
									}
								}
								paquets_data->temp_pointer_1->package_nb_data = paquets_data->temp_pointer_2->package_nb_data + paquets_data->temp_pointer_1->package_nb_data - nb_duplicate_data;

								paquets_data->temp_pointer_1->expected_time += paquets_data->temp_pointer_2->expected_time;

								paquets_data->temp_pointer_2->package_nb_data = 0;
								nb_duplicate_data = 0;
								paquets_data->temp_pointer_2->nb_task_in_sub_list = 0;
								temp_nb_min_task_packages--;
								if(paquets_data->NP == number_of_package_to_build)
								{
									goto break_merging_1;
								}
								if (temp_nb_min_task_packages > 1)
								{
									goto debut_while;
								}
								else
								{
									j = number_task; i = number_task;
								}
							}
						}
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

		/* Code to get the coordinates of each data in the order in which tasks get out of pull_task */
		while (paquets_data->temp_pointer_1 != NULL)
		{
			/* if ((strcmp(_starpu_HFP_appli,"starpu_sgemm_gemm") == 0) && (_print_in_terminal == 1)) {
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
		/* if (_print_in_terminal == 1) { visualisation_tache_matrice_format_tex(coordinate_visualization_matrix,coordinate_order_visualization_matrix,nb_of_loop,link_index); } */

		/* Checking if we have the right number of packages. if MULTIGPU is equal to 0 we want only one package. if it is equal to 1 we want |GPU| packages */
		if (link_index == number_of_package_to_build)
		{
			goto end_while_packaging_impossible;
		}

		for (i = 0; i < number_task; i++)
		{
			int j;
			for (j = 0; j < number_task; j++)
			{
				matrice_donnees_commune[i][j] = 0;
			}
		}
		/* Reset number_task for the matrix initialisation */
		number_task = link_index;
		/* If we have only one package we don't have to do more packages */
		if (number_task == 1)
		{
			packaging_impossible = 1;
		}
	} /* End of while (packaging_impossible == 0) { */
	/* We are in algorithm 3, we remove the size limit of a package */
	GPU_limit_switch = 0; goto beginning_while_packaging_impossible;

 end_while_packaging_impossible:

	/* Add tasks or packages that were not connexe */
	while(!starpu_task_list_empty(&non_connexe))
	{
		starpu_task_list_push_back(&paquets_data->first_link->sub_list, starpu_task_list_pop_front(&non_connexe));
		paquets_data->first_link->nb_task_in_sub_list++;
	}

	return paquets_data->first_link->sub_list;
}

/* Check if our struct is empty */
bool _starpu_HFP_is_empty(struct _starpu_HFP_my_list* a)
{
	if (a == NULL)
	{
		return true;
	}
	if (!starpu_task_list_empty(&a->sub_list))
	{
		return false;
	}
	while (a->next != NULL)
	{
		a = a->next;
		if (!starpu_task_list_empty(&a->sub_list))
		{
			return false;
		}
	}
	return true;
}

/* TODO : a supprimer une fois les mesures du temps terminées */
//~ struct timeval time_start_loadbalanceexpectedtime;
//~ struct timeval time_end_loadbalanceexpectedtime;
//~ long long time_total_loadbalanceexpectedtime = 0;

/* Equilibrates package in order to have packages with the exact same expected task time
 * Called in HFP_pull_task once all packages are done
 * Used for MULTIGPU == 4
 */
static void load_balance_expected_time(struct _starpu_HFP_paquets *a, int number_gpu)
{
	//~ gettimeofday(&time_start_loadbalanceexpectedtime, NULL);

	struct starpu_task *task;
	double ite = 0; int i = 0; int index = 0;
	int package_with_min_expected_time, package_with_max_expected_time;
	double min_expected_time, max_expected_time, expected_time_to_steal = 0;
	bool load_balance_needed = true;
	//~ if (_print_in_terminal == 1) { printf("A package should have %f time +/- 5 percent\n", EXPECTED_TIME/number_gpu); }
	/* Selecting the smallest and biggest package */
	while (load_balance_needed == true)
	{
		a->temp_pointer_1 = a->first_link;
		min_expected_time = a->temp_pointer_1->expected_time;
		max_expected_time = a->temp_pointer_1->expected_time;
		package_with_min_expected_time = 0;
		package_with_max_expected_time = 0;
		i = 0;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		while (a->temp_pointer_1 != NULL)
		{
			i++;
			if (min_expected_time > a->temp_pointer_1->expected_time)
			{
				min_expected_time = a->temp_pointer_1->expected_time;
				package_with_min_expected_time = i;
			}
			if (max_expected_time < a->temp_pointer_1->expected_time)
			{
				max_expected_time = a->temp_pointer_1->expected_time;
				package_with_max_expected_time = i;
			}
			a->temp_pointer_1 = a->temp_pointer_1->next;
		}
		//~ if (_print_in_terminal == 1) { printf("min et max : %f et %f\n",min_expected_time, max_expected_time); }
		//~ exit(0);
		/* Stealing as much task from the last tasks of the biggest packages */
		//~ if (package_with_min_expected_time == package_with_max_expected_time || min_expected_time >=  max_expected_time - ((5*max_expected_time)/100)) {
		if (package_with_min_expected_time == package_with_max_expected_time)
		{
			//~ if (_print_in_terminal == 1) { printf("All packages have the same expected time\n"); }
			load_balance_needed = false;
		}
		else
		{
			/* Getting on the right packages */
			a->temp_pointer_1 = a->first_link;
			for (i = 0; i < package_with_min_expected_time; i++)
			{
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_2 = a->first_link; index = 0;
			for (i = 0; i < package_with_max_expected_time; i++)
			{
				a->temp_pointer_2 = a->temp_pointer_2->next;
				index++;
			}
			if (a->temp_pointer_2->expected_time - ((EXPECTED_TIME/number_gpu) - a->temp_pointer_1->expected_time) >= EXPECTED_TIME/number_gpu)
			{
				//~ printf("if\n");
				expected_time_to_steal = (EXPECTED_TIME/number_gpu) - a->temp_pointer_1->expected_time;
			}
			else
			{
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
			ite = 0;

			//Pour visu python. Pas implémenté dans load_balance et load_balance_expected_package_time
			FILE *f = fopen("Output_maxime/Data_stolen_load_balance.txt", "a");

			while (ite < expected_time_to_steal)
			{
				task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
				ite += starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
				a->temp_pointer_2->expected_time = a->temp_pointer_2->expected_time - starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);

				//Pour visu python
				if (_print_in_terminal == 1)
				{
					int temp_tab_coordinates[2];
					if (_print3d != 0)
					{
						starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, temp_tab_coordinates);
						fprintf(f, "%d	%d", temp_tab_coordinates[0], temp_tab_coordinates[1]);
						starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, temp_tab_coordinates);
						fprintf(f, "	%d	%d\n", temp_tab_coordinates[0], index);
					}
					else
					{
						starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, temp_tab_coordinates);
						fprintf(f, "%d	%d	%d\n", temp_tab_coordinates[0], temp_tab_coordinates[1], index);
					}
				}

				/* Merging */
				merge_task_and_package(a->temp_pointer_1, task);
				a->temp_pointer_2->nb_task_in_sub_list--;
			}

			fclose(f);
		}
	}

	//~ gettimeofday(&time_end_loadbalanceexpectedtime, NULL);
	//~ time_total_loadbalanceexpectedtime += (time_end_loadbalanceexpectedtime.tv_sec - time_start_loadbalanceexpectedtime.tv_sec)*1000000LL + time_end_loadbalanceexpectedtime.tv_usec - time_start_loadbalanceexpectedtime.tv_usec;
}

/* Equilibrates package in order to have packages with the exact same number of tasks +/-1 task
 * Called in HFP_pull_task once all packages are done
 */
static void load_balance(struct _starpu_HFP_paquets *a, int number_gpu)
{
	int min_number_task_in_package, package_with_min_number_task, i, max_number_task_in_package, package_with_max_number_task, number_task_to_steal = 0;
	bool load_balance_needed = true;
	struct starpu_task *task = NULL;
	//~ if (_print_in_terminal == 1){ printf("A package should have %d or %d tasks\n", NT/number_gpu, NT/number_gpu+1); }
	/* Selecting the smallest and biggest package */
	while (load_balance_needed == true)
	{
		a->temp_pointer_1 = a->first_link;
		min_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
		max_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
		package_with_min_number_task = 0;
		package_with_max_number_task = 0;
		i = 0;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		while (a->temp_pointer_1 != NULL)
		{
			i++;
			if (min_number_task_in_package > a->temp_pointer_1->nb_task_in_sub_list)
			{
				min_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
				package_with_min_number_task = i;
			}
			if (max_number_task_in_package < a->temp_pointer_1->nb_task_in_sub_list)
			{
				max_number_task_in_package = a->temp_pointer_1->nb_task_in_sub_list;
				package_with_max_number_task = i;
			}
			a->temp_pointer_1 = a->temp_pointer_1->next;
		}
		/* Stealing as much task from the last tasks of the biggest packages */
		if (package_with_min_number_task == package_with_max_number_task || min_number_task_in_package ==  max_number_task_in_package-1)
		{
			load_balance_needed = false;
		}
		else
		{
			a->temp_pointer_1 = a->first_link;
			for (i = 0; i < package_with_min_number_task; i++)
			{
				a->temp_pointer_1 = a->temp_pointer_1->next;
			}
			a->temp_pointer_2 = a->first_link;
			for (i = 0; i < package_with_max_number_task; i++)
			{
				a->temp_pointer_2 = a->temp_pointer_2->next;
			}
			if ((_starpu_HFP_NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list == 0)
			{
				number_task_to_steal = 1;
			}
			else if (a->temp_pointer_2->nb_task_in_sub_list - ((_starpu_HFP_NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list) >= _starpu_HFP_NT/number_gpu)
			{
				number_task_to_steal = (_starpu_HFP_NT/number_gpu) - a->temp_pointer_1->nb_task_in_sub_list;
			}
			else
			{
				number_task_to_steal = a->temp_pointer_2->nb_task_in_sub_list - _starpu_HFP_NT/number_gpu;
			}
			for (i = 0; i < number_task_to_steal; i++)
			{
				task = starpu_task_list_pop_back(&a->temp_pointer_2->sub_list);
				merge_task_and_package(a->temp_pointer_1, task);
				a->temp_pointer_2->expected_time -= starpu_task_expected_length(task, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
				a->temp_pointer_2->nb_task_in_sub_list--;
			}
		}
	}
}

/* Printing in a .tex file the number of GPU that a data is used in.
 * With red = 255 if it's on GPU 1, blue if it's on GPU 2 and green on GPU 3.
 * Thus it only work for 3 GPUs.
 * Also print the number of use in each GPU.
 * TODO : Faire marcher cette fonction avec n GPUs
 */
//static void visualisation_data_gpu_in_file_hfp_format_tex(struct _starpu_HFP_paquets *p)
//{
//	struct starpu_task *task;
//	int i = 0;
//	int j = 0;
//	int k = 0;
//	int red, green, blue;
//	int temp_tab_coordinates[2];
//	FILE *f = fopen("Output_maxime/Data_in_gpu_HFP.tex", "w");
//	fprintf(f, "\\documentclass{article}\\usepackage{diagbox}\\usepackage{color}\\usepackage{fullpage}\\usepackage{colortbl}\\usepackage{caption}\\usepackage{subcaption}\\usepackage{float}\\usepackage{graphics}\\begin{document}\n\n\n\\begin{figure}[H]\n");
//	int data_use_in_gpus[_starpu_HFP_N*2][_nb_gpus + 1];
//	for (j = 0; j < 2; j++)
//	{
//		printf("premier for\n");
//		for (i = 0; i < _starpu_HFP_N*2; i++)
//		{
//			for (k = 0; k < _nb_gpus + 1; k++)
//			{
//				data_use_in_gpus[i][k] = 0;
//			}
//		}
//		fprintf(f, "\\begin{subfigure}{.5\\textwidth}\\centering\\begin{tabular}{|");
//		for (i = 0; i < _starpu_HFP_N; i++)
//		{
//			fprintf(f,"c|");
//		}
//		fprintf(f,"c|}\\hline\\diagbox{GPUs}{Data}&");
//		p->temp_pointer_1 = p->first_link;
//		i = 0;
//		while (p->temp_pointer_1 != NULL)
//		{
//			for (task = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task != starpu_task_list_end(&p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
//			{
//				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task,2),2,temp_tab_coordinates);
//				data_use_in_gpus[temp_tab_coordinates[j]][i]++;
//			}
//			p->temp_pointer_1 = p->temp_pointer_1->next;
//			i++;
//		}
//		for (i = 0; i < _starpu_HFP_N - 1; i++)
//		{
//			red = 0;
//			green = 0;
//			blue = 0;
//			for (k = 0; k < _nb_gpus; k++)
//			{
//				if (data_use_in_gpus[i][k] != 0)
//				{
//					if (k%3 == 0)
//					{
//						red = 1;
//					}
//					if (k%3 == 1)
//					{
//						green = 1;
//					}
//					if (k%3 == 2)
//					{
//						blue = 1;
//					}
//				}
//			}
//			fprintf(f,"\\cellcolor[RGB]{%d,%d,%d}%d&", red*255, green*255, blue*255, i);
//		}
//		red = 0;
//		green = 0;
//		blue = 0;
//		for (k = 0; k < _nb_gpus; k++)
//		{
//			if (data_use_in_gpus[_starpu_HFP_N - 1][k] != 0)
//			{
//				if (k%3 == 0)
//				{
//					red = 1;
//				}
//				if (k%3 == 1)
//				{
//					green = 1;
//				}
//				if (k%3 == 2)
//				{
//					blue = 1;
//				}
//			}
//		}
//		fprintf(f,"\\cellcolor[RGB]{%d,%d,%d}%d\\\\\\hline", red*255, green*255, blue*255, _starpu_HFP_N - 1);
//		for (i = 0; i < _nb_gpus; i++)
//		{
//			red = 0;
//			green = 0;
//			blue = 0;
//			if (i%3 == 0)
//			{
//				red = 1;
//			}
//			if (i%3 == 1)
//			{
//				green = 1;
//			}
//			if (i%3 == 2)
//			{
//				blue = 1;
//			}
//			fprintf(f, " \\cellcolor[RGB]{%d,%d,%d}GPU %d&", red*255, green*255, blue*255, i);
//			for (k = 0; k < _starpu_HFP_N - 1; k++)
//			{
//				fprintf(f, "%d&", data_use_in_gpus[k][i]);
//			}
//			fprintf(f, "%d\\\\\\hline", data_use_in_gpus[_starpu_HFP_N - 1][i]);
//		}
//		fprintf(f, "\\end{tabular}\\caption{Data from matrix ");
//		if (j == 0)
//		{
//			fprintf(f, "A");
//		}
//		else
//		{
//			fprintf(f, "B");
//		}
//		fprintf(f, "}\\end{subfigure}\n\n");
//	}
//	fprintf(f, "\\caption{Number of use of a data in each GPU}\\end{figure}\n\n\n\\end{document}");
//	fclose(f);
//}

/* Print the order in one file for each GPU and also print in a tex file the coordinate for 2D matrix, before the ready, so it's only planned */
static void print_order_in_file_hfp(struct _starpu_HFP_paquets *p)
{
	char str[2];
	unsigned i = 0;
	int size = 0;
	char *path = NULL;

	p->temp_pointer_1 = p->first_link;
	struct starpu_task *task;
	while (p->temp_pointer_1 != NULL)
	{
		sprintf(str, "%d", i);
		size = strlen("Output_maxime/Task_order_HFP_") + strlen(str);
		path = malloc(sizeof(char)*size);
		strcpy(path, "Output_maxime/Task_order_HFP_");
		strcat(path, str);
		FILE *f = fopen(path, "w");
		for (task = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task != starpu_task_list_end(&p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
		{
			fprintf(f, "%p\n",task);
		}
		p->temp_pointer_1 = p->temp_pointer_1->next;
		i++;
		fclose(f);
	}
	if (_print_in_terminal == 1 && (strcmp(_starpu_HFP_appli,"starpu_sgemm_gemm") == 0))
	{
		i = 0;
		p->temp_pointer_1 = p->first_link;
		FILE *f = fopen("Output_maxime/Data_coordinates_order_last_HFP.txt", "w");
		int temp_tab_coordinates[2];
		while (p->temp_pointer_1 != NULL)
		{
			for (task = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task != starpu_task_list_end(&p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
			{
				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task,2),2,temp_tab_coordinates);
				fprintf(f, "%d	%d	%d\n", temp_tab_coordinates[0], temp_tab_coordinates[1], i);
			}
			p->temp_pointer_1 = p->temp_pointer_1->next;
			i++;
		}
		fclose(f);
		//visualisation_tache_matrice_format_tex("HFP");
	}
}

/* Attention, la dedans je vide la liste l. Et donc si tu lui donne sched_list et que
 * derrière t'essaye de la lire comme je fesais dans MST, et bah ca va crasher.
 * Aussi si tu lance hMETIS dans un do_schedule, attention de bien mettre do_schedule_done à true
 * et de sortir de la fonction avec un return;.
 */
void _starpu_hmetis_scheduling(struct _starpu_HFP_paquets *p, struct starpu_task_list *l, int nb_gpu)
{
	FILE *f1 = fopen("Output_maxime/temp_input_hMETIS.txt", "w+");
	_starpu_HFP_NT = 0;
	struct starpu_task *task_1;
	int NT = 0;
	bool first_write_on_line = true, already_counted = false;
	int index_task_1 = 1; int index_task_2 = 0; int number_hyperedge = 0;

	for (task_1 = starpu_task_list_begin(l); task_1 != starpu_task_list_end(l); task_1 = starpu_task_list_next(task_1))
	{
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task_1); i++)
		{
			struct starpu_task *task_3;
			task_3 = starpu_task_list_begin(l);
			already_counted = false;
			int k;
			for (k = 1; k < index_task_1; k++)
			{
				unsigned m;
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
				struct starpu_task *task_2;
				for (task_2 = starpu_task_list_next(task_1); task_2 != starpu_task_list_end(l); task_2 = starpu_task_list_next(task_2))
				{
					unsigned j;
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task_2); j++)
					{
						if (STARPU_TASK_GET_HANDLE(task_1, i) == STARPU_TASK_GET_HANDLE(task_2, j))
						{
							if (first_write_on_line == true)
							{
								first_write_on_line = false;
								fprintf(f1, "%d %d", index_task_1, index_task_2);
								number_hyperedge++;
							}
							else
							{
								fprintf(f1, " %d", index_task_2);
							}
						}
					}
					index_task_2++;
				}
				if (first_write_on_line == false)
				{
					fprintf(f1, "\n");
				}
			}
		}
		index_task_1++;
		NT++;
	}
	_starpu_HFP_N = sqrt(NT);
	if(_print3d == 1)
	{
		_starpu_HFP_N = _starpu_HFP_N/2; /* So i can print just like a 2D matrix */
	}
	/* Printing expected time of each task */
	for (task_1 = starpu_task_list_begin(l); task_1 != starpu_task_list_end(l); task_1 = starpu_task_list_next(task_1))
	{
		fprintf(f1, "%f\n", starpu_task_expected_length(task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0));
	}
	/* Printing information for hMETIS on the first line */
	FILE *f_3 = fopen("Output_maxime/input_hMETIS.txt", "w+");
	fprintf(f_3, "%d %d 10\n", number_hyperedge, NT); /* Number of hyperedges, number of task, 10 for weighted vertices but non weighted */
	char ch;
	rewind(f1);
	while ((ch = fgetc(f1)) != EOF)
		fputc(ch, f_3);
	fclose(f1);
	fclose(f_3);
	//TODO : remplacer le 3 par nb_gpu ici
	//TODO tester différents paramètres de hmetis et donc modifier ici
	FILE *f2 = fopen("Output_maxime/hMETIS_parameters.txt", "r");

	//~ Nparts : nombre de paquets.
	//~ UBfactor : 1 - 49, a tester. Déséquilibre autorisé.
	//~ Nruns : 1 - inf, a tester. 1 par défaut. Plus on test plus ce sera bon mais ce sera plus long.
	//~ CType : 1 - 5, a tester. 1 par défaut.
	//~ RType :  1 - 3, a tester. 1 par défaut.
	//~ Vcycle : 1. Sélectionne la meilleure des Nruns.
	//~ Reconst : 0 - 1, a tester. 0 par défaut. Normalement ca ne devrait rien changer car ca joue juste sur le fait de reconstruire les hyperedges ou non.
	//~ dbglvl : 0. Sert à montrer des infos de debug; Si besoin mettre (1, 2 ou 4).

	int size = strlen("../these_gonthier_maxime/hMETIS/hmetis-1.5-linux/hmetis Output_maxime/input_hMETIS.txt_");
	char buffer[100];
	while (fscanf(f2, "%s", buffer) == 1)
	{
		size += sizeof(buffer);
	}
	rewind(f2);
	char *system_call = (char *)malloc(size);
	strcpy(system_call, "../these_gonthier_maxime/hMETIS/hmetis-1.5-linux/hmetis Output_maxime/input_hMETIS.txt");
	while (fscanf(f2, "%s", buffer)== 1)
	{
		strcat(system_call, " ");
		strcat(system_call, buffer);
	}
	fclose(f2);

	int cr = system(system_call);
	if (cr != 0)
	{
		printf("Error when calling system(../these_gonthier_maxime/hMETIS/hmetis-1.5-linux/hmetis\n");
		exit(0);
	}
	starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
	int i;
	for (i = 1; i < nb_gpu; i++)
	{
		_starpu_HFP_insertion(p);
		starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
	}
	p->first_link = p->temp_pointer_1;
	char str[2];
	sprintf(str, "%d", nb_gpu);
	size = strlen("Output_maxime/input_hMETIS.txt.part.") + strlen(str);
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
		p->temp_pointer_1 = p->first_link;
		int j;
		for (j = 0; j < number; j++)
		{
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
		task_1 = starpu_task_list_pop_front(l);
		p->temp_pointer_1->expected_time += starpu_task_expected_length(task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
		starpu_task_list_push_back(&p->temp_pointer_1->sub_list, task_1);
		p->temp_pointer_1->nb_task_in_sub_list++;
	}
	fclose(f_2);

	/* Apply HFP on each package if we have the right option */
	if (_starpu_HFP_hmetis == 2)
	{
		if (_print_in_terminal == 1)
		{
			i = 0;
			p->temp_pointer_1 = p->first_link;
			FILE *f = fopen("Output_maxime/Data_coordinates_order_last_hMETIS.txt", "w");
			int temp_tab_coordinates[2];
			while (p->temp_pointer_1 != NULL)
			{
				for (task_1 = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task_1 != starpu_task_list_end(&p->temp_pointer_1->sub_list); task_1 = starpu_task_list_next(task_1))
				{
					starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task_1,2),2,temp_tab_coordinates);
					fprintf(f, "%d	%d	%d\n", temp_tab_coordinates[0], temp_tab_coordinates[1], i);
				}
				p->temp_pointer_1 = p->temp_pointer_1->next;
				i++;
			}
			fclose(f);
			//visualisation_tache_matrice_format_tex("hMETIS"); /* So I can get the matrix visualisation before tempering it with HFP */
		}
		p->temp_pointer_1 = p->first_link;
		for (i = 0; i < nb_gpu; i++)
		{
			p->temp_pointer_1->sub_list = hierarchical_fair_packing_one_task_list(p->temp_pointer_1->sub_list, p->temp_pointer_1->nb_task_in_sub_list);
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
	}
}

/*
 * This function is used when a partition has already be generated by hMETIS, and we now want to apply the execution.
 */
static void hmetis_input_already_generated(struct _starpu_HFP_paquets *p, struct starpu_task_list *l, int nb_gpu)
{
	int random_task_order = starpu_get_env_number_default("RANDOM_TASK_ORDER", 0);
	sparse_matrix = starpu_get_env_number_default("SPARSE_MATRIX", 0);

	_starpu_HFP_NT = starpu_task_list_size(l);
	int i = 0; struct starpu_task *task_1;
	int j = 0;

	_starpu_HFP_N = hmetis_n;
	if (_starpu_HFP_N == 0)
	{
		printf("Error, please precis env var HMETIS_N.\n");
		exit(0);
	}
	int size =  0;
	starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
	for (i = 1; i < nb_gpu; i++)
	{
		_starpu_HFP_insertion(p);
		starpu_task_list_init(&p->temp_pointer_1->refused_fifo_list);
	}
	p->first_link = p->temp_pointer_1;

	char str[2];
	char Nchar[4];
	sprintf(str, "%d", nb_gpu);
	sprintf(Nchar, "%d", _starpu_HFP_N);
	if (random_task_order == 1)
	{
		size = strlen("Output_maxime/Data/input_hMETIS/") + strlen(str) + strlen("GPU_Random_task_order/input_hMETIS_N") + strlen(Nchar) + strlen(".txt");
	}
	else if (_starpu_HFP_hmetis == 5) /* Cas matrice 3D */
	{
		size = strlen("Output_maxime/Data/input_hMETIS/") + strlen(str) + strlen("GPU_Matrice3D/input_hMETIS_N") + strlen(Nchar) + strlen(".txt");
	}
	else if (_starpu_HFP_hmetis == 6) /* Cas cholesky */
	{
		size = strlen("Output_maxime/Data/input_hMETIS/") + strlen(str) + strlen("GPU_Cholesky/input_hMETIS_N") + strlen(Nchar) + strlen(".txt");
	}
	else
	{
		size = strlen("Output_maxime/Data/input_hMETIS/") + strlen(str) + strlen("GPU/input_hMETIS_N") + strlen(Nchar) + strlen(".txt");
	}

	/* Cas sparse */
	if (sparse_matrix != 0)
	{
		size += strlen("_sparse");
	}

	char *path2 = (char *)malloc(size);
	strcpy(path2, "Output_maxime/Data/input_hMETIS/");
	strcat(path2, str);

	if (random_task_order == 1)
	{
		strcat(path2, "GPU_Random_task_order");
	}
	else if (_starpu_HFP_hmetis == 5) /* Cas matrice 3D */
	{
		strcat(path2, "GPU_Matrice3D");
	}
	else if (_starpu_HFP_hmetis == 6) /* Cas Cholesky */
	{
		strcat(path2, "GPU_Cholesky");
	}
	else
	{
		strcat(path2, "GPU");
	}

	/* Cas sparse */
	if (sparse_matrix != 0)
	{
		strcat(path2, "_sparse/input_hMETIS_N");
	}
	else
	{
		strcat(path2, "/input_hMETIS_N");
	}

	strcat(path2, Nchar);
	strcat(path2, ".txt");
	FILE *f_2 = fopen(path2, "r");
	free(path2);
	int number; int error;
	for (i = 0; i < _starpu_HFP_NT; i++)
	{
		error = fscanf(f_2, "%d", &number);
		if (error == 0)
		{
			printf("error fscanf in hMETIS input already generated\n");
			exit(0);
		}
		p->temp_pointer_1 = p->first_link;
		for (j = 0; j < number; j++)
		{
			p->temp_pointer_1 = p->temp_pointer_1->next;
		}
		task_1 = starpu_task_list_pop_front(l);
		p->temp_pointer_1->expected_time += starpu_task_expected_length(task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
		p->temp_pointer_1->expected_package_computation_time += starpu_task_expected_length(task_1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0);
		p->temp_pointer_1->nb_task_in_sub_list++;
		starpu_task_list_push_back(&p->temp_pointer_1->sub_list, task_1);
	}
	fclose(f_2);
}

static void init_visualisation(struct _starpu_HFP_paquets *a)
{
	print_order_in_file_hfp(a);
	//~ if (multigpu != 0 && (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") == 0))
	//~ {
		//~ visualisation_data_gpu_in_file_hfp_format_tex(a);
	//~ }
	//TODO corriger la manière dont je vide si il y a plus de 3 GPUs
	FILE *f = fopen("Output_maxime/Task_order_effective_0", "w"); /* Just to empty it before */
	fclose(f);
	f = fopen("Output_maxime/Task_order_effective_1", "w"); /* Just to empty it before */
	fclose(f);
	f = fopen("Output_maxime/Task_order_effective_2", "w"); /* Just to empty it before */
	fclose(f);
	f = fopen("Output_maxime/Data_coordinates_order_last_scheduler.txt", "w");
	fclose(f);
}

/* The function that sort the tasks in packages */
static struct starpu_task *HFP_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	//~ printf("Début de HFP_pull_task\n"); fflush(stdout);
	struct _starpu_HFP_sched_data *data = component->data;
	int i = 0;
	struct starpu_task *task1 = NULL;

	if (_starpu_HFP_do_schedule_done == true)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		/* If one or more task have been refused */
		data->p->temp_pointer_1 = data->p->first_link;
		if (data->p->temp_pointer_1->next != NULL)
		{
			for (i = 0; i < _nb_gpus; i++)
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
		//~ if (i == _nb_gpus) /* To return NULL to the cpus */
		//~ {
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ return NULL;
		//~ }
		if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list))
		{
			//~ printf("refused not empty.\n");
			task1 = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list);
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ printf("Return in pull_task because data->p->temp_pointer_1->refused_fifo_list not empty %p.\n", task1);
			return task1;
		}

		/* If the linked list is empty */
		if (_starpu_HFP_is_empty(data->p->first_link) == true)
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ printf("Return NULL linked list empty.\n");
			return NULL;
		}
		//~ printf("go to get task to return.\n");
		task1 = _starpu_HFP_get_task_to_return(component, to, data->p, _nb_gpus);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		//~ printf("Return in pull_task %p.\n", task1);
		return task1;
	}
	//~ printf("Return NULL.\n");
	return NULL;
}

static int HFP_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct _starpu_HFP_sched_data *data = component->data;
	int didwork = 0;
	int i = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//~ printf("%p failed\n");
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		data->p->temp_pointer_1 = data->p->first_link;
		int nb_gpu = _get_number_GPU();
		if (data->p->temp_pointer_1->next == NULL)
		{
			starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
		}
		else
		{
			//A corriger. En fait il faut push back dans une fifo a part puis pop back dans cette fifo dans pull task
			//Ici le pb c'est si plusieurs taches se font refusé
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
				//~ if (_print_in_terminal == 1) { fprintf(stderr, "pushed some tasks to %p\n", to); }
			//~ else
				//~ if (_print_in_terminal == 1) { fprintf(stderr, "I didn't have anything for %p\n", to); }
		//~ }
	}

	/* There is room now */
	return didwork || starpu_sched_component_can_push(component, to);
}

static int HFP_can_pull(struct starpu_sched_component * component)
{
	return starpu_sched_component_can_pull(component);
}

/* Fonction qui va appeller le scheduling en fonction multigpu et de hmetis. On peut ignorer son temps dans xgemm directement */
static void HFP_do_schedule(struct starpu_sched_component *component)
{
	struct _starpu_HFP_sched_data *data = component->data;
	struct starpu_task *task1 = NULL;
	int nb_of_loop = 0; /* Number of iteration of the while loop */
	int number_of_package_to_build = 0;
	number_of_package_to_build = _get_number_GPU(); /* Getting the number of GPUs */
	_starpu_HFP_GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx)))); /* Here we calculate the size of the RAM of the GPU. We allow our packages to have half of this size */

	/* If the linked list is empty, we can pull more tasks */
	if (_starpu_HFP_is_empty(data->p->first_link) == true)
	{
		if (!starpu_task_list_empty(&data->sched_list))
		{
			/* Si la liste initiale (sched_list) n'est pas vide, ce sont des tâches non traitées */
			EXPECTED_TIME = 0;
			_starpu_HFP_appli = starpu_task_get_name(starpu_task_list_begin(&data->sched_list));

			if (_starpu_HFP_hmetis != 0)
			{
				if (_starpu_HFP_hmetis == 3 || _starpu_HFP_hmetis == 4 || _starpu_HFP_hmetis == 5 || _starpu_HFP_hmetis == 6)
				{
					hmetis_input_already_generated(data->p, &data->sched_list, number_of_package_to_build);
				}
				else
				{
					_starpu_hmetis_scheduling(data->p, &data->sched_list, number_of_package_to_build);
				}
				if (_print_in_terminal == 1)
				{
					init_visualisation(data->p);
				}
				_starpu_HFP_do_schedule_done = true;
				return;
			}

			/* Pulling all tasks and counting them
			 * TODO : pas besoin de faire ca on peut faire size. Du coup faut suppr popped task list et la remplacer par sched list
			 */

			struct starpu_task_list *temp_task_list = starpu_task_list_new();
			starpu_task_list_init(temp_task_list);

			_starpu_HFP_NT = starpu_task_list_size(&data->sched_list);
			while (!starpu_task_list_empty(&data->sched_list))
			{
				task1 = starpu_task_list_pop_front(&data->sched_list);
				if (_print_in_terminal != 0)
				{
					printf("Tâche %p, %d donnée(s) : ",task1, STARPU_TASK_GET_NBUFFERS(task1));
					unsigned i;
					for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task1); i++)
					{
						printf("%p ",STARPU_TASK_GET_HANDLE(task1, i));
					}
					printf("\n");
				}
				if (multigpu != 0) { EXPECTED_TIME += starpu_task_expected_length(task1, starpu_worker_get_perf_archtype(STARPU_CUDA_WORKER, 0), 0); }

				//~ starpu_task_list_push_back(&data->popped_task_list, task1);
				starpu_task_list_push_back(temp_task_list, task1);
			}
			_starpu_HFP_N = sqrt(_starpu_HFP_NT);

			if(_print3d == 1)
			{
				_starpu_HFP_N = _starpu_HFP_N/2; /* So i can print just like a 2D matrix */
			}
			data->p->NP = _starpu_HFP_NT;

			//~ task1 = starpu_task_list_begin(&data->popped_task_list);
			//~ printf("%p\n", task1);
			//~ data->p = hierarchical_fair_packing(data->popped_task_list, NT, number_of_package_to_build);
			data->p = hierarchical_fair_packing(temp_task_list, _starpu_HFP_NT, number_of_package_to_build);

			/* Printing in terminal and also visu python */
			if (_print_in_terminal == 1)
			{
				_sched_visu_print_packages_in_terminal(data->p, nb_of_loop, "After first execution of HFP we have ---\n");
				int i = 0;
				int j = 0;
				int temp_tab_coordinates[2];
				FILE *f_last_package = fopen("Output_maxime/last_package_split.txt", "w");
				data->p->temp_pointer_1 = data->p->first_link;
				int sub_package = 0;

				while (data->p->temp_pointer_1 != NULL)
				{
					//~ printf("Début while.\n");
					j = 1;
					for (task1 = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); task1 != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); task1 = starpu_task_list_next(task1))
					{
						//~ _print_in_terminal ("On %p in printing.\n", task1);
						/* + 1 cause it's the next one that is in the other sub package */
						if (j == data->p->temp_pointer_1->split_last_ij + 1)
						{
							sub_package++;
						}
						if (_print3d != 0)
						{
							starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task1, 2), 2, temp_tab_coordinates);
							fprintf(f_last_package, "%d	%d", temp_tab_coordinates[0], temp_tab_coordinates[1]);
							starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task1, 0), 2, temp_tab_coordinates);
							fprintf(f_last_package, "	%d	%d	%d\n", temp_tab_coordinates[0], i, sub_package);
						}
						else
						{
							starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task1, 2), 2, temp_tab_coordinates);
							/* Printing X Y GPU SUBPACKAGE(1 - NSUBPACKAGES) */
							fprintf(f_last_package, "%d	%d	%d	%d\n", temp_tab_coordinates[0], temp_tab_coordinates[1], i, sub_package);
						}
						j++;
					}
					sub_package++;
					i++;
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
					//~ printf("Next.\n");
				}
				fclose(f_last_package);
				//~ printf("End of printing1.\n"); fflush(stdout);
			}

			/* Task stealing based on the number of tasks. Only in cases of multigpu */
			if (multigpu == 2 || multigpu == 3)
			{
				load_balance(data->p, number_of_package_to_build);
				_sched_visu_print_packages_in_terminal(data->p, nb_of_loop, "After load balance we have ---\n");
			}
			else if (multigpu == 4 || multigpu == 5) /* Task stealing with expected time of each task */
			{
				load_balance_expected_time(data->p, number_of_package_to_build);
				_sched_visu_print_packages_in_terminal(data->p, nb_of_loop, "After load balance we have with expected time ---\n");
			}
			else if (multigpu == 6 || multigpu == 7)
			{
				/* Task stealing with expected time of each package, with transfers and overlap */
				load_balance_expected_package_computation_time(data->p, _starpu_HFP_GPU_RAM_M);
				_sched_visu_print_packages_in_terminal(data->p, nb_of_loop, "After load balance we have with expected package computation time ---\n");
			}
			/* Re-apply HFP on each package.
			 * Once task stealing is done we need to re-apply HFP. For this I use an other instance of HFP_sched_data.
			 * It is in another function, if it work we can also put the packing above in it.
			 * Only with MULTIGPU = 2 because if we don't do load balance there is no point in re-applying HFP.
			 */
			 if (multigpu == 3 || multigpu == 5 || multigpu == 7)
			 {
				 data->p->temp_pointer_1 = data->p->first_link;
				 while (data->p->temp_pointer_1 != NULL)
				 {
					data->p->temp_pointer_1->sub_list = hierarchical_fair_packing_one_task_list(data->p->temp_pointer_1->sub_list, data->p->temp_pointer_1->nb_task_in_sub_list);
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				 }
				 _sched_visu_print_packages_in_terminal(data->p, nb_of_loop, "After execution of HFP on each package we have ---\n");
			 }

			 /* Interlacing package task list order */
			 if (interlacing != 0)
			 {
				 _sched_visu_print_packages_in_terminal(data->p, 0, "Before interlacing we have:\n");
				 interlacing_task_list(data->p);
				 _sched_visu_print_packages_in_terminal(data->p, 0, "After interlacing we have:\n");
			 }

			/* Belady */
			if (belady == 1)
			{
				get_ordre_utilisation_donnee(data->p, number_of_package_to_build);
			}

			/* If you want to get the sum of weight of all different data. Only works if you have only one package */
			//~ //if (_print_in_terminal == 1) { get_weight_all_different_data(data->p->first_link, _starpu_HFP_GPU_RAM_M); }

			/* We prefetch data for each task for modular-heft-HFP */
			if (modular_heft_hfp_mode != 0)
			{
				prefetch_each_task(data->p, component);
			}

			/* Printing in a file the order produced by HFP. If we use modular-heft-HFP, we can compare this order with the one done by modular-heft. We also print here the number of gpu in which a data is used for HFP's order. */
			if (_print_in_terminal == 1)
			{
				/* Todo a remetrre quand j'aurais corrigé le print_order_in_file_hfp */
				init_visualisation(data->p);
			}

			_starpu_HFP_do_schedule_done = true;
		}
	}
}

/* TODO a suppr */
//~ struct timeval time_start_eviction;
//~ struct timeval time_end_eviction;
//~ long long time_total_eviction = 0;

/* TODO a suppr */
//~ struct timeval time_start_createtolasttaskfinished;
//~ struct timeval time_end_createtolasttaskfinished;
//~ long long time_total_createtolasttaskfinished = 0;

static starpu_data_handle_t belady_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{
	(void)toload;
	STARPU_PTHREAD_MUTEX_LOCK(&HFP_mutex);
	//~ gettimeofday(&time_start_eviction, NULL);
	/* Checking if all task are truly valid. Else I return a non valid data
	 * pas indispensable en 2D mais sera utile plus tard. */
	/* for (i = 0; i < nb_data_on_node; i++)
	{
		if (valid[i] == 0 && starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		{
			free(valid);
			returned_handle = data_on_node[i];
			free(data_on_node);
			return returned_handle;
		}
	} */

	struct starpu_sched_component *temp_component = component;
	struct _starpu_HFP_sched_data *data = temp_component->data;
	starpu_data_handle_t returned_handle = NULL;
	int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;

	/* Je check si une eviction n'a pas été refusé. */
	data->p->temp_pointer_1 = data->p->first_link;
	int i;
	for (i = 0; i < current_gpu; i++)
	{
		data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	}
	if (data->p->temp_pointer_1->data_to_evict_next != NULL)
	{
		returned_handle = data->p->temp_pointer_1->data_to_evict_next;
		data->p->temp_pointer_1->data_to_evict_next = NULL;
		STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);

		//~ gettimeofday(&time_end_eviction, NULL);
		//~ time_total_eviction += (time_end_eviction.tv_sec - time_start_eviction.tv_sec)*1000000LL + time_end_eviction.tv_usec - time_start_eviction.tv_usec;

		//~ printf("Return 1 %p.\n", returned_handle); fflush(stdout);
		return returned_handle;
	}
	/* Sinon je cherche dans la mémoire celle utilisé dans le plus longtemps et que j'ai le droit d'évincer */
	starpu_data_handle_t *data_on_node;
	unsigned nb_data_on_node = 0;
	int *valid;
	starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);
	int latest_use = 0;
	int index_latest_use = 0;
	struct _starpu_HFP_next_use *b = NULL;

	unsigned j;
	for (j = 0; j < nb_data_on_node; j++)
	{
		if (starpu_data_can_evict(data_on_node[j], node, is_prefetch))
		{
			struct _starpu_HFP_next_use_by_gpu *c = _starpu_HFP_next_use_by_gpu_new();
			b = data_on_node[j]->sched_data;

			if (_starpu_HFP_next_use_by_gpu_list_empty(b->next_use_tab[current_gpu])) /* Si c'est vide alors je peux direct renvoyer cette donnée, elle ne sera jamais ré-utilisé */
			{
				STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);
				//~ printf("Return %p that is not used again.\n", data_on_node[i]);

				//~ gettimeofday(&time_end_eviction, NULL);
				//~ time_total_eviction += (time_end_eviction.tv_sec - time_start_eviction.tv_sec)*1000000LL + time_end_eviction.tv_usec - time_start_eviction.tv_usec;

				//~ printf("Return 2 %p.\n", data_on_node[i]); fflush(stdout);
				return data_on_node[j];
			}

			c = _starpu_HFP_next_use_by_gpu_list_begin(b->next_use_tab[current_gpu]);
			if (latest_use < c->value_next_use)
			{
				latest_use = c->value_next_use;
				index_latest_use = j;
			}
		}
	}
	if (latest_use == 0) /* Si je n'ai eu aucune donnée valide, je renvoie NO_VICTIM */
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);

		//~ gettimeofday(&time_end_eviction, NULL);
		//~ time_total_eviction += (time_end_eviction.tv_sec - time_start_eviction.tv_sec)*1000000LL + time_end_eviction.tv_usec - time_start_eviction.tv_usec;

		//~ printf("Return NO_VICTIM\n"); fflush (stdout);
		return STARPU_DATA_NO_VICTIM;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);

	//~ gettimeofday(&time_end_eviction, NULL);
	//~ time_total_eviction += (time_end_eviction.tv_sec - time_start_eviction.tv_sec)*1000000LL + time_end_eviction.tv_usec - time_start_eviction.tv_usec;

	return data_on_node[index_latest_use];
}

static void belady_victim_eviction_failed(starpu_data_handle_t victim, void *component)
{
	STARPU_PTHREAD_MUTEX_LOCK(&HFP_mutex);

	struct starpu_sched_component *temp_component = component;
	struct _starpu_HFP_sched_data *data = temp_component->data;

	/* If a data was not truly evicted I put it back in the list. */
	data->p->temp_pointer_1 = data->p->first_link;
	unsigned i;
	for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
	{
		data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	}
	data->p->temp_pointer_1->data_to_evict_next = victim;

	STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);
}

struct starpu_sched_component *starpu_sched_component_HFP_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	//~ gettimeofday(&time_start_createtolasttaskfinished, NULL);

	/* Global variable */
	order_u = starpu_get_env_number_default("ORDER_U", 1);
	belady = starpu_get_env_number_default("BELADY", 1);
	multigpu = starpu_get_env_number_default("MULTIGPU", 0);
	modular_heft_hfp_mode = starpu_get_env_number_default("MODULAR_HEFT_HFP_MODE", 0);
	_starpu_HFP_hmetis = starpu_get_env_number_default("HMETIS", 0);
	hmetis_n = starpu_get_env_number_default("HMETIS_N", 0);
	task_stealing = starpu_get_env_number_default("TASK_STEALING", 0);
	interlacing = starpu_get_env_number_default("INTERLACING", 0);
	faster_first_iteration = starpu_get_env_number_default("FASTER_FIRST_ITERATION", 0);

	_starpu_visu_init();

	starpu_srand48(starpu_get_env_number_default("SEED", 0));
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "HFP");

	assert(0);
	if (_print_in_terminal == 1)
	{
		FILE *f = fopen("Output_maxime/Data_stolen_load_balance.txt", "w");
		fclose(f);
	}

	struct _starpu_HFP_sched_data *data;
	struct _starpu_HFP_my_list *my_data = malloc(sizeof(*my_data));
	struct _starpu_HFP_paquets *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));

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
	data->p->temp_pointer_1->data_weight = 0;

	data->p->temp_pointer_1->expected_time = 0;

	component->data = data;
	component->do_schedule = HFP_do_schedule;
	component->push_task = HFP_push_task;
	component->pull_task = HFP_pull_task;
	component->can_push = HFP_can_push;
	component->can_pull = HFP_can_pull;

	STARPU_PTHREAD_MUTEX_INIT(&HFP_mutex, NULL);

	number_task_out = 0;
	iteration = 0;

	/* TODO init du temps a suppr si on mesure plus le temps. A suppr */
	//~ time_total_getorderbelady = 0;
	//~ time_total_getcommondataorderu = 0;
	//~ time_total_gettasktoreturn = 0;
	//~ time_total_scheduling = 0;
	//~ time_total_loadbalanceexpectedtime = 0;
	//~ time_total_createtolasttaskfinished = 0;
	//~ time_total_eviction = 0;

	if (belady == 1)
	{
		starpu_data_register_victim_selector(belady_victim_selector, belady_victim_eviction_failed, component);
	}

	return component;
}

static void initialize_HFP_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_HFP_create, NULL,
							   STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
							   STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY | /* ready of dmdar plugged into HFP */
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
							   STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_HFP_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

/* Get the task that was last executed. Used to update the task list of pulled task	 */
static void get_task_done_HFP(struct starpu_task *task, unsigned sci)
{
	STARPU_PTHREAD_MUTEX_LOCK(&HFP_mutex);
	number_task_out++;
	/* Je me place sur le bon gpu. */
	int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;

	/* Si j'utilse Belady, je pop les valeurs dans les données de la tâche qui vient de se terminer */
	if (belady == 1)
	{
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			struct _starpu_HFP_next_use *b = STARPU_TASK_GET_HANDLE(task, i)->sched_data;
			if (!_starpu_HFP_next_use_by_gpu_list_empty(b->next_use_tab[current_gpu])) /* Test empty car avec le task stealing ca n'a plus aucun sens */
			{
				_starpu_HFP_next_use_by_gpu_list_pop_front(b->next_use_tab[current_gpu]);
				STARPU_TASK_GET_HANDLE(task, i)->sched_data = b;
			}
		}
	}

	/* Reset pour prochaine itération à faire ici quand le nombe de tâches sortie == NT si besoin */
	if (_starpu_HFP_NT == number_task_out)
	{
		iteration++;
		_starpu_HFP_do_schedule_done = false;
		number_task_out = 0;

		/* TODO a suppr. PRINT_TIME sur 2 permet de forcer l'écriture en simulation car il y a 1 seule itération. */
		//~ if ((iteration == 3 && starpu_get_env_number_default("PRINT_TIME", 0) == 1) || starpu_get_env_number_default("PRINT_TIME", 0) == 2)
		//~ {
			//~ FILE *f = fopen("Output_maxime/HFP_time.txt", "a");
			//~ fprintf(f, "%0.0f	", sqrt(NT));
			//~ fprintf(f, "%lld	", time_total_scheduling);
			//~ fprintf(f, "%lld	", time_total_eviction);
			//~ fprintf(f, "%lld	", time_total_getorderbelady);
			//~ fprintf(f, "%lld	", time_total_getcommondataorderu);
			//~ fprintf(f, "%lld	", time_total_gettasktoreturn);
			//~ fprintf(f, "%lld	", time_total_loadbalanceexpectedtime);
			//~ gettimeofday(&time_end_createtolasttaskfinished, NULL);
			//~ time_total_createtolasttaskfinished += (time_end_createtolasttaskfinished.tv_sec - time_start_createtolasttaskfinished.tv_sec)*1000000LL + time_end_createtolasttaskfinished.tv_usec - time_start_createtolasttaskfinished.tv_usec;
			//~ fprintf(f, "%lld	", time_total_createtolasttaskfinished);
			//~ fprintf(f, "%lld	", time_total_find_min_size);
			//~ fprintf(f, "%lld	", time_total_init_packages);
			//~ fprintf(f, "%lld	", time_total_fill_matrix_common_data_plus_get_max);
			//~ fprintf(f, "%lld	", time_total_reset_init_start_while_loop);
			//~ fprintf(f, "%lld	", time_total_order_u_total);
			//~ fprintf(f, "%lld\n", time_total_merge);
			//~ fclose(f);
		//~ }
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&HFP_mutex);
	starpu_sched_component_worker_pre_exec_hook(task, sci);
}

struct starpu_sched_policy _starpu_sched_HFP_policy =
{
	.init_sched = initialize_HFP_center_policy,
	.deinit_sched = deinitialize_HFP_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = _sched_visu_get_data_to_load, /* To get the number of data needed for the current task, still return the task that we got with starpu_sched_tree_pop_task */
	.pre_exec_hook = _sched_visu_get_current_tasks, /* Getting current task for printing diff later on. Still call starpu_sched_component_worker_pre_exec_hook(task,sci); at the end */
	.post_exec_hook = get_task_done_HFP, /* Sert pour Belady et aussi pour afficher les temps d'exec. A ne pas retirer pour Belady */
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
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = _sched_visu_get_current_tasks_for_visualization,
	//~ .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.policy_name = "modular-heft-HFP",
	.policy_description = "heft modular policy",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};
