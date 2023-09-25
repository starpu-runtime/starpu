/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SCHED_HFP_H__
#define __SCHED_HFP_H__

#include <starpu.h>
#include <starpu_sched_component.h>

#include <core/task.h>
#include <core/sched_policy.h>
#include <common/list.h>
#include <sched_policies/prio_deque.h>
#include <sched_policies/darts.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>

#pragma GCC visibility push(hidden)

#define ORDER_U /* O or 1 */
#define BELADY /* O or 1 */
#define MULTIGPU /* 0 : on ne fais rien, 1 : on construit |GPU| paquets et on attribue chaque paquet à un GPU au hasard, 2 : pareil que 1 + load balance, 3 : pareil que 2 + HFP sur chaque paquet, 4 : pareil que 2 mais avec expected time a la place du nb de données, 5 pareil que 4 + HFP sur chaque paquet, 6 : load balance avec expected time d'un paquet en comptant transferts et overlap, 7 : pareil que 6 + HFP sur chaque paquet */
#define MODULAR_HEFT_HFP_MODE /* 0 we don't use heft, 1 we use starpu_prefetch_task_input_on_node_prio, 2 we use starpu_idle_prefetch_task_input_on_node_prio. Put it at 1 or 2 if you use modular-heft-HFP, else it will crash. The 0 is just here so we don't do prefetch when we use regular HFP. If we do not use modular-heft-HFP, always put this environemment variable on 0. */
#define HMETIS /* 0 we don't use hMETIS, 1 we use it to form |GPU| package, 2 same as 1 but we then apply HFP on each package. For mst if it is equal to 1 we form |GPU| packages then apply mst on each package, 3 we use hMETIS with already produced input files, 4 same
but we apply HFP on each package (pas codé en réalité car j'avais changé la manière d'appeller hfp, il faudrait mettre hierarchical_fair_packing_one_task_list ou un truc du genre, 5 we use hMETIS in 3D already generated, 6 it's cholesky with already generated. + Il faut préciser en var d'env le N avec HMETIS_N. Already generated is used in Grid5k or PlaFRIM when we don't have access to the hMETIS's executable + it saves precious time. */
#define HMETIS_N /* Préciser N */
#define PRINT3D /* 1 we print coordinates and visualize data. 2 same but it is 3D with Z = N. Needed to differentiate 2D from 3D. */
#define TASK_STEALING /* 0 we don't use it, 1 when a gpu (so a package) has finished all it tasks, it steal a task, starting by the end of the package of the package that has the most tasks left. It can be done with load balance on but was first thinked to be used with no load balance bbut |GPU| packages (MULTIGPU=1), 2 same than 1 but we steal from the package that has the biggest expected package time, 3 same than 2 but we always steal half (arondi à l'inférieur) of the package at once (in term of task duration). All that is implemented in get_task_to_return */
#define INTERLACING /* 0 we don't use it, 1 we start giving task at the middle of the package then do right, left and so on. */
#define FASTER_FIRST_ITERATION /* A 0 on ne fais rien, a 1 on le fais. Permet de faire une première itération où on merge ensemble els taches partageant une données sans regarder le max et donc sans calculer la matrice. Ne marche que pour matrice 2D, 3D. */

extern int _starpu_HFP_hmetis;

extern const char* _starpu_HFP_appli;
extern int _starpu_HFP_NT;
//extern int N;
extern starpu_ssize_t _starpu_HFP_GPU_RAM_M;
extern bool _starpu_HFP_do_schedule_done;

/* Structure used to acces the struct my_list. There are also task's list */
struct _starpu_HFP_sched_data
{
	//~ struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct _starpu_HFP_paquets *p;
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

/* Structure used to store all the variable we need and the tasks of each package. Each link is a package */
struct _starpu_HFP_my_list
{
	int package_nb_data;
	int nb_task_in_sub_list;
	int index_package; /* Utilisé dans MST pour le scheduling */
	starpu_data_handle_t * package_data; /* List of all the data in the packages. We don't put two times the duplicates */
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct starpu_task_list refused_fifo_list; /* if a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */
	struct _starpu_HFP_my_list *next;
	int split_last_ij; /* The separator of the last state of the current package */
	//~ starpu_data_handle_t * data_use_order; /* Order in which data will be loaded. used for Belady */
	double expected_time; /* Only task's time */
	double expected_time_pulled_out; /* for load balance but only MULTIGPU = 4, 5 */
	double expected_package_computation_time; /* Computation time with transfer and overlap */
	struct _starpu_HFP_data_on_node *pointer_node; /* linked list of handle use to simulate the memory in load balance with package with expected time */
	long int data_weight;

	starpu_data_handle_t data_to_evict_next;
};

struct _starpu_HFP_paquets
{
	/* All the pointer use to navigate through the linked list */
	struct _starpu_HFP_my_list *temp_pointer_1;
	struct _starpu_HFP_my_list *temp_pointer_2;
	struct _starpu_HFP_my_list *temp_pointer_3;
	struct _starpu_HFP_my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	int NP; /* Number of packages */
};

/* TODO : ou est-ce que j'utilise ca ? A suppr si inutile */
struct _starpu_HFP_data_on_node /* Simulate memory, list of handles */
{
	struct _starpu_HFP_handle *pointer_data_list;
	struct _starpu_HFP_handle *first_data;
	long int memory_used;
};

struct _starpu_HFP_handle /* The handles from above */
{
	starpu_data_handle_t h;
	int last_use;
	struct _starpu_HFP_handle *next;
};

/** Dans sched_data des données pour avoir la liste des prochaines utilisations que l'on peut pop à chaque utilisation dans get_task_done **/
LIST_TYPE(_starpu_HFP_next_use_by_gpu,
	  /* int to the next use, one by GPU */
	  int value_next_use;
	);

struct _starpu_HFP_next_use
{
	struct _starpu_HFP_next_use_by_gpu_list **next_use_tab;
};

void _starpu_HFP_initialize_global_variable(struct starpu_task *task);

/* Put a link at the beginning of the linked list */
void _starpu_HFP_insertion(struct _starpu_HFP_paquets *a);

/*
 * Called in HFP_pull_task when we need to return a task. It is used
 * when we have multiple GPUs.
 * In case of modular-heft-HFP, it needs to do a round robin on the
 * task it returned. So we use expected_time_pulled_out, an element of
 * struct my_list in order to track which package pulled out the least
 * expected task time. So heft can better divide tasks between GPUs
 */
struct starpu_task *_starpu_HFP_get_task_to_return(struct starpu_sched_component *component, struct starpu_sched_component *to, struct _starpu_HFP_paquets* a, int nb_gpu);

/* Check if our struct is empty */
bool _starpu_HFP_is_empty(struct _starpu_HFP_my_list* a);

void _starpu_hmetis_scheduling(struct _starpu_HFP_paquets *p, struct starpu_task_list *l, int nb_gpu);

void _starpu_visu_init();

#pragma GCC visibility pop

#endif // __SCHED_HFP_H__
