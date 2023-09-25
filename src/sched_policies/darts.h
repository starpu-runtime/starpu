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

#ifndef __DARTS_H__
#define __DARTS_H__

/** @file */

#include <stdbool.h>

#pragma GCC visibility push(hidden)

#define STARPU_DARTS_GRAPH_DESCENDANTS /* To use the graph as a way to add priorities. 0: Not used. 1: With a graph reading descendants in DARTS, use a pause in the task submit. 2: With a graph reading descendants in DARTS, but don't use pause and the graph is read at each new batch of tasks in pull_task. */
// 0 by default for all the following global variables
#define STARPU_DARTS_EVICTION_STRATEGY_DARTS /* 0: LRU, 1: special eviction for DARTS */
#define STARPU_DARTS_THRESHOLD /* Pour arrêter de regarder dans la liste des données plus tôt. 0 = no threshold, 1 = threshold à 14400 tâches pour une matrice 2D (donc STARPU_DARTS_APP == 0) et à 1599 tâches aussi pour matrice 3D (donc STARPU_DARTS_APP == 1), 2 = on s'arrete des que on a trouvé 1 donnée qui permet de faire au moins une tache gratuite ou si il y en a pas 1 donnée qui permet de faire au moins 1 tache a 1 d'ere gratuite. 0 par défaut */
#define STARPU_DARTS_APP /* 0 matrice 2D. 1 matrice 3D. Sur 1 on regarde les tâches à 1 d'être gratuite galement. Pas plus loin. */
#define STARPU_DARTS_CHOOSE_BEST_DATA_FROM /* Pour savoir où on regarde pour choisir la meilleure donnée. 0, on regarde la liste des données pas encore utilisées. 1 on regarde les données en mémoire et à partir des tâches de ces données on cherche une donnée pas encore en mémoire qui permet de faire le plus de tâches gratuite ou 1 from free. */
#define STARPU_DARTS_SIMULATE_MEMORY /* Default 0, means we use starpu_data_is_on_node, 1 we also look at nb of task in planned and pulled task. */
#define STARPU_DARTS_TASK_ORDER /* 0, signifie qu'on randomize entièrement la liste des tâches. 1 je ne randomise que les nouvelles tâches entre elle et les met à la fin des listes de taches. 2 je ne randomise pas et met chaque GPU sur un m/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de tâches. Dans le cas avec dépendances il n'y a pas de points de départs différents juste je ne randomise pas. */
#define STARPU_DARTS_DATA_ORDER /* 0, signifie qu'on randomize entièrement la liste des données. 1 je ne randomise que les nouvelles données entre elle et les met à la fin des listes de données. 2 je ne randomise pas et met chaque GPU sur un Ndata/NGPU portion différentes pour qu'ils commencent à différent endroit de la liste de données.*/
#define STARPU_DARTS_DEPENDANCES /* 0 non, 1 utile pour savoir si on fais des points de départs différents dans main task list (on ne le fais pas si il y a des dependances). TODO: pas forcément utile à l'avenir à voir si on l'enlève. */
#define STARPU_DARTS_PRIO /* 0 non, 1 tiebreak data selection with the that have the highest priority task */
#define STARPU_DARTS_FREE_PUSHED_TASK_POSITION /* To detail where a free task from push_task is pushed in planned_task. 0: at the top of planned task, 1: after the last free task of planned task. */
#define STARPU_DARTS_DOPT_SELECTION_ORDER /* In which order do I tiebreak when choosing the optimal data:
* 0: Nfree N1fromfree Prio Timeremaining
* 1: Nfree Prio N1fromfree Timeremaining
* 2: TransferTime NFree Prio N1FromFree TimeRemaining
* 3: NFree TransferTime Prio N1FromFree TimeRemaining
* 4: NFree Prio TransferTime N1FromFree TimeRemaining
* 5: NFree Prio N1FromFree TransferTime TimeRemaining
* 6: NFree Prio N1FromFree TimeRemaining TransferTime
* 7: Ratio_Transfer/Free_Task_Time NFree Prio N1FromFree TimeRemaining
*/
#define STARPU_DARTS_HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE /* 0: no, 1: I return the highesth priority task of the list when I didn't found a data giving free or 1 from free task. Only makes sense if STARPU_DARTS_TASK_ORDER is set to 2. else you are defeating the purpose of randomization with TASK_ORDEr on 0 or 1. */
#define STARPU_DARTS_CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET /* 0: no, 1 : yes */
#define STARPU_DARTS_PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK /* 0: no, 1: yes, 2: round robin */
#define STARPU_DARTS_CPU_ONLY /* 0: we use only GPUs, 1: we use only CPUs, 2: we use both (not functionnal) */

#pragma GCC visibility pop

#endif // __DARTS_H__
