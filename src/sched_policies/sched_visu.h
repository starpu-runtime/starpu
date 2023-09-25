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

#ifndef __SCHED_VISU_H__
#define __SCHED_VISU_H__

// environment variables:
// STARPU_SCHED_PRINT_IN_TERMINAL
//       O we print nothing, 1 we print in terminal and also fill data
//       coordinate order, task order etc... so it can take more time.
// STARPU_SCHED_PRINT_N
//       to precise the value of N for visualization in scheduelers
//       that does not count the toal number of tasks. Also use
//       PRINT3D=1 or 2 so we know we are in 3D
// STARPU_SCHED_PRINT_TIME
//        Pour afficher le temps d'exécution des fonctions dans HFP. A
//        1 on print à la 11ème itération et on fera la moyenne. A 2
//        on print à la première itération. Utile pour que cela
//        fonctionne avec Grid5k.

#include <starpu.h>
#include <sched_policies/HFP.h>
#include <common/config.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_DARTS_VERBOSE
#define PRINT /* A dé-commenter pour afficher les printfs dans le code, les mesures du temps et les écriture dans les fichiers. A pour objectif de remplacer la var d'env PRINTF de HFP. Pour le moment j'ai toujours besoin de PRINTF=1 pour les visualisations par exemple. Attention pour DARTS j'ai besoin de PRINTF=1 et de PRINT pour les visu pour le moment. */
#define PRINT_PYTHON /* Visu python */
#endif

#ifdef PRINT
#  define _STARPU_SCHED_PRINT(fmt, ...) do { fprintf(stderr, fmt, ## __VA_ARGS__); fflush(stderr); } while(0)
#else
#  define _STARPU_SCHED_PRINT(fmt, ...) do { } while (0)
#endif

extern int _print3d;
extern int _print_in_terminal;
extern int _print_n;
extern struct starpu_task *task_currently_treated;

void _sched_visu_init(int nb_gpus);
char *_sched_visu_get_output_directory();
void _sched_visu_print_data_to_load_prefetch(struct starpu_task *task, int gpu_id, int force);
void _sched_visu_pop_ready_task(struct starpu_task *task);
struct starpu_task *_sched_visu_get_data_to_load(unsigned sched_ctx);
void _sched_visu_print_packages_in_terminal(struct _starpu_HFP_paquets *a, int nb_of_loop, const char *msg);
void _sched_visu_print_effective_order_in_file(struct starpu_task *task, int index_task);
void _sched_visu_get_current_tasks(struct starpu_task *task, unsigned sci);
void _sched_visu_get_current_tasks_for_visualization(struct starpu_task *task, unsigned sci);

void _sched_visu_print_matrix(int **matrix, int x, int y, char *msg);
void _sched_visu_print_vector(int *vector, int x, char *msg);
void _sched_visu_print_data_for_task(struct starpu_task *task, const char *msg);

void _sched_visu_get_current_tasks_for_visualization(struct starpu_task *task, unsigned sci);

#pragma GCC visibility pop

#endif // __SCHED_VISU_H__
