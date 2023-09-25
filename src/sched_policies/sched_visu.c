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

#include <common/config.h>
#include <sched_policies/sched_visu.h>
#include <datawizard/memory_nodes.h>

#ifdef PRINT_PYTHON
static int *_index_current_popped_task;
static int _index_current_popped_task_all_gpu;
static int *_index_current_popped_task_prefetch;
static int _index_current_popped_task_all_gpu_prefetch;
#endif
int _print3d;
int _print_in_terminal;
int _print_n;
#ifdef PRINT_PYTHON
static int index_task_currently_treated=0;
#endif
static int _index_current_task_for_visualization=0;
struct starpu_task *task_currently_treated = NULL;

static char *_output_directory = NULL;
char *_sched_visu_get_output_directory()
{
	if (_output_directory == NULL)
	{
		_output_directory = starpu_getenv("STARPU_SCHED_OUTPUT");
		if (_output_directory == NULL)
			_output_directory = "/tmp";
		_starpu_mkpath_and_check(_output_directory, S_IRWXU);
	}
	return _output_directory;
}

void _sched_visu_init(int nb_gpus)
{
	(void)nb_gpus;
#ifdef PRINT_PYTHON
	_index_current_popped_task = malloc(sizeof(int)*nb_gpus);
	_index_current_popped_task_prefetch = malloc(sizeof(int)*nb_gpus);
	_index_current_popped_task_all_gpu = 0;
	_index_current_popped_task_all_gpu_prefetch = 0;
#endif
	_print3d = starpu_get_env_number_default("STARPU_SCHED_PRINT3D", 0);
	_print_in_terminal = starpu_get_env_number_default("STARPU_SCHED_PRINT_IN_TERMINAL", 0);
	_print_n = starpu_get_env_number_default("STARPU_SCHED_PRINT_N", 0);
}

/* Printing in a file the coordinates and the data loaded during prefetch for each task for visu python */
void _sched_visu_print_data_to_load_prefetch(struct starpu_task *task, int gpu_id, int force)
{
	(void)task;
	(void)gpu_id;
	(void)force;
#ifdef PRINT_PYTHON
	int current_gpu = gpu_id;
	_index_current_popped_task_prefetch[current_gpu]++; /* Increment popped task on the right GPU */
	_index_current_popped_task_all_gpu_prefetch++;
	int nb_data_to_load = 0;
	int x_to_load = 0;
	int y_to_load = 0;
	int z_to_load = 0;
	unsigned int i = 0;
	/* Getting the number of data to load */
	for (i = 0; i <  STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		if(!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), gpu_id))
		{
			nb_data_to_load++;

			/* To know if I load a line or a column */
			if (i == 0)
			{
				x_to_load = 1;
			}
			if (i == 1)
			{
				y_to_load = 1;
			}
			if (i == 2)
			{
				z_to_load = 1;
			}
		}
	}
	/* Printing the number of data to load */
	FILE *f2 = NULL;
	int tab_coordinates[2];

	if (strcmp(starpu_task_get_name(task), "starpu_sgemm_gemm") == 0)
	{
		starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
		int size = strlen(_output_directory) + strlen("/Data_to_load_prefetch_SCHEDULER.txt") + 1;
		char path[size];
		snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_prefetch_SCHEDULER.txt");
		if (_index_current_popped_task_all_gpu_prefetch == 1)
		{
			f2 = fopen(path, "w");
		}
		else
		{
			f2 = fopen(path, "a");
		}
		STARPU_ASSERT_MSG(f2, "cannot open file <%s>\n", path);
		if (_print3d != 0)
		{
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
			fprintf(f2, "%d	%d", tab_coordinates[0], tab_coordinates[1]);
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, tab_coordinates);
			fprintf(f2, "	%d	%d	%d	%d	%d\n", tab_coordinates[0], x_to_load, y_to_load, z_to_load, current_gpu);
		}
		else
		{
			fprintf(f2, "%d	%d	%d	%d	%d\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, current_gpu);
		}
	}
	else if (strcmp(starpu_task_get_name(task), "POTRF") == 0 || strcmp(starpu_task_get_name(task), "SYRK") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0 || strcmp(starpu_task_get_name(task), "GEMM") == 0)
	{
		/* Ouverture du fichier. */
		int size = strlen(_output_directory) + strlen("/Data_to_load_prefetch_SCHEDULER.txt") + 1;
		char path[size];
		snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_prefetch_SCHEDULER.txt");
		if (_index_current_popped_task_all_gpu_prefetch == 1)
		{
			f2 = fopen(path, "w");
			fprintf(f2, "TASK	COORDY	COORDX	XTOLOAD YTOLOAD ZTOLOAD	GPU	ITERATIONK\n");
		}
		else
		{
			f2 = fopen(path, "a");
		}

		/* Impression du type de tâche. */
		if (strcmp(starpu_task_get_name(task), "POTRF") == 0)
		{
			fprintf(f2, "POTRF");
		}
		else if (strcmp(starpu_task_get_name(task), "TRSM") == 0)
		{
			fprintf(f2, "TRSM");
		}
		else
		{
			/* Cas SYRK et GEMM que je distingue avec la donnée en double pour SYRK. */
			if (STARPU_TASK_GET_HANDLE(task, 0) == STARPU_TASK_GET_HANDLE(task, 1))
			{
				fprintf(f2, "SYRK");
				/* Attention pour SYRK il ne faut pas compter en double la donnée à charger. Donc je regarde si je l'a compté en double je fais --. */
				if(!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, 0), current_gpu))
				{
					y_to_load = 0;
				}
			}
			else
			{
				fprintf(f2, "GEMM");
			}
		}
		/* La je n'imprime que les coords de la dernière donnée de la tâche car c'est ce qui me donne la place dans le triangle de Cholesky. */
		starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, STARPU_TASK_GET_NBUFFERS(task) - 1), 2, tab_coordinates);
		fprintf(f2, "	%d	%d	%d	%d	%d	%d	%ld\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, z_to_load, current_gpu, task->iterations[0]);

	}
	else
	{
		_STARPU_DISP("There is only support for GEMM and CHOLESKY currently. Task %s is not supported.\n", starpu_task_get_name(task));
	}
	if (f2) fclose(f2);
#endif
}

void _sched_visu_pop_ready_task(struct starpu_task *task)
{
	(void)task;
/* Getting the data we need to fetch for visualization */
#ifdef PRINT_PYTHON
	if (_index_current_task_for_visualization == 0)
	{
		_output_directory = _sched_visu_get_output_directory();
	}

	if (task != NULL)
	{
		int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
		_index_current_popped_task[current_gpu]++; /* Increment popped task on the right GPU */
		_index_current_popped_task_all_gpu++;
		int nb_data_to_load = 0;
		int x_to_load = 0;
		int y_to_load = 0;
		int z_to_load = 0;
		unsigned i;
		/* Getting the number of data to load */
		for (i = 0; i <  STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			if(!starpu_data_is_on_node_excluding_prefetch(STARPU_TASK_GET_HANDLE(task, i), current_gpu))
			{
				nb_data_to_load++;

				/* To know if I load a line or a column. Attention ca marche pas si plus de 3 données dans la tâche. */
				if (i == 0)
				{
					x_to_load = 1;
				}
				else if (i == 1)
				{
					y_to_load = 1;
				}
				else if (i == 2)
				{
					z_to_load = 1;
				}
				else
				{
					perror("Cas pas géré dans get data to load.\n"); exit(0);
				}
			}
		}
		FILE *f2 = NULL;
		int tab_coordinates[2];
		/* Cas 2D et 3D qui marche. */
		if (strcmp(starpu_task_get_name(task), "starpu_sgemm_gemm") == 0)
		{
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
			int size = strlen(_output_directory) + strlen("/Data_to_load_SCHEDULER.txt") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_SCHEDULER.txt");
			if (_index_current_popped_task_all_gpu == 1)
			{
				f2 = fopen(path, "w");
			}
			else
			{
				f2 = fopen(path, "a");
			}
			if (_print3d != 0)
			{
				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
				fprintf(f2, "%d	%d", tab_coordinates[0], tab_coordinates[1]);
				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, tab_coordinates);
				fprintf(f2, "	%d	%d	%d	%d	%d\n", tab_coordinates[0], x_to_load, y_to_load, z_to_load, current_gpu - 1);
			}
			else
			{
				fprintf(f2, "%d	%d	%d	%d	%d\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, current_gpu - 1);
			}
		}
		else if (strcmp(starpu_task_get_name(task), "POTRF") == 0 || strcmp(starpu_task_get_name(task), "SYRK") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0 || strcmp(starpu_task_get_name(task), "GEMM") == 0)
		{
			/* Ouverture du fichier. */
			int size = strlen(_output_directory) + strlen("/Data_to_load_SCHEDULER.txt") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_SCHEDULER.txt");

			if (_index_current_popped_task_all_gpu == 1)
			{
				f2 = fopen(path, "w");
				fprintf(f2, "TASK	COORDY	COORDX	XTOLOAD YTOLOAD ZTOLOAD	GPU	ITERATIONK\n");
			}
			else
			{
				f2 = fopen(path, "a");
			}

			/* Impression du type de tâche. */
			if (strcmp(starpu_task_get_name(task), "chol_model_11") == 0 || strcmp(starpu_task_get_name(task), "POTRF") == 0)
			{
				fprintf(f2, "POTRF");
			}
			else if (strcmp(starpu_task_get_name(task), "chol_model_21") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0)
			{
				fprintf(f2, "TRSM");
			}
			else
			{
				/* Cas SYRK et GEMM que je distingue avec la donnée en double pour SYRK. */
				if (STARPU_TASK_GET_HANDLE(task, 0) == STARPU_TASK_GET_HANDLE(task, 1))
				{
					fprintf(f2, "SYRK");
					if(!starpu_data_is_on_node_excluding_prefetch(STARPU_TASK_GET_HANDLE(task, 0), current_gpu))
					{
						y_to_load = 0;
					}
				}
				else
				{
					fprintf(f2, "GEMM");
				}
			}

			/* La je n'imprime que les coords de la dernière donnée de la tâche car c'est ce qui me donne la place dans le triangle de Cholesky. */
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, STARPU_TASK_GET_NBUFFERS(task) - 1), 2, tab_coordinates);
			fprintf(f2, "	%d	%d	%d	%d	%d	%d	%ld\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, z_to_load, current_gpu - 1, task->iterations[0]);
		}
		else
		{
			_STARPU_DISP("Dans get data to load je ne gère que GEMM et CHOLESKY. Task %s is not supported.\n", starpu_task_get_name(task));
		}
		fclose(f2);
	}
#endif
}

/* Used for visualisation python */
struct starpu_task *_sched_visu_get_data_to_load(unsigned sched_ctx)
{
#ifndef PRINT_PYTHON
	return starpu_sched_tree_pop_task(sched_ctx);
#else
	struct starpu_task *task = starpu_sched_tree_pop_task(sched_ctx);
	if (task != NULL)
	{
		//~ int current_gpu = starpu_worker_get_id();
		int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
		//~ printf("Ngpu = %d current = %d task = %p\n", _nb_gpus, current_gpu, task);
		_index_current_popped_task[current_gpu]++; /* Increment popped task on the right GPU */
		_index_current_popped_task_all_gpu++;
		int nb_data_to_load = 0;
		int x_to_load = 0;
		int y_to_load = 0;
		int z_to_load = 0;
		/* Getting the number of data to load */
		unsigned i;
		for (i = 0; i <  STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			if(!starpu_data_is_on_node_excluding_prefetch(STARPU_TASK_GET_HANDLE(task, i), current_gpu))
			{
				nb_data_to_load++;

				/* To know if I load a line or a column. Attention ca marche pas si plus de 3 données dans la tâche. */
				if (i == 0)
				{
					x_to_load = 1;
				}
				else if (i == 1)
				{
					y_to_load = 1;
				}
				else if (i == 2)
				{
					z_to_load = 1;
				}
				else
				{
					perror("Cas pas géré dans get data to load.\n"); exit(0);
				}
			}
		}

		//~ printf("%s in get_data_to_load.\n", starpu_task_get_name(task));

		/* Printing the number of data to load */
		FILE *f2 = NULL;

		int tab_coordinates[2];

		/* Cas 2D et 3D qui marche. */
		if (strcmp(starpu_task_get_name(task), "starpu_sgemm_gemm") == 0)
		{
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);

			int size = strlen(_output_directory) + strlen("/Data_to_load_SCHEDULER.txt") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_SCHEDULER.txt");

			if (_index_current_popped_task_all_gpu == 1)
			{
				f2 = fopen(path, "w");
			}
			else
			{
				f2 = fopen(path, "a");
			}
			if (_print3d != 0)
			{
				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
				fprintf(f2, "%d	%d", tab_coordinates[0], tab_coordinates[1]);
				starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, tab_coordinates);
				fprintf(f2, "	%d	%d	%d	%d	%d\n", tab_coordinates[0], x_to_load, y_to_load, z_to_load, current_gpu - 1);
			}
			else
			{
				fprintf(f2, "%d	%d	%d	%d	%d\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, current_gpu - 1);
			}
		}
		else if (strcmp(starpu_task_get_name(task), "chol_model_11") == 0 || strcmp(starpu_task_get_name(task), "chol_model_21") == 0 || strcmp(starpu_task_get_name(task), "chol_model_22") == 0 || strcmp(starpu_task_get_name(task), "POTRF") == 0 || strcmp(starpu_task_get_name(task), "SYRK") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0 || strcmp(starpu_task_get_name(task), "GEMM") == 0)
		{
			/* Ouverture du fichier. */
			int size = strlen(_output_directory) + strlen("/Data_to_load_SCHEDULER.txt") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_to_load_SCHEDULER.txt");
			if (_index_current_popped_task_all_gpu == 1)
			{
				f2 = fopen(path, "w");
				fprintf(f2, "TASK	COORDY	COORDX	XTOLOAD YTOLOAD ZTOLOAD	GPU	ITERATIONK\n");
			}
			else
			{
				f2 = fopen(path, "a");
			}

			/* Impression du type de tâche. */
			if (strcmp(starpu_task_get_name(task), "chol_model_11") == 0 || strcmp(starpu_task_get_name(task), "POTRF") == 0)
			{
				fprintf(f2, "POTRF");
			}
			else if (strcmp(starpu_task_get_name(task), "chol_model_21") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0)
			{
				fprintf(f2, "TRSM");
			}
			else
			{
				/* Cas SYRK et GEMM que je distingue avec la donnée en double pour SYRK. */
				if (STARPU_TASK_GET_HANDLE(task, 0) == STARPU_TASK_GET_HANDLE(task, 1))
				{
					fprintf(f2, "SYRK");
					/* Attention pour SYRK il ne faut pas compter en double la donnée à charger. Donc je regarde si je l'a compté en double je fais --. */
					if(!starpu_data_is_on_node_excluding_prefetch(STARPU_TASK_GET_HANDLE(task, 0), current_gpu))
					{
						y_to_load = 0;
					}
				}
				else
				{
					fprintf(f2, "GEMM");
				}
			}

			/* La je n'imprime que les coords de la dernière donnée de la tâche car c'est ce qui me donne la place dans le triangle de Cholesky. */
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, STARPU_TASK_GET_NBUFFERS(task) - 1), 2, tab_coordinates);
			fprintf(f2, "	%d	%d	%d	%d	%d	%d	%ld\n", tab_coordinates[0], tab_coordinates[1], x_to_load, y_to_load, z_to_load, current_gpu - 1, task->iterations[0]);
		}
		else
		{
			printf("Dans get data to load je ne gère que GEMM et CHOLESKY. Task %s is not supported.\n", starpu_task_get_name(task));
			exit(0);
		}
		fclose(f2);
	}
	return task;
#endif
}

/* Visu python.
 * Print in a file the effective order
 * (we do it from get_current_task because the ready heuristic
 * can change our planned order).
 * Also print in a file each task and it data to compute later the data needed
 * to load at each iteration.
 */
void _sched_visu_print_effective_order_in_file(struct starpu_task *task, int index_task)
{
	FILE *f = NULL;
	int tab_coordinates[2];

	int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	/* For the coordinates It write the coordinates (with Z for 3D), then the GPU and then the number of data needed to load for this task */
	//~ if (_print_n != 0 && (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") == 0))
	if (strcmp(_starpu_HFP_appli, "starpu_sgemm_gemm") == 0)
	{
		int size = strlen(_output_directory) + strlen("/Data_coordinates_order_last_SCHEDULER.txt") + 1;
		char path[size];
		snprintf(path, size, "%s%s", _output_directory, "/Data_coordinates_order_last_SCHEDULER.txt");
		if (index_task == 0)
		{
			f = fopen(path, "w");
		}
		else
		{
			f = fopen(path, "a");
		}
		/* Pour matrice 3D je récupère la coord de Z aussi */
		if (_print3d != 0)
		{
			/* 3 for 3D no ? */
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
			fprintf(f, "%d	%d", tab_coordinates[0], tab_coordinates[1]);

			/* TODO a suppr */
			//~ printf("Tâche n°%d %p : x = %d | y = %d | ", index_task, task, temp_tab_coordinates[0], temp_tab_coordinates[1]);

			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, tab_coordinates);
			fprintf(f, "	%d	%d\n", tab_coordinates[0], current_gpu - 1);

			/* TODO a suppr */
			//~ printf("z = %d\n", temp_tab_coordinates[0]);
		}
		else
		{
			starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, tab_coordinates);
			fprintf(f, "%d	%d	%d\n", tab_coordinates[0], tab_coordinates[1], current_gpu - 1);
		}
		fclose(f);
		//~ if (index_task == NT - 1)
		//~ {
			//~ if (_print3d == 0)
			//~ {
				//~ visualisation_tache_matrice_format_tex_with_data_2D();
			//~ }
		//~ }
	}
	else if (strcmp(starpu_task_get_name(task), "chol_model_11") == 0 || strcmp(starpu_task_get_name(task), "chol_model_21") == 0 || strcmp(starpu_task_get_name(task), "chol_model_22") == 0 || strcmp(starpu_task_get_name(task), "POTRF") == 0 || strcmp(starpu_task_get_name(task), "SYRK") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0 || strcmp(starpu_task_get_name(task), "GEMM") == 0)
	{
		int size = strlen(_output_directory) + strlen("/Data_coordinates_order_last_SCHEDULER.txt") + 1;
		char path[size];
		snprintf(path, size, "%s%s", _output_directory, "/Data_coordinates_order_last_SCHEDULER.txt");
		if (index_task == 0)
		{
			/* Ouverture du fichier. */
			f = fopen(path, "w");
			fprintf(f, "TASK COORDY COORDX GPU ITERATIONK\n");
		}
		else
		{
			f = fopen(path, "a");
		}

		/* Impression du type de tâche. */
		if (strcmp(starpu_task_get_name(task), "chol_model_11") == 0 || strcmp(starpu_task_get_name(task), "POTRF") == 0)
		{
			fprintf(f, "POTRF");
		}
		else if (strcmp(starpu_task_get_name(task), "chol_model_21") == 0 || strcmp(starpu_task_get_name(task), "TRSM") == 0)
		{
			fprintf(f, "TRSM");
		}
		else
		{
			/* Cas SYRK et GEMM que je distingue avec la donnée en double pour SYRK. */
			if (STARPU_TASK_GET_HANDLE(task, 0) == STARPU_TASK_GET_HANDLE(task, 1))
			{
				fprintf(f, "SYRK");
			}
			else
			{
				fprintf(f, "GEMM");
			}
		}

		/* La je n'imprime que les coords de la dernière donnée de la tâche car c'est ce qui me donne la place dans le triangle de Cholesky. */
		starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, STARPU_TASK_GET_NBUFFERS(task) - 1), 2, tab_coordinates);
		fprintf(f, "	%d	%d	%d	%ld\n", tab_coordinates[0], tab_coordinates[1], current_gpu - 1, task->iterations[0]);

		fclose(f);
	}
	else
	{
		printf("Dans print effective orer in file je ne gère que GEMM et CHOLESKY %s not suported.\n", starpu_task_get_name(task));
		exit(0);
	}
}

/* Printing each package and its content for visualisation */
void _sched_visu_print_packages_in_terminal(struct _starpu_HFP_paquets *a, int nb_of_loop, const char *msg)
{
	if (_print_in_terminal != 1) return;
	fprintf(stderr, "%s\n", msg);
	int link_index = 0;
	struct starpu_task *task;
	a->temp_pointer_1 = a->first_link;
	while (a->temp_pointer_1 != NULL)
	{
		link_index++; a->temp_pointer_1 = a->temp_pointer_1->next;
	}
	a->temp_pointer_1 = a->first_link;
	printf("-----\nOn a fais %d tour(s) de la boucle while et on a fais %d paquet(s)\n",nb_of_loop,link_index);
	printf("-----\n");
	link_index = 0;
	while (a->temp_pointer_1 != NULL)
	{
		printf("Le paquet %d contient %d tâche(s) et %d données, expected task time = %f, expected package time = %f, split last package = %d\n",link_index,a->temp_pointer_1->nb_task_in_sub_list, a->temp_pointer_1->package_nb_data,a->temp_pointer_1->expected_time, a->temp_pointer_1->expected_package_computation_time, a->temp_pointer_1->split_last_ij);
		for (task = starpu_task_list_begin(&a->temp_pointer_1->sub_list); task != starpu_task_list_end(&a->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
		{
			printf("%p : ",task);
			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				printf("%p ", STARPU_TASK_GET_HANDLE(task, i));
			}
			printf("\n");
		}
		link_index++;
		a->temp_pointer_1 = a->temp_pointer_1->next;
		printf("-----\n");
	}
	a->temp_pointer_1 = a->first_link;
}

/* Appellé par DARTS et HFP pour visu python. */
void _sched_visu_get_current_tasks(struct starpu_task *task, unsigned sci)
{
#ifdef PRINT_PYTHON
	if (index_task_currently_treated == 0)
	{
		_starpu_HFP_initialize_global_variable(task);
	}
	_sched_visu_print_effective_order_in_file(task, index_task_currently_treated);
	task_currently_treated = task;
	index_task_currently_treated++;
#endif
	starpu_sched_component_worker_pre_exec_hook(task, sci);
}

void _sched_visu_get_current_tasks_for_visualization(struct starpu_task *task, unsigned sci)
{
	(void)task;
	(void)sci;
#ifdef PRINT_PYTHON
	if (_index_current_task_for_visualization == 0)
	{
		_starpu_HFP_initialize_global_variable(task);
	}
	_sched_visu_print_effective_order_in_file(task, _index_current_task_for_visualization);
	task_currently_treated = task;
	index_task_currently_treated++;
	_index_current_task_for_visualization++;
#endif
}

void _sched_visu_print_matrix(int **matrix, int x, int y, char *msg)
{
	(void)matrix;
	(void)x;
	(void)y;
	(void)msg;
#ifdef PRINT
	_STARPU_SCHED_PRINT("%s", msg);
	int i;
	for (i = 0; i < x; i++)
	{
		int j;
		for (j = 0; j < y; j++)
		{
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
#endif
}

void _sched_visu_print_vector(int *vector, int x, char *msg)
{
	(void)vector;
	(void)x;
	(void)msg;
#ifdef PRINT
	_STARPU_SCHED_PRINT("%s", msg);
	int i;
	for (i = 0; i < x; i++)
	{
		printf("%d ", vector[i]);
	}
	printf("\n");
#endif
}

void _sched_visu_print_data_for_task(struct starpu_task *task, const char *msg)
{
	(void)task;
	(void)msg;
#ifdef PRINT
	_STARPU_SCHED_PRINT(msg, task);
	unsigned x;
	for (x = 0; x < STARPU_TASK_GET_NBUFFERS(task); x++)
	{
		printf("%p ", STARPU_TASK_GET_HANDLE(task, x));
	}
	printf("\n");
#endif
}
