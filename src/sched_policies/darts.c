/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Dynamic Data Aware reactive Task (DARTS) scheduling.
 * Look for the "best" data, i.e. the data that have the minimal transfer time to computation made available ratio.
 * Computes all task using this data and the data already loaded on memory.
 * If no task is available compute a task with highest priority.
 * DARTS works especially well with GPUs. With both CPUs and GPUs, it does not take into account the speed difference, which leads to poor results.
 * Using STARPU_NCPU=0 with STARPU_NOPENCL=0 is thus highly recommended to achieve peak performance when using GPUs.
 * Otherwise, DARTS can work with CPUs only and with both GPUs and CPUs.
 */

#include <common/config.h>
#include <sched_policies/sched_visu.h>
#include <sched_policies/darts.h>
#include <sched_policies/helper_mct.h>
#include <common/graph.h> /* To compute the descendants and consequently add priorities */
#include <datawizard/memory_nodes.h>

struct _starpu_darts_sched_data
{
	struct starpu_task_list main_task_list; /* List used to randomly pick a task. We use a second list because it's easier to randomize sched_list. */
	struct starpu_task_list sched_list;
	starpu_pthread_mutex_t policy_mutex;
};

struct _starpu_darts_pointer_in_task
{
	/* Pointer to the data used by the current task */
	starpu_data_handle_t *pointer_to_D;
	struct _starpu_darts_task_using_data **tud;
	struct starpu_task *pointer_to_cell; /* Pointer to the cell in the main task list */
};

/** Planned task. One planned task = one processing unit. **/
struct _starpu_darts_gpu_planned_task
{
	struct starpu_task_list planned_task;

	struct starpu_task_list refused_fifo_list; /* If a task is refused, it goes in this fifo list so it can be the next task processed by the right gpu */

	void *gpu_data; /* Data not loaded yet. */
	void *new_gpu_data; /* Data not loaded yet that are from the new tasks. This is used only with STARPU_DARTS_DATA_ORDER=1 that randomize the new data and put them at the end of the list. */

	starpu_data_handle_t data_to_evict_next; /* If an eviction fails, it allows to evict it next. */

	bool first_task; /* If it's the first task of a GPU, we can directly return it and not look for the "best" data. */

	int number_data_selection;

	struct starpu_task *first_task_to_pop; /* First task to return if the task order is not randomized, i.e. STARPU_DARTS_TASK_ORDER == 2, which is the default case. */
};

/* Struct dans user_data des handles pour reset MAIS aussi pour savoir le nombre de tâches dans pulled task qui utilise cette donnée */
struct _starpu_darts_handle_user_data
{
	int last_iteration_DARTS;
	int* nb_task_in_pulled_task;
	int* nb_task_in_planned_task;
	int* last_check_to_choose_from; /* To clarify the last time I looked at this data, so as not to look at it twice in choose best data from 1 in the same iteration of searching for the best data. */
	int* is_present_in_data_not_used_yet; /* Array of the number of GPUs used in push_task to find out whether data is present in a GPU's datanotusedyet. Updated when data is used and removed from the list, and when data is pushed. Provides a quick indication of whether data should be added or not. */
	double sum_remaining_task_expected_length; /* Sum of expected job durations using this data. Used to tie break. Initialized in push task, decreased when adding a task in planned task and increased when removing a task from planned task after an eviction. */
};

/** Task out of pulled task. Updated by post_exec. I'm forced to use a list of single task and not task list because else starpu doesn't allow me to push a tasks in two different task_list **/
LIST_TYPE(_starpu_darts_pulled_task,
	  struct starpu_task *pointer_to_pulled_task;
);

struct _starpu_darts_gpu_pulled_task
{
	int test;
	struct _starpu_darts_pulled_task_list *ptl;
};

/** In the "packages" of dynamic data aware, each representing a gpu **/
LIST_TYPE(_starpu_darts_gpu_data_not_used,
	  starpu_data_handle_t D; /* The data not used yet by the GPU. */
);

/** In the handles **/
LIST_TYPE(_starpu_darts_task_using_data,
	  /* Pointer to the main task list T */
	  struct starpu_task *pointer_to_T;
);

static starpu_data_handle_t *Dopt;
static bool *data_conflict;

/** Mutex **/
#ifdef STARPU_DARTS_LINEAR_MUTEX
static starpu_pthread_mutex_t linear_mutex; /* Mutex that make almost everything linear. Used in the IPDPS version of this algorithm and also to ease debugs. Not used in the default case. */
#define _LINEAR_MUTEX_LOCK()   STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex)
#define _LINEAR_MUTEX_UNLOCK() STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex)
#define _LINEAR_MUTEX_INIT()   STARPU_PTHREAD_MUTEX_INIT(&linear_mutex, NULL)

#define _REFINED_MUTEX_LOCK()
#define _REFINED_MUTEX_UNLOCK()
#define _REFINED_MUTEX_INIT()

#else

static starpu_pthread_mutex_t refined_mutex; /* Protect the main task list and the data. This is the mutex used by default */
#define _REFINED_MUTEX_LOCK()   STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex)
#define _REFINED_MUTEX_UNLOCK() STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex)
#define _REFINED_MUTEX_INIT()   STARPU_PTHREAD_MUTEX_INIT(&refined_mutex, NULL)

#define _LINEAR_MUTEX_LOCK()
#define _LINEAR_MUTEX_UNLOCK()
#define _LINEAR_MUTEX_INIT()
#endif


static int can_a_data_be_in_mem_and_in_not_used_yet;
static int eviction_strategy_darts;
static int threshold;
static int app;
static int choose_best_data_from;
static int simulate_memory;
static int task_order;
static int data_order;
static int prio;
static int free_pushed_task_position;
static int dependances;
static int graph_descendants;
static int dopt_selection_order;
static int highest_priority_task_returned_in_default_case;
static int push_free_task_on_gpu_with_least_task_in_planned_task;
static int round_robin_free_task;
static int cpu_only;
static int _nb_gpus;

static bool new_tasks_initialized;
static struct _starpu_darts_gpu_planned_task *tab_gpu_planned_task;
static struct _starpu_darts_gpu_pulled_task *tab_gpu_pulled_task;
static int NT_DARTS;
static int iteration_DARTS;
static struct starpu_perfmodel_arch *perf_arch;
static int *memory_nodes;
static char *_output_directory;

#ifdef STARPU_DARTS_STATS
static int nb_return_null_after_scheduling;
static int nb_return_task_after_scheduling;
static int nb_return_null_because_main_task_list_empty;
static int nb_new_task_initialized;
static int nb_refused_task;
static int victim_selector_refused_not_on_node;
static int victim_selector_refused_cant_evict;
static int victim_selector_return_refused;
static int victim_selector_return_unvalid;
static int victim_selector_return_data_not_in_planned_and_pulled;
static int number_data_conflict;
static int number_critical_data_conflict;
static int victim_evicted_compteur;
static int victim_selector_compteur;
static int victim_selector_return_no_victim;
static int victim_selector_belady;
static int nb_1_from_free_task_not_found;
static int number_random_selection;
static int nb_free_choice;
static int nb_1_from_free_choice;
static int nb_data_selection_per_index;
static int nb_task_added_in_planned_task;
static bool data_choice_per_index;
static int nb_data_selection;

static long long time_total_selector;
static long long time_total_evicted;
static long long time_total_belady;
static long long time_total_schedule;
static long long time_total_choose_best_data;
static long long time_total_fill_planned_task_list;
static long long time_total_initialisation;
static long long time_total_randomize;
static long long time_total_pick_random_task;
static long long time_total_least_used_data_planned_task;
static long long time_total_createtolasttaskfinished;
struct timeval time_start_createtolasttaskfinished;
#endif

/** If a data is a redux or a scratch it's only used to optimize a computation and
 * does not contain any valuable information. Thus we ignore it. **/
#define STARPU_IGNORE_UTILITIES_HANDLES(task, index) if ((STARPU_TASK_GET_MODE(task, index) & STARPU_SCRATCH) || (STARPU_TASK_GET_MODE(task, index) & STARPU_REDUX)) { continue; }
#define STARPU_IGNORE_UTILITIES_HANDLES_FROM_DATA(handle) if ((handle->current_mode == STARPU_SCRATCH) || (handle->current_mode == STARPU_REDUX)) { continue; }

static int _get_number_GPU()
{
	int return_value = starpu_memory_nodes_get_count_by_kind(STARPU_CUDA_RAM);

	if (return_value == 0) /* We are not using GPUs so we are in an out-of-core case using CPUs. Need to return 1. If I want to deal with GPUs AND CPUs we need to adpt this function to return NGPU + 1 */
	{
		return 1;
	}

	return return_value;
}

/* Return the number of handle used by a task without considering the scratch data used by a cusolver. */
static int get_nbuffer_without_scratch(struct starpu_task *t)
{
	int count = 0;
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(t); i++)
	{
		if ((STARPU_TASK_GET_MODE(t, i) & STARPU_SCRATCH) || (STARPU_TASK_GET_MODE(t, i) & STARPU_REDUX))
		{
			continue;
		}
		else
		{
			count += 1;
		}
	}
	return count;
}

/* Set priority to the tasks depending on the progression on the task graph. Used when STARPU_DARTS_GRAPH_DESCENDANTS is set to 1 or 2. by default STARPU_DARTS_GRAPH_DESCENDANTS is set to 0. */
static void set_priority(void *_data, struct _starpu_graph_node *node)
{
	(void)_data;
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	starpu_worker_relax_off();
	struct _starpu_job *job = node->job;

	if (job)
	{
		job->task->priority = node->descendants;

		_STARPU_SCHED_PRINT("Descendants of job %p (%s): %d\n", job->task, starpu_task_get_name(job->task), job->task->priority);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}

void _starpu_darts_tab_gpu_planned_task_init()
{
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		starpu_task_list_init(&tab_gpu_planned_task[i].planned_task);
		starpu_task_list_init(&tab_gpu_planned_task[i].refused_fifo_list);
		tab_gpu_planned_task[i].data_to_evict_next = NULL;
		tab_gpu_planned_task[i].first_task = true;
		tab_gpu_planned_task[i].number_data_selection = 0;

		tab_gpu_planned_task[i].gpu_data = _starpu_darts_gpu_data_not_used_list_new();
		_starpu_darts_gpu_data_not_used_list_init(tab_gpu_planned_task[i].gpu_data);

		if (data_order == 1)
		{
			tab_gpu_planned_task[i].new_gpu_data = _starpu_darts_gpu_data_not_used_list_new();
			_starpu_darts_gpu_data_not_used_list_init(tab_gpu_planned_task[i].new_gpu_data);
		}

		tab_gpu_planned_task[i].first_task_to_pop = NULL;
	}
}

void _starpu_darts_tab_gpu_pulled_task_init()
{
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		struct _starpu_darts_pulled_task_list *p = _starpu_darts_pulled_task_list_new();
		tab_gpu_pulled_task[i].ptl = p;
		tab_gpu_pulled_task[i].test = 0;
	}
}

/* Function called directly in the applications of starpu to reinit the struct of darts. Used when multiple iteration of a same application are lauched in the same execution. */
void starpu_darts_reinitialize_structures()
{
	_REFINED_MUTEX_LOCK();
	_LINEAR_MUTEX_LOCK();

	/* Printing stats in files. Préciser PRINT_N dans les var d'env. */
	_STARPU_SCHED_PRINT("############### Itération n°%d ###############\n", iteration_DARTS + 1);

#ifdef STARPU_DARTS_STATS
	printf("Nb \"random\" task selection: %d\n", number_random_selection);
	printf("Nb \"index\" data selection: %d/%d\n", nb_data_selection_per_index, nb_data_selection);
	if (iteration_DARTS == 11 || starpu_get_env_number_default("STARPU_SCHED_PRINT_TIME", 0) == 2) /* PRINT_TIME = 2 pour quand on a 1 seule itération. */
	{
		{
			int size = strlen(_output_directory) + strlen("/Data_DARTS_Nb_conflit_donnee.csv") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_DARTS_Nb_conflit_donnee.csv");
			FILE *f_new_iteration = fopen(path, "a");
			STARPU_ASSERT_MSG(f_new_iteration, "cannot open file <%s>\n", path);
			fprintf(f_new_iteration , "%d,%d,%d\n", _print_n, number_data_conflict/11 + number_data_conflict%11, number_critical_data_conflict/11 + number_critical_data_conflict%11);
			fclose(f_new_iteration);
		}

		struct timeval time_end_createtolasttaskfinished;
		gettimeofday(&time_end_createtolasttaskfinished, NULL);
		time_total_createtolasttaskfinished += (time_end_createtolasttaskfinished.tv_sec - time_start_createtolasttaskfinished.tv_sec)*1000000LL + time_end_createtolasttaskfinished.tv_usec - time_start_createtolasttaskfinished.tv_usec;

		{
			int size = strlen(_output_directory) + strlen("/Data_DARTS_time.csv") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_DARTS_time.csv");
			FILE *f_new_iteration = fopen(path, "a");
			STARPU_ASSERT_MSG(f_new_iteration, "cannot open file <%s>\n", path);
			fprintf(f_new_iteration, "%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", _print_n, time_total_selector/11 + time_total_selector%11, time_total_evicted/11 + time_total_evicted%11, time_total_belady/11 + time_total_belady%11, time_total_schedule/11 + time_total_schedule%11, time_total_choose_best_data/11 + time_total_choose_best_data%11, time_total_fill_planned_task_list/11 + time_total_fill_planned_task_list%11, time_total_initialisation/11 + time_total_initialisation%11, time_total_randomize/11 + time_total_randomize%11, time_total_pick_random_task/11 + time_total_pick_random_task%11, time_total_least_used_data_planned_task/11 + time_total_least_used_data_planned_task%11, time_total_createtolasttaskfinished/11 + time_total_createtolasttaskfinished%11);
			fclose(f_new_iteration);
		}

		{
			int size = strlen(_output_directory) + strlen("/Data_DARTS_Choice_during_scheduling.csv") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_DARTS_Choice_during_scheduling.csv");
			FILE *f_new_iteration = fopen(path, "a");
			STARPU_ASSERT_MSG(f_new_iteration, "cannot open file <%s>\n", path);
			fprintf(f_new_iteration, "%d,%d,%d,%d,%d,%d,%d,%d\n", _print_n, nb_return_null_after_scheduling/11 + nb_return_null_after_scheduling%11, nb_return_task_after_scheduling/11 + nb_return_task_after_scheduling%11, nb_return_null_because_main_task_list_empty/11 + nb_return_null_because_main_task_list_empty%11, number_random_selection/11 + number_random_selection%11, nb_1_from_free_task_not_found/11 + nb_1_from_free_task_not_found%11, nb_free_choice/11 + nb_free_choice%11, nb_1_from_free_choice/11 + nb_1_from_free_choice%11);
			fclose(f_new_iteration);
		}

		{
			int size = strlen(_output_directory) + strlen("/Data_DARTS_Choice_victim_selector.csv") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_DARTS_Choice_victim_selector.csv");
			FILE *f_new_iteration = fopen(path, "a");
			STARPU_ASSERT_MSG(f_new_iteration, "cannot open file <%s>\n", path);
			fprintf(f_new_iteration, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", _print_n, victim_selector_refused_not_on_node/11 + victim_selector_refused_not_on_node%11, victim_selector_refused_cant_evict/11 + victim_selector_refused_cant_evict%11, victim_selector_return_refused/11 + victim_selector_return_refused%11, victim_selector_return_unvalid/11 + victim_selector_return_unvalid%11, victim_selector_return_data_not_in_planned_and_pulled/11 + victim_selector_return_data_not_in_planned_and_pulled%11, victim_evicted_compteur/11 + victim_evicted_compteur%11, victim_selector_compteur/11 + victim_selector_compteur%11, victim_selector_return_no_victim/11 + victim_selector_return_no_victim%11, victim_selector_belady/11 + victim_selector_belady%11);
			fclose(f_new_iteration);
		}

		{
			int size = strlen(_output_directory) + strlen("/Data_DARTS_DARTS_Misc.csv") + 1;
			char path[size];
			snprintf(path, size, "%s%s", _output_directory, "/Data_DARTS_DARTS_Misc.csv");
			FILE *f_new_iteration = fopen(path, "a");
			STARPU_ASSERT_MSG(f_new_iteration, "cannot open file <%s>\n", path);
			fprintf(f_new_iteration, "%d,%d,%d\n", _print_n, nb_refused_task/11 + nb_refused_task%11, nb_new_task_initialized/11 + nb_new_task_initialized%11);
			fclose(f_new_iteration);
		}
	}
#endif

	/* Re-init for the next iteration of the application */
	free(tab_gpu_planned_task);
	iteration_DARTS++; /* Used to know if a data must be added again in the list of data of each planned task. */
	tab_gpu_planned_task = calloc(_nb_gpus, sizeof(struct _starpu_darts_gpu_planned_task));
	_starpu_darts_tab_gpu_planned_task_init();

	_REFINED_MUTEX_UNLOCK();
	_LINEAR_MUTEX_UNLOCK();
}

static void print_task_info(struct starpu_task *task)
{
	(void)task;
#ifdef PRINT
	printf("Task %p has %d data:", task, get_nbuffer_without_scratch(task));

	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		printf(" %p", STARPU_TASK_GET_HANDLE(task, i));
	}
	printf("\n");
#endif
}

static void print_task_list(struct starpu_task_list *l, char *s)
{
	(void)l; (void)s;
#ifdef PRINT
	printf("%s :\n", s); fflush(stdout);
	struct starpu_task *task;
	for (task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
	{
		printf("%p (prio: %d):", task, task->priority); fflush(stdout);
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			STARPU_IGNORE_UTILITIES_HANDLES(task, i);
			printf("	%p", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
		}
		printf("\n"); fflush(stdout);
	}
#endif
}

static void print_data_not_used_yet()
{
#ifdef PRINT
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		printf("On GPU %d, there are %d data not used yet:", i, _starpu_darts_gpu_data_not_used_list_size(tab_gpu_planned_task[i].gpu_data)); fflush(stdout);
		struct _starpu_darts_gpu_data_not_used *e;
		for (e = _starpu_darts_gpu_data_not_used_list_begin(tab_gpu_planned_task[i].gpu_data);
		     e != _starpu_darts_gpu_data_not_used_list_end(tab_gpu_planned_task[i].gpu_data);
		     e = _starpu_darts_gpu_data_not_used_list_next(e))
		{
			printf(" %p", e->D); fflush(stdout);
		}
		printf("\n"); fflush(stdout);
	}
	printf("\n"); fflush(stdout);
#endif
}

static void print_pulled_task_one_gpu(struct _starpu_darts_gpu_pulled_task *g, int current_gpu)
{
	(void)g; (void)current_gpu;
#ifdef PRINT
	printf("Pulled task for GPU %d:\n", current_gpu); fflush(stdout);
	struct _starpu_darts_pulled_task *p;
	for (p = _starpu_darts_pulled_task_list_begin(tab_gpu_pulled_task[current_gpu].ptl); p != _starpu_darts_pulled_task_list_end(tab_gpu_pulled_task[current_gpu].ptl); p = _starpu_darts_pulled_task_list_next(p))
	{
		printf("%p\n", p->pointer_to_pulled_task); fflush(stdout);
	}
#endif
}

static void print_data_not_used_yet_one_gpu(struct _starpu_darts_gpu_planned_task *g, int current_gpu)
{
	(void)g; (void)current_gpu;
#ifdef PRINT
	printf("Data not used yet on GPU %d are:\n", current_gpu); fflush(stdout);
	if (g->gpu_data != NULL)
	{
		struct _starpu_darts_gpu_data_not_used *e;
		for (e = _starpu_darts_gpu_data_not_used_list_begin(tab_gpu_planned_task[current_gpu].gpu_data);
		     e != _starpu_darts_gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data);
		     e = _starpu_darts_gpu_data_not_used_list_next(e))
		{
			printf(" %p", e->D); fflush(stdout);
		}
	}
	printf("\n"); fflush(stdout);
#endif
}

#ifdef PRINT
static void check_double_in_data_not_used_yet(struct _starpu_darts_gpu_planned_task *g, int current_gpu)
{
	(void)g;
	printf("Careful you are using check_double_in_data_not_used_yet it cost time!\n"); fflush(stdout);
	struct _starpu_darts_gpu_data_not_used *e1;
	for (e1 = _starpu_darts_gpu_data_not_used_list_begin(tab_gpu_planned_task[current_gpu].gpu_data);
	     e1 != _starpu_darts_gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data);
	     e1 = _starpu_darts_gpu_data_not_used_list_next(e1))
	{
		struct _starpu_darts_gpu_data_not_used *e2;
		for (e2 = _starpu_darts_gpu_data_not_used_list_next(e1);
		     e2 != _starpu_darts_gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data);
		     e2 = _starpu_darts_gpu_data_not_used_list_next(e2))
		{
			if (e1->D == e2->D)
			{
				printf("Data %p is in double on GPU %d!\n", e1->D, current_gpu); fflush(stdout);
				print_data_not_used_yet_one_gpu(&tab_gpu_planned_task[current_gpu], current_gpu); fflush(stdout);
				exit(1);
			}
		}
	}
}
#endif

/* Looking if the task can freely be computed by looking at the memory and the data associated from free task in planned task */
static bool is_my_task_free(int current_gpu, struct starpu_task *task)
{
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			return false;
		}
		struct _starpu_darts_handle_user_data *hud;
		hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), memory_nodes[current_gpu]) && hud->nb_task_in_planned_task[current_gpu] == 0)
		//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), memory_nodes[current_gpu]) && hud->nb_task_in_planned_task[current_gpu] == 0 && hud->nb_task_in_pulled_task[current_gpu] == 0)
		{
			return false;
		}
	}
	return true;
}

/* Initialize for:
 * tasks -> pointer to the data it uses, pointer to the pointer of task list in the data,
 * pointer to the cell in the main task list (main_task_list).
 * data -> pointer to the tasks using this data.
 * GPUs -> data not used yet by this GPU.
 * In the case with dependencies I have to check if a data need to be added again even if it's struct is empty.
 */
/* Version when you don't have dependencies */
static void initialize_task_data_gpu_single_task_no_dependencies(struct starpu_task *task, int also_add_data_in_not_used_yet_list)
{
	if (also_add_data_in_not_used_yet_list == 1)
	{
		/* Adding the data not used yet in all the GPU(s). */
		int i;
		for (i = 0; i < _nb_gpus; i++)
		{
			unsigned j;
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(task, j);
				struct _starpu_darts_gpu_data_not_used *e = _starpu_darts_gpu_data_not_used_new();
				e->D = STARPU_TASK_GET_HANDLE(task, j);

				/* If the data already has an existing structure */
				if (STARPU_TASK_GET_HANDLE(task, j)->user_data != NULL)
				{
					struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, j)->user_data;

					if (hud->last_iteration_DARTS != iteration_DARTS || hud->is_present_in_data_not_used_yet[i] == 0) /* It is a new iteration of the same application, so the data must be re-initialized. */
					{
						if (data_order == 1)
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].new_gpu_data, e);
						}
						else
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, e);
						}
					}
				}
				else
				{
					if (data_order == 1)
					{
						_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].new_gpu_data, e);
					}
					else
					{
						_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, e);
					}
				}
			}
		}
	}

	/* Adding the pointer in the task. */
	struct _starpu_darts_pointer_in_task *pt = malloc(sizeof(*pt));
	pt->pointer_to_cell = task;
	pt->pointer_to_D = malloc(get_nbuffer_without_scratch(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
	pt->tud = malloc(get_nbuffer_without_scratch(task)*sizeof(_starpu_darts_task_using_data_new()));

	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		/* Pointer toward the main task list in the handles. */
		struct _starpu_darts_task_using_data *e = _starpu_darts_task_using_data_new();
		e->pointer_to_T = task;

		/* Adding the task in the list of task using the data */
		if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
		{
			struct _starpu_darts_task_using_data_list *tl = _starpu_darts_task_using_data_list_new();
			_starpu_darts_task_using_data_list_push_front(tl, e);
			STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
		}
		else
		{
			_starpu_darts_task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
		}

		/* Init hud in the data containing a way to track the number of task in
		 * planned and pulled_task but also a way to check last iteration_DARTS for this data and last check for CHOOSE_FROM_MEM=1
		 * so we don't look twice at the same data. */
		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			struct _starpu_darts_handle_user_data *hud = malloc(sizeof(*hud));
			hud->last_iteration_DARTS = iteration_DARTS;

			/* Need to init them with the number of GPU */
			hud->nb_task_in_pulled_task = malloc(_nb_gpus*sizeof(int));
			hud->nb_task_in_planned_task = malloc(_nb_gpus*sizeof(int));
			hud->last_check_to_choose_from = malloc(_nb_gpus*sizeof(int));
			hud->is_present_in_data_not_used_yet = malloc(_nb_gpus*sizeof(int));
			hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);

			int j;
			for (j = 0; j < _nb_gpus; j++)
			{
				hud->nb_task_in_pulled_task[j] = 0;
				hud->nb_task_in_planned_task[j] = 0;
				hud->last_check_to_choose_from[j] = 0;
				hud->is_present_in_data_not_used_yet[j] = 1;
			}

			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		else
		{
			struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);
			if (hud->last_iteration_DARTS != iteration_DARTS || hud->is_present_in_data_not_used_yet[i] == 0) /* Re-init values in hud. */
			{
				int j;
				for (j = 0; j < _nb_gpus; j++)
				{
					hud->nb_task_in_pulled_task[j] = 0;
					hud->nb_task_in_planned_task[j] = 0;
					hud->last_check_to_choose_from[j] = 0;
					hud->is_present_in_data_not_used_yet[j] = 1;
				}
				hud->last_iteration_DARTS = iteration_DARTS;
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
		}

		/* Adding the pointer in the task toward the data. */
		pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
		pt->tud[i] = e;
	}
	task->sched_data = pt;
}

/* V3 used for dependencies */
static void initialize_task_data_gpu_single_task_dependencies(struct starpu_task *task, int also_add_data_in_not_used_yet_list)
{
	unsigned i;

	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);

		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			struct _starpu_darts_handle_user_data *hud = malloc(sizeof(*hud));
			hud->last_iteration_DARTS = iteration_DARTS;
			hud->nb_task_in_pulled_task = malloc(_nb_gpus*sizeof(int));
			hud->nb_task_in_planned_task = malloc(_nb_gpus*sizeof(int));
			hud->last_check_to_choose_from = malloc(_nb_gpus*sizeof(int));
			hud->is_present_in_data_not_used_yet = malloc(_nb_gpus*sizeof(int));
			hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);

			_STARPU_SCHED_PRINT("Data is new. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, i), hud->sum_remaining_task_expected_length);

			int j;
			for (j = 0; j < _nb_gpus; j++)
			{
				struct _starpu_darts_gpu_data_not_used *e = _starpu_darts_gpu_data_not_used_new();
				e->D = STARPU_TASK_GET_HANDLE(task, i);

				hud->nb_task_in_pulled_task[j] = 0;
				hud->nb_task_in_planned_task[j] = 0;
				hud->last_check_to_choose_from[j] = 0;
				hud->is_present_in_data_not_used_yet[j] = 0;

				if (also_add_data_in_not_used_yet_list == 1 && (can_a_data_be_in_mem_and_in_not_used_yet == 1 || !starpu_data_is_on_node(e->D, memory_nodes[j])))
				{
					hud->is_present_in_data_not_used_yet[j] = 1;
					if (data_order == 1)
					{
						_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
					}
					else
					{
						_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
					}
				}
			}
			_STARPU_SCHED_PRINT("%p gets 1 at is_present_in_data_not_used_yet from NULL struct hud\n", STARPU_TASK_GET_HANDLE(task, i));
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		else
		{
			struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			_STARPU_SCHED_PRINT("New task. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, i), hud->sum_remaining_task_expected_length);

			if (hud->last_iteration_DARTS != iteration_DARTS)
			{
				hud->last_iteration_DARTS = iteration_DARTS;
				hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);

				int j;
				for (j = 0; j < _nb_gpus; j++)
				{
					struct _starpu_darts_gpu_data_not_used *e = _starpu_darts_gpu_data_not_used_new();
					e->D = STARPU_TASK_GET_HANDLE(task, i);

					hud->nb_task_in_pulled_task[j] = 0;
					hud->nb_task_in_planned_task[j] = 0;
					hud->last_check_to_choose_from[j] = 0;
					hud->is_present_in_data_not_used_yet[j] = 0;

					if (also_add_data_in_not_used_yet_list == 1 && (can_a_data_be_in_mem_and_in_not_used_yet == 1 || !starpu_data_is_on_node(e->D, memory_nodes[j])))
					{
						hud->is_present_in_data_not_used_yet[j] = 1;
						if (data_order == 1)
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
						}
						else
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
						}
						print_data_not_used_yet();
					}
				}
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
			else
			{
				hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);

				int j;
				for (j = 0; j < _nb_gpus; j++)
				{
					struct _starpu_darts_gpu_data_not_used *e = _starpu_darts_gpu_data_not_used_new();
					e->D = STARPU_TASK_GET_HANDLE(task, i);

					if (hud->is_present_in_data_not_used_yet[j] == 0 && also_add_data_in_not_used_yet_list == 1 && (can_a_data_be_in_mem_and_in_not_used_yet == 1 || !starpu_data_is_on_node(e->D, memory_nodes[j])))
					{
						_STARPU_SCHED_PRINT("%p gets 1 at is_present_in_data_not_used_yet on GPU %d\n", STARPU_TASK_GET_HANDLE(task, i), j);
						hud->is_present_in_data_not_used_yet[j] = 1;
						if (data_order == 1)
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
						}
						else
						{
							_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
						}
					}
				}
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
		}
	}

	/* Adding the pointer in the task. */
	struct _starpu_darts_pointer_in_task *pt = malloc(sizeof(*pt));
	pt->pointer_to_cell = task;
	pt->pointer_to_D = malloc(get_nbuffer_without_scratch(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
	pt->tud = malloc(get_nbuffer_without_scratch(task)*sizeof(_starpu_darts_task_using_data_new()));

	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		/* Pointer toward the main task list in the handles. */
		struct _starpu_darts_task_using_data *e = _starpu_darts_task_using_data_new();
		e->pointer_to_T = task;

		/* Adding the task in the list of task using the data */
		if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
		{
			struct _starpu_darts_task_using_data_list *tl = _starpu_darts_task_using_data_list_new();
			_starpu_darts_task_using_data_list_push_front(tl, e);
			STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
		}
		else
		{
			_starpu_darts_task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
		}

		//printf("Adding in tab at position %d out of %d\n", i, get_nbuffer_without_scratch(task) - 1); fflush(stdout);

		/* Adding the pointer in the task toward the data. */
		pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
		pt->tud[i] = e;
	}
	task->sched_data = pt;
}

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
static void merge_tab_of_int(int arr[], int l, int m, int r, int tab_of_int[])
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	/* create temp arrays */
	int L[n1], R[n2];
	int L_task_tab[n1];
	int R_task_tab[n2];

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < n1; i++)
	{
		L[i] = arr[l + i];
		L_task_tab[i] = tab_of_int[l + i];
	}
	for (j = 0; j < n2; j++)
	{
		R[j] = arr[m + 1 + j];
		R_task_tab[j] = tab_of_int[m + 1 + j];
	}

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0; // Initial index of first subarray
	j = 0; // Initial index of second subarray
	k = l; // Initial index of merged subarray
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			tab_of_int[k] = L_task_tab[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			tab_of_int[k] =  R_task_tab[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of L[], if there
	   are any */
	while (i < n1)
	{
		arr[k] = L[i];
		tab_of_int[k] =  L_task_tab[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there
	   are any */
	while (j < n2)
	{
		arr[k] = R[j];
		tab_of_int[k] =  R_task_tab[j];
		j++;
		k++;
	}
}

/* l is for left index and r is right index of the
sub-array of arr to be sorted */
static void mergeSort_tab_of_int(int *arr, int l, int r, int *tab_of_int)
{
	if (l < r)
	{
		// Same as (l+r)/2, but avoids overflow for
		// large l and h
		int m = l + (r - l) / 2;

		// Sort first and second halves
		mergeSort_tab_of_int(arr, l, m, tab_of_int);
		mergeSort_tab_of_int(arr, m + 1, r, tab_of_int);

		merge_tab_of_int(arr, l, m, r, tab_of_int);
	}
}

static void _starpu_darts_increment_planned_task_data(struct starpu_task *task, int current_gpu)
{
	/* Careful, we do not want duplicates here */
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		hud->nb_task_in_planned_task[current_gpu] = hud->nb_task_in_planned_task[current_gpu] + 1;
		STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
	}
}

/* Pushing the tasks. Each time a new task enter here, we initialize it. */
static int darts_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
#ifdef PRINT
	printf("New task %p (%s, prio: %d, length: %f) in push_task with data(s):", task, starpu_task_get_name(task), task->priority, starpu_task_expected_length(task, perf_arch, 0)); fflush(stdout);
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_SCRATCH)
		{
			printf(" %p mode is STARPU_SCRATCH\n", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
		}
		else if (STARPU_TASK_GET_MODE(task, i) & STARPU_REDUX)
		{
			printf(" %p mode is STARPU_REDUX\n", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
		}
		else
		{
			printf(" %p mode is R-RW-W", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
		}
	}
	printf("\n"); fflush(stdout);
#endif

	_REFINED_MUTEX_LOCK();
	_LINEAR_MUTEX_LOCK();

#ifdef STARPU_DARTS_STATS
	struct timeval time_start_initialisation;
	gettimeofday(&time_start_initialisation, NULL);
#endif

#ifdef PRINT
	int x;
	for (x = 0; x < _nb_gpus; x++)
	{
		check_double_in_data_not_used_yet(&tab_gpu_planned_task[x], x);
	}
#endif

	/* If push_free_task_on_gpu_with_least_task_in_planned_task is not set to 1, these two variables are not useful */
	int *sorted_gpu_list_by_nb_task_in_planned_task = NULL;
	int *planned_task_sizes = NULL;

	/* Pushing free task directly in a gpu's planned task. */
	if (push_free_task_on_gpu_with_least_task_in_planned_task == 1) /* Getting the gpu with the least tasks in planned task */
	{
		sorted_gpu_list_by_nb_task_in_planned_task = malloc(_nb_gpus*sizeof(int));
		planned_task_sizes = malloc(_nb_gpus*sizeof(int));
		int j;
		for (j = 0; j < _nb_gpus; j++)
		{
			sorted_gpu_list_by_nb_task_in_planned_task[j] = j;
			planned_task_sizes[j] = starpu_task_list_size(&tab_gpu_planned_task[j].planned_task);
		}
		mergeSort_tab_of_int(planned_task_sizes, 0, _nb_gpus - 1, sorted_gpu_list_by_nb_task_in_planned_task);
	}

	if (push_free_task_on_gpu_with_least_task_in_planned_task == 2)
	{
		round_robin_free_task++;
	}

	int j;
	for (j = 0; j < _nb_gpus; j++)
	{
		int gpu_looked_at;
		if (push_free_task_on_gpu_with_least_task_in_planned_task == 1)
		{
			gpu_looked_at = sorted_gpu_list_by_nb_task_in_planned_task[j];
		}
		else if (push_free_task_on_gpu_with_least_task_in_planned_task == 2)
		{
			gpu_looked_at = (j + round_robin_free_task)%_nb_gpus;
		}
		else
		{
			gpu_looked_at = j;
		}

		_STARPU_SCHED_PRINT("gpu_looked_at = %d\n", gpu_looked_at);

		if (is_my_task_free(gpu_looked_at, task))
		{
			_STARPU_SCHED_PRINT("Task %p is free from push_task\n", task);
			if (dependances == 1)
			{
				initialize_task_data_gpu_single_task_dependencies(task, 0);
			}
			else
			{
				initialize_task_data_gpu_single_task_no_dependencies(task, 0);
			}

			_starpu_darts_increment_planned_task_data(task, gpu_looked_at);
			struct _starpu_darts_pointer_in_task *pt = task->sched_data;
			unsigned y;
			for (y = 0; y < STARPU_TASK_GET_NBUFFERS(task); y++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(task, y);
				if (pt->tud[y] != NULL)
				{
					_starpu_darts_task_using_data_list_erase(pt->pointer_to_D[y]->sched_data, pt->tud[y]);
					pt->tud[y] = NULL;
				}
			}

			_STARPU_SCHED_PRINT("Free task from push %p is put in planned task\n", task);

			/* Now push this free task into planned task. This can be done at the beginning of the list or after the last free task in planned task. */
			if (free_pushed_task_position == 0)
			{
				starpu_task_list_push_front(&tab_gpu_planned_task[gpu_looked_at].planned_task, task);
			}
			else
			{
				/* Après la dernière tâche gratuite de planned task. */
				struct starpu_task *checked_task;
				for (checked_task = starpu_task_list_begin(&tab_gpu_planned_task[gpu_looked_at].planned_task); checked_task != starpu_task_list_end(&tab_gpu_planned_task[gpu_looked_at].planned_task); checked_task = starpu_task_list_next(checked_task))
				{
					for (y = 0; y < STARPU_TASK_GET_NBUFFERS(checked_task); y++)
					{
						STARPU_IGNORE_UTILITIES_HANDLES(checked_task, y);
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(checked_task, y), memory_nodes[gpu_looked_at]))
						{
							starpu_task_list_insert_before(&tab_gpu_planned_task[gpu_looked_at].planned_task, task, checked_task);

							if (push_free_task_on_gpu_with_least_task_in_planned_task == 1)
							{
								free(sorted_gpu_list_by_nb_task_in_planned_task);
								free(planned_task_sizes);
							}

							/* End now push task, no push in main task list or GPUs data */
							starpu_push_task_end(task);
							_REFINED_MUTEX_UNLOCK();
							_LINEAR_MUTEX_UNLOCK();
							component->can_pull(component);
							return 0;
						}
					}
				}

				/* Else push back */
				starpu_task_list_push_back(&tab_gpu_planned_task[gpu_looked_at].planned_task, task);
			}

			if (push_free_task_on_gpu_with_least_task_in_planned_task == 1)
			{
				free(sorted_gpu_list_by_nb_task_in_planned_task);
				free(planned_task_sizes);
			}

			/* End now push task, no push in main task list or GPUs data */
			starpu_push_task_end(task);
			_REFINED_MUTEX_UNLOCK();
			_LINEAR_MUTEX_UNLOCK();
			component->can_pull(component);
			return 0;
		}
	}

	if (push_free_task_on_gpu_with_least_task_in_planned_task == 1)
	{
		free(sorted_gpu_list_by_nb_task_in_planned_task);
		free(planned_task_sizes);
	}

	new_tasks_initialized = true;

	if (dependances == 1)
	{
		initialize_task_data_gpu_single_task_dependencies(task, 1);
	}
	else
	{
		initialize_task_data_gpu_single_task_no_dependencies(task, 1);
	}

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_initialisation;
	gettimeofday(&time_end_initialisation, NULL);
	time_total_initialisation += (time_end_initialisation.tv_sec - time_start_initialisation.tv_sec)*1000000LL + time_end_initialisation.tv_usec - time_start_initialisation.tv_usec;
#endif

	/* Pushing the task in sched_list. It's this list that will be randomized
	 * and put in main_task_list in pull_task.
	 */
	struct _starpu_darts_sched_data *data = component->data;
	if (task_order == 2 && dependances == 1) /* Cas ordre naturel mais avec dépendances. Pas de points de départs différents. Je met dans le back de la liste de tâches principales. */
	{
		starpu_task_list_push_back(&data->main_task_list, task);
	}
	else
	{
		starpu_task_list_push_front(&data->sched_list, task);
		NT_DARTS++;
	}
	starpu_push_task_end(task);

	_REFINED_MUTEX_UNLOCK();
	_LINEAR_MUTEX_UNLOCK();

	component->can_pull(component);
	return 0;
}

static void merge(int arr[], int l, int m, int r, struct starpu_task **task_tab)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	/* create temp arrays */
	int L[n1], R[n2];
	struct starpu_task *L_task_tab[n1];
	struct starpu_task *R_task_tab[n2];

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < n1; i++)
	{
		L[i] = arr[l + i];
		L_task_tab[i] = task_tab[l + i];
	}
	for (j = 0; j < n2; j++)
	{
		R[j] = arr[m + 1 + j];
		R_task_tab[j] = task_tab[m + 1 + j];
	}

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0; // Initial index of first subarray
	j = 0; // Initial index of second subarray
	k = l; // Initial index of merged subarray
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			task_tab[k] = L_task_tab[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			task_tab[k] =  R_task_tab[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of L[], if there
	   are any */
	while (i < n1)
	{
		arr[k] = L[i];
		task_tab[k] =  L_task_tab[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there
	   are any */
	while (j < n2)
	{
		arr[k] = R[j];
		task_tab[k] =  R_task_tab[j];
		j++;
		k++;
	}
}

/* l is for left index and r is right index of the
sub-array of arr to be sorted */
static void mergeSort(int *arr, int l, int r, struct starpu_task **task_tab)
{
	if (l < r)
	{
		// Same as (l+r)/2, but avoids overflow for
		// large l and h
		int m = l + (r - l) / 2;

		// Sort first and second halves
		mergeSort(arr, l, m, task_tab);
		mergeSort(arr, m + 1, r, task_tab);

		merge(arr, l, m, r, task_tab);
	}
}

/* Randomize only sched_data, so new task that were made available. */
static void randomize_new_task_list(struct _starpu_darts_sched_data *d)
{
	int i = 0;
	struct starpu_task *task_tab[NT_DARTS]; /* NT_DARTS is the number of "new" available tasks. */

	for (i = 0; i < NT_DARTS; i++)
	{
		task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
	}
	for (i = 0; i < NT_DARTS; i++)
	{
		int random = rand()%(NT_DARTS - i);
		starpu_task_list_push_back(&d->main_task_list, task_tab[random]);
		task_tab[random] = task_tab[NT_DARTS - i - 1];
	}
}

/* Randomize the full set of tasks. */
static void randomize_full_task_list(struct _starpu_darts_sched_data *d)
{
	/* Version where I choose random numbers for each task, then sort
	 * at the same time with a merge sort of the array of random integers and the array of
	 * tasks. Then I run through the main task list, inserting 1 by 1 the
	 * tasks at their positions. */
	int i;
	int size_main_task_list = starpu_task_list_size(&d->main_task_list);
	struct starpu_task *task_tab[NT_DARTS];
	struct starpu_task *task = NULL;
	int random_number[NT_DARTS];
	int avancement_main_task_list = 0;

	/* Fill in a table with the new tasks + draw a random number
	 * for each task. */
	for (i = 0; i < NT_DARTS; i++)
	{
		task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
		random_number[i] = rand()%size_main_task_list;
	}

	/* Appel du tri fusion. */
	mergeSort(random_number, 0, NT_DARTS - 1, task_tab);

	/* Filling the main task list in order and according to the number drawn. */
	task = starpu_task_list_begin(&d->main_task_list);
	for (i = 0; i < NT_DARTS; i++)
	{
		int j;
		for (j = avancement_main_task_list; j < random_number[i]; j++)
		{
			task = starpu_task_list_next(task);
			avancement_main_task_list++;
		}
		starpu_task_list_insert_before(&d->main_task_list, task_tab[i], task);
	}
}

/* Each GPU has a pointer to its first task to pop.
 * Then in scheduling, when you pop for the first time, that's the one.
 * In addition, the first pop is managed directly outside random
 * thanks to the first_task attribute of struct planned_task. */
static void natural_order_task_list(struct _starpu_darts_sched_data *d)
{
	int j=0;
	int i;
	for (i = 0; i < NT_DARTS; i++)
	{
		if (i == (NT_DARTS/_nb_gpus)*j && j < _nb_gpus)
		{
			struct starpu_task *task;
			task = starpu_task_list_pop_front(&d->sched_list);
			tab_gpu_planned_task[j].first_task_to_pop = task;
			starpu_task_list_push_back(&d->main_task_list, task);
			j++;
		}
		else
		{
			starpu_task_list_push_back(&d->main_task_list, starpu_task_list_pop_front(&d->sched_list));
		}
    }
}

/* Randomize the full list of data not used yet for all the GPU. */
static void randomize_full_data_not_used_yet()
{
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		int number_of_data = _starpu_darts_gpu_data_not_used_list_size(tab_gpu_planned_task[i].gpu_data);
		struct _starpu_darts_gpu_data_not_used *data_tab[number_of_data];
		int j;
		for (j = 0; j < number_of_data; j++)
		{
			data_tab[j] = _starpu_darts_gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data);
		}
		struct _starpu_darts_gpu_data_not_used_list *randomized_list = _starpu_darts_gpu_data_not_used_list_new();

		for (j = 0; j < number_of_data; j++)
		{
			int random = rand()%(number_of_data - j);
			_starpu_darts_gpu_data_not_used_list_push_back(randomized_list, data_tab[random]);

			/* I replace the box with the last task in the table */
			data_tab[random] = data_tab[number_of_data - j - 1];
		}
		/* Then replace the list with it. */
		tab_gpu_planned_task[i].gpu_data = randomized_list;
	}
}

/* Randomize the new data and put them at the end of datanotused for all the GPU. */
static void randomize_new_data_not_used_yet()
{
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		if (!_starpu_darts_gpu_data_not_used_list_empty(tab_gpu_planned_task[i].new_gpu_data))
		{
			int number_new_data = _starpu_darts_gpu_data_not_used_list_size(tab_gpu_planned_task[i].new_gpu_data);
			struct _starpu_darts_gpu_data_not_used *data_tab[number_new_data];
			int j;
			for (j = 0; j < number_new_data; j++)
			{
				data_tab[j] = _starpu_darts_gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].new_gpu_data);
			}
			for (j = 0; j < number_new_data; j++)
			{
				int random = rand()%(number_new_data - j);
				_starpu_darts_gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, data_tab[random]);

				data_tab[random] = data_tab[number_new_data - j - 1];
			}
		}
	}
}

/* The set of task is not randomized.
 * To make GPU work on different part of the applications they all have a version of the task list that start at a different position.
 * GPU1 starts at the first task, GPU2 at the n/NGPUth task etc... */
static void natural_order_data_not_used_yet()
{
	/* I need this for the %random. */
	int number_of_data = _starpu_darts_gpu_data_not_used_list_size(tab_gpu_planned_task[0].gpu_data);

	struct _starpu_darts_gpu_data_not_used *data_tab[number_of_data];
	int i;
	for (i = 1; i < _nb_gpus; i++)
	{
		int j;
		for (j = 0; j < (number_of_data/_nb_gpus)*i; j++)
		{
			data_tab[j] = _starpu_darts_gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data);
		}
		struct _starpu_darts_gpu_data_not_used_list *natural_order_list = _starpu_darts_gpu_data_not_used_list_new();
		for (j = 0; j < number_of_data - ((number_of_data/_nb_gpus)*i); j++)
		{
			_starpu_darts_gpu_data_not_used_list_push_back(natural_order_list, _starpu_darts_gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data));
		}
		for (j = 0; j < (number_of_data/_nb_gpus)*i; j++)
		{
			_starpu_darts_gpu_data_not_used_list_push_back(natural_order_list, data_tab[j]);
		}

		/* Then replace the list with it. */
		tab_gpu_planned_task[i].gpu_data = natural_order_list;
	}
}

/* Update the "best" data if the candidate data has better values. */
static void update_best_data_single_decision_tree(int *number_free_task_max, double *remaining_expected_length_max, starpu_data_handle_t *handle_popped, int *priority_max, int *number_1_from_free_task_max, int nb_free_task_candidate, double remaining_expected_length_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int *data_chosen_index, int i, struct starpu_task* *best_1_from_free_task, struct starpu_task *best_1_from_free_task_candidate, double transfer_min_candidate, double *transfer_min, double temp_length_free_tasks_max, double *ratio_transfertime_freetask_min)
{
	(void)data_chosen_index;
	(void)i;
	double ratio_transfertime_freetask_candidate = 0;

	/* There are more return that updates, so the if are reversed and if we don't return at all, then we have a new "best" data. */
	if (dopt_selection_order == 0)
	{
		/* First tiebreak with most free task */
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		/* Then with number of 1 from free */
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
			{
				return;
			}
			else if (number_1_from_free_task_candidate == *number_1_from_free_task_max)
			{
				/* Then with priority */
				if (prio == 1 && *priority_max > priority_candidate)
				{
					return;
				}
				/* Then with time of task in the list of task using this data */
				else if ((*priority_max == priority_candidate || prio == 0) && remaining_expected_length_candidate <= *remaining_expected_length_max)
				{
#ifdef STARPU_DARTS_STATS
					if (remaining_expected_length_candidate == *remaining_expected_length_max)
					{
						data_choice_per_index = true;
					}
#endif
					return;
				}
			}
		}
	}
	else if (dopt_selection_order == 1)
	{
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (prio == 1 && *priority_max > priority_candidate)
			{
				return;
			}
			else if (*priority_max == priority_candidate)
			{
				if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
				{
					return;
				}
				else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
				{
#ifdef STARPU_DARTS_STATS
					if (remaining_expected_length_candidate == *remaining_expected_length_max)
					{
						data_choice_per_index = true;
					}
#endif
					return;
				}
			}
		}
	}
	else if (dopt_selection_order == 2)
	{
		if (transfer_min_candidate > *transfer_min)
		{
			return;
		}
		else if (transfer_min_candidate == *transfer_min)
		{
			if (nb_free_task_candidate < *number_free_task_max)
			{
				return;
			}
			else if (nb_free_task_candidate == *number_free_task_max)
			{
				if (prio == 1 && *priority_max > priority_candidate)
				{
					return;
				}
				else if (*priority_max == priority_candidate)
				{
					if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
					{
						return;
					}
					else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
					{
#ifdef STARPU_DARTS_STATS
						if (remaining_expected_length_candidate == *remaining_expected_length_max)
						{
							data_choice_per_index = true;
						}
#endif
						return;
					}
				}
			}
		}
	}
	else if (dopt_selection_order == 3)
	{
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (transfer_min_candidate > *transfer_min)
			{
				return;
			}
			else if (transfer_min_candidate == *transfer_min)
			{
				if (prio == 1 && *priority_max > priority_candidate)
				{
					return;
				}
				else if (*priority_max == priority_candidate)
				{
					if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
					{
						return;
					}
					else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
					{
#ifdef STARPU_DARTS_STATS
						if (remaining_expected_length_candidate == *remaining_expected_length_max)
						{
							data_choice_per_index = true;
						}
#endif
						return;
					}
				}
			}
		}
	}
	else if (dopt_selection_order == 4)
	{
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (prio == 1 && *priority_max > priority_candidate)
			{
				return;
			}
			else if (*priority_max == priority_candidate)
			{
				if (transfer_min_candidate > *transfer_min)
				{
					return;
				}
				else if (transfer_min_candidate == *transfer_min)
				{
					if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
					{
						return;
					}
					else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
					{
#ifdef STARPU_DARTS_STATS
						if (remaining_expected_length_candidate == *remaining_expected_length_max)
						{
							data_choice_per_index = true;
						}
#endif
						return;
					}
				}
			}
		}
	}
	else if (dopt_selection_order == 5)
	{
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (prio == 1 && *priority_max > priority_candidate)
			{
				return;
			}
			else if (*priority_max == priority_candidate)
			{
				if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
				{
					return;
				}
				else if (number_1_from_free_task_candidate == *number_1_from_free_task_max)
				{
					if (transfer_min_candidate > *transfer_min)
					{
						return;
					}
					else if ((transfer_min_candidate == *transfer_min) && remaining_expected_length_candidate <= *remaining_expected_length_max)
					{
#ifdef STARPU_DARTS_STATS
						if (remaining_expected_length_candidate == *remaining_expected_length_max)
						{
							data_choice_per_index = true;
						}
#endif
						return;
					}
				}
			}
		}
	}
	else if (dopt_selection_order == 6)
	{
		if (nb_free_task_candidate < *number_free_task_max)
		{
			return;
		}
		else if (nb_free_task_candidate == *number_free_task_max)
		{
			if (prio == 1 && *priority_max > priority_candidate)
			{
				return;
			}
			else if (*priority_max == priority_candidate)
			{
				if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
				{
					return;
				}
				else if (number_1_from_free_task_candidate == *number_1_from_free_task_max)
				{
					if (remaining_expected_length_candidate < *remaining_expected_length_max)
					{
						return;
					}
					else if ((remaining_expected_length_candidate == *remaining_expected_length_max) && transfer_min_candidate >= *transfer_min)
					{
#ifdef STARPU_DARTS_STATS
						if (transfer_min_candidate == *transfer_min)
						{
							data_choice_per_index = true;
						}
#endif
						return;
					}
				}
			}
		}
	}
	else if (dopt_selection_order == 7)
	{
		if (temp_length_free_tasks_max == 0)
		{
			ratio_transfertime_freetask_candidate = DBL_MAX;
		}
		else
		{
			ratio_transfertime_freetask_candidate = transfer_min_candidate/temp_length_free_tasks_max;
		}
		if (ratio_transfertime_freetask_candidate > *ratio_transfertime_freetask_min)
		{
			return;
		}
		else if (ratio_transfertime_freetask_candidate == *ratio_transfertime_freetask_min)
		{
			if (nb_free_task_candidate < *number_free_task_max)
			{
				return;
			}
			else if (nb_free_task_candidate == *number_free_task_max)
			{
				if (prio == 1 && *priority_max > priority_candidate)
				{
					return;
				}
				else if (*priority_max == priority_candidate)
				{
					if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
					{
						return;
					}
					else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
					{
#ifdef STARPU_DARTS_STATS
						if (remaining_expected_length_candidate == *remaining_expected_length_max)
						{
							data_choice_per_index = true;
						}
#endif

						return;
					}
				}
			}
		}
	}
	else
	{
		printf("Wrong value for STARPU_DARTS_DOPT_SELECTION_ORDER\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}

#ifdef STARPU_DARTS_STATS
	data_choice_per_index = false;
#endif

	/* We have a new "best" data! pdate */
	*number_free_task_max = nb_free_task_candidate;
	*remaining_expected_length_max = remaining_expected_length_candidate;
	*number_1_from_free_task_max = number_1_from_free_task_candidate;
	*handle_popped = handle_candidate;
	*priority_max = priority_candidate;
	*best_1_from_free_task = best_1_from_free_task_candidate;
	*transfer_min = transfer_min_candidate;
	*ratio_transfertime_freetask_min = ratio_transfertime_freetask_candidate;

#ifdef STARPU_DARTS_STATS
	*data_chosen_index = i + 1;
#endif
}

static struct starpu_task *get_highest_priority_task(struct starpu_task_list *l)
{
	int max_priority = INT_MIN;
	struct starpu_task *highest_priority_task = starpu_task_list_begin(l);
	struct starpu_task *t;
	for (t = starpu_task_list_begin(l); t != starpu_task_list_end(l); t = starpu_task_list_next(t))
	{
		if (t->priority > max_priority)
		{
			max_priority = t->priority;
			highest_priority_task = t;
		}
	}
	return highest_priority_task;
}

/* Erase a task from the main task list.
 * Also erase pointer in the data.
 * Only for one GPU.
 * Also update the expected length of task using this data.
 */
static void _starpu_darts_erase_task_and_data_pointer(struct starpu_task *task, struct starpu_task_list *l)
{
	struct _starpu_darts_pointer_in_task *pt = task->sched_data;
	unsigned j;
	for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, j);
		if (pt->tud[j] != NULL)
		{
			_starpu_darts_task_using_data_list_erase(pt->pointer_to_D[j]->sched_data, pt->tud[j]);
			pt->tud[j] = NULL;
		}

		/* Reduce expected length of task using this data */
		struct _starpu_darts_handle_user_data *hud = pt->pointer_to_D[j]->user_data;
		hud->sum_remaining_task_expected_length -= starpu_task_expected_length(task, perf_arch, 0);
		_STARPU_SCHED_PRINT("Adding in planned task. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, j), hud->sum_remaining_task_expected_length);

		pt->pointer_to_D[j]->user_data = hud;
	}
	starpu_task_list_erase(l, pt->pointer_to_cell);
}

/* Main function of DARTS scheduling.
 * Takes the set of available task, the GPU (or CPU) asking for work as an input.
 * Chooses the best data that is not yet in memory of the PU and fill a buffer of task with task associated with this data. */
static void _starpu_darts_scheduling_3D_matrix(struct starpu_task_list *main_task_list, int current_gpu, struct _starpu_darts_gpu_planned_task *g)
{
	_STARPU_SCHED_PRINT("Début de sched 3D avec GPU %d.\n", current_gpu);

	Dopt[current_gpu] = NULL;

#ifdef STARPU_DARTS_STATS
	struct timeval time_start_schedule;
	gettimeofday(&time_start_schedule, NULL);
#endif
	double remaining_expected_length_max = 0;
	struct starpu_task *best_1_from_free_task = NULL;
	struct starpu_task *temp_best_1_from_free_task = NULL;

	/* Values used to know if the currently selected data is better that the pne already chosen */
	int number_free_task_max = 0;  /* Number of free task with selected data */
	int temp_number_free_task_max = 0;
	int number_1_from_free_task_max = 0;  /* Number of task on from free with selected data */
	int temp_number_1_from_free_task_max = 0;
	int priority_max = INT_MIN; /* Highest priority of a task using the chosen data that we know will be pushed to planned task. */
	int temp_priority_max = INT_MIN; /* Highest priority of a task using the chosen data that we know will be pushed to planned task. */

	/* To ty and use starpu_data_expected_transfer_time */
	double transfer_time_min = DBL_MAX;
	double temp_transfer_time_min = DBL_MAX;

	/* For the case where DOPT_SELECTION_OERDER >= 7. In this case I look at the transfertime/timeoffree task. thus I need to keep track of the length of the free tasks. */
	double ratio_transfertime_freetask_min = DBL_MAX;
	double temp_length_free_tasks_max = 0;

	starpu_data_handle_t handle_popped = NULL; /* Pointer to chosen best data */

	struct _starpu_darts_handle_user_data *hud = NULL;

	int data_chosen_index = 0; /* Forced to declare it here because of the fonction update */
#ifdef STARPU_DARTS_STATS
	int nb_data_looked_at = 0; /* Uniquement le cas ou on choisis depuis la mémoire */
#endif

	_STARPU_SCHED_PRINT("Il y a %d données parmi lesquelles choisir pour le GPU %d.\n", _starpu_darts_gpu_data_not_used_list_size(g->gpu_data), current_gpu);
	/* If it's the first task of the GPU, no need to schedule anything, just return it. */
	if (g->first_task == true)
	{
		_STARPU_SCHED_PRINT("Hey! C'est la première tâche du GPU n°%d!\n", current_gpu);
#ifdef STARPU_DARTS_STATS
		if (iteration_DARTS == 1)
		{
			FILE *f = NULL;
			int size = strlen(_output_directory) + strlen("/Data_DARTS_data_chosen_stats_GPU_.csv") + 3;
			char path[size];
			snprintf(path, size, "%s%s%d%s", _output_directory, "/Data_DARTS_data_chosen_stats_GPU_", current_gpu, ".csv");
			f = fopen(path, "a");
			fprintf(f, "%d,%d,%d,%d\n", g->number_data_selection, 0, 0, 0);
			fclose(f);
		}
#endif

		g->first_task = false;

		if (task_order == 2 && dependances == 0) /* Cas liste des taches et données naturelles et pas de dpendances donc points de départs différents à l'aide de first_task_to_pop. */
		{
			struct starpu_task *task = NULL;
			task = g->first_task_to_pop;
			g->first_task_to_pop = NULL;
			if (!starpu_task_list_ismember(main_task_list, task))
			{
				goto random;
			}

			unsigned x;
			for (x = 0; x < STARPU_TASK_GET_NBUFFERS(task); x++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(task, x);
				if (!_starpu_darts_gpu_data_not_used_list_empty(g->gpu_data))
				{
					struct _starpu_darts_gpu_data_not_used *e;
					for (e = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data); e != _starpu_darts_gpu_data_not_used_list_end(g->gpu_data); e = _starpu_darts_gpu_data_not_used_list_next(e))
					{
						if (e->D == STARPU_TASK_GET_HANDLE(task, x))
						{
							_starpu_darts_gpu_data_not_used_list_erase(g->gpu_data, e);
							hud = e->D->user_data;
							_STARPU_SCHED_PRINT("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu);
							hud->is_present_in_data_not_used_yet[current_gpu] = 0;
							e->D->user_data = hud;

							_starpu_darts_gpu_data_not_used_delete(e);
							break;
						}
					}
				}
			}

			_REFINED_MUTEX_LOCK();

			/* Add it from planned task compteur */
			_starpu_darts_increment_planned_task_data(task, current_gpu);

			_starpu_darts_erase_task_and_data_pointer(task, main_task_list);
			starpu_task_list_push_back(&g->planned_task, task);

			_REFINED_MUTEX_UNLOCK();

			goto end_scheduling;
		}
		else
		{
			goto random;
		}
	}

	if (_starpu_darts_gpu_data_not_used_list_empty(g->gpu_data))
	{
		_STARPU_SCHED_PRINT("Random selection car liste des données non utilisées vide.\n");
		goto random;
	}

	/* To know if all the data needed for a task are loaded in memory. */
	int data_not_available = 0;
	bool data_available = true;

	/* If threshold is different than 0 (set with env var STARPU_DARTS_THRESHOLD), less data are explored to reduce complexity. */
	int choose_best_data_threshold = INT_MAX;
	if (threshold == 1)
	{
		if (app == 0)
		{
			if (NT_DARTS > 14400)
			{
				choose_best_data_threshold = 110;
			}
		}
		else if (NT_DARTS > 1599 && dependances == 0)
		{
			choose_best_data_threshold = 200;
		}
		else if (dependances == 1)
		{
			choose_best_data_threshold = 200;
		}
	}

#ifdef STARPU_DARTS_STATS
	struct timeval time_start_choose_best_data;
	gettimeofday(&time_start_choose_best_data, NULL);
#endif

#ifdef STARPU_DARTS_STATS
	data_choice_per_index = false;
#endif

	if (choose_best_data_from == 0) /* We explore all unused data. In the else case we look at data from the task associated with the data already in memory. */
	{
#ifdef STARPU_DARTS_STATS
		g->number_data_selection++;
#endif

		int i=0;
		struct _starpu_darts_gpu_data_not_used *e;
		for (e = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data); e != _starpu_darts_gpu_data_not_used_list_end(g->gpu_data) && i != choose_best_data_threshold; e = _starpu_darts_gpu_data_not_used_list_next(e), i++)
		{
			temp_transfer_time_min = starpu_data_expected_transfer_time(e->D, current_gpu, STARPU_R);
			_STARPU_SCHED_PRINT("Temp transfer time is %f\n", temp_transfer_time_min);
			temp_number_free_task_max = 0;
			temp_number_1_from_free_task_max = 0;
			temp_priority_max = INT_MIN;
			temp_best_1_from_free_task = NULL;
			temp_length_free_tasks_max = 0;

#ifdef STARPU_DARTS_STATS
			nb_data_looked_at++;
#endif

			if (e->D->sched_data != NULL)
			{
				struct _starpu_darts_task_using_data *t;
				for (t = _starpu_darts_task_using_data_list_begin(e->D->sched_data); t != _starpu_darts_task_using_data_list_end(e->D->sched_data); t = _starpu_darts_task_using_data_list_next(t))
				{
					/* I put it at false if at least one data is missing. */
					data_not_available = 0;
					unsigned j;
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
					{
						STARPU_IGNORE_UTILITIES_HANDLES(t->pointer_to_T, j);
						/* I test if the data is on memory */
						if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
						{
							if (simulate_memory == 0)
							{
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]))
								{
									data_not_available++;
								}
							}
							else if (simulate_memory == 1)
							{
								hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]) && hud->nb_task_in_pulled_task[current_gpu] == 0 && hud->nb_task_in_planned_task[current_gpu] == 0)
								{
									data_not_available++;
								}
							}
						}
					}

					if (data_not_available == 0)
					{
						temp_number_free_task_max++;

						/* With threshold == 2, we stop as soon as we find a data that allow at least one fee task. */
						if (threshold == 2)
						{
							hud = e->D->user_data;
							update_best_data_single_decision_tree(&number_free_task_max, &remaining_expected_length_max, &handle_popped, &priority_max, &number_1_from_free_task_max, temp_number_free_task_max, hud->sum_remaining_task_expected_length, e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_chosen_index, i, &best_1_from_free_task, temp_best_1_from_free_task, temp_transfer_time_min, &transfer_time_min, temp_length_free_tasks_max, &ratio_transfertime_freetask_min);

							goto end_choose_best_data;
						}

						/* For the first one I want to forget priority of one from free tasks. */
						if (temp_number_free_task_max == 1)
						{
							temp_priority_max = t->pointer_to_T->priority;
						}
						else if (t->pointer_to_T->priority > temp_priority_max)
						{
							temp_priority_max = t->pointer_to_T->priority;
						}

						temp_length_free_tasks_max += starpu_task_expected_length(t->pointer_to_T, perf_arch, 0);
					}
					else if (data_not_available == 1)
					{
						temp_number_1_from_free_task_max++;

						/* Getting the max priority */
						if (t->pointer_to_T->priority > temp_priority_max)
						{
							temp_priority_max = t->pointer_to_T->priority;
							temp_best_1_from_free_task = t->pointer_to_T;
						}
					}
				}

				/* Checking if current data is better */
				hud = e->D->user_data;
				update_best_data_single_decision_tree(&number_free_task_max, &remaining_expected_length_max, &handle_popped, &priority_max, &number_1_from_free_task_max, temp_number_free_task_max, hud->sum_remaining_task_expected_length, e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_chosen_index, i, &best_1_from_free_task, temp_best_1_from_free_task, temp_transfer_time_min, &transfer_time_min, temp_length_free_tasks_max, &ratio_transfertime_freetask_min);
			}
		}
	}
	else if (choose_best_data_from == 1) /* The case where I only look at data (not yet in memory) from tasks using data in memory. */
	{
		/* To avoid looking at the same data twice in the same iteration. */
		struct _starpu_darts_handle_user_data *hud_last_check = NULL;

		/* Be careful here it's useful not to put it between ifdef! Because I use it to know if I haven't already looked at the data */
		g->number_data_selection++;

		/* Récupération des données en mémoire */
		starpu_data_handle_t *data_on_node;
		unsigned nb_data_on_node = 0;
		int *valid;
		starpu_data_get_node_data(current_gpu, &data_on_node, &valid, &nb_data_on_node);

		/* I put myself on a data in memory. */
		unsigned x;
		for (x = 0; x < nb_data_on_node; x++)
		{
			STARPU_IGNORE_UTILITIES_HANDLES_FROM_DATA(data_on_node[x]);
			_STARPU_SCHED_PRINT("On data nb %d/%d from memory\n", x, nb_data_on_node);

			/* Je me met sur une tâche de cette donnée en question. */
			struct _starpu_darts_task_using_data *t2;
			for (t2 = _starpu_darts_task_using_data_list_begin(data_on_node[x]->sched_data); t2 != _starpu_darts_task_using_data_list_end(data_on_node[x]->sched_data); t2 = _starpu_darts_task_using_data_list_next(t2))
			{
				_STARPU_SCHED_PRINT("On task %p from this data\n", t2);

				/* I set myself to a data item of this task (which is not the one in memory). */
				unsigned k;
				for (k = 0; k < STARPU_TASK_GET_NBUFFERS(t2->pointer_to_T); k++)
				{
					STARPU_IGNORE_UTILITIES_HANDLES(t2->pointer_to_T, k);
					_STARPU_SCHED_PRINT("On data %p from this task\n", STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k));
					hud_last_check = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data;

					/* Here you should not look at the same data twice if possible. It can happen. */
					if (STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k) != data_on_node[x] && hud_last_check->last_check_to_choose_from[current_gpu] != g->number_data_selection && !starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k), memory_nodes[current_gpu]))
					{
						_STARPU_SCHED_PRINT("Data %p is being looked at\n", STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k));
#ifdef STARPU_DARTS_STATS
						nb_data_looked_at++;
#endif

						/* Update the iteration for the data so as not to look at it twice at that iteration. */
						hud_last_check->last_check_to_choose_from[current_gpu] = g->number_data_selection;
						STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data = hud_last_check;

						temp_number_free_task_max = 0;
						temp_number_1_from_free_task_max = 0;
						temp_priority_max = INT_MIN;
						temp_best_1_from_free_task = NULL;
						temp_length_free_tasks_max = 0;

#ifdef STARPU_DARTS_STATS
						nb_data_looked_at++;
#endif

						struct _starpu_darts_task_using_data *t;
						for (t = _starpu_darts_task_using_data_list_begin(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t != _starpu_darts_task_using_data_list_end(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t = _starpu_darts_task_using_data_list_next(t))
						{
							_STARPU_SCHED_PRINT("Task %p is using this data\n", t);

							/* I put it at 1 if at least one data is missing. */
							data_not_available = 0;

							unsigned j;
							for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
							{
								STARPU_IGNORE_UTILITIES_HANDLES(t->pointer_to_T, j);
								if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k))
								{
									if (simulate_memory == 0)
									{
										if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]))
										{
											data_not_available++;
										}
									}
									else if (simulate_memory == 1)
									{
										hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
										if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]) && hud->nb_task_in_pulled_task[current_gpu] == 0 && hud->nb_task_in_planned_task[current_gpu] == 0)
										{
											data_not_available++;
										}
									}
								}
							}

							if (data_not_available == 0)
							{
								temp_number_free_task_max++;

								/* Version where I stop as soon as I get a free task. */
								if (threshold == 2)
								{
									hud = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data;
									update_best_data_single_decision_tree(&number_free_task_max, &remaining_expected_length_max, &handle_popped, &priority_max, &number_1_from_free_task_max, temp_number_free_task_max, hud->sum_remaining_task_expected_length, STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k), temp_priority_max, temp_number_1_from_free_task_max, &data_chosen_index, x, &best_1_from_free_task, temp_best_1_from_free_task, temp_transfer_time_min, &transfer_time_min, temp_length_free_tasks_max, &ratio_transfertime_freetask_min);

									goto end_choose_best_data;
								}

								/* For the first one I want to forget priority of one from free tasks. */
								if (temp_number_free_task_max == 1)
								{
									temp_priority_max = t->pointer_to_T->priority;
								}
								else if (t->pointer_to_T->priority > temp_priority_max)
								{
									temp_priority_max = t->pointer_to_T->priority;
								}
								temp_length_free_tasks_max += starpu_task_expected_length(t->pointer_to_T, perf_arch, 0);
							}
							else if (data_not_available == 1)
							{
								temp_number_1_from_free_task_max++;
								if (t->pointer_to_T->priority > temp_priority_max)
								{
									temp_priority_max = t->pointer_to_T->priority;
									temp_best_1_from_free_task = t->pointer_to_T;
								}
							}
						}

						temp_transfer_time_min = starpu_data_expected_transfer_time(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k), current_gpu ,STARPU_R);

						/* Update best data if needed */
						hud = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data;
						update_best_data_single_decision_tree(&number_free_task_max, &remaining_expected_length_max, &handle_popped, &priority_max, &number_1_from_free_task_max, temp_number_free_task_max, hud->sum_remaining_task_expected_length, STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k), temp_priority_max, temp_number_1_from_free_task_max, &data_chosen_index, x, &best_1_from_free_task, temp_best_1_from_free_task, temp_transfer_time_min, &transfer_time_min, temp_length_free_tasks_max, &ratio_transfertime_freetask_min);
					}
				}
			}
		}
	}

	_STARPU_SCHED_PRINT("Best data is = %p: %d free tasks and %d 1 from free tasks. Transfer time %f\n", handle_popped, number_free_task_max, number_1_from_free_task_max, transfer_time_min);

 end_choose_best_data : ;

#ifdef STARPU_DARTS_STATS
	if (data_choice_per_index == true)
	{
		nb_data_selection_per_index++;
	}
	nb_data_selection++;
#endif

	/* Look at data conflict. If there is one I need to re-start the schedule for one of the GPU. */
	data_conflict[current_gpu] = false;
	Dopt[current_gpu] = handle_popped;
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		if (i != current_gpu)
		{
			if (Dopt[i] == handle_popped && handle_popped != NULL)
			{
				_STARPU_SCHED_PRINT("Iteration %d. Same data between GPU %d and GPU %d: %p.\n", iteration_DARTS, current_gpu, i + 1, handle_popped);
#ifdef STARPU_DARTS_STATS
				number_data_conflict++;
#endif

				data_conflict[current_gpu] = true;
			}
		}
	}

#ifdef STARPU_DARTS_STATS
	if (iteration_DARTS == 1)
	{
		FILE *f = NULL;
		int size = strlen(_output_directory) + strlen("/Data_DARTS_data_chosen_stats_GPU_.csv") + 3;
		char path[size];
		snprintf(path, size, "%s%s%d%s", _output_directory, "/Data_DARTS_data_chosen_stats_GPU_", current_gpu, ".csv");
		f = fopen(path, "a");
		if (number_free_task_max != 0)
		{
			nb_task_added_in_planned_task = number_free_task_max;
		}
		else
		{
			nb_task_added_in_planned_task = 1;
		}
		fprintf(f, "%d,%d,%d,%d\n", g->number_data_selection, data_chosen_index, nb_data_looked_at - data_chosen_index, nb_task_added_in_planned_task);
		fclose(f);
	}
	struct timeval time_end_choose_best_data;
	gettimeofday(&time_end_choose_best_data, NULL);
	time_total_choose_best_data += (time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)*1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec;
#endif

	if (number_free_task_max != 0)
	{
#ifdef STARPU_DARTS_STATS
		struct timeval time_start_fill_planned_task_list;
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		nb_free_choice++;
#endif

		/* I erase the data from the list of data not used. */
		if (choose_best_data_from == 0)
		{
			struct _starpu_darts_gpu_data_not_used *e;
			e = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data);
			while (e->D != handle_popped)
			{
				  e = _starpu_darts_gpu_data_not_used_list_next(e);
			}

			_starpu_darts_gpu_data_not_used_list_erase(g->gpu_data, e);
			hud = e->D->user_data;
			hud->is_present_in_data_not_used_yet[current_gpu] = 0;
			e->D->user_data = hud;

			_starpu_darts_gpu_data_not_used_delete(e);

			_STARPU_SCHED_PRINT("Erased data %p\n", e->D);
			print_data_not_used_yet_one_gpu(g->gpu_data, current_gpu);
		}

		_STARPU_SCHED_PRINT("The data adding the most free tasks is %p and %d task.\n", handle_popped, number_free_task_max);

		_REFINED_MUTEX_LOCK();

		struct _starpu_darts_task_using_data *t;
		for (t = _starpu_darts_task_using_data_list_begin(handle_popped->sched_data); t != _starpu_darts_task_using_data_list_end(handle_popped->sched_data); t = _starpu_darts_task_using_data_list_next(t))
		{
			data_available = true;
			unsigned j;
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(t->pointer_to_T, j);
				if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
				{
					if (simulate_memory == 0)
					{
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]))
						{
							data_available = false;
							break;
						}
					}
					else if (simulate_memory == 1)
					{
						hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), memory_nodes[current_gpu]) && hud->nb_task_in_pulled_task[current_gpu] == 0 && hud->nb_task_in_planned_task[current_gpu] == 0)
						{
							data_available = false;
							break;
						}
					}
				}
			}
			if (data_available == true)
			{
				_starpu_darts_increment_planned_task_data(t->pointer_to_T, current_gpu);

#ifdef PRINT
				printf("Pushing free %p in planned_task of GPU %d :", t->pointer_to_T, current_gpu); fflush(stdout);
				for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
				{
					STARPU_IGNORE_UTILITIES_HANDLES(t->pointer_to_T, j);
					printf(" %p", STARPU_TASK_GET_HANDLE(t->pointer_to_T, j));
					fflush(stdout);
				}
				printf("\n"); fflush(stdout);
#endif

				_starpu_darts_erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
				starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
			}
		}

		_REFINED_MUTEX_UNLOCK();

#ifdef STARPU_DARTS_STATS
		struct timeval time_end_fill_planned_task_list;
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
#endif
	}
	else if (number_1_from_free_task_max != 0 && app != 0)
	{
#ifdef STARPU_DARTS_STATS
		struct timeval time_start_fill_planned_task_list;
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		nb_1_from_free_choice++;
#endif
		_STARPU_SCHED_PRINT("The data adding the most (%d) 1_from_free tasks is %p.\n", number_1_from_free_task_max, handle_popped);

		_REFINED_MUTEX_LOCK();

		if (!starpu_task_list_ismember(main_task_list, best_1_from_free_task))
		{
			_REFINED_MUTEX_UNLOCK();
			goto random;
		}

		if (best_1_from_free_task == NULL)
		{
#ifdef STARPU_DARTS_STATS
			nb_1_from_free_task_not_found++;
#endif
			_REFINED_MUTEX_UNLOCK();
			goto random;
		}

		/* Removing the data from datanotused of the GPU. */
		if (choose_best_data_from == 0)
		{
			print_task_info(best_1_from_free_task);
			unsigned x;
			for (x = 0; x < STARPU_TASK_GET_NBUFFERS(best_1_from_free_task); x++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(best_1_from_free_task, x);
				if (!_starpu_darts_gpu_data_not_used_list_empty(g->gpu_data)) /* TODO : utile ? */
				{
					struct _starpu_darts_gpu_data_not_used *e1;
					for (e1 = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data); e1 != _starpu_darts_gpu_data_not_used_list_end(g->gpu_data); e1 = _starpu_darts_gpu_data_not_used_list_next(e1))
					{
						if (e1->D == STARPU_TASK_GET_HANDLE(best_1_from_free_task, x))
						{
							_starpu_darts_gpu_data_not_used_list_erase(g->gpu_data, e1);
							hud = e1->D->user_data;
							_STARPU_SCHED_PRINT("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e1->D, current_gpu);

							hud->is_present_in_data_not_used_yet[current_gpu] = 0;
							e1->D->user_data = hud;

							_starpu_darts_gpu_data_not_used_delete(e1);

							break;
						}
					}
				}
			}
		}

		_starpu_darts_increment_planned_task_data(best_1_from_free_task, current_gpu);
		_STARPU_SCHED_PRINT("Pushing 1_from_free task %p in planned_task of GPU %d\n", best_1_from_free_task, current_gpu);

		_starpu_darts_erase_task_and_data_pointer(best_1_from_free_task, main_task_list);
		starpu_task_list_push_back(&g->planned_task, best_1_from_free_task);
		_REFINED_MUTEX_UNLOCK();

#ifdef STARPU_DARTS_STATS
		struct timeval time_end_fill_planned_task_list;
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
#endif

	}
	else
	{
		goto random;
	}

	/* If no task have been added to the list. */
	if (starpu_task_list_empty(&g->planned_task))
	{
		/* If there was a conflict (two PU chose the same data to load), then we restart the schedule from one of them. */
		if (data_conflict[current_gpu] == true)
		{
#ifdef STARPU_DARTS_STATS
			number_critical_data_conflict++;
			number_data_conflict--;
#endif
			_STARPU_SCHED_PRINT("Critical data conflict.\n");
			_starpu_darts_scheduling_3D_matrix(main_task_list, current_gpu, g);
		}

	random: ; /* We pop a task from the main task list. Either the head (from a randomized list or not depending on STARPU_DARTS_TASK_ORDER) or the highest priority task. */

		Dopt[current_gpu] = NULL;

#ifdef STARPU_DARTS_STATS
		struct timeval time_start_pick_random_task;
		gettimeofday(&time_start_pick_random_task, NULL);
		number_random_selection++;
#endif

		struct starpu_task *task = NULL;

		_REFINED_MUTEX_LOCK();

		if (!starpu_task_list_empty(main_task_list))
		{
			if (highest_priority_task_returned_in_default_case == 1) /* Highest priority task is returned. */
			{
				task = get_highest_priority_task(main_task_list);
			}
			else /* Head of the task list is returned. */
			{
				task = starpu_task_list_pop_front(main_task_list);
			}

			_STARPU_SCHED_PRINT("\"Random\" task for GPU %d is %p.\n", current_gpu, task);
		}
		else
		{
			_STARPU_SCHED_PRINT("Return void in scheduling for GPU %d.\n", current_gpu);
			_REFINED_MUTEX_UNLOCK();
			return;
		}

		if (choose_best_data_from == 0)
		{
			unsigned x;
			for (x= 0; x < STARPU_TASK_GET_NBUFFERS(task); x++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(task, x);
				if (!_starpu_darts_gpu_data_not_used_list_empty(g->gpu_data))
				{
					struct _starpu_darts_gpu_data_not_used *e;
					for (e = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data); e != _starpu_darts_gpu_data_not_used_list_end(g->gpu_data); e = _starpu_darts_gpu_data_not_used_list_next(e))
					{
						if (e->D == STARPU_TASK_GET_HANDLE(task, x))
						{
							_starpu_darts_gpu_data_not_used_list_erase(g->gpu_data, e);
							hud = e->D->user_data;
							_STARPU_SCHED_PRINT("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu);
							hud->is_present_in_data_not_used_yet[current_gpu] = 0;
							e->D->user_data = hud;

							_starpu_darts_gpu_data_not_used_delete(e);
							break;
						}
					}
				}
			}
		}

		/* Add it from planned task compteur */
		_starpu_darts_increment_planned_task_data(task, current_gpu);
		_STARPU_SCHED_PRINT("For GPU %d, returning head of the randomized main task list in planned_task: %p.\n", current_gpu, task);
		_starpu_darts_erase_task_and_data_pointer(task, main_task_list);
		starpu_task_list_push_back(&g->planned_task, task);

		_REFINED_MUTEX_UNLOCK();

#ifdef STARPU_DARTS_STATS
		struct timeval time_end_pick_random_task;
		gettimeofday(&time_end_pick_random_task, NULL);
		time_total_pick_random_task += (time_end_pick_random_task.tv_sec - time_start_pick_random_task.tv_sec)*1000000LL + time_end_pick_random_task.tv_usec - time_start_pick_random_task.tv_usec;
#endif
		return;
	}

	end_scheduling: ;

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_schedule;
	gettimeofday(&time_end_schedule, NULL);
	time_total_schedule += (time_end_schedule.tv_sec - time_start_schedule.tv_sec)*1000000LL + time_end_schedule.tv_usec - time_start_schedule.tv_usec;
#endif

	/* TODO: Do we need this at the top and at the end of this function? */
	Dopt[current_gpu] = NULL;
}

static void _starpu_darts_add_task_to_pulled_task(int current_gpu, struct starpu_task *task)
{
	/* We increment for each data using the task, the number of task in pulled task associated with this data. */
	unsigned i;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES(task, i);
		struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		hud->nb_task_in_pulled_task[current_gpu] += 1;
		STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
	}

	struct _starpu_darts_pulled_task *p = _starpu_darts_pulled_task_new();
	p->pointer_to_pulled_task = task;
	_starpu_darts_pulled_task_list_push_back(tab_gpu_pulled_task[current_gpu].ptl, p);
}

/*
 * Get a task to return to pull_task.
 * In multi GPU it allows me to return a task from the right element in the
 * linked list without having another GPU comme and ask a task in pull_task.
 */
static struct starpu_task *get_task_to_return_pull_task_darts(int current_gpu, struct starpu_task_list *l)
{
	/* If there are still tasks either in the packages, the main task list or the refused task,
	 * I enter here to return a task or start darts_scheduling. Else I return NULL.
	 */
	/* If one or more task have been refused. Need to update planned task but not pulled task as it was already done before. */
	if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu].refused_fifo_list))
	{
		struct starpu_task *task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu].refused_fifo_list);
		_STARPU_SCHED_PRINT("Return refused task %p.\n", task);
		return task;
	}

	_REFINED_MUTEX_LOCK();
	/* If the package is not empty I can return the head of the task list. */
	if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu].planned_task))
	{
		struct starpu_task *task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu].planned_task);
		/* Remove it from planned task compteur. Could be done in an external function as I use it two times */
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			STARPU_IGNORE_UTILITIES_HANDLES(task, i);
			struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->nb_task_in_planned_task[current_gpu] = hud->nb_task_in_planned_task[current_gpu] - 1;
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		_starpu_darts_add_task_to_pulled_task(current_gpu, task);

		/* For visualisation in python. */
		_sched_visu_print_data_to_load_prefetch(task, current_gpu, 1);
		_STARPU_SCHED_PRINT("Task: %p is getting out of pull_task from planned task not empty on GPU %d.\n", task, current_gpu);
		_REFINED_MUTEX_UNLOCK();
		return task;
	}

	/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
	if (!starpu_task_list_empty(l))
	{
		_REFINED_MUTEX_UNLOCK();
		_starpu_darts_scheduling_3D_matrix(l, current_gpu, &tab_gpu_planned_task[current_gpu]);
		_REFINED_MUTEX_LOCK();
		struct starpu_task *task;
		if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu].planned_task))
		{
			task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu].planned_task);
			_starpu_darts_add_task_to_pulled_task(current_gpu, task);

			/* Remove it from planned task compteur */
			unsigned i;
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				STARPU_IGNORE_UTILITIES_HANDLES(task, i);
				struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
				hud->nb_task_in_planned_task[current_gpu] = hud->nb_task_in_planned_task[current_gpu] - 1;

				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
		}
		else
		{
			_REFINED_MUTEX_UNLOCK();
			_STARPU_SCHED_PRINT("Return NULL after scheduling call.\n");
#ifdef STARPU_DARTS_STATS
			nb_return_null_after_scheduling++;
#endif
			return NULL;
		}

		/* For visualisation in python. */
		_sched_visu_print_data_to_load_prefetch(task, current_gpu, 1);
		_STARPU_SCHED_PRINT("Return task %p from the scheduling call GPU %d.\n", task, current_gpu);
#ifdef STARPU_DARTS_STATS
		nb_return_task_after_scheduling++;
#endif
		_REFINED_MUTEX_UNLOCK();
		return task;
	}
	else
	{
		_STARPU_SCHED_PRINT("Return NULL because main task list is empty.\n");
#ifdef STARPU_DARTS_STATS
		nb_return_null_because_main_task_list_empty++;
#endif
		_REFINED_MUTEX_UNLOCK();
		return NULL;
	}
}

static bool graph_read = false; /* TODO: a suppr si j'utilise pas graph_descendants == 1 */

/* Pull tasks. When it receives new task it either append tasks to the task list or randomize
 * the task list woth the new task, depending on the parameters of STARPU_DARTS_TASK_ORDER.
 * Similarly, data can be appended or randomized with STARPU_DARTS_DATA_ORDER.
 * By default, task and data are just appended.
 * This function return task from the head of planned task.
 * If it is empty it calls darts scheduling. */
static struct starpu_task *darts_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	(void)to;
	_STARPU_SCHED_PRINT("Début de pull_task.\n");

	_LINEAR_MUTEX_LOCK();

	struct _starpu_darts_sched_data *data = component->data;

	/* If GRAPHE!=0, we compute descendants of new tasks. */
	if (graph_descendants == 1 && graph_read == false && new_tasks_initialized == true)
	{
		graph_read = true;
		_starpu_graph_compute_descendants();
		_starpu_graph_foreach(set_priority, data);
	}

	_REFINED_MUTEX_LOCK();

	if (new_tasks_initialized == true)
	{
		if (graph_descendants == 2)
		{
			_starpu_graph_compute_descendants();
			_starpu_graph_foreach(set_priority, data);
		}

#ifdef STARPU_DARTS_STATS
		nb_new_task_initialized++;
#endif

		_STARPU_SCHED_PRINT("New tasks in pull_task.\n");
		new_tasks_initialized = false;

		_STARPU_SCHED_PRINT("\n-----\nPrinting GPU's data list and NEW task list before randomization:\n");
		print_data_not_used_yet();
		print_task_list(&data->sched_list, "Main task list");

#ifdef STARPU_DARTS_STATS
		struct timeval time_start_randomize;
		gettimeofday(&time_start_randomize, NULL);
#endif

		/* Randomizing or not the task in the set of task to compute. */
		if (task_order == 0) /* Randomize every task. */
		{
			if (!starpu_task_list_empty(&data->main_task_list))
			{
				randomize_full_task_list(data);
			}
			else
			{
				randomize_new_task_list(data);
			}
		}
		else if (task_order == 1) /* Randomize new tasks. */
		{
			randomize_new_task_list(data);
		}
		else if (dependances == 0) /* Do not randomize. */
		{
			natural_order_task_list(data);
		}

		/* Randomize or not data order in datanorusedyet. */
		if (choose_best_data_from != 1) /* If we use this parametere with CHOOSE_FROM_MEM=1, we do not need to randomize data as we directly choose them from memory. */
		{
			if (data_order == 0) /* Randomize all data. */
			{
				randomize_full_data_not_used_yet();
			}
			else if (data_order == 1) /* Randomize only new data. */
			{
				randomize_new_data_not_used_yet();
			}
			else if (dependances == 0) /* Do not randomize data. */
			{
				natural_order_data_not_used_yet();
			}
		}

		NT_DARTS = 0;

#ifdef STARPU_DARTS_STATS
		struct timeval time_end_randomize;
		gettimeofday(&time_end_randomize, NULL);
		time_total_randomize += (time_end_randomize.tv_sec - time_start_randomize.tv_sec)*1000000LL + time_end_randomize.tv_usec - time_start_randomize.tv_usec;
#endif

		_STARPU_SCHED_PRINT("Il y a %d tâches.\n", NT_DARTS);
		_STARPU_SCHED_PRINT("Printing GPU's data list and main task list after randomization (STARPU_DARTS_TASK_ORDER = %d, STARPU_DARTS_DATA_ORDER = %d):\n", task_order, data_order);
		print_task_list(&data->main_task_list, "Main task list");
		_STARPU_SCHED_PRINT("-----\n\n");
	}

	_REFINED_MUTEX_UNLOCK();

	int current_gpu; /* Index in tabs of structs */
	if (cpu_only == 1)
	{
		current_gpu = 0;
	}
	else if (cpu_only == 2)
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	}
	else
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;
	}

	struct starpu_task *task = get_task_to_return_pull_task_darts(current_gpu, &data->main_task_list);

	_LINEAR_MUTEX_UNLOCK();

	_STARPU_SCHED_PRINT("Pulled %stask %p on GPU %d.\n", task?"":"NO ", task, current_gpu);
	return task;
}

static void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct _starpu_darts_gpu_planned_task *g, int gpu_id)
{
	struct _starpu_darts_gpu_data_not_used *new_element = _starpu_darts_gpu_data_not_used_new();
	new_element->D = h;

	_STARPU_SCHED_PRINT("%p gets 1 at is_present_in_data_not_used_yet with random push\n", h);
	struct _starpu_darts_handle_user_data *hud = h->user_data;
	hud->is_present_in_data_not_used_yet[gpu_id] = 1;
	h->user_data = hud;

	if (_starpu_darts_gpu_data_not_used_list_empty(g->gpu_data))
	{
		_starpu_darts_gpu_data_not_used_list_push_back(g->gpu_data, new_element);
		return;
	}
	int random = rand()%_starpu_darts_gpu_data_not_used_list_size(g->gpu_data);
	struct _starpu_darts_gpu_data_not_used *ptr;
	ptr = _starpu_darts_gpu_data_not_used_list_begin(g->gpu_data);
	int i = 0;
	for (i = 0; i < random; i++)
	{
		ptr = _starpu_darts_gpu_data_not_used_list_next(ptr);
	}
	_starpu_darts_gpu_data_not_used_list_insert_before(g->gpu_data, new_element, ptr);
}

/* If an eviction was refused, we will try to evict it if possible at the next eviction call. To do this we retrieve the data here. */
static void darts_victim_eviction_failed(starpu_data_handle_t victim, void *component)
{
	(void)component;
	_REFINED_MUTEX_LOCK();
	_LINEAR_MUTEX_LOCK();

#ifdef STARPU_DARTS_STATS
	struct timeval time_start_evicted;
	gettimeofday(&time_start_evicted, NULL);
	victim_evicted_compteur++;
#endif

	int current_gpu; /* Index in tabs of structs */
	if (cpu_only == 1)
	{
		current_gpu = 0;
	}
	else if (cpu_only == 2)
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	}
	else
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;
	}

	tab_gpu_planned_task[current_gpu].data_to_evict_next = victim;

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_evicted;
	gettimeofday(&time_end_evicted, NULL);
	time_total_evicted += (time_end_evicted.tv_sec - time_start_evicted.tv_sec)*1000000LL + time_end_evicted.tv_usec - time_start_evicted.tv_usec;
#endif

	_REFINED_MUTEX_UNLOCK();
	_LINEAR_MUTEX_UNLOCK();
}

/* Applying Belady on tasks from pulled task. */
static starpu_data_handle_t _starpu_darts_belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct _starpu_darts_gpu_pulled_task *g)
{
#ifdef STARPU_DARTS_STATS
	struct timeval time_start_belady;
	gettimeofday(&time_start_belady, NULL);
#endif
	int index_next_use = 0;
	int max_next_use = -1;
	starpu_data_handle_t returned_handle = NULL;

	int i;
	for (i = 0; i < nb_data_on_node; i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES_FROM_DATA(data_tab[i]);
		if (starpu_data_can_evict(data_tab[i], node, is_prefetch)) /* TODO: This could be replaced by just looking in a tab of int as we already call this function on all data in victim_selector. */
		{
			index_next_use = 0;
			struct _starpu_darts_pulled_task *p;
			for (p = _starpu_darts_pulled_task_list_begin(g->ptl); p != _starpu_darts_pulled_task_list_end(g->ptl); p = _starpu_darts_pulled_task_list_next(p))
			{
				unsigned j;
				for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
				{
					STARPU_IGNORE_UTILITIES_HANDLES(p->pointer_to_pulled_task, j);
					index_next_use++;
					if (STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j) == data_tab[i])
					{
						if (max_next_use < index_next_use)
						{
							max_next_use = index_next_use;
							returned_handle = data_tab[i];
						}
						goto break_nested_for_loop;
					}
				}
			}
			break_nested_for_loop : ;
		}
    }

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_belady;
	gettimeofday(&time_end_belady, NULL);
	time_total_belady += (time_end_belady.tv_sec - time_start_belady.tv_sec)*1000000LL + time_end_belady.tv_usec - time_start_belady.tv_usec;
#endif

	return returned_handle;
}

static starpu_data_handle_t _starpu_darts_least_used_data_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, int *nb_task_in_pulled_task, int current_gpu)
{
#ifdef STARPU_DARTS_STATS
	struct timeval time_start_least_used_data_planned_task;
	gettimeofday(&time_start_least_used_data_planned_task, NULL);
#endif

	int min_nb_task_in_planned_task = INT_MAX;
	starpu_data_handle_t returned_handle = NULL;

	struct _starpu_darts_handle_user_data *hud = malloc(sizeof(struct _starpu_darts_handle_user_data));
	int i;
	for (i = 0; i < nb_data_on_node; i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES_FROM_DATA(data_tab[i]);
		if (nb_task_in_pulled_task[i] == 0)
		{
			hud = data_tab[i]->user_data;

			if (hud->nb_task_in_planned_task[current_gpu] < min_nb_task_in_planned_task)
			{
				min_nb_task_in_planned_task = hud->nb_task_in_planned_task[current_gpu];
				returned_handle = data_tab[i];
			}
		}
	}

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_least_used_data_planned_task;
	gettimeofday(&time_end_least_used_data_planned_task, NULL);
	time_total_least_used_data_planned_task += (time_end_least_used_data_planned_task.tv_sec - time_start_least_used_data_planned_task.tv_sec)*1000000LL + time_end_least_used_data_planned_task.tv_usec - time_start_least_used_data_planned_task.tv_usec;
#endif

	return returned_handle;
}

/* Return a data to evict following Least Used in the Future eviction policy.
 * This function is called each time a PU's memory is full and it needs to load a data, either as a fetch or prefetch. */
static starpu_data_handle_t darts_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{
	(void)toload;
	_REFINED_MUTEX_LOCK();
	_LINEAR_MUTEX_LOCK();

#ifdef STARPU_DARTS_STATS
	struct timeval time_start_selector;
	victim_selector_compteur++;
	gettimeofday(&time_start_selector, NULL);
#endif

	int current_gpu; /* Index in tabs of structs */
	if (cpu_only == 1)
	{
		current_gpu = 0;
	}
	else if (cpu_only == 2)
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	}
	else
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;
	}

	/* If an eviction was refused we try to evict it again. */
	if (tab_gpu_planned_task[current_gpu].data_to_evict_next != NULL)
	{
		starpu_data_handle_t temp_handle = tab_gpu_planned_task[current_gpu].data_to_evict_next;
		tab_gpu_planned_task[current_gpu].data_to_evict_next = NULL;

#ifdef STARPU_DARTS_STATS
		struct timeval time_end_selector;
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
#endif

		if (!starpu_data_is_on_node(temp_handle, node))
		{
			_STARPU_SCHED_PRINT("Refused %p is not on node %d. Restart eviction\n", temp_handle, node);
#ifdef STARPU_DARTS_STATS
			victim_selector_refused_not_on_node++;
#endif

			goto debuteviction;
		}
		if (!starpu_data_can_evict(temp_handle, node, is_prefetch))
		{
			_STARPU_SCHED_PRINT("Refused data can't be evicted. Restart eviction selection.\n");
#ifdef STARPU_DARTS_STATS
			victim_selector_refused_cant_evict++;
#endif
			goto debuteviction;
		}

		_STARPU_SCHED_PRINT("Evict refused data %p for GPU %d.\n", temp_handle, current_gpu);
#ifdef STARPU_DARTS_STATS
		victim_selector_return_refused++;
#endif

		_REFINED_MUTEX_UNLOCK();
		_LINEAR_MUTEX_UNLOCK();

		return temp_handle;
    }

 debuteviction: ;

	/* Getting the set of data on node. */
	starpu_data_handle_t *data_on_node;
	unsigned nb_data_on_node = 0;
	int *valid;
	starpu_data_handle_t returned_handle = STARPU_DATA_NO_VICTIM;
	starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);

	int min_number_task_in_pulled_task = INT_MAX;
	int nb_task_in_pulled_task[nb_data_on_node];

	unsigned i;
	for (i = 0; i < nb_data_on_node; i++)
	{
		nb_task_in_pulled_task[i] = 0;
	}

	/* Compute the number of task in pulled_task associated with each data. */
	for (i = 0; i < nb_data_on_node; i++)
	{
		STARPU_IGNORE_UTILITIES_HANDLES_FROM_DATA(data_on_node[i]);
		if (starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		{
			struct _starpu_darts_handle_user_data *hud = data_on_node[i]->user_data;
			nb_task_in_pulled_task[i] = hud->nb_task_in_pulled_task[current_gpu];
			_STARPU_SCHED_PRINT("%d task in pulled_task for %p.\n", hud->nb_task_in_pulled_task[current_gpu], data_on_node[i]);

			if (hud->nb_task_in_pulled_task[current_gpu] == 0 && hud->nb_task_in_planned_task[current_gpu] == 0)
			{
#ifdef STARPU_DARTS_STATS
				victim_selector_return_data_not_in_planned_and_pulled++;
#endif
				_STARPU_SCHED_PRINT("%d task in planned task as well for %p.\n", hud->nb_task_in_pulled_task[current_gpu], data_on_node[i]);
				returned_handle = data_on_node[i];
				goto deletion_in_victim_selector;
			}

			if (hud->nb_task_in_pulled_task[current_gpu] < min_number_task_in_pulled_task)
			{
				min_number_task_in_pulled_task = hud->nb_task_in_pulled_task[current_gpu];
			}
		}
		else
		{
			/* - 1 means that the associated data in the tab of data cannot be evicted. */
			nb_task_in_pulled_task[i] = -1;
		}
	}

	if (min_number_task_in_pulled_task == INT_MAX)
	{
#ifdef STARPU_DARTS_STATS
		struct timeval time_end_selector;
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		victim_selector_return_no_victim++;
#endif
		_STARPU_SCHED_PRINT("Evict NO_VICTIM because min_number_task_in_pulled_task == INT_MAX.\n");

		_REFINED_MUTEX_UNLOCK();
		_LINEAR_MUTEX_UNLOCK();

		return STARPU_DATA_NO_VICTIM;
	}
	else if (min_number_task_in_pulled_task == 0)
	{
		/* At least 1 data is not used in pulled_task */
		returned_handle = _starpu_darts_least_used_data_on_planned_task(data_on_node, nb_data_on_node, nb_task_in_pulled_task, current_gpu);
	}
	else /* At least 1 data is necessary in pulled_task */
	{
		/* If a prefetch is requesting the eviction, we return NO_VICTIM because we don't want to favor prefetch over task that are in pulled_task. */
		if (is_prefetch >= 1)
		{
#ifdef STARPU_DARTS_STATS
			struct timeval time_end_selector;
			gettimeofday(&time_end_selector, NULL);
			time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
			victim_selector_return_no_victim++;
#endif
			_STARPU_SCHED_PRINT("Evict NO_VICTIM because is_prefetch >= 1.\n");

			_REFINED_MUTEX_UNLOCK();
			_LINEAR_MUTEX_UNLOCK();

			return STARPU_DATA_NO_VICTIM;
		}

#ifdef STARPU_DARTS_STATS
		victim_selector_belady++;
#endif

		returned_handle = _starpu_darts_belady_on_pulled_task(data_on_node, nb_data_on_node, node, is_prefetch, &tab_gpu_pulled_task[current_gpu]);
	}

	/* TODO: DOes it really happens sometimes ? To check. */
	if (returned_handle == NULL)
	{
#ifdef STARPU_DARTS_STATS
		struct timeval time_end_selector;
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		victim_selector_return_no_victim++;
#endif

		_REFINED_MUTEX_UNLOCK();
		_LINEAR_MUTEX_UNLOCK();

		return STARPU_DATA_NO_VICTIM;
	}

 deletion_in_victim_selector : ;

	struct starpu_sched_component *temp_component = component;
	struct _starpu_darts_sched_data *data = temp_component->data;

	struct starpu_task *task;
	for (task = starpu_task_list_begin(&tab_gpu_planned_task[current_gpu].planned_task); task != starpu_task_list_end(&tab_gpu_planned_task[current_gpu].planned_task); task = starpu_task_list_next(task))
	{
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			STARPU_IGNORE_UTILITIES_HANDLES(task, i);
			if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
			{
				/* Removing task using the handle selected for eviction from the planned_task list. */
				struct _starpu_darts_pointer_in_task *pt = task->sched_data;
				starpu_task_list_erase(&tab_gpu_planned_task[current_gpu].planned_task, pt->pointer_to_cell);

				pt->pointer_to_cell = task;
				pt->pointer_to_D = malloc(get_nbuffer_without_scratch(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
				pt->tud = malloc(get_nbuffer_without_scratch(task)*sizeof(_starpu_darts_task_using_data_new()));

				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					STARPU_IGNORE_UTILITIES_HANDLES(task, i);
					/* Pointer toward the main task list in the handles. */
					struct _starpu_darts_task_using_data *e = _starpu_darts_task_using_data_new();
					e->pointer_to_T = task;

					if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
					{
						struct _starpu_darts_task_using_data_list *tl = _starpu_darts_task_using_data_list_new();
						_starpu_darts_task_using_data_list_push_front(tl, e);
						STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
					}
					else
					{
						_starpu_darts_task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
					}

					/* Adding the pointer in the task. */
					pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
					pt->tud[i] = e;

					/* Increase expected length of task using this data */
					struct _starpu_darts_handle_user_data *hud = pt->pointer_to_D[i]->user_data;
					hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);

					_STARPU_SCHED_PRINT("Eviction of data %p.\n", STARPU_TASK_GET_HANDLE(task, i));

					pt->pointer_to_D[i]->user_data = hud;

				}
				task->sched_data = pt;

				starpu_task_list_push_back(&data->main_task_list, task);
				break;
			}
		}
	}

	/* Pushing the evicted data in datanotusedyet if it is still usefull to some tasks or if we are in a case with dependencies. */
	if (choose_best_data_from == 0)
	{
		if (!_starpu_darts_task_using_data_list_empty(returned_handle->sched_data))
		{
			if (dependances == 1) /* Checking if other PUs have this handle in datanotusedtyet. */
			{
				struct _starpu_darts_handle_user_data *hud = returned_handle->user_data;
				if (hud->is_present_in_data_not_used_yet[current_gpu] == 0)
				{
					push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[current_gpu], current_gpu);
				}

				int x;
				for (x = 0; x < _nb_gpus; x++)
				{
					if (x != current_gpu)
					{
						if (hud->is_present_in_data_not_used_yet[x] == 0 && (can_a_data_be_in_mem_and_in_not_used_yet == 1 || !starpu_data_is_on_node(returned_handle, memory_nodes[x])))
						{
							push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[x], i);
						}
					}
				}
			}
			else
			{
				push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[current_gpu], current_gpu);
			}
		}
	}

#ifdef STARPU_DARTS_STATS
	struct timeval time_end_selector;
	gettimeofday(&time_end_selector, NULL);
	time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
#endif
	_STARPU_SCHED_PRINT("Evict %p on GPU %d.\n", returned_handle, current_gpu);

	_REFINED_MUTEX_UNLOCK();
	_LINEAR_MUTEX_UNLOCK();

	return returned_handle;
}

static int darts_can_push(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	int didwork = 0;
	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);
	if (task)
	{
		/* If a task is refused it is pushed it in the refused fifo list of the associated processing unit.
		 * This list is looked at first when a PU is asking for a task. */

		_REFINED_MUTEX_LOCK();
		_LINEAR_MUTEX_LOCK();

		_STARPU_SCHED_PRINT("Refused %p in can_push.\n", task);
#ifdef STARPU_DARTS_STATS
		nb_refused_task++;
#endif

		int current_gpu; /* Index in tabs of structs */
		if (cpu_only == 1)
		{
			current_gpu = 0;
		}
		else if (cpu_only == 2)
		{
			current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
		}
		else
		{
			current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;
		}

		starpu_task_list_push_back(&tab_gpu_planned_task[current_gpu].refused_fifo_list, task);

		_REFINED_MUTEX_UNLOCK();
		_LINEAR_MUTEX_UNLOCK();
	}

	/* There is room now */
	return didwork || starpu_sched_component_can_push(component, to);
}

static int darts_can_pull(struct starpu_sched_component *component)
{
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_darts_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	/* Global variables extracted from environement variables. */
	eviction_strategy_darts = starpu_get_env_number_default("STARPU_DARTS_EVICTION_STRATEGY_DARTS", 1);
	threshold = starpu_get_env_number_default("STARPU_DARTS_THRESHOLD", 0);
	app = starpu_get_env_number_default("STARPU_DARTS_APP", 1);
	choose_best_data_from = starpu_get_env_number_default("STARPU_DARTS_CHOOSE_BEST_DATA_FROM", 0);
	simulate_memory = starpu_get_env_number_default("STARPU_DARTS_SIMULATE_MEMORY", 0);
	task_order = starpu_get_env_number_default("STARPU_DARTS_TASK_ORDER", 2);
	data_order = starpu_get_env_number_default("STARPU_DARTS_DATA_ORDER", 2);
	dependances = starpu_get_env_number_default("STARPU_DARTS_DEPENDANCES", 1);
	prio = starpu_get_env_number_default("STARPU_DARTS_PRIO", 1);
	free_pushed_task_position = starpu_get_env_number_default("STARPU_DARTS_FREE_PUSHED_TASK_POSITION", 1);
	graph_descendants = starpu_get_env_number_default("STARPU_DARTS_GRAPH_DESCENDANTS", 0);
	dopt_selection_order = starpu_get_env_number_default("STARPU_DARTS_DOPT_SELECTION_ORDER", 7);
	highest_priority_task_returned_in_default_case = starpu_get_env_number_default("STARPU_DARTS_HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE", 1);
	can_a_data_be_in_mem_and_in_not_used_yet = starpu_get_env_number_default("STARPU_DARTS_CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET", 0);
	push_free_task_on_gpu_with_least_task_in_planned_task = starpu_get_env_number_default("STARPU_DARTS_PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK", 2);

	if (starpu_cpu_worker_get_count() > 0 && starpu_cuda_worker_get_count() == 0 && starpu_hip_worker_get_count() == 0 && starpu_opencl_worker_get_count() == 0 && starpu_mpi_ms_worker_get_count() == 0 && starpu_tcpip_ms_worker_get_count() == 0)
	{
		cpu_only = 1; // Only CPUs
	}
	else if (starpu_cpu_worker_get_count() > 0 && (starpu_cuda_worker_get_count() > 0 || starpu_hip_worker_get_count() == 0 || starpu_opencl_worker_get_count() == 0 || starpu_mpi_ms_worker_get_count() == 0 || starpu_tcpip_ms_worker_get_count() == 0))
	{
		cpu_only = 2; // Both GPUs and CPUs
	}
	else
	{
		cpu_only = 0; // Only GPUs
	}

	_STARPU_SCHED_PRINT("-----\nSTARPU_DARTS_EVICTION_STRATEGY_DARTS = %d\nSTARPU_DARTS_THRESHOLD = %d\nSTARPU_DARTS_APP = %d\nSTARPU_DARTS_CHOOSE_BEST_DATA_FROM = %d\nSTARPU_DARTS_SIMULATE_MEMORY = %d\nSTARPU_DARTS_TASK_ORDER = %d\nSTARPU_DARTS_DATA_ORDER = %d\nSTARPU_DARTS_DEPENDANCES = %d\nSTARPU_DARTS_PRIO = %d\nSTARPU_DARTS_FREE_PUSHED_TASK_POSITION = %d\nSTARPU_DARTS_GRAPH_DESCENDANTS = %d\nSTARPU_DARTS_DOPT_SELECTION_ORDER = %d\nSTARPU_DARTS_HIGHEST_PRIORITY_TASK_RETURNED_IN_DEFAULT_CASE = %d\nSTARPU_DARTS_CAN_A_DATA_BE_IN_MEM_AND_IN_NOT_USED_YET = %d\nSTARPU_DARTS_PUSH_FREE_TASK_ON_GPU_WITH_LEAST_TASK_IN_PLANNED_TASK = %d\n-----\n", eviction_strategy_darts, threshold, app, choose_best_data_from, simulate_memory, task_order, data_order, dependances, prio, free_pushed_task_position, graph_descendants, dopt_selection_order, highest_priority_task_returned_in_default_case, can_a_data_be_in_mem_and_in_not_used_yet, push_free_task_on_gpu_with_least_task_in_planned_task);

	_nb_gpus = _get_number_GPU();
	if (cpu_only == 2)
	{
		_nb_gpus += starpu_memory_nodes_get_count_by_kind(STARPU_CPU_RAM); /* Adding one to account for the NUMA node because we are in a CPU+GPU case. */
	}
	NT_DARTS = 0;
	new_tasks_initialized = false;
	round_robin_free_task = -1; /* Starts at -1 because it is updated at the beginning and not the end of push_task. Thus to start at 0 on the first task you need to init it at -1. */

	/* Initialize memory node of each GPU or CPU */
	memory_nodes = malloc(sizeof(int)*_nb_gpus);
	int i;
	for (i = 0; i < _nb_gpus; i++)
	{
		if (cpu_only == 0) /* Using GPUs so memory nodes are 1->Ngpu */
		{
			memory_nodes[i] = i + 1;
		}
		else if (cpu_only == 2) /* Using GPUs and CPUs so memory nodes are 0->Ngpu+Ncpu */
		{
			memory_nodes[i] = i;
		}
		else /* Using CPUs so memory nodes are 0->N_numa_nodes */
		{
			memory_nodes[i] = i;
		}
	}

#ifdef STARPU_DARTS_STATS
	int size = strlen(_output_directory) + strlen("/Data_DARTS_data_chosen_stats_GPU_.csv") + 3;
	for (i = 0; i < _nb_gpus; i++)
	{
		char path[size];
		snprintf(path, size, "%s%s%d%s", _output_directory, "/Data_DARTS_data_chosen_stats_GPU_", i+1, ".csv");
		FILE *f = fopen(path, "w");
		STARPU_ASSERT_MSG(f, "cannot open file <%s>\n", path);
		fprintf(f, "Data selection,Data chosen,Number of data read,Number of task added in planned_task\n");
		fclose(f);
	}

	gettimeofday(&time_start_createtolasttaskfinished, NULL);
	nb_return_null_after_scheduling = 0;
	nb_return_task_after_scheduling = 0;
	nb_data_selection = 0;
	nb_return_null_because_main_task_list_empty = 0;
	nb_new_task_initialized = 0;
	nb_refused_task = 0;
	victim_selector_refused_not_on_node = 0;
	victim_selector_refused_cant_evict = 0;
	victim_selector_return_refused = 0;
	victim_selector_return_unvalid = 0;
	victim_selector_return_data_not_in_planned_and_pulled = 0;
	number_data_conflict = 0;
	number_critical_data_conflict = 0;
	victim_evicted_compteur = 0;
	victim_selector_compteur = 0;
	victim_selector_return_no_victim = 0;
	victim_selector_belady = 0;
	number_random_selection = 0;
	nb_free_choice = 0;
	nb_1_from_free_choice = 0;
	nb_data_selection_per_index = 0;
	nb_task_added_in_planned_task = 0;
	nb_1_from_free_task_not_found = 0;
	time_total_selector = 0;
	time_total_evicted = 0;
	time_total_belady = 0;
	time_total_schedule = 0;
	time_total_choose_best_data = 0;
	time_total_fill_planned_task_list = 0;
	time_total_initialisation = 0;
	time_total_randomize = 0;
	time_total_pick_random_task = 0;
	time_total_least_used_data_planned_task = 0;
	time_total_createtolasttaskfinished = 0;
	data_choice_per_index = false;
#endif

	_sched_visu_init(_nb_gpus);

	struct starpu_sched_component *component = starpu_sched_component_create(tree, "darts");
	starpu_srand48(starpu_get_env_number_default("SEED", 0));

	struct _starpu_darts_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->main_task_list);

	tab_gpu_planned_task = malloc(_nb_gpus*sizeof(struct _starpu_darts_gpu_planned_task));
	_starpu_darts_tab_gpu_planned_task_init();
	tab_gpu_pulled_task = malloc(_nb_gpus*sizeof(struct _starpu_darts_gpu_pulled_task));
	_starpu_darts_tab_gpu_pulled_task_init();

	_REFINED_MUTEX_INIT();
	_LINEAR_MUTEX_INIT();

	Dopt = calloc(_nb_gpus, sizeof(starpu_data_handle_t));
	data_conflict = malloc(_nb_gpus*sizeof(bool));

	component->data = data;
	component->push_task = darts_push_task;
	component->pull_task = darts_pull_task;
	component->can_push = darts_can_push;
	component->can_pull = darts_can_pull;

	if (eviction_strategy_darts == 1)
	{
		starpu_data_register_victim_selector(darts_victim_selector, darts_victim_eviction_failed, component);
	}

	return component;
}

static void initialize_darts_center_policy(unsigned sched_ctx_id)
{
	_output_directory = _sched_visu_get_output_directory();
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_darts_create, NULL,
							   STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
							   STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY_FIRST |
							   STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);


	/* To avoid systematic prefetch in sched_policy.c */
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	sched_ctx->sched_policy->prefetches = 1;

	perf_arch = starpu_worker_get_perf_archtype(0, sched_ctx_id); /* Getting the perfmodel. Used to get the expected length of a task to tiebreak when choosing Dopt. We use 0 here because we assume all processing units to be homogeneous. TODO: use the mean performance of each processing unit or each time the performance model is needed, use the one of the selected processing unit to get the exepected task length or data transfer duration. */

	if (prio != 0)
	{
		if (graph_descendants != 0)
		{
			_starpu_graph_record = 1;
		}

		/* To get the priority of each task. */
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
	}
}

static void deinitialize_darts_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

/* Get the task that was last executed. Used to update the task list of pulled task. */
static void get_task_done(struct starpu_task *task, unsigned sci)
{
	_LINEAR_MUTEX_LOCK();

	int current_gpu;
	if (cpu_only == 1)
	{
		current_gpu = 0;
	}
	else if (cpu_only == 2)
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	}
	else
	{
		current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()) - 1;
	}

	if (eviction_strategy_darts == 1)
	{
		_REFINED_MUTEX_LOCK();
		unsigned i;
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			STARPU_IGNORE_UTILITIES_HANDLES(task, i);
			struct _starpu_darts_handle_user_data *hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->nb_task_in_pulled_task[current_gpu] -= 1;
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		_REFINED_MUTEX_UNLOCK();
	}

	int trouve = 0;

	if (!_starpu_darts_pulled_task_list_empty(tab_gpu_pulled_task[current_gpu].ptl))
	{
		struct _starpu_darts_pulled_task *temp;
		for (temp = _starpu_darts_pulled_task_list_begin(tab_gpu_pulled_task[current_gpu].ptl); temp != _starpu_darts_pulled_task_list_end(tab_gpu_pulled_task[current_gpu].ptl); temp = _starpu_darts_pulled_task_list_next(temp))
		{
			if (temp->pointer_to_pulled_task == task)
			{
				trouve = 1;
				break;
			}
		}
		if (trouve == 1)
		{
			_starpu_darts_pulled_task_list_erase(tab_gpu_pulled_task[current_gpu].ptl, temp);

			if (cpu_only == 0)
			{
				_starpu_darts_pulled_task_delete(temp);
			}
		}
	}

	_LINEAR_MUTEX_UNLOCK();

	starpu_sched_component_worker_pre_exec_hook(task, sci);
}

struct starpu_sched_policy _starpu_sched_darts_policy =
{
	.init_sched = initialize_darts_center_policy,
	.deinit_sched = deinitialize_darts_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	//~ .pop_task = _sched_visu_get_data_to_load, /* Modified from starpu_sched_tree_pop_task */
	.pop_task = starpu_sched_tree_pop_task, /* Modified from starpu_sched_tree_pop_task */
	//~ .pre_exec_hook = _sched_visu_get_current_tasks, /* Modified from starpu_sched_component_worker_pre_exec_hookstarpu_sched_component_worker_pre_exec_hook */
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook, /* Modified from starpu_sched_component_worker_pre_exec_hook */
	.reset_scheduler = starpu_darts_reinitialize_structures,
	.post_exec_hook = get_task_done,
	.policy_name = "darts",
	.policy_description = "Dynamic scheduler that select data in order to maximize the amount of task that can be computed without any additional data load",
	.worker_type = STARPU_WORKER_LIST,
};
