/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2022	Maxime Gonthier
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

/* Dynamic Data Aware scheduling. Look for the "best" data.
 * Computes all task using this data and the data already loaded on memory.
 * If no task is available compute a random task not computed yet.
 */

#include <schedulers/HFP.h>
#include <schedulers/dynamic_data_aware.h>
#include "helper_mct.h"
#include <starpu_data_maxime.h> /* pour l'appel de la fonction qui reinit a la nouvelle itération */
#include <graph.h> /* Pour compute_descendants */

/* Var globales déclaré en extern */
int eviction_strategy_dynamic_data_aware;
int threshold;
int app;
int choose_best_data_from;
int simulate_memory;
int task_order;
int data_order;
//~ int erase_data_strategy;
int prio;
int free_pushed_task_position;
int dependances; /* Utile pour les ordres de données et le push back de données dans datanotusedyet. */
int graph_descendants;
int dopt_selection_order;

bool gpu_memory_initialized;
bool new_tasks_initialized;
//~ struct gpu_planned_task_control *my_planned_task_control;
//~ struct gpu_pulled_task_control *my_pulled_task_control;
struct gpu_planned_task *tab_gpu_planned_task;
struct gpu_pulled_task *tab_gpu_pulled_task;
//~ int number_task_out_DARTS; /* Utile pour savoir quand réinit quand il y a plusieurs itérations. */
//~ int number_task_out_DARTS_2; /* Utile pour savoir quand réinit quand il y a plusieurs itérations. */
int NT_DARTS;
int iteration_DARTS;
struct starpu_perfmodel_arch* perf_arch;

#ifdef PRINT_STATS
/* Pour les compteurs. */
int nb_return_null_after_scheduling;
int nb_return_task_after_scheduling;
int nb_return_null_because_main_task_list_empty;
int nb_new_task_initialized;
int nb_refused_task;
int victim_selector_refused_not_on_node;
int victim_selector_refused_cant_evict;
int victim_selector_return_refused;
int victim_selector_return_unvalid;
int victim_selector_return_data_not_in_planned_and_pulled;
int number_data_conflict;
int number_critical_data_conflict;
int victim_evicted_compteur;
int victim_selector_compteur;
int victim_selector_return_no_victim;
int victim_selector_belady;
int nb_1_from_free_task_not_found;
int number_random_selection;
int nb_free_choice;
int nb_1_from_free_choice;
int nb_data_selection_per_index;
int nb_task_added_in_planned_task;
bool data_choice_per_index;
int nb_data_selection;
/* Pour la mesure du temps. */
struct timeval time_start_selector;
struct timeval time_end_selector;
long long time_total_selector;
struct timeval time_start_evicted;
struct timeval time_end_evicted;
long long time_total_evicted;
struct timeval time_start_belady;
struct timeval time_end_belady;
long long time_total_belady;
struct timeval time_start_schedule;
struct timeval time_end_schedule;
long long time_total_schedule;
struct timeval time_start_choose_best_data;
struct timeval time_end_choose_best_data;
long long time_total_choose_best_data;
struct timeval time_start_fill_planned_task_list;
struct timeval time_end_fill_planned_task_list;
long long time_total_fill_planned_task_list;
struct timeval time_start_initialisation;
struct timeval time_end_initialisation;
long long time_total_initialisation;
struct timeval time_start_randomize;
struct timeval time_end_randomize;
long long time_total_randomize;
struct timeval time_start_pick_random_task;
struct timeval time_end_pick_random_task;
long long time_total_pick_random_task;
struct timeval time_start_least_used_data_planned_task;
struct timeval time_end_least_used_data_planned_task;
long long time_total_least_used_data_planned_task;
struct timeval time_start_createtolasttaskfinished;
struct timeval time_end_createtolasttaskfinished;
long long time_total_createtolasttaskfinished;
#endif

/* TODO : a suppr */
int total_task_done;

/* GRAPHE */
static void set_priority(void *_data, struct _starpu_graph_node *node)
{
	//~ struct _starpu_graph_test_policy_data *data = _data;
	//~ struct dynamic_data_aware_sched_data *data = _data;
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	starpu_worker_relax_off();
	struct _starpu_job *job = node->job;

	if (job)
	{
		//~ if (data->descendants)
		//~ {
			job->task->priority = node->descendants;
			printf("Descendants of job %p (%s): %d\n", job->task, starpu_task_get_name(job->task), job->task->priority);
		//~ }
		//~ else
		//~ {
			//~ job->task->priority = node->depth;
			//~ printf("Depth of job %p (%s): %d\n", job->task, starpu_task_get_name(job->task), job->task->priority);
		//~ }	
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}

/* Fonction appellé directement dans l'application à la fin d'une itération. Je l'ai fais pour Cholesky et GEMM faut le faire pour de futures applications. */
void new_iteration()
{
	/* Printing stats in files. Préciser PRINT_N dans les var d'env. */	
	#ifdef PRINT
	printf("############### Itération n°%d ###############\n", iteration_DARTS + 1); fflush(stdout);
	#endif
		
	#ifdef PRINT_STATS
	printf("Nb \"random\" task selection: %d\n", number_random_selection);
	printf("Nb \"index\" data selection: %d/%d\n", nb_data_selection_per_index, nb_data_selection);
	if (iteration_DARTS == 11 || starpu_get_env_number_default("PRINT_TIME", 0) == 2) /* PRINT_TIME = 2 pour quand on a 1 seule itération. */
	{
		FILE *f_new_iteration = fopen("Output_maxime/Data/DARTS/Nb_conflit_donnee.csv", "a");
		fprintf(f_new_iteration , "%d,%d,%d\n", print_n, number_data_conflict/11 + number_data_conflict%11, number_critical_data_conflict/11 + number_critical_data_conflict%11);
		fclose(f_new_iteration);

		gettimeofday(&time_end_createtolasttaskfinished, NULL);
		time_total_createtolasttaskfinished += (time_end_createtolasttaskfinished.tv_sec - time_start_createtolasttaskfinished.tv_sec)*1000000LL + time_end_createtolasttaskfinished.tv_usec - time_start_createtolasttaskfinished.tv_usec;

		f_new_iteration = fopen("Output_maxime/Data/DARTS/DARTS_time.csv", "a");
		fprintf(f_new_iteration, "%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", print_n, time_total_selector/11 + time_total_selector%11, time_total_evicted/11 + time_total_evicted%11, time_total_belady/11 + time_total_belady%11, time_total_schedule/11 + time_total_schedule%11, time_total_choose_best_data/11 + time_total_choose_best_data%11, time_total_fill_planned_task_list/11 + time_total_fill_planned_task_list%11, time_total_initialisation/11 + time_total_initialisation%11, time_total_randomize/11 + time_total_randomize%11, time_total_pick_random_task/11 + time_total_pick_random_task%11, time_total_least_used_data_planned_task/11 + time_total_least_used_data_planned_task%11, time_total_createtolasttaskfinished/11 + time_total_createtolasttaskfinished%11);
		fclose(f_new_iteration);
		
		f_new_iteration = fopen("Output_maxime/Data/DARTS/Choice_during_scheduling.csv", "a");
		fprintf(f_new_iteration, "%d,%d,%d,%d,%d,%d,%d,%d\n", print_n, nb_return_null_after_scheduling/11 + nb_return_null_after_scheduling%11, nb_return_task_after_scheduling/11 + nb_return_task_after_scheduling%11, nb_return_null_because_main_task_list_empty/11 + nb_return_null_because_main_task_list_empty%11, number_random_selection/11 + number_random_selection%11, nb_1_from_free_task_not_found/11 + nb_1_from_free_task_not_found%11, nb_free_choice/11 + nb_free_choice%11, nb_1_from_free_choice/11 + nb_1_from_free_choice%11);
		fclose(f_new_iteration);
		
		f_new_iteration = fopen("Output_maxime/Data/DARTS/Choice_victim_selector.csv", "a");
		fprintf(f_new_iteration, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", print_n, victim_selector_refused_not_on_node/11 + victim_selector_refused_not_on_node%11, victim_selector_refused_cant_evict/11 + victim_selector_refused_cant_evict%11, victim_selector_return_refused/11 + victim_selector_return_refused%11, victim_selector_return_unvalid/11 + victim_selector_return_unvalid%11, victim_selector_return_data_not_in_planned_and_pulled/11 + victim_selector_return_data_not_in_planned_and_pulled%11, victim_evicted_compteur/11 + victim_evicted_compteur%11, victim_selector_compteur/11 + victim_selector_compteur%11, victim_selector_return_no_victim/11 + victim_selector_return_no_victim%11, victim_selector_belady/11 + victim_selector_belady%11);
		fclose(f_new_iteration);
		
		f_new_iteration = fopen("Output_maxime/Data/DARTS/Misc.csv", "a");
		fprintf(f_new_iteration, "%d,%d,%d\n", print_n, nb_refused_task/11 + nb_refused_task%11, nb_new_task_initialized/11 + nb_new_task_initialized%11);
		fclose(f_new_iteration);
	}
	#endif
	
	iteration_DARTS++; /* Variable globale qui sert à ré-init les données et tâches. */

	/* Re-init of planned task struct containing datanotused and other things. */
	//~ int i = 0;
	//~ free(my_planned_task_control);
	free(tab_gpu_planned_task);
	tab_gpu_planned_task = malloc(Ngpu*sizeof(struct gpu_planned_task));
	tab_gpu_planned_task_init();
	
	total_task_done = 0;
	
	//~ gpu_planned_task_initialisation();
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
		//~ gpu_planned_task_insertion();
	//~ }
	//~ my_planned_task_control->first = my_planned_task_control->pointer;

	/* TODO : Utile ? On pourrait free le tableau de pulled_task non ? */
	//~ free(my_pulled_task_control);
	//~ gpu_pulled_task_initialisation();
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
		//~ gpu_pulled_task_insertion();
	//~ }
	//~ my_pulled_task_control->first = my_pulled_task_control->pointer;
}

void print_task_info(struct starpu_task* task)
{
	int i = 0;
	printf("Task %p has %d data:", task, STARPU_TASK_GET_NBUFFERS(task));
	
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		printf(" %p", STARPU_TASK_GET_HANDLE(task, i));
	}
	printf("\n");
}

void print_task_list(struct starpu_task_list *l, char *s)
{
    int i = 0;
    printf("%s :\n", s); fflush(stdout);
    for (struct starpu_task *task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
    {
		printf("%p (prio: %d):", task, task->priority); fflush(stdout);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			printf("	%p", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
		}
		printf("\n"); fflush(stdout);
    }
}

void print_data_not_used_yet()
{
    int i = 0;
    
    for (i = 0; i < Ngpu; i++)
    {
		printf("On GPU %d, there are %d data not used yet:", i, gpu_data_not_used_list_size(tab_gpu_planned_task[i].gpu_data)); fflush(stdout);
		for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(tab_gpu_planned_task[i].gpu_data); e != gpu_data_not_used_list_end(tab_gpu_planned_task[i].gpu_data); e = gpu_data_not_used_list_next(e))
		{
			printf(" %p", e->D); fflush(stdout);
		}
		printf("\n"); fflush(stdout);
    }
    printf("\n"); fflush(stdout);
}

void print_planned_task_one_gpu(struct gpu_planned_task *g, int current_gpu)
{
    struct starpu_task *task = NULL;
    
    if (starpu_task_list_empty(&tab_gpu_planned_task[current_gpu - 1].planned_task))
    {
		printf("GPU %d's planned task list is empty.\n\n", current_gpu);
	}
	else
	{
		printf("Planned task for GPU %d:\n", current_gpu);
		for (task = starpu_task_list_begin(&g->planned_task); task != starpu_task_list_end(&g->planned_task); task = starpu_task_list_next(task))
		{
			printf("%p\n", task);
		}
		printf("\n");
	}
}

void print_planned_task_all_gpu()
{
    struct starpu_task *task = NULL;
    int i = 0;
    //~ struct gpu_planned_task *temp_pointer = my_planned_task_control->first;
    for (i = 0; i < Ngpu; i++)
    {
		if (starpu_task_list_empty(&tab_gpu_planned_task[i].planned_task))
		{
			printf("GPU %d's planned task list is empty.\n", i + 1); fflush(stdout);
		}
		else
		{
			printf("Planned task for GPU %d:\n", i + 1); fflush(stdout);
			for (task = starpu_task_list_begin(&tab_gpu_planned_task[i].planned_task); task != starpu_task_list_end(&tab_gpu_planned_task[i].planned_task); task = starpu_task_list_next(task))
			{
				printf("%p\n", task); fflush(stdout);
			}
			printf("\n"); fflush(stdout);
		}
		//~ temp_pointer = temp_pointer->next;
	}
}

void print_pulled_task_all_gpu()
{
    int i = 0;
    int j = 0;
    //~ struct gpu_pulled_task *temp_pointer = my_pulled_task_control->first;
	struct pulled_task *p = pulled_task_new();
    for (i = 0; i < Ngpu; i++)
    {
		if (pulled_task_list_empty(tab_gpu_pulled_task[i].ptl))
		{
			printf("GPU %d's pulled task list is empty.\n", i + 1); fflush(stdout);
		}
		printf("Pulled task for GPU %d:\n", i + 1); fflush(stdout);
		for (p = pulled_task_list_begin(tab_gpu_pulled_task[i].ptl); p != pulled_task_list_end(tab_gpu_pulled_task[i].ptl); p = pulled_task_list_next(p))
		{
			printf("%p :", p->pointer_to_pulled_task); fflush(stdout);
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
			{
				printf(" %p", STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j));
			}
			printf("\n");
		}
	}
}

void print_pulled_task_one_gpu(struct gpu_pulled_task *g, int current_gpu)
{
    struct pulled_task *p = pulled_task_new();
    
    printf("Pulled task for GPU %d:\n", current_gpu); fflush(stdout);
    for (p = pulled_task_list_begin(tab_gpu_pulled_task[current_gpu - 1].ptl); p != pulled_task_list_end(tab_gpu_pulled_task[current_gpu - 1].ptl); p = pulled_task_list_next(p))
    {
		printf("%p\n", p->pointer_to_pulled_task); fflush(stdout);
    }
}

void print_data_not_used_yet_one_gpu(struct gpu_planned_task *g, int current_gpu)
{
    printf("Data not used yet on GPU %d are:\n", current_gpu); fflush(stdout);
    //~ for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(tab_gpu_planned_task[current_gpu].gpu_data); e != gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data); e = gpu_data_not_used_list_next(e))
    if (g->gpu_data != NULL)
    {
    for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
    {
		printf(" %p", e->D); fflush(stdout);
    }}
    printf("\n"); fflush(stdout);
}

void check_double_in_data_not_used_yet(struct gpu_planned_task *g, int current_gpu)
{
	//~ printf("Careful you are using check_double_in_data_not_used_yet it cost time!\n");
    for (struct gpu_data_not_used *e1 = gpu_data_not_used_list_begin(tab_gpu_planned_task[current_gpu].gpu_data); e1 != gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data); e1 = gpu_data_not_used_list_next(e1))
    {
		for (struct gpu_data_not_used *e2 = gpu_data_not_used_list_next(e1); e2 != gpu_data_not_used_list_end(tab_gpu_planned_task[current_gpu].gpu_data); e2 = gpu_data_not_used_list_next(e2))
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

void print_task_using_data(starpu_data_handle_t d)
{
    printf("Task(s) using %p are:", d);
    for (struct task_using_data *t = task_using_data_list_begin(d->sched_data); t != task_using_data_list_end(d->sched_data); t = task_using_data_list_next(t))
    {
		printf(" %p", t->pointer_to_T);
    }
    printf("\n\n");
}

/* Print data loaded on a node */
void print_data_on_node(unsigned node)
{
    starpu_data_handle_t *data_on_node;
    unsigned nb_data_on_node = 0;
    int *valid;
    starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);

    int i = 0;
    printf("Data on node %d are:", node);
    for (i = 0; i < nb_data_on_node; i++)
    {
		printf(" %p", data_on_node[i]);
    }
    printf("\n");
}

void print_nb_task_in_list_one_data_one_gpu(starpu_data_handle_t d, int current_gpu)
{
	struct handle_user_data * hud = d->user_data;
	printf("Number of task used by %p in the tasks list:", d);
	printf("pulled_task = %d tasks | planned_tasks = %d tasks.\n", hud->nb_task_in_pulled_task[current_gpu - 1], hud->nb_task_in_planned_task[current_gpu - 1]);
}

bool is_my_task_free(int current_gpu, struct starpu_task *task)
{
	struct handle_user_data * hud = NULL;
	int i = 0;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		if (simulate_memory == 0)
		{
			if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), current_gpu))
			{
				return false;
			}
		}
		else if (simulate_memory == 1)
		{
			if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
			{
				return false;
			}
			hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
			{
				return false;
			}
		}
	}
	return true;
}

/* Pushing the tasks. Each time a new task enter here, we initialize it. */		
static int dynamic_data_aware_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	int i = 0;
	int j = 0;
		
	#ifdef PRINT
	printf("New task %p (%s, prio: %d, length: %f) in push_task with data(s):", task, starpu_task_get_name(task), task->priority, starpu_task_expected_length(task, perf_arch, 0)); fflush(stdout);
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		printf(" %p", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
	}	
	printf("\n"); fflush(stdout);
	#endif
	
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
	#endif
	
	#ifdef PRINT_STATS
	gettimeofday(&time_start_initialisation, NULL);
	#endif
	
	struct dynamic_data_aware_sched_data *data = component->data;
	
	#ifdef PRINT
	for (i = 0; i < Ngpu; i++)
	{
		check_double_in_data_not_used_yet(&tab_gpu_planned_task[i], i);
	}
	#endif
	
	/* Voir si ca marche apres le init ou si ca devrait le remplacer. 
	 * Ne pas faire le if else en dessous si la tache est gratuite. */
	for (i = 0; i < Ngpu; i++) 
	{
		if (is_my_task_free(i + 1, task)) /* +1 cause GPU start at 1, 2, etc ... and you have ram on 0 when checking the memory. */
		{
			#ifdef PRINT
			printf("Task %p is free from push_task\n", task);
			#endif
			
			if (dependances == 1)
			{
				initialize_task_data_gpu_single_task_v3(task, 0);
			}
			else
			{
				initialize_task_data_gpu_single_task_v1(task, 0);
			}
						
			increment_planned_task_data(task, i + 1);	
			struct pointer_in_task *pt = task->sched_data;
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
			{
				if (pt->tud[j] != NULL) 
				{
					task_using_data_list_erase(pt->pointer_to_D[j]->sched_data, pt->tud[j]);
					pt->tud[j] = NULL;
				}
			}
			
			/* Maintenant il faut push cette tâche gratuite dans planned task. On peut le faire au début de la liste ou après la dernière tâche gratuite de planned task. */
			/* Au début */
			if (free_pushed_task_position == 0)
			{
				starpu_task_list_push_front(&tab_gpu_planned_task[i].planned_task, task);
			}
			else
			{
				/* Après la dernière tâche gratuite de planned task. */
				struct starpu_task* checked_task = NULL;
				for (checked_task = starpu_task_list_begin(&tab_gpu_planned_task[i].planned_task); checked_task != starpu_task_list_end(&tab_gpu_planned_task[i].planned_task); checked_task = starpu_task_list_next(checked_task))
				{
					//~ printf("Is task %p in planned_task free ?\n", checked_task);
					//~ print_data_on_node(i + 1);
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(checked_task); j++)
					{
						//~ printf("%p ?\n", STARPU_TASK_GET_HANDLE(checked_task, j));
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(checked_task, j), i + 1))
						{
							starpu_task_list_insert_before(&tab_gpu_planned_task[i].planned_task, task, checked_task);
							
							//~ printf("Insert before task %p that doesn't have data %p.\n", checked_task, STARPU_TASK_GET_HANDLE(checked_task, j));
							//~ print_data_on_node(i + 1);
							//~ print_planned_task_one_gpu(&tab_gpu_planned_task[i], i + 1);
							
							/* End now push task, no push in main task list or GPUs data */		
							starpu_push_task_end(task);
							#ifdef REFINED_MUTEX
							STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
							#endif
							#ifdef LINEAR_MUTEX
							STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
							#endif	
							component->can_pull(component);
							return 0;
						}
					}
				}
				
				/* Else push back */
				starpu_task_list_push_back(&tab_gpu_planned_task[i].planned_task, task);
			}
						
			// add data in other GPUS ? In current GPU ? No but do add pointer from data to task and vice versa.
			
			/* End now push task, no push in main task list or GPUs data */		
			starpu_push_task_end(task);
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			#ifdef LINEAR_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
			#endif	
			component->can_pull(component);
			return 0;
		}
	}
	
	new_tasks_initialized = true; 
		
	if (dependances == 1)
	{
		initialize_task_data_gpu_single_task_v3(task, 1);
	}
	else
	{
		initialize_task_data_gpu_single_task_v1(task, 1);
	}

	#ifdef PRINT_STATS
	gettimeofday(&time_end_initialisation, NULL);
	time_total_initialisation += (time_end_initialisation.tv_sec - time_start_initialisation.tv_sec)*1000000LL + time_end_initialisation.tv_usec - time_start_initialisation.tv_usec;
	#endif
	
	
	/* Pushing the task in sched_list. It's this list that will be randomized
	 * and put in main_task_list in pull_task.
	 */
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
		
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
	#endif
		
	component->can_pull(component);
    return 0;
}

/* Initialize for:
 * tasks -> pointer to the data it uses, pointer to the pointer of task list in the data, 
 * pointer to the cell in the main task list (main_task_list).
 * data -> pointer to the tasks using this data.
 * GPUs -> datas not used yet by this GPU.
 * In the case with dependencies I have to check if a data need to be added again even if it's struct is empty.
 */
 // TODO : je peux simplifier la complexité de cetter version avec celle ci-dessous en comentaires mais ca crash en 2GPU matrice 2D et je ne sais pas pourquoi.

/* V1 */
void initialize_task_data_gpu_single_task_v1(struct starpu_task *task, int also_add_data_in_not_used_yet_list)
{
    int i = 0;
    int j = 0;
    
    if (also_add_data_in_not_used_yet_list == 1)
    {
		/* Adding the data not used yet in all the GPU(s). */
		for (i = 0; i < Ngpu; i++)
		{
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
			{
				struct gpu_data_not_used *e = gpu_data_not_used_new();
				e->D = STARPU_TASK_GET_HANDLE(task, j);
								
				/* Si la donnée a déjà une structure existante */
				if (STARPU_TASK_GET_HANDLE(task, j)->user_data != NULL)
				{					
					struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, j)->user_data;
					
					if (hud->last_iteration_DARTS != iteration_DARTS || hud->is_present_in_data_not_used_yet[i] == 0) /* On est sur une nouvelle itération où la donnée n'y est plus donc on peut l'ajouter */
					{						
						if (data_order == 1)
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].new_gpu_data, e);
						}
						else
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, e);
						}
					}
				}
				else
				{	
					if (data_order == 1)
					{
						gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].new_gpu_data, e);
					}
					else
					{
						gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, e);
					}
				}
			}
		}
	}
    
    /* Adding the pointer in the task. */
    struct pointer_in_task *pt = malloc(sizeof(*pt));
    pt->pointer_to_cell = task;
    pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
    pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
	
    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
    {
		/* Pointer toward the main task list in the handles. */
		struct task_using_data *e = task_using_data_new();
		e->pointer_to_T = task;
		
		/* Adding the task in the list of task using the data */
		if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
		{
			struct task_using_data_list *tl = task_using_data_list_new();
			task_using_data_list_push_front(tl, e);
			STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
		}
		else
		{
			task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
		}
		
		/* Init hud in the data containing a way to track the number of task in 
		 * planned and pulled_task but also a way to check last iteration_DARTS for this data and last check for CHOOSE_FROM_MEM=1
		 * so we don't look twice at the same data. */
		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			struct handle_user_data * hud = malloc(sizeof(*hud));
			hud->last_iteration_DARTS = iteration_DARTS;
			
			/* Need to init them with the number of GPU */
			hud->nb_task_in_pulled_task = malloc(Ngpu*sizeof(int));
			hud->nb_task_in_planned_task = malloc(Ngpu*sizeof(int));
			hud->last_check_to_choose_from = malloc(Ngpu*sizeof(int));
			hud->is_present_in_data_not_used_yet = malloc(Ngpu*sizeof(int));
			hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);
			for (j = 0; j < Ngpu; j++)
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
			struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);
			if (hud->last_iteration_DARTS != iteration_DARTS || hud->is_present_in_data_not_used_yet[i] == 0) /* Re-init values in hud. */
			{
				for (j = 0; j < Ngpu; j++)
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
void initialize_task_data_gpu_single_task_v3(struct starpu_task *task, int also_add_data_in_not_used_yet_list)
{
    int i = 0;
    int j = 0;
    /* 1 elem par liste */
    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
    {
		//~ struct gpu_data_not_used *e = gpu_data_not_used_new();
		//~ e->D = STARPU_TASK_GET_HANDLE(task, i);
		
		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			struct handle_user_data * hud = malloc(sizeof(*hud));
			hud->last_iteration_DARTS = iteration_DARTS;
			hud->nb_task_in_pulled_task = malloc(Ngpu*sizeof(int));
			hud->nb_task_in_planned_task = malloc(Ngpu*sizeof(int));
			hud->last_check_to_choose_from = malloc(Ngpu*sizeof(int));
			hud->is_present_in_data_not_used_yet = malloc(Ngpu*sizeof(int));
			hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);
			
			#ifdef PRINT
			printf("Data is new. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, i), hud->sum_remaining_task_expected_length);
			#endif
			
			for (j = 0; j < Ngpu; j++)
			{
				struct gpu_data_not_used *e = gpu_data_not_used_new();
				e->D = STARPU_TASK_GET_HANDLE(task, i);
		
				hud->nb_task_in_pulled_task[j] = 0;
				hud->nb_task_in_planned_task[j] = 0;
				hud->last_check_to_choose_from[j] = 0;
				hud->is_present_in_data_not_used_yet[j] = 0;
				
				if (also_add_data_in_not_used_yet_list == 1)
				{
					//~ printf("Data is new. Adding %p in data_not_used_yet of GPU %d\n", e->D, j); fflush(stdout);
					hud->is_present_in_data_not_used_yet[j] = 1;
					if (data_order == 1)
					{
						gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
					}
					else
					{
						gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
					}
					//~ print_data_not_used_yet();
				}
			}
			#ifdef PRINT
			printf("%p gets 1 at is_present_in_data_not_used_yet from NULL struct hud\n", STARPU_TASK_GET_HANDLE(task, i)); fflush(stdout);
			#endif
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		else
		{
			struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			//~ hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);
			
			#ifdef PRINT
			printf("New task. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, i), hud->sum_remaining_task_expected_length); fflush(stdout);
			#endif
			
			if (hud->last_iteration_DARTS != iteration_DARTS)
			{
				//~ printf("%d get %d\n", hud->last_iteration_DARTS, iteration_DARTS); fflush(stdout);
				hud->last_iteration_DARTS = iteration_DARTS;
				hud->sum_remaining_task_expected_length = starpu_task_expected_length(task, perf_arch, 0);
				for (j = 0; j < Ngpu; j++)
				{
					struct gpu_data_not_used *e = gpu_data_not_used_new();
					e->D = STARPU_TASK_GET_HANDLE(task, i);
				
					hud->nb_task_in_pulled_task[j] = 0;
					hud->nb_task_in_planned_task[j] = 0;
					hud->last_check_to_choose_from[j] = 0;
					hud->is_present_in_data_not_used_yet[j] = 0;
					
					if (also_add_data_in_not_used_yet_list == 1)
					{
						printf("New iteration. Adding %p in data_not_used_yet of GPU %d\n", e->D, j); fflush(stdout);
						hud->is_present_in_data_not_used_yet[j] = 1;
						if (data_order == 1)
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
						}
						else
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
						}
						print_data_not_used_yet();
					}
				}
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
			else 
			{
				hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);
				for (j = 0; j < Ngpu; j++)
				{
					
					struct gpu_data_not_used *e = gpu_data_not_used_new();
					e->D = STARPU_TASK_GET_HANDLE(task, i);
				
					if (hud->is_present_in_data_not_used_yet[j] == 0 && also_add_data_in_not_used_yet_list == 1)
					{							
						#ifdef PRINT
						printf("%p gets 1 at is_present_in_data_not_used_yet on GPU %d\n", STARPU_TASK_GET_HANDLE(task, i), j); fflush(stdout);
						#endif
						
						//~ printf("Data already exist but not anymore on this GPU. Adding %p in data_not_used_yet of GPU %d\n", e->D, j); fflush(stdout);
						hud->is_present_in_data_not_used_yet[j] = 1;
						if (data_order == 1)
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
						}
						else
						{
							gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);	
						}
						//~ print_data_not_used_yet();
					}
				}
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;				
			}
		}
	}
        
    /* Adding the pointer in the task. */
    struct pointer_in_task *pt = malloc(sizeof(*pt));
    pt->pointer_to_cell = task;
    pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
    pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
	
    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
    {
		/* Pointer toward the main task list in the handles. */
		struct task_using_data *e = task_using_data_new();
		e->pointer_to_T = task;
		
		/* Adding the task in the list of task using the data */
		if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
		{
			struct task_using_data_list *tl = task_using_data_list_new();
			task_using_data_list_push_front(tl, e);
			STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
		}
		else
		{
			task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
		}
					
		/* Adding the pointer in the task toward the data. */
		pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
		pt->tud[i] = e;
    }
    task->sched_data = pt;
}

//~ /* Nouvelle version où je pars des données. Permet de plus facilement gérer les dépendances et savoir si une donnée dois être ajouté à la liste des données pas encore utilisées */
//~ void initialize_task_data_gpu_single_task(struct starpu_task *task)
//~ {
    //~ int i = 0;
    //~ int j = 0;
    //~ struct handle_user_data* hud = malloc(sizeof(*hud));
    //~ struct pointer_in_task* pt = malloc(sizeof(*pt));
    //~ pt->pointer_to_cell = task;
    //~ pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
    //~ pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
	
    //~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
    //~ {
		//~ /* Pointer toward the main task list in the handles. */
		//~ struct task_using_data *tud = task_using_data_new();
		//~ tud->pointer_to_T = task;
		
		//~ /* Adding the task in the list of task using the data */
		//~ if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL)
		//~ {
			//~ struct task_using_data_list *tl = task_using_data_list_new();
			//~ task_using_data_list_push_front(tl, tud);
			//~ STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
		//~ }
		//~ else
		//~ {
			//~ task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, tud);
		//~ }
		
		//~ struct gpu_data_not_used *e = gpu_data_not_used_new();
		//~ e->D = STARPU_TASK_GET_HANDLE(task, i);
		
		//~ /* The handle has never been initialized. */
		//~ if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		//~ {
			//~ hud = malloc(sizeof(*hud));
			//~ hud->last_iteration_DARTS = iteration_DARTS;
			
			//~ /* Need to init them with the number of GPU */
			//~ hud->nb_task_in_pulled_task = malloc(Ngpu*sizeof(int));
			//~ hud->nb_task_in_planned_task = malloc(Ngpu*sizeof(int));
			//~ hud->last_check_to_choose_from = malloc(Ngpu*sizeof(int));
			//~ hud->is_present_in_data_not_used_yet = malloc(Ngpu*sizeof(int));
			
			//~ for (j = 0; j < Ngpu; j++)
			//~ {
				//~ hud->nb_task_in_pulled_task[j] = 0;
				//~ hud->nb_task_in_planned_task[j] = 0;
				//~ hud->last_check_to_choose_from[j] = 0;
				//~ hud->is_present_in_data_not_used_yet[j] = 1; /* To say that this data is present in each GPU's list */
			//~ }
	
			//~ STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			
			//~ /* Adding the data in the list of data not used of each GPU. */
			//~ if (data_order == 1)
			//~ {
				//~ for (j = 0; j < Ngpu; j++)
				//~ {
					//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
				//~ }
			//~ }
			//~ else
			//~ {
				//~ for (j = 0; j < Ngpu; j++)
				//~ {
					//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
				//~ }
			//~ }
		//~ }
		//~ else /* The data already exist from the same or a previous iteration. Need to check if it's a new iteration first. */ 
		//~ {
			//~ hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			//~ if (hud->last_iteration_DARTS != iteration_DARTS) /* On est sur une nouvelle itération donc on peut re-init. */
			//~ {
				//~ for (j = 0; j < Ngpu; j++)
				//~ {
					//~ hud->nb_task_in_pulled_task[j] = 0;
					//~ hud->nb_task_in_planned_task[j] = 0;
					//~ hud->last_check_to_choose_from[j] = 0;
					//~ hud->is_present_in_data_not_used_yet[j] = 1; /* To say that this data is present in each GPU's list */
				//~ }
				//~ hud->last_iteration_DARTS = iteration_DARTS;
				
				//~ STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
				
				//~ /* Adding the data in the list of data not used of each GPU. */
				//~ if (data_order == 1)
				//~ {
					//~ for (j = 0; j < Ngpu; j++)
					//~ {
						//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
					//~ }
				//~ }
				//~ else
				//~ {
					//~ for (j = 0; j < Ngpu; j++)
					//~ {
						//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
					//~ }
				//~ }
			//~ }
			//~ else /* So the data exist from the same iteration. We need to check if it's still in the list of unused task before adding it or not! */
			//~ {
				//~ for (j = 0; j < Ngpu; j++)
				//~ {
					//~ if (hud->is_present_in_data_not_used_yet[j] == 0) /* It's not in it I can add it. */
					//~ {
						//~ if (data_order == 1)
						//~ {
							//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].new_gpu_data, e);
						//~ }
						//~ else
						//~ {
							//~ gpu_data_not_used_list_push_back(tab_gpu_planned_task[j].gpu_data, e);
						//~ }
						//~ hud->is_present_in_data_not_used_yet[j] = 1;
					//~ }
					//~ /* Else I don't. */
				//~ }
			//~ }
		//~ }
		
		//~ /* Adding the pointer in the task toward the data. */
		//~ pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
		//~ pt->tud[i] = tud;
	//~ }
	//~ task->sched_data = pt;
//~ }

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge(int arr[], int l, int m, int r, struct starpu_task **task_tab)
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
void mergeSort(int *arr, int l, int r, struct starpu_task **task_tab)
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

/* Randomise sched_data uniquement, càd dire les nouvelles tâches et les mets à la fin de main_task_list */
void randomize_new_task_list(struct dynamic_data_aware_sched_data *d)
{
    int random = 0;
    int i = 0;
    struct starpu_task *task_tab[NT_DARTS]; /* NT_DARTS correspond au nombre de nouvelles tâches. */
    
    for (i = 0; i < NT_DARTS; i++)
    {
		task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
    }
    for (i = 0; i < NT_DARTS; i++)
    {
		random = rand()%(NT_DARTS - i);
		starpu_task_list_push_back(&d->main_task_list, task_tab[random]);
		
		/* Je remplace la case par la dernière tâche du tableau */
		task_tab[random] = task_tab[NT_DARTS - i - 1];
	}
}

/* Randomise ensemble main_task_list et les nouvelles tâches. */
void randomize_full_task_list(struct dynamic_data_aware_sched_data *d)
{
	/* Version où je les choisis des chiffres random pour chaque tâche, puis je trie
	 * en même temps avec un tri fusion le tableau d'entiers random et le tableau de
	 * tâches. Ensuite je parcours la liste de tâche principale en insérant 1 à 1 les
	 * tâches à leurs position. */
	//~ print_task_list(&d->main_task_list, "Avant randomisation totale");
    int i = 0;
    int j = 0;
    int size_main_task_list = starpu_task_list_size(&d->main_task_list);
    struct starpu_task *task_tab[NT_DARTS];
    struct starpu_task *task = NULL;
    int random_number[NT_DARTS];
    int avancement_main_task_list = 0;
    /* Remplissage d'un tableau avec les nouvelles tâches + tirage de chiffre aléatoire 
     * pour chaque tâche. */
    for (i = 0; i < NT_DARTS; i++)
    {
		task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
		random_number[i] = rand()%size_main_task_list;
    }
    
    /* Appel du tri fusion. */
    mergeSort(random_number, 0, NT_DARTS - 1, task_tab);

    /* Remplissage de main task list dans l'ordre et en fonction du chiffre tirée. */
    task = starpu_task_list_begin(&d->main_task_list);
    for (i = 0; i < NT_DARTS; i++)
    {
		for (j = avancement_main_task_list; j < random_number[i]; j++)
		{
			task = starpu_task_list_next(task);
			avancement_main_task_list++;
		}
		starpu_task_list_insert_before(&d->main_task_list, task_tab[i], task);
	}
    
    //~ print_task_list(&d->main_task_list, "Après randomisation totale");

	/* Version où je les merge les 2 listes de tâches puis je les mélange */
    //~ int random = 0;
    //~ int i = 0;
    //~ int size_main_task_list = starpu_task_list_size(&d->main_task_list);
    //~ struct starpu_task *task_tab[NT_DARTS + size_main_task_list];
    //~ for (i = 0; i < NT_DARTS; i++)
    //~ {
		//~ task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
    //~ }
    //~ for (i = NT_DARTS; i < NT_DARTS + size_main_task_list; i++)
    //~ {
		//~ task_tab[i] = starpu_task_list_pop_front(&d->main_task_list);
    //~ }
    //~ for (i = 0; i < NT_DARTS + size_main_task_list; i++)
    //~ {
		//~ random = rand()%(NT_DARTS + size_main_task_list - i);
		//~ starpu_task_list_push_back(&d->main_task_list, task_tab[random]);
		
		//~ /* Je remplace la case par la dernière tâche du tableau */
		//~ task_tab[random] = task_tab[NT_DARTS + size_main_task_list - i - 1];
	//~ }
}

/* Chaque GPU a un pointeur vers sa première tâche à pop.
 * Ensuite dans le scheduling quand on pop pour la première fois c'est celle la.
 * En plus ca tombe bien le premier pop est géré direct en dehors de random
 * grâce à l'attribut first_task de la struct planned_task. */
void natural_order_task_list(struct dynamic_data_aware_sched_data *d)
{
    int i = 0;
    int j = 0;
    struct starpu_task *task = NULL;
    
    for (i = 0; i < NT_DARTS; i++)
    {
		if (i == (NT_DARTS/Ngpu)*j && j < Ngpu)
		{
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
void randomize_full_data_not_used_yet()
{
    /* NEW */
    int i = 0;
    int j = 0;
    int random = 0;
    int number_of_data = 0;
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    		
	for (i = 0; i < Ngpu; i++)
	{
		number_of_data = gpu_data_not_used_list_size(tab_gpu_planned_task[i].gpu_data);
		struct gpu_data_not_used *data_tab[number_of_data];

		for (j = 0; j < number_of_data; j++)
		{
			data_tab[j] = gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data);
		}
		struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
		
		for (j = 0; j < number_of_data; j++)
		{
			random = rand()%(number_of_data - j);
			gpu_data_not_used_list_push_back(randomized_list, data_tab[random]);
			
			/* Je remplace la case par la dernière tâche du tableau */
			data_tab[random] = data_tab[number_of_data - j - 1];
		}
		/* Then replace the list with it. */
		tab_gpu_planned_task[i].gpu_data = randomized_list;
		//~ my_planned_task_control->pointer = tab_gpu_planned_task[i].next;
	}
}

/* Randomize the new data and put them at the end of datanotused for all the GPU. */
void randomize_new_data_not_used_yet()
{
    int i = 0;
    int j = 0;
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    int random = 0;
    int number_new_data = 0;
    
	for (i = 0; i < Ngpu; i++)
	{
		if (!gpu_data_not_used_list_empty(tab_gpu_planned_task[i].new_gpu_data))
		{
			number_new_data = gpu_data_not_used_list_size(tab_gpu_planned_task[i].new_gpu_data);
			struct gpu_data_not_used *data_tab[number_new_data];
			for (j = 0; j < number_new_data; j++)
			{
				data_tab[j] = gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].new_gpu_data);
			}
			for (j = 0; j < number_new_data; j++)
			{
				random = rand()%(number_new_data - j);
				gpu_data_not_used_list_push_back(tab_gpu_planned_task[i].gpu_data, data_tab[random]);
				
				/* Je remplace la case par la dernjère tâche du tableau */
				data_tab[random] = data_tab[number_new_data - j - 1];
			}
		}
		//~ my_planned_task_control->pointer = tab_gpu_planned_task[i].next;
	}
		
	//~ int random = 0;
    //~ struct starpu_task *task_tab[NT_DARTS]; /* NT_DARTS correspond au nombre de nouvelles tâches. */
    
    //~ for (i = 0; i < NT_DARTS; i++)
    //~ {
		//~ task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
    //~ }
    //~ for (i = 0; i < NT_DARTS; i++)
    //~ {
		//~ random = rand()%(NT_DARTS - i);
		//~ starpu_task_list_push_back(&d->main_task_list, task_tab[random]);
		
		//~ /* Je remplace la case par la dernière tâche du tableau */
		//~ task_tab[random] = task_tab[NT_DARTS - i - 1];
	//~ }
	
    //~ /* NEW */
    //~ int i = 0;
    //~ int j = 0;
    //~ int random = 0;
    //~ int number_of_data = 0;
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    

		//~ for (i = 0; i < Ngpu; i++)
		//~ {
			//~ number_of_data = gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data);
			//~ struct gpu_data_not_used *data_tab[number_of_data];


			//~ for (j = 0; j < number_of_data; j++)
			//~ {
				//~ data_tab[j] = gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data);
			//~ }
			//~ struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
			
			//~ for (j = 0; j < number_of_data; j++)
			//~ {
				//~ random = rand()%(number_of_data - j);
				//~ gpu_data_not_used_list_push_back(randomized_list, data_tab[random]);
				
				//~ /* Je remplace la case par la dernière tâche du tableau */
				//~ data_tab[random] = data_tab[number_of_data - j - 1];
			//~ }
			//~ /* Then replace the list with it. */
			//~ my_planned_task_control->pointer->gpu_data = randomized_list;
			//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
		//~ }
}

/* Chaque GPU commence au début de la liste des données a différents endroits en fonction du nombre de GPU.
 * Donc GPU 1 a la premiere donnée, GPU 2 la n/NGPU ème donnée et ainsi de suite. */
void natural_order_data_not_used_yet()
{
    int i = 0;
    int j = 0;
    int number_of_data = 0;
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    
    /* I need this for the %random. */
    number_of_data = gpu_data_not_used_list_size(tab_gpu_planned_task[0].gpu_data);
    
    struct gpu_data_not_used *data_tab[number_of_data];
    
    /* On ne fais rien pour le premier GPU. Le deuxième GPU commence à Ndata/Ngpu,
     * puis le deuxième à (Ndata/Ngpu)*1 et esnuite (Ndata/Ngpu)*2 ainsi de suite ...
     */
    for (i = 1; i < Ngpu; i++)
    {
		for (j = 0; j < (number_of_data/Ngpu)*i; j++)
		{
			data_tab[j] = gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data);
		}
		struct gpu_data_not_used_list *natural_order_list = gpu_data_not_used_list_new();
		for (j = 0; j < number_of_data - ((number_of_data/Ngpu)*i); j++)
		{
			gpu_data_not_used_list_push_back(natural_order_list, gpu_data_not_used_list_pop_front(tab_gpu_planned_task[i].gpu_data));
		}
		for (j = 0; j < (number_of_data/Ngpu)*i; j++)
		{
			gpu_data_not_used_list_push_back(natural_order_list, data_tab[j]);
		}

		/* Then replace the list with it. */
		tab_gpu_planned_task[i].gpu_data = natural_order_list;
		//~ my_planned_task_control->pointer = tab_gpu_planned_task[i].next;
    }
}

/** 
 * Get a task to return to pull_task. 
 * In multi GPU it allows me to return a task from the right element in the 
 * linked list without having another GPU comme and ask a task in pull_task.
 **/
struct starpu_task *get_task_to_return_pull_task_dynamic_data_aware(int current_gpu, struct starpu_task_list *l)
{
	#ifdef PRINT
	printf("\nDebut get task to return GPU n°%d.\n", current_gpu); fflush(stdout);
	#endif
			
	int i = 0;
    
	//~ print_planned_task_all_gpu(); fflush(stdout);
	//~ print_pulled_task_all_gpu(); fflush(stdout);
    
    /* If there are still tasks either in the packages, the main task list or the refused task,
     * I enter here to return a task or start dynamic_data_aware_scheduling. Else I return NULL.
     */
    //~ if (!starpu_task_list_empty(&temp_pointer->planned_task) || !starpu_task_list_empty(l) || !starpu_task_list_empty(&temp_pointer->refused_fifo_list))
    //~ {
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("GPU %d is asking for a task.\n", current_gpu); }
		struct starpu_task *task = NULL;

		/* If one or more task have been refused */
		if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu - 1].refused_fifo_list)) 
		{
			/* Ici je ne met pas à jour pulled_task car je l'ai déjà fais pour la tâche avant qu'elle ne soit refusé. */
			//~ task = starpu_task_list_pop_back(&temp_pointer->refused_fifo_list); 
			task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu - 1].refused_fifo_list);
			
			//~ #ifdef PRINT_PYTHON /* Il ne faut pas le faire ici non ? */
			//~ print_data_to_load_prefetch(task, current_gpu);
			//~ #endif
			#ifdef PRINT
			printf("Return refused task %p.\n", task); fflush(stdout);
			#endif
			return task;
		}
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
		
		/* If the package is not empty I can return the head of the task list. */
		if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu - 1].planned_task))
		{
			task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu - 1].planned_task);

			/* Remove it from planned task compteur. Could be done in an external function as I use it two times */
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
				hud->nb_task_in_planned_task[current_gpu - 1] = hud->nb_task_in_planned_task[current_gpu - 1] - 1;				
				STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			}
			
			/* Fonction qui ajoute la tâche à pulled_task. Elle est aussi dans le else if en dessous. */
			add_task_to_pulled_task(current_gpu, task);
				
			/* For visualisation in python. */
			#ifdef PRINT_PYTHON
			print_data_to_load_prefetch(task, current_gpu);
			#endif
			#ifdef PRINT
			printf("Task: %p is getting out of pull_task from planned task not empty on GPU %d\n", task, current_gpu); fflush(stdout);
			#endif
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			return task;
		}
		
		/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
		if (!starpu_task_list_empty(l))
		{			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			
			/* La j'appelle 3D dans les deux cas car j'ai regroupé les 2 fonctions en 1 seule. 
			 * La différence se fais avec la var d'env APP. */
			dynamic_data_aware_scheduling_3D_matrix(l, current_gpu, &tab_gpu_planned_task[current_gpu - 1]);
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
			#endif
			
			if (!starpu_task_list_empty(&tab_gpu_planned_task[current_gpu - 1].planned_task))
			{
				task = starpu_task_list_pop_front(&tab_gpu_planned_task[current_gpu - 1].planned_task);
				
				add_task_to_pulled_task(current_gpu, task);
				
				/* Remove it from planned task compteur */
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
				{
					struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
					hud->nb_task_in_planned_task[current_gpu - 1] = hud->nb_task_in_planned_task[current_gpu - 1] - 1;
										
					STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
				}
			}
			else
			{
				#ifdef REFINED_MUTEX
				STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
				#endif
				
				#ifdef PRINT
				printf("Return NULL after scheduling call.\n"); fflush(stdout);
				#endif
				#ifdef PRINT_STATS
				nb_return_null_after_scheduling++;
				#endif
				return NULL;
			}
			
			/* For visualisation in python. */
			#ifdef PRINT_PYTHON
			print_data_to_load_prefetch(task, current_gpu);
			#endif
			#ifdef PRINT
			printf("Return task %p from the scheduling call.\n", task); fflush(stdout);
			#endif
			#ifdef PRINT_STATS
			nb_return_task_after_scheduling++;
			#endif
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			//~ printf("Return task %p from the scheduling call.\n", task); fflush(stdout);
			return task;
		}
		else
		{
			#ifdef PRINT
			printf("Return NULL because main task list is empty.\n"); fflush(stdout);
			#endif
			#ifdef PRINT_STATS
			nb_return_null_because_main_task_list_empty++;
			#endif
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			return NULL;
		}
}

bool graph_read = false; /* TODO: a suppr si j'utilise pas graph_descendants == 1 */

/* Pull tasks. When it receives new task it will randomize the task list and the GPU data list.
 * If it has no task it return NULL. Else if a task was refused it return it. Else it return the
 * head of the GPU task list. Else it calls dyanmic_outer_scheuling to fill this package. */
static struct starpu_task *dynamic_data_aware_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	//~ #ifdef PRINT
	//~ printf("Début de pull_task.\n"); fflush(stdout);
	//~ #endif
		
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
	#endif
		
    struct dynamic_data_aware_sched_data *data = component->data;

    /* Inutile pour DARTS pour le moment. */
    //~ if (gpu_memory_initialized == false)
    //~ {
		//~ GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));
		//~ gpu_memory_initialized = true;
    //~ }
    
	/* GRAPHE si je fais une pause dans cholesky */
	if (graph_descendants == 1 && graph_read == false && new_tasks_initialized == true)
	{
		graph_read = true;
		_starpu_graph_compute_descendants();
		_starpu_graph_foreach(set_priority, data);
	}
    
    /* New tasks from push_task. We need to randomize. */
	#ifdef REFINED_MUTEX
    STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
    #endif
	
    if (new_tasks_initialized == true)
    {
		/* GRAPHE si je ne fais pas de pause et je veux le faire a chaque nouvel soumission de jobs */
		if (graph_descendants == 2)
		{
			_starpu_graph_compute_descendants();
			_starpu_graph_foreach(set_priority, data);
		}
		
		#ifdef PRINT_STATS
		nb_new_task_initialized++;
		#endif
		#ifdef PRINT
		printf("New tasks in pull_task.\n"); fflush(stdout);
		#endif
		
		new_tasks_initialized = false;
		
		#ifdef PRINT
		printf("\n-----\nPrinting GPU's data list and NEW task list before randomization:\n");
		print_data_not_used_yet();
		print_task_list(&data->sched_list, "Main task list"); fflush(stdout);
		#endif
		
		//~ NT_DARTS = starpu_task_list_size(&data->sched_list); /* Nombre de nouvelles tâches */
		
		#ifdef PRINT
		printf("NT_DARTS in pull_task = %d.\n", NT_DARTS); fflush(stdout);
		#endif
		#ifdef PRINT_STATS
		gettimeofday(&time_start_randomize, NULL);
		#endif
		
		/* Ordre des tâches dans main_task_list */
		if (task_order == 0) /* Randomise la liste des tâches entièrement à chaque nouvelle tâches. */
		{
			/* Si main task list est vide pas la peine d'appeller la fonction qui randomise ensemble les 2 listes de tâches. */
			if (!starpu_task_list_empty(&data->main_task_list))
			{
				randomize_full_task_list(data);
			}
			else
			{
				randomize_new_task_list(data);
			}
		}
		else if (task_order == 1) /* Randomise que les nouvelles tâches */
		{
			randomize_new_task_list(data);
		}
		else if (dependances == 0) /* task_order == 2 and no dependances */
		{
			natural_order_task_list(data);
		}
		
		/* Ordre des données dans datanotuse de chaque GPU */
		if (choose_best_data_from != 1) /* Si on regarde dans la mémoire pour choisir les données, il n'y a aucun intérêt à toucher à la liste des données. */
		{
			if (data_order == 0) /* Randomise la liste des données entièrement et différement pour chaque GPU. */
			{
				randomize_full_data_not_used_yet();
			}
			else if (data_order == 1) /* Randomise que les nouvelles données. */
			{
				randomize_new_data_not_used_yet();
			}
			else if (dependances == 0) /* DATA_ORDER == 2 et DEPENDANCE == 0, ordre naturel avec point de départs différent pour les GPU. */
			{
				natural_order_data_not_used_yet();
			}
			/* De même si il y a des dépendances et que DATA_ORDER == 2 on va juste mettre les une après les autres les données. */
		}
		
		/* Test */
		NT_DARTS = 0;
		/* Test */
		
		#ifdef PRINT_STATS
		gettimeofday(&time_end_randomize, NULL);
		time_total_randomize += (time_end_randomize.tv_sec - time_start_randomize.tv_sec)*1000000LL + time_end_randomize.tv_usec - time_start_randomize.tv_usec;
		#endif
		
		#ifdef PRINT		
		printf("Il y a %d tâches.\n", NT_DARTS);
		printf("Printing GPU's data list and main task list after randomization (TASK_ORDER = %d, DATA_ORDER = %d):\n", task_order, data_order);
		print_data_not_used_yet();
		print_task_list(&data->main_task_list, "Main task list"); fflush(stdout);
		printf("-----\n\n");
		#endif
    }
    
    #ifdef REFINED_MUTEX
    STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
    #endif
    
	//~ int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()); /* Attention le premier GPU vaut 1 et non 0. */
    //~ STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
        
    //~ STARPU_PTHREAD_MUTEX_LOCK(&local_mutex[current_gpu - 1]);
    struct starpu_task *task = get_task_to_return_pull_task_dynamic_data_aware(starpu_worker_get_memory_node(starpu_worker_get_id()), &data->main_task_list);
    //~ STARPU_PTHREAD_MUTEX_UNLOCK(&local_mutex[current_gpu - 1]);
    
    #ifdef LINEAR_MUTEX
    STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
    #endif
	
	#ifdef PRINT
	if (task != NULL)
	{
		printf("Pulled task %p.\n", task);
	}
	#endif
	
    return task;
}

void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct gpu_planned_task *g, int gpu_id)
{
	struct gpu_data_not_used *new_element = gpu_data_not_used_new();
    new_element->D = h;
    
    
    struct handle_user_data * hud = h->user_data;
    
    #ifdef PRINT
    printf("%p gets 1 at is_present_in_data_not_used_yet with random push\n", h); fflush(stdout);
    #endif
    
	hud->is_present_in_data_not_used_yet[gpu_id] = 1;
	h->user_data = hud;
	
	//~ printf("1.\n"); fflush(stdout);
    
	if (gpu_data_not_used_list_empty(g->gpu_data))
	{
		//~ printf("1.1\n"); fflush(stdout);
		printf("After eviction. Push data %p in GPU %d\n", new_element->D, gpu_id); fflush(stdout);
		gpu_data_not_used_list_push_back(g->gpu_data, new_element);
		//~ printf("2.\n"); fflush(stdout);
		return;
	}
	//~ printf("1.2\n"); fflush(stdout);
    struct gpu_data_not_used *ptr = gpu_data_not_used_new();
	//~ printf("1.2.1 %d\n", gpu_data_not_used_list_size(g->gpu_data)); fflush(stdout);
    int random = rand()%gpu_data_not_used_list_size(g->gpu_data);
	//~ printf("1.3\n"); fflush(stdout);
    int i = 0;
    ptr = gpu_data_not_used_list_begin(g->gpu_data);

    for (i = 0; i < random; i++)
    {
		ptr = gpu_data_not_used_list_next(ptr);
    }
    gpu_data_not_used_list_insert_before(g->gpu_data, new_element, ptr);
    //~ printf("3.\n"); fflush(stdout);
}

void update_best_data(int* number_free_task_max, int* task_available_max, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int* task_available_max_1_from_free, int nb_free_task_candidate, int task_using_data_list_size_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i)
{
	/* If we are in a case with free tasks */
	if (nb_free_task_candidate > *number_free_task_max)
	{
		*number_free_task_max = nb_free_task_candidate;
		*task_available_max = task_using_data_list_size_candidate;
		*handle_popped = handle_candidate;
		*priority_max = priority_candidate;
					
		#ifdef PRINT_STATS
		*data_choosen_index = i + 1;
		#endif
	}
	else if (nb_free_task_candidate == *number_free_task_max && *number_free_task_max != 0)
	{
		/* I first tiebreak with 1 from free task */
		/* Then with priority */
		if (prio == 1 && *priority_max < priority_candidate)
		{
			*task_available_max = task_using_data_list_size_candidate;
			*handle_popped = handle_candidate;
			*priority_max = priority_candidate;
		}
		else if ((*priority_max == priority_candidate || prio == 0) && task_using_data_list_size_candidate > *task_available_max)
		{
			*task_available_max = task_using_data_list_size_candidate;
			*handle_popped = handle_candidate;
			*priority_max = priority_candidate;
						
			#ifdef PRINT_STATS
			*data_choosen_index = i + 1;
			#endif
		}
	}
	else if (number_1_from_free_task_candidate > *number_1_from_free_task_max)
	{
		*number_1_from_free_task_max = number_1_from_free_task_candidate;
		*task_available_max_1_from_free = task_using_data_list_size_candidate;
		*handle_popped = handle_candidate;
		*priority_max = priority_candidate;
					
		#ifdef PRINT_STATS
		*data_choosen_index = i + 1;
		#endif
	}
	/* Si il y a égalité je regarde la priorité d'abord puis celle qui peut faire le plus de tâches globalement. */
	else if (number_1_from_free_task_candidate == *number_1_from_free_task_max && *number_1_from_free_task_max != 0)
	{					
		/* I first tiebreak with priority */
		if (prio == 1 && *priority_max < priority_candidate)
		{
			*task_available_max_1_from_free = task_using_data_list_size_candidate;
			*handle_popped = handle_candidate;
			*priority_max = priority_candidate;
		}
		else if ((*priority_max == priority_candidate || prio == 0) && task_using_data_list_size_candidate > *task_available_max_1_from_free) /* Then glovbal number of tasks remaining */
		{
			*task_available_max_1_from_free = task_using_data_list_size_candidate;
			*handle_popped = handle_candidate;
			*priority_max = priority_candidate;
						
			#ifdef PRINT_STATS
			*data_choosen_index = i + 1;
			#endif
		}
	}
}

/* The call is: update_best_data_single_decision_tree(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i); */
//~ void update_best_data_single_decision_tree(int* number_free_task_max, int* task_available_max, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int* task_available_max_1_from_free, int nb_free_task_candidate, int task_using_data_list_size_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i)
void update_best_data_single_decision_tree(int* number_free_task_max, double* remaining_expected_length_max, starpu_data_handle_t* handle_popped, int* priority_max, int* number_1_from_free_task_max, int nb_free_task_candidate, double remaining_expected_length_candidate, starpu_data_handle_t handle_candidate, int priority_candidate, int number_1_from_free_task_candidate, int* data_choosen_index, int i, struct starpu_task** best_1_from_free_task, struct starpu_task* best_1_from_free_task_candidate)
{
	/* Il y a bien plus de return que de update. Je met donc les if dans ce sens pour pouvoir gagner en complexité et s'arrêter plus tot. */
	/* First tiebreak with most free task */
	if (nb_free_task_candidate < *number_free_task_max)
	{
		return;
	}
	/* Then with number of 1 from free */
	else if (nb_free_task_candidate == *number_free_task_max)
	{
		
		/* V1 1 from free then prio */
		if  (dopt_selection_order == 0)
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
					#ifdef PRINT_STATS
					if (remaining_expected_length_candidate == *remaining_expected_length_max)
					{
						data_choice_per_index = true;
					}
					#endif
					
					return;
				}
			}
		}
		else
		{
		/* V2 prio then 1 from free */
			if (prio == 1 && *priority_max > priority_candidate)
			{
				return;
			}
			else if (*priority_max == priority_candidate)
			{				
				/* Then with 1 from free */
				if (number_1_from_free_task_candidate < *number_1_from_free_task_max)
				{				
					return;
				}
				/* Then with time of task in the list of task using this data */
				else if ((number_1_from_free_task_candidate == *number_1_from_free_task_max) && remaining_expected_length_candidate <= *remaining_expected_length_max)
				{
					#ifdef PRINT_STATS
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
	
	#ifdef PRINT_STATS
	data_choice_per_index = false;
	#endif
	/* Update */
	*number_free_task_max = nb_free_task_candidate;
	//~ *task_available_max = task_using_data_list_size_candidate;
	*remaining_expected_length_max = remaining_expected_length_candidate;
	*number_1_from_free_task_max = number_1_from_free_task_candidate;
	*handle_popped = handle_candidate;
	*priority_max = priority_candidate;
	*best_1_from_free_task = best_1_from_free_task_candidate;
		
	#ifdef PRINT_STATS
	*data_choosen_index = i + 1;
	#endif
	
	//~ /* First tiebreak with most free task */
	//~ if (nb_free_task_candidate > *number_free_task_max)
	//~ {
		//~ goto update;
	//~ }
	//~ /* Then with number of 1 from free */
	//~ else if (nb_free_task_candidate == *number_free_task_max)
	//~ {
		//~ if (number_1_from_free_task_candidate > *number_1_from_free_task_max)
		//~ {
			//~ goto update;
		//~ }
		//~ /* Then with priority */
		//~ else if (number_1_from_free_task_candidate == *number_1_from_free_task_max)
		//~ {
			//~ if (prio == 1 && *priority_max < priority_candidate)
			//~ {
				//~ goto update;
			//~ }
			//~ /* Then with number of task in the list of task using this data */
			//~ else if ((*priority_max == priority_candidate || prio == 0) && task_using_data_list_size_candidate > *task_available_max)
			//~ {
				//~ goto update;
			//~ }
		//~ }
	//~ }
	//~ return;
	
	//~ update: ;
	//~ *number_free_task_max = nb_free_task_candidate;
	//~ *task_available_max = task_using_data_list_size_candidate;
	//~ *number_1_from_free_task_max = number_1_from_free_task_candidate;
	//~ *handle_popped = handle_candidate;
	//~ *priority_max = priority_candidate;
					
	//~ #ifdef PRINT_STATS
	//~ *data_choosen_index = i + 1;
	//~ #endif
}

/**
 * Fill a package's task list following dynamic_data_aware algorithm.
 * Si je trouve une donnée qui me donne des taches gratuites je prends et j'ajoute a planned task.
 * Sinon en chargeant 1 nouvelle donnée, je charge une tache dont la donnée amène le plus de tache a 1 seul chargement : (si c'est ce cas on créé une liste planned_task_1dtata__to_load).
 * Sinon random.
 **/
void dynamic_data_aware_scheduling_3D_matrix(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g)
{
	#ifdef PRINT
	printf("Début de sched 3D avec GPU %d.\n", current_gpu - 1); fflush(stdout);
	#endif
	
	Dopt[current_gpu - 1] = NULL;
		
	#ifdef PRINT_STATS
	gettimeofday(&time_start_schedule, NULL);
	#endif
	
    int i = 0;
    int j = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    
    //~ int task_available_max = 0;
    //~ int task_available_max_1_from_free = 0;
    double remaining_expected_length_max = 0;
    
    struct starpu_task* best_1_from_free_task = NULL;
    struct starpu_task* temp_best_1_from_free_task = NULL;
    
    /* Values used to know if the currently selected data is better that the pne already choosen */
    int number_free_task_max = 0;  /* Number of free task with selected data */
    int temp_number_free_task_max = 0;
    int number_1_from_free_task_max = 0;  /* Number of task on from free with selected data */
    int temp_number_1_from_free_task_max = 0;
    int priority_max = INT_MIN; /* Highest priority of a task using the choosen data that we know will be pushed to planned task. */
    int temp_priority_max = INT_MIN; /* Highest priority of a task using the choosen data that we know will be pushed to planned task. */
    starpu_data_handle_t handle_popped = NULL; /* Pointer to choosen best data */
    
    struct handle_user_data * hud = NULL;
    
    //~ struct task_using_data_list *tudl = task_using_data_list_new();
    
	#ifdef PRINT
	printf("Il y a %d données parmi lesquelles choisir pour le GPU %d.\n", gpu_data_not_used_list_size(g->gpu_data), current_gpu); fflush(stdout);
	#endif
	
	int data_choosen_index = 0; /* Forced to declare it here because of the fonction update */
	#ifdef PRINT_STATS
	//~ int data_choosen_index = 0;
	int nb_data_looked_at = 0; /* Uniquement le cas ou on choisis depuis la mémoire */
	#endif
	
	//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	
    /* Si c'est la première tâche, on regarde pas le reste on fais random. */
    if (g->first_task == true)
    {
		#ifdef PRINT
		printf("Hey! C'est la première tâche du GPU n°%d!\n", current_gpu); fflush(stdout);	
		#endif
		#ifdef PRINT_STATS
		if (iteration_DARTS == 1)
		{
			FILE *f = NULL;
			char str[2];
			int size = strlen("Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_.csv") + strlen(str);
			char* path = (char *)malloc(size);
			sprintf(str, "%d", current_gpu);
			strcpy(path, "Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_");
			strcat(path, str);
			strcat(path, ".csv");
			f = fopen(path, "a");
			fprintf(f, "%d,%d,%d,%d\n", g->number_data_selection, 0, 0, 0);
			fclose(f);
			free(path);
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
						
			/* old place, a check que c'était pas important si ca crash. */
			//~ g->first_task_to_pop = NULL;
						
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (!gpu_data_not_used_list_empty(g->gpu_data)) /* TODO : Est-ce vraiment utile ? Pas sûr, ptet en réel sur Grid5k ? A tester. Si je l'enlève enlever le deuxième qu'il y a plus bas dans le cas random. */
				{
					for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
					{
						if(e->D == STARPU_TASK_GET_HANDLE(task, i))
						{
							printf("First task. Erase data %p from GPU %d\n", e->D, current_gpu - 1); fflush(stdout);
							gpu_data_not_used_list_erase(g->gpu_data, e);
							print_data_not_used_yet();
							hud = e->D->user_data;
							
							#ifdef PRINT
							printf("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu - 1); fflush(stdout);
							#endif
							
							hud->is_present_in_data_not_used_yet[current_gpu - 1] = 0;
							e->D->user_data = hud;
							
							gpu_data_not_used_delete(e);
							break;
						}
					}
				}
			}
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
			#endif
			
			/* Add it from planned task compteur */			
			increment_planned_task_data(task, current_gpu);
			
			#ifdef PRINT
			printf("Returning first task of GPU n°%d in natural order: %p.\n", current_gpu, task);
			#endif
			
			erase_task_and_data_pointer(task, main_task_list);
			starpu_task_list_push_back(&g->planned_task, task);
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			
			goto end_scheduling;
		}
		else
		{
			goto random;
		}
	}
    
    /* Ce cas arrive avec le cas ou je gère pas les evictions. Car quand je ne gère pas les évictions je ne remet pas les données évincées dans la liste des données
     * à faire. */
    if (gpu_data_not_used_list_empty(g->gpu_data))
    {
		#ifdef PRINT
		printf("Random selection car liste des données non utilisées vide.\n");
		#endif
		
		goto random;
    }
    
    /* To know if all the data needed for a task are loaded in memory. */
    int data_not_available = 0;
    bool data_available = true;
    	
	/* Pour diminuer au début de l'execution. 
	 * TODO : Ne marche plus si il y a des dépendanceces car NT_DARTS est le nombre de nouvelles données. Donc il faudrait faire une somme totale des tâches qui sont arrivées ou qui vont arriver car le but c'est des le debut de mettre un trhasold pour les grosses appli. En réupérant N peut etre ? */
	int choose_best_data_threshold = INT_MAX;
	if (threshold == 1)
	{
		/* En 2D on fais cela */
		if (app == 0)
		{
			if (NT_DARTS > 14400)
			{
				choose_best_data_threshold = 110;
			}
		}
		else if (NT_DARTS > 1599) /* Pour que ca se déclanche au 4ème point en 3D */
		{
			choose_best_data_threshold = 200;
		}
	}
	
	//~ struct handle_user_data * hud = NULL;
	
	#ifdef PRINT_STATS
	gettimeofday(&time_start_choose_best_data, NULL);
	#endif
	
	#ifdef PRINT_STATS
	data_choice_per_index = false;
	#endif
	
	//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);	
		
	/* Recherche de la meilleure donnée. Je regarde directement pour chaque donnée, le nombre de tâche qu'elle met à 1 donnée d'être possible si j'ai toujours
	 * 0 à number_free_task_max. */
	if (choose_best_data_from == 0) /* Le cas de base où je regarde les données pas encore utilisées. */
	{
		#ifdef PRINT_STATS
		g->number_data_selection++;
		#endif
		
		for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data) && i != choose_best_data_threshold; e = gpu_data_not_used_list_next(e), i++)
		{
			temp_number_free_task_max = 0;
			temp_number_1_from_free_task_max = 0;
			temp_priority_max = INT_MIN;
			temp_best_1_from_free_task = NULL;
			
			#ifdef PRINT_STATS
			nb_data_looked_at++;
			#endif
			
			/* Il y a deux cas pour simplifier un peu la complexité. Si j'ai au moins 1 tâche qui peut être gratuite, je ne fais plus les compteurs qui permettent 
			 * d'avoir des tâches à 1 donnée d'être free et on ajoute un break pour gagner un peu de temps. */
			//~ if (number_free_task_max == 0 && app != 0) /* Regardons les 1 from free tasks */
			//~ {
				for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
				{
					//~ printf("Testing task %p\n", t->pointer_to_T);
					/* I put it at false if at least one data is missing. */
					data_not_available = 0; 
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
					{
						/* I test if the data is on memory */ 
						if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
						{
							/* Sans ifdef */
							if (simulate_memory == 0)
							{
								/* Ancien */
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
								{
									data_not_available++;
								}
							}
							else if (simulate_memory == 1)
							{
								hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
								{
									data_not_available++;
								}
							}
						}
					}
					
					if (data_not_available == 0)
					{
						temp_number_free_task_max++;
						
						/* Version où je m'arrête dès que j'ai une tâche gratuite.
						 * Nouvelle place du threshold == 2. TODO: mettre un ifdef */
						if (threshold == 2)
						{
							handle_popped = e->D;
							number_free_task_max = temp_number_free_task_max;
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
					}
					else if (data_not_available == 1)
					{
						temp_number_1_from_free_task_max++;
						
						/* Getting the max priority */
						if (t->pointer_to_T->priority > temp_priority_max)
						{
							temp_priority_max = t->pointer_to_T->priority;
							temp_best_1_from_free_task = t->pointer_to_T;
							//~ printf("New best 1 from free %p\n", temp_best_1_from_free_task);
						}
					}
				}
				
				/* Checking if current data is better */
				/* Looking at free OR 1 from free to choose */
				//~ update_best_data(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i);
				/* Looking at free THEN 1 from free to choose */
				//~ update_best_data_single_decision_tree(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i);
				
				hud = e->D->user_data;

				update_best_data_single_decision_tree(&number_free_task_max, &remaining_expected_length_max, &handle_popped, &priority_max, &number_1_from_free_task_max, temp_number_free_task_max, hud->sum_remaining_task_expected_length, e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i, &best_1_from_free_task, temp_best_1_from_free_task);
								
				
			//~ }
			/* number_free_task_max > 0 or 2D case. Only cheking data that gives free tasks. */
			//~ else
			//~ {
				//~ for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
				//~ {
					//~ /* I put it at false if at least one data is missing. */
					//~ data_available = true;
					//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
					//~ {
						//~ /* I test if the data is on memory */ 
						//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
						//~ {							
							//~ if (simulate_memory == 0)
							//~ {
								//~ /* Ancien */
								//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
								//~ {
									//~ data_available = false;
									//~ break;
								//~ }
							//~ }
							//~ else if (simulate_memory == 1)
							//~ {
								//~ /* Nouveau */
								//~ hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
								//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
								//~ {
									//~ data_available = false;
									//~ break;
								//~ }
							//~ }
						//~ }
					//~ }
					//~ if (data_available == true)
					//~ {
						//~ temp_number_free_task_max++;
						//~ /* Version où je m'arrête dès que j'ai une tâche gratuite.
						 //~ * Nouvelle place du threshold == 2. */
						//~ if (threshold == 2)
						//~ {
							//~ handle_popped = e->D;
							//~ number_free_task_max = temp_number_free_task_max;
							//~ goto end_choose_best_data;
						//~ }
						
						//~ if (t->pointer_to_T->priority > temp_priority_max)
						//~ {
							//~ temp_priority_max = t->pointer_to_T->priority;
						//~ }
					//~ }
				//~ }
			
				//~ /* Update best data if needed */
				//~ update_best_data(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i);				
			//~ }
		}
	}
	/* TODO : a refaire avec single ree et expected length */
	//~ else if (choose_best_data_from == 1) /* Le cas où je regarde uniquement les données (pas encore en mémoire) des tâches des données en mémoire. */
	//~ {
		//~ /* Pour ne pas regarder deux fois à la même itération la même donnée. */
		//~ struct handle_user_data * hud_last_check = NULL;
		
		//~ /* Attention ici c'est utile ne pas le metre entre des ifdef!!!! */
		//~ g->number_data_selection++;
		
		//~ starpu_data_handle_t *data_on_node;
		//~ unsigned nb_data_on_node = 0;
		//~ int *valid;
		//~ starpu_data_get_node_data(current_gpu, &data_on_node, &valid, &nb_data_on_node);
		//~ struct task_using_data *t2 = NULL;
		//~ int k = 0;
		//~ /* Je me met sur une donnée de la mémoire. */
		//~ for (i = 0; i < nb_data_on_node; i++)
		//~ {
			//~ /* Je me met sur une tâche de cette donnée en question. */
			//~ for (t2 = task_using_data_list_begin(data_on_node[i]->sched_data); t2 != task_using_data_list_end(data_on_node[i]->sched_data); t2 = task_using_data_list_next(t2))
			//~ {
				//~ /* Je me met sur une donnée de cette tâche (qui n'est pas celle en mémoire). */
				//~ for (k = 0; k < STARPU_TASK_GET_NBUFFERS(t2->pointer_to_T); k++)
				//~ {						
					//~ hud_last_check = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data;
					//~ /* Ici il faudrait ne pas regarder 2 fois la même donnée si possible. Ca peut arriver oui. */
					//~ if (STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k) != data_on_node[i] && hud_last_check->last_check_to_choose_from[current_gpu - 1] != g->number_data_selection)
					//~ {
						//~ #ifdef PRINT_STATS
						//~ nb_data_looked_at++;
						//~ #endif	
						
						//~ /* Mise à jour de l'itération pour la donnée pour ne pas la regarder deux fois à cette itération. */
						//~ hud_last_check->last_check_to_choose_from[current_gpu - 1] = g->number_data_selection;
						//~ STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data = hud_last_check;
		
						//~ temp_number_free_task_max = 0;
						//~ temp_number_1_from_free_task_max = 0;
						//~ temp_priority_max = INT_MIN;
						
						//~ if (number_free_task_max == 0 && app != 0)
						//~ {
							//~ /* Je regarde le nombre de free ou 1 from free tâche de cette donnée. */
							//~ for (t = task_using_data_list_begin(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t != task_using_data_list_end(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t = task_using_data_list_next(t))
							//~ {
								//~ data_not_available = 0; 
								//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
								//~ {
									//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k))
									//~ {
										//~ if (simulate_memory == 0)
										//~ {
											//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
											//~ {
												//~ data_not_available++;
											//~ }
										//~ }
										//~ else if (simulate_memory == 1)
										//~ {
											//~ hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
											//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
											//~ {
												//~ data_not_available++;
											//~ }
										//~ }
									//~ }
								//~ }
								//~ if (data_not_available == 0)
								//~ {
									//~ temp_number_free_task_max++;
									
									//~ /* Version où je m'arrête dès que j'ai une tâche gratuite.
									 //~ * Nouvelle place du threshold == 2. */
									//~ if (threshold == 2)
									//~ {
										//~ number_free_task_max = temp_number_free_task_max;
										//~ handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
										//~ goto end_choose_best_data;
									//~ }
									
									//~ /* For the first one I want to forget priority of one from free tasks. */
									//~ if (temp_number_free_task_max == 1)
									//~ {
										//~ temp_priority_max = t->pointer_to_T->priority;
									//~ }
									//~ else if (t->pointer_to_T->priority > temp_priority_max)
									//~ {
										//~ temp_priority_max = t->pointer_to_T->priority;
									//~ }
								//~ }
								//~ else if (data_not_available == 1)
								//~ {
									//~ temp_number_1_from_free_task_max++;
									//~ if (t->pointer_to_T->priority > temp_priority_max)
									//~ {
										//~ temp_priority_max = t->pointer_to_T->priority;
									//~ }
								//~ }
							//~ }
							
							//~ /* Update best data if needed */
							//~ update_best_data(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i);
						//~ }
						//~ else /* Cas 2D */
						//~ {
							//~ for (t = task_using_data_list_begin(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t != task_using_data_list_end(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t = task_using_data_list_next(t))
							//~ {
								//~ data_available = true;
								//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
								//~ {
									//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k))
									//~ {
										//~ if (simulate_memory == 0)
										//~ {
											//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
											//~ {
												//~ data_available = false;
												//~ break;
											//~ }
										//~ }
										//~ else if (simulate_memory == 1)
										//~ {
											//~ hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
											//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
											//~ {
												//~ data_available = false;
												//~ break;
											//~ }
										//~ }
									//~ }
								//~ }
								//~ if (data_available == true)
								//~ {
									//~ temp_number_free_task_max++;
									
									//~ /* Version où je m'arrête dès que j'ai une tâche gratuite.
									 //~ * Nouvelle place du threshold == 2. */
									//~ if (threshold == 2)
									//~ {
										//~ handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
										//~ number_free_task_max = temp_number_free_task_max;
										//~ goto end_choose_best_data;
									//~ }
									
									//~ if (t->pointer_to_T->priority > temp_priority_max)
									//~ {
										//~ temp_priority_max = t->pointer_to_T->priority;
									//~ }
								//~ }
							//~ }
							
							//~ /* Update best data if needed */
							//~ update_best_data(&number_free_task_max, &task_available_max, &handle_popped, &priority_max, &number_1_from_free_task_max, &task_available_max_1_from_free, temp_number_free_task_max, task_using_data_list_size(e->D->sched_data), e->D, temp_priority_max, temp_number_1_from_free_task_max, &data_choosen_index, i);
						//~ }
					//~ }	
				//~ }
			//~ }
		//~ }
	//~ }	
		
	#ifdef PRINT
	printf("Best data is = %p: %d free tasks and/or %d 1 from free tasks.\n", handle_popped, number_free_task_max, number_1_from_free_task_max); fflush(stdout);
	#endif

	end_choose_best_data : ;
	
	#ifdef PRINT_STATS
	if (data_choice_per_index == true)
	{
		nb_data_selection_per_index++;
	}
	nb_data_selection++;
	#endif
		
	/* Look at data conflict. If there is one I need to re-start the schedule for one of the GPU. */
	data_conflict[current_gpu - 1] = false;
	Dopt[current_gpu - 1] = handle_popped;
	for (i = 0; i < Ngpu; i++)
	{
		if(i != current_gpu - 1)
		{
			if (Dopt[i] == handle_popped && handle_popped != NULL)
			{				
				#ifdef PRINT
				printf("Iteration %d. Same data between GPU %d and GPU %d: %p.\n", iteration_DARTS, current_gpu, i + 1, handle_popped); fflush(stdout);
				#endif
				#ifdef PRINT_STATS
				number_data_conflict++;
				#endif
				
				data_conflict[current_gpu - 1] = true;
			}
		}
	}
		
	#ifdef PRINT_STATS
	if (iteration_DARTS == 1)
	{
		FILE *f = NULL;
		char str[2];
		int size = strlen("Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_.csv") + strlen(str);
		char* path = (char *)malloc(size);
		sprintf(str, "%d", current_gpu);
		strcpy(path, "Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_");
		strcat(path, str);
		strcat(path, ".csv");
		f = fopen(path, "a");
		if (number_free_task_max != 0)
		{
			nb_task_added_in_planned_task = number_free_task_max;
		}
		else
		{
			nb_task_added_in_planned_task = 1;
		}
		fprintf(f, "%d,%d,%d,%d\n", g->number_data_selection, data_choosen_index, nb_data_looked_at - data_choosen_index, nb_task_added_in_planned_task);
		fclose(f);
		free(path);
	}
	gettimeofday(&time_end_choose_best_data, NULL);
	time_total_choose_best_data += (time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)*1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec;
    #endif
            
    if (number_free_task_max != 0) /* Cas comme dans 2D, je met dans planned_task les tâches gratuites, sauf que j'ai 3 données à check et non 2. */
    {
		#ifdef PRINT_STATS
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		nb_free_choice++;
		#endif
	
		/* I erase the data from the list of data not used. */
		if (choose_best_data_from == 0)
		{			
				/* J'efface la données des datanotuse du GPU en question. Il n'y a que celle la a effacé car la tâche est gratuite 
				 * pour les autres données. Atention dans le cas N_from_free c'est différent. */
				e = gpu_data_not_used_list_begin(g->gpu_data);
				while (e->D != handle_popped)
				{
					  e = gpu_data_not_used_list_next(e);
				}
				
				//~ printf("Free task. Erase data %p from GPU %d\n", e->D, current_gpu - 1); fflush(stdout);
				gpu_data_not_used_list_erase(g->gpu_data, e);
				//~ print_data_not_used_yet();
				
				hud = e->D->user_data;
				
				#ifdef PRINT
				printf("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu - 1); fflush(stdout);
				#endif
				
				hud->is_present_in_data_not_used_yet[current_gpu - 1] = 0;
				e->D->user_data = hud;
				
				gpu_data_not_used_delete(e);
		}
		
		#ifdef PRINT
		printf("Erased data %p\n", e->D);
		print_data_not_used_yet_one_gpu(g->gpu_data, current_gpu - 1);
		#endif
		
		#ifdef PRINT
		printf("The data adding the most free tasks is %p and %d task.\n", handle_popped, number_free_task_max);
		#endif
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
		
		for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
		{
			data_available = true; 
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			{		    		
				if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
				{
					if (simulate_memory == 0)
					{
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
						{
							data_available = false;
							break;
						}
					}
					else if (simulate_memory == 1)
					{
						hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
						{
							data_available = false;
							break;
						}
					}
				}
			}
			if (data_available == true)
			{
				//~ #ifdef REFINED_MUTEX
				//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
				//~ #endif
				
				/* Add it from planned task compteur */
				increment_planned_task_data(t->pointer_to_T, current_gpu);
				
				#ifdef PRINT
				printf("Pushing free %p in planned_task of GPU %d :", t->pointer_to_T, current_gpu);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); i++)
				{
					printf(" %p", STARPU_TASK_GET_HANDLE(t->pointer_to_T, i));
				}
				printf("\n");
				print_data_on_node(1);
				#endif
								
				erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
				starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
				
				//~ #ifdef REFINED_MUTEX
				//~ STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
				//~ #endif
			}
		}
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		
		#ifdef PRINT_STATS
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
		#endif
	}
	/* La je change par rapport à 2D, si à la fois free et 1_from_free sont à 0 je renvoie random */   
	else if (number_1_from_free_task_max != 0 && app != 0) /* On prend une tâche de la donnée 1_from_free, dans l'ordre randomisé de la liste de tâches. 1 seule tâche est ajouté à planned_task. */
	{
		#ifdef PRINT_STATS
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		nb_1_from_free_choice++;
		#endif
		
		#ifdef PRINT
		printf("The data adding the most (%d) 1_from_free tasks is %p.\n", number_1_from_free_task_max, handle_popped);
		#endif
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
				
		/* OLD: Je vais regarder si la tâche est bien 1 from free. */
		//~ for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
		//~ {
			//~ /* Finding just one task that is one from free with the choosen data. */
			//~ data_not_available = 0; 
			//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			//~ {
				//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
				//~ {
					//~ if (simulate_memory == 0)
					//~ {
						//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
						//~ {
							//~ data_not_available++;
						//~ }
					//~ }
					//~ else if (simulate_memory == 1)
					//~ {
						//~ hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
						//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
						//~ {
							//~ data_not_available++;
						//~ }
					//~ }
				//~ }
			//~ }
			//~ if (data_not_available <= 1)
			//~ {
				//~ break;
			//~ }
		//~ }
		
		/* NEW */
		//~ t = best_1_from_free_task->pointer_to_T;
		//~ printf("best_1_from_free_task is %p\n", best_1_from_free_task);

		//~ if (t == task_using_data_list_end(handle_popped->sched_data))
		if (best_1_from_free_task == NULL)
		{
			#ifdef PRINT
			printf("Rien trouvé.\n"); fflush(stdout);
			#endif
			#ifdef PRINT_STATS
			nb_1_from_free_task_not_found++;
			#endif
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
						
			goto random;
		}
		
		struct gpu_data_not_used* e1 = NULL;
		
		/* Removing the datas from datanotused of the GPU. */		
		if (choose_best_data_from == 0) /* Que dans le cas où je choisis depuis la liste des pas encore utilisées. */
		{
			//~ print_task_info(best_1_from_free_task);
			/* J'efface toutes les données qui sont utilisé par la tâche 1_from_free que l'ont va retourner. */
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(best_1_from_free_task); i++)
			{
				if (!gpu_data_not_used_list_empty(g->gpu_data)) /* TODO : utile ? */
				{
					//~ for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
					for (e1 = gpu_data_not_used_list_begin(g->gpu_data); e1 != gpu_data_not_used_list_end(g->gpu_data); e1 = gpu_data_not_used_list_next(e1))
					{
						//~ printf("%p == %p ?\n", e->D, STARPU_TASK_GET_HANDLE(best_1_from_free_task, i)); fflush(stdout);
						//~ printf("%p == %p ?\n", e1->D, STARPU_TASK_GET_HANDLE(best_1_from_free_task, i)); fflush(stdout);
						//~ if(e->D == STARPU_TASK_GET_HANDLE(best_1_from_free_task, i))
						if(e1->D == STARPU_TASK_GET_HANDLE(best_1_from_free_task, i))
						{
							//~ printf("1 from free task. Erase data %p from GPU %d\n", e1->D, current_gpu - 1); fflush(stdout);
							//~ printf("1 from free task. Erase data %p from GPU %d\n", e->D, current_gpu - 1); fflush(stdout);
							//~ printf("Printing list I will erase from\n"); fflush(stdout);
							//~ print_data_not_used_yet_one_gpu(g, current_gpu - 1);
							//~ g->gpu_data = gpu_data_not_used_list_begin(g->gpu_data);
							//~ e = e1;
							//~ gpu_data_not_used_list_erase(g->gpu_data, e);
							//~ printf("%p\n", e1);
							
							gpu_data_not_used_list_erase(g->gpu_data, e1);
							/* TODO:  a chaque erase faire : */
							
							//~ remove_cell_from_gpu_data_not_used(g->gpu_data, e1);
							//~ print_data_not_used_yet();
							
							//~ hud = e->D->user_data;
							hud = e1->D->user_data;
							#ifdef PRINT
							//~ printf("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu - 1); fflush(stdout);
							printf("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e1->D, current_gpu - 1); fflush(stdout);
							#endif
							hud->is_present_in_data_not_used_yet[current_gpu - 1] = 0;
							//~ e->D->user_data = hud;
							e1->D->user_data = hud;
							
							/* TODO:  a chaque erase faire */
							gpu_data_not_used_delete(e1);
							
							/* TEST, refaire boucle apres pour suppr le deuxieme TODO */
							//~ i = STARPU_TASK_GET_NBUFFERS(best_1_from_free_task) + 1; break;
							break;
						}
					}
				}
			}
		}
						
		//~ increment_planned_task_data(t->pointer_to_T, current_gpu);
		increment_planned_task_data(best_1_from_free_task, current_gpu);
		
		#ifdef PRINT
		//~ printf("Pushing 1_from_free task %p in planned_task of GPU %d\n", t->pointer_to_T, current_gpu);
		printf("Pushing 1_from_free task %p in planned_task of GPU %d\n", best_1_from_free_task, current_gpu);
		#endif
		
		//~ erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
		erase_task_and_data_pointer(best_1_from_free_task, main_task_list);
		//~ starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
		starpu_task_list_push_back(&g->planned_task, best_1_from_free_task);
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		
		#ifdef PRINT_STATS
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
		#endif
		
	}
	else /* Sinon random */
	{
		#ifdef PRINT
		printf("Random selection because no data allow to get free or 1 from free tasks.\n"); fflush(stdout);
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		goto random;
	}
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&g->planned_task)) 
    {
		/* Si il y a eu un conflit, alors on veut recommencer au lieu de faire random. */
		//~ data_conflict = false;
		//~ Dopt[current_gpu - 1] = handle_popped;
		//~ for (i = 0; i < Ngpu; i++)
		//~ {
			//~ if(i != current_gpu - 1)
			//~ {
				if (data_conflict[current_gpu - 1] == true)
				{
					//printf("CRITICAL DATA CONFLICT! Iteration %d, %d task(s) out. Same data between GPU %d and GPU %d: %p.\n", iteration_DARTS, number_task_out_DARTS_2, current_gpu, i + 1, handle_popped); fflush(stdout);
					//~ printf("Goto\n");
					#ifdef PRINT_STATS
					number_critical_data_conflict++;
					number_data_conflict--;
					#endif
					#ifdef PRINT
					printf("Critical data conflict.\n"); fflush(stdout);
					#endif
					
					dynamic_data_aware_scheduling_3D_matrix(main_task_list, current_gpu, g);
					//~ goto debut_choix_Dopt;
					//~ data_conflict = true;
				}
			//~ }
		//~ }
	
		random: ; /* Va pop front main_task_list. Avec les listes de tâches randomisé ça reviens à faire du random. Si task_order == 2 comme on pourrait vouloir le faire avec dépendances, ça permet de garder l'ordre naturel. */
		
		/* TODO : a suppr ? */
		Dopt[current_gpu - 1] = NULL;

		#ifdef PRINT_STATS
		gettimeofday(&time_start_pick_random_task, NULL);
		number_random_selection++;
		#endif
						
		struct starpu_task *task = NULL;
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
		
		if (!starpu_task_list_empty(main_task_list))
		{
			#ifdef PRINT
			printf("Will pop random for GPU %d.\n", current_gpu); fflush(stdout);
			#endif
			
			task = starpu_task_list_pop_front(main_task_list);
		}
		else
		{
			#ifdef PRINT
			printf("Return void in scheduling for GPU %d.\n", current_gpu); fflush(stdout);
			#endif
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			return;
		}
		
		if (choose_best_data_from == 0)
		{	
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (!gpu_data_not_used_list_empty(g->gpu_data))
				{
					for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
					{
						if(e->D == STARPU_TASK_GET_HANDLE(task, i))
						{
							//~ printf("Random. Erase data %p from GPU %d\n", e->D, current_gpu - 1); fflush(stdout);
							gpu_data_not_used_list_erase(g->gpu_data, e);
							//~ print_data_not_used_yet();
							
							hud = e->D->user_data;
							
							#ifdef PRINT
							printf("%p gets 0 at is_present_in_data_not_used_yet GPU is %d\n", e->D, current_gpu - 1); fflush(stdout);
							#endif
							
							hud->is_present_in_data_not_used_yet[current_gpu - 1] = 0;
							e->D->user_data = hud;
							
							gpu_data_not_used_delete(e);
							break;
						}
					}
				}
			}
		}
		
		/* Add it from planned task compteur */
		increment_planned_task_data(task, current_gpu);
		
		#ifdef PRINT
		printf("For GPU %d, returning head of the randomized main task list: %p.\n", current_gpu, task);
		#endif
		
		erase_task_and_data_pointer(task, main_task_list);
		starpu_task_list_push_back(&g->planned_task, task);
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		
		#ifdef PRINT_STATS
		gettimeofday(&time_end_pick_random_task, NULL);
		time_total_pick_random_task += (time_end_pick_random_task.tv_sec - time_start_pick_random_task.tv_sec)*1000000LL + time_end_pick_random_task.tv_usec - time_start_pick_random_task.tv_usec;
		#endif
		return;
    }
    
    end_scheduling: ;
    
    #ifdef PRINT_STATS
    gettimeofday(&time_end_schedule, NULL);
    time_total_schedule += (time_end_schedule.tv_sec - time_start_schedule.tv_sec)*1000000LL + time_end_schedule.tv_usec - time_start_schedule.tv_usec;
	#endif
	
	/* TODO : besoin de le faire en haut et en bas de la fonction ? */
	Dopt[current_gpu - 1] = NULL;
}

void increment_planned_task_data(struct starpu_task *task, int current_gpu)
{
	/* Attention aux doubles içi!!! */
	int i = 0;
	//~ starpu_data_handle_t last_handle = NULL;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{	
		//~ if (last_handle != STARPU_TASK_GET_HANDLE(task, i))
		//~ {
			struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->nb_task_in_planned_task[current_gpu - 1] = hud->nb_task_in_planned_task[current_gpu - 1] + 1;
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
			//~ last_handle = STARPU_TASK_GET_HANDLE(task, i);
		//~ }
	}
}

/* Pour savoir si la donnée évincé est bien celle que l'on avais prévu.
 * Si ce n'est pas le cas ou si ca vaut NULL alors cela signifie qu'une donnée non prévu a 
 * été évincé. Il faut donc mettre à jour les listes dans les tâches et les données en conséquence.
 * Cependant si on est sur la fin de l'éxécution et que les éviction sont juste la pour vider la mémoire ce n'est pas
 * nécessaire. En réalité pour le moment je ne me rend pas compte qu'on est a la fin de l'exec. 
 * TODO : se rendre compte qu'on est a la fin et arreter de mettre à jour les listes du coup ?
 * Du coup je ne sais pas si c'est utile, à vérifier.
 */
 /* starpu_data_handle_t planned_eviction; */

void dynamic_data_aware_victim_eviction_failed(starpu_data_handle_t victim, void *component)
{	
	//~ #ifdef PRINT
	//~ printf("Début de victim eviction failed with data %p.\n", victim); fflush(stdout);
	//~ #endif
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
	#endif
		
	#ifdef PRINT_STATS
	gettimeofday(&time_start_evicted, NULL);
	victim_evicted_compteur++;
	#endif
		
	/* If a data was not truly evicted I put it back in the list. */
	//~ int i = 0;
			
	//~ my_planned_task_control->pointer = my_planned_task_control->first;
	//~ for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
	//~ {
		//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	//~ }
	//~ my_planned_task_control->pointer->data_to_evict_next = victim;
	tab_gpu_planned_task[starpu_worker_get_memory_node(starpu_worker_get_id()) - 1].data_to_evict_next = victim;
		
	#ifdef PRINT_STATS
	gettimeofday(&time_end_evicted, NULL);
	time_total_evicted += (time_end_evicted.tv_sec - time_start_evicted.tv_sec)*1000000LL + time_end_evicted.tv_usec - time_start_evicted.tv_usec;
	#endif
	
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
	#endif
}


/* Return NULL ou ne rien faire si la dernière tâche est sorti du post exec hook ? De même pour la mise à jour des listes à chaque eviction de donnée : J'ai pas la vision que la dernière tâche est sortie donc ce n'est pas possible.
 * Je rentre bcp trop dans cette fonction on perds du temps car le timing avance lui. Résolu en réduisant le threshold et en adaptant aussi CUDA_PIPELINE. */
starpu_data_handle_t dynamic_data_aware_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{
	//~ #ifdef PRINT
	//~ printf("Début de victim selector.\n"); fflush(stdout);
	//~ #endif
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
	#endif
	
	#ifdef PRINT_STATS
	victim_selector_compteur++;
	gettimeofday(&time_start_selector, NULL);
	#endif
	
    int i = 0;
    int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
    
    //~ #ifdef PRINT
	//~ printf("Début de victim_selector GPU %d node %d.\n", current_gpu, node); fflush(stdout);
	//~ #endif
    
    /* Se placer sur le bon GPU pour planned_task */
    //~ struct gpu_planned_task *temp_pointer = my_planned_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
    //~ {
		//~ temp_pointer = temp_pointer->next;
    //~ }

    /* Je check si une eviction n'a pas été refusé. */
    if (tab_gpu_planned_task[current_gpu - 1].data_to_evict_next != NULL) 
    {
		starpu_data_handle_t temp_handle = tab_gpu_planned_task[current_gpu - 1].data_to_evict_next;
		tab_gpu_planned_task[current_gpu - 1].data_to_evict_next = NULL;
		
		#ifdef PRINT_STATS
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		#endif
		
		/* TODO : pas vraiment une solution ces deux boucles non ? Est-ce vraiment utile ? J'ai pas l'impression. */
		if (!starpu_data_is_on_node(temp_handle, node))
		{ 			
			#ifdef PRINT
			printf("Refused %p is not on node %d. ??? Restart eviction\n", temp_handle, node); fflush(stdout); 
			#endif
			#ifdef PRINT_STATS
			victim_selector_refused_not_on_node++;
			#endif
			
			goto debuteviction; 
		}
		if (!starpu_data_can_evict(temp_handle, node, is_prefetch))
		{ 			
			#ifdef PRINT
			printf("Refused data can't be evicted ??? Restart eviction selection.\n"); fflush(stdout);
			#endif
			#ifdef PRINT_STATS
			victim_selector_refused_cant_evict++;
			#endif
			
			goto debuteviction; 
		}
		
		#ifdef PRINT
		printf("Evict refused data %p for GPU %d.\n", temp_handle, current_gpu); fflush(stdout);
		#endif
		#ifdef PRINT_STATS
		victim_selector_return_refused++;
		#endif
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		#ifdef LINEAR_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
		#endif
		
		return temp_handle;
    }
    
	debuteviction: ;
    
    /* Getting data on node. */
    starpu_data_handle_t *data_on_node;
    unsigned nb_data_on_node = 0;
    int *valid;
    starpu_data_handle_t returned_handle = STARPU_DATA_NO_VICTIM;
    starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);
      
   	/* Checking if all task are truly valid. TODO : a garder dans le cas avec dependances ? */
	//~ for (i = 0; i < nb_data_on_node; i++)
	//~ {
		//~ if (valid[i] == 0 && starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		//~ {
			//~ free(valid);
			//~ returned_handle = data_on_node[i];
			//~ free(data_on_node);
			
			//~ #ifdef PRINT
			//~ victim_selector_return_unvalid++;
			//~ printf("Return unvalid data %p.\n", returned_handle); fflush(stdout);
			//~ #endif
			
			//~ #ifdef REFINED_MUTEX
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			//~ #endif
			//~ #ifdef LINEAR_MUTEX
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
			//~ #endif
			
			//~ return returned_handle;
		//~ }
	//~ }
                
    /* Get the the min number of task a data can do in pulled_task */
    /* Se placer sur le bon GPU pour pulled_task */
	//~ my_pulled_task_control->pointer = my_pulled_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
 	//~ {
		//~ my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
	//~ }
    
    int min_number_task_in_pulled_task = INT_MAX;
    int nb_task_in_pulled_task[nb_data_on_node];
    
    for (i = 0; i < nb_data_on_node; i++)
    {
		nb_task_in_pulled_task[i] = 0;
    }
	
	//~ #ifdef REFINED_MUTEX
	//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	//~ #endif
	
	/* Je cherche le nombre de tâche dans le pulled_task que peut faire chaque données */
    struct handle_user_data *hud = malloc(sizeof(hud));
    for (i = 0; i < nb_data_on_node; i++)
    {
		if (starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		{
			hud = data_on_node[i]->user_data;
			nb_task_in_pulled_task[i] = hud->nb_task_in_pulled_task[current_gpu - 1];
			
			#ifdef PRINT
			printf("%d task in pulled_task for %p.\n", hud->nb_task_in_pulled_task[current_gpu - 1], data_on_node[i]);
			#endif
			
			/* Ajout : si sur les deux lists c'est 0 je la return direct la data */
			if (hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
			{
				#ifdef PRINT_STATS
				victim_selector_return_data_not_in_planned_and_pulled++;
				#endif
				
				#ifdef PRINT
				printf("%d task in planned task as well for %p.\n", hud->nb_task_in_pulled_task[current_gpu - 1], data_on_node[i]);
				#endif
				
				returned_handle = data_on_node[i];
				goto deletion_in_victim_selector;
			}
			
			if (hud->nb_task_in_pulled_task[current_gpu - 1] < min_number_task_in_pulled_task)
			{
				min_number_task_in_pulled_task = hud->nb_task_in_pulled_task[current_gpu - 1];
			}
		}
		else
		{
			/* - 1 si j'ai pas le droit d'évincer cette donnée */
			nb_task_in_pulled_task[i] = -1;
		}
    }
    
    //~ #ifdef PRINT
    //~ printf("Min number of task in pulled task = %d from %d data.\n", min_number_task_in_pulled_task, nb_data_on_node); 
	//~ #endif
	
	/* TODO : ce premier if on ne peut jamais entrer dedans normalement. A supprimer. */
    if (min_number_task_in_pulled_task == INT_MAX)
    {
		#ifdef PRINT_STATS
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		victim_selector_return_no_victim++;
		#endif
		#ifdef PRINT
		printf("Evict NO_VICTIM because min_number_task_in_pulled_task == INT_MAX.\n"); fflush(stdout);
		#endif
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		#ifdef LINEAR_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
		#endif
		
		return STARPU_DATA_NO_VICTIM;
    }
    else if (min_number_task_in_pulled_task == 0)
    {
		/* Au moins 1 donnée ne sert pas dans pulled_task */
		/* OLD */
		//~ returned_handle = min_weight_average_on_planned_task(data_on_node, nb_data_on_node, node, is_prefetch, temp_pointer, nb_task_in_pulled_task);
		
		/* NEW */
		returned_handle = least_used_data_on_planned_task(data_on_node, nb_data_on_node, nb_task_in_pulled_task, current_gpu);
    }
    else /* Au moins 1 donnée sert dans pulled_task */
    {
		/* Si c'est un prefetch qui demande une eviction de ce qui est utile pour les tâches de pulled task je renvoie NO VICTIM si >= à STARPU_TASK_PREFETCH */
		if (is_prefetch >= 1)
		{
			#ifdef PRINT_STATS
			gettimeofday(&time_end_selector, NULL);
			time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
			victim_selector_return_no_victim++;
			#endif
			#ifdef PRINT
			printf("Evict NO_VICTIM because is_prefetch >= 1.\n"); fflush(stdout);
			#endif
			
			#ifdef REFINED_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
			#endif
			#ifdef LINEAR_MUTEX
			STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
			#endif
		
			return STARPU_DATA_NO_VICTIM;
		}
		
		#ifdef PRINT_STATS
		victim_selector_belady++;
		#endif
		
		//~ returned_handle = belady_on_pulled_task(data_on_node, nb_data_on_node, node, is_prefetch, my_pulled_task_control->pointer);
		returned_handle = belady_on_pulled_task(data_on_node, nb_data_on_node, node, is_prefetch, &tab_gpu_pulled_task[current_gpu - 1]);
    }
    
    /* TODO: Ca devrait pas arriver a enleevr et a tester */
    if (returned_handle == NULL)
    {
		#ifdef PRINT_STATS
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		victim_selector_return_no_victim++;
		#endif
		#ifdef PRINT
		printf("Evict NO_VICTIM because returned_handle == NULL.\n"); fflush(stdout);
		#endif
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		#ifdef LINEAR_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
		#endif
		
		return STARPU_DATA_NO_VICTIM; 
    }
    
    deletion_in_victim_selector : ;

    //~ #ifdef REFINED_MUTEX
	//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	//~ #endif
    
    struct starpu_task *task = NULL;
    struct starpu_sched_component *temp_component = component;
    struct dynamic_data_aware_sched_data *data = temp_component->data;
    /* Enlever de la liste de tache a faire celles qui utilisais cette donnée. Et donc ajouter cette donnée aux données
     * à pop ainsi qu'ajouter la tache dans les données. Also add it to the main task list. */
        
    /* Suppression de la liste de planned task les tâches utilisant la donnée que l'on s'apprête à évincer. 
     * On ne veut pas de ce if en commentaires car on veut mettre à jour même quand la donnée évincé viens de pulled task. */
    //~ if (min_number_task_in_pulled_task == 0)
    //~ {
		for (task = starpu_task_list_begin(&tab_gpu_planned_task[current_gpu - 1].planned_task); task != starpu_task_list_end(&tab_gpu_planned_task[current_gpu - 1].planned_task); task = starpu_task_list_next(task))
		{
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
				{				
					/* Suppression de la liste de tâches à faire */
					struct pointer_in_task *pt = task->sched_data;
					starpu_task_list_erase(&tab_gpu_planned_task[current_gpu - 1].planned_task, pt->pointer_to_cell);
						
					pt->pointer_to_cell = task;
					pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
					pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
						
					for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
					{
						/* Pointer toward the main task list in the handles. */
						struct task_using_data *e = task_using_data_new();
						e->pointer_to_T = task;
						
						if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL) 
						{
							struct task_using_data_list *tl = task_using_data_list_new();
							task_using_data_list_push_front(tl, e);
							STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
						}
						else
						{
							task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
						}
							
						/* Adding the pointer in the task. */
						pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
						pt->tud[i] = e;
						
						/* Increase expected length of task using this data */
						struct handle_user_data* hud = pt->pointer_to_D[i]->user_data;
						hud->sum_remaining_task_expected_length += starpu_task_expected_length(task, perf_arch, 0);
						
						//~ #ifdef PRINT
						//~ printf("Eviction of data %p.\n", STARPU_TASK_GET_HANDLE(task, i));
						//~ #endif
						
						pt->pointer_to_D[i]->user_data = hud;
						
					}	
					task->sched_data = pt;
						
					/* Ajout a la liste de tâches principales ces mêmes tâches. */
					starpu_task_list_push_back(&data->main_task_list, task);
					break;
				}
			}
		}   
    //~ }
	
    /* Placing in a random spot of the data list to use the evicted handle. */
    /* Je ne le fais pas dans le cas ou on choisis depuis la mémoire.
     * Je ne le fais pas non plus si la donnée ne sert plus à rien.
     * dans le cas avec dépendances, si elle sert encore, je vérifie sur les autres GPUs
     * si ils ont unecopie valide de la donnée dans leurs not used yet. Si non je la 
     * rajoute. */
    if (choose_best_data_from == 0)
    {
		if (!task_using_data_list_empty(returned_handle->sched_data))
		{						
			if (dependances == 1) /* Checking if other GPUs have this handle in datanotusedtyet. */
			{
				struct handle_user_data * hud = returned_handle->user_data;
				if (hud->is_present_in_data_not_used_yet[current_gpu - 1] == 0)
				{
					//~ #ifdef PRINT
					//~ printf("Pushing %p on GPU %d at random spot.\n", returned_handle, current_gpu - 1); fflush(stdout);
					//~ #endif
					
					push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[current_gpu - 1], current_gpu - 1);
				}

				//~ struct gpu_data_not_used *e = NULL;
				for (i = 0; i < Ngpu; i++)
				{
					if (i != current_gpu - 1)
					{
						//~ printf("Checking GPU %d.\n", i);

						/* Version où je check à la main */
						//~ for (e = gpu_data_not_used_list_begin(tab_gpu_planned_task[i].gpu_data); e != gpu_data_not_used_list_end(tab_gpu_planned_task[i].gpu_data); e = gpu_data_not_used_list_next(e))
						//~ {
							//~ if (e->D == returned_handle)
							//~ {
								//~ goto data_already_present;
								//~ break;
							//~ }
						//~ }
						//~ printf("Not there, pusing data %p on GPU %d at random spot\n", returned_handle, i);
						//~ push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[i], i);
						//~ print_data_not_used_yet_one_gpu(&tab_gpu_planned_task[i], i);
						//~ data_already_present: ;
						
						/* Version où j'utilise is_present_in_data_not_used_yet de la struct de handle_user_data */
						if (hud->is_present_in_data_not_used_yet[i] == 0)
						{
							//~ #ifdef PRINT
							//~ printf("Not there, pushing data %p on GPU %d at random spot\n", returned_handle, i); fflush(stdout);
							//~ #endif
							
							push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[i], i);
						}
					}
				}
			}
			else
			{
				//~ #ifdef PRINT
				//~ printf("Pushing %p on GPU %d at random spot.\n", returned_handle, current_gpu - 1); fflush(stdout);
				//~ #endif
				
				push_data_not_used_yet_random_spot(returned_handle, &tab_gpu_planned_task[current_gpu - 1], current_gpu - 1);
			}
		}
	}
	
    #ifdef PRINT_STATS
    gettimeofday(&time_end_selector, NULL);
	time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
	#endif
	#ifdef PRINT
	printf("Evict %p on GPU %d.\n", returned_handle, current_gpu); fflush(stdout);
	#endif
	
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
	#endif
	
    return returned_handle;
}

starpu_data_handle_t belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_pulled_task *g)
{
	#ifdef PRINT_STATS
	gettimeofday(&time_start_belady, NULL);
	#endif
    int i = 0;
    int j = 0;
    int index_next_use = 0;
    int max_next_use = -1;
    struct pulled_task *p = pulled_task_new();
    starpu_data_handle_t returned_handle = NULL;
    
    //print_pulled_task_one_gpu(g, node);
    for (i = 0; i < nb_data_on_node; i++)
    {
		if (starpu_data_can_evict(data_tab[i], node, is_prefetch)) /* TODO : il y aurait moyen de remplacer ce can evict juste par une lecture dans un tableau car de toute facon on le fais avant dans victim_selector. */
		{
			index_next_use = 0;
			for (p = pulled_task_list_begin(g->ptl); p != pulled_task_list_end(g->ptl); p = pulled_task_list_next(p))
			{
				for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
				{
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
			/* TODO: le break nested loop peut etre dans l'accolad suivante non ? */
			break_nested_for_loop : ;
		}
    }

    #ifdef PRINT_STATS
    gettimeofday(&time_end_belady, NULL);
    time_total_belady += (time_end_belady.tv_sec - time_start_belady.tv_sec)*1000000LL + time_end_belady.tv_usec - time_start_belady.tv_usec;
    #endif
    
    return returned_handle;
}

/* Belady sur lesp lanned task. Je check pas is on node car c'est -1 dans le tab du nb de taches dans pulled task */
starpu_data_handle_t least_used_data_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, int *nb_task_in_pulled_task, int current_gpu)
{
	#ifdef PRINT_STATS
	gettimeofday(&time_start_least_used_data_planned_task, NULL);
    #endif
    
    int i = 0;
    int min_nb_task_in_planned_task = INT_MAX;
    starpu_data_handle_t returned_handle = NULL;
    
    struct handle_user_data *hud = malloc(sizeof(hud));
    
    for (i = 0; i < nb_data_on_node; i++)
    {
		if (nb_task_in_pulled_task[i] == 0)
		{
			hud = data_tab[i]->user_data;
			
			if (hud->nb_task_in_planned_task[current_gpu - 1] < min_nb_task_in_planned_task)
			{
				min_nb_task_in_planned_task = hud->nb_task_in_planned_task[current_gpu - 1];
				returned_handle = data_tab[i];
			}
		}
	}
	
	#ifdef PRINT_STATS
	gettimeofday(&time_end_least_used_data_planned_task, NULL);
	time_total_least_used_data_planned_task += (time_end_least_used_data_planned_task.tv_sec - time_start_least_used_data_planned_task.tv_sec)*1000000LL + time_end_least_used_data_planned_task.tv_usec - time_start_least_used_data_planned_task.tv_usec;
    #endif
    
    return returned_handle;
}

/* Erase a task from the main task list.
 * Also erase pointer in the data.
 * Only of one GPU.
 * Also update the expected length of task using this data.
 */
void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l)
{
    int j = 0;
    struct pointer_in_task *pt = task->sched_data;
    
    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
    {
		if (pt->tud[j] != NULL) 
		{
			task_using_data_list_erase(pt->pointer_to_D[j]->sched_data, pt->tud[j]);
			pt->tud[j] = NULL;
		}
		
		/* Reduce expected length of task using this data */
		struct handle_user_data* hud = pt->pointer_to_D[j]->user_data;
		hud->sum_remaining_task_expected_length -= starpu_task_expected_length(task, perf_arch, 0);
		
		#ifdef PRINT
		printf("Adding in planned task. Expected length in data %p: %f\n", STARPU_TASK_GET_HANDLE(task, j), hud->sum_remaining_task_expected_length);
		#endif
		
		pt->pointer_to_D[j]->user_data = hud;
		
    }
    starpu_task_list_erase(l, pt->pointer_to_cell);
}

static int dynamic_data_aware_can_push(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	#ifdef PRINT
	printf("Début de dynamic_data_aware_can_push.\n"); fflush(stdout);
	#endif
    //~ struct dynamic_data_aware_sched_data *data = component->data;
    int didwork = 0;
    struct starpu_task *task;
    task = starpu_sched_component_pump_to(component, to, &didwork);
    if (task)
    {	    
	    /* If a task is refused I push it in the refused fifo list of the appropriate GPU's package.
	     * This list is looked at first when a GPU is asking for a task so we don't break the planned order. */
	     
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
		#ifdef LINEAR_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
		#endif
		
		#ifdef PRINT
		printf("Refused %p in can_push.\n", task); fflush(stdout);
		#endif
		#ifdef PRINT_STATS
		nb_refused_task++;
		#endif
		
	    //~ my_planned_task_control->pointer = my_planned_task_control->first;
	    //~ int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	    //~ for (int i = 1; i < current_gpu; i++) 
	    //~ {
			//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	    //~ }
	    starpu_task_list_push_back(&tab_gpu_planned_task[starpu_worker_get_memory_node(starpu_worker_get_id()) - 1].refused_fifo_list, task);
	    
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
		#ifdef LINEAR_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
		#endif
    }
    
    /* There is room now */
    return didwork || starpu_sched_component_can_push(component, to);
}

static int dynamic_data_aware_can_pull(struct starpu_sched_component *component)
{
    return starpu_sched_component_can_pull(component);
}

//~ void gpu_planned_task_initialisation()
//~ {
    //~ _STARPU_MALLOC( my_planned_task_control, sizeof(*my_planned_task_control));
    //~ struct gpu_planned_task *new = malloc(sizeof(*new));
    
    //~ starpu_task_list_init(&new->planned_task);
    //~ starpu_task_list_init(&new->refused_fifo_list);
    //~ new->data_to_evict_next = NULL;
    //~ new->next = NULL;
    //~ new->first_task = true;
    //~ new->number_data_selection = 0;
    
    //~ /* Pour les init avec dépendances */
    //~ new->gpu_data = gpu_data_not_used_list_new();
    //~ new->new_gpu_data = gpu_data_not_used_list_new();
    
    //~ my_planned_task_control->pointer = new;
    //~ my_planned_task_control->first = my_planned_task_control->pointer;
//~ }

//~ void gpu_planned_task_insertion()
//~ {
    //~ struct gpu_planned_task *new = malloc(sizeof(*new));
    
    //~ starpu_task_list_init(&new->planned_task);
    //~ starpu_task_list_init(&new->refused_fifo_list);
    //~ new->data_to_evict_next = NULL;
    //~ new->next = my_planned_task_control->pointer;
    //~ new->first_task = true;
    //~ new->number_data_selection = 0;
    
	//~ /* Pour les init avec dépendances */
    //~ new->gpu_data = gpu_data_not_used_list_new();
    //~ new->new_gpu_data = gpu_data_not_used_list_new();
    
    //~ my_planned_task_control->pointer = new;
//~ }

//~ void gpu_pulled_task_initialisation()
//~ {
    //~ _STARPU_MALLOC(my_pulled_task_control, sizeof(*my_pulled_task_control));
    //~ struct gpu_pulled_task *new = malloc(sizeof(*new));
    //~ struct pulled_task_list *p = pulled_task_list_new();
    //~ new->ptl = p;
    
    //~ my_pulled_task_control->pointer = new;
    //~ my_pulled_task_control->first = my_pulled_task_control->pointer;
//~ }

//~ void gpu_pulled_task_insertion()
//~ {
    //~ struct gpu_pulled_task *new = malloc(sizeof(*new));
    //~ struct pulled_task_list *p = pulled_task_list_new();
    //~ new->ptl = p;
    
    //~ new->next = my_pulled_task_control->pointer;    
    //~ my_pulled_task_control->pointer = new;
//~ }

void tab_gpu_planned_task_init()
{
    int i = 0;
	for (i = 0; i < Ngpu; i++)
	{
		//~ struct gpu_planned_task *new = malloc(sizeof(*new));
		
		starpu_task_list_init(&tab_gpu_planned_task[i].planned_task);
		starpu_task_list_init(&tab_gpu_planned_task[i].refused_fifo_list);
		tab_gpu_planned_task[i].data_to_evict_next = NULL;
		tab_gpu_planned_task[i].first_task = true;
		tab_gpu_planned_task[i].number_data_selection = 0;
		
		/* Pour les init avec dépendances */
		//~ tab_gpu_planned_task[i].gpu_data = gpu_data_not_used_list_new();
		//~ tab_gpu_planned_task[i].new_gpu_data = gpu_data_not_used_list_new();
		//~ gpu_data_not_used_list_init(&tab_gpu_planned_task[i].gpu_data);
		//~ gpu_data_not_used_list_init(&tab_gpu_planned_task[i].new_gpu_data);
		
		struct gpu_data_not_used_list *p = gpu_data_not_used_list_new();
		tab_gpu_planned_task[i].gpu_data = p;
		gpu_data_not_used_list_init(tab_gpu_planned_task[i].gpu_data);
		
		if (data_order == 1)
		{
			struct gpu_data_not_used_list *p = gpu_data_not_used_list_new();
			tab_gpu_planned_task[i].new_gpu_data = p;
			gpu_data_not_used_list_init(tab_gpu_planned_task[i].new_gpu_data);
		}

		tab_gpu_planned_task[i].first_task_to_pop = NULL;
	}
}


void tab_gpu_pulled_task_init()
{
	int i = 0;
	for (i = 0; i < Ngpu; i++)
	{
		struct pulled_task_list *p = pulled_task_list_new();
		tab_gpu_pulled_task[i].ptl = p;
		tab_gpu_pulled_task[i].test = 0;
	}
}

void add_task_to_pulled_task(int current_gpu, struct starpu_task *task)
{
	int i = 0;	

	/* J'incrémente le nombre de tâches dans pulled task pour les données de task */
    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		//~ hud->nb_task_in_pulled_task[current_gpu - 1] = hud->nb_task_in_pulled_task[current_gpu - 1] + 1;
		hud->nb_task_in_pulled_task[current_gpu - 1] += 1;
		STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
	}
	
    struct pulled_task *p = pulled_task_new();
    p->pointer_to_pulled_task = task;
    
    //~ my_pulled_task_control->pointer = my_pulled_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
    //~ {
		//~ my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    //~ }
    //~ pulled_task_list_push_back(my_pulled_task_control->pointer->ptl, p);
    pulled_task_list_push_back(tab_gpu_pulled_task[current_gpu - 1].ptl, p);
}

struct starpu_sched_component *starpu_sched_component_dynamic_data_aware_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{	
	/* TODO a suppr */
	total_task_done = 0;
	
	/* Var globale pour n'appeller qu'une seule fois get_env_number */
	eviction_strategy_dynamic_data_aware = starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_DATA_AWARE", 0);
	threshold = starpu_get_env_number_default("THRESHOLD", 0);
	app = starpu_get_env_number_default("APP", 0);
	choose_best_data_from = starpu_get_env_number_default("CHOOSE_BEST_DATA_FROM", 0);
	simulate_memory = starpu_get_env_number_default("SIMULATE_MEMORY", 0);
	task_order = starpu_get_env_number_default("TASK_ORDER", 0);
	data_order = starpu_get_env_number_default("DATA_ORDER", 0);
	dependances = starpu_get_env_number_default("DEPENDANCES", 0);
	prio = starpu_get_env_number_default("PRIO", 0);
	free_pushed_task_position = starpu_get_env_number_default("FREE_PUSHED_TASK_POSITION", 0);
	graph_descendants = starpu_get_env_number_default("GRAPH_DESCENDANTS", 0); /* 0 ou 1 ou 2 */
	dopt_selection_order = starpu_get_env_number_default("DOPT_SELECTION_ORDER", 0);
	
	printf("-----\nEVICTION_STRATEGY_DYNAMIC_DATA_AWARE = %d\nTHRESHOLD = %d\nAPP = %d\nCHOOSE_BEST_DATA_FROM = %d\nSIMULATE_MEMORY = %d\nTASK_ORDER = %d\nDATA_ORDER = %d\nDEPENDANCES = %d\nPRIO = %d\nFREE_PUSHED_TASK_POSITION = %d\nGRAPH_DESCENDANTS = %d\nDOPT_SELECTION_ORDER = %d\nPRIORITY_ATTRIBUTION = %d\n-----\n", eviction_strategy_dynamic_data_aware, threshold, app, choose_best_data_from, simulate_memory, task_order, data_order, dependances, prio, free_pushed_task_position, graph_descendants, dopt_selection_order, starpu_get_env_number_default("PRIORITY_ATTRIBUTION", 0));
	
	/* Initialization of global variables. */
	Ngpu = get_number_GPU();
	NT_DARTS = 0;
	new_tasks_initialized = false;
	gpu_memory_initialized = false;
	
	int i = 0;

	/* Prints and stats. */
	#if defined PRINT || defined PRINT_STATS || defined PRINT_PYTHON
	print_in_terminal = starpu_get_env_number_default("PRINT_IN_TERMINAL", 0);
	print3d = starpu_get_env_number_default("PRINT3D", 0);
	print_n = starpu_get_env_number_default("PRINT_N", 0);
	print_time = starpu_get_env_number_default("PRINT_TIME", 0);
	#endif
	
	#ifdef PRINT_STATS
	/* If I want to empty the files. I don't do it if I want to test on multiple working set sizes. Pour data_choosen je ne le fais pas pour différentes working set sizes de toute facon. */
	FILE *f = NULL;
	char str[2];
	int size = strlen("Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_.csv") + strlen(str);
	char* path = NULL;
	for (i = 0; i < Ngpu; i++)
	{
		path = (char *)malloc(size);
		sprintf(str, "%d", i + 1); /* To get the index of the current GPU */
		strcpy(path, "Output_maxime/Data/DARTS/DARTS_data_choosen_stats_GPU_");
		strcat(path, str);
		strcat(path, ".csv");
		f = fopen(path, "w");
		fprintf(f, "Data selection,Data choosen,Number of data read,Number of task added in planned_task\n");
		fclose(f);
		free(path);
	}
	
	/* A commenter pour ces 5 fichiers si on veut tester sur plusieurs working set. */
	//~ f = fopen("Output_maxime/Data/DARTS/Nb_conflit_donnee.csv", "w");
	//~ fprintf(f, "N,Nb conflits,Nb conflits critiques\n");
	//~ fclose(f);
	
	//~ f = fopen("Output_maxime/Data/DARTS/Choice_during_scheduling.csv", "w");
	//~ fprintf(f, "N,Return NULL, Return task, Return NULL because main task list empty,Nb of random selection,nb_1_from_free_task_not_found\n");
	//~ fclose(f);
	
	//~ f = fopen("Output_maxime/Data/DARTS/Choice_victim_selector.csv", "w");
	//~ fprintf(f, "N,victim_selector_refused_not_on_node,victim_selector_refused_cant_evict,victim_selector_return_refused,victim_selector_return_unvalid,victim_selector_return_data_not_in_planned_and_pulled,victim_evicted_compteur,victim_selector_compteur,victim_selector_return_no_victim,victim_selector_belady\n");
	//~ fclose(f);
	
	//~ f = fopen("Output_maxime/Data/DARTS/Misc.csv", "w");
	//~ fprintf(f, "N,Nb refused tasks,Nb new task initialized\n");
	//~ fclose(f);
	
	//~ f = fopen("Output_maxime/Data/DARTS/DARTS_time.csv", "w");
	//~ fprintf(f, "N,selector,evicted,belady,schedule,choose_best_data,fill_planned_task_list,initialisation, randomize, pick_random_task,least_used_data_planned_task,createtolasttaskfinished\n");
	//~ fclose(f);
	
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
	
	#ifdef PRINT_PYTHON
	/* For visualisation in python. */
	index_current_popped_task = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_all_gpu = 0;
	index_current_popped_task_all_gpu_prefetch = 0;
	#endif
	
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_data_aware");
	srandom(starpu_get_env_number_default("SEED", 0));
	
	/* Initialization of structures. */
	struct dynamic_data_aware_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	//~ STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->main_task_list);
	
	/* Initialisation des structs de liste de tâches */
	//~ gpu_planned_task_initialisation();
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
	    //~ gpu_planned_task_insertion();
	//~ }
	//~ my_planned_task_control->first = my_planned_task_control->pointer;
	
	//~ gpu_pulled_task_initialisation();
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
	    //~ gpu_pulled_task_insertion();
	//~ }
	//~ my_pulled_task_control->first = my_pulled_task_control->pointer;
	//~ printf("%d GPU(s)\n", Ngpu); fflush(stdout);
	tab_gpu_planned_task = malloc(Ngpu*sizeof(struct gpu_planned_task));
	tab_gpu_planned_task_init();
	tab_gpu_pulled_task = malloc(Ngpu*sizeof(struct gpu_pulled_task));
	tab_gpu_pulled_task_init();
	
	/* Initialisation des mutexs. */
	#ifdef REFINED_MUTEX
	STARPU_PTHREAD_MUTEX_INIT(&refined_mutex, NULL);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_INIT(&linear_mutex, NULL);
	#endif
	
	/* Pour gérer les conflits si c'est nécessaire. */
	Dopt = malloc(Ngpu*sizeof(starpu_data_handle_t));
	for (i = 0; i < Ngpu; i++)
	{
		Dopt[i] = NULL;
	}	
	data_conflict = malloc(Ngpu*sizeof(bool));

	component->data = data;
	/* component->do_schedule = dynamic_data_aware_do_schedule; */
	component->push_task = dynamic_data_aware_push_task;
	component->pull_task = dynamic_data_aware_pull_task;
	component->can_push = dynamic_data_aware_can_push;
	component->can_pull = dynamic_data_aware_can_pull;
	
	/* TODO: Aussi faire cela pour HFP. */
	if (eviction_strategy_dynamic_data_aware == 1) 
	{
	    starpu_data_register_victim_selector(dynamic_data_aware_victim_selector, dynamic_data_aware_victim_eviction_failed, component); 
	}
	
	return component;
}

static void initialize_dynamic_data_aware_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_dynamic_data_aware_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW | /* Ne ré-ordonne plus par prio maintenant */
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			//~ STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP | /* Pas utile */
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
	
	perf_arch = starpu_worker_get_perf_archtype(0, sched_ctx_id); /* Getting the perfmodel. Used to get the expected length of a task to tiebreak when choosing Dopt. I put 0 in place of worker id because I assume we are in an homogenous case with only identical GPUs. */
	
	if (prio != 0)
	{
		/* GRAPHE */
		if (graph_descendants != 0)
		{
			_starpu_graph_record = 1;
		}
		
		/* To initialize and get prioriies on each tasks (it's the application that set priorities. It reduces my perfs ? why ? It takes time ? Test with -no prio see if the perfs stays the same */
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
	}
}

static void deinitialize_dynamic_data_aware_center_policy(unsigned sched_ctx_id)
{
	//~ printf("Début de deinitialize\n"); fflush(stdout);
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

/* Get the task that was last executed. Used to update the task list of pulled task	 */
void get_task_done(struct starpu_task *task, unsigned sci)
{
	#ifdef PRINT
	printf("Début de get task done.\n"); fflush(stdout);
	#endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_LOCK(&linear_mutex);
	#endif
	//~ #ifdef REFINED_MUTEX
	//~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
	//~ #endif
	
	total_task_done++; /* TODO : utile ? */
	
	#ifdef PRINT
	printf("%dème task done in the post_exec_hook: %p.\n", total_task_done, task); fflush(stdout);
	#endif
		
	/* Je me place sur la liste correspondant au bon gpu. */
	int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	int i = 0;
		
	if (eviction_strategy_dynamic_data_aware == 1) 
	{
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
		#endif
		
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			struct handle_user_data* hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->nb_task_in_pulled_task[current_gpu - 1] -= 1;
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		
		#ifdef REFINED_MUTEX
		STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
		#endif
	}
	
	    
    struct pulled_task *temp = NULL;
    int trouve = 0;
	
    /* J'efface la tâche dans la liste de tâches */
    if (!pulled_task_list_empty(tab_gpu_pulled_task[current_gpu - 1].ptl))
    {
		for (temp = pulled_task_list_begin(tab_gpu_pulled_task[current_gpu - 1].ptl); temp != pulled_task_list_end(tab_gpu_pulled_task[current_gpu - 1].ptl); temp = pulled_task_list_next(temp))
		{	
			if (temp->pointer_to_pulled_task == task)
			{
				trouve = 1;
				break;
			}
		}
		if (trouve == 1)
		{
			#ifdef PRINT
			printf("Popped task in get task done is %p.\n", temp->pointer_to_pulled_task); fflush(stdout);
			#endif
			
			pulled_task_list_erase(tab_gpu_pulled_task[current_gpu - 1].ptl, temp);
			
			/* New delete */
			pulled_task_delete(temp);
		}
    }
    
    //~ #ifdef REFINED_MUTEX /* TODO suppr ce mutex ? */
    //~ STARPU_PTHREAD_MUTEX_LOCK(&refined_mutex);
    //~ #endif
    
    //~ number_task_out_DARTS_2++; /* TODO utile cela ? */
    /* Reset pour prochaine itération, a modifier */
    //~ if (iteration_DARTS == 10)
	//~ {
		//~ printf("RESET in get task done\n"); fflush(stdout);
		//~ number_task_out_DARTS_2 = 0;
		//~ reset_all_struct();
		//~ need_to_reinit = true;
		
		/* TODO : commenté car je ne sais pas gérer plusieurs itérions pour le moment. */
		//~ iteration_DARTS++;
		
		/* TODO : a suppr */
		//~ if (iteration_DARTS == 10)
		//~ {
			//~ FILE *f2 = fopen("Output_maxime/Data/Nb_conflit_donnee.txt", "a");
			//~ fprintf(f2 , "%d\n", number_data_conflict);
			//~ fclose(f2);
			//~ f2 = fopen("Output_maxime/Data/Nb_conflit_donnee_critique.txt", "a");
			//~ fprintf(f2 , "%d\n", number_critical_data_conflict);
			//~ fclose(f2);
		//~ }
		
		//~ #ifdef PRINT /* TODO : Il faudrat metre 10 la cr la 11ème je ne la ++ pas dans la fonction de l'appli ? */
		//~ if ((iteration_DARTS == 10 && starpu_get_env_number_default("PRINT_TIME", 0) == 1) || starpu_get_env_number_default("PRINT_TIME", 0) == 2) //PRINT_TIME = 2 pour quand on a 1 seule itération
		//~ {
			//~ gettimeofday(&time_end_createtolasttaskfinished, NULL);
			//~ time_total_createtolasttaskfinished += (time_end_createtolasttaskfinished.tv_sec - time_start_createtolasttaskfinished.tv_sec)*1000000LL + time_end_createtolasttaskfinished.tv_usec - time_start_createtolasttaskfinished.tv_usec;

			//~ int print_N = 0;
			//~ if (app == 0) // Cas M2D
			//~ {
				//~ print_N = sqrt(NT_DARTS);
			//~ }
			//~ else // Cas M3D
			//~ {
				//~ print_N = sqrt(NT_DARTS)/2;
			//~ }
			//~ if (threshold == 1)
			//~ {
				//~ FILE *f = fopen("Output_maxime/DARTS_time.txt", "a");
				//~ fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				//~ fclose(f);
			//~ }
			//~ else if (choose_best_data_from == 1)
			//~ {
				//~ FILE *f = fopen("Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt", "a");
				//~ fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				//~ fclose(f);
			//~ }
			//~ else
			//~ {
				//~ FILE *f = fopen("Output_maxime/DARTS_time_no_threshold.txt", "a");
				//~ fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				//~ fclose(f);
			//~ }
			//~ printf("Nombre d'entrée dans victim selector = %d, nombre de return no victim = %d. Temps passé dans victim_selector = %lld.\n", victim_selector_compteur, victim_selector_return_no_victim, time_total_selector);
			//~ printf("Nombre d'entrée dans Belady = %d. Temps passé dans Belady = %lld.\n", victim_selector_belady, time_total_belady);
			//~ printf("Nombre d'entrée dans victim evicted = %d. Temps passé dans victim_evicted = %lld.\n", victim_evicted_compteur, time_total_evicted);
			//~ printf("Nombre de choix random = %d.\n", number_random_selection);
		//~ }
		//~ #endif
	//~ }
	
	//~ #ifdef REFINED_MUTEX
	//~ STARPU_PTHREAD_MUTEX_UNLOCK(&refined_mutex);
	//~ #endif
	#ifdef LINEAR_MUTEX
	STARPU_PTHREAD_MUTEX_UNLOCK(&linear_mutex);
	#endif

	#ifdef PRINT
	printf("End of get_task_done\n"); fflush(stdout);
	#endif
	
    starpu_sched_component_worker_pre_exec_hook(task, sci);
}

void get_submit_hook(struct starpu_task *task)
{
	#ifdef PRINT
	printf("Saw submission of task %p prio %d.\n", task, task->priority); fflush(stdout);
	#endif
}

#ifdef PRINT_PYTHON
/* Version avec print et visualisation */
struct starpu_sched_policy _starpu_sched_dynamic_data_aware_policy =
{
	.init_sched = initialize_dynamic_data_aware_center_policy,
	.deinit_sched = deinitialize_dynamic_data_aware_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	//~ .submit_hook = get_submit_hook,
	/* .do_schedule = starpu_sched_tree_do_schedule, */
	.push_task = starpu_sched_tree_push_task,
	/* //~ .pop_task = starpu_sched_tree_pop_task, */
	.pop_task = get_data_to_load,
	/* .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook, */
	.pre_exec_hook = get_current_tasks,
	/* .post_exec_hook = starpu_sched_component_worker_post_exec_hook, */
	.post_exec_hook = get_task_done,
	.pop_every_task = NULL,
	.policy_name = "dynamic-data-aware",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading the data adding the most tasks",
	.worker_type = STARPU_WORKER_LIST,
};
#else
/* Version pour performances */
struct starpu_sched_policy _starpu_sched_dynamic_data_aware_policy =
{
	.init_sched = initialize_dynamic_data_aware_center_policy,
	.deinit_sched = deinitialize_dynamic_data_aware_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	//~ .submit_hook = get_submit_hook,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = get_task_done, /* Utile pour la stratégie d'éviction */
	//~ /* Mettre un data unregister qui oublie les données temporaires, existe deja starpu_data_unregister y ajouter l'appel a la methode de l'ordo et en plus un reste ailleurs (depuis l'appli). Le unregsiter sera utile pour dautrs aplli comme QRmems */
	.pop_every_task = NULL,
	.policy_name = "dynamic-data-aware",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading the data adding the most tasks",
	.worker_type = STARPU_WORKER_LIST,
};
#endif
