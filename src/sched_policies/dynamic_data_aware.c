/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021	Maxime Gonthier
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

/* Dynamic Data Aware scheduling. Pop best data.
 * Computes all task using this data and the data already loaded on memory.
 * If no task is available compute a random task not computed yet.
 */
 
#include <schedulers/HFP.h>
#include <schedulers/dynamic_data_aware.h>
#include "helper_mct.h"

/* Var globales déclaré en extern */
int eviction_strategy_dynamic_data_aware;
int threshold;
int app;
int choose_best_data_from;
int simulate_memory;
//~ int data_order;
int natural_order;
//~ int erase_data_strategy;

bool gpu_memory_initialized;
bool new_tasks_initialized;
struct gpu_planned_task_control *my_planned_task_control;
struct gpu_pulled_task_control *my_pulled_task_control;
int number_task_out_DARTS; /* Utile pour savoir quand réinit quand il y a plusieurs itérations. */
int NT_dynamic_outer;
int iteration_DARTS;

/* TODO a supprimer pour meilleures perfs ? */
#ifdef PRINT
//~ int victim_evicted_compteur = 0;
//~ int victim_selector_compteur = 0;
//~ int victim_selector_return_no_victim = 0;
//~ int victim_selector_belady = 0;
int number_random_selection = 0;
struct timeval time_start_selector;
struct timeval time_end_selector;
long long time_total_selector = 0;
struct timeval time_start_evicted;
struct timeval time_end_evicted;
long long time_total_evicted = 0;
struct timeval time_start_belady;
struct timeval time_end_belady;
long long time_total_belady = 0;
struct timeval time_start_schedule;
struct timeval time_end_schedule;
long long time_total_schedule = 0;
struct timeval time_start_choose_best_data;
struct timeval time_end_choose_best_data;
long long time_total_choose_best_data = 0;
struct timeval time_start_fill_planned_task_list;
struct timeval time_end_fill_planned_task_list;
long long time_total_fill_planned_task_list = 0;
struct timeval time_start_initialisation;
struct timeval time_end_initialisation;
long long time_total_initialisation = 0;
struct timeval time_start_randomize;
struct timeval time_end_randomize;
long long time_total_randomize = 0;
struct timeval time_start_pick_random_task;
struct timeval time_end_pick_random_task;
long long time_total_pick_random_task = 0;
struct timeval time_start_least_used_data_planned_task;
struct timeval time_end_least_used_data_planned_task;
long long time_total_least_used_data_planned_task = 0;
struct timeval time_start_createtolasttaskfinished;
struct timeval time_end_createtolasttaskfinished;
long long time_total_createtolasttaskfinished = 0;
#endif

void print_task_list(struct starpu_task_list *l, char *s)
{
    int i = 0;
    printf("%s :\n", s);
    for (struct starpu_task *task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
    {
		printf("%p:", task);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			printf("	%p", STARPU_TASK_GET_HANDLE(task, i));
		}
		printf("\n");
    }
    printf("\n");
}

void print_data_not_used_yet()
{
    int i = 0;
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    for (i = 0; i < Ngpu; i++)
    {
		printf("On GPU %d, there are %d data not used yet are:", i + 1, gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data));
		for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(my_planned_task_control->pointer->gpu_data); e != gpu_data_not_used_list_end(my_planned_task_control->pointer->gpu_data); e = gpu_data_not_used_list_next(e))
		{
			printf(" %p", e->D);
		}
		printf("\n");
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    my_planned_task_control->pointer = my_planned_task_control->first;
    printf("\n");
}

void print_planned_task_one_gpu(struct gpu_planned_task *g, int current_gpu)
{
    struct starpu_task *task = NULL;
    
    if (starpu_task_list_empty(&g->planned_task))
    {
		printf("GPU %d's planned task list is empty.\n\n", current_gpu);
	}
	else
	{
		printf("Planned task for GPU %d:\n", current_gpu);
		for (task = starpu_task_list_begin(&g->planned_task); task != starpu_task_list_end(&g->planned_task); task = starpu_task_list_next(task))
		{
			printf("%p: %p %p\n", task, STARPU_TASK_GET_HANDLE(task, 0), STARPU_TASK_GET_HANDLE(task, 1));
		}
		printf("\n");
	}
}

void print_pulled_task_one_gpu(struct gpu_pulled_task *g, int current_gpu)
{
    struct pulled_task *p = pulled_task_new();
    
    printf("Pulled task for GPU %d:\n", current_gpu); fflush(stdout);
    for (p = pulled_task_list_begin(g->ptl); p != pulled_task_list_end(g->ptl); p = pulled_task_list_next(p))
    {
		printf("%p\n", p->pointer_to_pulled_task); fflush(stdout);
    }
}

void print_data_not_used_yet_one_gpu(struct gpu_planned_task *g)
{
    printf("Data not used yet are:\n");
    for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
    {
		printf(" %p", e->D);
    }
    printf("\n");
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

void print_data_on_node(starpu_data_handle_t *data_tab, int nb_data_on_node)
{
    int i = 0;
    printf("Data on node are:");
    for (i = 0; i < nb_data_on_node; i++)
    {
		printf(" %p", data_tab[i]);
    }
    printf("\n");
}

void print_nb_task_in_list_one_data_one_gpu(starpu_data_handle_t d, int current_gpu)
{
	struct handle_user_data * hud = d->user_data;
	printf("Number of task used by %p in the tasks list:", d);
	printf("pulled_task = %d tasks | planned_tasks = %d tasks.\n", hud->nb_task_in_pulled_task[current_gpu - 1], hud->nb_task_in_planned_task[current_gpu - 1]);
}

bool need_to_reinit = true;

/* Pushing the tasks. Each time a new task enter here, we initialize it. */		
static int dynamic_data_aware_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	
    /* If this boolean is true, pull_task will know that new tasks have arrived and
     * thus it will be able to randomize both the task list and the data list not used yet in the GPUs. 
     */
     	
     if (need_to_reinit == true && iteration_DARTS != 1)
	 {
		int i = 0;
		//~ for (i = 0; i < Ngpu; i++) { STARPU_PTHREAD_MUTEX_LOCK(&local_mutex[i]); }

		//~ printf("REINIT STRUCT in push task.\n"); fflush(stdout);
			
		free(my_planned_task_control);
			
		gpu_planned_task_initialisation();
		for (i = 0; i < Ngpu - 1; i++)
		{
			gpu_planned_task_insertion();
		}
		my_planned_task_control->first = my_planned_task_control->pointer;

		//~ gpu_pulled_task_initialisation();
		//~ for (i = 0; i < Ngpu - 1; i++)
		//~ {
			//~ gpu_pulled_task_insertion();
		//~ }
		//~ my_pulled_task_control->first = my_pulled_task_control->pointer;

		need_to_reinit = false;
			
		//~ for (i = 0; i < Ngpu; i++) { STARPU_PTHREAD_MUTEX_UNLOCK(&local_mutex[i]); }
	}
     
    new_tasks_initialized = true; 
    struct dynamic_data_aware_sched_data *data = component->data;
    
    #ifdef PRINT
    gettimeofday(&time_start_initialisation, NULL);
    #endif
    
    initialize_task_data_gpu_single_task(task);
    
    #ifdef PRINT
    gettimeofday(&time_end_initialisation, NULL);
	time_total_initialisation += (time_end_initialisation.tv_sec - time_start_initialisation.tv_sec)*1000000LL + time_end_initialisation.tv_usec - time_start_initialisation.tv_usec;
	#endif
    
    /* Pushing the task in sched_list. It's this list that will be randomized
     * and put in main_task_list in pull_task.
     */
    starpu_task_list_push_front(&data->sched_list, task);
    starpu_push_task_end(task);
    component->can_pull(component);
    
    STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    return 0;
}

/* Initialize for:
 * tasks -> pointer to the data it uses, pointer to the pointer of task list in the data, 
 * pointer to the cell in the main task list (main_task_list).
 * data -> pointer to the tasks using this data.
 * GPUs -> datas not used yet by this GPU.
 * TODO : Y a pas moyen de fusionner les deux boucles pour parcourir juste une fois la liste des données d'une tâche ? Non vu que la première boucle coucle sur NGPU, faudrait faire un test if on est sur le gpu 1 seulement alors je fais le truc sinon, non.
 * Création d'un tableau 3D des données, marche que en 3D. TODO : ajouter que APP=1 c'est que pour ca et APP=2 sera pour les variantes 3D mais sans cette init ou autrement mais le différencier quoi pour pas el faire sur cholesky ou M2D.
 */
void initialize_task_data_gpu_single_task(struct starpu_task *task)
{
    int i = 0;
    int j = 0;
    
    /* Adding the data not used yet in all the GPU(s). */
    my_planned_task_control->pointer = my_planned_task_control->first;
    for (i = 0; i < Ngpu; i++)
    {
		for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
		{
			struct gpu_data_not_used *e = gpu_data_not_used_new();
			e->D = STARPU_TASK_GET_HANDLE(task, j);
			
			/* If the void * of struct paquet is empty I initialize it. */ 
			if (my_planned_task_control->pointer->gpu_data == NULL)
			{
				struct gpu_data_not_used_list *gd = gpu_data_not_used_list_new();
				gpu_data_not_used_list_push_front(gd, e);
				my_planned_task_control->pointer->gpu_data = gd; 
			}
			else
			{
				/* La je ne dois pas ne rien faire a l'iteration_DARTS 2 */
				/* Il faudrait une liste exeterne des data pour les reset ? */
				if (STARPU_TASK_GET_HANDLE(task, j)->sched_data == NULL)
				{
					gpu_data_not_used_list_push_front(my_planned_task_control->pointer->gpu_data, e);
				}
				else
				{
					if (STARPU_TASK_GET_HANDLE(task, j)->user_data != NULL)
					{
						struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, j)->user_data;
						if (hud->last_iteration_DARTS != iteration_DARTS)
						{
							gpu_data_not_used_list_push_front(my_planned_task_control->pointer->gpu_data, e);
						}
					}
					else
					{
						gpu_data_not_used_list_push_front(my_planned_task_control->pointer->gpu_data, e);
					}
				}
			}
		}
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    
    /* Adding the pointer in the task. */
    struct pointer_in_task *pt = malloc(sizeof(*pt));
    pt->pointer_to_cell = task;
    pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
    pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
    /* pt->state = 0; for state of the task in pulled task or planned task */
	
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
		/* Added for reset but also compteur in both task list */
		if (STARPU_TASK_GET_HANDLE(task, i)->user_data == NULL)
		{
			struct handle_user_data * hud = malloc(sizeof(*hud));
			hud->last_iteration_DARTS = iteration_DARTS;
			
			/* Need to init them with the number of GPU */
			hud->nb_task_in_pulled_task = malloc(Ngpu*sizeof(int));
			hud->nb_task_in_planned_task = malloc(Ngpu*sizeof(int));
			hud->last_check_to_choose_from = malloc(Ngpu*sizeof(int));
			for (j = 0; j < Ngpu; j++)
			{
				hud->nb_task_in_pulled_task[j] = 0;
				hud->nb_task_in_planned_task[j] = 0;
				hud->last_check_to_choose_from[j] = 0;
			}
			
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		else
		{
			struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->last_iteration_DARTS = iteration_DARTS;
			for (j = 0; j < Ngpu; j++)
			{
				hud->nb_task_in_pulled_task[j] = 0;
				hud->nb_task_in_planned_task[j] = 0;
				hud->last_check_to_choose_from[j] = 0;
			}
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
			
		/* Adding the pointer in the task. */
		pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
		pt->tud[i] = e;
    }
    
    /* Matrice 3D des coords pour ordre Z. */
    /* A modif peut etre la condition. */
    /* Pourquoi je m'embete ? l'ordre naturel peut permettre de faire ca sans check les coord nan ? SUffit de connaitre N ? */
	//~ if (app == 1)
	//~ {
		//~ int temp_tab_coordinates[2];
		//~ starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 2), 2, temp_tab_coordinates);
		//~ printf("Tâche %p : A (%p) x = %d | ", task, STARPU_TASK_GET_HANDLE(task, 0), temp_tab_coordinates[0]);
		//~ printf("B (%p) y = %d | ", STARPU_TASK_GET_HANDLE(task, 1), temp_tab_coordinates[1]);
		//~ starpu_data_get_coordinates_array(STARPU_TASK_GET_HANDLE(task, 0), 2, temp_tab_coordinates);
		//~ printf("C (%p) z = %d.\n", STARPU_TASK_GET_HANDLE(task, 2), temp_tab_coordinates[0]);
	//~ }
    
    task->sched_data = pt;
}

void randomize_task_list(struct dynamic_data_aware_sched_data *d)
{
    /* NEW */
    int random = 0;
    int i = 0;
    struct starpu_task *task_tab[NT_dynamic_outer];
    
    for (i = 0; i < NT_dynamic_outer; i++)
    {
		task_tab[i] = starpu_task_list_pop_front(&d->sched_list);
    }
    for (i = 0; i < NT_dynamic_outer; i++)
    {
		random = rand()%(NT_dynamic_outer - i);
		starpu_task_list_push_back(&d->main_task_list, task_tab[random]);
		
		/* Je remplace la case par la dernière tâche du tableau */
		task_tab[random] = task_tab[NT_dynamic_outer - i - 1];
	}
}

/* Chaque GPU a un pointeur vers sa première tâche à pop.
 * Ensuite dans le scheduling quand on pop pour la première fois c'est celle la.
 * En plus ca tombe bien le premier pop est géré direct en dehors de random. */
void natural_order_task_list(struct dynamic_data_aware_sched_data *d)
{
	my_planned_task_control->pointer = my_planned_task_control->first;
    int i = 0;
    int j = 0;
    struct starpu_task *task = NULL;
    
    for (i = 0; i < NT_dynamic_outer; i++)
    {
		if (i == (NT_dynamic_outer/Ngpu)*j && j < Ngpu)
		{
			task = starpu_task_list_pop_front(&d->sched_list);
			my_planned_task_control->pointer->first_task_to_pop = task;
			starpu_task_list_push_back(&d->main_task_list, task);
			my_planned_task_control->pointer = my_planned_task_control->pointer->next;
			j++;
		}
		else
		{
			starpu_task_list_push_back(&d->main_task_list, starpu_task_list_pop_front(&d->sched_list));
		}
    }
}

/* Randomize the list of data not used yet for all the GPU. */
void randomize_data_not_used_yet()
{
    /* NEW */
    int i = 0;
    int j = 0;
    int random = 0;
    int number_of_data = 0;
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    /* I need this for the %random. */
    number_of_data = gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data);
    
    struct gpu_data_not_used *data_tab[number_of_data];
    
    for (i = 0; i < Ngpu; i++)
    {
		for (j = 0; j < number_of_data; j++)
		{
			data_tab[j] = gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data);
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
		my_planned_task_control->pointer->gpu_data = randomized_list;
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
}

void natural_order_data_not_used_yet()
{
    int i = 0;
    int j = 0;
    int number_of_data = 0;
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    /* I need this for the %random. */
    number_of_data = gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data);
    
    struct gpu_data_not_used *data_tab[number_of_data];
    
    /* On ne fais rien pour le premier GPU. Le deuxième GPU commence à Ndata/Ngpu,
     * puis le deuxième à (Ndata/Ngpu)*1 et esnuite (Ndata/Ngpu)*2 ainsi de suite ...
     */
    for (i = 1; i < Ngpu; i++)
    {
		for (j = 0; j < (number_of_data/Ngpu)*i; j++)
		{
			data_tab[j] = gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data);
		}
		struct gpu_data_not_used_list *natural_order_list = gpu_data_not_used_list_new();
		for (j = 0; j < number_of_data - ((number_of_data/Ngpu)*i); j++)
		{
			gpu_data_not_used_list_push_back(natural_order_list, gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data));
		}
		for (j = 0; j < (number_of_data/Ngpu)*i; j++)
		{
			gpu_data_not_used_list_push_back(natural_order_list, data_tab[j]);
		}

		/* Then replace the list with it. */
		my_planned_task_control->pointer->gpu_data = natural_order_list;
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
}

//~ void order_z_data_not_used_yet()
//~ {
	//~ /* Je créé une liste d'entier random pour i (1:N), j (1:N) et k (1:4). */
	//~ int i = 0;
    //~ int j = 0;
    //~ int random = 0;
        
    //~ int N = sqrt(NT_dynamic_outer)/2;
    
    //~ int ordre_i[N];
    //~ int ordre_j[N];
    //~ int ordre_k[4];
    //~ int temp_ordre_i[N];
    //~ int temp_ordre_j[N];
    //~ int temp_ordre_k[4];
    
    //~ for (i = 0; i < N; i++)
    //~ {
		//~ temp_ordre_i[i] = i;
		//~ temp_ordre_j[i] = i;
	//~ }
    //~ for (i = 0; i < 4; i++)
    //~ {
		//~ temp_ordre_k[i] = i;
	//~ }
    
    //~ /* Je le fais différement pour chaque GPU ou c'est comun aux 3 GPUs ? */
		//~ /* Je créé une liste d'entier random pour i (1:N), j (1:N) et k (1:4). */	
		//~ for (j = 0; j < N; j++)
		//~ {
			//~ random = rand()%(N - j);
			//~ ordre_i[j] = temp_ordre_i[random];
			
			//~ /* Je remplace la case par la dernière tâche du tableau */
			//~ temp_ordre_i[random] = temp_ordre_i[N - j - 1];
			
			//~ random = rand()%(N - j);
			//~ ordre_j[j] = temp_ordre_j[random];
			
			//~ /* Je remplace la case par la dernière tâche du tableau */
			//~ temp_ordre_j[random] = temp_ordre_j[N - j - 1];
		//~ }
		//~ for (j = 0; j < 4; j++)
		//~ {
			//~ random = rand()%(4 - j);
			//~ ordre_k[j] = temp_ordre_k[random];
			
			//~ /* Je remplace la case par la dernière tâche du tableau */
			//~ temp_ordre_k[random] = temp_ordre_k[4 - j - 1];
		//~ }
	
	//~ printf("Ordre i :");
	//~ for (i = 0; i < N; i++)
	//~ {
		//~ printf(" %d", ordre_i[i]);
	//~ }
	//~ printf("\n");
	//~ printf("Ordre j :");
	//~ for (i = 0; i < N; i++)
	//~ {
		//~ printf(" %d", ordre_j[i]);
	//~ }
	//~ printf("\n");
	//~ printf("Ordre k :");
	//~ for (i = 0; i < 4; i++)
	//~ {
		//~ printf(" %d", ordre_k[i]);
	//~ }
	//~ printf("\n");
//~ }

//~ /* Randomize the list of data not used yet for a single GPU. */
//~ void randomize_data_not_used_yet_single_GPU(struct gpu_planned_task *g)
//~ {
    //~ int j = 0;
    //~ int k = 0;
    //~ int random = 0;
    //~ int number_of_data = 0;
    
    //~ number_of_data = gpu_data_not_used_list_size(g->gpu_data);
    
    //~ struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
    //~ for (j = 0; j < number_of_data; j++)
    //~ {
	//~ /* After each time I remove a data I can choose between a smaller number of value for random. */
	//~ random = rand()%(number_of_data - j);
	//~ for (k = 0; k < random; k++)
	//~ {
	    //~ gpu_data_not_used_list_push_back(g->gpu_data, gpu_data_not_used_list_pop_front(g->gpu_data));
	//~ }
	//~ /* I use an external list. */
	//~ gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(g->gpu_data));
    //~ }
    //~ /* Then replace the list with it. */
    //~ g->gpu_data = randomized_list;
//~ }

/* Get a task to put out of pull_task. In multi GPU it allows me to return a task from the right element in the 
 * linked list without having an other GPU comme and ask a task in pull_task. At least I hope it does so.
 */
struct starpu_task *get_task_to_return_pull_task_dynamic_data_aware(int current_gpu, struct starpu_task_list *l)
{
	//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	int i = 0;
    /* Getting on the right GPU's package.
     * TODO: Can I do this faster with pointer directly to the cell ? */
    my_planned_task_control->pointer = my_planned_task_control->first;
    for (i = 1; i < current_gpu; i++) /* Parce que le premier GPU vaut 1 et pas 0. */
    {
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    
    /* If there are still tasks either in the packages, the main task list or the refused task,
     * I enter here to return a task or start dynamic_data_aware_scheduling. Else I return NULL.
     */
    //~ if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task) || !starpu_task_list_empty(l) || !starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list))
    //~ {
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("GPU %d is asking for a task.\n", current_gpu); }
		struct starpu_task *task = NULL;

		/* If one or more task have been refused */
		if (!starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list)) 
		{
			/* Ici je ne met pas à jour pulled_task car je l'ai déjà fais pour la tâche avant qu'elle ne soit refusé. */
			task = starpu_task_list_pop_back(&my_planned_task_control->pointer->refused_fifo_list); 
			
			#ifdef PRINT
			print_data_to_load_prefetch(task, starpu_worker_get_id());
			#endif
			
			return task;
		}
		
		STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		/* If the package is not empty I can return the head of the task list. */
		if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task))
		{
			number_task_out_DARTS++;
			task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);

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
			#ifdef PRINT
			print_data_to_load_prefetch(task, starpu_worker_get_id());
			#endif
			
			//~ printf("Task %d: %p is getting out of pull_task from planned task not empty on GPU %d\n", number_task_out_DARTS, task, current_gpu); fflush(stdout);
			
			STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			return task;
		}
		
		/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
		if (!starpu_task_list_empty(l))
		{
			number_task_out_DARTS++;
			
			//~ /* Cas matrice 2D-3D séparé */
			//~ if (app == 0)
			//~ {
				//~ dynamic_data_aware_scheduling_one_data_popped(l, current_gpu, my_planned_task_control->pointer);
			//~ }
			//~ else if (app == 1)
			//~ {
				//~ dynamic_data_aware_scheduling_3D_matrix(l, current_gpu, my_planned_task_control->pointer);
			//~ }
			
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			/* La j'appelle 3D dans les deux cas car j'ai voulu regrouper. */
			dynamic_data_aware_scheduling_3D_matrix(l, current_gpu, my_planned_task_control->pointer);
			
			//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
			if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task))
			{
				task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);
				
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
				STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
				return NULL;
			}
			
			/* For visualisation in python. */
			#ifdef PRINT
			print_data_to_load_prefetch(task, starpu_worker_get_id());
			printf("Task %d: %p is getting out of pull_task after scheduling on GPU %d\n", number_task_out_DARTS, task, current_gpu); fflush(stdout);
			#endif
			
			STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			return task;
		}
		else
		{
    //~ }
    STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    return NULL; }
}

void reset_all_struct()
{
	NT_dynamic_outer = -1;
	/* For visualisation in python. */
	//~ index_current_popped_task = malloc(sizeof(int)*Ngpu);
	//~ index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	//~ index_current_popped_task_all_gpu = 0;
	//~ index_current_popped_task_all_gpu_prefetch = 0;
	
	number_task_out_DARTS = -1;
}

/* Pull tasks. When it receives new task it will randomize the task list and the GPU data list.
 * If it has no task it return NULL. Else if a task was refused it return it. Else it return the
 * head of the GPU task list. Else it calls dyanmic_outer_scheuling to fill this package. */
static struct starpu_task *dynamic_data_aware_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
    struct dynamic_data_aware_sched_data *data = component->data;

    /* Need only to be done once if all GPU have the same memory. */
    if (gpu_memory_initialized == false)
    {
		GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));
		gpu_memory_initialized = true;
    }
    
    /* New tasks from push_task. We need to randomize. 
     * TODO: check that with other applications where task are not
     * all available at once, this works.
     */
    if (new_tasks_initialized == true)
    {	    
		//~ int i = 0;
		//~ for (i = 0; i < Ngpu; i++) { STARPU_PTHREAD_MUTEX_LOCK(&local_mutex[i]); }

		new_tasks_initialized = false;
		
		#ifdef PRINT
		printf("Printing GPU's data list and main task list before:\n\n");
		print_data_not_used_yet();
		print_task_list(&data->sched_list, "");
		#endif
		
		NT_dynamic_outer = starpu_task_list_size(&data->sched_list);
		NT = NT_dynamic_outer;
		
		#ifdef PRINT
		gettimeofday(&time_start_randomize, NULL);
		#endif
		
		if (natural_order == 0)
		{
			randomize_task_list(data);
			
			if (choose_best_data_from != 1)
			{
				randomize_data_not_used_yet();
			}	
		}
		else if (natural_order == 1) /* Randomise pas les liste de données. */
		{
			randomize_task_list(data);
			
			if (choose_best_data_from != 1)
			{
				natural_order_data_not_used_yet();
			}
		}
		else /* Randomise rien du tout, NATURAL_ORDER == 2. */
		{
			natural_order_task_list(data);
			
			if (choose_best_data_from != 1)
			{
				natural_order_data_not_used_yet();
			}
		}

		//~ if (data_order == 0)
		//~ {
		//~ }
		//~ else /* ordre Z des données */
		//~ {
			//~ order_z_data_not_used_yet();
		//~ }
		
		#ifdef PRINT
		gettimeofday(&time_end_randomize, NULL);
		time_total_randomize += (time_end_randomize.tv_sec - time_start_randomize.tv_sec)*1000000LL + time_end_randomize.tv_usec - time_start_randomize.tv_usec;		
		printf("Il y a %d tâches.\n", NT_dynamic_outer);
		printf("Printing GPU's data list and main task list after randomization (NATURAL_ORDER = %d):\n\n", natural_order);
		print_data_not_used_yet();
		print_task_list(&data->main_task_list, ""); fflush(stdout);
		#endif
		
		//~ for (i = 0; i < Ngpu; i++) { STARPU_PTHREAD_MUTEX_UNLOCK(&local_mutex[i]); }
    }
    
    /* Nouveau */
	//~ int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id()); /* Attention le premier GPU vaut 1 et non 0. */
    STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
        
    /* Nouveau */
    //~ STARPU_PTHREAD_MUTEX_LOCK(&local_mutex[current_gpu - 1]);
    struct starpu_task *task = get_task_to_return_pull_task_dynamic_data_aware(starpu_worker_get_memory_node(starpu_worker_get_id()), &data->main_task_list);
    //~ STARPU_PTHREAD_MUTEX_UNLOCK(&local_mutex[current_gpu - 1]);
    
    /* Ancienne location du mutex global. */
    //~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    return task;
}

void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct gpu_planned_task *g)
{
    struct gpu_data_not_used *ptr = gpu_data_not_used_new();
    struct gpu_data_not_used *new_element = gpu_data_not_used_new();
    new_element->D = h;
    int random = rand()%gpu_data_not_used_list_size(g->gpu_data);
    int i = 0;
    
    ptr = gpu_data_not_used_list_begin(g->gpu_data);
    for (i = 0; i < random; i++)
    {
		ptr = gpu_data_not_used_list_next(ptr);
    }
    gpu_data_not_used_list_insert_before(g->gpu_data, new_element, ptr);
}

/* Fill a package's task list following dynamic_data_aware algorithm. It pop only one data, the one that achieve the most tasks. */
//~ void dynamic_data_aware_scheduling_one_data_popped(struct starpu_task_list main_task_list, int current_gpu, struct gpu_planned_task g)
//~ {
	//~ // TODO : A suppr ou a ifdef pour meilleur perf 
	//~ #ifdef PRINT
	//~ gettimeofday(&time_start_schedule, NULL);
	//~ #endif
	
    //~ int i = 0;
    //~ int j = 0;
    //~ struct task_using_data t = NULL;
    //~ struct gpu_data_not_used e = NULL;
    //~ int number_of_task_max = 0;
    //~ int task_available_max = 0;
    //~ int temp_number_of_task_max = 0;
    //~ starpu_data_handle_t handle_popped = NULL;
    //~ struct task_using_data_list tudl = task_using_data_list_new();
    
    //~ // Si c'est la première tâche, on regarde pas le reste on fais random. 
    //~ if (g->first_task == true)
    //~ {
		//~ #ifdef PRINT
		//~ printf("Go to random car c'est la première tâche du GPU %d.\n", current_gpu);
		//~ #endif
		
		//~ g->first_task = false;
		//~ goto random;
	//~ }
    
    //~ // Ce cas arrive avec le cas ou je gère pas les evictions. Car quand je ne gère pas les évictions je ne remet pas les données évincées dans la liste des données
      //~ à faire. 
    //~ if (gpu_data_not_used_list_empty(g->gpu_data))
    //~ {
		//~ goto random;
    //~ }
    
    //~ // To know if all the data needed for a task are loaded in memory. 
    //~ bool data_available = true;
    
    //~ // Pour faire varier le TH des data que l'on regarde 
    /*
    int choose_best_data_threshold = INT_MAX;
    if (starpu_get_env_number_default("LIFT_THRESHOLD_MODE", 0) != 0)
    {
		if (starpu_get_env_number_default("LIFT_THRESHOLD_MODE", 0) == 1)
		{
			// Version avec un threshold fixe de data que l'on regarde. 
			// getting the number of data on which we will loop. for 0 we take infinite. 
			choose_best_data_threshold = starpu_get_env_number_default("CHOOSE_BEST_DATA_THRESHOLD", 0);
		}
		else if (starpu_get_env_number_default("LIFT_THRESHOLD_MODE", 0) == 2)
		{
			// Version avec un threshold qui varie en fonction du nombres de tâches envoyées à pulled task.
			  Quand NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD == number_of_task_out je met le th à INT_MAX. 
			if (number_task_out_DARTS >= starpu_get_env_number_default("NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD", 0)/Ngpu)
			{
				choose_best_data_threshold = INT_MAX;
			}
			else
			{
				choose_best_data_threshold = starpu_get_env_number_default("CHOOSE_BEST_DATA_THRESHOLD", 0);
			}
		}
		else if (starpu_get_env_number_default("LIFT_THRESHOLD_MODE", 0) == 3)
		{
			// Version avec un threshold qui varie en fonction du nombres de tâches envoyées à pulled task mais progressivement.
			  Quand NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD == number_of_task_out je met le th à INT_MAX. 
			if (number_task_out_DARTS >= starpu_get_env_number_default("NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD", 0)/Ngpu)
			{
				choose_best_data_threshold = INT_MAX;
			}
			else
			{
				choose_best_data_threshold = starpu_get_env_number_default("CHOOSE_BEST_DATA_THRESHOLD", 0) + number_task_out_DARTS/2;
			}
		}
		else if (starpu_get_env_number_default("LIFT_THRESHOLD_MODE", 0) == 4)
		{
			// Version avec un threshold qui varie en fonction du pourcentage de tâches envoyées à pulled task.
			  Quand PERCENTAGE_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD == number_of_task_out je met le th à INT_MAX.
			  Fait augmenter en douceur jusqu'au % max. 
			if ((number_task_out_DARTS100)/NT_dynamic_outer >= starpu_get_env_number_default("PERCENTAGE_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD", 0)/Ngpu)
			{
				choose_best_data_threshold = INT_MAX;
			}
			else
			{
				choose_best_data_threshold = starpu_get_env_number_default("CHOOSE_BEST_DATA_THRESHOLD", 0) + ((number_task_out_DARTS100)/NT_dynamic_outer)2;
			}
		}
	} */
	// The costly loop 
	
	//~ // pour diminuer a la fin de l'execution 
	/*
	int choose_best_data_threshold = INT_MAX;
	if (number_task_out_DARTS > 40000)
	{
		choose_best_data_threshold = 100;
	}
	*/
	
	//~ // Pour diminuer au début de l'execution 
	//~ int choose_best_data_threshold = INT_MAX;
	//~ if (threshold == 1)
	//~ {
		//~ if (NT_dynamic_outer > 14400)
		//~ {
			//~ choose_best_data_threshold = 110;
		//~ }
	//~ }

	//~ #ifdef PRINT
    //~ gettimeofday(&time_start_choose_best_data, NULL);
    //~ #endif
    	
	//~ // Version avec threshold 
	//~ i = 0;
    //~ for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data) && i != choose_best_data_threshold; e = gpu_data_not_used_list_next(e), i++)
    //~ {
		//~ temp_number_of_task_max = 0;
				
		//~ for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
		//~ {
			//~ // I put it at false if at least one data is missing. 
			//~ data_available = true; 
			//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			//~ {
				//~ // I test if the data is on memory  
				//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
				//~ {
					//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
					//~ {
						//~ data_available = false;
						//~ break;
					//~ }
				//~ }
			//~ }
			//~ if (data_available == true)
			//~ {
				//~ temp_number_of_task_max++;
			//~ }
		//~ }
	
		//~ if (temp_number_of_task_max > number_of_task_max)
		//~ {
			//~ number_of_task_max = temp_number_of_task_max;
			//~ task_available_max = task_using_data_list_size(e->D->sched_data);
			//~ handle_popped = e->D;
		//~ }
		//~ // Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. 
		//~ else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
		//~ {
			//~ tudl = e->D->sched_data;
			//~ // TODO : la en 3D on voudra check les data qui peuvent permettre de faire des tâches avec 1 data de load. Puius pour rendre ca général avec 2 data de plus, 3 de plus etc... Du coup rendre ca géénral et déjà tester que en 2d ca donne les mêmes résultats exactement, car normalement ca devrait. 
			//~ if (task_using_data_list_size(tudl) > task_available_max)
			//~ {
				//~ task_available_max = task_using_data_list_size(tudl);
				//~ handle_popped = e->D;
			//~ }
		//~ }
    //~ }
    //~ // Fin de version avec threshold 
    
    //~ // Version pour tester en continuant à perdre du temps après le threshold.
    //~ gettimeofday(&time_start_choose_best_data, NULL);
    //~ for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e), i++)
    //~ {
		//~ temp_number_of_task_max = 0;
				
		//~ for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
		//~ {
			//~ data_available = true; 
			//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			//~ {
				//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
				//~ {
					//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
					//~ {
						//~ data_available = false;
						//~ break;
					//~ }
				//~ }
			//~ }
			//~ if (data_available == true)
			//~ {
				//~ temp_number_of_task_max++;
			//~ }
		//~ }
		
		//~ if (temp_number_of_task_max > number_of_task_max)
		//~ {
			//~ if (i < choose_best_data_threshold)
			//~ {
				//~ number_of_task_max = temp_number_of_task_max;
				//~ task_available_max = task_using_data_list_size(e->D->sched_data);
				//~ handle_popped = e->D;
			//~ }
		//~ }
		//~ else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
		//~ {
			//~ tudl = e->D->sched_data;
			//~ if (task_using_data_list_size(tudl) > task_available_max)
			//~ {
				//~ if (i < choose_best_data_threshold)
				//~ {
					//~ task_available_max = task_using_data_list_size(tudl);
					//~ handle_popped = e->D;
				//~ }
			//~ }
		//~ }
    //~ } Fin de version pour tester en perdant du temps après le threshold. 
    /*
    if(starpu_get_env_number_default("CHOOSE_BEST_DATA_TYPE", 0) == 0)
    {
		//~ // The costly loop. Version originale. 
		i = 0;
		gettimeofday(&time_start_choose_best_data, NULL);
		for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
		{
			temp_number_of_task_max = 0;
					
			for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
			{
				// I put it at false if at least one data is missing. 
				data_available = true; 
				for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
				{
					// I test if the data is on memory  
					if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
					{
						if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
						{
							data_available = false;
							break;
						}
					}
				}
				if (data_available == true)
				{
					temp_number_of_task_max++;
				}
			}
		
			if (temp_number_of_task_max > number_of_task_max)
			{
				number_of_task_max = temp_number_of_task_max;
				task_available_max = task_using_data_list_size(e->D->sched_data);
				handle_popped = e->D;
			}
			// Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. 
			else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
			{
				tudl = e->D->sched_data;
				// TODO : la en 3D on voudra check les data qui peuvent permettre de faire des tâches avec 1 data de load. Puius pour rendre ca général avec 2 data de plus, 3 de plus etc... Du coup rendre ca géénral et déjà tester que en 2d ca donne les mêmes résultats exactement, car normalement ca devrait. 
				if (task_using_data_list_size(tudl) > task_available_max)
				{
					task_available_max = task_using_data_list_size(tudl);
					handle_popped = e->D;
				}
			}
		}
		gettimeofday(&time_end_choose_best_data, NULL);
		time_total_choose_best_data += (time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec;
		//~ // Seulement pour les traces pour les faires ressembler à celle sur Grid5000 
		//~ // usleep(((time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec));
		//~ // Fin de la version originale 
    }
    else
    {
		// The costly loop. Version où je récup les 10 meilleures données et ensuite tant que je les ai pas toutes envoyé je test sur ces 10 données la seulement. 
		  Utilise une struct globale et un int dans le .h a suppr si on utilise pas cette méthode 
		gettimeofday(&time_start_choose_best_data, NULL);
		if (data_to_pop_next_list_empty(g->my_data_to_pop_next))
		{
			int k = 0;
			int number_task_data_to_pop_next = malloc(N_data_to_pop_nextsizeof(int));
			int global_number_task_data_to_pop_next = malloc(N_data_to_pop_nextsizeof(int));
			for (i = 0; i < N_data_to_pop_next; i++)
			{
				number_task_data_to_pop_next[i] = -1;
				global_number_task_data_to_pop_next[i] = -1;
			}
			
			for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
			{
				temp_number_of_task_max = 0;
						
				for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
				{
					// I put it at false if at least one data is missing. 
					data_available = true; 
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
					{
						// I test if the data is on memory  
						if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
						{
							if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
							{
								data_available = false;
								break;
							}
						}
					}
					if (data_available == true)
					{
						temp_number_of_task_max++;
					}
				}
				
				for (i = 0; i < N_data_to_pop_next; i++)
				{
					struct data_to_pop_next ptr = data_to_pop_next_new();
					struct data_to_pop_next new = data_to_pop_next_new();
					if (temp_number_of_task_max > number_task_data_to_pop_next[i])
					{
						for (k = i; k < N_data_to_pop_next - 1; k++)
						{
							number_task_data_to_pop_next[k + 1] = number_task_data_to_pop_next[k];
							global_number_task_data_to_pop_next[k + 1] = global_number_task_data_to_pop_next[k];
						}
						number_of_task_max = temp_number_of_task_max;
						number_task_data_to_pop_next[i] = temp_number_of_task_max;
						global_number_task_data_to_pop_next[i] = task_using_data_list_size(e->D->sched_data);
						
						ptr = data_to_pop_next_list_begin(g->my_data_to_pop_next);
						new->pointer_to_data_to_pop_next = e->D;
						for (k = 0; k < i; k++)
						{
							ptr = data_to_pop_next_list_next(ptr);
						}
						if (data_to_pop_next_list_empty(g->my_data_to_pop_next))
						{
							data_to_pop_next_list_push_front(g->my_data_to_pop_next, new);
						}
						else if (ptr == NULL)
						{
							data_to_pop_next_list_push_back(g->my_data_to_pop_next, new);
						}
						else
						{
							data_to_pop_next_list_insert_before(g->my_data_to_pop_next, new, ptr);
						}
						break;
					}
					else if (temp_number_of_task_max == number_task_data_to_pop_next[i] && number_task_data_to_pop_next[i] != 0)
					{
						if (task_using_data_list_size(tudl) > global_number_task_data_to_pop_next[i])
						{
							for (k = i + 1; k < N_data_to_pop_next; k++)
							{
								number_task_data_to_pop_next[k] = number_task_data_to_pop_next[k - 1];
								global_number_task_data_to_pop_next[k] = global_number_task_data_to_pop_next[k - 1];
							}
							number_of_task_max = temp_number_of_task_max;
							number_task_data_to_pop_next[i] = temp_number_of_task_max;
							global_number_task_data_to_pop_next[i] = task_using_data_list_size(e->D->sched_data);
							
							ptr = data_to_pop_next_list_begin(g->my_data_to_pop_next);
							new->pointer_to_data_to_pop_next = e->D;
							for (k = 0; k < i; k++)
							{
								ptr = data_to_pop_next_list_next(ptr);
							}
							if (data_to_pop_next_list_empty(g->my_data_to_pop_next))
							{
								data_to_pop_next_list_push_front(g->my_data_to_pop_next, new);
							}
							else if (ptr == NULL)
							{
								data_to_pop_next_list_push_back(g->my_data_to_pop_next, new);
							}
							else
							{
								data_to_pop_next_list_insert_before(g->my_data_to_pop_next, new, ptr);
							}
							break;
						}
					}
				}
			}
			struct data_to_pop_next handle_to_return = data_to_pop_next_new();
			handle_to_return = data_to_pop_next_list_pop_front(g->my_data_to_pop_next);
			handle_popped = handle_to_return->pointer_to_data_to_pop_next;
		}
		else
		{
			if(starpu_get_env_number_default("CHOOSE_BEST_DATA_TYPE", 0) == 1)
			{
				// version ou je renvoie juste sans choisir 
				struct data_to_pop_next handle_to_return = data_to_pop_next_new();
				handle_to_return = data_to_pop_next_list_pop_front(g->my_data_to_pop_next);
				handle_popped = handle_to_return->pointer_to_data_to_pop_next;
			}
			else
			{
				// choose best one among them 
				number_of_task_max = -1;
				task_available_max = 0;
				temp_number_of_task_max = 0;
				struct data_to_pop_next e2 = data_to_pop_next_new();
				for (e2 = data_to_pop_next_list_begin(g->my_data_to_pop_next); e2 != data_to_pop_next_list_end(g->my_data_to_pop_next); e2 = data_to_pop_next_list_next(e2))
				{
					temp_number_of_task_max = 0;		
					for (t = task_using_data_list_begin(e2->pointer_to_data_to_pop_next->sched_data); t != task_using_data_list_end(e2->pointer_to_data_to_pop_next->sched_data); t = task_using_data_list_next(t))
					{
						// I put it at false if at least one data is missing. 
						data_available = true; 
						for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
						{
							// I test if the data is on memory  
							if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e2->pointer_to_data_to_pop_next)
							{
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
								{
									data_available = false;
									break;
								}
							}
						}
						if (data_available == true)
						{
							temp_number_of_task_max++;
						}
					}
				
					if (temp_number_of_task_max > number_of_task_max)
					{
						number_of_task_max = temp_number_of_task_max;
						task_available_max = task_using_data_list_size(e2->pointer_to_data_to_pop_next->sched_data);
						handle_popped = e2->pointer_to_data_to_pop_next;
					}
					// Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. 
					else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
					{
						tudl = e2->pointer_to_data_to_pop_next->sched_data;
						// TODO : la en 3D on voudra check les data qui peuvent permettre de faire des tâches avec 1 data de load. Puius pour rendre ca général avec 2 data de plus, 3 de plus etc... Du coup rendre ca géénral et déjà tester que en 2d ca donne les mêmes résultats exactement, car normalement ca devrait. 
						if (task_using_data_list_size(tudl) > task_available_max)
						{
							task_available_max = task_using_data_list_size(tudl);
							handle_popped = e2->pointer_to_data_to_pop_next;
						}
					}
				}
				// erase le handle de la liste des N data a pop 
					e2 = data_to_pop_next_list_begin(g->my_data_to_pop_next);
					while (e2->pointer_to_data_to_pop_next != handle_popped)
					{
					   e2 = data_to_pop_next_list_next(e2);
					} 
					data_to_pop_next_list_erase(g->my_data_to_pop_next, e2);
				number_of_task_max = 1;
			}
		}*/
		//~ #ifdef PRINT
		//~ gettimeofday(&time_end_choose_best_data, NULL);
		//~ time_total_choose_best_data += (time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec;
		//~ #endif
		// Fin de version où je récup les 10 meilleures données et ensuite tant que je les ai pas toutes envoyé je test sur ces 10 données la. 
    /*}*/
    
    //~ if (number_of_task_max == 0)
    //~ {
		//~ goto random;
    //~ }
    //~ else // I erase the data. 
    //~ {
		//~ e = gpu_data_not_used_list_begin(g->gpu_data);
		//~ while (e->D != handle_popped)
		//~ {
		   //~ e = gpu_data_not_used_list_next(e);
		//~ } 
		//~ gpu_data_not_used_list_erase(g->gpu_data, e);
    //~ }
	
     //~ // Adding the task to the list. TODO : this is a copy paste of the code above to test the available tasks.
      //~ TODO : cette partie ne marchera que en 2D ? Je sais pas à tester 
     
     //~ // a tester apres, borner à 50 taches par exemple si les perfs ne s'améliore pas. Mais en réalité cela prend tellement peu de temps que ce n'est pas nécessaire. 
     /*
    int fill_planned_task_list_threshold = starpu_get_env_number_default("FILL_PLANNED_TASK_LIST_THRESHOLD", 0);
    if (fill_planned_task_list_threshold == 0)
    {
		fill_planned_task_list_threshold = INT_MAX;
	}
	i = 0; */
    //~ #ifdef PRINT
    //~ gettimeofday(&time_start_fill_planned_task_list, NULL);
    //~ #endif
    
    //~ for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
    //~ {
		//~ data_available = true; 
		//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
		//~ {		    		
			//~ if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
			//~ {
				//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
				//~ {
					//~ data_available = false;
					//~ break;
				//~ }
			//~ }
		//~ }
		//~ if (data_available == true)
		//~ {
			//~ // Add it from planned task compteur 
			//~ increment_planned_task_data(t->pointer_to_T, current_gpu);
			
			//~ erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
			//~ starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
		//~ }
    //~ }
    
    //~ #ifdef PRINT
    //~ gettimeofday(&time_end_fill_planned_task_list, NULL);
    //~ time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
    //~ #endif
    
    //~ // If no task have been added to the list. 
    //~ if (starpu_task_list_empty(&g->planned_task)) 
    //~ {
		//~ random: ;
		//~ struct starpu_task task = starpu_task_list_pop_front(main_task_list);		
		//~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		//~ {
			//~ if (!gpu_data_not_used_list_empty(g->gpu_data))
			//~ {
				//~ for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
				//~ {
					//~ if(e->D == STARPU_TASK_GET_HANDLE(task, i))
					//~ {
						//~ gpu_data_not_used_list_erase(g->gpu_data, e);
					//~ }
				//~ }
			//~ }
		//~ }
		
		//~ // Add it from planned task compteur 
		//~ increment_planned_task_data(task, current_gpu);
		
		//~ erase_task_and_data_pointer(task, main_task_list);
		//~ starpu_task_list_push_back(&g->planned_task, task);
    //~ }
    
    //~ #ifdef PRINT
    //~ gettimeofday(&time_end_schedule, NULL);
    //~ time_total_schedule += (time_end_schedule.tv_sec - time_start_schedule.tv_sec)1000000LL + time_end_schedule.tv_usec - time_start_schedule.tv_usec;
	//~ #endif
//~ }

/* Fill a package's task list following dynamic_data_aware algorithm but with the 3D variant. */
//~ Chargement
//~ Si je trouve une donnée qui me donne des taches gratuites je prends et j'ajoute a planned task
//~ Sinon en chargeant 1 nouvelle donnée, je charge une tache donbt la donnée amene le plus de tache a 1 seul chargement : (si c'est ce cas on créé une liste planned_task_1dtata__to_load)
//~ Sinon random
//~ Itération suivante
//~ (On regarde les données des taches de la liste planned_task_1dtata_to_load, si on trouve 1 donnée la dedans)

//~ Eviction
//~ Pareil pour le moment. Attention a bien toujours mettre à jour planned task

//~ Dans un coin de la tete : idée de liste intermédiaire
void dynamic_data_aware_scheduling_3D_matrix(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g)
{
	//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	#ifdef PRINT
	gettimeofday(&time_start_schedule, NULL);
	#endif
	
    int i = 0;
    int j = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    int task_available_max = 0;
    int task_available_max_1_from_free = 0;
    
    /* Le nombre de tâches gratuites où à 1 données d'être gratuite par donnée. */
    int number_free_task_max = 0;
    int temp_number_free_task_max = 0;
    int number_1_from_free_task_max = 0;
    int temp_number_1_from_free_task_max = 0;
    
    starpu_data_handle_t handle_popped = NULL;
    struct task_using_data_list *tudl = task_using_data_list_new();
    
	#ifdef PRINT
	printf("Il y a %d données parmi lesquelles choisir.\n", gpu_data_not_used_list_size(g->gpu_data));
	#endif
	
	#ifdef PRINT
	int data_choosen_index = 0;
	FILE *f = NULL;
	f = fopen("Output_maxime/DARTS_data_choosen_stats.csv", "a");
	int nb_data_looked_at = 0; //Uniquement e cas ou on choisis depuis la mémoire
	#endif
	
    /* Si c'est la première tâche, on regarde pas le reste on fais random. */
    if (g->first_task == true)
    {
		#ifdef PRINT
		printf("Hey! C'est la première tâche du GPU n°%d!\n", current_gpu);
		fprintf(f, "%d,%d,%d\n", g->number_data_selection, 0, 0);
		fclose(f);
		#endif
		
		g->first_task = false;
				
		if (natural_order == 2)
		{
			struct starpu_task *task = g->first_task_to_pop;
			
			if (!starpu_task_list_ismember(main_task_list, task))
			{
				goto random;
			}
			
			g->first_task_to_pop = NULL;		
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (!gpu_data_not_used_list_empty(g->gpu_data))
				{
					for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
					{
						if(e->D == STARPU_TASK_GET_HANDLE(task, i))
						{
							gpu_data_not_used_list_erase(g->gpu_data, e);
						}
					}
				}
			}
			/* Add it from planned task compteur */
			//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
			
			increment_planned_task_data(task, current_gpu);
			
			#ifdef PRINT
			printf("Returning first task of GPU n°%d in natural order: %p.\n", current_gpu, task);
			#endif
			
			erase_task_and_data_pointer(task, main_task_list);
			starpu_task_list_push_back(&g->planned_task, task);
			
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			
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
    	
	/* Pour diminuer au début de l'execution */
	int choose_best_data_threshold = INT_MAX;
	if (threshold == 1)
	{
		/* En 2D on fais cela */
		if (app == 0)
		{
			if (NT_dynamic_outer > 14400)
			{
				choose_best_data_threshold = 110;
			}
		}
		else if (NT_dynamic_outer > 1599) /* Pour que ca se déclanche au 4ème point en 3D */
		{
			choose_best_data_threshold = 200;
		}
	}
	
	struct handle_user_data * hud = NULL;
	
	#ifdef PRINT
	gettimeofday(&time_start_choose_best_data, NULL);
	#endif
	
	//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		
	/* Recherche de la meilleure donnée. Je regarde directement pour chaque donnée, le nombre de tâche qu'elle met à 1 donnée d'être possible si j'ai toujours
	 * 0 à number_free_task_max. */
	if (choose_best_data_from == 0) /* Le cas de base où je regarde les données pas encore utilisées. */
	{
		#ifdef PRINT
		g->number_data_selection++;
		#endif
		
		for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data) && i != choose_best_data_threshold; e = gpu_data_not_used_list_next(e), i++)
		{			
			temp_number_free_task_max = 0;
			temp_number_1_from_free_task_max = 0;
			
			#ifdef PRINT
			nb_data_looked_at++;
			#endif
			
			/* Il y a deux cas pour simplifier un peu la complexité. Si j'ai au moins 1 tâche qui peut être gratuite, je ne fais plus les compteurs qui permettent 
			 * d'avoir des tâches à 1 donnée d'être free et on ajoute un break pour gagner un peu de temps. */
			if (number_free_task_max == 0 && app != 0)
			//~ if (number_free_task_max == 0)
			{	
				for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
				{
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
								/* Nouveau */
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
						 * Nouvelle place du threshold == 2. */
						if (threshold == 2)
						{
							handle_popped = e->D;
							number_free_task_max = temp_number_free_task_max;
							goto end_choose_best_data;
						}
					}
					else if (data_not_available == 1)
					{
						temp_number_1_from_free_task_max++;
					}
				}
				if (temp_number_free_task_max > 0)
				{
					number_free_task_max = temp_number_free_task_max;
					task_available_max = task_using_data_list_size(e->D->sched_data);
					handle_popped = e->D;
					
					#ifdef PRINT
					data_choosen_index = i + 1;
					#endif
									
					//~ /* Version où je m'arrête dès que j'ai une tâche gratuite. */
					//~ if (threshold == 2)
					//~ {
						//~ goto end_choose_best_data;
					//~ }
				}
				else if (temp_number_1_from_free_task_max > number_1_from_free_task_max)
				{
					number_1_from_free_task_max = temp_number_1_from_free_task_max;
					task_available_max_1_from_free = task_using_data_list_size(e->D->sched_data);
					handle_popped = e->D;
					
					#ifdef PRINT
					data_choosen_index = i + 1;
					#endif
				}
				/* Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. */
				/* TODO : est-ce vraiment nécessaire ? Si ca prend du temps autant le retirer */
				else if (temp_number_1_from_free_task_max == number_1_from_free_task_max && number_1_from_free_task_max != 0)
				{
					tudl = e->D->sched_data;
					if (task_using_data_list_size(tudl) > task_available_max_1_from_free)
					{
						task_available_max_1_from_free = task_using_data_list_size(tudl);
						handle_popped = e->D;
						
						#ifdef PRINT
						data_choosen_index = i + 1;
						#endif
					}
				}
			}
			else /* La version plus courte similaire à celle dans M2D. */
			{
				for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
				{
					/* I put it at false if at least one data is missing. */
					data_available = true;
					for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
					{
						/* I test if the data is on memory */ 
						if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
						{							
							if (simulate_memory == 0)
							{
								/* Ancien */
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
								{
									data_available = false;
									break;
								}
							}
							else if (simulate_memory == 1)
							{
								/* Nouveau */
								hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
								if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
								{
									data_available = false;
									break;
								}
							}
							
							//~ /* Avec ifdef */
							//~ #ifdef SIMMEM
								//~ hud = STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)->user_data;
								//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu) && hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
								//~ {
									//~ data_available = false;
									//~ break;
								//~ }
							//~ #else
								//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
								//~ {
									//~ data_available = false;
									//~ break;
								//~ }
							//~ #endif
						}
					}
					if (data_available == true)
					{
						temp_number_free_task_max++;
						/* Version où je m'arrête dès que j'ai une tâche gratuite.
						 * Nouvelle place du threshold == 2. */
						if (threshold == 2)
						{
							handle_popped = e->D;
							number_free_task_max = temp_number_free_task_max;
							goto end_choose_best_data;
						}
					}
				}
			
				if (temp_number_free_task_max > number_free_task_max)
				{
					number_free_task_max = temp_number_free_task_max;
					task_available_max = task_using_data_list_size(e->D->sched_data);
					handle_popped = e->D;
					
					#ifdef PRINT
					data_choosen_index = i + 1;
					#endif
					
					//~ /* Version où je m'arrête dès que j'ai une tâche gratuite. */
					//~ if (threshold == 2)
					//~ {
						//~ goto end_choose_best_data;
					//~ }
				}
				/* Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. Attention içi ce n'est pas adapté au 3D
				 * car tu pourrais avoir des tâches qui amène plus de tache a 1 seule donnée d'être free*/
				else if (temp_number_free_task_max == number_free_task_max && number_free_task_max != 0)
				{
					tudl = e->D->sched_data;
					if (task_using_data_list_size(tudl) > task_available_max)
					{
						task_available_max = task_using_data_list_size(tudl);
						handle_popped = e->D;
						
						#ifdef PRINT
						data_choosen_index = i + 1;
						#endif
					}
				}
			}
		}
	}
	else if (choose_best_data_from == 1) /* Le cas où je regarde uniquement les données (pas encore en mémoire) des tâches des données en mémoire. */
	{
		/* Pour ne pas regarder deux fois à la même itération la même donnée. */
		struct handle_user_data * hud_last_check = NULL;
		
		/* Attention ici c'est utile ne pas le metre entre des ifdef!!!! */
		g->number_data_selection++;
		
		starpu_data_handle_t *data_on_node;
		unsigned nb_data_on_node = 0;
		int *valid;
		starpu_data_get_node_data(current_gpu, &data_on_node, &valid, &nb_data_on_node);
		struct task_using_data *t2 = NULL;
		int k = 0;
		//~ printf("nb data on node = %d.\n", nb_data_on_node);
		/* Je me met sur une donnée de la mémoire. */
		for (i = 0; i < nb_data_on_node; i++)
		{
			/* Je me met sur une tâche de cette donnée en question. */
			for (t2 = task_using_data_list_begin(data_on_node[i]->sched_data); t2 != task_using_data_list_end(data_on_node[i]->sched_data); t2 = task_using_data_list_next(t2))
			{
				//~ printf("task = %p.\n", task);
				/* Je me met sur une donnée de cette tâche (qui n'est pas celle en mémoire). */
				for (k = 0; k < STARPU_TASK_GET_NBUFFERS(t2->pointer_to_T); k++)
				{						
					hud_last_check = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data;
					/* Ici il faudrait ne pas regarder 2 fois la même donnée si possible. Ca peut arriver oui. */
					if (STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k) != data_on_node[i] && hud_last_check->last_check_to_choose_from[current_gpu - 1] != g->number_data_selection)
					{
						#ifdef PRINT
						nb_data_looked_at++;
						#endif	
						
						/* Mise à jour de l'itération pour la donnée pour ne pas la regarder deux fois à cette itération. */
						hud_last_check->last_check_to_choose_from[current_gpu - 1] = g->number_data_selection;
						STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->user_data = hud_last_check;
		
						temp_number_free_task_max = 0;
						temp_number_1_from_free_task_max = 0;
				
						if (number_free_task_max == 0 && app != 0)
						{
							/* Je regarde le nombre de free ou 1 from free tâche de cette donnée. */
							for (t = task_using_data_list_begin(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t != task_using_data_list_end(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t = task_using_data_list_next(t))
							{
								data_not_available = 0; 
								for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
								{
									if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k))
									{
										if (simulate_memory == 0)
										{
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
									 * Nouvelle place du threshold == 2. */
									if (threshold == 2)
									{
										number_free_task_max = temp_number_free_task_max;
										handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
										goto end_choose_best_data;
									}	
								}
								else if (data_not_available == 1)
								{
									temp_number_1_from_free_task_max++;
								}
							}
							if (temp_number_free_task_max > 0)
							{
								number_free_task_max = temp_number_free_task_max;
								task_available_max = task_using_data_list_size(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data);
								handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
								
								#ifdef PRINT
								data_choosen_index = nb_data_looked_at;
								#endif
								
								//~ /* Anciene place du threshold==2 */
								//~ if (threshold == 2)
								//~ {
									//~ printf("ancien go to de th==2 %p.\n", STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k));
									//~ goto end_choose_best_data;
								//~ }			
							}
							else if (temp_number_1_from_free_task_max > number_1_from_free_task_max)
							{
								number_1_from_free_task_max = temp_number_1_from_free_task_max;
								task_available_max_1_from_free = task_using_data_list_size(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data);
								handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
								
								#ifdef PRINT
								data_choosen_index = nb_data_looked_at;
								#endif
								
							}
							else if (temp_number_1_from_free_task_max == number_1_from_free_task_max && number_1_from_free_task_max != 0)
							{
								tudl = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data;
								if (task_using_data_list_size(tudl) > task_available_max_1_from_free)
								{
									task_available_max_1_from_free = task_using_data_list_size(tudl);
									handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
									
									#ifdef PRINT
									data_choosen_index = nb_data_looked_at;
									#endif
									
								}
							}
						}
						else /* Cas 2D */
						{
							for (t = task_using_data_list_begin(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t != task_using_data_list_end(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data); t = task_using_data_list_next(t))
							{
								data_available = true;
								for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
								{
									if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k))
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
									temp_number_free_task_max++;
										/* Version où je m'arrête dès que j'ai une tâche gratuite.
										 * Nouvelle place du threshold == 2. */
										if (threshold == 2)
										{
											handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
											number_free_task_max = temp_number_free_task_max;
											goto end_choose_best_data;
										}
								}
							}
							if (temp_number_free_task_max > number_free_task_max)
							{
								number_free_task_max = temp_number_free_task_max;
								task_available_max = task_using_data_list_size(STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data);
								handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
								
								#ifdef PRINT
								data_choosen_index = nb_data_looked_at;
								#endif
								
								//~ /* Version où je m'arrête dès que j'ai une tâche gratuite. */
								//~ if (threshold == 2)
								//~ {
									//~ goto end_choose_best_data;
								//~ }
							}
							else if (temp_number_free_task_max == number_free_task_max && number_free_task_max != 0)
							{
								tudl = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k)->sched_data;
								if (task_using_data_list_size(tudl) > task_available_max)
								{
									task_available_max = task_using_data_list_size(tudl);
									handle_popped = STARPU_TASK_GET_HANDLE(t2->pointer_to_T, k);
									
									#ifdef PRINT
									data_choosen_index = nb_data_looked_at;
									#endif
								}
							}
						}
					}	
				}
			}
		}
	}	
	//~ #ifdef PRINT
	//~ printf("best data is = %p %d free tasks and/or %d 1 from free tasks.\n", handle_popped, number_free_task_max, number_1_from_free_task_max);   
	//~ #endif
	
	end_choose_best_data : ;
	
	#ifdef PRINT
	fprintf(f, "%d,%d,%d\n", g->number_data_selection, data_choosen_index, nb_data_looked_at - data_choosen_index);
	fclose(f);
	#endif
	
	#ifdef PRINT
    gettimeofday(&time_end_choose_best_data, NULL);
	time_total_choose_best_data += (time_end_choose_best_data.tv_sec - time_start_choose_best_data.tv_sec)*1000000LL + time_end_choose_best_data.tv_usec - time_start_choose_best_data.tv_usec;
    #endif
    
    if (number_free_task_max != 0) /* cas comme dans 2D, je met dans planned_task les tâches gratuites, sauf que j'ai 3 données à check et non 2. */
    {
		#ifdef PRINT
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		#endif
	
		/* I erase the data from the list of data not used. See env var ERASE_DATA_STRATEGY */
		if (choose_best_data_from == 0)
		{
			/* Je l'erase de 1 seul GPU ou de tous. */
			//~ if (erase_data_strategy == 0)
			//~ {
				e = gpu_data_not_used_list_begin(g->gpu_data);
				while (e->D != handle_popped)
				{
					  e = gpu_data_not_used_list_next(e);
				}
				gpu_data_not_used_list_erase(g->gpu_data, e);
			//~ }
			//~ else
			//~ {
				//~ my_planned_task_control->pointer = my_planned_task_control->first;
				//~ for (i = 0; i < Ngpu; i++)
				//~ {
					//~ e = gpu_data_not_used_list_begin(my_planned_task_control->pointer->gpu_data);
					//~ for (j = 0; j < gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data); j++)
					//~ {
						//~ if (e->D == handle_popped)
						//~ {
							//~ gpu_data_not_used_list_erase(my_planned_task_control->pointer->gpu_data, e);
							//~ break;
						//~ }
						//~ e = gpu_data_not_used_list_next(e);
					//~ }
					//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
				//~ }
			//~ }
		}

		#ifdef PRINT
		printf("The data adding the most free tasks is %p.\n", handle_popped);
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
				//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
				/* Add it from planned task compteur */
				increment_planned_task_data(t->pointer_to_T, current_gpu);
				
				#ifdef PRINT
				printf("Pushing free %p in planned_task of GPU %d :", t->pointer_to_T, current_gpu);
				for (i = 0; i < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); i++)
				{
					printf(" %p", STARPU_TASK_GET_HANDLE(t->pointer_to_T, i));
				}
				printf("\n");
				#endif
				
				erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
				starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
				//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			}
		}
		
		#ifdef PRINT
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
		#endif
	}
	/* La je change par rapport à 2D, si à la fois free et 1_from_free sont à 0 je renvoie random */   
	else if (number_1_from_free_task_max != 0 && app != 0) /* On prend une tâche de la donnée 1_from_free, dans l'ordre randomisé de la liste de tâches. */
	{
		#ifdef PRINT
		gettimeofday(&time_start_fill_planned_task_list, NULL);
		printf("The data adding the most (%d) 1_from_free tasks is %p.\n", number_1_from_free_task_max, handle_popped);
		#endif
		
		/* I erase the data from the list of data not used. See env var ERASE_DATA_STRATEGY */
		if (choose_best_data_from == 0)
		{
			/* Je l'erase de 1 seul GPU ou de tous. */
			//~ if (erase_data_strategy == 0)
			//~ {
				e = gpu_data_not_used_list_begin(g->gpu_data);
				while (e->D != handle_popped)
				{
					  e = gpu_data_not_used_list_next(e);
				}
				gpu_data_not_used_list_erase(g->gpu_data, e);
			//~ }
			//~ else
			//~ {
				//~ my_planned_task_control->pointer = my_planned_task_control->first;
				//~ for (i = 0; i < Ngpu; i++)
				//~ {
					//~ e = gpu_data_not_used_list_begin(my_planned_task_control->pointer->gpu_data);
					//~ for (j = 0; j < gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data); j++)
					//~ {
						//~ if (e->D == handle_popped)
						//~ {
							//~ gpu_data_not_used_list_erase(my_planned_task_control->pointer->gpu_data, e);
							//~ break;
						//~ }
						//~ e = gpu_data_not_used_list_next(e);
					//~ }
					//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
				//~ }
			//~ }
		}
		
		/* Nouvelle version où au lieu de bêtement prendre une tâche de la donnée élu, je vais regarder si la tâche est bien 1 from free. */
		for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
		{
			/* Finding just one task that is one from free with the choosen data. */
			data_not_available = 0; 
			for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
			{
				if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
				{
					if (simulate_memory == 0)
					{
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
			if (data_not_available == 1)
			{
				break;
			}
		}
		
		/* OLD version : Random car la liste est randomisé */
		//~ t = task_using_data_list_begin(handle_popped->sched_data);
		
		/* Add it from planned task compteur */
		//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		increment_planned_task_data(t->pointer_to_T, current_gpu);
		
		#ifdef PRINT
		printf("Pushing 1_from_free task %p in planned_task of GPU %d\n", t->pointer_to_T, current_gpu);
		#endif
		
		erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
		starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		#ifdef PRINT
		gettimeofday(&time_end_fill_planned_task_list, NULL);
		time_total_fill_planned_task_list += (time_end_fill_planned_task_list.tv_sec - time_start_fill_planned_task_list.tv_sec)*1000000LL + time_end_fill_planned_task_list.tv_usec - time_start_fill_planned_task_list.tv_usec;
		#endif
		
	}
	else /* Sinon random */
	{
		#ifdef PRINT
		printf("Random selection because no data allow to get free or 1 from free tasks.\n");
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		goto random;
	}
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&g->planned_task)) 
    {
		random: ;

		#ifdef PRINT
		gettimeofday(&time_start_pick_random_task, NULL);
		number_random_selection++;
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		
		struct starpu_task *task = starpu_task_list_pop_front(main_task_list);	
		
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
							gpu_data_not_used_list_erase(g->gpu_data, e);
						}
					}
				}
			}
		}
		
		/* Add it from planned task compteur */
		increment_planned_task_data(task, current_gpu);
		
		#ifdef PRINT
		printf("Returning head of the randomized main task list: %p.\n", task);
		#endif
		
		erase_task_and_data_pointer(task, main_task_list);
		starpu_task_list_push_back(&g->planned_task, task);
		
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		
		#ifdef PRINT
		gettimeofday(&time_end_pick_random_task, NULL);
		time_total_pick_random_task += (time_end_pick_random_task.tv_sec - time_start_pick_random_task.tv_sec)*1000000LL + time_end_pick_random_task.tv_usec - time_start_pick_random_task.tv_usec;
		#endif
		
		return;
    }
    
    end_scheduling: ;
	//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    
    #ifdef PRINT
    gettimeofday(&time_end_schedule, NULL);
    time_total_schedule += (time_end_schedule.tv_sec - time_start_schedule.tv_sec)*1000000LL + time_end_schedule.tv_usec - time_start_schedule.tv_usec;
	#endif
}

void increment_planned_task_data(struct starpu_task *task, int current_gpu)
{
	int i = 0;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		hud->nb_task_in_planned_task[current_gpu - 1] = hud->nb_task_in_planned_task[current_gpu - 1] + 1;
		STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
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
	#ifdef PRINT
	gettimeofday(&time_start_evicted, NULL);
	#endif
	
	STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	//~ printf("Début de victim evicted avec %p. Success = %d. Nb de fois dans victim evicted total %d.\n", victim, success, victim_evicted_compteur); fflush(stdout);
     /* If a data was not truly evicted I put it back in the list. */
	int i = 0;
			
	my_planned_task_control->pointer = my_planned_task_control->first;
	for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
	{
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	}
	my_planned_task_control->pointer->data_to_evict_next = victim;
	
	STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);

	#ifdef PRINT
	gettimeofday(&time_end_evicted, NULL);
	time_total_evicted += (time_end_evicted.tv_sec - time_start_evicted.tv_sec)*1000000LL + time_end_evicted.tv_usec - time_start_evicted.tv_usec;
	#endif
}

/* TODO: return NULL ou ne rien faire si la dernière tâche est sorti du post exec hook ? De même pour la mise à jour des listes à chaque eviction de donnée.
 * TODO je rentre bcp trop dans cette fonction on perds du temps car le timing avance lui. Résolu en réduisant le threshold et en adaptant aussi CUDA_PIPELINE. */
starpu_data_handle_t dynamic_data_aware_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{    
	//~ STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	
	#ifdef PRINT
	gettimeofday(&time_start_selector, NULL);
	#endif
	
    int i = 0;
    int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
    
    /* Se placer sur le bon GPU pour planned_task */
    my_planned_task_control->pointer = my_planned_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    
    /* Je check si une eviction n'a pas été refusé. */
    if (my_planned_task_control->pointer->data_to_evict_next != NULL) 
    { 
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Return data %p that was refused.\n", my_planned_task_control->pointer->data_to_evict_next); }
		starpu_data_handle_t temp_handle = my_planned_task_control->pointer->data_to_evict_next;
		my_planned_task_control->pointer->data_to_evict_next = NULL;
		
		#ifdef PRINT
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		return temp_handle;
    }
        
    starpu_data_handle_t *data_on_node;
    unsigned nb_data_on_node = 0;
    int *valid;
    starpu_data_handle_t returned_handle = STARPU_DATA_NO_VICTIM;
    starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);
    
    /* Get the the min number of task a data can do in pulled_task */
    /* Se placer sur le bon GPU pour pulled_task */
     my_pulled_task_control->pointer = my_pulled_task_control->first;
     for (i = 1; i < current_gpu; i++)
 	 {
		my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
	 }
    
    int min_number_task_in_pulled_task = INT_MAX;
    int nb_task_in_pulled_task[nb_data_on_node];
    
    for (i = 0; i < nb_data_on_node; i++)
    {
		nb_task_in_pulled_task[i] = 0;
    }
    
    /* Je cherche le nombre de tâche dans le pulled_task que peut faire chaque données */
    /* OLD */
    //~ for (i = 0; i < nb_data_on_node; i++)
    //~ {
		//~ if (starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		//~ {
			//~ for (p = pulled_task_list_begin(my_pulled_task_control->pointer->ptl); p!= pulled_task_list_end(my_pulled_task_control->pointer->ptl); p = pulled_task_list_next(p))
			//~ {
				//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
				//~ {
					//~ if (STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j) == data_on_node[i])
					//~ {
						//~ nb_task_in_pulled_task[i]++;
						//~ break;
					//~ }
				//~ }
			//~ }
			//~ if (nb_task_in_pulled_task[i] < min_number_task_in_pulled_task)
			//~ {
				//~ min_number_task_in_pulled_task = nb_task_in_pulled_task[i];
			//~ }
		//~ }
		//~ else
		//~ {
			//~ /* - 1 si j'ai pas le droit d'évincer cette donnée */
			//~ nb_task_in_pulled_task[i] = -1;
		//~ }
    //~ }
    
    /* NEW avec la struct nb_task_in_pulled_task */
    //~ if (starpu_get_env_number_default("PRINTF", 0) == 1) 
    //~ { 
		//~ struct gpu_pulled_task *g = my_pulled_task_control->first;
		//~ print_pulled_task_one_gpu(g, 1);
	//~ }
	
    struct handle_user_data *hud = malloc(sizeof(hud));
    for (i = 0; i < nb_data_on_node; i++)
    {
		if (starpu_data_can_evict(data_on_node[i], node, is_prefetch))
		{
			hud = data_on_node[i]->user_data;
			nb_task_in_pulled_task[i] = hud->nb_task_in_pulled_task[current_gpu - 1];
			
			/* Ajout : si sur les deux lists c'est 0 je la return direct la data */
			if (hud->nb_task_in_pulled_task[current_gpu - 1] == 0 && hud->nb_task_in_planned_task[current_gpu - 1] == 0)
			{
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
    //~ if (starpu_get_env_number_default("PRINTF", 0) == 1) { printf("Min = %d.\n", min_number_task_in_pulled_task); }
    
    /* Si il vaut 0 je fais belady sur les données utilisé par les tâches de pulled_task, 
     * sinon je choisis min(NT_dynamic_outer/W(D(T)) + W(Di) * W(Di)).
     * Si il vaut -1 c'est que je n'avais aucune donnée à renvoyer car aucune n'est évincable.
     */
    if (min_number_task_in_pulled_task == INT_MAX)
    {		
		#ifdef PRINT
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		return STARPU_DATA_NO_VICTIM;
    }
    else if (min_number_task_in_pulled_task == 0)
    {
		/* Au moins 1 donnée ne sert pas dans pulled_task */
		/* OLD */
		//~ returned_handle = min_weight_average_on_planned_task(data_on_node, nb_data_on_node, node, is_prefetch, my_planned_task_control->pointer, nb_task_in_pulled_task);
		
		/* NEW */
		returned_handle = least_used_data_on_planned_task(data_on_node, nb_data_on_node, my_planned_task_control->pointer, nb_task_in_pulled_task, current_gpu);
    }
    else
    {
		/* Au moins 1 donnée sert dans pulled_task */
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("#warning min number of task done by data on node is != 0.\n"); }
	
		/* Si c'est un prefetch qui demande une eviction de ce qui est utile pour les tâches de pulled task je renvoie NO VICTIM si >= à STARPU_TASK_PREFETCH */
		if (is_prefetch >= 1)
		{
			#ifdef PRINT
			gettimeofday(&time_end_selector, NULL);
			time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
			#endif
			
			//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
			return STARPU_DATA_NO_VICTIM;
		}
		
		returned_handle = belady_on_pulled_task(data_on_node, nb_data_on_node, node, is_prefetch, my_pulled_task_control->pointer);
    }
    
    /* Ca devrait pas arriver a enleevr et a tester */
    if (returned_handle == NULL)
    {
		//~ victim_selector_return_no_victim++;
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Return STARPU_DATA_NO_VICTIM in victim selector car NULL dans returned_handle de belady planned ou pulled task.\n"); }
		#ifdef PRINT
		gettimeofday(&time_end_selector, NULL);
		time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
		#endif
		
		//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		
		return STARPU_DATA_NO_VICTIM; 
    }
    
    deletion_in_victim_selector : ;
    
    STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
    
    struct starpu_task *task = NULL;
    struct starpu_sched_component *temp_component = component;
    struct dynamic_data_aware_sched_data *data = temp_component->data;
    /* Enlever de la liste de tache a faire celles qui utilisais cette donnée. Et donc ajouter cette donnée aux données
     * à pop ainsi qu'ajouter la tache dans les données. Also add it to the main task list. */
     
    /* Suppression de la liste de planned task les tâches utilisant la donnée que l'on s'apprête à évincer. */
    if (min_number_task_in_pulled_task == 0)
    {
		for (task = starpu_task_list_begin(&my_planned_task_control->pointer->planned_task); task != starpu_task_list_end(&my_planned_task_control->pointer->planned_task); task = starpu_task_list_next(task))
		{
			for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			{
				if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
				{
					/* Suppression de la liste de tâches à faire */
					struct pointer_in_task *pt = task->sched_data;
					starpu_task_list_erase(&my_planned_task_control->pointer->planned_task, pt->pointer_to_cell);
						
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
					}	
					task->sched_data = pt;
						
					/* Ajout a la liste de tâches principales ces mêmes tâches. */
					starpu_task_list_push_back(&data->main_task_list, task);
					break;
				}
			}
		}   
    }
	
    /*Placing in a random spot of the data list to use the evicted handle */
    /* Je ne le fais pas dans le cas ou on choisis depuis la mémoire ou ailleurs */
    if (choose_best_data_from == 0)
    {
		/* Si une donnée n'a plus rien à faire je ne la remet pas dans la liste des donnée parmi lesquelles choisir. */
		if (!task_using_data_list_empty(returned_handle->sched_data))
		{
			push_data_not_used_yet_random_spot(returned_handle, my_planned_task_control->pointer);
		}
	}
	
    //~ if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Return %p in victim selector.\n", returned_handle); }
    #ifdef PRINT
    gettimeofday(&time_end_selector, NULL);
	time_total_selector += (time_end_selector.tv_sec - time_start_selector.tv_sec)*1000000LL + time_end_selector.tv_usec - time_start_selector.tv_usec;
	#endif
	
	STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
	
    return returned_handle;
}

starpu_data_handle_t belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_pulled_task *g)
{
	#ifdef PRINT
	gettimeofday(&time_start_belady, NULL);
	#endif
	
    int i = 0;
    int j = 0;
    int next_use = 0;
    int max_next_use = -1;
    struct pulled_task *p = pulled_task_new();
    starpu_data_handle_t returned_handle = NULL;
    
    //print_pulled_task_one_gpu(g, node);
    for (i = 0; i < nb_data_on_node; i++)
    {
		if (starpu_data_can_evict(data_tab[i], node, is_prefetch))
		{
			next_use = 0;
			for (p = pulled_task_list_begin(g->ptl); p != pulled_task_list_end(g->ptl); p = pulled_task_list_next(p))
			{
				//~ printf("On %p\n", p->pointer_to_pulled_task); fflush(stdout);
				for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
				{
					next_use++;
					if (STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j) == data_tab[i])
					{
						//printf("Next use of %p is %d.\n", STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j), next_use);
						if (max_next_use < next_use)
						{
							max_next_use = next_use;
							returned_handle = data_tab[i];
						}
						goto break_nested_for_loop;
					}
				}
			}
			break_nested_for_loop : ;
		}
    }

    #ifdef PRINT
    gettimeofday(&time_end_belady, NULL);
    time_total_belady += (time_end_belady.tv_sec - time_start_belady.tv_sec)*1000000LL + time_end_belady.tv_usec - time_start_belady.tv_usec;
    #endif
    
    return returned_handle;
}

/* Belady sur lesp lanned task. Je check pas is on node car c'est -1 dans le tab du nb de taches dans pulled task */
starpu_data_handle_t least_used_data_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, struct gpu_planned_task *g, int *nb_task_in_pulled_task, int current_gpu)
{
	#ifdef PRINT
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
	
	#ifdef PRINT
	gettimeofday(&time_end_least_used_data_planned_task, NULL);
	time_total_least_used_data_planned_task += (time_end_least_used_data_planned_task.tv_sec - time_start_least_used_data_planned_task.tv_sec)*1000000LL + time_end_least_used_data_planned_task.tv_usec - time_start_least_used_data_planned_task.tv_usec;
    #endif
    
    return returned_handle;
}

/* TODO : normalement j'ai pas besoin de refaire un coup de if(data is on node) car j'ai mis -1 dans le tableau pour cela.
 * Fonction pas utilisé */
//~ starpu_data_handle_t min_weight_average_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_planned_task *g, int *nb_task_in_pulled_task)
//~ {
    //~ int i = 0;
    //~ int j = 0;
    //~ int k = 0;
    //~ int next_use = 0;
    //~ int data_count = 0;
    //~ int max_next_use = 0;
    //~ float min_weight_average = FLT_MAX;
    //~ float weight_average = 0;
    //~ float weight_missing_data = 0;
    //~ int NT_Di = 0;
    //~ struct starpu_task *task = NULL;
    //~ starpu_data_handle_t returned_handle = NULL;
    
    //~ /* To avoid duplicate data */
    //~ struct data_weighted_list *dwl = data_weighted_list_new();
    //~ struct data_weighted *dw = data_weighted_new();
    
    //~ for (i = 0; i < nb_data_on_node; i++)
    //~ {
		//~ if (nb_task_in_pulled_task[i] == 0)
		//~ {
			//~ NT_Di = 0;
			//~ weight_missing_data = 0;
			//~ next_use = 0;
			//~ data_count = 0;
			//~ for (task = starpu_task_list_begin(&g->planned_task); task != starpu_task_list_end(&g->planned_task); task = starpu_task_list_next(task))
			//~ {
				//~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
				//~ {
					//~ /* Au cas où il y a égalité et que je fais Belady */
					//~ data_count++;
					
					//~ if (STARPU_TASK_GET_HANDLE(task, j) == data_tab[i])
					//~ {
					//~ /* J'arrête de compter pour Belady */
					//~ next_use = data_count;
					
					//~ /* Je suis sur une tâche qui utilise Di */
					//~ NT_Di++;
					
					//~ /* J'ajoute ses données manquantes sauf Di au poids */
					//~ for (k = 0; k < STARPU_TASK_GET_NBUFFERS(task); k++)
					//~ {
						//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, k), node))
						//~ {
						//~ /* Je ne dois ajouter le poids de cette donnée que si elle n'a pas déjà été compté. */
						//~ if (data_weighted_list_empty(dwl))
						//~ {
							//~ struct data_weighted *new = data_weighted_new();
							//~ new->pointer_to_data_weighted = STARPU_TASK_GET_HANDLE(task, k);
							//~ data_weighted_list_push_back(dwl, new);
							//~ weight_missing_data += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, k));
						//~ }
						//~ else
						//~ {
							//~ for (dw = data_weighted_list_begin(dwl); dw != data_weighted_list_end(dwl); dw = data_weighted_list_next(dw))
							//~ {
							//~ if (STARPU_TASK_GET_HANDLE(task, k) == dw->pointer_to_data_weighted)
							//~ {
								//~ break;
							//~ }
							//~ struct data_weighted *new = data_weighted_new();
							//~ new->pointer_to_data_weighted = STARPU_TASK_GET_HANDLE(task, k);
							//~ data_weighted_list_push_back(dwl, new);
							//~ weight_missing_data += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, k));
							//~ }
						//~ }
						//~ }
					//~ }
					//~ break;
					//~ }
				//~ }
			//~ }
			//~ weight_average = (NT_Di/(weight_missing_data + starpu_data_get_size(data_tab[i])))*starpu_data_get_size(data_tab[i]);
			//~ if (min_weight_average > weight_average)
			//~ {
			//~ max_next_use = next_use; /* Au cas ou Belady */
			//~ min_weight_average = weight_average;
			//~ returned_handle = data_tab[i];
			//~ }
			//~ else if (min_weight_average == weight_average)
			//~ {
			//~ /* Je fais Belady sur planned_task */
			//~ if (next_use > max_next_use)
			//~ {
				//~ max_next_use = next_use;
				//~ returned_handle = data_tab[i];
			//~ }
			//~ }
		//~ }
    //~ }
    //~ return returned_handle;
//~ }

/* Erase a task from the main task list.
 * Also erase pointer in the data.
 * Only of one GPU.
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
    }
    starpu_task_list_erase(l, pt->pointer_to_cell);
}

static int dynamic_data_aware_can_push(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	//~ printf("Début de dynamic_data_aware_can_push.\n"); fflush(stdout);
    //~ struct dynamic_data_aware_sched_data *data = component->data;
    int didwork = 0;
    struct starpu_task *task;
    task = starpu_sched_component_pump_to(component, to, &didwork);
    //printf("task dans can push = %p.\n", task); fflush(stdout);
    if (task)
    {	    
	    /* If a task is refused I push it in the refused fifo list of the appropriate GPU's package.
	     * This list is looked at first when a GPU is asking for a task so we don't break the planned order. */
	     
	    STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
	    
	    my_planned_task_control->pointer = my_planned_task_control->first;
	    for (int i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++) 
	    {
			my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	    }
	    starpu_task_list_push_back(&my_planned_task_control->pointer->refused_fifo_list, task);
	    
	    STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    }
    
    /* There is room now */
    return didwork || starpu_sched_component_can_push(component, to);
}

static int dynamic_data_aware_can_pull(struct starpu_sched_component *component)
{
    return starpu_sched_component_can_pull(component);
}

void gpu_planned_task_initialisation()
{
    _STARPU_MALLOC( my_planned_task_control, sizeof(*my_planned_task_control));
    struct gpu_planned_task *new = malloc(sizeof(*new));
    
    starpu_task_list_init(&new->planned_task);
    starpu_task_list_init(&new->refused_fifo_list);
    new->gpu_data = NULL;
    new->data_to_evict_next = NULL;
    new->next = NULL;
    //~ new->my_data_to_pop_next = data_to_pop_next_list_new();
    new->first_task = true;
    new->number_data_selection = 0;
    
    my_planned_task_control->pointer = new;
    my_planned_task_control->first = my_planned_task_control->pointer;
}

void gpu_planned_task_insertion()
{
    struct gpu_planned_task *new = malloc(sizeof(*new));
    
    starpu_task_list_init(&new->planned_task);
    starpu_task_list_init(&new->refused_fifo_list);
    new->gpu_data = NULL;
    new->data_to_evict_next = NULL;
    new->next = my_planned_task_control->pointer;
    //~ new->my_data_to_pop_next = data_to_pop_next_list_new();
    new->first_task = true;
    new->number_data_selection = 0;
    
    my_planned_task_control->pointer = new;
}

void gpu_pulled_task_initialisation()
{
    _STARPU_MALLOC(my_pulled_task_control, sizeof(*my_pulled_task_control));
    struct gpu_pulled_task *new = malloc(sizeof(*new));
    struct pulled_task_list *p = pulled_task_list_new();
    new->ptl = p;
    
    my_pulled_task_control->pointer = new;
    my_pulled_task_control->first = my_pulled_task_control->pointer;
}

void gpu_pulled_task_insertion()
{
    struct gpu_pulled_task *new = malloc(sizeof(*new));
    struct pulled_task_list *p = pulled_task_list_new();
    new->ptl = p;
    
    new->next = my_pulled_task_control->pointer;    
    my_pulled_task_control->pointer = new;
}

void add_task_to_pulled_task(int current_gpu, struct starpu_task *task)
{
	int i = 0;	

	/* J'incrémente le nombre de tâches dans pulled task pour les données de task */
    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
		hud->nb_task_in_pulled_task[current_gpu - 1] = hud->nb_task_in_pulled_task[current_gpu - 1] + 1;
		STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
	}
	
    struct pulled_task *p = pulled_task_new();
    p->pointer_to_pulled_task = task;
    
    my_pulled_task_control->pointer = my_pulled_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
		my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    }
    pulled_task_list_push_back(my_pulled_task_control->pointer->ptl, p);
}

struct starpu_sched_component *starpu_sched_component_dynamic_data_aware_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	/* Var globale pour n'appeller qu'une seule fois get_env_number */
	eviction_strategy_dynamic_data_aware = starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_DATA_AWARE", 0);
	threshold = starpu_get_env_number_default("THRESHOLD", 0);
	app = starpu_get_env_number_default("APP", 0);
	choose_best_data_from = starpu_get_env_number_default("CHOOSE_BEST_DATA_FROM", 0);
	simulate_memory = starpu_get_env_number_default("SIMULATE_MEMORY", 0);
	//~ data_order = starpu_get_env_number_default("DATA_ORDER", 0);
	natural_order = starpu_get_env_number_default("NATURAL_ORDER", 0);
	//~ erase_data_strategy = starpu_get_env_number_default("ERASE_DATA_STRATEGY", 0);

	#ifdef PRINT
	print_in_terminal = starpu_get_env_number_default("PRINT_IN_TERMINAL", 0);
	print3d = starpu_get_env_number_default("PRINT3D", 0);
	print_n = starpu_get_env_number_default("PRINT_N", 0);
	print_time = starpu_get_env_number_default("PRINT_TIME", 0);
	FILE *f = NULL;
	f = fopen("Output_maxime/DARTS_data_choosen_stats.csv", "w");
	fprintf(f, "Iteration,Data choosen,Number of data read\n");
	fclose(f);	
	gettimeofday(&time_start_createtolasttaskfinished, NULL);
	#endif
	
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_data_aware");
	srandom(starpu_get_env_number_default("SEED", 0));
	int i = 0;
	
	/* Initialization of global variables. */
	Ngpu = get_number_GPU();
	NT_dynamic_outer = 0;
	NT = 0;
	new_tasks_initialized = false;
	
	/* For visualisation in python. */
	#ifdef PRINT
	index_current_popped_task = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_all_gpu = 0;
	index_current_popped_task_all_gpu_prefetch = 0;
	#endif
	
	gpu_memory_initialized = false;
	number_task_out_DARTS = -1;
	iteration_DARTS = 1;
	
	/* Initialization of structures. */
	struct dynamic_data_aware_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	//~ STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->main_task_list);
	
	/* Initialisation des structs de liste de tâches */
	gpu_planned_task_initialisation();
	for (i = 0; i < Ngpu - 1; i++)
	{
	    gpu_planned_task_insertion();
	}
	my_planned_task_control->first = my_planned_task_control->pointer;
	//~ STARPU_PTHREAD_MUTEX_INIT(&my_planned_task_control->planned_task_mutex, NULL);
	
	gpu_pulled_task_initialisation();
	for (i = 0; i < Ngpu - 1; i++)
	{
	    gpu_pulled_task_insertion();
	}
	my_pulled_task_control->first = my_pulled_task_control->pointer;
	//~ STARPU_PTHREAD_MUTEX_INIT(&my_pulled_task_control->first->pulled_task_mutex, NULL);
	//~ my_pulled_task_control->pointer = my_pulled_task_control->first;
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
		//~ my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
	    //~ STARPU_PTHREAD_MUTEX_INIT(&my_pulled_task_control->pointer->pulled_task_mutex, NULL);
	//~ }
	
	/* initialisation des mutexs. */
	STARPU_PTHREAD_MUTEX_INIT(&global_mutex, NULL);
	//~ local_mutex = malloc(Ngpu*sizeof(starpu_pthread_mutex_t));
	//~ for (i = 0; i < Ngpu; i++)
	//~ {
		//~ STARPU_PTHREAD_MUTEX_INIT(&local_mutex[i], NULL);
	//~ }
	
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
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_dynamic_data_aware_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

/* Get the task that was last executed. Used to update the task list of pulled task	 */
void get_task_done(struct starpu_task *task, unsigned sci)
{
	//~ STARPU_PTHREAD_MUTEX_LOCK(&local_mutex[starpu_worker_get_memory_node(starpu_worker_get_id()) - 1]);
	
	/* Je me place sur la liste correspondant au bon gpu. */
	int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
	int i = 0;
	
	/* TODO : increment de number_task_out_DARTS a faire ici */
	
	if (eviction_strategy_dynamic_data_aware == 1) 
	{
		STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		{
			struct handle_user_data * hud = STARPU_TASK_GET_HANDLE(task, i)->user_data;
			hud->nb_task_in_pulled_task[current_gpu - 1] = hud->nb_task_in_pulled_task[current_gpu - 1] - 1;
			STARPU_TASK_GET_HANDLE(task, i)->user_data = hud;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
	}
	
	
    //~ /* Je me place sur la liste correspondant au bon gpu. */
    //~ my_pulled_task_control->pointer = my_pulled_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
    //~ {
		//~ my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    //~ }
    
    struct pulled_task *temp = NULL;
    int trouve = 0;
    
    STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
    my_pulled_task_control->pointer = my_pulled_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
		my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    }
	
    /* J'efface la tâche dans la liste de tâches */
    if (!pulled_task_list_empty(my_pulled_task_control->pointer->ptl))
    {
		for (temp = pulled_task_list_begin(my_pulled_task_control->pointer->ptl); temp != pulled_task_list_end(my_pulled_task_control->pointer->ptl); temp = pulled_task_list_next(temp))
		{	
			if (temp->pointer_to_pulled_task == task)
			{
				trouve = 1;
				break;
			}
		}
		if (trouve == 1)
		{
			//~ printf("Popped task in get task done is %p.\n", temp->pointer_to_pulled_task); fflush(stdout);	
			pulled_task_list_erase(my_pulled_task_control->pointer->ptl, temp);
		}
		//~ else
		//~ {
			//~ printf("%p n'a pas été trouvé.\n", task); fflush(stdout);
		//~ }
    }
    STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    
    //~ STARPU_PTHREAD_MUTEX_UNLOCK(&local_mutex[starpu_worker_get_memory_node(starpu_worker_get_id()) - 1]);

    /* Reset pour prochaine itération */
    if (NT_dynamic_outer - 1 == number_task_out_DARTS)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&global_mutex);
		reset_all_struct();
		need_to_reinit = true;
		iteration_DARTS++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
		
		#ifdef PRINT
		if ((iteration_DARTS == 11 && starpu_get_env_number_default("PRINT_TIME", 0) == 1) || starpu_get_env_number_default("PRINT_TIME", 0) == 2) //PRINT_TIME = 2 pour quand on a 1 seule itération
		{
			gettimeofday(&time_end_createtolasttaskfinished, NULL);
			time_total_createtolasttaskfinished += (time_end_createtolasttaskfinished.tv_sec - time_start_createtolasttaskfinished.tv_sec)*1000000LL + time_end_createtolasttaskfinished.tv_usec - time_start_createtolasttaskfinished.tv_usec;

			int print_N = 0;
			if (app == 0) // Cas M2D
			{
				print_N = sqrt(NT);
			}
			else // Cas M3D
			{
				print_N = sqrt(NT)/2;
			}
			if (threshold == 1)
			{
				FILE *f = fopen("Output_maxime/DARTS_time.txt", "a");
				fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				fclose(f);
			}
			else if (choose_best_data_from == 1)
			{
				FILE *f = fopen("Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt", "a");
				fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				fclose(f);
			}
			else
			{
				FILE *f = fopen("Output_maxime/DARTS_time_no_threshold.txt", "a");
				fprintf(f, "%d	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld	%lld\n", print_N, time_total_selector, time_total_evicted, time_total_belady, time_total_schedule, time_total_choose_best_data, time_total_fill_planned_task_list, time_total_initialisation, time_total_randomize, time_total_pick_random_task, time_total_least_used_data_planned_task, time_total_createtolasttaskfinished);
				fclose(f);
			}
			//~ printf("Nombre d'entrée dans victim selector = %d, nombre de return no victim = %d. Temps passé dans victim_selector = %lld.\n", victim_selector_compteur, victim_selector_return_no_victim, time_total_selector);
			//~ printf("Nombre d'entrée dans Belady = %d. Temps passé dans Belady = %lld.\n", victim_selector_belady, time_total_belady);
			//~ printf("Nombre d'entrée dans victim evicted = %d. Temps passé dans victim_evicted = %lld.\n", victim_evicted_compteur, time_total_evicted);
			printf("Nombre de choix random = %d.\n", number_random_selection);
		}
		#endif
	}
	
	//~ STARPU_PTHREAD_MUTEX_UNLOCK(&global_mutex);
    starpu_sched_component_worker_pre_exec_hook(task, sci);
}

//~ /* Version avec print et visualisation */
//~ struct starpu_sched_policy _starpu_sched_dynamic_data_aware_policy =
//~ {
	//~ .init_sched = initialize_dynamic_data_aware_center_policy,
	//~ .deinit_sched = deinitialize_dynamic_data_aware_center_policy,
	//~ .add_workers = starpu_sched_tree_add_workers,
	//~ .remove_workers = starpu_sched_tree_remove_workers,
	//~ /* .do_schedule = starpu_sched_tree_do_schedule, */
	//~ .push_task = starpu_sched_tree_push_task,
	//~ /* //~ .pop_task = starpu_sched_tree_pop_task, */
	//~ .pop_task = get_data_to_load,
	//~ /* .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook, */
	//~ .pre_exec_hook = get_current_tasks,
	//~ /* .post_exec_hook = starpu_sched_component_worker_post_exec_hook, */
	//~ .post_exec_hook = get_task_done,
	//~ .pop_every_task = NULL,
	//~ .policy_name = "dynamic-data-aware",
	//~ .policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading the data adding the most tasks",
	//~ .worker_type = STARPU_WORKER_LIST,
//~ };

/* Version pour performances */
struct starpu_sched_policy _starpu_sched_dynamic_data_aware_policy =
{
	.init_sched = initialize_dynamic_data_aware_center_policy,
	.deinit_sched = deinitialize_dynamic_data_aware_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = get_task_done, /* Utile pour la stratégie d'éviction */
	.pop_every_task = NULL,
	.policy_name = "dynamic-data-aware",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading the data adding the most tasks",
	.worker_type = STARPU_WORKER_LIST,
};
