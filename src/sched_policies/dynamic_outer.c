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

/* Dynamic Outer scheuling. Pop as much random data not used yet by a GPU
 * as there are different data type.
 * Computes all task using these data and the data already loaded on memory.
 * if no task is available compute a random task not computed yet.
 */

#include <schedulers/HFP.h>
#include <schedulers/dynamic_outer.h>
#include "helper_mct.h"

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
	printf("On GPU %d, the data not used yet are:", i + 1);
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
    
    printf("Planned task for GPU %d:\n", current_gpu);
    for (task = starpu_task_list_begin(&g->planned_task); task != starpu_task_list_end(&g->planned_task); task = starpu_task_list_next(task))
    {
	printf("%p: %p %p\n", task, STARPU_TASK_GET_HANDLE(task, 0), STARPU_TASK_GET_HANDLE(task, 1));
    }
}

void print_pulled_task_one_gpu(struct gpu_pulled_task *g, int current_gpu)
{
    struct pulled_task *p = pulled_task_new();
    
    printf("Pulled task for GPU %d:\n", current_gpu);
    for (p = pulled_task_list_begin(g->ptl); p != pulled_task_list_end(g->ptl); p = pulled_task_list_next(p))
    {
	printf("%p\n", p->pointer_to_pulled_task);
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

/* Pushing the tasks. Each time a new task enter here, we initialize it. */		
static int dynamic_outer_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
    /* If this boolean is true, pull_task will know that new tasks have arrived and
     * thus it will be able to randomize both the task list and the data list not used yet in the GPUs. 
     */
    new_tasks_initialized = true; 
    struct dynamic_outer_sched_data *data = component->data;

    initialize_task_data_gpu_single_task(task);
    
    /* Pushing the task in sched_list. It's this list that will be randomized
     * and put in main_task_list in pull_task.
     */
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
    starpu_task_list_push_front(&data->sched_list, task);
    starpu_push_task_end(task);
    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    component->can_pull(component);
    return 0;
}

/* Initialize for:
 * tasks -> pointer to the data it uses, pointer to the pointer of task list in the data, 
 * pointer to the cell in the main task list (main_task_list).
 * data -> pointer to the tasks using this data.
 * GPUs -> datas not used yet by this GPU.
 */
void initialize_task_data_gpu_single_task(struct starpu_task *task)
{
    int i = 0;
    int j = 0;
    
    /* Adding the data not used yet in the all the GPU(s). */
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
		if (STARPU_TASK_GET_HANDLE(task, j)->sched_data == NULL)
		{
		    gpu_data_not_used_list_push_front(my_planned_task_control->pointer->gpu_data, e);
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
    //~ pt->state = 0;
	
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
}

void randomize_task_list(struct dynamic_outer_sched_data *d)
{
    int random = 0;
    int i = 0;
    for (i = 0; i < NT; i++)
    {
	random = rand()%(NT - i);
	while (random != 0)
	{
	    random--;
	    starpu_task_list_push_back(&d->sched_list, starpu_task_list_pop_front(&d->sched_list));
	}
	starpu_task_list_push_back(&d->main_task_list, starpu_task_list_pop_front(&d->sched_list));
    }
}

/* Randomize the list of data not used yet for all the GPU. */
void randomize_data_not_used_yet()
{
    int i = 0;
    int k = 0;
    int l = 0;
    int random = 0;
    int number_of_data = 0;
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    /* I need this for the %random. */
    number_of_data = gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data);
    
    for (i = 0; i < Ngpu; i++)
    {
	struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
	for (l = 0; l < number_of_data; l++)
	{
	    /* After each time I remove a data I can choose between a smaller number of value for random. */
	    random = rand()%(number_of_data - l);
	    for (k = 0; k < random; k++)
	    {
		gpu_data_not_used_list_push_back(my_planned_task_control->pointer->gpu_data, gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data));
	    }
	    /* I use an external list. */
	    gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data));
	}
	/* Then replace the list with it. */
	my_planned_task_control->pointer->gpu_data = randomized_list;
	my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
}

/* Randomize the list of data not used yet for a single GPU. */
void randomize_data_not_used_yet_single_GPU(struct gpu_planned_task *g)
{
    int j = 0;
    int k = 0;
    int random = 0;
    int number_of_data = 0;
    
    number_of_data = gpu_data_not_used_list_size(g->gpu_data);
    
    struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
    for (j = 0; j < number_of_data; j++)
    {
	/* After each time I remove a data I can choose between a smaller number of value for random. */
	random = rand()%(number_of_data - j);
	for (k = 0; k < random; k++)
	{
	    gpu_data_not_used_list_push_back(g->gpu_data, gpu_data_not_used_list_pop_front(g->gpu_data));
	}
	/* I use an external list. */
	gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(g->gpu_data));
    }
    /* Then replace the list with it. */
    g->gpu_data = randomized_list;
}

/* Get a task to put out of pull_task. In multi GPU it allows me to return a task from the right element in the 
 * linked list without having an other GPU comme and ask a task in pull_task. At least I hope it does so.
 */
struct starpu_task *get_task_to_return_pull_task_dynamic_outer(int current_gpu, struct starpu_task_list *l)
{
     int i = 0;
     
    /* Getting on the right GPU's package.
     * TODO: Can I do this faster with pointer directly to the cell ? */
    my_planned_task_control->pointer = my_planned_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
	my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    
    /* If there are still tasks either in the packages, the main task list or the refused task,
     * I enter here to return a task or start dynamic_outer_scheduling. Else I return NULL.
     */
    if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task) || !starpu_task_list_empty(l) || !starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list))
    {	
	printf("GPU %d is asking for a task.\n", current_gpu);
	struct starpu_task *task = NULL;

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list)) 
	{
	    /* Ici je ne met pas à jour pulled_task car je l'ai déjà fais pour la tâche avant qu'elle ne soit refusé. */
	    task = starpu_task_list_pop_back(&my_planned_task_control->pointer->refused_fifo_list); 
	    printf("Task %d: %p is getting out of pull_task from fifo refused list on GPU %d\n", number_task_out, task, current_gpu);
	    return task;
	}

	/* If the package is not empty I can return the head of the task list. */
	if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task))
	{
	    number_task_out++;
	    task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);
	    
	    /* Fonction qui ajoute la tâche à pulled_task. Elle est aussi dans le else if en dessous. */
	    add_task_to_pulled_task(current_gpu, task);

	    printf("Task %d: %p is getting out of pull_task from GPU %d\n", number_task_out, task, current_gpu);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, current_gpu - 1);
	    }

	    return task;
	}
	/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
	else if (!starpu_task_list_empty(l))
	{
	    number_task_out++;
	    dynamic_outer_scheduling_one_data_popped(l, current_gpu, my_planned_task_control->pointer);
	    task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);
	    add_task_to_pulled_task(current_gpu, task);
	    printf("Task %d, %p is getting out of pull_task from GPU %d\n", number_task_out, task, current_gpu);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, current_gpu - 1);
	    }
	    
	    return task;
	}
    }
    return NULL;
}

/* Pull tasks. When it receives new task it will randomize the task list and the GPU data list.
 * If it has no task it return NULL. Else if a task was refused it return it. Else it return the
 * head of the GPU task list. Else it calls dyanmic_outer_scheuling to fill this package. */
static struct starpu_task *dynamic_outer_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct dynamic_outer_sched_data *data = component->data;
    //~ int i = 0;
    //~ int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
    
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
	printf("Printing GPU's data list and main task list before randomization:\n\n");
	print_data_not_used_yet();
	print_task_list(&data->sched_list, "");
	NT = starpu_task_list_size(&data->sched_list);
	printf("Il y a %d tâches.\n", NT);
	randomize_task_list(data);
	randomize_data_not_used_yet(my_planned_task_control->first);
	new_tasks_initialized = false;
	printf("Printing GPU's data list and main task list after randomization:\n\n");
	print_data_not_used_yet();
	print_task_list(&data->main_task_list, "");
    }
    
    /* Même sans les mutex ca marche, c'est bizare ... */
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
    struct starpu_task *task = get_task_to_return_pull_task_dynamic_outer(starpu_worker_get_memory_node(starpu_worker_get_id()), &data->main_task_list);
    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    return task;
}

void push_data_not_used_yet_random_spot(starpu_data_handle_t h, struct gpu_planned_task *g)
{
    printf("Avant pushing random\n");
    print_data_not_used_yet_one_gpu(g);
    struct gpu_data_not_used *ptr = gpu_data_not_used_new();
    struct gpu_data_not_used *new_element = gpu_data_not_used_new();
    new_element->D = h;
    int random = rand()%gpu_data_not_used_list_size(g->gpu_data);
    int i = 0;

    printf("Random = %d. Je veux push %p.\n", random, h);
    
    ptr = gpu_data_not_used_list_begin(g->gpu_data);
    for (i = 0; i < random; i++)
    {
	ptr = gpu_data_not_used_list_next(ptr);
    }
    gpu_data_not_used_list_insert_before(g->gpu_data, new_element, ptr);
    printf("Après pushing random\n");
    print_data_not_used_yet_one_gpu(g);
}

/* Fill a package's task list following dynamic_outer algorithm. It pop only one data, the one that achieve the most tasks. */
void dynamic_outer_scheduling_one_data_popped(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g)
{
    int i = 0;
    int j = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    int number_of_task_max = 0;
    int task_available_max = 0;
    int temp_number_of_task_max = 0;
    starpu_data_handle_t handle_popped = NULL;
    struct task_using_data_list *tudl = task_using_data_list_new();
    
    /* Ce cas arrive avec le cas ou je gère pas les evictions. Car quand je ne gère pas les évictions je ne remet pas les données évincées dans la liste des données
     * à faire. */
    if (gpu_data_not_used_list_empty(g->gpu_data))
    {
	goto random;
    }
    
    /* To know if all the data needed for a task are loaded in memory. */
    bool data_available = true; 
    
    for (e = gpu_data_not_used_list_begin(g->gpu_data); e != gpu_data_not_used_list_end(g->gpu_data); e = gpu_data_not_used_list_next(e))
    {
	temp_number_of_task_max = 0;
	
	printf("If e->D is %p.\n", e->D);
	for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
	{
	    /* I put it at false if at least one data is missing. */
	    data_available = true; 
	    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
	    {
		/* I test if the data is on memory */ 
		if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != e->D)
		{
		    printf("Testing if %p is on node.\n", STARPU_TASK_GET_HANDLE(t->pointer_to_T, j)); 
		    if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, j), current_gpu))
		    {
			data_available = false;
			break;
		    }
		}
	    }
	    if (data_available == true)
	    {
		printf("Data available for task %p.\n", t->pointer_to_T);
		temp_number_of_task_max++;
	    }
	}
	
	if (temp_number_of_task_max > number_of_task_max)
	{
	    number_of_task_max = temp_number_of_task_max;
	    task_available_max = task_using_data_list_size(e->D->sched_data);
	    handle_popped = e->D;
	}
	/* Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. */
	else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
	{
	    tudl = e->D->sched_data;
	    /* TODO : la en 3D on voudra check les data qui peuvent permettre de faire des tâches avec 1 data de load. Puius pour rendre ca général avec 2 data de plus, 3 de plus etc... Du coup rendre ca géénral et déjà tester que en 2d ca donne les mêmes résultats exactement, car normalement ca devrait. */
	    if (task_using_data_list_size(tudl) > task_available_max)
	    {
		printf("Egalité mais plus de data available.\n");
		task_available_max = task_using_data_list_size(tudl);
		handle_popped = e->D;
	    }
	}
    }
    printf("number of task max: %d.\n", number_of_task_max);
    if (number_of_task_max == 0)
    {
	goto random;
    }
    else /* I erase the data. */
    {
	e = gpu_data_not_used_list_begin(g->gpu_data);
	while (e->D != handle_popped)
	{
	   e = gpu_data_not_used_list_next(e);
        } 
	gpu_data_not_used_list_erase(g->gpu_data, e);
    }
    printf("The data adding the most is %p.\n", handle_popped);
	
    /* Adding the task to the list. TODO : this is a copy paste of the code above to test the available tasks.
     * TODO : cette partie ne marchera que en 2D ? Je sais pas à tester */
    for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
    {
	data_available = true; 
	for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T); j++)
	{		    		
	    if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, j) != handle_popped)
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
	    printf("Pushing %p in planned task of GPU %d\n", t->pointer_to_T, current_gpu);
	    erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
	    starpu_task_list_push_back(&g->planned_task, t->pointer_to_T);
	    //~ print_planned_task_one_gpu(g, current_gpu);
	}
    }
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&g->planned_task)) 
    {
	random: ;
	struct starpu_task *task = starpu_task_list_pop_front(main_task_list);
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
	printf("No task were possible with the popped handles. Returning head of the randomized main task list: %p.\n", task);
	erase_task_and_data_pointer(task, main_task_list);
	starpu_task_list_push_back(&g->planned_task, task);
    }
}

/* Pour savoir si la donnée évincé est bien celle que l'on avais prévu.
 * Si ce n'est pas le cas ou si ca vaut NULL alors cela signifie qu'une donnée non prévu a 
 * été évincé. Il faut donc mettre à jour les listes dans les tâches et les données en conséquence.
 * Cependant si on est sur la fin de l'éxécution et que les éviction sont juste la pour vider la mémoire ce n'est pas
 * nécessaire. En réalité pour le moment je ne me rend pas compte qu'on est a la fin de l'exec. 
 * TODO : se rendre compte qu'on est a la fin et arreter de mettre à jour les listes du coup.
 * Du coup je ne sais pas si c'est utile, à vérifier.
 */
 /* starpu_data_handle_t planned_eviction; */

void dynamic_outer_victim_evicted(int success, starpu_data_handle_t victim, void *component)
{
     /* If a data was not truly evicted I put it back in the list. */
    if (success == 0)
    {
	int i = 0;
		
	my_planned_task_control->pointer = my_planned_task_control->first;
	for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
	{
	    my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	}
	
	/* Version 1 seule donnée. A voir si ca marche en multi GPU */
	my_planned_task_control->pointer->data_to_evict_next = victim;
    }
    else
    {
	return;
    }
}

/* TODO: return NULL ou ne rien faire si la dernière tâche est sorti du post exec hook ? De même pour la mise à jour des listes à chaque eviction de donnée.
 * TODO je rentre bcp trop dans cette fonction on perds du temps car le timing avance lui. */
starpu_data_handle_t dynamic_outer_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{        
    int i = 0;
    int j = 0;
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
	printf("Return data %p that was refused.\n", my_planned_task_control->pointer->data_to_evict_next);
	starpu_data_handle_t temp_handle = my_planned_task_control->pointer->data_to_evict_next;
	my_planned_task_control->pointer->data_to_evict_next = NULL;
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
    struct pulled_task *p = pulled_task_new();
    int nb_task_in_pulled_task[nb_data_on_node];
    for (i = 0; i < nb_data_on_node; i++)
    {
	nb_task_in_pulled_task[i] = 0;
    }
    
    /* Je cherche le nombre de tâche dans le pulled_task que peut faire chaque données */
    for (i = 0; i < nb_data_on_node; i++)
    {
	if (starpu_data_can_evict(data_on_node[i], node, is_prefetch))
	{
	    for (p = pulled_task_list_begin(my_pulled_task_control->pointer->ptl); p!= pulled_task_list_end(my_pulled_task_control->pointer->ptl); p = pulled_task_list_next(p))
	    {
		for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
		{
		    if (STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j) == data_on_node[i])
		    {
			nb_task_in_pulled_task[i]++;
			break;
		    }
		}
	    }
	    printf("Nb task for data %p: %d.\n", data_on_node[i], nb_task_in_pulled_task[i]);
	    if (nb_task_in_pulled_task[i] < min_number_task_in_pulled_task)
	    {
		min_number_task_in_pulled_task = nb_task_in_pulled_task[i];
	    }
	}
	else
	{
	    /* - 1 si j'ai pas le droit d'évincer cette donnée */
	    nb_task_in_pulled_task[i] = -1;
	}
    }
    printf("Min number of task in pulled task is %d.\n", min_number_task_in_pulled_task);
    
    /* Si il vaut 0 je fais belady sur les données utilisé par les tâches de pulled_task, 
     * sinon je choisis min(NT/W(D(T)) + W(Di) * W(Di)).
     * Si il vaut -1 c'est que je n'avais aucune donnée à renvoyer car aucune n'est évincable.
     */
    if (min_number_task_in_pulled_task == INT_MAX)
    {
	printf("Return STARPU_DATA_NO_VICTIM in victim selector.\n");
	return STARPU_DATA_NO_VICTIM;
    }
    else if (min_number_task_in_pulled_task == 0)
    {
	/* Au moins 1 donnée ne sert pas dans pulled_task */
	returned_handle = min_weight_average_on_planned_task(data_on_node, nb_data_on_node, node, is_prefetch, my_planned_task_control->pointer, nb_task_in_pulled_task);
    }
    else
    {
	/* Au moins 1 donnée sert dans pulled_task */
	printf("#warning min number of task done by data on node is != 0.\n");
	
	/* Si c'est un prefetch qui demande une eviction de ce qui est utile pour les tâches de pulled task je renvoie NO VICTIM si >= à STARPU_TASK_PREFETCH */
	if (is_prefetch >= 1)
	{
	    printf("A prefetch is asking for an eviction.\n");
	    return STARPU_DATA_NO_VICTIM;
	}

	returned_handle = belady_on_pulled_task(data_on_node, nb_data_on_node, node, is_prefetch, my_pulled_task_control->pointer);
    }
    
    //~ returned_handle = get_handle_to_evict(data_on_node, nb_data_on_node, node, is_prefetch, current_gpu);
    /* Ca devrait pas arriver a enleevr et a tester */
    if (returned_handle == NULL)
    {
	printf("#Warning: the returned handle in dynamic_outer_victim_selector was NULL.\n");
	return STARPU_DATA_NO_VICTIM; 
    }
    
    struct starpu_task *task = NULL;
    struct starpu_sched_component *temp_component = component;
    struct dynamic_outer_sched_data *data = temp_component->data;
    /* Enlever de la liste de tache a faire celles qui utilisais cette donnée. Et donc ajouter cette donnée aux données
     * à pop ainsi qu'ajouter la tache dans les données. Also add it to the main task list. */
    //Suppression de la liste de planned task les tâches utilisant la données
    if (min_number_task_in_pulled_task == 0)
    {
	for (task = starpu_task_list_begin(&my_planned_task_control->pointer->planned_task); task != starpu_task_list_end(&my_planned_task_control->pointer->planned_task); task = starpu_task_list_next(task))
	{
	    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	    {
		if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
		{
		    //Suppression de la liste de tâches à faire 
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
			    
			//Ajout a la liste de tâches principales ces mêmes tâches
			starpu_task_list_push_back(&data->main_task_list, task);
		    break;
		}
	    }
	}
	    
    }
	
    /*Placing in a random spot of the data list to use the evicted handle */
    push_data_not_used_yet_random_spot(returned_handle, my_planned_task_control->pointer);
	
    printf("Return %p in victim selector.\n", returned_handle);
    return returned_handle;
}

starpu_data_handle_t belady_on_pulled_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_pulled_task *g)
{
    int i = 0;
    int j = 0;
    int next_use = 0;
    int max_next_use = -1;
    struct pulled_task *p = pulled_task_new();
    starpu_data_handle_t returned_handle = STARPU_DATA_NO_VICTIM;
    
    //~ print_pulled_task_one_gpu(g, node);
    
    for (i = 0; i < nb_data_on_node; i++)
    {
	if (starpu_data_can_evict(data_tab[i], node, is_prefetch))
	{
	    next_use = 0;
	    for (p = pulled_task_list_begin(g->ptl); p != pulled_task_list_end(g->ptl); p = pulled_task_list_next(p))
	    {
		for (j = 0; j < STARPU_TASK_GET_NBUFFERS(p->pointer_to_pulled_task); j++)
		{
		    next_use++;
		    if (STARPU_TASK_GET_HANDLE(p->pointer_to_pulled_task, j) == data_tab[i])
		    {
			printf("Next use of %p is %d.\n", data_tab[i], next_use);
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
    printf("Return in belady %p.\n", returned_handle);
    return returned_handle;
}

starpu_data_handle_t min_weight_average_on_planned_task(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, struct gpu_planned_task *g, int *nb_task_in_pulled_task)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int next_use = 0;
    int data_count = 0;
    int max_next_use = 0;
    float min_weight_average = FLT_MAX;
    float weight_average = 0;
    float weight_missing_data = 0;
    int NT_Di = 0;
    struct starpu_task *task = NULL;
    starpu_data_handle_t returned_handle = STARPU_DATA_NO_VICTIM;
    
    print_planned_task_one_gpu(g, node);
    print_data_on_node(data_tab, nb_data_on_node);
    
    /* To avoid duplicate data */
    struct data_weighted_list *dwl = data_weighted_list_new();
    struct data_weighted *dw = data_weighted_new();
    
    for (i = 0; i < nb_data_on_node; i++)
    {
	if (nb_task_in_pulled_task[i] == 0)
	{
	    NT_Di = 0;
	    weight_missing_data = 0;
	    next_use = 0;
	    data_count = 0;
	    for (task = starpu_task_list_begin(&g->planned_task); task != starpu_task_list_end(&g->planned_task); task = starpu_task_list_next(task))
	    {
		for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
		{
		    /* Au cas où il y a égalité et que je fais Belady */
		    data_count++;
		    
		    printf("Comparing %p and %p.\n", STARPU_TASK_GET_HANDLE(task, j), data_tab[i]);
		    if (STARPU_TASK_GET_HANDLE(task, j) == data_tab[i])
		    {
			/* J'arrête de compter pour Belady */
			next_use = data_count;
			
			/* Je suis sur une tâche qui utilise Di */
			NT_Di++;
			
			/* J'ajoute ses données manquantes sauf Di au poids */
			for (k = 0; k < STARPU_TASK_GET_NBUFFERS(task); k++)
			{
			    printf("Test if is on node %p.\n", STARPU_TASK_GET_HANDLE(task, k));
			    if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, k), node))
			    {
				printf("Not on node.\n");
				/* Je ne dois ajouter le poids de cette donnée que si elle n'a pas déjà été compté. */
				if (data_weighted_list_empty(dwl))
				{
				    printf("Data counted list empty, adding the weight.\n");
				    struct data_weighted *new = data_weighted_new();
				    new->pointer_to_data_weighted = STARPU_TASK_GET_HANDLE(task, k);
				    data_weighted_list_push_back(dwl, new);
				    weight_missing_data += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, k));
				}
				else
				{
				    for (dw = data_weighted_list_begin(dwl); dw != data_weighted_list_end(dwl); dw = data_weighted_list_next(dw))
				    {
					if (STARPU_TASK_GET_HANDLE(task, k) == dw->pointer_to_data_weighted)
					{
					    printf("Déjà compté.\n");
					    break;
					}
					struct data_weighted *new = data_weighted_new();
					new->pointer_to_data_weighted = STARPU_TASK_GET_HANDLE(task, k);
					data_weighted_list_push_back(dwl, new);
					printf("%p is not on node and not counted yet, adding it to missing data weight.\n", STARPU_TASK_GET_HANDLE(task, k));
					weight_missing_data += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, k));
				    }
				}
			    }
			}
			break;
		    }
		}
	    }
	    weight_average = (NT_Di/(weight_missing_data + starpu_data_get_size(data_tab[i])))*starpu_data_get_size(data_tab[i]);
	    printf("Weight average of %p is %f with %d task and %f missing data.\n", data_tab[i], weight_average, NT_Di, weight_missing_data);
	    if (min_weight_average > weight_average)
	    {
		max_next_use = next_use; /* Au cas ou Belady */
		min_weight_average = weight_average;
		returned_handle = data_tab[i];
	    }
	    else if (min_weight_average == weight_average)
	    {
		printf("Egalité entre %p et %p, les next use sont %d et %d.\n", returned_handle, data_tab[i], max_next_use, next_use);
		/* Je fais Belady sur planned_task */
		if (next_use > max_next_use)
		{
		    max_next_use = next_use;
		    returned_handle = data_tab[i];
		    printf("Pour le moment returned handle devient %p.\n", data_tab[i]);
		}
	    }
	}
    }
    
    printf("Return in min_weight_average %p.\n", returned_handle);
    return returned_handle;
}

/* Erase a task from the main task list.
 * Also erase pointer in the data.
 * There was a problem here. I evict a task in a data even tho it's not on it!
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

static int dynamic_outer_can_push(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct dynamic_outer_sched_data *data = component->data;
    int didwork = 0;
    struct starpu_task *task;
    task = starpu_sched_component_pump_to(component, to, &didwork);
    if (task)
    {	    
	    /* If a task is refused I push it in the refused fifo list of the appropriate GPU's package.
	     * This list is looked at first when a GPU is asking for a task so we don't break the planned order. */
	    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	    my_planned_task_control->pointer = my_planned_task_control->first;
	    for (int i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++) 
	    {
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	    }
	    starpu_task_list_push_back(&my_planned_task_control->pointer->refused_fifo_list, task);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    }
    
    /* There is room now */
    return didwork || starpu_sched_component_can_push(component, to);
}

static int dynamic_outer_can_pull(struct starpu_sched_component *component)
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
    my_pulled_task_control->pointer = my_pulled_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
	my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    }
    
    struct pulled_task *p = pulled_task_new();
    p->pointer_to_pulled_task = task;
    pulled_task_list_push_back(my_pulled_task_control->pointer->ptl, p);
    
    //~ print_pulled_task_one_gpu(my_pulled_task_control->pointer, current_gpu);
}

struct starpu_sched_component *starpu_sched_component_dynamic_outer_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_outer");
	srandom(starpu_get_env_number_default("SEED", 0));
	int i = 0;
	
	/* Initialization of global variables. */
	Ngpu = get_number_GPU();
	NT = 0;
	new_tasks_initialized = false;
	index_current_popped_task = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_all_gpu = 0;
	index_current_popped_task_all_gpu_prefetch = 0;
	gpu_memory_initialized = false;
	number_task_out = -1;
	
	printf("Ngpu = %d\n", Ngpu); 
		
	/* Initialization of structures. */
	struct dynamic_outer_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->main_task_list);
	
	/* Initialisation des structs de liste de tâches */
	gpu_planned_task_initialisation();
	for (i = 0; i < Ngpu - 1; i++)
	{
	    gpu_planned_task_insertion();
	}
	my_planned_task_control->first = my_planned_task_control->pointer;
	
	gpu_pulled_task_initialisation();
	for (i = 0; i < Ngpu - 1; i++)
	{
	    gpu_pulled_task_insertion();
	}
	my_pulled_task_control->first = my_pulled_task_control->pointer;
	
	component->data = data;
	/* component->do_schedule = dynamic_outer_do_schedule; */
	component->push_task = dynamic_outer_push_task;
	component->pull_task = dynamic_outer_pull_task;
	component->can_push = dynamic_outer_can_push;
	component->can_pull = dynamic_outer_can_pull;
	
	/* TODO: Aussi faire cela pour HFP. */
	if (starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_OUTER", 0) == 1) 
	{ 
	    starpu_data_register_victim_selector(dynamic_outer_victim_selector, dynamic_outer_victim_evicted, component); 
	}
	
	return component;
}

static void initialize_dynamic_outer_center_policy(unsigned sched_ctx_id)
{
    
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_dynamic_outer_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_dynamic_outer_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

/* Get the task that was last executed. Used to update the task list of pulled task	 */
void get_task_done(struct starpu_task *task, unsigned sci)
{
    if (starpu_worker_get_memory_node(starpu_worker_get_id()) == 4)
    { printf("Dans le post exec hook avec la tâche %p.\n", task); }
    int i = 0;
    
    /* Je me place sur la liste correspondant au bon gpu. */
    my_pulled_task_control->pointer = my_pulled_task_control->first;
    for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
    {
	printf("next\n");
	my_pulled_task_control->pointer = my_pulled_task_control->pointer->next;
    }
        
    /* J'efface la tâche dans la liste de tâches */
    pulled_task_list_pop_front(my_pulled_task_control->pointer->ptl);
    
    starpu_sched_component_worker_post_exec_hook(task, sci);
}

struct starpu_sched_policy _starpu_sched_dynamic_outer_policy =
{
	.init_sched = initialize_dynamic_outer_center_policy,
	.deinit_sched = deinitialize_dynamic_outer_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	/* .do_schedule = starpu_sched_tree_do_schedule, */
	.push_task = starpu_sched_tree_push_task,
	/* .pop_task = starpu_sched_tree_pop_task, */
	.pop_task = get_data_to_load,
	/* .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook, */
	.pre_exec_hook = get_current_tasks,
	/* .post_exec_hook = starpu_sched_component_worker_post_exec_hook, */
	.post_exec_hook = get_task_done,
	.pop_every_task = NULL,
	.policy_name = "dynamic-outer",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading 2 random data",
	.worker_type = STARPU_WORKER_LIST,
};
