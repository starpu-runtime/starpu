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
    int j = 0;
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    for (i = 0; i < Ngpu; i++)
    {
	printf("On GPU %d, the data not used yet are:", i + 1);
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    printf("\nFor the data type %d:", j);
	    for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(my_planned_task_control->pointer->gpu_data[j]); e != gpu_data_not_used_list_end(my_planned_task_control->pointer->gpu_data[j]); e = gpu_data_not_used_list_next(e))
	    {
		printf(" %p", e->D);
	    }
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
	printf("%p\n", task);
    }
}

void print_data_not_used_yet_one_gpu(struct gpu_planned_task *g)
{
    int j = 0;    
    for (j = 0; j < Ndifferent_data_type; j++)
    {
	printf("\nFor the data type n°%d:", j);
	for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(g->gpu_data[j]); e != gpu_data_not_used_list_end(g->gpu_data[j]); e = gpu_data_not_used_list_next(e))
	{
	    printf(" %p", e->D);
	}
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
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    struct gpu_data_not_used *e = gpu_data_not_used_new();
	    e->D = STARPU_TASK_GET_HANDLE(task, j);
	    
	    /* To get the data type of each data. It's in user_data. 
	     * sched_data in the handles is used for the task using this data. */
	    struct datatype *d = malloc(sizeof(*d));
	    d->type = j; 
	    STARPU_TASK_GET_HANDLE(task, j)->user_data = d;
	    
	    /* If the void * of struct paquet is empty I initialize it. */ 
	    if (my_planned_task_control->pointer->gpu_data[j] == NULL)
	    {
		struct gpu_data_not_used_list *gd = gpu_data_not_used_list_new();
		gpu_data_not_used_list_push_front(gd, e);
		my_planned_task_control->pointer->gpu_data[j] = gd; 
	    }
	    else
	    {
		if (STARPU_TASK_GET_HANDLE(task, j)->sched_data == NULL)
		{
		    gpu_data_not_used_list_push_front(my_planned_task_control->pointer->gpu_data[j], e);
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
    int j = 0;
    int k = 0;
    int l = 0;
    int random = 0;
    int number_of_data[Ndifferent_data_type];
    my_planned_task_control->pointer = my_planned_task_control->first;
    
    /* I need this for the %random. */
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	number_of_data[i] = gpu_data_not_used_list_size(my_planned_task_control->pointer->gpu_data[i]);
    }
    for (i = 0; i < Ngpu; i++)
    {
	my_planned_task_control->pointer->number_handle_to_pop = 0;
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
	    for (l = 0; l < number_of_data[j]; l++)
	    {
		/* After each time I remove a data I can choose between a smaller number of value for random. */
		random = rand()%(number_of_data[j]- l);
		for (k = 0; k < random; k++)
		{
		    gpu_data_not_used_list_push_back(my_planned_task_control->pointer->gpu_data[j], gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data[j]));
		}
		/* I use an external list. */
		gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(my_planned_task_control->pointer->gpu_data[j]));
	    }
	    /* Then replace the list with it. */
	    my_planned_task_control->pointer->gpu_data[j] = randomized_list;
	    my_planned_task_control->pointer->number_handle_to_pop += number_of_data[j];
	}
	my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
}

/* Randomize the list of data not used yet for a single GPU. */
void randomize_data_not_used_yet_single_GPU(struct gpu_planned_task *g)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int random = 0;
    int number_of_data[Ndifferent_data_type];
    g->number_handle_to_pop = 0;
    
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	number_of_data[i] = gpu_data_not_used_list_size(g->gpu_data[i]);
    }
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
	for (j = 0; j < number_of_data[i]; j++)
	{
	    /* After each time I remove a data I can choose between a smaller number of value for random. */
	    random = rand()%(number_of_data[i]- j);
	    for (k = 0; k < random; k++)
	    {
		gpu_data_not_used_list_push_back(g->gpu_data[i], gpu_data_not_used_list_pop_front(g->gpu_data[i]));
	    }
	    /* I use an external list. */
	    gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(g->gpu_data[i]));
	}
	/* Then replace the list with it. */
	g->gpu_data[i] = randomized_list;
	g->number_handle_to_pop += number_of_data[i];
    }
}

/* Pull tasks. When it receives new task it will randomize the task list and the GPU data list.
 * If it has no task it return NULL. Else if a task was refused it return it. Else it return the
 * head of the GPU task list. Else it calls dyanmic_outer_scheuling to fill this package. */
static struct starpu_task *dynamic_outer_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct dynamic_outer_sched_data *data = component->data;
    int i = 0;
    int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
    
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
	    
    /* Getting on the right GPU's package.
     * TODO: Can I do this faster with pointer directly to the cell ?
     */
    my_planned_task_control->pointer =  my_planned_task_control->first;
    for (i = 1; i < current_gpu; i++)
    {
	my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    }
    
    /* If there are still tasks either in the packages, the main task list or the refused task,
     * I enter here to return a task or start dynamic_outer_scheduling. Else I return NULL.
     */
    if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task) || !starpu_task_list_empty(&data->main_task_list) || !starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list))
    {	
	printf("GPU %d is asking for a task.\n", current_gpu);
	struct starpu_task *task = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&my_planned_task_control->pointer->refused_fifo_list)) 
	{
	    task = starpu_task_list_pop_back(&my_planned_task_control->pointer->refused_fifo_list); 
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    printf("Task %d: %p is getting out of pull_task from fifo refused list on GPU %d\n", number_task_out, task, current_gpu);
	    return task;
	}

	/* If the package is not empty I can return the head of the task list. */
	if (!starpu_task_list_empty(&my_planned_task_control->pointer->planned_task))
	{
	    number_task_out++;
	    task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);
	    printf("Task %d: %p is getting out of pull_task from GPU %d\n", number_task_out, task, current_gpu);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, current_gpu - 1);
	    }
	    
	    return task;
	}
	/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
	else if (!starpu_task_list_empty(&data->main_task_list))
	{
	    /* Je me remet à nouveau sur le bon gpu, car si entre temps pull_task est rappellé ca me remet au début de la liste chainbée -__- */ 
	    my_planned_task_control->pointer = my_planned_task_control->first;
	    for (i = 1; i < current_gpu; i++)
	    {
		my_planned_task_control->pointer = my_planned_task_control->pointer->next;
	    }
		    
	    number_task_out++;
	    if (starpu_get_env_number_default("DATA_POP_POLICY", 0) == 0)
	    {
		dynamic_outer_scheduling(&data->main_task_list, current_gpu, my_planned_task_control->pointer);
	    }
	    else
	    {
		dynamic_outer_scheduling_one_data_popped(&data->main_task_list, current_gpu, my_planned_task_control->pointer);
	    }
	    task = starpu_task_list_pop_front(&my_planned_task_control->pointer->planned_task);
	    printf("Task %d, %p is getting out of pull_task from GPU %d\n", number_task_out, task, current_gpu);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, current_gpu - 1);
	    }
	    
	    return task;
	}
	/* Else I will return NULL. But I still need to unlock the mutex. */
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    }
    
    return NULL;
}

void push_back_data_not_used_yet(starpu_data_handle_t h, struct gpu_planned_task *g, int data_type)
{
    struct gpu_data_not_used *e = gpu_data_not_used_new();
    e->D = h;
    gpu_data_not_used_list_push_back(g->gpu_data[data_type], e);
}

/* Fill a package's task list following dynamic_outer algorithm. It pop only one data, the one that achieve the most tasks. */
void dynamic_outer_scheduling_one_data_popped(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g)
{
    int i = 0;
    int j = 0;
    int next_handle = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    int number_of_task_max = 0;
    int task_available_max = 0;
    int temp_number_of_task_max = 0;
    starpu_data_handle_t handle_popped = NULL;
    struct task_using_data_list *tudl = task_using_data_list_new();
    
    if (gpu_data_not_used_list_empty(g->gpu_data[g->data_type_to_pop]))
    {
	goto random;
    }
    
    /* To know if all the data needed for a task are loaded in memory. */
    bool data_available = true; 
    
    if (starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_OUTER", 0) == 1)
    {
	/* If the number of handle popped is equal to the number of original handle it
	 * means that we are on the set of data evicted. So we want to reshuffle it. 
	 * TODO : it doesn't work in multi gpu because package ae stealing task to one another!!*/
	 g->number_handle_to_pop--;
	 if (g->number_handle_to_pop == 0)
	 {
	     printf("Re-shuffle.\n");
	     print_data_not_used_yet_one_gpu(g);
	     randomize_data_not_used_yet_single_GPU(g);
	     print_data_not_used_yet_one_gpu(g);
	 }
    }
    for (e = gpu_data_not_used_list_begin(g->gpu_data[g->data_type_to_pop]); e != gpu_data_not_used_list_end(g->gpu_data[g->data_type_to_pop]); e = gpu_data_not_used_list_next(e))
    {
	temp_number_of_task_max = 0;
	
	for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
	{
	    /* I put it at false if at least one data is missing. */
	    data_available = true; 
	    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T) - 1; j++)
	    {
		/* I use %nb_data_for_a_task because I don't want to check the current data type I'm on.*/
		next_handle = (g->data_type_to_pop + 1 + j)%STARPU_TASK_GET_NBUFFERS(t->pointer_to_T);
		    /* I test if the data is on memory including prefetch.
		     * TODO: Will we need one day to test without the prefetch ?
		     */ 
		    if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle) != e->D)
		    {
			if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
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
	    handle_popped = e->D;
	}
	/* Si il y a égalité je pop celle qui peut faire le plus de tâches globalement. */
	else if (temp_number_of_task_max == number_of_task_max && number_of_task_max != 0)
	{
	    tudl = e->D->sched_data;
	    if (task_using_data_list_size(tudl) > task_available_max)
	    {
		printf("Egalité mais plus de data available.\n");
		task_available_max = task_using_data_list_size(tudl);
		handle_popped = e->D;
	    }
	}
    }
    if (number_of_task_max == 0)
    {
	goto random;
    }
    else /* I erase the data. */
    {
	e = gpu_data_not_used_list_begin(g->gpu_data[g->data_type_to_pop]);
	while (e->D != handle_popped)
	{
	   e = gpu_data_not_used_list_next(e);
        } 
	gpu_data_not_used_list_erase(g->gpu_data[g->data_type_to_pop], e);
    }
    
    /* Adding the task to the list. TODO : this is a copy paste of the code above to test the available tasks. */
    for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
    {
	data_available = true; 
	for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T) - 1; j++)
	{
	    next_handle = (g->data_type_to_pop + 1 + j)%STARPU_TASK_GET_NBUFFERS(t->pointer_to_T);
		    		
	    if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle) != handle_popped)
	    {
		if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
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
	    print_planned_task_one_gpu(g, current_gpu);
	}
    }
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&g->planned_task)) 
    {
	random: ;
	struct starpu_task *task = starpu_task_list_pop_front(main_task_list);
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
	    if (!gpu_data_not_used_list_empty(g->gpu_data[i]))
	    {
		for (e = gpu_data_not_used_list_begin(g->gpu_data[i]); e != gpu_data_not_used_list_end(g->gpu_data[i]); e = gpu_data_not_used_list_next(e))
		{
		    if(e->D == STARPU_TASK_GET_HANDLE(task, i))
		    {
			gpu_data_not_used_list_erase(g->gpu_data[i], e);
		    }
		}
	    }
	}
	printf("No task were possible with the popped handles. Returning head of the randomized main task list: %p.\n", task);
	erase_task_and_data_pointer(task, main_task_list);
	starpu_task_list_push_back(&g->planned_task, task);
    }
    
    /* On veut pop un autre type de donnée la prochaine fois. */
    g->data_type_to_pop = (g->data_type_to_pop + 1)%Ndifferent_data_type;
}

/* Fill a package task list following dynamic_outer algorithm. */
void dynamic_outer_scheduling(struct starpu_task_list *main_task_list, int current_gpu, struct gpu_planned_task *g)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int pushed_task = 0;
    int next_handle = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    void *task_tab[Ndifferent_data_type];
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	struct starpu_task_list *tl = starpu_task_list_new();
	starpu_task_list_init(tl);
	task_tab[i] = tl;
    }

    /* TODO: Here if you have a random graph, you need to know exactly what data need the tasks. We can also always 
     * pop a N number of data and check each time if we can do a task with what's in memory.
     */
    /* Data popped that we will use. */
    starpu_data_handle_t *handle_popped = malloc(Ndifferent_data_type*sizeof(STARPU_TASK_GET_HANDLE(starpu_task_list_begin(main_task_list), 0)));
    
    /* To know if all the data needed for a task are loaded in memory. */
    bool data_available = true; 
    /* To know if it's the task using the Ndifferent_data_type data popped. */
    bool handle_popped_task = true;

    for (i = 0; i < Ndifferent_data_type; i++)
    {
	/* If at least one data type is empty in the GPU I return a random task.
	 * TODO: This is useless if we add correctly data at the back of the list when evicting, so to remove.
	 */
	if (gpu_data_not_used_list_empty(g->gpu_data[i]))
	{
	    goto return_random_task;
	}
	/* Else I can pop a data. */
	e = gpu_data_not_used_list_pop_front(g->gpu_data[i]);
	handle_popped[i] = e->D;
    }
    
    /* TODO : a enlever ici et plus bas car mtn je check dans le victim selector.
     * TODO : vérifier les shuffle a la main et eneever ce qu'il faut. */
    starpu_data_handle_t *evicted_handles = malloc(Ndifferent_data_type*sizeof(STARPU_TASK_GET_HANDLE(starpu_task_list_begin(main_task_list), 0)));
    if (starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_OUTER", 0) == 1) 
    {
	/* If the number of handle popped is equal to the number of original handle it
	 * means that we are on the set of data evicted. So we want to reshuffle it. */
	 g->number_handle_to_pop--;
	 if (g->number_handle_to_pop == 0)
	 {
	     printf("Re-shuffle\n");
	     randomize_data_not_used_yet_single_GPU(g);
	 }
     }
	
    /* Here, I need to find the task I can do with the data already in memory + the new data A and B.
     * It can also be the task using A and B.
     */
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	for (t = task_using_data_list_begin(handle_popped[i]->sched_data); t != task_using_data_list_end(handle_popped[i]->sched_data); t = task_using_data_list_next(t))
	{
	    /* I put it at false if at least one data is missing. */
	    data_available = true; 
	    handle_popped_task = true;
	    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T) - 1; j++)
	    {
		/* I use %nb_data_for_a_task because I don't want to check the current data type I'm on.*/
		next_handle = (i + 1 + j)%STARPU_TASK_GET_NBUFFERS(t->pointer_to_T);
		    		
		/* I test if the data is on memory including prefetch.
		 * TODO: Will we need one day to test without the prefetch ?
		 */ 
		if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle) != handle_popped[next_handle])
		{
		    handle_popped_task = false;
		    /* I also test if it's not an evicted handle. TODO: not sure it works well. */
		    for (k = 0; k < Ndifferent_data_type; k++)
		    {
			if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle) == evicted_handles[k])
			{
			    data_available = false;
			    break;
			}
		    }
		    if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
		    {
			data_available = false;
			break;
		    }
		}
	    }
	    if (data_available == true)
	    {
		/* Deleting the task from the task list of data A, B (and C) and from the main task list. */
		erase_task_and_data_pointer(t->pointer_to_T, main_task_list);
		
		/* Pushing on top the task using all popped handles. */
		if (handle_popped_task == true)
		{
		    starpu_task_list_push_front(&g->planned_task, t->pointer_to_T);
		}
		else
		{
		    starpu_task_list_push_back(task_tab[i], t->pointer_to_T);
		    pushed_task++;
		}
	    }
	}
    }
        
    /* Pushing back interlacing all different data types. */
    while (pushed_task > 0)
    {
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    if (!starpu_task_list_empty(task_tab[j]))
	    {
		starpu_task_list_push_back(&g->planned_task, starpu_task_list_pop_front(task_tab[j]));
		pushed_task--;
	    }
	}
    }
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&g->planned_task)) 
    {
	return_random_task: ;
	struct starpu_task *task = starpu_task_list_pop_front(main_task_list);
	printf("No task were possible with the popped handles. Returning head of the randomized main task list: %p.\n", task);
	erase_task_and_data_pointer(task, main_task_list);
	starpu_task_list_push_back(&g->planned_task, task);
    }
    free(handle_popped);
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

/* Return the handle that can do the least tasks that already have all
 * it data on memory. If there is a draw or if there are no task in the task list, return the 
 * data that has the least remaining task (even if their data are not loaded on memory.
 */
starpu_data_handle_t get_handle_least_tasks(starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch, int current_gpu)
{
    //~ printf("On GPU %d in get_handle_least_task.\n", current_gpu);
    
    //~ starpu_data_handle_t returned_handle = NULL;
    //~ int i = 0;
    //~ int min = 0;
    //~ struct planned_task *pt = planned_task_new();
    
    //~ /* Se placer su la liste corespondant au gpu actuel */
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
    //~ {
	//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    //~ }
    //~ if (planned_task_list_empty(my_planned_task_control->pointer->ptpt))
    //~ {
	//~ /* Je cherche la donnée qui permet de faire le moins de tâches globalement */
	//~ min = INT_MAX;
	//~ for (i = 0; i < nb_data_on_node; i++)
	//~ {
	    //~ if (task_using_data_list_size(tudl) < min && starpu_data_can_evict(data_tab[i], node, is_prefetch))
	    //~ {
		//~ min = task_using_data_list_size(tudl);
		//~ returned_handle = data_tab[i];
	    //~ }
	//~ }
	//~ return returned_handle;
    //~ }
    //~ else
    //~ {
	//~ int j = 0;
	//~ struct starpu_task *task = NULL;
	//~ int nb_task_done_by_data[nb_data_on_node];
	//~ for (i = 0; i < nb_data_on_node; i++) { nb_task_done_by_data[i] = 0; }
	//~ bool all_data_available = true;
	 //~ /* Cherche nb de tache fais par chaque donnée parmis les tâches prévus qu'il reste à faire */
	 //~ for (task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
	 //~ {
	     //~ all_data_available = true;
	     //~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	     //~ {
		//~ if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), node))
		//~ {
		     //~ all_data_available = false;
		     //~ break;
		//~ }
	     //~ }
	     //~ if (all_data_available == true)
	     //~ {
		 //~ for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
		 //~ {
		     //~ for (i = 0; i < nb_data_on_node; i++)
		      //~ {
			  //~ if (data_tab[i] == STARPU_TASK_GET_HANDLE(task, j))
			  //~ {
			      //~ nb_task_done_by_data[i]++;
			      //~ break;
			  //~ }
		      //~ }
		 //~ }
	     //~ }
	 //~ }
	//~ /* Cherche le min dans le tab */
	//~ min = INT_MAX;
	 //~ for (i = 0; i < nb_data_on_node; i++)
	 //~ {
	     //~ if (min > nb_task_done_by_data[i])
	     //~ {
	     //~ if (starpu_data_can_evict(data_tab[i], node, is_prefetch))
	     //~ {
		 //~ min = nb_task_done_by_data[i];
		 //~ returned_handle = data_tab[i];
	     //~ }
	    //~ }
	    //~ else if (min == nb_task_done_by_data[i] && starpu_data_can_evict(data_tab[i], node, is_prefetch))
	    //~ {
		//~ tudl = data_tab[i]->sched_data;
		//~ if (task_using_data_list_size(tudl) < task_using_data_list_size(returned_handle->sched_data))
		//~ {
		    //~ min = nb_task_done_by_data[i];
		    //~ returned_handle = data_tab[i];
		//~ }
	    //~ }
	 //~ }
	 //~ return returned_handle;
    //~ }
}

/* TODO: return NULL ou ne rien faie si la dernière tâche est sorti du post exec hook ? De même pour la mise à jour des listes à chaque eviction de donnée.
 * TODO je rentre bcp trop dans cete fonction on perds du temps car le timing avance lui. */
starpu_data_handle_t dynamic_outer_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{    
    //~ int i = 0;
    //~ int current_gpu = starpu_worker_get_memory_node(starpu_worker_get_id());
    
    //~ /* Se placer sur le bon GPU */
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    //~ for (i = 1; i < current_gpu; i++)
    //~ {
	//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    //~ }
    
    //~ if (my_planned_task_control->pointer->data_to_evict_next != NULL) 
    //~ { 
	//~ printf("Return data %p that was refused.\n", data->my_planned_task_control->pointer->data_to_evict_next);
	//~ starpu_data_handle_t temp_handle = data->my_planned_task_control->pointer->data_to_evict_next;
	//~ data->my_planned_task_control->pointer->data_to_evict_next = NULL;
	//~ return temp_handle;
    //~ }
        
    //~ struct starpu_task *task = NULL;
    //~ starpu_data_handle_t *data_on_node;
    //~ unsigned nb_data_on_node = 0;
    //~ int *valid;
    //~ starpu_data_handle_t returned_handle = NULL;
    //~ starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);
    
    //~ returned_handle = get_handle_least_tasks(data_on_node, nb_data_on_node, node, is_prefetch, current_gpu);
    //~ if (returned_handle == NULL)
    //~ {
	    //~ return STARPU_DATA_NO_VICTIM; 
    //~ }
	//~ /* Enlever de la liste de tache a faire celles qui utilisais cette donnée. Et donc ajouter cette donnée aux données
	  //~ * à pop ainsi qu'ajouter la tache dans les données. Also add it to the main task list. */
	//~ //Suppression de la liste de planned task les tâches utilisant la données
	//~ for (task = starpu_task_list_begin(&data->my_planned_task_control->pointer->planned_task); task != starpu_task_list_end(&data->my_planned_task_control->pointer->planned_task); task = starpu_task_list_next(task))
	//~ {
	    //~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	    //~ {
		//~ if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
		//~ {
		    //~ //Suppression de la liste de tâches à faire 
		    //~ struct pointer_in_task *pt = task->sched_data;
		    //~ starpu_task_list_erase(&data->my_planned_task_control->pointer->planned_task, pt->pointer_to_cell);
			
			 //~ /* NEW */
			//~ pt->pointer_to_cell = task;
			//~ pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
			//~ pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
			    
			//~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			//~ {
			    //~ /* Pointer toward the main task list in the handles. */
			    //~ struct task_using_data *e = task_using_data_new();
			    //~ e->pointer_to_T = task;
			    
			    //~ if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL) 
			    //~ {
				//~ struct task_using_data_list *tl = task_using_data_list_new();
				//~ task_using_data_list_push_front(tl, e);
				//~ STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
			    //~ }
			    //~ else
			    //~ {
				//~ task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
			    //~ }
				
			    //~ /* Adding the pointer in the task. */
			    //~ pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
			    //~ pt->tud[i] = e;
			//~ }	
			//~ task->sched_data = pt;
			
			//~ //Ajout a la liste de tâches principales ces mêmes tâches
			//~ starpu_task_list_push_back(&data->main_task_list, task);

		    //~ break;
		//~ }
	    //~ }
	//~ } 
	
	//~ //Ajout de la données aux données pas encore traitées du gpu
	//~ struct datatype *d = malloc(sizeof(*d));
	//~ d = returned_handle->user_data;

	//~ push_back_data_not_used_yet(returned_handle, data->my_planned_task_control->pointer, d->type);
	
	 //~ printf("Return %p in victim selector.\n", returned_handle);
	 //~ return returned_handle;
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
    int i = 0;
    _STARPU_MALLOC( my_planned_task_control, sizeof(*my_planned_task_control));
    struct gpu_planned_task *new = malloc(sizeof(*new));
    
    starpu_task_list_init(&new->planned_task);
    starpu_task_list_init(&new->refused_fifo_list);
    new->gpu_data = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
    new->data_type_to_pop = 0;
    new->data_to_evict_next = NULL;
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	new->gpu_data[i] = NULL;
    }
    new->next = NULL;
    
    my_planned_task_control->pointer = new;
    my_planned_task_control->first = my_planned_task_control->pointer;
}

void gpu_planned_task_insertion()
{
    int i = 0;
    struct gpu_planned_task *new = malloc(sizeof(*new));
    
    starpu_task_list_init(&new->planned_task);
    starpu_task_list_init(&new->refused_fifo_list);
    new->gpu_data = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
    new->data_type_to_pop = 0;
    new->data_to_evict_next = NULL;
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	new->gpu_data[i] = NULL;
    }
    new->next = my_planned_task_control->pointer;
    my_planned_task_control->pointer = new;
}

void gpu_pulled_task_initialisation()
{
    _STARPU_MALLOC(my_pulled_task_control, sizeof(*my_pulled_task_control));
    struct gpu_pulled_task *new = malloc(sizeof(*new));
    starpu_task_list_init(&new->pulled_task);
    new->next = NULL;
    
    my_pulled_task_control->pointer = new;
    my_pulled_task_control->first = my_pulled_task_control->pointer;
}

void gpu_pulled_task_insertion()
{
    struct gpu_pulled_task *new = malloc(sizeof(*new));
     starpu_task_list_init(&new->pulled_task);
    new->next = my_pulled_task_control->pointer;    
    my_pulled_task_control->pointer = new;
}

struct starpu_sched_component *starpu_sched_component_dynamic_outer_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_outer");
	srandom(starpu_get_env_number_default("SEED", 0));
	int i = 0;
	
	/* Initialization of global variables. */
	Ndifferent_data_type = 2; // TODO: changer cela si on est en 3D ou autre
	Ngpu = get_number_GPU();
	NT = 0;
	new_tasks_initialized = false;
	index_current_popped_task = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_prefetch = malloc(sizeof(int)*Ngpu);
	index_current_popped_task_all_gpu = 0;
	index_current_popped_task_all_gpu_prefetch = 0;
	gpu_memory_initialized = false;
	number_task_out = -1;
	
	/* Initialization of structures. */
	//~ struct HFP_sched_data *data;
	struct dynamic_outer_sched_data *data;
	//~ struct my_list *my_data = malloc(sizeof(*my_data));
	//~ struct paquets *paquets_data = malloc(sizeof(*paquets_data));
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
	
	//~ starpu_task_list_init(&my_data->planned_task);
	//~ starpu_task_list_init(&my_data->refused_fifo_list);
 	//~ my_data->next = NULL;
	//~ paquets_data->temp_pointer_1 = my_data;
	//~ paquets_data->first_link = paquets_data->temp_pointer_1;
	//~ data->p = paquets_data;
	
	//~ data->my_planned_task_control->pointer->gpu_data = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
	//~ data->my_planned_task_control->pointer->gpu_data_loaded = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
	//~ data->my_planned_task_control->pointer->memory_used = 0;
	
	//~ data->my_planned_task_control->pointer->data_type_to_pop = 0;
	
	/* Creating as much package as there are GPUs. */
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
	    //~ printf("Insertion.\n");
	    //~ dynamic_outer_insertion(data->p);
	//~ }
	//~ data->p->first_link = data->my_planned_task_control->pointer;
	//~ data->p->first_link->data_to_evict_next = NULL;
	
	/* Initiliazing global struct for eviction. */
	//~ data_to_evict_element_e = malloc(sizeof(*data_to_evict_element_e)); 
	//~ data_to_evict_control_c = malloc(sizeof(*data_to_evict_control_c));
	//~ data_to_evict_element_e->element = NULL;
	//~ data_to_evict_element_e->next = NULL;
	//~ data_to_evict_control_c->pointeur = data_to_evict_element_e;
	//~ data_to_evict_control_c->first = data_to_evict_control_c->pointeur;
	//~ for (i = 0; i < Ngpu - 1; i++)
	//~ {
	    //~ data_to_evict_insertion(data_to_evict_control_c);
	//~ }
	//~ data_to_evict_control_c->first = data_to_evict_control_c->pointeur;
	
	/* Init data that was refused at eviction. */
	//~ data_to_evict_next = NULL;
	
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

/* Get the task that was last executed. Used to update the task list of planned task. */
void get_task_done(struct starpu_task *task, unsigned sci)
{
    //~ printf("Dans le post exec hook avec la tâche %p.\n", task);
    //~ printf("The planned order was:\n");
    //~ print_planned_task();
    //~ /* Je supprime de la liste de tâches prévus celle qui vient de se terminer */
    //~ int i = 0;
    //~ struct planned_task *pt = planned_task_new();
    
    //~ /* Je me place sur la liste correspondant au bon gpu. */
    //~ my_planned_task_control->pointer = my_planned_task_control->first;
    //~ for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
    //~ {
	//~ my_planned_task_control->pointer = my_planned_task_control->pointer->next;
    //~ }
    
    //~ /* J'efface la tâche dans la liste de tâches */
    //~ for (pt = planned_task_list_begin(my_planned_task_control->pointer->ptpt); pt != planned_task_list_end(my_planned_task_control->pointer->ptpt); pt = planned_task_list_next(pt))
    //~ {
	//~ if (pt->pointer_to_planned_task == task)
	//~ {
	    //~ planned_task_list_erase(my_planned_task_control->pointer->ptpt, pt);
	    //~ break;
	//~ }
    //~ }
    
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
