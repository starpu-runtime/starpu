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

/* Pushing the tasks. Each time a new task enter here, we initialize it. */		
static int dynamic_outer_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
    /* If this boolean is true, pull_task will know that new tasks have arrived and
     * thus it will be able to randomize both the task list and the data list not used yet in the GPUs. 
     */
    new_tasks_initialized = true; 
    struct HFP_sched_data *data = component->data;

    initialize_task_data_gpu_single_task(task, data->p);
    
    /* Pushing the task in sched_list. It's this list that will be randomized
     * and put in popped_task_list in pull_task.
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
 * pointer to the cell in the main task list (popped_task_list).
 * data -> pointer to the tasks using this data.
 * GPUs -> datas not used yet by this GPU.
 */
void initialize_task_data_gpu_single_task(struct starpu_task *task, struct paquets *p)
{
    int i = 0;
    int j = 0;
    
    /* Adding the data not used yet in the corresponding GPU. */
    p->temp_pointer_1 = p->first_link;
    for (i = 0; i < Ngpu; i++)
    {
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    struct gpu_data_not_used *e = gpu_data_not_used_new();
	    e->D = STARPU_TASK_GET_HANDLE(task, j);
	    
	    /* To get the data type of each data. It's in user_data. 
	     * sched_data og the handles is used for the task using this data. */
	    struct datatype *d = malloc(sizeof(*d));
	    d->type = j; 
	    STARPU_TASK_GET_HANDLE(task, j)->user_data = d;
	    
	    /* If the void * of struct paquet is empty I initialize it. */ 
	    if (p->temp_pointer_1->gpu_data[j] == NULL)
	    {
		struct gpu_data_not_used_list *gd = gpu_data_not_used_list_new();
		gpu_data_not_used_list_push_front(gd, e);
		p->temp_pointer_1->gpu_data[j] = gd; 
	    }
	    else
	    {
		if (STARPU_TASK_GET_HANDLE(task, j)->sched_data == NULL)
		{
		    gpu_data_not_used_list_push_front(p->temp_pointer_1->gpu_data[j], e);
		}
	    }
	}
	p->temp_pointer_1 = p->temp_pointer_1->next;
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

/* Randomize the list of data not used yet for all the GPU. */
void randomize_data_not_used_yet(struct paquets *p)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    int random = 0;
    int number_of_data[Ndifferent_data_type];
    p->temp_pointer_1 = p->first_link;
    
    /* I need this for the %random. */
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	number_of_data[i] = gpu_data_not_used_list_size(p->temp_pointer_1->gpu_data[i]);
    }
    for (i = 0; i < Ngpu; i++)
    {
	p->temp_pointer_1->number_handle_to_pop = 0;
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    struct gpu_data_not_used_list *randomized_list = gpu_data_not_used_list_new();
	    for (l = 0; l < number_of_data[j]; l++)
	    {
		/* After each time I remove a data I can choose between a smaller number of value for random. */
		random = rand()%(number_of_data[j]- l);
		for (k = 0; k < random; k++)
		{
		    gpu_data_not_used_list_push_back(p->temp_pointer_1->gpu_data[j], gpu_data_not_used_list_pop_front(p->temp_pointer_1->gpu_data[j]));
		}
		/* I use an external list. */
		gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(p->temp_pointer_1->gpu_data[j]));
	    }
	    /* Then replace the list with it. */
	    p->temp_pointer_1->gpu_data[j] = randomized_list;
	    p->temp_pointer_1->number_handle_to_pop += number_of_data[j];
	}
	//~ p->temp_pointer_1->number_handle_to_pop = number_of_data[0];
	p->temp_pointer_1 = p->temp_pointer_1->next;
    }
}

/* Randomize the list of data not used yet for a single GPU. */
void randomize_data_not_used_yet_single_GPU(struct my_list *l)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int random = 0;
    int number_of_data[Ndifferent_data_type];
    
    l->number_handle_to_pop = 0;
    
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	number_of_data[i] = gpu_data_not_used_list_size(l->gpu_data[i]);
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
		gpu_data_not_used_list_push_back(l->gpu_data[i], gpu_data_not_used_list_pop_front(l->gpu_data[i]));
	    }
	    /* I use an external list. */
	    gpu_data_not_used_list_push_back(randomized_list, gpu_data_not_used_list_pop_front(l->gpu_data[i]));
	}
	/* Then replace the list with it. */
	l->gpu_data[i] = randomized_list;
	l->number_handle_to_pop += number_of_data[i];
    }
    //~ l->number_handle_to_pop = number_of_data[0];
}

/* Just to track where I am on the exec.
 * TODO : A supprimer quand j'aurais tout finis car c'est inutile.
 */
int number_task_out = -1;

/* Pull tasks. When it receives new task it will randomize the task list and the GPU data list.
 * If it has no task it return NULL. Else if a task was refused it return it. Else it return the
 * head of the GPU task list. Else it calls dyanmic_outer_scheuling to fill this package. */
static struct starpu_task *dynamic_outer_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct HFP_sched_data *data = component->data;
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
	print_data_not_used_yet(data->p);
	print_task_list(&data->sched_list, "");
	NT = starpu_task_list_size(&data->sched_list);
	printf("Il y a %d tâches.\n", NT);
	randomize_task_list(data);
	randomize_data_not_used_yet(data->p);
	new_tasks_initialized = false;
	printf("Printing GPU's data list and main task list after randomization:\n\n");
	print_data_not_used_yet(data->p);
	print_task_list(&data->popped_task_list, "");
    }
	    
    /* Getting on the right GPU's package.
     * TODO: Can I do this faster with pointer directly to the cell ?
     */
    data->p->temp_pointer_1 = data->p->first_link;
    for (i = 1; i < current_gpu; i++)
    {
	data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
    }
    
    /* If there are still tasks either in the packages, the main task list or the refused task,
     * I enter here to return a task or start dynamic_outer_scheduling. Else I return NULL.
     */
    if (!starpu_task_list_empty(&data->p->temp_pointer_1->sub_list) || !starpu_task_list_empty(&data->popped_task_list) || !starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list))
    {	
	printf("GPU n°%d is asking for a task.\n", current_gpu);
	struct starpu_task *task = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list)) 
	{
	    task = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list); 
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    printf("Task n°%d: %p is getting out of pull_task from fifo refused list on GPU n°%d\n", number_task_out, task, current_gpu);
	    return task;
	}

	/* If the package is not empty I can return the head of the task list. */
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->sub_list))
	{
	    number_task_out++;
	    task = starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list);
	    printf("Task n°%d: %p is getting out of pull_task from GPU n°%d\n", number_task_out, task, current_gpu);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, starpu_worker_get_memory_node(starpu_worker_get_id()) - 1);
	    }
	    
	    return task;
	}
	/* Else if there are still tasks in the main task list I call dynamic outer algorithm. */
	else if (!starpu_task_list_empty(&data->popped_task_list))
	{
	    /* Je me remet à nouveau sur le bon gpu, car si entre temps pull_task est rappellé ca me remet au début de la liste chainbée -__- */ 
	    data->p->temp_pointer_1 = data->p->first_link;
	    for (i = 1; i < current_gpu; i++)
	    {
		data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	    }
	    
		    
	    number_task_out++;
	    if (starpu_get_env_number_default("DATA_POP_POLICY", 0) == 0)
	    {
		dynamic_outer_scheduling(&data->popped_task_list, current_gpu, data->p->temp_pointer_1);
	    }
	    else
	    {
		printf("Before calling dynamic outer : on est sur le paquet %d.\n", data->p->temp_pointer_1->index_package);
		dynamic_outer_scheduling_one_data_popped(&data->popped_task_list, current_gpu, data->p->temp_pointer_1);
	    }
	    task = starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list);
	    printf("Task n°%d, %p is getting out of pull_task from GPU n°%d\n", number_task_out, task, current_gpu);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    
	    /* For visualisation in python. */
	    if (starpu_get_env_number_default("PRINTF", 0) == 1)
	    {
		print_data_to_load_prefetch(task, starpu_worker_get_memory_node(starpu_worker_get_id()) - 1);
	    }
	    
	    return task;
	}
	/* Else I will return NULL. But I still need to unlock the mutex. */
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    }
    
    return NULL;
}

/* Add data in the list of data loaded on memory. */
void add_data_to_gpu_data_loaded(struct my_list *l, starpu_data_handle_t h, int data_type)
{    
    struct gpu_data_in_memory *e = gpu_data_in_memory_new();
    e->D = h;
	    
    if (l->gpu_data_loaded[data_type] == NULL)
    {
	struct gpu_data_in_memory_list *gd = gpu_data_in_memory_list_new();
	gpu_data_in_memory_list_push_back(gd, e);
	l->gpu_data_loaded[data_type] = gd; 
    }
    else
    {
	gpu_data_in_memory_list_push_back(l->gpu_data_loaded[data_type], e);
    }
}

void push_back_data_not_used_yet(starpu_data_handle_t h, struct my_list *l, int data_type)
{
    struct gpu_data_not_used *e = gpu_data_not_used_new();
    e->D = h;
    gpu_data_not_used_list_push_back(l->gpu_data[data_type], e);
}

/* Fill a package's task list following dynamic_outer algorithm. It pop only one data, the one that achieve the mos tasks. */
void dynamic_outer_scheduling_one_data_popped(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l)
{
    printf("Data type paquets %d : %d.\n", l->index_package, l->data_type_to_pop);
    
    int i = 0;
    int j = 0;
    int next_handle = 0;
    struct task_using_data *t = NULL;
    struct gpu_data_not_used *e = NULL;
    int number_of_task_max = 0;
    int temp_number_of_task_max = 0;
    starpu_data_handle_t handle_popped = NULL;
    
    if (gpu_data_not_used_list_empty(l->gpu_data[l->data_type_to_pop]))
    {
	goto random;
    }
    
    /* To know if all the data needed for a task are loaded in memory. */
    bool data_available = true; 
    
    if (starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_OUTER", 0) == 1)
    {
	/* If the number of handle popped is equal to the number of original handle it
	 * means that we are on the set of data evicted. So we want to reshuffle it. */
	 l->number_handle_to_pop--;
	 if (l->number_handle_to_pop == 0)
	 {
	     printf("Re-shuffle.\n");
	     //~ l->number_handle_to_pop = gpu_data_not_used_list_size(l->gpu_data[l->data_type_to_pop]);
	     print_data_not_used_yet_one_gpu(l);
	     randomize_data_not_used_yet_single_GPU(l);
	     print_data_not_used_yet_one_gpu(l);
	 }
    }
    for (e = gpu_data_not_used_list_begin(l->gpu_data[l->data_type_to_pop]); e != gpu_data_not_used_list_end(l->gpu_data[l->data_type_to_pop]); e = gpu_data_not_used_list_next(e))
    {
	temp_number_of_task_max = 0;
	
	for (t = task_using_data_list_begin(e->D->sched_data); t != task_using_data_list_end(e->D->sched_data); t = task_using_data_list_next(t))
	{
	    /* I put it at false if at least one data is missing. */
	    data_available = true; 
	    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T) - 1; j++)
	    {
		/* I use %nb_data_for_a_task because I don't want to check the current data type I'm on.*/
		next_handle = (l->data_type_to_pop + 1 + j)%STARPU_TASK_GET_NBUFFERS(t->pointer_to_T);
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
    }
    if (number_of_task_max == 0)
    {
	goto random;
    }
    else /* I erase the data. */
    {
	e = gpu_data_not_used_list_begin(l->gpu_data[l->data_type_to_pop]);
	while (e->D != handle_popped)
	{
	   e = gpu_data_not_used_list_next(e);
        } 
	gpu_data_not_used_list_erase(l->gpu_data[l->data_type_to_pop], e);
	l->memory_used += starpu_data_get_size(handle_popped);
	add_data_to_gpu_data_loaded(l, handle_popped, l->data_type_to_pop);
    }
    printf("The data adding the most task is: %p.\n", handle_popped);
    
    /* Adding the task to the list. TODO : this is a copy paste of the code above to test the available tasks. */
    for (t = task_using_data_list_begin(handle_popped->sched_data); t != task_using_data_list_end(handle_popped->sched_data); t = task_using_data_list_next(t))
    {
	data_available = true; 
	for (j = 0; j < STARPU_TASK_GET_NBUFFERS(t->pointer_to_T) - 1; j++)
	{
	    next_handle = (l->data_type_to_pop + 1 + j)%STARPU_TASK_GET_NBUFFERS(t->pointer_to_T);
		    		
	    if (STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle) != handle_popped)
	    {
		printf("Test if %p is on node for task %p?\n", STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), t->pointer_to_T);
		//~ struct _starpu_data_state *state = STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle);
		//~ printf("%p.\n", &STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle)->per_node[1].state);
		//~ starpu_data_handle_t h;
		//~ printf("%p.\n", &h->per_node[1].state);
		if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
		{
		    data_available = false;
		    printf("Not on node!\n");
		    break;
		}
	    }
	}
	if (data_available == true)
	{
	    printf("Pushing %p in the package.\n", t->pointer_to_T);
	    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(t->pointer_to_T, 0));
	    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(t->pointer_to_T, 1));
	    erase_task_and_data_pointer(t->pointer_to_T, popped_task_list);
	    //~ printf("After erase\n");
	    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(t->pointer_to_T, 0));
	    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(t->pointer_to_T, 1));
	    starpu_task_list_push_front(&l->sub_list, t->pointer_to_T);
	}
    }
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&l->sub_list)) 
    {
	random: ;
	struct starpu_task *task = starpu_task_list_pop_front(popped_task_list);
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
	    if (!gpu_data_not_used_list_empty(l->gpu_data[i]))
	    {
		for (e = gpu_data_not_used_list_begin(l->gpu_data[i]); e != gpu_data_not_used_list_end(l->gpu_data[i]); e = gpu_data_not_used_list_next(e))
		{
		    if(e->D == STARPU_TASK_GET_HANDLE(task, i))
		    {
			gpu_data_not_used_list_erase(l->gpu_data[i], e);
			add_data_to_gpu_data_loaded(l, STARPU_TASK_GET_HANDLE(task, i), i);
		    }
		}
	    }
	    l->memory_used += starpu_data_get_size(STARPU_TASK_GET_HANDLE(task, i));
	}
	
	printf("No task were possible with the popped handles. Returning head of the randomized main task list: %p.\n", task);
	erase_task_and_data_pointer(task, popped_task_list);
	starpu_task_list_push_back(&l->sub_list, task);
    }
    
    /* On veut pop un autre type de donnée la prochaine fois. */
    l->data_type_to_pop = (l->data_type_to_pop + 1)%Ndifferent_data_type;
}

/* Fill a package task list following dynamic_outer algorithm. */
void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l)
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
    starpu_data_handle_t *handle_popped = malloc(Ndifferent_data_type*sizeof(STARPU_TASK_GET_HANDLE(starpu_task_list_begin(popped_task_list), 0)));
    
    /* To know if all the data needed for a task are loaded in memory. */
    bool data_available = true; 
    /* To know if it's the task using the Ndifferent_data_type data popped. */
    bool handle_popped_task = true;

    for (i = 0; i < Ndifferent_data_type; i++)
    {
	/* If at least one data type is empty in the GPU I return a random task.
	 * TODO: This is useless if we add correctly data at the back of the list when evicting, so to remove.
	 */
	if (gpu_data_not_used_list_empty(l->gpu_data[i]))
	{
	    goto return_random_task;
	}
	/* Else I can pop a data. */
	e = gpu_data_not_used_list_pop_front(l->gpu_data[i]);
	handle_popped[i] = e->D;
	
	/* And I add this data to the data loaded on the GPU in the same package.
	 * I also increment the memory used.
	 */
	l->memory_used += starpu_data_get_size(handle_popped[i]);
	add_data_to_gpu_data_loaded(l, handle_popped[i], i);
    }
    
    /* TODO : a enlever ici et plus bas car mtn je check dans le victim selector.
     * TODO : vérifier les shuffle a la main et eneever ce qu'il faut. */
    starpu_data_handle_t *evicted_handles = malloc(Ndifferent_data_type*sizeof(STARPU_TASK_GET_HANDLE(starpu_task_list_begin(popped_task_list), 0)));
    if (starpu_get_env_number_default("EVICTION_STRATEGY_DYNAMIC_OUTER", 0) == 1) 
    {
	/* If we exceed the GPU's memory with the new data I need to evict as much data. */
	//~ if (l->memory_used > GPU_RAM_M)
	//~ {
	    //~ printf("Memory exceeded with the new data.\n");
	    
	    /* This is eviction method n°1 where we evict immediatly. 
	     * The problem is that the data is often not loaded when I try to evict it.
	     */
	    //~ int result = 0;
	    //~ struct gpu_data_in_memory *evicted_handle = NULL;
	    //~ for (i = 0; i < Ndifferent_data_type; i++)
	    //~ {
		//~ /* I take data from the data already loaded following a FIFO rule. */
		//~ evicted_handle = gpu_data_in_memory_list_pop_front(l->gpu_data_loaded[i]);
		//~ l->memory_used -= starpu_data_get_size(evicted_handle->D);
			
		//~ /* I call the function that evict two data from the memory immediatly. */
		//~ result = starpu_data_evict_from_node(evicted_handle->D, current_gpu);
		//~ printf("Result of eviction = %d\n", result);
		
		//~ /* I add it at the end of the data list not used by the GPU. */
		//~ push_back_data_not_used_yet(evicted_handle->D, l, i);
	    //~ }
	    /* End of eviction method n°1. */
	    
	    /* This is eviction method n°2 where we evict with starpu_data_register_victim_selector
	     * when we are asked for a data to evict. In this case we evict the head of the data list
	     * in gpu_data_in_memory of the corresponding package. We also need to ignore the tasks using
	     * these data when we fill the package with tasks.
	     */
	    //~ struct gpu_data_in_memory *eh = NULL;
	    //~ /* Get on the right gpu list of data to evict. */
	    //~ data_to_evict_control_c->pointeur = data_to_evict_control_c->first;
	    
	    //~ for (i = 0; i < current_gpu - 1; i++)
	    //~ {
		//~ data_to_evict_control_c->pointeur = data_to_evict_control_c->pointeur->next;
	    //~ }
	    //~ for (i = 0; i < Ndifferent_data_type; i++)
	    //~ {
		//~ /* So here we suppose that this handle will be evicted. */
		//~ eh = gpu_data_in_memory_list_pop_front(l->gpu_data_loaded[i]);
		//~ l->memory_used -= starpu_data_get_size(eh->D);
		//~ evicted_handles[i] = eh->D;
		//~ push_back_data_not_used_yet(eh->D, l, i);
			
		//~ /* And I add these handles in the list of handle of the corresponding gpu in a global struct. */
		//~ struct data_to_evict *d = data_to_evict_new();
		//~ d->D = eh->D;
		//~ /* If the void * of struct paquet is empty I initialize it. */ 
		//~ if (data_to_evict_control_c->pointeur->element == NULL)
		//~ {
		    //~ struct data_to_evict_list *dl = data_to_evict_list_new();
		    //~ data_to_evict_list_push_back(dl, d);
		    //~ data_to_evict_control_c->pointeur->element = dl; 
		//~ }
		//~ else
		//~ {
		    //~ data_to_evict_list_push_back(data_to_evict_control_c->pointeur->element, d);
		//~ }
	    //~ }
	    //~ /* End of eviction method n°2. */
	//~ }
	
	/* If the number of handle popped is equal to the number of original handle it
	 * means that we are on the set of data evicted. So we want to reshuffle it. */
	 l->number_handle_to_pop--;
	 if (l->number_handle_to_pop == 0)
	 {
	     printf("Re-shuffle\n");
	     randomize_data_not_used_yet_single_GPU(l);
	 }
     }
    
    /* Just printing. */
    //~ printf("Handles popped:");
    //~ for (i = 0; i < Ndifferent_data_type; i++)
    //~ {
	//~ printf(" %p", handle_popped[i]);
    //~ }
    //~ printf("\n\n");
    //~ printf("Task using these handles:");
    //~ for (struct task_using_data *t = task_using_data_list_begin(handle_popped[0]->sched_data); t != task_using_data_list_end(handle_popped[0]->sched_data); t = task_using_data_list_next(t))
    //~ {
	//~ printf(" %p", t->pointer_to_T);
    //~ }
    //~ printf(" /");
    //~ for (struct task_using_data *t = task_using_data_list_begin(handle_popped[1]->sched_data); t != task_using_data_list_end(handle_popped[1]->sched_data); t = task_using_data_list_next(t))
    //~ {
	//~ printf(" %p", t->pointer_to_T);
    //~ }
    //~ printf("\n\n");
    /* End of printing. */
	
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
	    //~ printf("Task %p use %p.\n", t->pointer_to_T, handle_popped[i]);	
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
			    //~ printf("Data %p is the one we will evict soon.\n", STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle)); 
			    data_available = false;
			    break;
			}
		    }
		    
		    if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
		    {
			//~ printf("Data %p is not on memory nor is popped.\n", STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle)); 
			data_available = false;
			break;
		    }
		}
	    }
	    if (data_available == true)
	    {
		//~ printf("Pushing %p in the package.\n", t->pointer_to_T);
		/* Deleting the task from the task list of data A, B (and C) and from the main task list. */
		erase_task_and_data_pointer(t->pointer_to_T, popped_task_list);
		
		/* Pushing on top the task using all popped handles. */
		if (handle_popped_task == true)
		{
		    starpu_task_list_push_front(&l->sub_list, t->pointer_to_T);
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
		starpu_task_list_push_back(&l->sub_list, starpu_task_list_pop_front(task_tab[j]));
		pushed_task--;
	    }
	}
    }
    
    /* If no task have been added to the list. */
    if (starpu_task_list_empty(&l->sub_list)) 
    {
	return_random_task: ;
	struct starpu_task *task = starpu_task_list_pop_front(popped_task_list);
	printf("No task were possible with the popped handles. Returning head of the randomized main task list: %p.\n", task);
	erase_task_and_data_pointer(task, popped_task_list);
	starpu_task_list_push_back(&l->sub_list, task);
    }
    
    free(handle_popped);
    //~ printf("\n");
}

/* Pour savoir si la donnée évincé est bien celle que l'on avais prévu.
 * Si ce n'est pas le cas ou si ca vaut NULL alors cela signifie qu'une donnée non prévu a 
 * été évincé. Il faut donc mettre à jour les listes dans les tâches et les données en conséquence.
 * Cependant si on est sur la fin de l'éxécution et que les éviction sont juste la pour vider la mémoire ce n'est pas
 * nécessaire. En réalité pour le moment je ne me rend pas compte qu'on est a la fin de l'exec. 
 * TODO : se rendre compte qu'on est a la fin et arreter de mettre à jour les listes du coup.
 */
 //~ starpu_data_handle_t planned_eviction;

void dynamic_outer_victim_evicted(int success, starpu_data_handle_t victim, void *component)
{
     /* If a data was not truly evicted I put it back in the list. */
    if (success == 0)
    {
	int i = 0;
	struct starpu_sched_component *temp_component = component;
	struct HFP_sched_data *data = temp_component->data;
	
	printf("Current gpu in victim evicted %d.\n", starpu_worker_get_memory_node(starpu_worker_get_id()));
	
	data->p->temp_pointer_1 = data->p->first_link;
	for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
	{
	    data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	}
	
	/* Version 1 seule donnée. A voir si ca marche en multi GPU */
	data->p->temp_pointer_1->data_to_evict_next = victim;
    }
    else
    {
	/* Si une autre donnée a été évincé je dois mettre à jour mes listes dans les tâches, les gpus et les données et la liste principale de tâches. */
	//~ if (victim != planned_eviction)
	//~ {
	    //~ printf("Victim != planned_eviction.\n");
	    //~ int i = 0;
	    //~ struct starpu_task *task = NULL;
	        //~ struct starpu_sched_component *temp_component = component;
	        //~ struct HFP_sched_data *data = temp_component->data;
	    //~ for (task = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); task != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
	    //~ {
		//~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		//~ {
		    //~ if (STARPU_TASK_GET_HANDLE(task, i) == victim)
		    //~ {
			//~ //Suppression de la liste de tâches à faire 
			//~ struct pointer_in_task *pt = task->sched_data;
			//~ starpu_task_list_erase(&data->p->temp_pointer_1->sub_list, pt->pointer_to_cell);

			    //~ pt->pointer_to_cell = task;
			    //~ pt->pointer_to_D = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(STARPU_TASK_GET_HANDLE(task, 0)));
			    //~ pt->tud = malloc(STARPU_TASK_GET_NBUFFERS(task)*sizeof(task_using_data_new()));
				
			    //~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
			    //~ {
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
				    
				//~ pt->pointer_to_D[i] = STARPU_TASK_GET_HANDLE(task, i);
				//~ pt->tud[i] = e;
			    //~ }	
			    //~ task->sched_data = pt;
			    
			    //~ //Ajout a la liste de tâches principales ces mêmes tâches
			    //~ starpu_task_list_push_back(&data->popped_task_list, task);

			//~ break;
		    //~ }
		//~ }
	//~ } 
	//~ //Ajout de la données aux données pas encore traitées du gpu
	//~ struct datatype *d = malloc(sizeof(*d));
	//~ d = victim->user_data;
	
	//~ print_task_using_data(victim);
	
	//~ printf("Pushing back data in not used yet.%p\n", victim);
	//~ push_back_data_not_used_yet(victim, data->p->temp_pointer_1, d->type);
	//~ }
	//~ else
	//~ {
	    //~ printf("La donnée évincé est la même que celle qui était prévu. Rien à faire.\n");
	//~ }
	return;
    }
}

/* Return the handle that can do the least tasks that already have all
 * it data on memory. If there is a draw or if there are no task in the task list, return the 
 * data that has the least remaining task (even if their data are not loaded on memory.
 */
starpu_data_handle_t get_handle_least_tasks(struct starpu_task_list *l, starpu_data_handle_t *data_tab, int nb_data_on_node, unsigned node, enum starpu_is_prefetch is_prefetch)
{
    starpu_data_handle_t returned_handle = NULL;
    int i = 0;
    int min = 0;
    struct task_using_data_list *tudl = task_using_data_list_new();
    if (starpu_task_list_empty(l))
    {
	min = INT_MAX;
	
	for (i = 0; i < nb_data_on_node; i++)
	{
	    tudl = data_tab[i]->sched_data;
	    if (task_using_data_list_size(tudl) < min && starpu_data_can_evict(data_tab[i], node, is_prefetch))
	    {
		min = task_using_data_list_size(tudl);
		returned_handle = data_tab[i];
	    }
	}
	//~ printf("Return %p, sub list empty.\n", returned_handle);
	return returned_handle;
    }
    else
    {
	int j = 0;
	struct starpu_task *task = NULL;
	int nb_task_done_by_data[nb_data_on_node];
	for (i = 0; i < nb_data_on_node; i++) { nb_task_done_by_data[i] = 0; }
	bool all_data_available = true;
	 //~ printf("Planned task are :");
	 /* Cherche nb de tache fais par chaque donnée parmis les données prévus qu'il reste à faire */
	 for (task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
	 {
	     all_data_available = true;
	     for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	     {
		if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(task, i), node))
		{
		     all_data_available = false;
		     break;
		}
	     }
	     if (all_data_available == true)
	     {
		 for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
		 {
		     for (i = 0; i < nb_data_on_node; i++)
		      {
			  if (data_tab[i] == STARPU_TASK_GET_HANDLE(task, j))
			  {
			      nb_task_done_by_data[i]++;
			      break;
			  }
		      }
		 }
	     }
	 }
	/* Cherche le min dans le tab */
	min = INT_MAX;
	 for (i = 0; i < nb_data_on_node; i++)
	 {
	     //~ printf("%p can do %d tasks.\n", data_tab[i], nb_task_done_by_data[i]);
	     if (min > nb_task_done_by_data[i])
	     {
		 //~ printf("It's inferior to min.\n");
	     if (starpu_data_can_evict(data_tab[i], node, is_prefetch))
	     {
		 //~ printf("I can evict it, new min.\n");
		 min = nb_task_done_by_data[i];
		 returned_handle = data_tab[i];
	     }
	    }
	    else if (min == nb_task_done_by_data[i] && starpu_data_can_evict(data_tab[i], node, is_prefetch))
	    {
		tudl = data_tab[i]->sched_data;
		if (task_using_data_list_size(tudl) < task_using_data_list_size(returned_handle->sched_data))
		{
		    min = nb_task_done_by_data[i];
		    returned_handle = data_tab[i];
		}
	    }
	 }
	 return returned_handle;
    }
}

/* TODO: return NULL ou ne rien faie si la dernière tâche est sorti ? De même pour la mise à jour des listes à chaque eviction de donnée.
 * TODO je renre bcp trop dans cete focntion on perdu du temps car le timing avance lui. */
starpu_data_handle_t dynamic_outer_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *component)
{
    printf("Beggining of victim_selector. On GPU n°%d. Timing is %f.\n", starpu_worker_get_memory_node(starpu_worker_get_id()), starpu_timing_now());
    
    int i = 0;
    struct starpu_sched_component *temp_component = component;
    struct HFP_sched_data *data = temp_component->data;
    
    /* Se placer sur le bon GPU. */
    data->p->temp_pointer_1 = data->p->first_link;
    for (i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++)
    {
	data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
    }
    
    if (data->p->temp_pointer_1->data_to_evict_next != NULL) 
    { 
	printf("Return data %p that was refused.\n", data->p->temp_pointer_1->data_to_evict_next);
	starpu_data_handle_t temp_handle = data->p->temp_pointer_1->data_to_evict_next;
	data->p->temp_pointer_1->data_to_evict_next = NULL;
	return temp_handle;
    }
        
    struct starpu_task *task = NULL;
    starpu_data_handle_t *data_on_node;
    unsigned nb_data_on_node = 0;
    int *valid;
    starpu_data_handle_t returned_handle = NULL;
    starpu_data_get_node_data(node, &data_on_node, &valid, &nb_data_on_node);
	
    /* TODO : Je ne rentre jamsi dedans c'est bizare non ?*/
    //~ for (i = 0; i < nb_data_on_node; i++)
    //~ {
	//~ if (valid[i] == 0 && starpu_data_can_evict(data_on_node[i], node, is_prefetch))
	//~ {
	    //~ exit(0);
	    //~ free(valid);
	    //~ returned_handle = data_on_node[i];
	    //~ free(data_on_node);
	    //~ printf("Returning an invalid data.\n");
	    //~ return returned_handle;
	//~ }
    //~ }
    
    returned_handle = get_handle_least_tasks(&data->p->temp_pointer_1->sub_list, data_on_node, nb_data_on_node, node, is_prefetch);
	if (returned_handle == NULL) { 
	    return STARPU_DATA_NO_VICTIM; 
	    //~ printf("Return NULL.\n");
	    //~ return NULL;
    }
	/* Enlever de la liste de tache a faire celles qui utilisais cette donnée. Et donc ajouter cette donnée aux données
	  * à pop ainsi qu'ajouter la tache dans les données. Also add it to the main task list. */
	//Suppression de la liste de planned task les tâches utilisant la données
	//~ printf("Avant suppression:\n");
	//~ print_packages_in_terminal(data->p, 0);
	for (task = starpu_task_list_begin(&data->p->temp_pointer_1->sub_list); task != starpu_task_list_end(&data->p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
	{
	    for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	    {
		if (STARPU_TASK_GET_HANDLE(task, i) == returned_handle)
		{
		    //~ printf("Deleting task %p\n", task);
		    //Suppression de la liste de tâches à faire 
		    struct pointer_in_task *pt = task->sched_data;
		    starpu_task_list_erase(&data->p->temp_pointer_1->sub_list, pt->pointer_to_cell);
		    //~ print_task_list(&data->p->temp_pointer_1->sub_list, "Après suppression.\n");
		    
		    //Ajout de la tâche dans la liste de tâche de la donnée
		    //~ struct task_using_data *e = task_using_data_new();
		    //~ e->pointer_to_T = task;
		    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(task, i));
		    //~ printf("pushing back %p\n", task);
		    //~ task_using_data_list_push_back(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
		    //~ print_task_using_data(STARPU_TASK_GET_HANDLE(task, i));
		    //~ printf("deleted some things\n");
		    
			/* Pointer toward the main task list in the handles. */
			/* OLD */
			//~ struct task_using_data *e = task_using_data_new();
			//~ e->pointer_to_T = task;
			
			//~ if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL) 
			//~ {
			    //~ printf("new list\n");
			    //~ struct task_using_data_list *tl = task_using_data_list_new();
			    //~ task_using_data_list_push_back(tl, e);
			    //~ STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
			//~ }
			//~ else
			//~ {
			    //~ printf("Adding %p in task to do with handle %p.\n", task, STARPU_TASK_GET_HANDLE(task, i));
			    //~ task_using_data_list_push_back(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
			//~ }
			
			 /* NEW */
			//~ struct pointer_in_task *pt = malloc(sizeof(*pt));
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
			//~ starpu_task_list_push_back(&data->popped_task_list, e->pointer_to_T);
			starpu_task_list_push_back(&data->popped_task_list, task);

		    break;
		}
	    }
	} 
	//~ printf("Apres suppression:\n");
	//~ print_packages_in_terminal(data->p, 0);
	//~ print_task_list(&data->popped_task_list, "after");
	
	//Ajout de la données aux données pas encore traitées du gpu
	//~ printf("Avant:\n");
	//~ print_data_not_used_yet(data->p);
	struct datatype *d = malloc(sizeof(*d));
	d = returned_handle->user_data;
	//~ printf("%p is type %d\n", returned_handle, d->type);
	
	//~ print_task_using_data(returned_handle);
	
	printf("Pushing back data in not used yet.%p\n", returned_handle);
	push_back_data_not_used_yet(returned_handle, data->p->temp_pointer_1, d->type);
	
	//~ printf("Après:\n");
	//~ print_data_not_used_yet(data->p);
	
	//Ajout des tâches supprimées dans la liste de tâche de la donnée
	/* Pointer toward the main task list in the handles. */
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
	
	 printf("Return %p in victim selector.\n", returned_handle);
	 //~ planned_eviction = returned_handle;
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
    struct HFP_sched_data *data = component->data;
    int didwork = 0;
    struct starpu_task *task;
    task = starpu_sched_component_pump_to(component, to, &didwork);
    if (task)
    {
	    //~ printf("Oops, task %p got refused.\n", task);
	    
	    /* If a task is refused I push it in the refused fifo list of the appropriate GPU's package.
	     * This list is lloked at first when a GPU is asking for a task so we don't break the planned order. */
	    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	    data->p->temp_pointer_1 = data->p->first_link;
	    for (int i = 1; i < starpu_worker_get_memory_node(starpu_worker_get_id()); i++) 
	    {
		data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	    }
	    starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    }
    //~ else
    //~ {
	/* Can I uncomment this part ? */
	//~ {
	    //~ if (didwork)
		//~ fprintf(stderr, "pushed some tasks to %p\n", to);
	    //~ else
		//~ fprintf(stderr, "I didn't have anything for %p\n", to);
	//~ }
    //~ }
    
    /* There is room now */
    return didwork || starpu_sched_component_can_push(component, to);
}

static int dynamic_outer_can_pull(struct starpu_sched_component *component)
{
    return starpu_sched_component_can_pull(component);
}

/* Put a link at the beginning of the linked list.
 * Different one from HFP_insertion because I also init
 * gpu_data and i don't init other fields.
 */
void dynamic_outer_insertion(struct paquets *a)
{
    int j = 0;
    struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
    starpu_task_list_init(&new->sub_list);
    new->next = a->temp_pointer_1;
    starpu_task_list_init(&new->refused_fifo_list);
    
    new->gpu_data = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
    new->gpu_data_loaded = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
    new->memory_used = 0;
    new->data_type_to_pop = 0;
    new->data_to_evict_next = NULL;
        
    for (j = 0; j < Ndifferent_data_type; j++)
    {
	new->gpu_data[j] = NULL;
	new->gpu_data_loaded[j] = NULL;
    }
    a->temp_pointer_1 = new;
}

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

void print_data_not_used_yet(struct paquets *p)
{
    int i = 0;
    int j = 0;
    p->temp_pointer_1 = p->first_link;
    
    for (i = 0; i < Ngpu; i++)
    {
	printf("On GPU n°%d, the data not used yet are:", i + 1);
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    printf("\nFor the data type n°%d:", j);
	    for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(p->temp_pointer_1->gpu_data[j]); e != gpu_data_not_used_list_end(p->temp_pointer_1->gpu_data[j]); e = gpu_data_not_used_list_next(e))
	    {
		printf(" %p", e->D);
	    }
	}
	printf("\n");
	p->temp_pointer_1 = p->temp_pointer_1->next;
    }
    p->temp_pointer_1 = p->first_link;
    printf("\n");
}

void print_data_not_used_yet_one_gpu(struct my_list *l)
{
    int j = 0;    
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    printf("\nFor the data type n°%d:", j);
	    for (struct gpu_data_not_used *e = gpu_data_not_used_list_begin(l->gpu_data[j]); e != gpu_data_not_used_list_end(l->gpu_data[j]); e = gpu_data_not_used_list_next(e))
	    {
		printf(" %p", e->D);
	    }
	}
	printf("\n");
}

void print_data_loaded(struct paquets *p)
{
    int i = 0;
    int j = 0;
    p->temp_pointer_1 = p->first_link;
    
    for (i = 0; i < Ngpu; i++)
    {
	printf("On GPU n°%d, the data loaded are:", i + 1);
	for (j = 0; j < Ndifferent_data_type; j++)
	{
	    printf("\nFor the data type n°%d:", j);
	    for (struct gpu_data_in_memory *e = gpu_data_in_memory_list_begin(p->temp_pointer_1->gpu_data_loaded[j]); e != gpu_data_in_memory_list_end(p->temp_pointer_1->gpu_data_loaded[j]); e = gpu_data_in_memory_list_next(e))
	    {
		printf(" %p", e->D);
	    }
	}
	printf("\nThese data weight: %ld\n", p->temp_pointer_1->memory_used);
	p->temp_pointer_1 = p->temp_pointer_1->next;
    }
    p->temp_pointer_1 = p->first_link;
    printf("\n");
}

void print_packages(struct paquets *p)
{
    int i = 0;
    p->temp_pointer_1 = p->first_link;
    
    for (i = 0; i < Ngpu; i++)
    {
	printf("On GPU n°%d we have:", i + 1);
	for (struct starpu_task *task = starpu_task_list_begin(&p->temp_pointer_1->sub_list); task != starpu_task_list_end(&p->temp_pointer_1->sub_list); task = starpu_task_list_next(task))
	{
	    printf(" %p", task);
	}
	printf("\n");
	p->temp_pointer_1 = p->temp_pointer_1->next;
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

void randomize_task_list(struct HFP_sched_data *d)
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
	starpu_task_list_push_back(&d->popped_task_list, starpu_task_list_pop_front(&d->sched_list));
    }
    //~ printf("J'ai parcouru la liste de tâche complète, puis la liste - 1 élément et ainsi de suite. Cela pour randomiser la liste de tâches initiale dans randomize_task_list. Complexité : O(NT^2)\n\n");
}

//~ void data_to_evict_insertion(struct data_to_evict_control *d)
//~ {
    //~ struct data_to_evict_element *new = malloc(sizeof(*new));
    //~ new->next = d->pointeur;    
    //~ new->element = NULL;
    //~ d->pointeur = new;
//~ }

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
	
	/* Initialization of structures. */
	struct HFP_sched_data *data;
	struct my_list *my_data = malloc(sizeof(*my_data));
	struct paquets *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);
 	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
	data->p = paquets_data;
	
	data->p->temp_pointer_1->gpu_data = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
	data->p->temp_pointer_1->gpu_data_loaded = malloc(Ndifferent_data_type*sizeof(starpu_data_handle_t));
	data->p->temp_pointer_1->memory_used = 0;
	
	data->p->temp_pointer_1->data_type_to_pop = 0;
	
	/* Creating as much package as there are GPUs. */
	for (i = 0; i < Ngpu - 1; i++)
	{
	    printf("Insertion.\n");
	    dynamic_outer_insertion(data->p);
	}
	
		//~ paquets_data->first_link = paquets_data->temp_pointer_1;
		
	data->p->first_link = data->p->temp_pointer_1;
	data->p->first_link->data_to_evict_next = NULL;
	
	printf("%d.\n", data->p->temp_pointer_1->index_package);
	data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
	printf("%d.\n", data->p->temp_pointer_1->index_package);
	//~ exit(0);
	
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
	//~ component->do_schedule = dynamic_outer_do_schedule;
	component->push_task = dynamic_outer_push_task;
	component->pull_task = dynamic_outer_pull_task;
	component->can_push = dynamic_outer_can_push;
	component->can_pull = dynamic_outer_can_pull;
	
	/* TODO: initialiser le victim_selector et victim_evicted. Et de même pour HFP. */
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

struct starpu_sched_policy _starpu_sched_dynamic_outer_policy =
{
	.init_sched = initialize_dynamic_outer_center_policy,
	.deinit_sched = deinitialize_dynamic_outer_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	//~ .do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	//~ .pop_task = starpu_sched_tree_pop_task,
	.pop_task = get_data_to_load,
	//~ .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.pre_exec_hook = get_current_tasks,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "dynamic-outer",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading 2 random data",
	.worker_type = STARPU_WORKER_LIST,
};
