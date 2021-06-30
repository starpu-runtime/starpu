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
 * Complexité :     //~ printf("J'ai parcouru la liste de tâche complète dans initialize_task_list_using_data(struct starpu_task_list *l) pour ajouter chaque tâche dans une liste dans les handles. Complexité : O(NT). J'ai également parcouru la liste de donnée de chaque GPU pour savoir lesquelles je n'avais pas encore mis dedans. complexité : O(ND^2).\n\n");
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

/* Randomize the list of data not used yet by a GPU. */
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
	}
	p->temp_pointer_1->number_handle_to_pop = number_of_data[0];
	p->temp_pointer_1 = p->temp_pointer_1->next;
    }
}

void randomize_data_not_used_yet_single_GPU(struct my_list *l)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int random = 0;
    int number_of_data[Ndifferent_data_type];
    
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
    }
}

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
	printf("GPU n°%d is asking for a task!\n", current_gpu);
	
	struct starpu_task *task = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list)) 
	{
	    task = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list); 
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    printf("Task %p is getting out of pull_task from fifo refused list on GPU n°%d\n", task, current_gpu);
	    return task;
	}

	/* If the package is not empty I can return the head of the task list. */
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->sub_list))
	{
	    task = starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list);
	    printf("Task %p is getting out of pull_task from GPU n°%d\n", task, current_gpu);
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
	    print_data_not_used_yet(data->p);
	    dynamic_outer_scheduling(&data->popped_task_list, current_gpu, data->p->temp_pointer_1);
	    print_data_loaded(data->p);
	    task = starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list);
	    printf("Task %p is getting out of pull_task from GPU n°%d\n", task, current_gpu);
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

/* Fill a package task list following dynamic_outer algorithm. */
void dynamic_outer_scheduling(struct starpu_task_list *popped_task_list, int current_gpu, struct my_list *l)
{
    printf("Beggining of dynamic_outer_scheduling.\n\n");
    
    /* Test mémoire */
    //~ printf("On GPU n°%d the memory available is: %ld\n", current_gpu, starpu_memory_get_available(current_gpu));
    
    int i = 0;
    int j = 0;
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
    
    /* If we exceed the GPu's memory with the new data I need to evict as much data. */
    if (l->memory_used > GPU_RAM_M)
    {
	struct gpu_data_in_memory *evicted_handle = NULL;
	printf("Memory exceeded with the new data.\n");
	for (i = 0; i < Ndifferent_data_type; i++)
	{
	    /* I take data from the data already loaded following a FIFO rule. */
	    evicted_handle = gpu_data_in_memory_list_pop_front(l->gpu_data_loaded[i]);
	    e->D = evicted_handle->D;
	    l->memory_used -= starpu_data_get_size(evicted_handle->D);
	    /* I call the function that evict two data from the memory immediatly. */
	    //TODO function_to_evict(evicted_handle->D);
	    /* I add them at the end of the data list not used by the GPU. */
	    gpu_data_not_used_list_push_back(l->gpu_data[i], e);
	}
    }
    
    /* If the number of handle popped is equal to the number of original handle it
     * means that we are on the set of data evicted. So we want to reshuffle it. */
     l->number_handle_to_pop--;
     if (l->number_handle_to_pop == 0)
     {
	 randomize_data_not_used_yet_single_GPU(l);
     }
    
    /* Just printing. */
    printf("Handles popped:");
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	printf(" %p", handle_popped[i]);
    }
    printf("\n\n");
    printf("Task using these handles:");
    for (struct task_using_data *t = task_using_data_list_begin(handle_popped[0]->sched_data); t != task_using_data_list_end(handle_popped[0]->sched_data); t = task_using_data_list_next(t))
    {
	printf(" %p", t->pointer_to_T);
    }
    printf(" /");
    for (struct task_using_data *t = task_using_data_list_begin(handle_popped[1]->sched_data); t != task_using_data_list_end(handle_popped[1]->sched_data); t = task_using_data_list_next(t))
    {
	printf(" %p", t->pointer_to_T);
    }
    printf("\n\n");
	
    /* Here, I need to find the task I can do with the data already in memory + the new data A and B.
     * It can also be the task using A and B.
     * TODO: Add first the one using A and B then alterning between task from A and from B.
     */
    for (i = 0; i < Ndifferent_data_type; i++)
    {
	for (t = task_using_data_list_begin(handle_popped[i]->sched_data); t != task_using_data_list_end(handle_popped[i]->sched_data); t = task_using_data_list_next(t))
	{
	    /* I put it at false if at least one data is missing. */
	    data_available = true; 
	    handle_popped_task = true;
	    printf("Task %p use %p.\n", t->pointer_to_T, handle_popped[i]);	
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
		    
		    if (!starpu_data_is_on_node(STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle), current_gpu))
		    {
			printf("Data %p is not on memory nor is popped.\n", STARPU_TASK_GET_HANDLE(t->pointer_to_T, next_handle)); 
			data_available = false;
			break;
		    }
		}
	    }
	    if (data_available == true)
	    {
		printf("Pushing %p in the package.\n", t->pointer_to_T);
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
    printf("\n");
}

/* Erase a task from the main task list.
 * Also erase pointer in the data.
 */
void erase_task_and_data_pointer (struct starpu_task *task, struct starpu_task_list *l)
{
    int j = 0;
    struct pointer_in_task *pt = task->sched_data;
    
    for (j = 0; j < STARPU_TASK_GET_NBUFFERS(task); j++)
    {
	task_using_data_list_erase(pt->pointer_to_D[j]->sched_data, pt->tud[j]);
	print_task_using_data(pt->pointer_to_D[j]);
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
	    printf("Oops, task %p got refused.\n", task);
	    
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

struct starpu_sched_component *starpu_sched_component_dynamic_outer_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_outer");
	srandom(time(NULL));
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
	/* Creating as much package as there are GPUs. */
	for (i = 0; i < Ngpu - 1; i++)
	{
	    dynamic_outer_insertion(data->p);
	}
	data->p->first_link = data->p->temp_pointer_1;

	component->data = data;
	//~ component->do_schedule = dynamic_outer_do_schedule;
	component->push_task = dynamic_outer_push_task;
	component->pull_task = dynamic_outer_pull_task;
	component->can_push = dynamic_outer_can_push;
	component->can_pull = dynamic_outer_can_pull;
	return component;
}

static void initialize_dynamic_outer_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_dynamic_outer_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			//~ STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
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
