/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2020       Maxime Gonthier
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

/* TODO : description
 * dynamic_outer_sched_data->sched_list = liste de tâches grand T
 */

#include <schedulers/HFP.h>
#include <schedulers/dynamic_outer.h>
#include "helper_mct.h"

/* Pushing the tasks */		
static int dynamic_outer_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
    struct dynamic_outer_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
    starpu_task_list_push_front(&data->sched_list, task);
    starpu_push_task_end(task);
    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    component->can_pull(component);
    return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *dynamic_outer_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    /* Step 8 : When a GPU ask for a task first check if initialization has been done.
     * If yes return the head of it package.
     * If the package is empty, go in do_schedule to pop 2 new data.
     */
     //TODO
     
    struct dynamic_outer_sched_data *data = component->data;	
    if (do_schedule_done == true)
    {
	int i = 0;
	struct starpu_task *task = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		
	/* Getting on the right GPU's package */
	data->p->pointer = data->p->first_link;
	if (data->p->pointer->next != NULL) 
	{
	    for (i = 0; i < Ngpu; i++) 
	    {
		if (to == component->children[i]) 
		{
		    break;
		}
		else 
		{
		    data->p->pointer = data->p->pointer->next;
		}
	    }
	}
	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->p->pointer->refused_fifo_list)) 
	{
	    task = starpu_task_list_pop_back(&data->p->pointer->refused_fifo_list); 
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    printf("Task %p is getting out of pull_task from fifo refused list on gpu %p\n", task, to);
	    return task;
	}
	/* If the linked list is empty */
	if (is_empty(data->p->first_link) == true) 
	{
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	    return NULL;
	}
	/* Else we take the next one in the package */
	//~ task = get_task_to_return(component, to, data->p, Ngpu);
	task = starpu_task_list_pop_front(&data->p->pointer->sub_list);
	printf("Task %p is getting out of pull_task from gpu %p\n", task, to);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	return task;
    }
    else
    {
	/* Do schedule not done yet */
	return NULL;
    }
}

static int dynamic_outer_can_push(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct dynamic_outer_sched_data *data = component->data;
    int didwork = 0;
    struct starpu_task *task;
    task = starpu_sched_component_pump_to(component, to, &didwork);
    if (task)
    {
	    if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, task %p got refused\n", task); }
	    /* If a task is refused I push it in the refused fifo list of the appropriate GPU's package.
	     * This list is lloked at first when a GPU is asking for a task so we don't break the planned order. */
	    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	    for (int i = 0; i < Ngpu; i++) 
	    {
		if (to == component->children[i]) 
		{
		    break;
		}
		else 
		{
		    data->p->pointer = data->p->pointer->next;
		}
	    }
	    starpu_task_list_push_back(&data->p->pointer->refused_fifo_list, task);
	    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    }
    else
    {
	/* Can I uncomment this part ? */
	//~ {
	    //~ if (didwork)
		//~ fprintf(stderr, "pushed some tasks to %p\n", to);
	    //~ else
		//~ fprintf(stderr, "I didn't have anything for %p\n", to);
	//~ }
    }
    /* There is room now */
    return didwork || starpu_sched_component_can_push(component, to);
}

void print_task_list(struct starpu_task_list *l, char *s)
{
    printf("%s :\n", s);
    for (struct starpu_task *task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
    {
	printf("%p\n", task);
    }
    printf("\n");
}

static int dynamic_outer_can_pull(struct starpu_sched_component *component)
{
    return starpu_sched_component_can_pull(component);
}

void initialize_task_list_using_data(struct starpu_task_list *l)
{
    int i = 0;
    
    for (struct starpu_task *task = starpu_task_list_begin(l); task != starpu_task_list_end(l); task = starpu_task_list_next(task))
    {
	printf("Sur la tâche %p, ", task);
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
	{
		//~ struct pointer_in_task_list *pt;
		//~ pt->pointer_to_A = 
		//~ pt->pointer_to_B = 
		//~ pt->pointer_to_cell = ???
		//~ task->sched_data = pt;
	    
	    /* Pointer toward the main task list in the handles. */
	    printf("sur la donnée %p, ", STARPU_TASK_GET_HANDLE(task, i));
	    if (STARPU_TASK_GET_HANDLE(task, i)->sched_data == NULL) 
	    {
		struct task_using_data_list *tl = task_using_data_list_new();
		struct task_using_data *e = task_using_data_new();
		e->pointer_to_T = task;
		task_using_data_list_push_front(tl, e);
		STARPU_TASK_GET_HANDLE(task, i)->sched_data = tl;
	    }
	    else
	    {
		struct task_using_data *e = task_using_data_new();
		e->pointer_to_T = task;
		task_using_data_list_push_front(STARPU_TASK_GET_HANDLE(task, i)->sched_data, e);
	    }
	    printf("contenu de sched_data dans le handle : ");
	    struct task_using_data *temp;
	    for(temp = task_using_data_list_begin(STARPU_TASK_GET_HANDLE(task, i)->sched_data); temp != task_using_data_list_end(STARPU_TASK_GET_HANDLE(task, i)->sched_data); temp  = task_using_data_list_next(temp))
	    {
		printf("%p	", temp->pointer_to_T);
	    }
	    printf("\n");
	}
    }
    printf("J'ai parcouru la liste de tâche complète dans initialize_task_list_using_data(struct starpu_task_list *l) pour ajouter chaque tâche dans une liste dans les handles. Complexité : O(NT)\n\n");
}

void randomize_task_list(struct dynamic_outer_sched_data *d)
{
    int random = 0;
    int i = 0;
    
    for (i = 0; i < NT; i++)
    {
	random = rand()%NT;
	while (random != 0)
	{
	    random--;
	    starpu_task_list_push_back(&d->sched_list, starpu_task_list_pop_front(&d->sched_list));
	}
	starpu_task_list_push_back(&d->popped_task_list, starpu_task_list_pop_front(&d->sched_list));
    }
    printf("J'ai parcouru la liste de tâche complète, puis la liste - 1 élément et ainsi de suite. Cela pour randomiser la liste de tâches initiale dans randomize_task_list. Complexité : O(NT^2)\n\n");
}

void initialization_dynamic_outer(struct starpu_sched_component *component)
{
    struct dynamic_outer_sched_data *data = component->data;
    
    /* Step 2 : Shuffling the task list */
    print_task_list(&data->sched_list, "Initial task list");
    randomize_task_list(data);
    print_task_list(&data->popped_task_list, "After shuffling");
    
    /* Step 3a : For each different data, create a list of task using this data. Using the field *sched_data of the handles.
     * I'm using the randomized list. I don't know if it changes something.
     */
    initialize_task_list_using_data(&data->popped_task_list);
    
    /* Step 3b : Add in each task a pointer to point toward the two (or three) data used by this task 
     * AND the cell in the main task list of the corresponding task.
     */
    //TODO
    
    /* Step 4 : Creating a data list for each GPU containing all the data they did not used yet. Initially it's all the different data.
     * One list per data type. It works only in 2D or 3D matrix multiplication for now.
     */
    /* I do it in initialize_task_list_using_data so I don't run through the list a 1000 times. */
    //TODO
    
}

static void dynamic_outer_do_schedule(struct starpu_sched_component *component)
{	
    struct dynamic_outer_sched_data *data = component->data;
    struct starpu_task *task = NULL;
    int i = 0;
    
    if (!starpu_task_list_empty(&data->sched_list) && initialization_dynamic_outer_done == false)
    {
	NT = starpu_task_list_size(&data->sched_list);
	printf("Il y a %d tâches\n\n", NT);
	initialization_dynamic_outer(component);
	initialization_dynamic_outer_done = true;
    }
    
    /* Step 5 : Pop a data from A and a data from B */
    //TODO
    
    /* Step 6 : Add to the corresponding package all the task we can do 
     * with A and/or B + the data already in memory of the corresponding GPU.
     */
    //TODO
    
    /* Step 7 : Delete from handle list (A and B (and C)) and T the tasks added in the package.
     */
    //TODO
     
    /* Temporary code just to test what I already coded */
    //~ struct task_using_data *temp;
    //~ task = starpu_task_list_begin(&data->popped_task_list);
    //~ temp = task_using_data_list_pop_front(STARPU_TASK_GET_HANDLE(task, 1)->sched_data);
    //~ starpu_task_list_push_back(&data->p->pointer->sub_list, &temp);
    //~ do_schedule_done = true;
     
    //~ while (!starpu_task_list_empty(&data->sched_list)) 
    //~ for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++) 

}

struct starpu_sched_component *starpu_sched_component_dynamic_outer_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_outer");
	
	Ngpu = get_number_GPU();
	NT = 0;
	do_schedule_done = false;
	initialization_dynamic_outer_done = false;	
	struct dynamic_outer_sched_data *data;
	struct dynamic_outer_packages_list *my_data = malloc(sizeof(*my_data));
	struct dynamic_outer_packages *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);
 	my_data->next = NULL;
	paquets_data->pointer = my_data;
	paquets_data->first_link = paquets_data->pointer;
	data->p = paquets_data;
	
	//~ struct control_task_using_di *d;
	//~ struct task_using_di *t_d = malloc(sizeof(*t_d));
	//~ starpu_task_list_init(&t_d->l);
	//~ t_d->next = NULL;
	//~ d->pointer = t_d;
	//~ d->first = d->pointer;
	
	component->data = data;
	component->do_schedule = dynamic_outer_do_schedule;
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
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "dynamic-outer",
	.policy_description = "Dynamic scheduler scheduling tasks whose data are in memory after loading 2 random data",
	.worker_type = STARPU_WORKER_LIST,
};
