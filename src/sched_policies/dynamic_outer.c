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
 */

#include <schedulers/HFP.h>
#include "helper_mct.h"

/* Structure used to acces the struct my_list. There are also task's list */
//~ struct dynamic_outer_sched_data
//~ {
	//~ struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	//~ struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	
	//~ /* All the pointer use to navigate through the linked list */
	//~ struct my_list *temp_pointer_1;
	//~ struct my_list *temp_pointer_2;
	//~ struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	
	//~ struct starpu_task_list sched_list;
     	//~ starpu_pthread_mutex_t policy_mutex;
//~ };

/* Pushing the tasks */		
static int dynamic_outer_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
    struct HFP_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
    starpu_task_list_push_front(&data->sched_list, task);
    starpu_push_task_end(task);
    STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
    /* Tell below that they can now pull */
    component->can_pull(component);
    return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *dynamic_outer_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
    struct HFP_sched_data *data = component->data;	
    if (do_schedule_done == true)
    {
	int i = 0;
	struct starpu_task *task = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		
	/* Getting on the right GPU's package */
	data->p->temp_pointer_1 = data->p->first_link;
	if (data->p->temp_pointer_1->next != NULL) 
	{
	    for (i = 0; i < Ngpu; i++) 
	    {
		if (to == component->children[i]) 
		{
		    break;
		}
		else 
		{
		    data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
		}
	    }
	}
	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list)) 
	{
	    task = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list); 
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
	task = starpu_task_list_pop_front(&data->p->temp_pointer_1->sub_list);
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

static int dynamic_outer_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
    struct HFP_sched_data *data = component->data;
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
		    data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
		}
	    }
	    starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
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

static int dynamic_outer_can_pull(struct starpu_sched_component * component)
{
    return starpu_sched_component_can_pull(component);
}

static void dynamic_outer_do_schedule(struct starpu_sched_component *component)
{	
    struct HFP_sched_data *data = component->data;
    struct starpu_task *task = NULL;
    while (!starpu_task_list_empty(&data->sched_list)) 
    {
	task = starpu_task_list_pop_front(&data->sched_list);
	printf("Tâche %p, %d donnée(s) : ",task, STARPU_TASK_GET_NBUFFERS(task));
	starpu_task_list_push_back(&data->p->temp_pointer_1->sub_list, task);
	do_schedule_done = true;
    }
}

struct starpu_sched_component *starpu_sched_component_dynamic_outer_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "dynamic_outer");
	
	Ngpu = get_number_GPU();
	do_schedule_done = false;	
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
	data->p->temp_pointer_1->nb_task_in_sub_list = 0;
	data->p->temp_pointer_1->expected_time_pulled_out = 0;
	
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
