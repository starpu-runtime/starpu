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

/* A remplir
 */

#include <stdlib.h>
#include <time.h>
#include <starpu.h>
#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include "core/task.h"
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include "starpu_stdlib.h"
#include "common/list.h"
#define PRINTF /* O or 1 */

/* Structure used to acces the struct my_list. There are also task's list */
struct a_remplir_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */
	
	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *temp_pointer_2;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

/* Pushing the tasks */		
static int a_remplir_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct a_remplir_sched_data *data = component->data;
    STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *a_remplir_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct a_remplir_sched_data *data = component->data;

	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;
	//~ struct starpu_task *task2 = NULL;
 
	int NT = 0;
		
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->list_if_fifo_full)) {
		task1 = starpu_task_list_pop_back(&data->list_if_fifo_full); 
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		return task1;
	}
	/* If the linked list is empty, we can pull more tasks */
	//~ if ((data->temp_pointer_1->next == NULL) && (starpu_task_list_empty(&data->temp_pointer_1->sub_list))) {
	if (starpu_task_list_empty(&data->popped_task_list)) {
		if (!starpu_task_list_empty(&data->sched_list)) {
			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list)) {				
				task1 = starpu_task_list_pop_front(&data->sched_list);
				NT++;
				starpu_task_list_push_back(&data->popped_task_list,task1);
			} 		
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("%d task(s) have been pulled\n",NT); }
			
			/* Write here */
			
			task1 = starpu_task_list_pop_front(&data->popped_task_list);
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			return task1;
		}
		else {
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
			return task1; 
		}
	}
	else { 
		task1 = starpu_task_list_pop_front(&data->popped_task_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (starpu_get_env_number_default("PRINTF",0) == 1) { printf("Task %p is getting out of pull_task\n",task1); }
		return task1;
	}
}

static int a_remplir_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct a_remplir_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, task %p got refused\n", task); }
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->list_if_fifo_full, task);
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

static int a_remplir_can_pull(struct starpu_sched_component * component)
{
	struct a_remplir_sched_data *data = component->data;
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_a_remplir_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	//~ srandom(time(0)); /* For the random selection in ALGO 4 */
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "a_remplir");
	
	struct a_remplir_sched_data *data;
	//~ struct my_list *my_data = malloc(sizeof(*my_data));
	_STARPU_MALLOC(data, sizeof(*data));
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	//~ starpu_task_list_init(&my_data->sub_list);
 
	//~ my_data->next = NULL;
	//~ data->temp_pointer_1 = my_data;
	
	component->data = data;
	component->push_task = a_remplir_push_task;
	component->pull_task = a_remplir_pull_task;
	component->can_push = a_remplir_can_push;
	component->can_pull = a_remplir_can_pull;

	return component;
}

static void initialize_a_remplir_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_a_remplir_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_a_remplir_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_a_remplir_policy =
{
	.init_sched = initialize_a_remplir_center_policy,
	.deinit_sched = deinitialize_a_remplir_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "a_remplir",
	.policy_description = "Description",
	.worker_type = STARPU_WORKER_LIST,
};
