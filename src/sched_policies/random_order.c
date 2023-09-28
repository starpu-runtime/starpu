/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

#include <starpu.h>
#include <starpu_sched_component.h>

#include <core/task.h>
#include <sched_policies/prio_deque.h>
#include <sched_policies/helper_mct.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include <common/list.h>

#include <stdlib.h>
#include <time.h>
#include <float.h>
#define PRINTF /* O or 1 */

/* Structure used to store all the variable we need and the tasks of each package. Each link is a package */
struct my_list
{
	int package_nb_data;
	//~ int nb_task_in_sub_list;
	int index_package; /* Used to write in Data_coordinates.txt and keep track of the initial index of the package */
	//~ starpu_data_handle_t * package_data; /* List of all the data in the packages. We don't put two times the duplicates */
	struct starpu_task_list sub_list; /* The list containing the tasks */
	struct my_list *next;
};

/* Structure used to access the struct my_list. There are also task's list */
struct random_order_sched_data
{
	struct starpu_task_list popped_task_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list random_list; /* List used to store all the tasks at the beginning of the pull_task function */
	struct starpu_task_list list_if_fifo_full; /* List used if the fifo list is not empty. It means that task from the last iteration haven't been pushed, thus we need to pop task from this list */

	/* All the pointer use to navigate through the linked list */
	struct my_list *temp_pointer_1;
	struct my_list *temp_pointer_2;
	struct my_list *temp_pointer_3;
	struct my_list *first_link; /* Pointer that we will use to point on the first link of the linked list */

	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

/* Put a link at the beginning of the linked list */
static void random_order_insertion(struct random_order_sched_data *a)
{
	struct my_list *new = malloc(sizeof(*new)); /* Creation of a new link */
	starpu_task_list_init(&new->sub_list);
	new->next = a->temp_pointer_1;
	a->temp_pointer_1 = new;
}

/* Delete all the empty packages */
static struct my_list* random_order_delete_link(struct random_order_sched_data* a)
{
	while (a->first_link != NULL && a->first_link->package_nb_data == 0)
	{
		a->temp_pointer_1 = a->first_link;
		a->first_link = a->first_link->next;
		free(a->temp_pointer_1);
	}
	if (a->first_link != NULL)
	{
		a->temp_pointer_2 = a->first_link;
		a->temp_pointer_3 = a->first_link->next;
		while (a->temp_pointer_3 != NULL)
		{
			while (a->temp_pointer_3 != NULL && a->temp_pointer_3->package_nb_data == 0)
			{
				a->temp_pointer_1 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
				a->temp_pointer_2->next = a->temp_pointer_3;
				free(a->temp_pointer_1);
			}
			if (a->temp_pointer_3 != NULL)
			{
				a->temp_pointer_2 = a->temp_pointer_3;
				a->temp_pointer_3 = a->temp_pointer_3->next;
			}
		}
	}
	return a->first_link;
}

/* Pushing the tasks */
static int random_order_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct random_order_sched_data *data = component->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	/* Tell below that they can now pull */
	component->can_pull(component);
	return 0;
}

/* The function that sort the tasks in packages */
static struct starpu_task *random_order_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	(void)to;
	struct random_order_sched_data *data = component->data;

	int random_number = 0;
	struct starpu_task *task1 = NULL;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;

	int NT = 0; int i = 0; int link_index = 0; int do_not_add_more = 0;

	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	if (!starpu_task_list_empty(&data->list_if_fifo_full))
	{
		task1 = starpu_task_list_pop_back(&data->list_if_fifo_full);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return task1;
	}
	if (starpu_task_list_empty(&data->random_list))
	{
		if (!starpu_task_list_empty(&data->sched_list))
		{
			time_t start, end; time(&start);
			while (!starpu_task_list_empty(&data->sched_list))
			{
				task1 = starpu_task_list_pop_front(&data->sched_list);
				NT++;
				if (starpu_get_env_number_default("PRINTF",0) == 1) printf("%p\n",task1);
				starpu_task_list_push_back(&data->popped_task_list,task1);
			}
			if (starpu_get_env_number_default("PRINTF",0) == 1) printf("%d task(s) have been pulled\n",NT);

			temp_task_1  = starpu_task_list_begin(&data->popped_task_list);
			//~ data->temp_pointer_1->package_data = malloc(STARPU_TASK_GET_NBUFFERS(temp_task_1)*sizeof(data->temp_pointer_1->package_data[0]));
			/* One task == one link in the linked list */
			do_not_add_more = NT - 1;
			for (temp_task_1  = starpu_task_list_begin(&data->popped_task_list); temp_task_1 != starpu_task_list_end(&data->popped_task_list); temp_task_1  = temp_task_2)
			{
				//~ printf("ok0.5\n");
				temp_task_2 = starpu_task_list_next(temp_task_1);
				temp_task_1 = starpu_task_list_pop_front(&data->popped_task_list);
				//~ printf("ok0.6\n");
				data->temp_pointer_1->package_nb_data = 1;
				//~ printf("ok0.7\n");
				/* We sort our data in the packages */
				/* Pushing the task and the number of the package in the package*/
				starpu_task_list_push_back(&data->temp_pointer_1->sub_list,temp_task_1);
				//~ printf("ok0.8\n");
				data->temp_pointer_1->index_package = link_index;
				/* Initialization of the lists last_packages */
				//~ printf("ok1\n");
				link_index++;
				//~ data->temp_pointer_1->nb_task_in_sub_list ++;

				if(do_not_add_more != 0)
				{
					random_order_insertion(data);
				}
				do_not_add_more--;
			}
			data->first_link = data->temp_pointer_1;
			int temp_NT = NT;
			for (i = 0; i < temp_NT; i++)
			{
				data->temp_pointer_1 = data->first_link;
				random_number = rand()%NT;
				//~ printf("Il y a %d tâche, random = %d\n",NT,random_number);
				while (random_number != 0)
				{
					data->temp_pointer_1 = data->temp_pointer_1->next;
					random_number--;
				}
				data->temp_pointer_1->package_nb_data = 0;
				starpu_task_list_push_back(&data->random_list,starpu_task_list_pop_front(&data->temp_pointer_1->sub_list));
				data->temp_pointer_1 = random_order_delete_link(data);
				NT--;
				//~ task1 = starpu_task_list_pop_front(&data->temp_pointer_1->sub_list);
			}
			//~ free(&data->temp_pointer_1->package_nb_data);
			//~ free(data);
			//~ random_order_free(data);

			time(&end);
			int time_taken = end - start;
			if (starpu_get_env_number_default("PRINTF",0) == 1) printf("Temps d'exec : %d secondes\n",time_taken);
			FILE *f_time = fopen("Output_maxime/Execution_time_raw.txt","a");
			fprintf(f_time,"%d\n",time_taken);
			fclose(f_time);

			task1 = starpu_task_list_pop_front(&data->random_list);
			//~ free(data->temp_pointer_1->package_nb_data);
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			if (starpu_get_env_number_default("PRINTF",0) == 1) printf("Task %p is getting out of pull_task\n",task1);
			return task1;
		}
		else
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
			//~ if (starpu_get_env_number_default("PRINTF",0) == 1) printf("Task %p is getting out of pull_task\n",task1);
			return task1;
		}
	}
	else
	{
		//~ task1 = starpu_task_list_pop_front(&data->popped_task_list);
		task1 = starpu_task_list_pop_front(&data->random_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		if (starpu_get_env_number_default("PRINTF",0) == 1) printf("Task %p is getting out of pull_task\n",task1);
		return task1;
	}
}

static int random_order_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct random_order_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//~ if (starpu_get_env_number_default("PRINTF",0) == 1) { fprintf(stderr, "oops, task %p got refused\n", task); }
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

static int random_order_can_pull(struct starpu_sched_component * component)
{
	return starpu_sched_component_can_pull(component);
}

struct starpu_sched_component *starpu_sched_component_random_order_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	starpu_srand48(starpu_get_env_number_default("SEED", 0));
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "random_order");

	struct random_order_sched_data *data;
	_STARPU_CALLOC(data, 1, sizeof(*data));

	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->list_if_fifo_full);
	starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&data->random_list);

	struct my_list *my_data = malloc(sizeof(*my_data));
	my_data->next = NULL;
	starpu_task_list_init(&my_data->sub_list);
	data->temp_pointer_1 = my_data;

	component->data = data;
	component->push_task = random_order_push_task;
	component->pull_task = random_order_pull_task;
	component->can_push = random_order_can_push;
	component->can_pull = random_order_can_pull;

	return component;
}

static void initialize_random_order_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_random_order_create, NULL,
							   STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
							   STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
							   STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_random_order_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_random_order_policy =
{
	.init_sched = initialize_random_order_center_policy,
	.deinit_sched = deinitialize_random_order_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.policy_name = "random_order",
	.policy_description = "Description",
	.worker_type = STARPU_WORKER_LIST,
};
