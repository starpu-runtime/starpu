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

/* Building a maximum spanning tree from the set of tasks using Prim's algorithm.
 * The task processing order is then the order in which tasks are added to the tree.
 * hMETIS can be used before executing MST on each package with the environemment variable HMETIS=1.
 */

#include <datawizard/memory_nodes.h>
#include <sched_policies/darts.h>
#include <sched_policies/HFP.h>
#include <sched_policies/sched_visu.h>

static int _nb_gpus;

static int mst_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{
	struct _starpu_HFP_sched_data *data = component->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_task_list_push_front(&data->sched_list, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	component->can_pull(component);
	return 0;
}

static struct starpu_task_list mst(struct starpu_task_list task_list, int number_task)
{
	struct starpu_task_list SIGMA;
	starpu_task_list_init(&SIGMA);
	int i = 0;
	int count = 0;
	int j = 0;
	int v = 0;
	unsigned int i_bis = 0;
	unsigned int j_bis = 0;
	int tab_runner = 0;
	struct starpu_task *temp_task_1 = NULL;
	struct starpu_task *temp_task_2 = NULL;
	int matrice_adjacence[number_task][number_task];
	for (i = 0; i < number_task; i++)
	{
		for (j = 0; j < number_task; j++)
		{
			matrice_adjacence[i][j] = 0;
		}
	}
	temp_task_1  = starpu_task_list_begin(&task_list);
	temp_task_2  = starpu_task_list_begin(&task_list);
	temp_task_2  = starpu_task_list_next(temp_task_2);
	/* Building the adjacency matrix */
	for (i = 0; i < number_task; i++)
	{
		for (j = i + 1; j < number_task; j++)
		{
			for (i_bis = 0; i_bis < STARPU_TASK_GET_NBUFFERS(temp_task_1); i_bis++)
			{
				for (j_bis = 0; j_bis < STARPU_TASK_GET_NBUFFERS(temp_task_2); j_bis++)
				{
					if (STARPU_TASK_GET_HANDLE(temp_task_1, i_bis) == STARPU_TASK_GET_HANDLE(temp_task_2, j_bis))
					{
						matrice_adjacence[i][j]++;
					}
				}
			}
			temp_task_2  = starpu_task_list_next(temp_task_2);
		}
		temp_task_1  = starpu_task_list_next(temp_task_1);
		temp_task_2 = temp_task_1;
		if (i + 1 != number_task)
		{
			temp_task_2  = starpu_task_list_next(temp_task_2);
		}
	}

	/* Printing the adjacency matrix */
	_sched_visu_print_matrix((int **)matrice_adjacence, number_task, number_task, "Matrice d'adjacence :\n");

	/* Struct of packages to have one task by package and thus being able to number each task.
	 * We need to number them to recognize them later on. */
	struct _starpu_HFP_my_list *temp_sub_list = malloc(sizeof(*temp_sub_list));
	struct _starpu_HFP_paquets *temp_paquets = malloc(sizeof(*temp_paquets));
	starpu_task_list_init(&temp_sub_list->sub_list);
	temp_sub_list->next = NULL;
	temp_paquets->temp_pointer_1 = temp_sub_list;
	temp_paquets->first_link = temp_paquets->temp_pointer_1;
	int do_not_add_more = 0;
	while (!starpu_task_list_empty(&task_list))
	{
		starpu_task_list_push_back(&temp_paquets->temp_pointer_1->sub_list, starpu_task_list_pop_front(&task_list));
		temp_paquets->temp_pointer_1->index_package = do_not_add_more;
		if (do_not_add_more != number_task - 1)
		{
			_starpu_HFP_insertion(temp_paquets);
		}
		do_not_add_more++;
	}
	temp_paquets->first_link = temp_paquets->temp_pointer_1;

	/* Start of the MST algorithm */
	// Key values used to pick minimum weight edge in cut
	int key[number_task];
	// To represent set of vertices included in MST
	bool mstSet[number_task];
	int tab_SIGMA[number_task];
	// Initialize all keys as 0
	for (i = 0; i < number_task; i++)
	{
		key[i] = 0, mstSet[i] = false;
	}

	// Always include first 1st vertex in MST.
	// Make key 0 so that this vertex is picked as first vertex.
	key[0] = 1;
	for (count = 0; count < number_task - 1; count++)
	{
		// Pick the minimum key vertex from the
		// set of vertices not yet included in MST
		int max = -1, max_index = 0;
		for (v = 0; v < number_task; v++)
			if (mstSet[v] == false && key[v] > max)
				max = key[v], max_index = v;

		int u = max_index;
		// Add the picked vertex to the MST Set
		mstSet[u] = true;
		tab_SIGMA[tab_runner] = u;
		tab_runner++;

		// Update key value and parent index of
		// the adjacent vertices of the picked vertex.
		// Consider only those vertices which are not
		// yet included in MST
		for (v = 0; v < number_task; v++)
			// matrice_adjacence[u][v] is non zero only for adjacent vertices of m
			// mstSet[v] is false for vertices not yet included in MST
			// Update the key only if graph[u][v] is greater than key[v]
			if (matrice_adjacence[u][v] && mstSet[v] == false && matrice_adjacence[u][v] > key[v])
				key[v] = matrice_adjacence[u][v];
	}
	/* End of the MST algorithm */

	/* Put last vertex in SIGMA */
	for (i = 0; i < number_task; i++)
	{
		if (mstSet[i] == false)
		{
			tab_SIGMA[number_task - 1] = i;
		}
	}

	_sched_visu_print_vector(tab_SIGMA, number_task, "tab_SIGMA[i] : ");

	i = 0;

	/* Filling our task list */
	temp_paquets->temp_pointer_1 = temp_paquets->first_link;
	while (i != number_task)
	{
		if (tab_SIGMA[i] == temp_paquets->temp_pointer_1->index_package)
		{
			starpu_task_list_push_back(&SIGMA, starpu_task_list_pop_front(&temp_paquets->temp_pointer_1->sub_list));
			i++;
			temp_paquets->temp_pointer_1 = temp_paquets->first_link;
		}
		else
		{
			temp_paquets->temp_pointer_1 = temp_paquets->temp_pointer_1->next;
		}
	}

	//Belady
	//~ if (starpu_get_env_number_default("BELADY",0) == 1)
	//~ {
		//~ get_ordre_utilisation_donnee_mst(data, NB_TOTAL_DONNEES);
	//~ }

	return SIGMA;
}

static struct starpu_task *mst_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct _starpu_HFP_sched_data *data = component->data;
	int i = 0;
	struct starpu_task *task = NULL;

	if (_starpu_HFP_do_schedule_done == true)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	/* If one or more task have been refused */
	data->p->temp_pointer_1 = data->p->first_link;
	if (data->p->temp_pointer_1->next != NULL)
	{
		for (i = 0; i < _nb_gpus; i++)
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
	if (!starpu_task_list_empty(&data->p->temp_pointer_1->refused_fifo_list))
	{
		task = starpu_task_list_pop_back(&data->p->temp_pointer_1->refused_fifo_list);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		_STARPU_SCHED_PRINT("Task %p is getting out of pull_task from fifo refused list on gpu %p\n",task, to);
		return task;
	}
	/* If the linked list is empty, we can pull more tasks */
	if (_starpu_HFP_is_empty(data->p->first_link) == true)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return NULL;
	}
	else
	{
		task = _starpu_HFP_get_task_to_return(component, to, data->p, _nb_gpus);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		_STARPU_SCHED_PRINT("Task %p is getting out of pull_task from gpu %p\n", task, to);
		return task;
	}
}
	return NULL;
}

static int _get_number_GPU()
{
	int return_value = starpu_memory_nodes_get_count_by_kind(STARPU_CUDA_RAM);

	if (return_value == 0) /* We are not using GPUs so we are in an out-of-core case using CPUs. Need to return 1. If I want to deal with GPUs AND CPUs we need to adpt this function to return NGPU + 1 */
	{
		return 1;
	}

	return return_value;
}

static int mst_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	int i = 0;
	struct _starpu_HFP_sched_data *data = component->data;
	int didwork = 0;

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);
	if (task)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		data->p->temp_pointer_1 = data->p->first_link;
		int nb_gpu = _get_number_GPU();
		if (data->p->temp_pointer_1->next == NULL)
		{
			starpu_task_list_push_back(&data->p->temp_pointer_1->refused_fifo_list, task);
		}
		else
		{
			for (i = 0; i < nb_gpu; i++)
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
		}
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

static int mst_can_pull(struct starpu_sched_component * component)
{
	return starpu_sched_component_can_pull(component);
}

static void mst_do_schedule(struct starpu_sched_component *component)
{
	int i = 0;
	struct starpu_task_list temp_task_list;
	starpu_task_list_init(&temp_task_list);
	int NB_TOTAL_DONNEES = 0;
	struct _starpu_HFP_sched_data *data = component->data;
	struct starpu_task *task = NULL;
	_starpu_HFP_NT = 0;
 	int number_of_package_to_build = _get_number_GPU();
	_starpu_HFP_GPU_RAM_M = (starpu_memory_get_total(starpu_worker_get_memory_node(starpu_bitmap_first(&component->workers_in_ctx))));

	/* If the linked list is empty, we can pull more tasks */
	if (_starpu_HFP_is_empty(data->p->first_link) == true)
	{
		if (!starpu_task_list_empty(&data->sched_list))
		{
			_starpu_HFP_appli = starpu_task_get_name(starpu_task_list_begin(&data->sched_list));
			if (_starpu_HFP_hmetis != 0)
			{
				_starpu_hmetis_scheduling(data->p, &data->sched_list, number_of_package_to_build);

				/* Apply mst on each package */
				data->p->temp_pointer_1 = data->p->first_link;
				for (i = 0; i < number_of_package_to_build; i++)
				{
					data->p->temp_pointer_1->sub_list = mst(data->p->temp_pointer_1->sub_list, data->p->temp_pointer_1->nb_task_in_sub_list);
					data->p->temp_pointer_1 = data->p->temp_pointer_1->next;
				}
				_starpu_HFP_do_schedule_done = true;
				return;
			}

			/* Pulling all tasks and counting them */
			while (!starpu_task_list_empty(&data->sched_list))
			{
				task = starpu_task_list_pop_front(&data->sched_list);
				NB_TOTAL_DONNEES+=STARPU_TASK_GET_NBUFFERS(task);
				_starpu_HFP_NT++;
				_STARPU_SCHED_PRINT("%p\n",task);
				starpu_task_list_push_back(&temp_task_list, task);
			}
			_STARPU_SCHED_PRINT("%d task(s) have been pulled\n", _starpu_HFP_NT);
			//~ task = starpu_task_list_begin(&data->popped_task_list);
			//~ printf("tache %p\n", task);
			/* Apply mst on the task list */
			//~ data->p->temp_pointer_1->sub_list = mst(data->popped_task_list, NT, GPU_RAM_M);
			data->p->temp_pointer_1->sub_list = mst(temp_task_list, _starpu_HFP_NT);

			_starpu_HFP_do_schedule_done = true;
		}
	}
}

struct starpu_sched_component *starpu_sched_component_mst_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	_starpu_HFP_hmetis = starpu_get_env_number_default("HMETIS", 0);

	struct starpu_sched_component *component = starpu_sched_component_create(tree, "mst");

	struct _starpu_HFP_sched_data *data;
	struct _starpu_HFP_my_list *my_data = malloc(sizeof(*my_data));
	struct _starpu_HFP_paquets *paquets_data = malloc(sizeof(*paquets_data));
	_STARPU_MALLOC(data, sizeof(*data));

	_starpu_visu_init();

	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	starpu_task_list_init(&data->sched_list);
	//~ starpu_task_list_init(&data->popped_task_list);
	starpu_task_list_init(&my_data->sub_list);
	starpu_task_list_init(&my_data->refused_fifo_list);

	my_data->next = NULL;
	paquets_data->temp_pointer_1 = my_data;
	paquets_data->first_link = paquets_data->temp_pointer_1;
	data->p = paquets_data;
	data->p->temp_pointer_1->nb_task_in_sub_list = 0;
	data->p->temp_pointer_1->expected_time_pulled_out = 0;

	component->data = data;
	component->do_schedule = mst_do_schedule;
	component->push_task = mst_push_task;
	component->pull_task = mst_pull_task;
	component->can_push = mst_can_push;
	component->can_pull = mst_can_pull;

	return component;
}

static void initialize_mst_center_policy(unsigned sched_ctx_id)
{
	_nb_gpus = _get_number_GPU();
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_mst_create, NULL,
							   STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
							   STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
							   STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
							   STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_mst_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

static void get_current_tasks_mst(struct starpu_task *task, unsigned sci)
{
#ifdef PRINT_PYTHON
	task_currently_treated = task;
#endif
	starpu_sched_component_worker_pre_exec_hook(task,sci);
}

struct starpu_sched_policy _starpu_sched_mst_policy =
{
	.init_sched = initialize_mst_center_policy,
	.deinit_sched = deinitialize_mst_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.do_schedule = starpu_sched_tree_do_schedule,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = get_current_tasks_mst,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.policy_name = "mst",
	.policy_description = "Maximum Spanning Tree",
	.worker_type = STARPU_WORKER_LIST,
};
