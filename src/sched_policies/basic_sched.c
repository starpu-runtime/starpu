/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Ordonnanceur de base */

#include <starpu.h>
#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
//#include <starpu_task_list.h>
#include "core/task.h"
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>
#define NTASKS 5

#define MEMORY_AFFINITY

struct child_data {
	double expected_start;
	double predicted;
	double predicted_transfer;
	double expected_end;
	unsigned child;
};


static int compar(const void *_a, const void *_b)
{
	const struct child_data *a = _a;
	const struct child_data *b = _b;
	if (a->expected_end < b->expected_end)
		return -1;
	if (a->expected_end == b->expected_end)
		return 0;
	return 1;
}

//~ static int basic_push_task(struct starpu_sched_component * component, struct starpu_task * task)
//~ {
	//~ unsigned n = component->nchildren;
	//~ unsigned i;

	//~ printf("Test_basic_push_task\n");


	//~ /* See if it's a GEMM task */
	//~ const char *name = starpu_task_get_model_name(task);
	//~ //fprintf(stderr, "it's %s\n", name);

	//~ if (name && (!strcmp(name, "gemm") ||
		//~ !strcmp(name, "dgemm") ||
		//~ !strcmp(name, "sgemm") ||
		//~ !strcmp(name, "chol_model_22") ||
		//~ !strcmp(name, "starpu_dlu_lu_model_22") ||
		//~ !strcmp(name, "starpu_slu_lu_model_22")))
	//~ {
		//~ /* It's a GEMM, try to push to GPUs */

		//~ struct child_data child_data[n];

		//~ for (i = 0; i < n; i++)
		//~ {
			//~ child_data[i].expected_end = -1;
			//~ child_data[i].child = i;
		//~ }

		//~ /* Look at GPU availability time */
		//~ for (i = 0; i < n; i++)
		//~ {
			//~ struct starpu_sched_component *child = component->children[i];
			//~ double predicted;
			//~ if (starpu_sched_component_execute_preds(child, task, &predicted))
			//~ {
				//~ double expected_start;
				//~ child_data[i].expected_start =
					//~ expected_start = child->estimated_end(child);
				//~ child_data[i].predicted = predicted;
				//~ child_data[i].expected_end = expected_start 
					//~ + predicted;

//~ #ifdef MEMORY_AFFINITY
				//~ double predicted_transfer;
				//~ child_data[i].predicted_transfer =
					//~ predicted_transfer = starpu_sched_component_transfer_length(child, task);
				//~ child_data[i].expected_end += predicted_transfer;
//~ #endif
			//~ }
		//~ }

		//~ /* Sort by increasing expected end */
		//~ qsort(child_data, n, sizeof(*child_data), compar);

		//~ /* Try to push to the GPU with minimum availability time, to balance the load.  */
		//~ for (i = 0; i < n; i++)
		//~ {
			//~ if (child_data[i].expected_end != -1)
			//~ {
				//~ struct starpu_sched_component *child = component->children[child_data[i].child];

				//~ /* Note it in the task so that estimated_end() has it */
				//~ task->predicted = child_data[i].predicted;
				//~ task->predicted_transfer = child_data[i].predicted_transfer;

				//~ int ret = starpu_sched_component_push_task(component,child,task);
				//~ if (!ret)
					//~ /* Ok, this GPU took it */
					//~ return 0;
			//~ } 
		//~ }
	//~ }

	//~ int workerid;
	//~ /* It's not a GEMM, or no GPU wanted to take it, find somebody else */
	//~ for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    //~ workerid != -1;
	    //~ workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	//~ {
		//~ int nimpl;
		//~ for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		//~ {
			//~ if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   //~ || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			//~ {
				//~ for (i = 0; i < n; i++)
				//~ {
					//~ struct starpu_sched_component *child = component->children[i];
					//~ int idworker;
					//~ for(idworker = starpu_bitmap_first(component->children[i]->workers);
						//~ idworker != -1;
						//~ idworker = starpu_bitmap_next(component->children[i]->workers, idworker))
					//~ {
						//~ if (idworker == workerid)
						//~ {
							//~ if ((starpu_cpu_worker_get_count() == 0 ||
									//~ starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
							 //~ && (starpu_worker_can_execute_task(workerid,task,nimpl)
							   //~ || starpu_combined_worker_can_execute_task(workerid, task, nimpl)))
							//~ {
								//~ int ret = starpu_sched_component_push_task(component,child,task);
								//~ if (!ret)
									//~ return 0;
							//~ }
						//~ }
					//~ }
				//~ }
			//~ }
		//~ }
	//~ }
	//~ /* FIFOs are full */
	//~ return 1;
//~ }

//HEFT
//~ struct _starpu_basic_data
//~ {
	//~ struct _starpu_prio_deque prio;
	//~ starpu_pthread_mutex_t mutex;
	//~ struct _starpu_mct_data *mct_data;
	//~ struct starpu_task_list sched_list;
     	//~ //starpu_pthread_mutex_t policy_mutex;
//~ };

//~ struct _starpu_basic_data_2
//~ {
	//~ struct starpu_task_list sched_list;
     	//~ starpu_pthread_mutex_t policy_mutex;
//~ };

//~ static int basic_progress_one(struct starpu_sched_component *component)
//~ {
	//~ struct _starpu_basic_data * data = component->data;
	//~ //struct basic_sched_data *data = component->data;
	//~ starpu_pthread_mutex_t * mutex = &data->mutex;
	
	
	//~ struct _starpu_prio_deque * prio = &data->prio;
	//~ struct starpu_task * (tasks[NTASKS]);
	//~ unsigned ntasks = 0;

	//~ STARPU_COMPONENT_MUTEX_LOCK(mutex);
	//~ tasks[0] = _starpu_prio_deque_pop_task(prio);
	//~ if (tasks[0])
	//~ {
		//~ int priority = tasks[0]->priority;
		//~ /* Try to look at NTASKS from the queue */
		//~ for (ntasks = 1; ntasks < NTASKS; ntasks++)
		//~ {
			//~ tasks[ntasks] = _starpu_prio_deque_highest_task(prio);
 			//~ if (!tasks[ntasks] || tasks[ntasks]->priority < priority)
 				//~ break;
 			//~ _starpu_prio_deque_pop_task(prio);		
		//~ }
	//~ }
	//~ STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	//~ if (!ntasks)
	//~ {
		//~ return 1;
	//~ }

	//~ {
		//~ struct _starpu_mct_data * d = data->mct_data;
		//~ struct starpu_sched_component * best_component;
		//~ unsigned n;

		//~ /* Estimated task duration for each child */
		//~ double estimated_lengths[component->nchildren * ntasks];
		//~ /* Estimated transfer duration for each child */
		//~ double estimated_transfer_length[component->nchildren * ntasks];
		//~ /* Estimated transfer+task termination for each child */
		//~ double estimated_ends_with_task[component->nchildren * ntasks];

		//~ /* Minimum transfer+task termination on all children */
		//~ double min_exp_end_with_task[ntasks];
		//~ /* Maximum transfer+task termination on all children */
		//~ double max_exp_end_with_task[ntasks];

		//~ unsigned suitable_components[component->nchildren * ntasks];

		//~ unsigned nsuitable_components[ntasks];

		//~ /* Estimate durations */
		//~ for (n = 0; n < ntasks; n++)
		//~ {
			//~ unsigned offset = component->nchildren * n;

			//~ nsuitable_components[n] = starpu_mct_compute_execution_times(component, tasks[n],
					//~ estimated_lengths + offset,
					//~ estimated_transfer_length + offset,
					//~ suitable_components + offset);

			//~ starpu_mct_compute_expected_times(component, tasks[n],
					//~ estimated_lengths + offset,
					//~ estimated_transfer_length + offset,
					//~ estimated_ends_with_task + offset,
					//~ &min_exp_end_with_task[n], &max_exp_end_with_task[n],
							  //~ suitable_components + offset, nsuitable_components[n]);
		//~ }

		//~ int best_task = 0;
		//~ double max_benefit = 0;

		//~ /* Find the task which provides the most computation time benefit */
		//~ for (n = 0; n < ntasks; n++)
		//~ {
			//~ double benefit = max_exp_end_with_task[n] - min_exp_end_with_task[n];
			//~ if (max_benefit < benefit)
			//~ {
				//~ max_benefit = benefit;
				//~ best_task = n;
			//~ }
		//~ }

		//~ STARPU_ASSERT(best_task >= 0);

		//~ /* Push back the other tasks */
		//~ STARPU_COMPONENT_MUTEX_LOCK(mutex);
		//~ for (n = ntasks - 1; n < ntasks; n--)
			//~ if ((int) n != best_task)
				//~ _starpu_prio_deque_push_front_task(prio, tasks[n]);
		//~ STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

		//~ unsigned offset = component->nchildren * best_task;

		//~ int best_icomponent = starpu_mct_get_best_component(d, tasks[best_task], estimated_lengths + offset, estimated_transfer_length + offset, estimated_ends_with_task + offset, min_exp_end_with_task[best_task], max_exp_end_with_task[best_task], suitable_components + offset, nsuitable_components[best_task]);

		//~ STARPU_ASSERT(best_icomponent != -1);
		//~ best_component = component->children[best_icomponent];

		//~ if(starpu_sched_component_is_worker(best_component))
		//~ {
			//~ best_component->can_pull(best_component);
			//~ return 1;
		//~ }

		//~ starpu_sched_task_break(tasks[best_task]);
		//~ int ret = starpu_sched_component_push_task(component, best_component, tasks[best_task]);

		//~ if (ret)
		//~ {
			//~ /* Could not push to child actually, push that one back too */
			//~ STARPU_COMPONENT_MUTEX_LOCK(mutex);
			//~ _starpu_prio_deque_push_front_task(prio, tasks[best_task]);
			//~ STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
			//~ return 1;
		//~ }
		//~ else
			//~ return 0;
	//~ }
//~ }

//~ int starpu_sched_component_is_basic(struct starpu_sched_component * component)
//~ {
	//~ return component->push_task == basic_push_task;
//~ }

//~ static void basic_progress(struct starpu_sched_component *component)
//~ {
	//~ STARPU_ASSERT(component && starpu_sched_component_is_basic(component));
	//~ while (!basic_progress_one(component));
//~ }

//~ static int basic_can_push(struct starpu_sched_component *component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
//~ {
	//~ basic_progress(component);
	//~ int ret = 0;
	//~ unsigned j;
	//~ for(j=0; j < component->nparents; j++)
	//~ {
		//~ if(component->parents[j] == NULL)
			//~ continue;
		//~ else
		//~ {
			//~ ret = component->parents[j]->can_push(component->parents[j], component);
			//~ if(ret)
				//~ break;
		//~ }
	//~ }
	//~ return ret;
//~ }
//FIN HEFT

//Dummy
struct basic_sched_data
{
	int mem;
	struct starpu_task_list tache_pop;
	struct basic_sched_data *next;
	
	struct starpu_task_list sched_list;
     	starpu_pthread_mutex_t policy_mutex;
};

static int basic_push_task(struct starpu_sched_component *component, struct starpu_task *task)
{	printf("Dans le push\n");
	struct basic_sched_data *data = component->data;
	//if (data->verbose)
		fprintf(stderr, "pushing task %p\n", task);

        STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);

	starpu_task_list_push_front(&data->sched_list, task);

	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	/* Tell below that they can now pull */
	component->can_pull(component);

	return 0;
}

static struct starpu_task *basic_pull_task(struct starpu_sched_component *component, struct starpu_sched_component *to)
{
	struct basic_sched_data *data = component->data;	
	printf("Debut de basic_pull_task\n");
	int i = 0; int j = 0; int nb_pop = 0; int temp_nb_pop = 0; int tab_runner = 0; int max_donnees_commune = 0; int k = 0;
	
//Marche pas dans mon cas	
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_task_list_empty(&data->sched_list))
		return NULL;
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	struct starpu_task *task1 = NULL;
	//struct starpu_task *task2 = NULL;
	//struct starpu_task *task_temp_1 = NULL;
	//struct starpu_task *task_temp_2 = NULL;
	
	if (starpu_task_list_empty(&data->tache_pop)) {
		if (!starpu_task_list_empty(&data->sched_list)) {
			while (!starpu_task_list_empty(&data->sched_list)) {
				task1 = starpu_task_list_pop_back(&data->sched_list);
				printf("Pull la tâche : %p\n",task1);
				nb_pop++;
				printf("Il y a eu %d pop(s) \n",nb_pop);
				starpu_task_list_push_back(&data->tache_pop,task1);
			}
			if (nb_pop > 2) {
				struct starpu_task *task_tab [nb_pop];
				int *handles[nb_pop*3]; for (i = 0; i < nb_pop*3; i++) { handles[i] = 0; }
				for (i = 0; i < nb_pop; i++) {
					task1 = starpu_task_list_pop_front(&data->tache_pop);
					handles[tab_runner] = STARPU_TASK_GET_HANDLE(task1, 0);
					handles[tab_runner + 1] = STARPU_TASK_GET_HANDLE(task1, 1);
					handles[tab_runner + 2] = STARPU_TASK_GET_HANDLE(task1, 2);
					tab_runner += 3;
					starpu_task_list_push_back(&data->tache_pop,task1);
					//Remplissage dans un tableau
					//task_tab[i]
				}
				printf("Tableau des handles : \n");
				for (i = 0; i < nb_pop*3; i++) {
					printf("%p\n",handles[i]);
				}
				tab_runner = 0;
				int *matrice_donnees_commune[nb_pop][nb_pop]; for (i = 0; i < nb_pop; i++) { for (j = 0; j < nb_pop; j++) { matrice_donnees_commune[i][j] = 0; }}
				for (i = 0; i < nb_pop - 1; i++) {
					//Compare les HANDLE(tâches,i)
					//Compare par rapport à la tâche 1 puis la 2 au deuxième tour du for ci-dessus
					for (tab_runner = i+1; tab_runner < nb_pop; tab_runner++) {
						if (handles[i*3] == handles[tab_runner*3]) { matrice_donnees_commune[i][tab_runner] += 1; }
						if (handles[i*3 + 1] == handles[tab_runner*3 + 1]) { matrice_donnees_commune[i][tab_runner] += 1; }
						if (handles[i*3 + 2] == handles[tab_runner*3 + 2]) { matrice_donnees_commune[i][tab_runner] += 1; }
					}				
				}
				printf("Matrice de données communes remplite : \n");
				for (i = 0; i < nb_pop; i++) {
					for (j = 0; j < nb_pop; j++) {
						printf (" %d ",matrice_donnees_commune[i][j]);
					}
					printf("\n");
				}
				//met dans un tableau
				for (i = 0; i < nb_pop; i++) {
					task_tab[i] = starpu_task_list_pop_front(&data->tache_pop);
				}
				
				for (i = 0; i < nb_pop; i++) {
					if (task_tab[i] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[i]); task_tab[i] = 0; }
					for (j = i + 1; j< nb_pop; j++) {
						if (matrice_donnees_commune[i][j] == 4) {
							printf ("Une donnée en commun\n");
							if (task_tab[j] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[j]); task_tab[j] = 0; }
						}
					}
				}
				for (i = 0; i < nb_pop; i++) {
					if (task_tab[i] != 0) { starpu_task_list_push_back(&data->tache_pop,task_tab[i]); task_tab[i] = 0; }
				}
				task1 = starpu_task_list_pop_front(&data->tache_pop);
			}
			else {
				task1 = starpu_task_list_pop_front(&data->tache_pop);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return task1;
	}
	task1 = starpu_task_list_pop_front(&data->tache_pop);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	return task1;
}

static int basic_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	struct basic_sched_data *data = component->data;
	int didwork = 0;

	//if (data->verbose)
		fprintf(stderr, "tells me I can push to him\n");

	struct starpu_task *task;
	task = starpu_sched_component_pump_to(component, to, &didwork);

	if (task)
	{
		//if (data->verbose)
			fprintf(stderr, "oops, %p couldn't take our task %p \n", to, task);
		/* Oops, we couldn't push everything, put back this task */
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_task_list_push_back(&data->sched_list, task);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
	}
	else
	{
		//if (data->verbose)
		{
			if (didwork)
				fprintf(stderr, "pushed some tasks to %p\n", to);
			else
				fprintf(stderr, "I didn't have anything for %p\n", to);
		}
	}

	/* There is room now */
	return didwork || starpu_sched_component_can_push(component, to);
}

static int basic_can_pull(struct starpu_sched_component * component)
{
	struct basic_sched_data *data = component->data;

	//if (data->verbose)
		fprintf(stderr,"telling below they can pull\n");

	return starpu_sched_component_can_pull(component);
}
//Fin dummy

struct starpu_sched_component *starpu_sched_component_basic_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "basic");
	
	struct basic_sched_data *data;
	_STARPU_MALLOC(data, sizeof(*data)); //a pas oublier
	data->mem = 1;

	//_starpu_prio_deque_init(&data->prio);
	//STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	//data->mct_data = mct_data;
	
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	starpu_task_list_init(&data->sched_list);
	starpu_task_list_init(&data->tache_pop);
	//starpu_task_list_init(&data->temp_list);
	
	component->data = data;
	component->push_task = basic_push_task;
	component->pull_task = basic_pull_task;
	component->can_push = basic_can_push;
	component->can_pull = basic_can_pull;

	//~ struct basic_sched_data *data = malloc(sizeof(*data));
	//~ STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
	/* Create a linked-list of tasks and a condition variable to protect it */
	//~ starpu_task_list_init(&data->sched_list);
	//~ data->verbose = params->verbose;
	//Fifo
	// struct starpu_sched_component *component = starpu_sched_component_create(tree, "fifo");
	// struct _starpu_fifo_data *data;
	// _STARPU_MALLOC(data, sizeof(*data));
	// data->fifo = _starpu_create_fifo();
	// STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	// //Fin fifo
	// component->data = data;
	// component->push_task = fifo_push_task;
	// component->pull_task = fifo_pull_task;
	// component->can_push = fifo_can_push;
	// component->can_pull = fifo_can_pull;
	//~ component->deinit_data = basic_deinit_data;

	return component;
}

static void initialize_basic_center_policy(unsigned sched_ctx_id)
{
	printf("Debut initialize\n");
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_basic_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_DECIDE_ALWAYS  |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			//STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_EXP |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);

}

static void deinitialize_basic_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_basic_sched_policy =
{
	.init_sched = initialize_basic_center_policy,
	.deinit_sched = deinitialize_basic_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "basic-sched",
	.policy_description = "sched de base pour tester",
	.worker_type = STARPU_WORKER_LIST,
};

