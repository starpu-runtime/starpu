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

/* This scheduler runs only GEMMs on GPUs, and tries to feed them with as many
 * GEMMs as possible. */

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>

/* Optionally, it can take memory affinity into account, to avoid too many GPU
 * data transfers */

#define MEMORY_AFFINITY

struct child_data
{
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

static int gemm_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	unsigned n = component->nchildren;
	unsigned i;

	/* See if it's a GEMM task */
	const char *name = starpu_task_get_model_name(task);
	//fprintf(stderr, "it's %s\n", name);

	if (name && (!strcmp(name, "gemm") ||
		!strcmp(name, "dgemm") ||
		!strcmp(name, "sgemm") ||
		!strcmp(name, "chol_model_22") ||
		!strcmp(name, "starpu_dlu_lu_model_22") ||
		!strcmp(name, "starpu_slu_lu_model_22")))
	{
		/* It's a GEMM, try to push to GPUs */

		struct child_data child_data[n];

		for (i = 0; i < n; i++)
		{
			child_data[i].expected_end = -1;
			child_data[i].child = i;
		}

		/* Look at GPU availability time */
		for (i = 0; i < n; i++)
		{
			struct starpu_sched_component *child = component->children[i];
			double predicted;
			if (starpu_sched_component_execute_preds(child, task, &predicted))
			{
				double expected_start;
				child_data[i].expected_start =
					expected_start = child->estimated_end(child);
				child_data[i].predicted = predicted;
				child_data[i].expected_end = expected_start 
					+ predicted;

#ifdef MEMORY_AFFINITY
				double predicted_transfer;
				child_data[i].predicted_transfer =
					predicted_transfer = starpu_sched_component_transfer_length(child, task);
				child_data[i].expected_end += predicted_transfer;
#endif
			}
		}

		/* Sort by increasing expected end */
		qsort(child_data, n, sizeof(*child_data), compar);

		/* Try to push to the GPU with minimum availability time, to balance the load.  */
		for (i = 0; i < n; i++)
		{
			if (child_data[i].expected_end != -1)
			{
				struct starpu_sched_component *child = component->children[child_data[i].child];

				/* Note it in the task so that estimated_end() has it */
				task->predicted = child_data[i].predicted;
				task->predicted_transfer = child_data[i].predicted_transfer;

				int ret = starpu_sched_component_push_task(component,child,task);
				if (!ret)
					/* Ok, this GPU took it */
					return 0;
			}
		}
	}

	int workerid;
	/* It's not a GEMM, or no GPU wanted to take it, find somebody else */
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				for (i = 0; i < n; i++)
				{
					struct starpu_sched_component *child = component->children[i];
					int idworker;
					for(idworker = starpu_bitmap_first(component->children[i]->workers);
						idworker != -1;
						idworker = starpu_bitmap_next(component->children[i]->workers, idworker))
					{
						if (idworker == workerid)
						{
							if ((starpu_cpu_worker_get_count() == 0 ||
									starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
							 && (starpu_worker_can_execute_task(workerid,task,nimpl)
							   || starpu_combined_worker_can_execute_task(workerid, task, nimpl)))
							{
								int ret = starpu_sched_component_push_task(component,child,task);
								if (!ret)
									return 0;
							}
						}
					}
				}
			}
		}
	}
	/* FIFOs are full */
	return 1;
}

struct starpu_sched_component *starpu_sched_component_gemm_create(struct starpu_sched_tree *tree, void *params STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "gemm");

	component->push_task = gemm_push_task;

	return component;
}

static void initialize_gemm_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_gemm_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_MEMNODES |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_gemm_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_modular_gemm_policy =
{
	.init_sched = initialize_gemm_center_policy,
	.deinit_sched = deinitialize_gemm_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-gemm",
	.policy_description = "gemm modular policy",
	.worker_type = STARPU_WORKER_LIST,
};
