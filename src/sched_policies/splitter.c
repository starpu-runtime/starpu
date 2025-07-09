/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <sched_policies/splitter.h>
#include <common/uthash.h>
#include <core/task.h>
#include <profiling/splitter_bound.h>
#include <sched_policies/deque_modeling_policy_data_aware.h>

#include <limits.h>
#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

struct worker_data_node
{
	unsigned total_nb_workers;
	unsigned workerid_canonical;
	unsigned nb_busy;
	double end_time_busy;
	unsigned nb_not_busy;
	double end_time_not_busy;
};

static inline double predict_best_end_of_task(struct starpu_task *task, struct worker_data_node *data_nodes, unsigned nb_nodes, int sched_ctx_id, double min_start, unsigned *best_node_p, double *length_of_task)
{
	unsigned best_node = -1, best_impl =-1;
	double best_exp_end = DBL_MAX;
	double local_task_length[nb_nodes][STARPU_MAXIMPLEMENTATIONS];

	unsigned n;
	for (n = 0; n < nb_nodes; n++)
	{
		unsigned nimpl;
		unsigned impl_mask;
		unsigned workerid = data_nodes[n].workerid_canonical;
		if (!starpu_worker_can_execute_task_impl(workerid, task, &impl_mask))
			continue;

		double exp_start_worker = data_nodes[n].nb_not_busy > 0 ? data_nodes[n].end_time_not_busy : data_nodes[n].end_time_busy;
		double exp_start = STARPU_MAX(exp_start_worker, min_start);
		for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!(impl_mask & (1U << nimpl)))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			local_task_length[n][nimpl] = starpu_task_worker_expected_length(task, workerid, sched_ctx_id, nimpl);
			if (isnan(local_task_length[n][nimpl]))
			{
				// we will try this size to compute the length
				*best_node_p = -1;
				return DBL_MAX;
			}
			double exp_end = exp_start + local_task_length[n][nimpl];

			if (exp_end < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = exp_end;
				best_impl = nimpl;
				best_node = n;
			}
		}
	}

	if (best_node < nb_nodes)
	{
		*best_node_p = best_node;
		*length_of_task = local_task_length[best_node][best_impl];
	}
	return best_exp_end;
}

#define STARPU_DEFAULT_CUT_RATIO 0. // 0.08
#define NMAX_CL 256
struct cl_data_splitter_entry
{
	double ratio_split;
	unsigned long nb_tasks_split;
	unsigned long nb_tasks_nsplit;
	unsigned long nb_tasks_ready_soon_split;
	unsigned long already_split;
	uint32_t cl_footprint;
};

struct cl_data_splitter_table
{
	UT_hash_handle hh;
	uint32_t footprint;
	struct cl_data_splitter_entry *entry;
};

#define HASH_ADD_UINT32_T(head,field,add) HASH_ADD(hh,head,field,sizeof(uint32_t),add)
#define HASH_FIND_UINT32_T(head,find,out) HASH_FIND(hh,head,find,sizeof(uint32_t),out)

#ifdef STARPU_RECURSIVE_TASKS

static struct cl_data_splitter_entry *create_splitter_history_entry(struct cl_data_splitter_table **complete_table, struct starpu_codelet *cl, unsigned level)
{
	struct cl_data_splitter_entry *entry;
	_STARPU_MALLOC(entry, sizeof(*entry));
	entry->ratio_split = STARPU_DEFAULT_CUT_RATIO;
	entry->nb_tasks_split = 0;
	entry->nb_tasks_nsplit = 0;
	entry->already_split = 0;
	entry->nb_tasks_ready_soon_split = 0;
	entry->cl_footprint = starpu_hash_crc32c_be_ptr(cl, 0);
	entry->cl_footprint = starpu_hash_crc32c_be(level, entry->cl_footprint); // add level

	struct cl_data_splitter_table *table;
	_STARPU_MALLOC(table, sizeof(*table));
	table->footprint = entry->cl_footprint;
	table->entry = entry;

	HASH_ADD_UINT32_T(*complete_table, footprint, table);
	return entry;
}

struct data_splitter
{
	struct cl_data_splitter_table *table;
	unsigned nb_tasks_registered;
	starpu_pthread_mutex_t *mutex;
	unsigned long nb_tasks_on_sched;
	unsigned long nb_tasks_cpus;
	double ratio_respected;
	struct cl_data_splitter_entry **cache_entries;
	unsigned long nb_cache_entries;
};

static struct data_splitter splitter_data;
static unsigned splitter_data_is_init = 0;
void _splitter_initialize_data()
{
	splitter_data.mutex = malloc(sizeof(starpu_pthread_mutex_t));
	STARPU_PTHREAD_MUTEX_INIT(splitter_data.mutex, NULL);
	splitter_data.cache_entries = calloc(NMAX_CL, sizeof(struct cl_data_splitter_entry*));
	splitter_data.nb_cache_entries = 0;
	splitter_data.nb_tasks_registered = 0;
	splitter_data.nb_tasks_cpus = 0;
	splitter_data.nb_tasks_on_sched = 0;
	splitter_data.ratio_respected = starpu_getenv_float_default("STARPU_SPLITTER_RATIO", 0.75);
	splitter_data_is_init = 1;
}

static struct cl_data_splitter_entry *get_splitter_history_entry_from_cl(struct starpu_codelet *cl, unsigned level)
{
	if (!splitter_data_is_init)
		_splitter_initialize_data();
	uint32_t key = starpu_hash_crc32c_be_ptr(cl, 0);
	key = starpu_hash_crc32c_be(level, key);
	struct cl_data_splitter_table *cur_t = NULL;
	struct cl_data_splitter_entry *entry = NULL;
	STARPU_PTHREAD_MUTEX_LOCK(splitter_data.mutex);
	HASH_FIND_UINT32_T(splitter_data.table, &key, cur_t);
	if (cur_t == NULL)
	{
		entry = create_splitter_history_entry(&splitter_data.table, cl, level);
	}
	else
	{
		entry = cur_t->entry;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(splitter_data.mutex);
	return entry;
}

static struct cl_data_splitter_entry *get_splitter_history_entry(struct starpu_task *t)
{
	return get_splitter_history_entry_from_cl(t->cl, _starpu_task_get_level(t));
}

void _starpu_update_task_level_end(struct starpu_task *task)
{
	int level = _starpu_task_get_level(task);
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	if (level == 0 && splitter_data_is_init)
	{
		struct cl_data_splitter_entry *entry = get_splitter_history_entry(task);
		if (j->recursive.is_recursive_task)
		{
			(void)STARPU_ATOMIC_ADD(&entry->nb_tasks_split, -1);
		}
		else
		{
			(void)STARPU_ATOMIC_ADD(&entry->nb_tasks_nsplit, -1);
		}
	}
	if (!j->recursive.is_recursive_task && j->recursive.is_recursive_task_processed && splitter_data_is_init)
	{
		(void)STARPU_ATOMIC_ADD(&splitter_data.nb_tasks_on_sched, -1);
		if (level > 1)
		{
			(void)STARPU_ATOMIC_ADD(&splitter_data.nb_tasks_cpus, -1);
		}
	}

}

void _starpu_update_task_level(struct starpu_task *task)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	if (!j->recursive.is_recursive_task && !j->recursive.is_recursive_task_processed && splitter_data_is_init)
	{
		j->recursive.is_recursive_task_processed = 1;
		if (j->recursive.level > 1)
		{
			(void)STARPU_ATOMIC_ADD(&splitter_data.nb_tasks_cpus, 1);
		}
		(void)STARPU_ATOMIC_ADD(&splitter_data.nb_tasks_on_sched, 1);
	}
}

void starpu_add_data_cut()
{
}

void starpu_remove_data_cut()
{
}

void _splitter_reinit_cache_entry()
{
	splitter_data.nb_cache_entries = 0;
}

void _splitter_actualize_ratio_to_split(struct starpu_codelet *cl, double ratio, unsigned level)
{
	struct cl_data_splitter_entry *entry = get_splitter_history_entry_from_cl(cl, level);
	unsigned long cur_nb = STARPU_ATOMIC_ADD(&splitter_data.nb_cache_entries, 1);
	entry->ratio_split = ratio;
	splitter_data.cache_entries[cur_nb-1] = entry;
}

#if 0
static unsigned long _splitter_get_nb_tasks_split(struct starpu_codelet *cl, unsigned level)
{
	struct cl_data_splitter_entry *entry = get_splitter_history_entry_from_cl(cl, level);
	return entry->nb_tasks_split;
}
#endif

#if 0
static unsigned long _splitter_get_nb_tasks_nsplit(struct starpu_codelet *cl, unsigned level)
{
	struct cl_data_splitter_entry *entry = get_splitter_history_entry_from_cl(cl, level);
	return entry->nb_tasks_nsplit;
}
#endif

#if 0
static int _splitter_choose_between_task_or_task_array_worker_load(struct starpu_task *ptask, int nb_subtasks)
{
	(void) nb_subtasks;
	int level = _starpu_task_get_level(ptask);
	//long nb_flops = (long) ptask->flops;
	if (level == 0)
	{
		struct cl_data_splitter_entry *entry = get_splitter_history_entry(ptask);
		struct _starpu_job *job =_starpu_get_job_associated_to_task(ptask);
		long nb_tasks_nsplit = _starpu_splitter_bound_get_nb_nsplit(job), nb_tasks_split = _starpu_splitter_bound_get_nb_split(job);
		double ratio_to_split = entry->ratio_split;
		long total_tasks = nb_tasks_nsplit + nb_tasks_split + 1;
		double cur_ratio_split = nb_tasks_split*1./(total_tasks*1.);
		int cutting = cur_ratio_split < splitter_data.ratio_respected*ratio_to_split;
		cutting +=  STARPU_ATOMIC_ADD(&entry->already_split, 1) == 1;

		if (!cutting && starpu_worker_type_can_execute_task(STARPU_CUDA_WORKER, ptask))
		{ // this is a useful line which refrain the scheduler to put big tasks on CPUs
			// it has to be introduced on the scheduler later
			ptask->where = STARPU_CUDA;
		}
		return cutting;
	}
	return 0;
}
#endif

int _splitter_simulate_three_dimensions(struct starpu_task *ptask, int nb_tasks, int sched_ctx_id)
{
	(void) nb_tasks;
	(void) sched_ctx_id;
	if (!splitter_data_is_init)
	{
		_splitter_initialize_data();
	}
	//unsigned level = _starpu_task_get_level(ptask);
	struct cl_data_splitter_entry *entry = get_splitter_history_entry(ptask);
	struct _starpu_job *job =_starpu_get_job_associated_to_task(ptask);
	long nb_tasks_nsplit = _starpu_splitter_bound_get_nb_nsplit(job), nb_tasks_split = _starpu_splitter_bound_get_nb_split(job);
	double ratio_to_split = entry->ratio_split;
	long total_tasks = nb_tasks_nsplit + nb_tasks_split + 1;
	double cur_ratio_split = nb_tasks_split*1./(total_tasks*1.);

	int mandatory_split = !starpu_worker_type_can_execute_task(STARPU_CUDA_WORKER, ptask); // no choice, we need to split
	int possible_split = cur_ratio_split < ratio_to_split; //&& (level == 1 || STARPU_ATOMIC_ADD(&splitter_data.nb_tasks_cpus, 0) < 2*starpu_worker_get_count_by_type(STARPU_CPU_WORKER));

	return (mandatory_split || (possible_split
				   // && (splitter_data.nb_tasks_on_sched - splitter_data.nb_tasks_cpus > 2*starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) || level == 1)
			));
}

int _splitter_choose_three_dimensions(struct starpu_task *ptask, int nb_tasks, int sched_ctx_id)
{
	int split = (_splitter_simulate_three_dimensions(ptask, nb_tasks, sched_ctx_id));
	struct cl_data_splitter_entry *entry = get_splitter_history_entry(ptask);
	unsigned long already_split = STARPU_ATOMIC_ADD(&entry->already_split, 1);
	split += (already_split <= 1);
	return split;
}

#endif /* STARPU_RECURSIVE_TASKS */
