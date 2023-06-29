/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

/* Distributed queues using performance modeling to assign tasks */

#include <starpu_config.h>
#include <starpu_scheduler.h>
#include <starpu_scheduler.h>
#include <schedulers/starpu_heteroprio.h>
#include <schedulers/starpu_scheduler_toolbox.h>

#include <common/graph.h>
#include "heteroprio.h"

#include <common/fxt.h>
#include <core/task.h>
#include <core/workers.h>
#include <core/debug.h>
#include <starpu_bitmap.h>

#include <datawizard/memory_nodes.h>

#include <starpu_task_list.h>
#include <sched_policies/prio_deque.h>
#include <limits.h>
#include <errno.h>

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

#define STARPU_NB_TYPES STARPU_NARCH

#define STR_MAX_SIZE 64

#define STRINGIFY(x) _STR(x)
#define _STR(x) #x

/** Push strategy for use_locality */
enum laheteroprio_push_strategy
{
	PUSH_LS_SDH,
	PUSH_LS_SDH2,
	PUSH_LS_SDHB,
	PUSH_LC_SMWB,
	PUSH_NB_AUTO, // Always last to limit auto
	PUSH_LcS,
	PUSH_WORKER,
	PUSH_AUTO
};

/** Queue used when use_locality is enabled */
struct laqueue
{
	unsigned char* data;
	long int capacity;
	long int current_index;
	long int size_of_element;
};

static struct laqueue laqueue_init(const long int size_of_element);
static void laqueue_destroy(struct laqueue* q);
//static long int laqueue_size(struct laqueue* q);
static void laqueue_push(struct laqueue* q, void* data);
static void* laqueue_pop(struct laqueue* q);
//static void* laqueue_top(struct laqueue* q);

struct starpu_laheteroprio_access_item
{
	unsigned prio_idx;
	unsigned wgroup_idx;
};

static struct laqueue laqueue_init(const long int size_of_element)
{
	struct laqueue q;
	q.data = NULL;
	q.capacity = 0;
	q.current_index = 0;
	q.size_of_element = size_of_element;
	return q;
}

static void laqueue_destroy(struct laqueue* q)
{
	STARPU_ASSERT(q->current_index == 0);
	free(q->data);
}

//static long int laqueue_size(struct laqueue* q)
//{
//	return q->capacity;
//}

static void laqueue_push(struct laqueue* q, void* data)
{
	if(q->current_index == q->capacity)
	{
		q->capacity = (q->capacity+10)*2;
		_STARPU_REALLOC(q->data, q->size_of_element*q->capacity);
	}
	memcpy(&q->data[(q->current_index++)*q->size_of_element], data, q->size_of_element);
}

static void* laqueue_pop(struct laqueue* q)
{
	STARPU_ASSERT(q->current_index-1 >= 0);
	unsigned char* data = &q->data[(q->current_index-1)*q->size_of_element];
	q->current_index -= 1;
	return data;
}

//static void* laqueue_top(struct laqueue* q)
//{
//	STARPU_ASSERT(q->current_index-1 >= 0);
//	return &q->data[(q->current_index-1)*q->size_of_element];
//}

/** How are codelet grouped by priority */
enum autoheteroprio_codelet_grouping_strategy
{
	BY_PERF_MODEL_OR_NAME = 0, 	/** Using perfmodel symbol or codelet's name if no perfmodel */
	BY_NAME_ONLY = 1			/** Based on the codelet's name only */
};

/* A bucket corresponds to a Pair of priorities
 * When a task is pushed with a priority X, it will be stored
 * into the bucket X.
 * All the tasks stored in the fifo should be computable by the arch
 * in valid_archs.
 * For example if valid_archs = (STARPU_CPU|STARPU_CUDA)
 * Then task->task->where should be at least (STARPU_CPU|STARPU_CUDA)
 */
struct _heteroprio_bucket
{
	/* Tasks of the current bucket */
	/* In case data locality is NOT used, only the first element of the array is used */
	/* In case data locality IS used, the element refers to a worker group */
	struct starpu_task_list tasks_queue[LAHETEROPRIO_MAX_WORKER_GROUPS];
	/* The correct arch for the current bucket */
	unsigned valid_archs;
	/* The slow factors for any archs */
	float slow_factors_per_index[STARPU_NB_TYPES];
	/* The base arch for the slow factor (the fatest arch for the current task in the bucket */
	unsigned factor_base_arch_index;

	/****  Fields used when use_locality == 1 :  ****/

	/* the number of tasks in all the queues (was previously tasks_queue.ntasks) */
	unsigned tasks_queue_ntasks;
	/* to keep track of the mn at push time */
	struct laqueue auto_mn[LAHETEROPRIO_MAX_WORKER_GROUPS];
};

static int use_la_mode = 0;
static int use_auto_mode = 0;

/* Init a bucket */
static void _heteroprio_bucket_init(struct _heteroprio_bucket* bucket)
{
	if(use_la_mode)
	{
		unsigned i;
		memset(bucket, 0, sizeof(*bucket));
		for(i = 0 ; i < LAHETEROPRIO_MAX_WORKER_GROUPS ; ++i)
		{
		        starpu_task_list_init(&bucket->tasks_queue[i]);
		        bucket->auto_mn[i] = laqueue_init(sizeof(unsigned)*PUSH_NB_AUTO);
		}
	}
	else
	{
		memset(bucket, 0, sizeof(*bucket));
		starpu_task_list_init(&bucket->tasks_queue[0]);
	}
}

/* Release a bucket */
static void _heteroprio_bucket_release(struct _heteroprio_bucket* bucket)
{
	if(use_la_mode)
	{
		unsigned i;
		for(i = 0 ; i < LAHETEROPRIO_MAX_WORKER_GROUPS ; ++i)
		{
			STARPU_ASSERT(starpu_task_list_empty(&bucket->tasks_queue[i]) != 0);
			laqueue_destroy(&bucket->auto_mn[i]);
		}
	}
	else
	{
		STARPU_ASSERT(starpu_task_list_empty(&bucket->tasks_queue[0]) != 0);
		// don't task_lists need to be destroyed ?
	}
}

// Must be manually add to get more stats
//#define LAHETEROPRIO_PRINT_STAT

static enum laheteroprio_push_strategy getEnvAdvPush()
{
	const char *push = starpu_getenv("STARPU_LAHETEROPRIO_PUSH");
	if (push)
	{
		if(strcmp(push, "WORKER") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_WORKER\n");
#endif
			return PUSH_WORKER;
		}
		if(strcmp(push, "LcS") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_LcS\n");
#endif
			return PUSH_LcS;
		}
		if(strcmp(push, "LS_SDH") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_LS_SDH\n");
#endif
			return PUSH_LS_SDH;
		}
		if(strcmp(push, "LS_SDH2") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_LS_SDH2\n");
#endif
			return PUSH_LS_SDH2;
		}
		if(strcmp(push, "LS_SDHB") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_LS_SDHB\n");
#endif
			return PUSH_LS_SDHB;
		}
		if(strcmp(push, "LC_SMWB") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_LC_SMWB\n");
#endif
			return PUSH_LC_SMWB;
		}
		if(strcmp(push, "AUTO") == 0)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			_STARPU_MSG("[LAHETEROPRIO] Use PUSH_AUTO\n");
#endif
			return PUSH_AUTO;
		}
		_STARPU_MSG("Undefined push strategy %s\n", push);
	}
#ifdef LAHETEROPRIO_PRINT_STAT
	_STARPU_MSG("[LAHETEROPRIO] Use PUSH_AUTO\n");
#endif
	return PUSH_AUTO;
}

/* A worker is mainly composed of a fifo for the tasks
 * and some direct access to worker properties.
 * The fifo is implemented with any array,
 * to read a task, access tasks_queue[tasks_queue_index]
 * to write a task, access tasks_queue[(tasks_queue_index+tasks_queue_size)%HETEROPRIO_MAX_PREFETCH]
 */
/* ANDRA_MODIF: can use starpu fifo + starpu sched_mutex*/
struct _heteroprio_worker_wrapper
{
	unsigned arch_type;
	unsigned arch_index;

	/** Only used when use_locality==0 : */
	struct starpu_st_prio_deque tasks_queue;
};

struct _starpu_heteroprio_data
{
	starpu_pthread_mutex_t policy_mutex;
	struct starpu_bitmap waiters;
	/* The bucket to store the tasks */
	struct _heteroprio_bucket buckets[HETEROPRIO_MAX_PRIO];
	/* Whether heteroprio should consider data locality or not */
	unsigned use_locality;
	/* The number of buckets for each arch */
	unsigned nb_prio_per_arch_index[STARPU_NB_TYPES];
	/* The mapping to the corresponding buckets */
	unsigned prio_mapping_per_arch_index[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	/* The number of available tasks for a given arch (not prefetched) */
	unsigned nb_remaining_tasks_per_arch_index[STARPU_NB_TYPES];
	/* The total number of tasks in the bucket (not prefetched) */
	unsigned total_tasks_in_buckets;
	/* The number of workers for a given arch */
	unsigned nb_workers_per_arch_index[STARPU_NB_TYPES];

	/* Information on all the workers */
	struct _heteroprio_worker_wrapper workers_heteroprio[STARPU_NMAXWORKERS];

	/*** use_locality==0 specific : */

	/* The total number of prefetched tasks for a given arch */
	unsigned nb_prefetched_tasks_per_arch_index[STARPU_NB_TYPES];

	/*** use_locality==1 (laheteroprio) specific : */

	/* Helps ensuring laheteroprio has been correctly initialized */
	unsigned map_wgroup_has_been_called;
	/* Helps ensuring laheteroprio has been correctly initialized */
	unsigned warned_change_nb_memory_nodes;
	/* Number of memory nodes */
	unsigned nb_memory_nodes;
	/* The mapping to the corresponding prio prio_mapping_per_arch_index[x][prio_mapping_per_arch_index[x][y]] = y */
	unsigned bucket_mapping_per_arch_index[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	/* The wgroup for all the workers */
	unsigned workers_laheteroprio_wgroup_index[STARPU_NMAXWORKERS];
	/* Number of wgroups */
	unsigned nb_wgroups;
	/* The task queue for the tasks inserted by the master thread */
	unsigned master_tasks_queue_idx;
	/* Arch related to each wgroup (for now only one kind of arch per wgroup */
	unsigned arch_of_wgroups[LAHETEROPRIO_MAX_WORKER_GROUPS];
	/* The pop offset per group */
	struct starpu_laheteroprio_access_item wgroup_pop_access_orders[LAHETEROPRIO_MAX_WORKER_GROUPS][LAHETEROPRIO_MAX_WORKER_GROUPS*HETEROPRIO_MAX_PRIO];
	/* Size of wgroup_pop_access_orders items */
	unsigned wgroup_pop_access_orders_size[LAHETEROPRIO_MAX_WORKER_GROUPS];
	/* The push strategy */
	enum laheteroprio_push_strategy pushStrategyToUse;
	enum laheteroprio_push_strategy pushStrategySet;
	int pushStrategyHistory[PUSH_NB_AUTO];
	starpu_pthread_mutex_t push_history_mutex;

	/*** auto-heteroprio specific : */

	/** Strategy to determine on which base which can assign same priority to codelets */
	enum autoheteroprio_codelet_grouping_strategy codelet_grouping_strategy;

	unsigned use_auto_calibration;

	starpu_pthread_mutex_t auto_calibration_mutex;

	// parameters:

	unsigned autoheteroprio_priority_ordering_policy;
	// reorder priority every priority_ordering_interval pushed tasks
	int priority_ordering_interval;
	// if set to 0: will gather data from execution (task time, NOD, etc.)
	unsigned freeze_data_gathering;


	unsigned autoheteroprio_print_prio_after_ordering;
	unsigned autoheteroprio_print_data_on_update;


	// 0 = if a task has no implementation on arch, expected time will be AUTOHETEROPRIO_LONG_TIME
	// 1 = if a task has no implementation on arch, expected time will be the shortest time among all archs
	unsigned autoheteroprio_time_estimation_policy;


	// environment hyperparameters

	double NTnodPond;
	double NTexpVal;
	double BNexpVal;
	double URTurt;
	double URT2urt;
	double URT2prop;
	double and2pond;
	double and3pond;
	double and4pond;
	double and5xoffset;
	double and5yoffset;
	double and9xoffset;
	double and9yoffset;
	double and10xoffset;
	double and10yoffset;
	double and11xoffset;
	double and11yoffset;
	double ANTnodPond;
	double ANTexpVal;

	int priority_last_ordering;

	// lightweight time profiling:

	// busy time and free time of each arch for current execution
	double current_arch_busy_time[STARPU_NB_TYPES];
	double current_arch_free_time[STARPU_NB_TYPES];

	// last time a worker executed either pre_exec or post_exec hook
	double last_hook_exec_time[STARPU_NMAXWORKERS];

	// task data:

	unsigned found_codelet_names_length;
	char found_codelet_names[HETEROPRIO_MAX_PRIO][CODELET_MAX_NAME_LENGTH];
	unsigned found_codelet_names_on_arch[STARPU_NB_TYPES];

	// busy time and free time of each arch
	double average_arch_busy_time[STARPU_NB_TYPES];
	double average_arch_free_time[STARPU_NB_TYPES];

	// average prio NOD for each task
	double prio_average_NOD[HETEROPRIO_MAX_PRIO];
	// NOD sample size
	unsigned prio_average_NOD_count[HETEROPRIO_MAX_PRIO];

	// average prio URT for each task
	double prio_average_URT[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	// URT sample size
	unsigned prio_average_URT_count[HETEROPRIO_MAX_PRIO];

	// average execution time for each arch
	double prio_average_time_arch[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	// sample size of execution times
	unsigned prio_average_time_arch_count[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	// true if we have at least one sample to compute the average execution time
	unsigned prio_arch_has_time_info[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];

	// proportion of each task during execution (sum of each prio should equal 1)
	double prio_overall_proportion[HETEROPRIO_MAX_PRIO];
	// sample size (number of added tasks of a type)
	unsigned prio_overall_proportion_count[HETEROPRIO_MAX_PRIO];

	// actual location of a task execution (~= probability of being executed on an arch) (sum of each arch for a prio should equal 1)
	double prio_arch_proportion[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	unsigned prio_arch_proportion_count[HETEROPRIO_MAX_PRIO];

	// sum of each successor's best time (better arch)
	double prio_average_successors_best_time_sum[HETEROPRIO_MAX_PRIO];
	// sample size
	unsigned prio_average_successors_best_time_sum_count[HETEROPRIO_MAX_PRIO];

	// best possible time of a prio (between archs)
	double prio_average_best[HETEROPRIO_MAX_PRIO];
	unsigned prio_average_best_count[HETEROPRIO_MAX_PRIO];
};

// declare prototypes
void starpu_heteroprio_map_wgroup_memory_nodes_hp(struct _starpu_heteroprio_data *hp);
static double get_best_autoheteroprio_estimated_time(struct _starpu_heteroprio_data *hp, unsigned priority);

static int starpu_heteroprio_types_to_arch(enum starpu_worker_archtype arch)
{
	if (arch >= STARPU_NARCH)
		return 0;
	return STARPU_WORKER_TO_MASK(arch);
}

static int arch_can_execute_prio(struct _starpu_heteroprio_data *hp, unsigned arch, unsigned prio)
{
	return (hp->buckets[prio].valid_archs&starpu_heteroprio_types_to_arch(arch))!=0;
}

void starpu_heteroprio_set_use_locality(unsigned sched_ctx_id, unsigned use_locality)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_ASSERT(use_locality == 0 || use_locality == 1);

	hp->use_locality = use_locality;
}

/** Tell how many prio there are for a given arch */
void starpu_heteroprio_set_nb_prios_hp(struct _starpu_heteroprio_data *hp, enum starpu_worker_archtype arch, unsigned max_prio)
{
	STARPU_ASSERT(max_prio <= HETEROPRIO_MAX_PRIO);

	hp->nb_prio_per_arch_index[arch] = max_prio;

	if(hp->use_locality)
	{
		starpu_heteroprio_map_wgroup_memory_nodes_hp(hp);
	}
}

/** Tell how many prio there are for a given arch */
void starpu_heteroprio_set_nb_prios(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned max_prio)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_set_nb_prios_hp(hp, arch, max_prio);
}

void starpu_heteroprio_set_mapping_hp_without_arch(struct _starpu_heteroprio_data *hp, enum starpu_worker_archtype arch, unsigned source_prio, unsigned dest_bucket_id)
{
	STARPU_ASSERT(dest_bucket_id < HETEROPRIO_MAX_PRIO);

	hp->prio_mapping_per_arch_index[arch][source_prio] = dest_bucket_id;

	if(hp->use_locality == 1)
	{
		hp->bucket_mapping_per_arch_index[arch][dest_bucket_id] = source_prio;
	}
}

void starpu_heteroprio_set_mapping_without_arch(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned source_prio, unsigned dest_bucket_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_set_mapping_hp_without_arch(hp, arch, source_prio, dest_bucket_id);
}

/** Set the mapping for a given arch prio=>bucket */
void starpu_heteroprio_set_mapping_hp(struct _starpu_heteroprio_data *hp, enum starpu_worker_archtype arch, unsigned source_prio, unsigned dest_bucket_id)
{
	starpu_heteroprio_set_mapping_hp_without_arch(hp, arch, source_prio, dest_bucket_id);

	hp->buckets[dest_bucket_id].valid_archs |= starpu_heteroprio_types_to_arch(arch);
	_STARPU_DEBUG("Adding arch %d to bucket %u\n", arch, dest_bucket_id);
}

inline void starpu_heteroprio_set_mapping(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned source_prio, unsigned dest_bucket_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_set_mapping_hp(hp, arch, source_prio, dest_bucket_id);
}

void starpu_heteroprio_clear_mapping_hp(struct _starpu_heteroprio_data *hp)
{
	// direct mapping for all archs (and overwrite any changes to bucket archs)
	unsigned arch;
	for(arch=0;arch<STARPU_NB_TYPES;++arch)
	{
		int prio;
		for(prio=0 ; prio<HETEROPRIO_MAX_PRIO ; prio++)
		{
			starpu_heteroprio_set_mapping_hp_without_arch(hp, arch, prio, prio);
			hp->buckets[prio].valid_archs = 0;
		}
	}
}

void starpu_heteroprio_set_faster_arch_hp(struct _starpu_heteroprio_data *hp, enum starpu_worker_archtype arch, unsigned bucket_id)
{
	STARPU_ASSERT(bucket_id < HETEROPRIO_MAX_PRIO);

	hp->buckets[bucket_id].factor_base_arch_index = arch;

	hp->buckets[bucket_id].slow_factors_per_index[arch] = 0;
}

/** Tell which arch is the faster for the tasks of a bucket (optional) */
inline void starpu_heteroprio_set_faster_arch(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned bucket_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_set_faster_arch_hp(hp, arch, bucket_id);
}

void starpu_heteroprio_set_arch_slow_factor_hp(struct _starpu_heteroprio_data *hp, enum starpu_worker_archtype arch, unsigned bucket_id, float slow_factor)
{
	STARPU_ASSERT(bucket_id < HETEROPRIO_MAX_PRIO);

	hp->buckets[bucket_id].slow_factors_per_index[arch] = slow_factor;
}

/** Tell how slow is a arch for the tasks of a bucket (optional) */
inline void starpu_heteroprio_set_arch_slow_factor(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned bucket_id, float slow_factor)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_set_arch_slow_factor_hp(hp, arch, bucket_id, slow_factor);
}

void starpu_heteroprio_set_pop_access_order_hp(struct _starpu_heteroprio_data *hp, unsigned wgroup_id, const struct starpu_laheteroprio_access_item access_items[], const unsigned size)
{
	STARPU_ASSERT(size <= LAHETEROPRIO_MAX_WORKER_GROUPS * HETEROPRIO_MAX_PRIO);
	const unsigned arch_of_wgroup = hp->arch_of_wgroups[wgroup_id];
	const unsigned nb_prios = hp->nb_prio_per_arch_index[arch_of_wgroup];
	const unsigned nb_wgroups = hp->nb_wgroups;
	STARPU_ASSERT(size <= nb_wgroups *nb_prios);
	memcpy(hp->wgroup_pop_access_orders[wgroup_id], access_items, sizeof(struct starpu_laheteroprio_access_item) *size);
	hp->wgroup_pop_access_orders_size[wgroup_id] = size;
}

void starpu_heteroprio_set_pop_access_order(unsigned sched_ctx_id, unsigned wgroup_id, const struct starpu_laheteroprio_access_item access_items[], const unsigned size)
{
	STARPU_ASSERT(size <= LAHETEROPRIO_MAX_WORKER_GROUPS * HETEROPRIO_MAX_PRIO);
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data *) starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_heteroprio_set_pop_access_order_hp(hp, wgroup_id, access_items, size);
}

struct dist
{
	double dist;
	unsigned wgroup_idx;
};

static int comp_dist(const void *elem1, const void *elem2)
{
	const struct dist *d1 = ((struct dist *) elem1);
	const struct dist *d2 = ((struct dist *) elem2);
	if (d1->dist > d2->dist) return 1;
	if (d1->dist < d2->dist) return -1;
	return 0;
}

void starpu_heteroprio_map_wgroup_memory_nodes_hp(struct _starpu_heteroprio_data *hp)
{
	STARPU_ASSERT_MSG(hp->use_locality == 1, "starpu_heteroprio_map_wgroup_memory_nodes has been called without enabling LA mode\n");
	hp->map_wgroup_has_been_called = 1; // Set flag to 1

	// Set the number of memory nodes
	hp->nb_memory_nodes = starpu_memory_nodes_get_count();
	const unsigned current_nb_memory_nodes = hp->nb_memory_nodes;

	hp->warned_change_nb_memory_nodes = 0;

	hp->nb_wgroups = current_nb_memory_nodes;
	// Set memory nodes' type
	{
		unsigned idx_memnode;
		for (idx_memnode = 0; idx_memnode < current_nb_memory_nodes; ++idx_memnode)
		{
			const enum starpu_node_kind memnode_kind = starpu_node_get_kind(idx_memnode);
			hp->arch_of_wgroups[idx_memnode] = starpu_memory_node_get_worker_archtype(memnode_kind);
		}
	}
	// Set workers' type
	{
		unsigned idx_worker;
		for (idx_worker = 0; idx_worker < starpu_worker_get_count(); ++idx_worker)
		{
			hp->workers_laheteroprio_wgroup_index[idx_worker] = starpu_worker_get_memory_node(idx_worker);
		}
	}
	if (starpu_cpu_worker_get_count() != 0)
	{
		unsigned cpu_0 = starpu_worker_get_by_type(STARPU_CPU_WORKER, 0);
		hp->master_tasks_queue_idx = starpu_worker_get_memory_node(cpu_0);
	}
	else
	{
		// Consider memory node 0 as the CPU
		STARPU_ASSERT(starpu_node_get_kind(0) == STARPU_CPU_RAM);
		hp->master_tasks_queue_idx = 0;
	}
	// Build memory distance matrix
	double dist_mem_matrix[LAHETEROPRIO_MAX_WORKER_GROUPS][LAHETEROPRIO_MAX_WORKER_GROUPS] = {{ 0 }};
	{
		unsigned idx_mem_node1;
		unsigned idx_mem_node2;
		double max_dist_mem = 0;
		for (idx_mem_node1 = 0; idx_mem_node1 < current_nb_memory_nodes; ++idx_mem_node1)
		{
			for (idx_mem_node2 = 0; idx_mem_node2 < current_nb_memory_nodes; ++idx_mem_node2)
			{
				if (idx_mem_node1 == idx_mem_node2)
				{
					dist_mem_matrix[idx_mem_node1][idx_mem_node2] = 0;
				}
				else
				{
					dist_mem_matrix[idx_mem_node1][idx_mem_node2] = starpu_transfer_predict(idx_mem_node2, idx_mem_node1, 1024 * 1024 *1024);
					max_dist_mem = STARPU_MAX(max_dist_mem, dist_mem_matrix[idx_mem_node1][idx_mem_node2]);
				}
			}
		}
		for (idx_mem_node1 = 0; idx_mem_node1 < current_nb_memory_nodes; ++idx_mem_node1)
		{
			for (idx_mem_node2 = 0; idx_mem_node2 < current_nb_memory_nodes; ++idx_mem_node2)
			{
				dist_mem_matrix[idx_mem_node1][idx_mem_node2] /= max_dist_mem;
			}
		}
	}
	// Build priority distance matrix
	double dist_prio_matrix[LAHETEROPRIO_MAX_WORKER_GROUPS][LAHETEROPRIO_MAX_WORKER_GROUPS] = {{ 0 }};
	{
		unsigned idx_prio_node1;
		unsigned idx_prio_node2;
		for (idx_prio_node1 = 0; idx_prio_node1 < current_nb_memory_nodes; ++idx_prio_node1)
		{
			for (idx_prio_node2 = 0; idx_prio_node2 < current_nb_memory_nodes; ++idx_prio_node2)
			{
				if (idx_prio_node1 == idx_prio_node2)
				{
					dist_prio_matrix[idx_prio_node1][idx_prio_node2] = 0;
				}
				else
				{
					const unsigned arch_wgroup1 = hp->arch_of_wgroups[idx_prio_node1];
					const unsigned arch_wgroup2 = hp->arch_of_wgroups[idx_prio_node2];
					double diff = 0;
					int cpt1 = 0;
					int cpt2 = 0;
					unsigned idx;
					for(idx = 0; idx < HETEROPRIO_MAX_PRIO; ++idx)
					{
						diff += fabs((double)(hp->bucket_mapping_per_arch_index[arch_wgroup1][idx] + 1) - (double)(hp->bucket_mapping_per_arch_index[arch_wgroup2][idx] + 1));
						if (hp->bucket_mapping_per_arch_index[arch_wgroup1][idx] != (unsigned)-1) cpt1 += 1;
						if (hp->bucket_mapping_per_arch_index[arch_wgroup2][idx] != (unsigned)-1) cpt2 += 1;
					}
					const int maxcpt = STARPU_MAX(cpt1, cpt2);
					diff /= (maxcpt + 1) *(maxcpt + 2) / 2.;
					dist_prio_matrix[idx_prio_node1][idx_prio_node2] = diff;
				}
			}
		}
	}
	// Build final distance matrix
	double dist_matrix[LAHETEROPRIO_MAX_WORKER_GROUPS][LAHETEROPRIO_MAX_WORKER_GROUPS] = {{ 0 }};
	{
		const double alpha = 0.5;
		unsigned idx_node1;
		unsigned idx_node2;
		for (idx_node1 = 0; idx_node1 < current_nb_memory_nodes; ++idx_node1)
		{
			for (idx_node2 = 0; idx_node2 < current_nb_memory_nodes; ++idx_node2)
			{
				dist_matrix[idx_node1][idx_node2] = (1 - dist_prio_matrix[idx_node1][idx_node2]) *alpha + dist_mem_matrix[idx_node1][idx_node2] *(1 - alpha);
			}
		}
	}
	unsigned nb_closed_nodes[STARPU_NB_TYPES];
	{
		char var_name[STR_MAX_SIZE];

		// Retrieving environment variable STARPU_LAHETEROPRIO_S_* for each architecture
		unsigned arch;
		for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
		{
			const char *arch_env_name = starpu_worker_get_type_as_env_var(arch);
			if(arch_env_name)
			{
				snprintf(var_name, STR_MAX_SIZE, "STARPU_LAHETEROPRIO_S_%s",
					arch_env_name);
				unsigned default_value = arch == STARPU_CPU_WORKER ? current_nb_memory_nodes - 1 : 1;

				nb_closed_nodes[arch] = starpu_getenv_number_default(var_name, default_value);
			}
		}
	}
	unsigned nb_prio_step[STARPU_NB_TYPES];
	{
		char var_name[STR_MAX_SIZE];

		// Retrieving environment variable STARPU_LAHETEROPRIO_PRIO_STEP_* for each architecture
		unsigned arch;
		for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
		{
			const char *arch_env_name = starpu_worker_get_type_as_env_var(arch);
			if(arch_env_name)
			{
				snprintf(var_name, STR_MAX_SIZE, "STARPU_LAHETEROPRIO_PRIO_STEP_%s",
					arch_env_name);
				unsigned default_value = arch != STARPU_CPU_WORKER ? hp->nb_prio_per_arch_index[arch] : 1;

				nb_prio_step[arch] = starpu_getenv_number_default(var_name, default_value);
			}
		}
	}
#ifdef LAHETEROPRIO_PRINT_STAT
	_STARPU_MSG("[LAHETEROPRIO] nb_closed_nodes[STARPU_CPU_WORKER] %u\n", nb_closed_nodes[STARPU_CPU_WORKER]);
	_STARPU_MSG("[LAHETEROPRIO] nb_closed_nodes[STARPU_CUDA_WORKER] %u\n", nb_closed_nodes[STARPU_CUDA_WORKER]);
	_STARPU_MSG("[LAHETEROPRIO] nb_prio_step[STARPU_CPU_WORKER] %u\n", nb_prio_step[STARPU_CPU_WORKER]);
	_STARPU_MSG("[LAHETEROPRIO] nb_prio_step[STARPU_CUDA_WORKER] %u\n", nb_prio_step[STARPU_CUDA_WORKER]);
#endif
	STARPU_ASSERT(hp->nb_wgroups == current_nb_memory_nodes);
	unsigned wgroup_idx;
	for (wgroup_idx = 0; wgroup_idx < current_nb_memory_nodes; ++wgroup_idx)
	{
		const unsigned wgroup_arch = hp->arch_of_wgroups[wgroup_idx];
		struct dist others[LAHETEROPRIO_MAX_WORKER_GROUPS];
		unsigned access_wgroup_idx;
		for (access_wgroup_idx = 0; access_wgroup_idx < current_nb_memory_nodes; ++access_wgroup_idx)
		{
			others[access_wgroup_idx].wgroup_idx = access_wgroup_idx;
			others[access_wgroup_idx].dist = dist_matrix[wgroup_idx][access_wgroup_idx];
		}
		{
			struct dist tmp = others[wgroup_idx];
			others[wgroup_idx] = others[0];
			others[0] = tmp;
		}
		qsort(others + 1, current_nb_memory_nodes - 1, sizeof(struct dist), comp_dist);
		struct starpu_laheteroprio_access_item buffer_access_items[LAHETEROPRIO_MAX_WORKER_GROUPS *HETEROPRIO_MAX_PRIO];
		const unsigned nb_prio_in_wgroup = hp->nb_prio_per_arch_index[hp->arch_of_wgroups[wgroup_idx]];
		unsigned access_idx = 0;
		unsigned prio_block_idx;
		for (prio_block_idx = 0; prio_block_idx < nb_prio_in_wgroup; prio_block_idx += nb_prio_step[wgroup_arch])
		{
			{
				access_wgroup_idx = 0;
				unsigned prio_idx;
				for (prio_idx = prio_block_idx; prio_idx < STARPU_MIN(prio_block_idx + nb_prio_step[wgroup_arch], nb_prio_in_wgroup); ++prio_idx)
				{
					buffer_access_items[access_idx].prio_idx = prio_idx;
					buffer_access_items[access_idx].wgroup_idx = others[access_wgroup_idx].wgroup_idx;
					access_idx += 1;
				}
			}
			unsigned prio_idx;
			for (prio_idx = prio_block_idx; prio_idx < STARPU_MIN(prio_block_idx + nb_prio_step[wgroup_arch], nb_prio_in_wgroup); ++prio_idx)
			{
				for (access_wgroup_idx = 1; access_wgroup_idx < STARPU_MIN(nb_closed_nodes[wgroup_arch] + 1, current_nb_memory_nodes); ++access_wgroup_idx)
				{
					buffer_access_items[access_idx].prio_idx = prio_idx;
					buffer_access_items[access_idx].wgroup_idx = others[access_wgroup_idx].wgroup_idx;
					access_idx += 1;
				}
			}
		}
		unsigned prio_idx;
		for (prio_idx = 0; prio_idx < nb_prio_in_wgroup; ++prio_idx)
		{
			for (access_wgroup_idx = nb_closed_nodes[wgroup_arch] + 1; access_wgroup_idx < current_nb_memory_nodes; ++access_wgroup_idx)
			{
				buffer_access_items[access_idx].prio_idx = prio_idx;
				buffer_access_items[access_idx].wgroup_idx = others[access_wgroup_idx].wgroup_idx;
				access_idx += 1;
			}
		}
		starpu_heteroprio_set_pop_access_order_hp(hp, wgroup_idx, buffer_access_items, access_idx);
	}
}

void starpu_heteroprio_map_wgroup_memory_nodes(unsigned sched_ctx_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data *) starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_heteroprio_map_wgroup_memory_nodes_hp(hp);
}

void starpu_heteroprio_print_wgroups(FILE *stream, unsigned sched_ctx_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data *) starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_ASSERT_MSG(hp->use_locality == 1, "starpu_heteroprio_print_wgroups has been called without enabling LA mode\n");

	fprintf(stream, "[STARPU-LAHETEROPRIO] There are %u groups\n", hp->nb_wgroups);
	char dest_name[512];
	unsigned worker_id;
	for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
	{
		starpu_worker_get_name(worker_id, dest_name, 512);
		fprintf(stream, "[STARPU-LAHETEROPRIO] Worker %u => group %u (%s)\n", worker_id, hp->workers_laheteroprio_wgroup_index[worker_id], dest_name);
	}
	fprintf(stream, "\n");
	unsigned idx_wgroup;
	for (idx_wgroup = 0; idx_wgroup < hp->nb_wgroups; ++idx_wgroup)
	{
		int access_order[LAHETEROPRIO_MAX_WORKER_GROUPS][HETEROPRIO_MAX_PRIO] = {{ 0 }};
		memset(access_order, -1, sizeof(access_order[0][0]) *LAHETEROPRIO_MAX_WORKER_GROUPS * HETEROPRIO_MAX_PRIO);
		const unsigned wgroup_arch = hp->arch_of_wgroups[idx_wgroup];
		const unsigned nb_prios = hp->nb_prio_per_arch_index[wgroup_arch];
		const unsigned nb_wgroups = hp->nb_wgroups;
		const struct starpu_laheteroprio_access_item *wgroup_access_order = hp->wgroup_pop_access_orders[idx_wgroup];
		const unsigned wgroup_access_order_size = hp->wgroup_pop_access_orders_size[idx_wgroup];
		unsigned idx_access_item;
		for (idx_access_item = 0; idx_access_item < wgroup_access_order_size; ++idx_access_item)
		{
			const unsigned current_wgroupid = wgroup_access_order[idx_access_item].wgroup_idx;
			const unsigned current_prio = wgroup_access_order[idx_access_item].prio_idx;
			access_order[current_wgroupid][current_prio] = idx_access_item;
		}
		fprintf(stream, "[STARPU-LAHETEROPRIO] Access order for wgroup %u (of arch type %u):\n", idx_wgroup, wgroup_arch);
		unsigned idx_prio;
		for (idx_prio = nb_prios; idx_prio > 0; --idx_prio)
		{
			const unsigned current_bucket = hp->prio_mapping_per_arch_index[wgroup_arch][idx_prio - 1];
			fprintf(stream, "[STARPU-LAHETEROPRIO]     Prio %3u (Bucket %3u) => ", idx_prio - 1, current_bucket);
			unsigned idx_wgroup_prio;
			for (idx_wgroup_prio = 0; idx_wgroup_prio < nb_wgroups; ++idx_wgroup_prio)
			{
				if (access_order[idx_wgroup][idx_prio - 1] == -1)
				{
					fprintf(stream, "[XX] ");
				}
				else
				{
					fprintf(stream, "[%2d] ", access_order[idx_wgroup][idx_prio - 1]);
				}
			}
			fprintf(stream, "\n");
		}
		fprintf(stream, "\n");
	}
}

/** If the user does not provide an init callback we create a single bucket for all architectures */
static inline void default_init_sched(unsigned sched_ctx_id)
{
	int min_prio = starpu_sched_ctx_get_min_priority(sched_ctx_id);
	int max_prio = starpu_sched_ctx_get_max_priority(sched_ctx_id);
	STARPU_ASSERT(min_prio >= 0);
	STARPU_ASSERT(max_prio >= 0);

	enum starpu_worker_archtype type;

	// By default each type of devices uses 1 bucket and no slow factor
	for (type = 0; type < STARPU_NARCH; type++)
		if (starpu_worker_get_count_by_type(type) > 0)
			starpu_heteroprio_set_nb_prios(sched_ctx_id, type, max_prio-min_prio+1);

	// Direct mapping
	int prio;
	for(prio=min_prio ; prio<=max_prio ; prio++)
	{
		// By default each type of devices uses 1 bucket and no slow factor
		for (type = 0; type < STARPU_NARCH; type++)
			if (starpu_worker_get_count_by_type(type) > 0)
				starpu_heteroprio_set_mapping(sched_ctx_id, type, prio, prio);
	}
}

/** stats of heteroprio when use_locality==1 */
#ifdef LAHETEROPRIO_PRINT_STAT
struct laheteropriostats
{
	long int nb_tasks;
	long int nb_tasks_per_worker[128][HETEROPRIO_MAX_PRIO];
	long int nb_tasks_per_wgroup[LAHETEROPRIO_MAX_WORKER_GROUPS][HETEROPRIO_MAX_PRIO];
	long int task_skipt_due_to_factor_per_worker[128][HETEROPRIO_MAX_PRIO];
	long int task_list_empty_per_worker[128][HETEROPRIO_MAX_PRIO];
	long int task_stolen_per_worker[128][HETEROPRIO_MAX_PRIO];
	long int task_stolen_in_wgroup[LAHETEROPRIO_MAX_WORKER_GROUPS][HETEROPRIO_MAX_PRIO];
	long int push_redirect[128+1][LAHETEROPRIO_MAX_WORKER_GROUPS];
	long int pop_redirect[128][LAHETEROPRIO_MAX_WORKER_GROUPS];
	long int push_to_use[128][PUSH_NB_AUTO];
};
struct laheteropriostats lastats;
#endif

static void check_heteroprio_mapping(struct _starpu_heteroprio_data *hp)
{
	//return 0;

	unsigned idx_prio;

	/* Ensure that information have been correctly filled */
	unsigned check_all_archs[HETEROPRIO_MAX_PRIO];
	memset(check_all_archs, 0, sizeof(unsigned)*HETEROPRIO_MAX_PRIO);
	unsigned arch_index;
	for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
	{
		STARPU_ASSERT(hp->nb_prio_per_arch_index[arch_index] <= HETEROPRIO_MAX_PRIO);

		unsigned check_archs[HETEROPRIO_MAX_PRIO];
		memset(check_archs, 0, sizeof(unsigned)*HETEROPRIO_MAX_PRIO);

		for(idx_prio = 0; idx_prio < hp->nb_prio_per_arch_index[arch_index]; ++idx_prio)
		{
			const unsigned mapped_prio = hp->prio_mapping_per_arch_index[arch_index][idx_prio];
			STARPU_ASSERT(mapped_prio <= HETEROPRIO_MAX_PRIO);
			STARPU_ASSERT(hp->buckets[mapped_prio].slow_factors_per_index[arch_index] >= 0.0);

			STARPU_ASSERT(hp->buckets[mapped_prio].valid_archs & starpu_heteroprio_types_to_arch(arch_index));

			check_archs[mapped_prio]      = 1;
			check_all_archs[mapped_prio] += 1;
		}
		for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
		{
			/* Ensure the current arch use a bucket or someone else can use it */
			STARPU_ASSERT(check_archs[idx_prio] == 1 || hp->buckets[idx_prio].valid_archs == 0
				      || (hp->buckets[idx_prio].valid_archs & ~starpu_heteroprio_types_to_arch(arch_index)) != 0);
		}
	}
	/* Ensure that if a valid_archs = (STARPU_CPU|STARPU_CUDA) then check_all_archs[] = 2 for example */

	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
	{
		unsigned nb_arch_on_bucket = 0;
		for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
		{
			if(hp->buckets[idx_prio].valid_archs & starpu_heteroprio_types_to_arch(arch_index))
			{
				nb_arch_on_bucket += 1;
			}
		}
		STARPU_ASSERT_MSG(check_all_archs[idx_prio] == nb_arch_on_bucket, "check_all_archs[idx_prio(%u)] = %u != nb_arch_on_bucket = %u\n", idx_prio, check_all_archs[idx_prio], nb_arch_on_bucket);
	}
}

static void starpu_autoheteroprio_add_task(struct _starpu_heteroprio_data *hp, const char name[CODELET_MAX_NAME_LENGTH], unsigned archs[STARPU_NB_TYPES])
{
	unsigned arch;
	for(arch=0;arch<STARPU_NB_TYPES;++arch)
	{
		if(archs[arch])
		{ // task is considered to be able to run on this arch
			starpu_heteroprio_set_mapping_hp(hp, arch, hp->found_codelet_names_on_arch[arch], hp->found_codelet_names_length);
			++hp->found_codelet_names_on_arch[arch];
			starpu_heteroprio_set_nb_prios_hp(hp, arch, hp->found_codelet_names_on_arch[arch]);
		}
	}

	// TODO: remap laheteroprio policy
	strncpy(&hp->found_codelet_names[hp->found_codelet_names_length][0], name, CODELET_MAX_NAME_LENGTH);
	++hp->found_codelet_names_length;

	check_heteroprio_mapping(hp); // ensures that priorities are correctly mapped
}

#define _HETEROPRIO_DIR_MAXLEN 256
static char _heteroprio_data_dir[_HETEROPRIO_DIR_MAXLEN];

/* Try to get the name of the program, to get specific data file for each program */
#ifdef STARPU_HAVE_PROGRAM_INVOCATION_SHORT_NAME
#define _progname program_invocation_short_name
#else
#define _progname "UNKNOWN_PROGRAM"
#endif

static char *_starpu_heteroprio_get_data_dir()
{
	static int directory_existence_was_tested = 0;

	if(!directory_existence_was_tested)
	{
		char *path = starpu_getenv("STARPU_HETEROPRIO_DATA_DIR");
		if(path)
		{
			snprintf(_heteroprio_data_dir, _HETEROPRIO_DIR_MAXLEN, "%s/", path);
		}
		else
		{
			snprintf(_heteroprio_data_dir, _HETEROPRIO_DIR_MAXLEN, "%s/heteroprio/", _starpu_get_perf_model_dir_default());
		}

		_starpu_mkpath_and_check(_heteroprio_data_dir, S_IRWXU);

		directory_existence_was_tested = 1;
	}

	return _heteroprio_data_dir;
}

static void starpu_autoheteroprio_fetch_task_data(struct _starpu_heteroprio_data *hp)
{
	const char *custom_path = starpu_getenv("STARPU_HETEROPRIO_DATA_FILE");

#ifndef STARPU_HAVE_PROGRAM_INVOCATION_SHORT_NAME
	if(!custom_path)
	{
		_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio can't determine the program's name to automatically store performance data. "
			"You can specify a path to store program associated data with STARPU_HETEROPRIO_DATA_FILE\n");
	}
#endif

	char path[_HETEROPRIO_DIR_MAXLEN+6];
	if(!custom_path)
	{
		snprintf(path, _HETEROPRIO_DIR_MAXLEN+6, "%s/%s.data", _starpu_heteroprio_get_data_dir(),
			_progname);
	}

	FILE *autoheteroprio_file;
	int locked;

	autoheteroprio_file = fopen(custom_path ? custom_path : path, "r");
	if(autoheteroprio_file == NULL)
	{
		// unable to open heteroprio data file
		return;
	}
	locked = _starpu_frdlock(autoheteroprio_file) == 0;

	_starpu_drop_comments(autoheteroprio_file);

	unsigned number_of_archs;
	unsigned archs[STARPU_NB_TYPES];
	unsigned arch_ind, arch_type;
	int c;

	if(fscanf(autoheteroprio_file, "%u", &number_of_archs) != 1)
	{
		fclose(autoheteroprio_file);
		_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing a number of architectures\n");
		return;
	}

	// Count number of archs not available in this version
	const unsigned ignored_archs = STARPU_MAX(0, (int) (number_of_archs - STARPU_NB_TYPES));

	const unsigned supported_archs = STARPU_MIN(STARPU_NB_TYPES, number_of_archs);

	// Reading list of supported architectures
	for(arch_ind = 0; arch_ind < supported_archs; ++arch_ind)
	{
		if(fscanf(autoheteroprio_file, "%u", &arch_type) != 1)
		{
			fclose(autoheteroprio_file);
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture id\n");
			return;
		}
		archs[arch_ind] = arch_type;
	}
	for(arch_ind = 0; arch_ind < ignored_archs; ++arch_ind)
	{
		if(fscanf(autoheteroprio_file, "%u", &arch_type) != 1)
		{
			fclose(autoheteroprio_file);
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture id\n");
			return;
		}
	}
	if(getc(autoheteroprio_file) != '\n')
	{
		fclose(autoheteroprio_file);
		_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is improperly formatted\n");
		return;
	}

	_starpu_drop_comments(autoheteroprio_file);

	// Reading architectures average times
	double avg_arch_busy_time, avg_arch_free_time;
	for(arch_ind = 0; arch_ind < supported_archs; ++arch_ind)
	{
		if(fscanf(autoheteroprio_file, "%lf %lf", &avg_arch_busy_time, &avg_arch_free_time) != 2)
		{
			fclose(autoheteroprio_file);
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture average times id\n");
			return;
		}
		else if(arch_ind < STARPU_NB_TYPES && archs[arch_ind] < STARPU_NB_TYPES)
		{
			hp->average_arch_busy_time[archs[arch_ind]] = avg_arch_busy_time;
			hp->average_arch_free_time[archs[arch_ind]] = avg_arch_free_time;
		}
	}
	for(arch_ind = 0; arch_ind < ignored_archs; ++arch_ind)
	{
		if(fscanf(autoheteroprio_file, "%lf %lf", &avg_arch_busy_time, &avg_arch_free_time) != 2)
		{
			fclose(autoheteroprio_file);
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture average times id\n");
			return;
		}
	}
	if(getc(autoheteroprio_file) != '\n')
	{
		fclose(autoheteroprio_file);
		_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is improperly formatted\n");
		return;
	}

	_starpu_drop_comments(autoheteroprio_file);

	unsigned codelet_archs[STARPU_NB_TYPES];
	unsigned codelet_exec_archs[STARPU_NB_TYPES];
	unsigned prio = hp->found_codelet_names_length;
	char codelet_name[CODELET_MAX_NAME_LENGTH+1];
	unsigned ignored_lines, arch_can_execute;

	// Read saved stats for each codelet
	while(fscanf(autoheteroprio_file, "%" STRINGIFY(CODELET_MAX_NAME_LENGTH) "s", codelet_name) == 1)
	{
		memset(codelet_exec_archs, 0, STARPU_NB_TYPES * sizeof(unsigned));

		// Read compatible architectures
		ignored_lines = 0;
		for(arch_ind = 0; arch_ind < supported_archs; ++arch_ind)
		{
			if(fscanf(autoheteroprio_file, "%u", &arch_can_execute) != 1)
			{
				fclose(autoheteroprio_file);
				_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture information for a codelet\n");
				return;
			}
			else if(arch_ind < STARPU_NB_TYPES)
			{
				codelet_archs[arch_ind] = arch_can_execute;
				if(archs[arch_ind] < STARPU_NB_TYPES)
					codelet_exec_archs[archs[arch_ind]] = arch_can_execute;
			}
		}
		for(arch_ind = 0; arch_ind < ignored_archs; ++arch_ind)
		{
			if(fscanf(autoheteroprio_file, "%u", &arch_can_execute) != 1)
			{
				fclose(autoheteroprio_file);
				_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is missing an architecture information for a codelet\n");
				return;
			}
			else if(arch_can_execute)
			{
				ignored_lines += 1;
			}
		}

		// Read general codelet data
		if(fscanf(autoheteroprio_file, "%lf %u %u %lf %u %u %lf %u %lf %u",
			&hp->prio_average_NOD[prio], &hp->prio_average_NOD_count[prio],
			&hp->prio_average_URT_count[prio],
			&hp->prio_overall_proportion[prio], &hp->prio_overall_proportion_count[prio],
			&hp->prio_arch_proportion_count[prio],
			&hp->prio_average_successors_best_time_sum[prio], &hp->prio_average_successors_best_time_sum_count[prio],
			&hp->prio_average_best[prio], &hp->prio_average_best_count[prio]
			) != 10)
		{
			fclose(autoheteroprio_file);
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is improperly formatted\n");
			return;
		}

		// Read architecture specific data
		for(arch_ind = 0; arch_ind < supported_archs; ++arch_ind)
		{
			if(codelet_archs[arch_ind] && archs[arch_ind] < STARPU_NB_TYPES)
			{
				if(fscanf(autoheteroprio_file, "%lf %lf %u %lf\n",
					&hp->prio_average_URT[archs[arch_ind]][prio],
					&hp->prio_average_time_arch[archs[arch_ind]][prio], &hp->prio_average_time_arch_count[archs[arch_ind]][prio],
					&hp->prio_arch_proportion[archs[arch_ind]][prio]
					) != 4)
				{
					fclose(autoheteroprio_file);
					_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file is improperly formatted\n");
					return;
				}

				if(hp->prio_average_time_arch_count[archs[arch_ind]][prio] > 0)
					hp->prio_arch_has_time_info[archs[arch_ind]][prio] = 1;
			}
			else if(codelet_archs[arch_ind] && archs[arch_ind] >= STARPU_NB_TYPES)
			{
				while((c = getc(autoheteroprio_file)) != '\n')
					if(c == EOF)
					{
						fclose(autoheteroprio_file);
						_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file ended abruptly\n");
						return;
					}
			}
		}
		for(arch_ind = 0; arch_ind < ignored_lines; ++arch_ind)
		{
			while((c = getc(autoheteroprio_file)) != '\n')
				if(c == EOF)
				{
					fclose(autoheteroprio_file);
					_STARPU_MSG("[HETEROPRIO][INITIALIZATION] Warning, autoheteroprio's data file ended abruptly\n");
					return;
				}
		}

		starpu_autoheteroprio_add_task(hp, codelet_name, codelet_exec_archs);
		prio = hp->found_codelet_names_length; // update current prio (+1)

		_starpu_drop_comments(autoheteroprio_file);
	}

	if(locked)
		_starpu_frdunlock(autoheteroprio_file);
	fclose(autoheteroprio_file);
}

static void starpu_autoheteroprio_save_task_data(struct _starpu_heteroprio_data *hp)
{
	const char *custom_path = starpu_getenv("STARPU_HETEROPRIO_DATA_FILE");

	char path[_HETEROPRIO_DIR_MAXLEN+6];
	if(!custom_path)
	{
		snprintf(path, _HETEROPRIO_DIR_MAXLEN+6, "%s/%s.data", _starpu_heteroprio_get_data_dir(),
			_progname);
	}

	FILE *autoheteroprio_file;
	int locked;

	autoheteroprio_file = fopen(custom_path ? custom_path : path, "w+");
	if(autoheteroprio_file == NULL)
	{
		_STARPU_MSG("[HETEROPRIO][DEINITIALIZATION] Warning: unable to save task data\n");
		return;
	}
	locked = _starpu_fwrlock(autoheteroprio_file) == 0;
	fseek(autoheteroprio_file, 0, SEEK_SET);
	_starpu_fftruncate(autoheteroprio_file, 0);

	unsigned number_of_archs = 0;
	unsigned is_arch_used[STARPU_NB_TYPES];
	unsigned arch_ind;

	fprintf(autoheteroprio_file, "##################\n");
	fprintf(autoheteroprio_file, "# Known architectures\n");
	fprintf(autoheteroprio_file, "# number_of_archs arch_ids (");
	for(arch_ind = 0; arch_ind < STARPU_NB_TYPES; ++arch_ind)
	{
		if(hp->found_codelet_names_on_arch[arch_ind] > 0)
		{
			// Architecture was used
			is_arch_used[arch_ind] = 1;
			number_of_archs += 1;
			fprintf(autoheteroprio_file, "%s - %u, ",
				starpu_worker_get_type_as_string(arch_ind), arch_ind);
		}
		else
			is_arch_used[arch_ind] = 0;
	}
	fprintf(autoheteroprio_file, ")\n");

	// List of used architectures designed by their id
	fprintf(autoheteroprio_file, "%u", number_of_archs);
	for(arch_ind = 0; arch_ind < STARPU_NB_TYPES; ++arch_ind)
	{
		if(is_arch_used[arch_ind])
			fprintf(autoheteroprio_file, " %u", arch_ind);
	}
	fprintf(autoheteroprio_file, "\n");

	fprintf(autoheteroprio_file, "##################\n");
	fprintf(autoheteroprio_file, "# Busy/Free proportion per architecture\n");
	fprintf(autoheteroprio_file, "# ARCH1_busy_time ARCH1_free_time ... ARCHn_busy_time ARCHn_free_time\n");

	// Busy and free proportion per architecture
	for(arch_ind = 0; arch_ind < STARPU_NB_TYPES; ++arch_ind)
	{
		if(is_arch_used[arch_ind])
			fprintf(autoheteroprio_file, " %lf %lf",
				hp->average_arch_busy_time[arch_ind], hp->average_arch_free_time[arch_ind]);
	}
	fprintf(autoheteroprio_file, "\n");

	fprintf(autoheteroprio_file, "##################\n");
	fprintf(autoheteroprio_file, "# Codelets specific data\n");
	fprintf(autoheteroprio_file, "# codelet_name arch_1_can_exec ... arch_n_can_exec\n");
	fprintf(autoheteroprio_file, "# average_NOD average_NOD_count average_URT_count overall_proportion overall_proportion_count arch_proportion_count avg_best_successor_time avg_best_successor_time_count prio_average_best prio_average_best_count\n");
	fprintf(autoheteroprio_file, "# for each arch which can exec: average_URT_ARCH average_time_ARCH average_time_ARCH_count ARCH_proportion\n");
	fprintf(autoheteroprio_file, "##########\n");

	unsigned prio;
	unsigned codelet_archs[STARPU_NB_TYPES];

	for(prio = 0; prio < hp->found_codelet_names_length; ++prio)
	{
		fprintf(autoheteroprio_file, "%s", hp->found_codelet_names[prio]);

		// Indicate if each can execute codelet
		for(arch_ind = 0; arch_ind < STARPU_NB_TYPES; ++arch_ind)
		{
			if(is_arch_used[arch_ind])
			{
				codelet_archs[arch_ind] = arch_can_execute_prio(hp, arch_ind, prio);
				fprintf(autoheteroprio_file, " %u", codelet_archs[arch_ind]);
			}
			else
				codelet_archs[arch_ind] = 0;
		}
		fprintf(autoheteroprio_file, "\n");

		// Non specific codelet data
		fprintf(autoheteroprio_file, "%lf %u %u %lf %u %u %lf %u %lf %u\n",
			hp->prio_average_NOD[prio], hp->prio_average_NOD_count[prio],
			hp->prio_average_URT_count[prio],
			hp->prio_overall_proportion[prio], hp->prio_overall_proportion_count[prio],
			hp->prio_arch_proportion_count[prio],
			hp->prio_average_successors_best_time_sum[prio], hp->prio_average_successors_best_time_sum_count[prio],
			hp->prio_average_best[prio], hp->prio_average_best_count[prio]);

		// Architecture specific data
		for(arch_ind = 0; arch_ind < STARPU_NB_TYPES; ++arch_ind)
		{
			if(codelet_archs[arch_ind])
			{
				fprintf(autoheteroprio_file, "%lf %lf %u %lf\n",
					hp->prio_average_URT[arch_ind][prio],
					hp->prio_average_time_arch[arch_ind][prio], hp->prio_average_time_arch_count[arch_ind][prio],
					hp->prio_arch_proportion[arch_ind][prio]);
			}
		}

		fprintf(autoheteroprio_file, "#####\n");
	}

	if(locked)
		_starpu_fwrunlock(autoheteroprio_file);
	fclose(autoheteroprio_file);
}

static void initialize_heteroprio_policy(unsigned sched_ctx_id)
{
#ifdef LAHETEROPRIO_PRINT_STAT
	memset(&lastats, 0, sizeof(lastats));
#endif

	int max_priority = starpu_sched_ctx_get_max_priority(sched_ctx_id);
	if(max_priority < HETEROPRIO_MAX_PRIO-1)
	{
		starpu_sched_ctx_set_max_priority(sched_ctx_id, HETEROPRIO_MAX_PRIO-1);
		_STARPU_DISP("[HETEROPRIO][INITIALIZATION] Max priority has been set to %d\n", HETEROPRIO_MAX_PRIO-1);
	}
	int min_priority = starpu_sched_ctx_get_min_priority(sched_ctx_id);
	if(min_priority > 0)
	{
		starpu_sched_ctx_set_min_priority(sched_ctx_id, 0);
		_STARPU_DISP("[HETEROPRIO][INITIALIZATION] Min priority has been set to 0\n");
	}

	/* Alloc the scheduler data  */
	struct _starpu_heteroprio_data *hp;
	_STARPU_MALLOC(hp, sizeof(struct _starpu_heteroprio_data));
	memset(hp, 0, sizeof(*hp));

	hp->use_locality = use_la_mode = starpu_getenv_number_default("STARPU_HETEROPRIO_USE_LA", 0);
	_STARPU_DISP("[HETEROPRIO] Data locality : %s\n", hp->use_locality?"ENABLED":"DISABLED");

	hp->codelet_grouping_strategy = use_auto_mode = starpu_getenv_number_default("STARPU_HETEROPRIO_CODELET_GROUPING_STRATEGY", 0);
	switch(hp->codelet_grouping_strategy)
	{
		case BY_PERF_MODEL_OR_NAME:
			_STARPU_DISP("[HETEROPRIO] Codelet grouping strategy : BY_PERF_MODEL_OR_NAME\n");
			break;
		case BY_NAME_ONLY:
			_STARPU_DISP("[HETEROPRIO] Codelet grouping strategy : BY_NAME\n");
			break;
		default:
			_STARPU_DISP("[HETEROPRIO] Codelet grouping strategy : UNKNOWN\n");

			hp->codelet_grouping_strategy = BY_PERF_MODEL_OR_NAME; // setting to default
	}

	hp->use_auto_calibration = use_auto_mode = starpu_getenv_number_default("STARPU_HETEROPRIO_USE_AUTO_CALIBRATION", 1);
	_STARPU_DISP("[HETEROPRIO] Auto calibration : %s\n", hp->use_auto_calibration?"ENABLED":"DISABLED");
	if(hp->use_auto_calibration)
	{
		const int ordering_policy = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY", STARPU_HETEROPRIO_URT_DOT_DIFF_4);
		STARPU_ASSERT_MSG(ordering_policy < STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY_COUNT, "STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY must be < %d.\n", STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY_COUNT);
		STARPU_ASSERT_MSG(ordering_policy >= 0, "STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY must be >= 0.\n");
		hp->autoheteroprio_priority_ordering_policy = ordering_policy;
		_STARPU_DISP("[AUTOHETEROPRIO] Priority ordering policy : %s\n", &starpu_autoheteroprio_priority_ordering_policy_names[hp->autoheteroprio_priority_ordering_policy][0]);

		hp->priority_ordering_interval = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_ORDERING_INTERVAL", 32);

		hp->freeze_data_gathering = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_FREEZE_GATHERING", 0);
		_STARPU_DISP("[AUTOHETEROPRIO] Data gathering : %s\n", !hp->freeze_data_gathering?"ENABLED":"DISABLED");

		hp->autoheteroprio_print_prio_after_ordering = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_PRINT_AFTER_ORDERING", 0);
		_STARPU_DISP("[AUTOHETEROPRIO] Print after ordering : %s\n", hp->autoheteroprio_print_prio_after_ordering?"ENABLED":"DISABLED");

		hp->autoheteroprio_print_data_on_update = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_PRINT_DATA_ON_UPDATE", 0);
		_STARPU_DISP("[AUTOHETEROPRIO] Print on update : %s\n", hp->autoheteroprio_print_data_on_update?"ENABLED":"DISABLED");

		hp->autoheteroprio_time_estimation_policy = starpu_getenv_number_default("STARPU_AUTOHETEROPRIO_TIME_ESTIMATION_POLICY", 0);
	}

	starpu_bitmap_init(&hp->waiters);
	if(hp->use_locality)
	{
		hp->pushStrategySet = getEnvAdvPush();
		if(hp->pushStrategySet != PUSH_AUTO)
		{
			hp->pushStrategyToUse = hp->pushStrategySet;
		}
		else
		{
			hp->pushStrategyToUse = PUSH_LS_SDHB;
		}
	}

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)hp);

	STARPU_PTHREAD_MUTEX_INIT(&hp->policy_mutex, NULL);
	if(hp->use_locality)
	{
		STARPU_PTHREAD_MUTEX_INIT(&hp->push_history_mutex, NULL);
	}
	if(hp->use_auto_calibration)
	{
		STARPU_PTHREAD_MUTEX_INIT(&hp->auto_calibration_mutex, NULL);
	}

	// get environment hyperparameters

	hp->NTnodPond = starpu_getenv_float_default("STARPU_HETEROPRIO_NOD_TIME_COMBINATION_NOD_MULTIPLIER", 0.3);
	hp->NTexpVal = starpu_getenv_float_default("STARPU_HETEROPRIO_NOD_TIME_COMBINATION_EXP_SELECTIVITY", 0.5);
	hp->BNexpVal = starpu_getenv_float_default("STARPU_HETEROPRIO_BEST_NODS_SCORE_EXP_SELECTIVITY", 0.5);
	hp->URTurt = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_URT_MULTIPLIER", 0.5);
	hp->URT2urt = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_2_URT_MULTIPLIER", 0.5);
	hp->URT2prop = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_2_ARCH_NEED_MULTIPLIER", 2.0);
	hp->and2pond = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_2_ARCH_NEED_MULTIPLIER", 1.0);
	hp->and3pond = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_3_ARCH_NEED_MULTIPLIER", 1.0);
	hp->and4pond = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_4_ARCH_NEED_MULTIPLIER", 1.0);
	hp->and5xoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_5_NOD_OFFSET", 1.3);
	hp->and5yoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_5_ARCH_DIFF_OFFSET", 1.0);
	hp->and9xoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_9_NOD_OFFSET", 1.3);
	hp->and9yoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_9_ARCH_DIFF_OFFSET", 1.0);
	hp->and10xoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_AURT_DOT_DIFF_10_NOD_OFFSET", 1.3);
	hp->and10yoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_10_ARCH_DIFF_OFFSET", 1.0);
	hp->and11xoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_11_NOD_OFFSET", 1.3);
	hp->and11yoffset = starpu_getenv_float_default("STARPU_HETEROPRIO_URT_DOT_DIFF_11_ARCH_DIFF_OFFSET", 1.0);
	hp->ANTnodPond = starpu_getenv_float_default("STARPU_HETEROPRIO_URTS_TIME_COMBINATION_NOD_MULTIPLIER", 0.3);
	hp->ANTexpVal = starpu_getenv_float_default("STARPU_HETEROPRIO_URTS_TIME_COMBINATION_EXP_SELECTIVITY", 0.5);

	unsigned idx_prio;
	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
		_heteroprio_bucket_init(&hp->buckets[idx_prio]);

	if(hp->use_locality)
	{
		hp->nb_wgroups = LAHETEROPRIO_MAX_WORKER_GROUPS;
		unsigned idx_wgroup;
		for(idx_wgroup = 0 ; idx_wgroup < LAHETEROPRIO_MAX_WORKER_GROUPS ; ++idx_wgroup)
		{
			hp->arch_of_wgroups[idx_wgroup] = STARPU_ANY_WORKER; // We set STARPU_ANY_WORKER = default (none) value
		}
		memset(hp->bucket_mapping_per_arch_index, -1, sizeof(unsigned)*STARPU_NB_TYPES*HETEROPRIO_MAX_PRIO);
	}

	void (*callback_sched)(unsigned) = starpu_sched_ctx_get_sched_policy_callback(sched_ctx_id);

	if(callback_sched)
	{
		if(hp->use_auto_calibration)
		{
			_STARPU_DISP("[HETEROPRIO][INITIALIZATION] Warning: a custom sched init function has been detected while being in auto calibration mode (STARPU_HETEROPRIO_USE_AUTO_CALIBRATION). Custom changes to priority mapping will be overwritten.\n");
		}
		callback_sched(sched_ctx_id);
	}
	else
	{
		default_init_sched(sched_ctx_id);
	}

	check_heteroprio_mapping(hp);

	if(hp->use_auto_calibration)
	{
		unsigned arch;
		for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
		{
			hp->prio_average_NOD[idx_prio] = 0.f;
			hp->prio_average_NOD_count[idx_prio] = 0;

			hp->prio_average_URT_count[idx_prio] = 0;

			hp->prio_overall_proportion[idx_prio] = 0.f;
			hp->prio_overall_proportion_count[idx_prio] = 0;

			hp->prio_arch_proportion_count[idx_prio] = 0;

			hp->prio_average_successors_best_time_sum[idx_prio] = 0.f;
			hp->prio_average_successors_best_time_sum_count[idx_prio] = 0;

			hp->prio_average_best[idx_prio] = 0.f;
			hp->prio_average_best_count[idx_prio] = 0;

			for(arch=0;arch<STARPU_NB_TYPES;++arch)
			{
				hp->prio_average_URT[arch][idx_prio] = 0.f;

				hp->prio_average_time_arch[arch][idx_prio] = 0.f;
				hp->prio_average_time_arch_count[arch][idx_prio] = 0;

				hp->prio_arch_proportion[arch][idx_prio] = 0.f;

				if(arch != STARPU_CPU_WORKER)
				{
					starpu_heteroprio_set_arch_slow_factor_hp(hp, arch, idx_prio, 1.0f);
				}
			}
			starpu_heteroprio_set_faster_arch_hp(hp, STARPU_CPU_WORKER, idx_prio);
		}

		starpu_heteroprio_clear_mapping_hp(hp);
		for(arch=0;arch<STARPU_NB_TYPES;++arch)
		{
			starpu_heteroprio_set_nb_prios(sched_ctx_id, arch, 0);
		}

		starpu_autoheteroprio_fetch_task_data(hp);

		if(!hp->freeze_data_gathering)
		{
			_starpu_graph_record = 1; // allow starpu graph recording
		}
	}
}

static void register_arch_times(struct _starpu_heteroprio_data *hp, unsigned arch, double busy_time, double free_time);

static void deinitialize_heteroprio_policy(unsigned sched_ctx_id)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	/* Ensure there are no more tasks */
	STARPU_ASSERT(hp->total_tasks_in_buckets == 0);
	unsigned arch_index;
	for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
	{
		if(hp->use_locality)
		{
			STARPU_ASSERT(hp->nb_remaining_tasks_per_arch_index[arch_index] == 0);
		}
		else
		{
			STARPU_ASSERT(hp->nb_remaining_tasks_per_arch_index[arch_index] == 0);
			STARPU_ASSERT(hp->nb_prefetched_tasks_per_arch_index[arch_index] == 0);
		}
	}

	unsigned idx_prio;
	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
	{
		STARPU_ASSERT(hp->buckets[idx_prio].tasks_queue_ntasks == 0); // potentially not wanted if use_la==0
		_heteroprio_bucket_release(&hp->buckets[idx_prio]);
	}

	if(hp->use_locality)
	{
#ifdef LAHETEROPRIO_PRINT_STAT
		_STARPU_MSG("[LASTATS] nb tasks %ld\n", lastats.nb_tasks);
		{
			_STARPU_MSG("[LASTATS] Tasks pushed per workers of kind:\n");
			unsigned nb_tasks = 0;
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				const unsigned worker_arch = hp->workers_heteroprio[worker_id].arch_index;
				const unsigned nb_prios = hp->nb_prio_per_arch_index[worker_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.nb_tasks_per_worker[worker_id][idx_prio]);
					nb_tasks += lastats.nb_tasks_per_worker[worker_id][idx_prio];
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] Total tasks pushed per workers of kind: %u\n\n", nb_tasks);
		}
		{
			_STARPU_MSG("[LASTATS] Tasks pushed per workers to mem node:\n");
			unsigned nb_tasks = 0;
			_STARPU_MSG("[LASTATS] Master: ");
			unsigned idx_mem;
			for (idx_mem = 0; idx_mem < hp->nb_wgroups; ++idx_mem)
			{
				fprintf(stderr, "[%3u] %9ld  ", idx_mem, lastats.push_redirect[0][idx_mem]);
				nb_tasks += lastats.push_redirect[0][idx_mem];
			}
			fprintf(stderr, "\n");
			_STARPU_MSG("[LASTATS] Total tasks pushed per workers to mem node: %u\n\n", nb_tasks);
		}
		{
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				_STARPU_MSG("[LASTATS] %u: ", worker_id);
				unsigned idx_mem;
				for (idx_mem = 0; idx_mem < hp->nb_wgroups; ++idx_mem)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_mem, lastats.push_redirect[worker_id + 1][idx_mem]);
				}
				fprintf(stderr, "\n");
			}
			fprintf(stderr, "\n");
		}
		{
			_STARPU_MSG("[LASTATS] Tasks per wgroup:\n");
			unsigned nb_tasks = 0;
			unsigned idx_wgroup;
			for (idx_wgroup = 0; idx_wgroup < hp->nb_wgroups; ++idx_wgroup)
			{
				const unsigned wgroup_arch = hp->arch_of_wgroups[idx_wgroup];
				const unsigned nb_prios = hp->nb_prio_per_arch_index[wgroup_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.nb_tasks_per_wgroup[idx_wgroup][idx_prio]);
					nb_tasks += lastats.nb_tasks_per_wgroup[idx_wgroup][idx_prio];
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] Total tasks pushed per wgroup: %u\n\n", nb_tasks);
		}
		{
			_STARPU_MSG("[LASTATS] Tasks skipt per workers:\n");
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				const unsigned worker_arch = hp->workers_heteroprio[worker_id].arch_index;
				const unsigned nb_prios = hp->nb_prio_per_arch_index[worker_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.task_skipt_due_to_factor_per_worker[worker_id][idx_prio]);
				}
				fprintf(stderr, "\n");
			}
			fprintf(stderr, "\n");
		}
		{
			_STARPU_MSG("[LASTATS] Tasks list empty per workers:\n");
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				const unsigned worker_arch = hp->workers_heteroprio[worker_id].arch_index;
				const unsigned nb_prios = hp->nb_prio_per_arch_index[worker_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.task_list_empty_per_worker[worker_id][idx_prio]);
				}
				fprintf(stderr, "\n");
			}
			fprintf(stderr, "\n");
		}
		{
			_STARPU_MSG("[LASTATS] Tasks stolen per workers:\n");
			unsigned nb_tasks = 0;
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				const unsigned worker_arch = hp->workers_heteroprio[worker_id].arch_index;
				const unsigned nb_prios = hp->nb_prio_per_arch_index[worker_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.task_stolen_per_worker[worker_id][idx_prio]);
					nb_tasks += lastats.task_stolen_per_worker[worker_id][idx_prio];
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] Total tasks stolen per worker: %u\n\n", nb_tasks);
		}
		{
			_STARPU_MSG("[LASTATS] Tasks stolen in wgroup:\n");
			unsigned nb_tasks = 0;
			unsigned idx_wgroup;
			for (idx_wgroup = 0; idx_wgroup < hp->nb_wgroups; ++idx_wgroup)
			{
				const unsigned wgroup_arch = hp->arch_of_wgroups[idx_wgroup];
				const unsigned nb_prios = hp->nb_prio_per_arch_index[wgroup_arch];
				_STARPU_MSG("[LASTATS] ");
				for (idx_prio = 0; idx_prio < nb_prios; ++idx_prio)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_prio, lastats.task_stolen_in_wgroup[idx_wgroup][idx_prio]);
					nb_tasks += lastats.task_stolen_in_wgroup[idx_wgroup][idx_prio];
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] Total tasks stolen in wgroup: %u\n\n", nb_tasks);
		}
		{
			_STARPU_MSG("[LASTATS] Tasks push/pop different wgroup:\n");
			unsigned nb_tasks = 0;
			unsigned worker_id;
			for (worker_id = 0; worker_id < starpu_worker_get_count(); ++worker_id)
			{
				_STARPU_MSG("[LASTATS] %u: ", worker_id);
				unsigned idx_mem;
				for (idx_mem = 0; idx_mem < hp->nb_wgroups; ++idx_mem)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_mem, lastats.pop_redirect[worker_id][idx_mem]);
					nb_tasks += lastats.pop_redirect[worker_id][idx_mem];
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] Total tasks push/pop different wgroup: %u\n\n", nb_tasks);
		}
		{
			_STARPU_MSG("[LASTATS] push strategy used:\n");
			unsigned worker_id;
			unsigned counter[PUSH_NB_AUTO] = { 0 };
			unsigned idx_more_used = 0;
			for (worker_id = 0; worker_id <= starpu_worker_get_count(); ++worker_id)
			{
				_STARPU_MSG("[LASTATS] %u: ", worker_id);
				unsigned idx_strategy;
				for (idx_strategy = 0; idx_strategy < PUSH_NB_AUTO; ++idx_strategy)
				{
					fprintf(stderr, "[%3u] %9ld  ", idx_strategy, lastats.push_to_use[worker_id][idx_strategy]);
					counter[idx_strategy] += lastats.push_to_use[worker_id][idx_strategy];
					if (counter[idx_strategy] > counter[idx_more_used])
					{
						idx_more_used = idx_strategy;
					}
				}
				fprintf(stderr, "\n");
			}
			_STARPU_MSG("[LASTATS] More used push: %u\n\n", idx_more_used);
		}
		{
			_STARPU_MSG("[LASTATS] correct MN pushes:\n");
			unsigned idx_strategy;
			for (idx_strategy = 0; idx_strategy < PUSH_NB_AUTO; ++idx_strategy)
			{
				_STARPU_MSG("[LASTATS][%u] %u \n", idx_strategy, hp->pushStrategyHistory[idx_strategy]);
			}
		}
#endif
	}

	STARPU_PTHREAD_MUTEX_DESTROY(&hp->policy_mutex);
	if(hp->use_locality)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&hp->push_history_mutex);
	}
	if(hp->use_auto_calibration)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&hp->auto_calibration_mutex);
	}
	if(hp->use_auto_calibration && !hp->freeze_data_gathering)
	{
		// update autoheteroprio data with free and busy worker time
		for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
		{
			register_arch_times(hp, arch_index, hp->current_arch_busy_time[arch_index], hp->current_arch_free_time[arch_index]);
		}

		starpu_autoheteroprio_save_task_data(hp);
	}

	_starpu_graph_record = 0; // disable starpu graph recording (that may have been activated due to hp->use_auto_calibration)

	free(hp);
}

static void add_workers_heteroprio_policy(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	// Retrieve current time to set as starting time for each worker
	struct timespec tsnow;
	_starpu_clock_gettime(&tsnow);
	const double now = starpu_timing_timespec_to_us(&tsnow);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];

		memset(&hp->workers_heteroprio[workerid], 0, sizeof(hp->workers_heteroprio[workerid]));
		if(!hp->use_locality)
		{
			/* if the worker has already belonged to this context
			   the queue and the synchronization variables have been already initialized */
			starpu_st_prio_deque_init(&hp->workers_heteroprio[workerid].tasks_queue);
		}

		enum starpu_worker_archtype arch_index = starpu_worker_get_type(workerid);
		hp->workers_heteroprio[workerid].arch_index = arch_index;
		hp->workers_heteroprio[workerid].arch_type = starpu_heteroprio_types_to_arch(arch_index);
		hp->nb_workers_per_arch_index[hp->workers_heteroprio[workerid].arch_index]++;

		hp->last_hook_exec_time[workerid] = now;
	}
}

static void remove_workers_heteroprio_policy(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	if(!hp->use_locality)
	{
		unsigned i;

		for (i = 0; i < nworkers; i++)
		{
			int workerid = workerids[i];
			starpu_st_prio_deque_destroy(&hp->workers_heteroprio[workerid].tasks_queue);
		}
	}
}

static unsigned get_best_mem_node(struct starpu_task *task, struct _starpu_heteroprio_data *hp, const enum laheteroprio_push_strategy pushStrategy)
{
	const unsigned workerid = starpu_worker_get_id();
	unsigned best_mem_node;

	STARPU_ASSERT(task != NULL);

	if (pushStrategy != PUSH_WORKER)
	{
		if(!hp->warned_change_nb_memory_nodes && starpu_memory_nodes_get_count() != hp->nb_memory_nodes)
		{
			_STARPU_MSG("[HETEROPRIO][INITIALIZATION][get_best_mem_node] Warning: current memory node number is different from the one retrieved at initialization.\n\
This warning will only be displayed once.\n");
			hp->warned_change_nb_memory_nodes = 1;
		}

		const unsigned nnodes = hp->nb_memory_nodes; // == starpu_memory_nodes_get_count() if number of mem nodes didn't change during execution

		if (pushStrategy == PUSH_LcS)
		{
			int node_to_worker[LAHETEROPRIO_MAX_WORKER_GROUPS];
			unsigned idx_worker;
			for (idx_worker = 0; idx_worker < starpu_worker_get_count(); ++idx_worker)
			{
			 	// overwrite, we simply need one worker per mem node
				node_to_worker[starpu_worker_get_memory_node(idx_worker)] = idx_worker;
			}
			double bestTransferTime = starpu_task_expected_data_transfer_time_for(task, node_to_worker[0]);
			best_mem_node = 0;
			unsigned idx_node;
			for (idx_node = 1; idx_node < nnodes; ++idx_node)
			{
				const double transferTime = starpu_task_expected_data_transfer_time_for(task, node_to_worker[idx_node]);
				if (transferTime < bestTransferTime)
				{
					bestTransferTime = transferTime;
					best_mem_node = idx_node;
				}
			}
		}
		else if (pushStrategy == PUSH_LS_SDH || pushStrategy == PUSH_LS_SDH2)
		{
			size_t max_size_so_far = 0;
			unsigned idx_max_size = 0;
			const unsigned wgroupid = (workerid == (unsigned)-1 ? hp->master_tasks_queue_idx : hp->workers_laheteroprio_wgroup_index[workerid]);
			size_t data_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			assert(nnodes <= LAHETEROPRIO_MAX_WORKER_GROUPS);
			unsigned idx_data;
			for (idx_data = 0; idx_data < STARPU_TASK_GET_NBUFFERS(task); ++idx_data)
			{
				const starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, idx_data);
				const size_t raw_data_size = starpu_data_get_size(handle);
				const unsigned is_read = (STARPU_TASK_GET_MODE(task, idx_data) == STARPU_R);
				// Easy:
				size_t data_size;
				if (pushStrategy == PUSH_LS_SDH)
				{
					data_size = raw_data_size;
				}
				else
				{
					assert(pushStrategy == PUSH_LS_SDH2);
					data_size = (is_read ? raw_data_size : raw_data_size *raw_data_size);
				}
				unsigned idx_node;
				for (idx_node = 0; idx_node < nnodes; ++idx_node)
				{
					if (starpu_data_is_on_node(handle, idx_node))
					{
						data_per_mem_node[idx_node] += data_size;
						if (max_size_so_far < data_per_mem_node[idx_node] ||
							(max_size_so_far == data_per_mem_node[idx_node] &&
								idx_node == wgroupid))
						{
							max_size_so_far = data_per_mem_node[idx_node];
							idx_max_size = idx_node;
						}
					}
				}
			}
			best_mem_node = idx_max_size;
		}
		else if (pushStrategy == PUSH_LC_SMWB)
		{
			const unsigned wgroupid = (workerid == (unsigned)-1 ? hp->master_tasks_queue_idx : hp->workers_laheteroprio_wgroup_index[workerid]);
			assert(nnodes <= LAHETEROPRIO_MAX_WORKER_GROUPS);
			const unsigned N = STARPU_TASK_GET_NBUFFERS(task);
			unsigned data_exist_every_where[128] = { 0 };
			unsigned nb_data_exist_every_where = 0;
			{
				unsigned idx_data;
				for (idx_data = 0; idx_data < N; ++idx_data)
				{
					const starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, idx_data);
					data_exist_every_where[idx_data] = 1;
					unsigned idx_node;
					for (idx_node = 0; idx_node < nnodes; ++idx_node)
					{
						if (starpu_data_is_on_node(handle, idx_node))
						{
						 				// Ok
						}
						else
						{
							data_exist_every_where[idx_data] = 0;
							break;
						}
					}
					if (data_exist_every_where[idx_data])
					{
						nb_data_exist_every_where += 1;
					}
				}
			}
			assert(N <= 128);
			unsigned data_is_read[128] = { 0 };
			unsigned Nw = 0;
			size_t total_size = 0;
			size_t total_size_in_read = 0;
			size_t total_size_in_write = 0;
			size_t data_sizes[128] = { 0 };
			unsigned data_Ri[128] = { 0 };
			size_t data_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			size_t data_per_mem_node_in_read[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			size_t data_per_mem_node_in_write[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			unsigned nb_data_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			unsigned nb_data_in_w_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			{
				unsigned idx_data;
				for (idx_data = 0; idx_data < N; ++idx_data)
				{
					if (data_exist_every_where[idx_data] == 0)
					{
						const starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, idx_data);
						data_sizes[idx_data] = starpu_data_get_size(handle);
						data_is_read[idx_data] = (STARPU_TASK_GET_MODE(task, idx_data) == STARPU_R);
						total_size += data_sizes[idx_data];
						if (data_is_read[idx_data])
						{
							total_size_in_read += data_sizes[idx_data];
						}
						else
						{
							total_size_in_write += data_sizes[idx_data];
							Nw += 1;
						}
						unsigned idx_node;
						for (idx_node = 0; idx_node < nnodes; ++idx_node)
						{
							if (starpu_data_is_on_node(handle, idx_node))
							{
								data_Ri[idx_data] += 1;
								data_per_mem_node[idx_node] += data_sizes[idx_data];
								nb_data_per_mem_node[idx_node] += 1;
								if (data_is_read[idx_data])
								{
									data_per_mem_node_in_read[idx_node] += data_sizes[idx_data];
								}
								else
								{
									data_per_mem_node_in_write[idx_node] += data_sizes[idx_data];
									nb_data_in_w_per_mem_node[idx_node] += 1;
								}
							}
						}
					}
				}
			}
			double max_score_so_far = 0;
			unsigned idx_max_score = 0;
			unsigned idx_node;
			for (idx_node = 0; idx_node < nnodes; ++idx_node)
			{
				double current_score = 0;
				current_score = (data_per_mem_node_in_read[idx_node]) +
					1000. *(data_per_mem_node_in_write[idx_node] *nb_data_in_w_per_mem_node[idx_node]);
				if (max_score_so_far < current_score ||
					(max_score_so_far == current_score &&
						idx_node == wgroupid))
				{
					max_score_so_far = current_score;
					idx_max_score = idx_node;
				}
			}
			best_mem_node = idx_max_score;
		}
		else
		{
			const unsigned wgroupid = (workerid == (unsigned)-1 ? hp->master_tasks_queue_idx : hp->workers_laheteroprio_wgroup_index[workerid]);
			assert(nnodes <= LAHETEROPRIO_MAX_WORKER_GROUPS);
			const unsigned N = STARPU_TASK_GET_NBUFFERS(task);
			assert(N <= 128);
			unsigned data_is_read[128] = { 0 };
			unsigned Nw = 0;
			size_t total_size = 0;
			size_t total_size_in_read = 0;
			size_t total_size_in_write = 0;
			size_t data_sizes[128] = { 0 };
			unsigned data_Ri[128] = { 0 };
			size_t data_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			size_t data_per_mem_node_in_read[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			size_t data_per_mem_node_in_write[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			unsigned nb_data_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			unsigned nb_data_in_w_per_mem_node[LAHETEROPRIO_MAX_WORKER_GROUPS] = { 0 };
			{
				unsigned idx_data;
				for (idx_data = 0; idx_data < N; ++idx_data)
				{
					const starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, idx_data);
					data_sizes[idx_data] = starpu_data_get_size(handle);
					data_is_read[idx_data] = (STARPU_TASK_GET_MODE(task, idx_data) == STARPU_R);
					total_size += data_sizes[idx_data];
					if (data_is_read[idx_data])
					{
						total_size_in_read += data_sizes[idx_data];
					}
					else
					{
						total_size_in_write += data_sizes[idx_data];
						Nw += 1;
					}
					unsigned idx_node;
					for (idx_node = 0; idx_node < nnodes; ++idx_node)
					{
						if (starpu_data_is_on_node(handle, idx_node))
						{
							data_Ri[idx_data] += 1;
							data_per_mem_node[idx_node] += data_sizes[idx_data];
							nb_data_per_mem_node[idx_node] += 1;
							if (data_is_read[idx_data])
							{
								data_per_mem_node_in_read[idx_node] += data_sizes[idx_data];
							}
							else
							{
								data_per_mem_node_in_write[idx_node] += data_sizes[idx_data];
								nb_data_in_w_per_mem_node[idx_node] += 1;
							}
						}
					}
				}
			}
			double max_score_so_far = DBL_MAX;
			unsigned idx_max_score = 0;
			unsigned idx_node;
			for (idx_node = 0; idx_node < nnodes; ++idx_node)
			{
/*
				const enum starpu_node_kind memnode_kind = starpu_node_get_kind(idx_node);
				if(memnode_kind == STARPU_DISK_RAM)
				{
					continue; // a disk has no associated worker
				}
*/

				double current_score = 0;
				assert(pushStrategy == PUSH_LS_SDHB);
				current_score = (total_size_in_read - data_per_mem_node_in_read[idx_node]) +
					((total_size_in_write - data_per_mem_node_in_write[idx_node]) *(2. - (double)(Nw) / (double)(N)));
				if (max_score_so_far > current_score ||
					(max_score_so_far == current_score &&
						idx_node == wgroupid))
				{
					max_score_so_far = current_score;
					idx_max_score = idx_node;
				}
			}
			best_mem_node = idx_max_score;
		}
#ifdef LAHETEROPRIO_PRINT_STAT
		lastats.push_redirect[workerid + 1][best_mem_node] += 1;
#endif	// LAHETEROPRIO_PRINT_STAT
	}
	else
	{
		if (workerid == (unsigned)-1)
		{ /*master thread */
			best_mem_node = hp->master_tasks_queue_idx;
		}
		else
		{
			const unsigned wgroupid = hp->workers_laheteroprio_wgroup_index[workerid];
			best_mem_node = wgroupid;
		}
	}
	return best_mem_node;
}

static void print_priorities(struct _starpu_heteroprio_data *hp)
{
	STARPU_ASSERT(hp->autoheteroprio_print_prio_after_ordering);

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->auto_calibration_mutex);
	starpu_worker_relax_off();

	fprintf(stderr, "Updated task priorities :\n");
	unsigned arch;
	for(arch=0;arch<STARPU_NARCH;++arch)
	{
		if(starpu_worker_archtype_is_valid(arch))
		{
			fprintf(stderr, "\t%s : ", starpu_worker_get_type_as_string(arch));
		}
		else
		{
			continue;
		}

		unsigned p;
		for(p=0;p<hp->found_codelet_names_on_arch[arch];++p)
		{
			fprintf(stderr, "%s  ", hp->found_codelet_names[hp->prio_mapping_per_arch_index[arch][p]]);
		}
		fprintf(stderr, "\n");
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->auto_calibration_mutex);
}


static double get_autoheteroprio_arch_busy_proportion(struct _starpu_heteroprio_data *hp, unsigned arch)
{
	double total = hp->average_arch_busy_time[arch] + hp->average_arch_free_time[arch];

	if(total <= 0)
	{ // if we have no info on workers times, we assume they are never busy (near-arbitrary choice)
		return 0;
	}

	return hp->average_arch_busy_time[arch]/total;
}

static double get_autoheteroprio_estimated_time(struct _starpu_heteroprio_data *hp, unsigned priority, unsigned arch)
{
	if(hp->prio_arch_has_time_info[arch][priority])
	{
		return hp->prio_average_time_arch[arch][priority];
	}

	if(arch_can_execute_prio(hp, arch, priority))
	{ // if arch is legit but we have no time information, return a decent arbitrary time
		return AUTOHETEROPRIO_FAIR_TIME;
	}


	if(hp->autoheteroprio_time_estimation_policy == 0)
	{
		return AUTOHETEROPRIO_LONG_TIME;
	}
	else if(hp->autoheteroprio_time_estimation_policy == 1)
	{
		// we can't execute this task on this arch, we therefore act as if it would be executed as fast as on the fastest architecture
		double bestTime = AUTOHETEROPRIO_EXTREMELY_LONG_TIME;

		unsigned a;
		for(a=0;a<STARPU_NB_TYPES;++a)
		{
			if(a == arch)
			{
				continue;
			}
			if(arch_can_execute_prio(hp, a, priority))
			{
				// recursive call, but guaranteed to stop because we know that arch_can_execute_prio(hp, a, prio)
				double time = get_autoheteroprio_estimated_time(hp, priority, a);
				if(time < bestTime)
				{
					bestTime = time;
				}
			}
		}

		return bestTime;
	}
	else
	{
		STARPU_ASSERT(hp->autoheteroprio_time_estimation_policy == 0 || hp->autoheteroprio_time_estimation_policy == 1);
		return 0.; // to get rid of warning
	}
}

static double get_autoheteroprio_prio_proportion(struct _starpu_heteroprio_data *hp, unsigned priority)
{
	if(hp->prio_overall_proportion_count[priority] > 0)
	{
		return hp->prio_overall_proportion[priority];
	}

	// no prio of this type has ever been recorded
	return 0;
}

// get normalized time (no unit, with average best arch executes tasks in 1.0)
static double get_autoheteroprio_normalized_time(struct _starpu_heteroprio_data *hp, unsigned priority, unsigned arch)
{
	double sum = 0.f;

	unsigned p;
	for(p=0;p<hp->found_codelet_names_length;++p)
	{
		sum += get_autoheteroprio_prio_proportion(hp, p) * get_best_autoheteroprio_estimated_time(hp, p);
	}

	if(sum <= 0.f)
	{
		return 1.0;
	}

	return get_autoheteroprio_estimated_time(hp, priority, arch) / sum;
}

static double get_autoheteroprio_prio_arch_proportion(struct _starpu_heteroprio_data *hp, unsigned priority, unsigned arch)
{
	if(hp->prio_arch_proportion_count[priority] > 0)
	{
		return hp->prio_arch_proportion[arch][priority];
	}

	// this prio has never been executed on this arch
	return 0;
}

static double get_autoheteroprio_successors_best_time_sum(struct _starpu_heteroprio_data *hp, unsigned priority)
{
	if(hp->prio_average_successors_best_time_sum_count[priority] > 0)
	{
		return hp->prio_average_successors_best_time_sum[priority];
	}

	return AUTOHETEROPRIO_FAIR_TIME;
}

// best execution time of a prio
//static double get_autoheteroprio_best_time(struct _starpu_heteroprio_data *hp, unsigned priority)
//{
//	if(hp->prio_average_best_count[priority] > 0)
//	{
//		return hp->prio_average_best[priority];
//	}
//
//	return AUTOHETEROPRIO_FAIR_TIME;
//}

static double get_autoheteroprio_NOD(struct _starpu_heteroprio_data *hp, unsigned priority)
{
	if(hp->prio_average_NOD_count[priority] > 0)
	{
		return hp->prio_average_NOD[priority];
	}

	return 1.0f;
}

static double get_autoheteroprio_URT(struct _starpu_heteroprio_data *hp, unsigned arch, unsigned priority)
{
	if(hp->prio_average_URT_count[priority] > 0)
	{
		return hp->prio_average_URT[arch][priority];
	}

	return AUTOHETEROPRIO_FAIR_TIME;
}

static double reLU(double x)
{
	if(x<0.0f)
	{
		return 0.0f;
	}
	return x;
}

static double rpg(double x)
{
	if(x > 1.0f)
	{
		return 1.0f;
	}
	return sqrt(x)*sqrt(2.0f-x);
}

struct prio_score
{
	unsigned index;
	double score;
};

static int compare_prio_scores(const void* elem1, const void* elem2)
{
	if(((const struct prio_score*)elem1)->score > ((const struct prio_score*)elem2)->score)
		return -1;
	return ((const struct prio_score*)elem1)->score < ((const struct prio_score*)elem2)->score;
}

static void order_priorities(struct _starpu_heteroprio_data *hp)
{
	STARPU_ASSERT(use_auto_mode);
	STARPU_ASSERT(hp->use_auto_calibration); // priorities should only be changed during execution if in auto calibration mode

	struct prio_score prio_arch[STARPU_NB_TYPES][HETEROPRIO_MAX_PRIO];
	unsigned prio_arch_index[STARPU_NB_TYPES] = {0};

	// lock the global policy mutex
	_starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->auto_calibration_mutex);
	starpu_worker_relax_off();

	unsigned p, a;
	for(p=0;p<hp->found_codelet_names_length;++p)
	{
		int worst_arch = -1;
		double worstTime = -1.0f;
		int second_worst_arch = -1;
		double secondWorstTime = -1.0f;

		// Find the worst architecture and the second worst if there is one
		for(a = 0; a < STARPU_NB_TYPES; ++a)
		{
			if((hp->buckets[p].valid_archs & starpu_heteroprio_types_to_arch(a)) == 0)
				continue;

			const double arch_time = get_autoheteroprio_normalized_time(hp, p, a);
			if(worstTime < arch_time)
			{
				second_worst_arch = worst_arch;
				secondWorstTime = worstTime;

				worst_arch = a;
				worstTime = arch_time;
			}
			else if(secondWorstTime < arch_time)
			{
				second_worst_arch = a;
				secondWorstTime = arch_time;
			}
		}

		// Ensure that there is at least one arch that can execute priority
		STARPU_ASSERT(worst_arch != -1);

		const double worstArchTaskProportion = get_autoheteroprio_prio_arch_proportion(hp, p, worst_arch);
		const double URT_worst = get_autoheteroprio_URT(hp, worst_arch, p);

		double secondWorstArchTaskProportion, URT_secondWorst;
		if(second_worst_arch == -1)
		{
			// If there's no second worst set values to worst possible values
			secondWorstTime = AUTOHETEROPRIO_EXTREMELY_LONG_TIME;
			secondWorstArchTaskProportion = 0.f;
			URT_secondWorst = 0.f;
		}
		else
		{
			secondWorstTime = get_autoheteroprio_normalized_time(hp, p, second_worst_arch);
			secondWorstArchTaskProportion = get_autoheteroprio_prio_arch_proportion(hp, p, second_worst_arch);
			URT_secondWorst = get_autoheteroprio_URT(hp, second_worst_arch, p);
		}

		// Compute scores
		for(a=0;a<STARPU_NB_TYPES;++a)
		{
			if(hp->buckets[p].valid_archs & starpu_heteroprio_types_to_arch(a))
			{
				double otherTime, otherArchTaskProportion, URT_other;
				unsigned prio = prio_arch_index[a]++;

				if(a == (unsigned) worst_arch)
				{
					// Compare the worst architecture to the second worst
					otherTime = secondWorstTime;
					URT_other = URT_secondWorst;
					otherArchTaskProportion = secondWorstArchTaskProportion;
				}
				else
				{
					// Compare to the worst architecture
					otherTime = worstTime;
					URT_other = URT_worst;
					otherArchTaskProportion = worstArchTaskProportion;
				}

				const double need_other = 1.0f - otherArchTaskProportion;

				double NOD = get_autoheteroprio_NOD(hp, p);
				double sum = get_autoheteroprio_successors_best_time_sum(hp, p);

				double ownTime = get_autoheteroprio_normalized_time(hp, p, a);
				double archDiff = otherTime - ownTime;
				double archRelDiff = otherTime/ownTime;

				double ownArchTaskProportion = get_autoheteroprio_prio_arch_proportion(hp, p, a);

				double URT_own = get_autoheteroprio_URT(hp, a, p);

				double need_own = 1.0f - get_autoheteroprio_arch_busy_proportion(hp, a);
				double archNeedDiff = need_own-need_other;

				double URT = (URT_own*need_own + URT_other*need_other);

				prio_arch[a][prio].index = p;

				if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_NOD_TIME_COMBINATION)
				{
					double relDiff = archRelDiff>1.0f?archRelDiff:1.0/archRelDiff;
					double multiplier = exp(-hp->NTexpVal*(relDiff-1)*(relDiff-1));
					prio_arch[a][prio].score = archDiff + hp->NTnodPond*multiplier*NOD;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_BEST_NODS_SCORE || hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_BEST_NODS)
				{ // TODO, implement BEST_NODS
					double multiplier = exp(-hp->BNexpVal*(archDiff)*(archDiff));
					if(archDiff > 0.0f)
					{ // my arch is faster
						multiplier = 1.0f;
					}
					multiplier = 2.0f*multiplier - 1.0f; // bad diff becomes -1, good or equal diff 1

					prio_arch[a][prio].score = multiplier*NOD;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_PURE)
				{
					prio_arch[a][prio].score = URT;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT)
				{
					prio_arch[a][prio].score = hp->URTurt * URT + archDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_2)
				{
					prio_arch[a][prio].score = hp->URT2urt * URT + archDiff + hp->URT2prop * reLU(ownArchTaskProportion*otherArchTaskProportion*archNeedDiff);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_PURE)
				{
					prio_arch[a][prio].score = URT*archDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_PURE_2)
				{
					prio_arch[a][prio].score = (1.0f + URT)*archDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE)
				{
					prio_arch[a][prio].score = URT*archRelDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE_2)
				{
					prio_arch[a][prio].score = (1.0f + URT)*archRelDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_2)
				{
					prio_arch[a][prio].score = (1.0f + URT)*archDiff + hp->and2pond * ownTime * archNeedDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_3)
				{
					prio_arch[a][prio].score = (1.0f + URT)*archDiff + hp->and3pond * ownTime * reLU(archNeedDiff);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_4)
				{
					prio_arch[a][prio].score = (1.0f + URT)*archDiff - hp->and4pond * ownTime * reLU(-archNeedDiff);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_5)
				{
					prio_arch[a][prio].score = (hp->and5xoffset + URT) * (hp->and5yoffset + archDiff);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_6)
				{
					prio_arch[a][prio].score = (1.0f + URT)*log1p(exp(archDiff));
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_7)
				{
					prio_arch[a][prio].score = rpg(URT)*(1+URT)*(1+archDiff)+(1-rpg(URT))*(-log1p(exp(-archDiff)));
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_8)
				{
					prio_arch[a][prio].score = (1/(1+exp(-URT))-0.5)*(1+URT)*(1+archDiff)+(1/(1+exp(-1/URT))-0.5)*(-exp(-archDiff));
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_9)
				{
					prio_arch[a][prio].score = log(hp->and9xoffset+URT)*atan(archDiff+hp->and9yoffset*URT);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_10)
				{
					prio_arch[a][prio].score = (hp->and10xoffset+URT)*atan(archDiff) + hp->and10yoffset*URT;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URT_DOT_DIFF_11)
				{
					prio_arch[a][prio].score = (hp->and11xoffset+URT)*(archDiff+hp->and11yoffset*URT);
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URTS_PER_SECONDS)
				{
					prio_arch[a][prio].score = URT / ownTime;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URTS_PER_SECONDS_2)
				{
					prio_arch[a][prio].score = (URT + archDiff) / ownTime;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URTS_PER_SECONDS_DIFF)
				{
					prio_arch[a][prio].score = URT / ownTime + archDiff;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URTS_TIME_RELEASED_DIFF)
				{
					prio_arch[a][prio].score = URT*(sum+archDiff)/ownTime;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_URTS_TIME_COMBINATION)
				{
					double relDiff = archRelDiff>1.0f?archRelDiff:1.0/archRelDiff;
					double multiplier = exp(-hp->ANTexpVal*(relDiff-1)*(relDiff-1));
					prio_arch[a][prio].score = archDiff + hp->ANTnodPond*multiplier*URT;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_NODS_PER_SECOND)
				{
					prio_arch[a][prio].score = NOD/ownTime;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_NODS_TIME_RELEASED)
				{
					prio_arch[a][prio].score = NOD*sum/ownTime;
				}
				else if(hp->autoheteroprio_priority_ordering_policy == STARPU_HETEROPRIO_NODS_TIME_RELEASED_DIFF)
				{
					prio_arch[a][prio].score = NOD*(sum+archDiff)/ownTime;
				}
				else
				{
					_STARPU_MSG("[AUTOHETEROPRIO] Warning: unknown ordering policy.\n");
					prio_arch[a][prio].score = 0;
				}

				if(!hp->freeze_data_gathering && hp->prio_average_time_arch_count[a][p] < AUTOHETEROPRIO_RELEVANT_SAMPLE_SIZE)
				{
					// if we dont have enough data on execution time, we push execution on it by increasing the score
					prio_arch[a][prio].score += 99999999.;
				}
			}
		}
	}

	for(a=0;a<STARPU_NB_TYPES;++a)
	{
		qsort(&prio_arch[a][0], hp->found_codelet_names_on_arch[a], sizeof(struct prio_score), compare_prio_scores);
	}

	starpu_heteroprio_clear_mapping_hp(hp);

	for(a=0;a<STARPU_NB_TYPES;++a)
	{
		for(p=0;p<hp->found_codelet_names_on_arch[a];++p)
		{
			starpu_heteroprio_set_mapping_hp(hp, a, p, prio_arch[a][p].index);
		}
	}

/* // uncomment to print task names ordered by priority (TODO : use environment variable)
	printf("priorities sorted:\n");
	printf("CPU:\n");
	for(p=0;p<hp->found_codelet_names_on_arch[STARPU_CPU_WORKER];++p)
	{
		printf("%d : %s bucket=%d (score = %f)\n", p, hp->found_codelet_names[prio_arch[STARPU_CPU_WORKER][p].index], prio_arch[STARPU_CPU_WORKER][p].index, prio_arch[STARPU_CPU_WORKER][p].score);
	}
	printf("GPU:\n");
	for(p=0;p<hp->found_codelet_names_on_arch[STARPU_CUDA_WORKER];++p)
	{
		printf("%d : %s bucket=%d (score = %f)\n", p, hp->found_codelet_names[prio_arch[STARPU_CUDA_WORKER][p].index], prio_arch[STARPU_CUDA_WORKER][p].index, prio_arch[STARPU_CUDA_WORKER][p].score);
	}
*/


	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->auto_calibration_mutex);
}

// used to get the name of a codelet, considering a codelet grouping strategy
static const char *_heteroprio_get_codelet_name(enum autoheteroprio_codelet_grouping_strategy strategy, struct starpu_codelet *cl)
{
	const char *name = NULL;
	switch(strategy)
	{
		case BY_PERF_MODEL_OR_NAME:
			name = _starpu_codelet_get_model_name(cl);
			break;

		case BY_NAME_ONLY:
			name = _starpu_codelet_get_name(cl);
			break;
	}

	return name ? name : AUTOHETEROPRIO_NO_NAME;
}

// used by get_task_auto_priority for knowing if a submitted codelet equals an other
static int are_same_codelets(struct _starpu_heteroprio_data *hp, const struct starpu_task *task, const char name[CODELET_MAX_NAME_LENGTH], unsigned valid_archs)
{
	unsigned task_valid_archs = task->where >= 0 ? (unsigned) task->where : task->cl->where;

	if(task_valid_archs != valid_archs)
	{
		// are not same codelet, because different architectures
		return 0;
	}

	const char *task_name = _heteroprio_get_codelet_name(hp->codelet_grouping_strategy, task->cl);

	return strncmp(name, task_name, CODELET_MAX_NAME_LENGTH) == 0;
}

static int get_task_auto_priority(struct _starpu_heteroprio_data *hp, const struct starpu_task *task)
{
	STARPU_ASSERT(use_auto_mode);
	STARPU_ASSERT(hp->use_auto_calibration);
	STARPU_ASSERT(hp->found_codelet_names_length <= HETEROPRIO_MAX_PRIO);

	if(task->cl->where == STARPU_NOWHERE)
	{
		return -1;
	}

	const char *name = _heteroprio_get_codelet_name(hp->codelet_grouping_strategy, task->cl);

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->auto_calibration_mutex);
	starpu_worker_relax_off();

	unsigned current_priority;
	for(current_priority = 0;current_priority<hp->found_codelet_names_length;++current_priority)
	{
		if(are_same_codelets(hp, task, &hp->found_codelet_names[current_priority][0], hp->buckets[current_priority].valid_archs))
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&hp->auto_calibration_mutex);
			return current_priority;
		}
	}

	// codelet's name does not exist in found_codelet_names, add it

	STARPU_ASSERT(hp->found_codelet_names_length < HETEROPRIO_MAX_PRIO);

	const unsigned found_codelet_names_length = hp->found_codelet_names_length;

	if(!task->cl->model)
	{ // The codelet does not have a perf model
		_STARPU_DISP("[HETEROPRIO] Warning: codelet %s does not have a perfmodel. This may negatively impact heteroprio's auto prioritizing.\n", name);
	}

	unsigned archs[STARPU_NB_TYPES] = {0};

	unsigned arch;
	for(arch=0;arch<STARPU_NARCH;++arch)
	{
		archs[arch] = starpu_worker_type_can_execute_task(arch, task);
	}

	starpu_autoheteroprio_add_task(hp, name, archs);

	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->auto_calibration_mutex);

	return found_codelet_names_length;
}

// checks that auto-heteroprio arrays are correctly set (for debugging purposes)
//static void check_auto_heteroprio_mapping(struct _starpu_heteroprio_data *hp)
//{
//	// may be useful
//	(void) hp;
//}

static double get_job_NOD(struct _starpu_heteroprio_data *hp, struct _starpu_job *job)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);
	STARPU_ASSERT(_starpu_graph_record == 1);

	double NOD = 0.f;

	//STARPU_PTHREAD_MUTEX_LOCK(&job->sync_mutex);

	/*if(!job->tag)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&job->sync_mutex);
		return 0;
	}*/

	//STARPU_PTHREAD_MUTEX_UNLOCK(&job->sync_mutex);

	_starpu_graph_wrlock();

	struct _starpu_graph_node *node = job->graph_node;

	if(!node)
	{
		// No information because the graph isn't available
		_starpu_graph_wrunlock();
		return 0.f;
	}

	unsigned n;
	for(n=0;n<node->n_outgoing;++n)
	{
		struct _starpu_graph_node *successor = node->outgoing[n]; // there is a node->outgoing_slot, but this ordering array does not seem useful here
		if(successor)
		{ // successor may be NULL
			NOD += 1.f/(double)successor->n_incoming;
		}
	}

	_starpu_graph_wrunlock();

	return NOD;
}

// get job's NRT (Normalized Released Time)
static double get_job_NRT(struct _starpu_heteroprio_data *hp, struct _starpu_job *job, unsigned arch)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);
	STARPU_ASSERT(_starpu_graph_record == 1);

	double NOD = 0.f;

	//STARPU_PTHREAD_MUTEX_LOCK(&job->sync_mutex);

	/*if(!job->tag)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&job->sync_mutex);
		return 0;
	}*/

	//STARPU_PTHREAD_MUTEX_UNLOCK(&job->sync_mutex);

	_starpu_graph_wrlock();

	struct _starpu_graph_node *node = job->graph_node;

	if(!node)
	{
		// No information because the graph isn't available
		_starpu_graph_wrunlock();
		return 0.f;
	}

	unsigned n;
	for(n=0;n<node->n_outgoing;++n)
	{
		struct _starpu_graph_node *successor = node->outgoing[n]; // there is a node->outgoing_slot, but this ordering array does not seem useful here
		if(successor)
		{
			// successor may be NULL
			struct _starpu_job *successor_job = successor->job;
			STARPU_PTHREAD_MUTEX_LOCK(&successor_job->sync_mutex);
			const struct starpu_task *successor_task = successor_job->task;
			STARPU_PTHREAD_MUTEX_UNLOCK(&successor_job->sync_mutex);

			if(successor_task->cl)
			{
				// if a codelet is associated to the task, we can count it in the NOD
				int successor_prio = get_task_auto_priority(hp, successor_task);
				double successor_arch_time;
				if(successor_prio == -1)
				{
					successor_arch_time = 0.f;
				}
				else
				{
					successor_arch_time = get_autoheteroprio_prio_arch_proportion(hp, successor_prio, arch) * get_autoheteroprio_normalized_time(hp, successor_prio, arch);
				}
				NOD += successor_arch_time/(double)successor->n_incoming;
			}
		}
	}

	_starpu_graph_wrunlock();

	return NOD;
}




static void register_arch_times(struct _starpu_heteroprio_data *hp, unsigned arch, double busy_time, double free_time)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	double summed_busy_time = hp->average_arch_busy_time[arch] + busy_time;
	double summed_free_time = hp->average_arch_free_time[arch] + free_time;

	double max_time = STARPU_MAX(summed_busy_time, summed_free_time);
	double scale_to_apply = 1.0f;

	if(max_time > AUTOHETEROPRIO_MAX_WORKER_PROFILING_TIME)
	{
		scale_to_apply = AUTOHETEROPRIO_MAX_WORKER_PROFILING_TIME/max_time;
	}

	hp->average_arch_busy_time[arch] = summed_busy_time*scale_to_apply;
	hp->average_arch_free_time[arch] = summed_free_time*scale_to_apply;
}

// gets the lowest expected time between each architectures
static double get_best_autoheteroprio_estimated_time(struct _starpu_heteroprio_data *hp, unsigned priority)
{
	double time = 999999999999999.f;

	unsigned arch;
	for(arch=0;arch<STARPU_NARCH;++arch)
	{
		if(arch_can_execute_prio(hp, arch, priority))
		{
			time = STARPU_MIN(time, get_autoheteroprio_estimated_time(hp, priority, arch));
		}
	}

	return time;
}

static double get_job_best_time(struct _starpu_heteroprio_data *hp, struct _starpu_job *job)
{
	int task_priority = get_task_auto_priority(hp, job->task);

	double time;

	if(task_priority == -1)
	{
		time = AUTOHETEROPRIO_DEFAULT_TASK_TIME;
	}
	else
	{
		time = get_best_autoheteroprio_estimated_time(hp, task_priority);
	}

	return time;
}

static double get_job_successors_best_time_sum(struct _starpu_heteroprio_data *hp, struct _starpu_job *job)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);
	STARPU_ASSERT(_starpu_graph_record == 1);

	double sum = 0.f;

	_starpu_graph_wrlock();

	struct _starpu_graph_node *node = job->graph_node;

	if(!node)
	{
		// No information because the graph isn't available
		_starpu_graph_wrunlock();
		return 0.f;
	}

	unsigned n;
	for(n=0;n<node->n_outgoing;++n)
	{
		struct _starpu_graph_node *successor = node->outgoing[n]; // there is a node->outgoing_slot, but this ordering array does not seem useful here
		if(successor && successor->job && successor->job->task->cl)
		{
			// successor may be NULL
			sum += get_job_best_time(hp, successor->job);
		}
	}

	_starpu_graph_wrunlock();

	return sum;
}

static void add_NOD_to_data(struct _starpu_heteroprio_data *hp, unsigned task_priority, double NOD)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_average_NOD_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_average_NOD_count[task_priority];
	}

	const unsigned count = hp->prio_average_NOD_count[task_priority];

	hp->prio_average_NOD[task_priority] = hp->prio_average_NOD[task_priority] * (double)(count - 1) / (double)count
					+ NOD / (double)count;
}

static void add_URTs_to_data(struct _starpu_heteroprio_data *hp, unsigned task_priority, double archs_URTs[STARPU_NARCH])
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_average_URT_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_average_URT_count[task_priority];
	}

	const unsigned count = hp->prio_average_URT_count[task_priority];

	unsigned arch;
	for(arch=0;arch<STARPU_NARCH;++arch)
	{
		hp->prio_average_URT[arch][task_priority] = hp->prio_average_URT[arch][task_priority] * (double)(count - 1) / (double)count
					+ archs_URTs[arch] / (double)count;
	}
}

static void register_execution_time(struct _starpu_heteroprio_data *hp, unsigned arch, unsigned task_priority, double time)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_average_time_arch_count[arch][task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_average_time_arch_count[arch][task_priority];
	}

	const unsigned count = hp->prio_average_time_arch_count[arch][task_priority];

	hp->prio_average_time_arch[arch][task_priority] = hp->prio_average_time_arch[arch][task_priority] * (double)(count - 1) / (double)count
					+ time / (double)count;
	hp->prio_arch_has_time_info[arch][task_priority] = 1;
}

static inline unsigned get_total_submitted_task_num(struct _starpu_heteroprio_data *hp)
{
	unsigned total = 0;

	unsigned idx_prio;
	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
	{
		total += hp->prio_overall_proportion_count[idx_prio];
	}

	return total;
}

static inline double get_sum_task_proportions(struct _starpu_heteroprio_data *hp)
{
	double total = 0.f;

	unsigned idx_prio;
	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
	{
		total += hp->prio_overall_proportion[idx_prio];
	}

	return total;
}

// noralizes tasks proportions so that their sum equals 1
static inline void normalize_task_proportions(struct _starpu_heteroprio_data *hp)
{
	const double total_task_proportions = get_sum_task_proportions(hp);

	STARPU_ASSERT(total_task_proportions > 0);

	unsigned idx_prio;
	for(idx_prio = 0; idx_prio < HETEROPRIO_MAX_PRIO; ++idx_prio)
	{
		hp->prio_overall_proportion[idx_prio] /= total_task_proportions;
	}
}

static void add_submitted_task_to_data(struct _starpu_heteroprio_data *hp, unsigned task_priority)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_overall_proportion_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_overall_proportion_count[task_priority];
	}

	const unsigned count = get_total_submitted_task_num(hp);

	STARPU_ASSERT(count > 0);

	hp->prio_overall_proportion[task_priority] += 1.f/(double)count;

	// take back task proportions to a valid value (sum = 1)
	normalize_task_proportions(hp);
}

// gets the sum of a task's architecture proportions
static inline double get_sum_task_arch_proportions(struct _starpu_heteroprio_data *hp, unsigned task_priority)
{
	double total = 0.f;
	unsigned arch;

	for(arch=0;arch<STARPU_NB_TYPES;++arch)
	{
		total += hp->prio_arch_proportion[arch][task_priority];
	}

	return total;
}

// noralizes tasks execution proportions so that the sum of proportions of a task on each arch equals 1
// EXAMPLE : task A : %CPU = 0.75, %GPU = 0.25
static inline void normalize_task_arch_proportions(struct _starpu_heteroprio_data *hp, unsigned task_priority)
{
	const double total_task_proportions = get_sum_task_arch_proportions(hp, task_priority);

	STARPU_ASSERT(total_task_proportions > 0);

	unsigned arch;
	for(arch=0;arch<STARPU_NB_TYPES;++arch)
	{
		hp->prio_arch_proportion[arch][task_priority] /= total_task_proportions;
	}
}

static void register_task_arch_execution(struct _starpu_heteroprio_data *hp, unsigned task_priority, unsigned arch)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_arch_proportion_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_arch_proportion_count[task_priority];
	}

	unsigned count = hp->prio_arch_proportion_count[task_priority];
	STARPU_ASSERT(count > 0);
	if(count >= 2)
	{
		// to have correct proportions and not divide by zero
		count -=1;
	}

	hp->prio_arch_proportion[arch][task_priority] += 1.f/(double)count;

	// take back task proportions to a valid value (sum = 1)
	normalize_task_arch_proportions(hp, task_priority);
}


static void add_successors_best_time_sum_to_data(struct _starpu_heteroprio_data *hp, unsigned task_priority, double sum)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_average_successors_best_time_sum_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_average_successors_best_time_sum_count[task_priority];
	}

	const unsigned count = hp->prio_average_successors_best_time_sum_count[task_priority];

	hp->prio_average_successors_best_time_sum[task_priority] = hp->prio_average_successors_best_time_sum[task_priority] * (double)(count - 1) / (double)count
					+ sum / (double)count;
}

static void add_best_time_to_data(struct _starpu_heteroprio_data *hp, unsigned task_priority, double sum)
{
	STARPU_ASSERT(!hp->freeze_data_gathering);

	if(hp->prio_average_best_count[task_priority] < AUTOHETEROPRIO_RELEVANT_TASK_LIFE)
	{
		++hp->prio_average_best_count[task_priority];
	}

	const unsigned count = hp->prio_average_best_count[task_priority];

	hp->prio_average_best[task_priority] = hp->prio_average_best[task_priority] * (double)(count - 1) / (double)count
					+ sum / (double)count;
}

static void autoheteroprio_update_slowdown_data(struct _starpu_heteroprio_data *hp)
{
	unsigned p, arch;
	for(p=0;p<hp->found_codelet_names_length;++p)
	{
		unsigned valid_archs[STARPU_NB_TYPES] = {0};
		double arch_times[STARPU_NB_TYPES] = {0.f};

		for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
		{
			valid_archs[arch] = arch_can_execute_prio(hp, arch, p);
			if(valid_archs[arch])
			{
				double arch_time = get_autoheteroprio_estimated_time(hp, p, arch);
				STARPU_ASSERT(arch_time > 0.f);
				arch_times[arch] = arch_time;
			}
		}

		// Assert that at least one architecture can execute priority
		for(arch = 0; arch < STARPU_NB_TYPES && !valid_archs[arch]; ++arch)
			;
		STARPU_ASSERT(arch < STARPU_NB_TYPES);

		arch = 0;
		while(!valid_archs[arch])
			++arch;
		unsigned fastest_arch = arch;
		double best_time = arch_times[arch];

		++arch;
		for(; arch < STARPU_NB_TYPES; ++arch)
		{
			if(valid_archs[arch] && arch_times[arch] < best_time)
			{
				fastest_arch = arch;
				best_time = arch_times[arch];
			}
		}

		starpu_heteroprio_set_faster_arch_hp(hp, fastest_arch, p);

		for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
		{
			if(valid_archs[arch] && arch != fastest_arch)
				starpu_heteroprio_set_arch_slow_factor_hp(hp, arch, p, arch_times[arch]/best_time);
		}
	}

	check_heteroprio_mapping(hp);
}

/* Push a new task (simply store it and update counters) */
static int push_task_heteroprio_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned computed_best_mem_node = 0;
	unsigned best_node_now[PUSH_NB_AUTO] = {0};

	if(hp->use_locality)
	{
#ifdef LAHETEROPRIO_PRINT_STAT
		lastats.push_to_use[starpu_worker_get_id()+1][hp->pushStrategyToUse] += 1;
#endif
		//unsigned best_node_now[PUSH_NB_AUTO] = {0};
		if(hp->pushStrategySet == PUSH_AUTO)
		{
			unsigned idx_strategy;
			for(idx_strategy = 0 ; idx_strategy < PUSH_NB_AUTO ; ++idx_strategy)
			{
				best_node_now[idx_strategy] = get_best_mem_node(task, hp, idx_strategy);
			}
		}
		computed_best_mem_node = (hp->pushStrategySet == PUSH_AUTO && hp->pushStrategyToUse < PUSH_NB_AUTO ?
					best_node_now[hp->pushStrategyToUse]
					: get_best_mem_node(task, hp, hp->pushStrategyToUse));

		STARPU_ASSERT_MSG(hp->map_wgroup_has_been_called, "starpu_laheteroprio_map_wgroup_memory_nodes \
has not been called while you are using the heteroprio in LA mode. To fix this, you can either turn LA mode off by setting \
the HETEROPRIO_USE_LA variable to 0, or calling starpu_laheteroprio_map_wgroup_memory_nodes after starpu_laheteroprio_set_nb_prios.\n");
	}


	const unsigned best_mem_node = computed_best_mem_node;

	/* One worker at a time uses heteroprio */
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->policy_mutex);
	starpu_worker_relax_off();

	/* Get tasks priority (ID) */
	int task_priority;
	if(hp->use_auto_calibration)
	{
		task_priority = get_task_auto_priority(hp, task);

		if(!hp->freeze_data_gathering && hp->priority_last_ordering >= hp->priority_ordering_interval)
		{
			hp->priority_last_ordering = 0;
		}

		if(hp->priority_last_ordering == 0)
		{
			// first pushed task OR at least "priority_ordering_interval" tasks have been pushed
			order_priorities(hp);
			if(hp->autoheteroprio_print_prio_after_ordering)
			{
				print_priorities(hp);
			}
			autoheteroprio_update_slowdown_data(hp);
		}

		++hp->priority_last_ordering;

		if(!hp->freeze_data_gathering)
		{
			struct _starpu_job *job = _starpu_get_job_associated_to_task(task);

			if(task_priority != -1)
			{
				// register that the task has been submitted
				add_submitted_task_to_data(hp, task_priority);

				double NOD = get_job_NOD(hp, job);
				add_NOD_to_data(hp, task_priority, NOD);

				double archs_NRTs[STARPU_NARCH];
				unsigned arch;
				for(arch=0;arch<STARPU_NARCH;++arch)
				{
					archs_NRTs[arch] = get_job_NRT(hp, job, arch);
				}
				add_URTs_to_data(hp, task_priority, archs_NRTs);

				double sum = get_job_successors_best_time_sum(hp, job);
				add_successors_best_time_sum_to_data(hp, task_priority, sum);

				double best_time = get_job_best_time(hp, job);
				add_best_time_to_data(hp, task_priority, best_time);

			}

			autoheteroprio_update_slowdown_data(hp);

			if(hp->autoheteroprio_print_data_on_update)
			{
				unsigned arch;
				char is_arch_used[STARPU_NB_TYPES];

				for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
				{
					if(hp->average_arch_busy_time[arch] + hp->average_arch_free_time[arch] > 0)
						is_arch_used[arch] = 1;
					else
						is_arch_used[arch] = 0;
				}

				fprintf(stderr, "Updated values :\n");

				fprintf(stderr, "Busy proportion :\n\t");
				for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
				{
					if(is_arch_used[arch])
						fprintf(stderr, "%s : %f, ",
							starpu_worker_get_type_as_string(arch),
							get_autoheteroprio_arch_busy_proportion(hp, arch));
				}
				fprintf(stderr, "\n");

				unsigned idx_prio;

				fprintf(stderr, "Assumed values for heuristic computation :\n");
				for(idx_prio = 0; idx_prio < hp->found_codelet_names_length; ++idx_prio)
				{
					fprintf(stderr, "task %s :\n\tNOD = %f",
						&hp->found_codelet_names[idx_prio][0],
						get_autoheteroprio_NOD(hp, idx_prio));

					for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
					{
						if(is_arch_used[arch])
							fprintf(stderr, ", URT_%s = %f",
								starpu_worker_get_type_as_string(arch),
								get_autoheteroprio_URT(hp, arch, idx_prio));
					}

					fprintf(stderr, "\n\testimated time : ");
					for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
					{
						if(is_arch_used[arch])
							fprintf(stderr, "%s : %f, ",
								starpu_worker_get_type_as_string(arch),
								get_autoheteroprio_estimated_time(hp, idx_prio, arch));
					}

					fprintf(stderr, "\n\tnormalized time : ");
					for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
					{
						if(is_arch_used[arch])
							fprintf(stderr, "%s : %f, ",
								starpu_worker_get_type_as_string(arch),
								get_autoheteroprio_normalized_time(hp, idx_prio, arch));
					}

					fprintf(stderr, "\n\tbestsum=%f, proportion=%f",
						get_autoheteroprio_successors_best_time_sum(hp, idx_prio),
						get_autoheteroprio_prio_proportion(hp, idx_prio));
					for(arch = 0; arch < STARPU_NB_TYPES; ++arch)
					{
						if(is_arch_used[arch])
							fprintf(stderr, ", prop%s=%f",
								starpu_worker_get_type_as_string(arch),
								get_autoheteroprio_prio_arch_proportion(hp, idx_prio, arch));
					}

					fprintf(stderr, "\n");
				}
			}
		}
	}
	else
	{
		task_priority = task->priority;
	}

	/* Retrieve the correct bucket */
	STARPU_ASSERT(task_priority >= 0);
	STARPU_ASSERT(task_priority < HETEROPRIO_MAX_PRIO);

	struct _heteroprio_bucket* bucket = &hp->buckets[task_priority];
	/* Ensure that any worker that check that list can compute the task */
	STARPU_ASSERT_MSG(bucket->valid_archs, "The bucket %d does not have any archs\n", task_priority);
	STARPU_ASSERT(((bucket->valid_archs ^ task->where) & bucket->valid_archs) == 0);

	if(hp->use_locality)
	{
		/* save the task */
		starpu_task_list_push_front(&bucket->tasks_queue[best_mem_node], task);
		if(hp->pushStrategySet == PUSH_AUTO)
		{
			laqueue_push(&bucket->auto_mn[best_mem_node], best_node_now);
		}
#ifdef LAHETEROPRIO_PRINT_STAT
		if(starpu_worker_get_id() != -1)
		{
			lastats.nb_tasks_per_wgroup[best_mem_node][task_priority] += 1;
			lastats.nb_tasks_per_worker[starpu_worker_get_id()][task_priority] += 1;
		}
#endif // LAHETEROPRIO_PRINT_STAT
		bucket->tasks_queue_ntasks += 1;
#ifdef LAHETEROPRIO_PRINT_STAT
		lastats.nb_tasks += 1;
#endif // LAHETEROPRIO_PRINT_STAT

	}
	else
	{
		/* save the task */
		starpu_task_list_push_front(&bucket->tasks_queue[0],task);
		/* Increase the total number of tasks */
		bucket->tasks_queue_ntasks += 1;
	}

	/* Inc counters */
	unsigned arch_index;
	for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
	{
		/* We test the archs on the bucket and not on task->where since it is restrictive */
		if(bucket->valid_archs & starpu_heteroprio_types_to_arch(arch_index))
		{
			hp->nb_remaining_tasks_per_arch_index[arch_index] += 1;
		}
	}

	hp->total_tasks_in_buckets += 1;

	starpu_push_task_end(task);

	/*if there are no tasks_queue block */
	/* wake people waiting for a task */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	char dowake[STARPU_NMAXWORKERS] = { 0 };
#endif

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);

#ifdef STARPU_NON_BLOCKING_DRIVERS
		if (!starpu_bitmap_get(&hp->waiters, worker))
			/* This worker is not waiting for a task */
			continue;
#endif

		if (starpu_worker_can_execute_task_first_impl(worker, task, NULL))
		{
			/* It can execute this one, tell him! */
#ifdef STARPU_NON_BLOCKING_DRIVERS
			starpu_bitmap_unset(&hp->waiters, worker);
			/* We really woke at least somebody, no need to wake somebody else */
			break;
#else
			dowake[worker] = 1;
#endif
		}
	}
	/* Let the task free */
	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->policy_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* Now that we have a list of potential workers, try to wake one */

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		if (dowake[worker])
			if (starpu_wake_worker_relax_light(worker))
				break; // wake up a single worker
	}
#endif

	return 0;
}

static struct starpu_task *pop_task_heteroprio_policy(unsigned sched_ctx_id)
{
	const unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _heteroprio_worker_wrapper* worker = &hp->workers_heteroprio[workerid];
	struct starpu_task* task = NULL;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* If no tasks available, no tasks in worker queue or some arch worker queue just return NULL */
	if (!STARPU_RUNNING_ON_VALGRIND
	    && (hp->total_tasks_in_buckets == 0 || hp->nb_remaining_tasks_per_arch_index[worker->arch_index] == 0)
	    && (hp->use_locality || (worker->tasks_queue.ntasks == 0 && hp->nb_prefetched_tasks_per_arch_index[worker->arch_index] == 0)))
	{
		return NULL;
	}

	if (!STARPU_RUNNING_ON_VALGRIND && starpu_bitmap_get(&hp->waiters, workerid))
	{
		/* Nobody woke us, avoid bothering the mutex */
		return NULL;
	}
#endif
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->policy_mutex);
	starpu_worker_relax_off();

	//	if(hp->use_locality)
	//	{
	// used only with use_locality==1
#ifdef LAHETEROPRIO_PRINT_STAT
	unsigned src_mem_node = (unsigned)-1;
#endif
	unsigned best_node_previous[PUSH_NB_AUTO] = {0};
	//	}
	//	else
	//	{
	// used only with use_locality==0
	/* keep track of the new added task to perform real prefetch on node */
	unsigned nb_added_tasks = 0;
	//	}

	if (hp->use_locality)
	{
		const unsigned wgroupid = hp->workers_laheteroprio_wgroup_index[workerid];

		if (hp->nb_remaining_tasks_per_arch_index[worker->arch_index] != 0)
		{
			const struct starpu_laheteroprio_access_item *wgroup_access_order = hp->wgroup_pop_access_orders[wgroupid];
			const unsigned wgroup_access_order_size = hp->wgroup_pop_access_orders_size[wgroupid];

			unsigned idx_access_item;
			for (idx_access_item = 0; task == NULL && idx_access_item < wgroup_access_order_size; ++idx_access_item)
			{
				const unsigned current_wgroupid = wgroup_access_order[idx_access_item].wgroup_idx;
				/*Retrieve the bucket using the mapping */
				struct _heteroprio_bucket *bucket = &hp->buckets[hp->prio_mapping_per_arch_index[worker->arch_index][wgroup_access_order[idx_access_item].prio_idx]];
				/*Ensure we can compute task from this bucket */
				STARPU_ASSERT(bucket->valid_archs &worker->arch_type);
				/*Take one task if possible */
				if (!starpu_task_list_empty(&bucket->tasks_queue[current_wgroupid]))
				{
					if ((bucket->factor_base_arch_index == 0 ||
							worker->arch_index == bucket->factor_base_arch_index ||
							(((float) bucket->tasks_queue_ntasks) / ((float) hp->nb_workers_per_arch_index[bucket->factor_base_arch_index])) >= bucket->slow_factors_per_index[worker->arch_index]))
					{
						task = starpu_task_list_pop_front(&bucket->tasks_queue[current_wgroupid]);
						if(!starpu_worker_can_execute_task(workerid, task, 0))
						{
							// Put the task back because worker can't execute it (e.g. codelet.can_execute)
							starpu_task_list_push_front(&bucket->tasks_queue[0], task);
							break;
						}
						if (hp->pushStrategySet == PUSH_AUTO)
						{
							memcpy(best_node_previous, laqueue_pop(&bucket->auto_mn[current_wgroupid]), sizeof(unsigned) *PUSH_NB_AUTO);
						}
						/*Save the task */
						STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), workerid);
						/*Update general counter */
						hp->total_tasks_in_buckets -= 1;
						bucket->tasks_queue_ntasks -= 1;
						unsigned arch_index;
						for (arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
						{
							/*We test the archs on the bucket and not on task->where since it is restrictive */
							if (bucket->valid_archs &starpu_heteroprio_types_to_arch(arch_index))
							{
								hp->nb_remaining_tasks_per_arch_index[arch_index] -= 1;
							}
						}
#ifdef LAHETEROPRIO_PRINT_STAT
						if (current_wgroupid != wgroupid)
						{
							lastats.task_stolen_per_worker[workerid][wgroup_access_order[idx_access_item].prio_idx] += 1;
							lastats.task_stolen_in_wgroup[current_wgroupid][wgroup_access_order[idx_access_item].prio_idx] += 1;
						}
						src_mem_node = current_wgroupid;
#endif
						break;
					}
#ifdef LAHETEROPRIO_PRINT_STAT
					else
					{
						lastats.task_skipt_due_to_factor_per_worker[workerid][wgroup_access_order[idx_access_item].prio_idx] += 1;
					}
#endif
				}
#ifdef LAHETEROPRIO_PRINT_STAT
				else
				{
					if (current_wgroupid == wgroupid)
					{
						lastats.task_list_empty_per_worker[workerid][wgroup_access_order[idx_access_item].prio_idx] += 1;
					}
				}
#endif
			}
		}
	}
	else
	{
		// !hp->use_locality
		/* Check that some tasks are available for the current worker arch */
		if(hp->nb_remaining_tasks_per_arch_index[worker->arch_index] != 0)
		{
			/* Ideally we would like to fill the prefetch array */
			unsigned nb_tasks_to_prefetch = (STARPU_HETEROPRIO_MAX_PREFETCH-worker->tasks_queue.ntasks);
			/* But there are maybe less tasks than that! */
			if(nb_tasks_to_prefetch > hp->nb_remaining_tasks_per_arch_index[worker->arch_index])
			{
				nb_tasks_to_prefetch = hp->nb_remaining_tasks_per_arch_index[worker->arch_index];
			}
			/* But in case there are less tasks than worker we take the minimum */
			if(hp->nb_remaining_tasks_per_arch_index[worker->arch_index] < starpu_sched_ctx_get_nworkers(sched_ctx_id))
			{
				if(worker->tasks_queue.ntasks == 0)
					nb_tasks_to_prefetch = 1;
				else
					nb_tasks_to_prefetch = 0;
			}

			unsigned idx_prio, arch_index;
			/* We iterate until we found all the tasks we need */
			for(idx_prio = 0; nb_tasks_to_prefetch && idx_prio < hp->nb_prio_per_arch_index[worker->arch_index]; ++idx_prio)
			{
				/* Retrieve the bucket using the mapping */
				struct _heteroprio_bucket* bucket = &hp->buckets[hp->prio_mapping_per_arch_index[worker->arch_index][idx_prio]];
				/* Ensure we can compute task from this bucket */
				STARPU_ASSERT(bucket->valid_archs & worker->arch_type);
				/* Take nb_tasks_to_prefetch tasks if possible */
				while(!starpu_task_list_empty(&bucket->tasks_queue[0]) && nb_tasks_to_prefetch &&
					(bucket->factor_base_arch_index == 0 ||
					worker->arch_index == bucket->factor_base_arch_index ||
					(((float)bucket->tasks_queue_ntasks)/((float)hp->nb_workers_per_arch_index[bucket->factor_base_arch_index])) >= bucket->slow_factors_per_index[worker->arch_index]
					))
				{
					task = starpu_task_list_pop_front(&bucket->tasks_queue[0]);
					if(!starpu_worker_can_execute_task(workerid, task, 0))
					{
						// Put the task back because worker can't execute it (e.g. codelet.can_execute)
						starpu_task_list_push_front(&bucket->tasks_queue[0], task);
						break;
					}
					/* Save the task */
					STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), workerid);
					starpu_st_prio_deque_push_front_task(&worker->tasks_queue, task);

					/* Update general counter */
					hp->nb_prefetched_tasks_per_arch_index[worker->arch_index] += 1;
					hp->total_tasks_in_buckets -= 1;
					bucket->tasks_queue_ntasks -= 1;

					for(arch_index = 0; arch_index < STARPU_NB_TYPES; ++arch_index)
					{
						/* We test the archs on the bucket and not on task->where since it is restrictive */
						if(bucket->valid_archs & starpu_heteroprio_types_to_arch(arch_index))
						{
							hp->nb_remaining_tasks_per_arch_index[arch_index] -= 1;
						}
					}
					/* Decrease the number of tasks to found */
					nb_tasks_to_prefetch -= 1;
					nb_added_tasks       += 1;
					// TODO starpu_prefetch_task_input_for(task, workerid);
				}
			}
		}

		task = NULL;

		/* The worker has some tasks in its queue */
		if(worker->tasks_queue.ntasks)
		{
			task = starpu_st_prio_deque_pop_task_for_worker(&worker->tasks_queue, workerid, NULL);
			hp->nb_prefetched_tasks_per_arch_index[worker->arch_index] -= 1;
		}
		/* Otherwise look if we can steal some work */
		else if(hp->nb_prefetched_tasks_per_arch_index[worker->arch_index])
		{
			/* If HETEROPRIO_MAX_PREFETCH==1 it should not be possible to steal work */
			STARPU_ASSERT(STARPU_HETEROPRIO_MAX_PREFETCH != 1);

			struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

			struct starpu_sched_ctx_iterator it;

			workers->init_iterator(workers, &it);
			unsigned victim;
			unsigned current_worker;

			/* Start stealing from just after ourself */
			while(workers->has_next(workers, &it))
			{
				current_worker = workers->get_next(workers, &it);
				if(current_worker == workerid)
					break;
			}

			/* circular loop */
			while (1)
			{
				if (!workers->has_next(workers, &it))
				{
					/* End of the list, restart from the beginning */
					workers->init_iterator(workers, &it);
				}
				while(workers->has_next(workers, &it))
				{
					victim = workers->get_next(workers, &it);
					/* When getting on ourself again, we're done trying to find work */
					if(victim == workerid)
						goto done;

					/* If it is the same arch and there is a task to steal */
					if(hp->workers_heteroprio[victim].arch_index == worker->arch_index
						&& hp->workers_heteroprio[victim].tasks_queue.ntasks)
					{
						/* ensure the worker is not currently prefetching its data */
						starpu_worker_lock(victim);

						if(hp->workers_heteroprio[victim].arch_index == worker->arch_index
						   && hp->workers_heteroprio[victim].tasks_queue.ntasks)
						{
							/* steal the last added task */
							task = starpu_st_prio_deque_pop_task_for_worker(&hp->workers_heteroprio[victim].tasks_queue, workerid, NULL);
							/* we steal a task update global counter */
							hp->nb_prefetched_tasks_per_arch_index[hp->workers_heteroprio[victim].arch_index] -= 1;

							starpu_worker_unlock(victim);
							goto done;
						}
						starpu_worker_unlock(victim);
					}
				}
			}
done:		;
		}
	}

	if (!task)
	{
		/* Tell pushers that we are waiting for tasks_queue for us */
		starpu_bitmap_set(&hp->waiters, workerid);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->policy_mutex);

	if(task &&_starpu_get_nsched_ctxs() > 1)
	{
		starpu_worker_relax_on();
		_starpu_sched_ctx_lock_write(sched_ctx_id);
		starpu_worker_relax_off();
		if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, task))
			task = NULL;
		_starpu_sched_ctx_unlock_write(sched_ctx_id);

		if(hp->use_locality)
		{
#ifdef LAHETEROPRIO_PRINT_STAT
			{
				const unsigned best_node_now = get_best_mem_node(task, hp, hp->pushStrategyToUse);
				if (best_node_now != src_mem_node)
				{
					lastats.pop_redirect[workerid][src_mem_node] += 1;
				}
			}
#endif
			if (hp->pushStrategySet == PUSH_AUTO)
			{
				unsigned best_node_now[PUSH_NB_AUTO] = { 0 };
				unsigned idx_strategy;
				for (idx_strategy = 0; idx_strategy < PUSH_NB_AUTO; ++idx_strategy)
				{
					best_node_now[idx_strategy] = get_best_mem_node(task, hp, idx_strategy);
				}
				STARPU_PTHREAD_MUTEX_LOCK(&hp->push_history_mutex);
				unsigned idx_best_strategy = 0;
				for (idx_strategy = 0; idx_strategy < PUSH_NB_AUTO; ++idx_strategy)
				{
					if (best_node_now[idx_strategy] == best_node_previous[idx_strategy])
					{
						hp->pushStrategyHistory[idx_strategy] += 1;
					}
					if (hp->pushStrategyHistory[idx_strategy] >=
						hp->pushStrategyHistory[idx_best_strategy])
					{
						idx_best_strategy = idx_strategy;
					}
				}
				hp->pushStrategyToUse = idx_best_strategy;
				STARPU_PTHREAD_MUTEX_UNLOCK(&hp->push_history_mutex);
			}
		}
	}

	if(!hp->use_locality)
	{
		/* if we have task (task) me way have some in the queue (worker->tasks_queue_size) that was freshly added (nb_added_tasks) */
		if(task && worker->tasks_queue.ntasks && nb_added_tasks && starpu_get_prefetch_flag())
		{
			/* TODO berenger: iterate in the other sense */
			struct starpu_task *task_to_prefetch = NULL;
			for (task_to_prefetch  = starpu_task_prio_list_begin(&worker->tasks_queue.list);
			     (task_to_prefetch != starpu_task_prio_list_end(&worker->tasks_queue.list) &&
			      nb_added_tasks && hp->nb_remaining_tasks_per_arch_index[worker->arch_index] != 0);
			     task_to_prefetch  = starpu_task_prio_list_next(&worker->tasks_queue.list, task_to_prefetch))
			{
				/* prefetch from closest to end task */
				if (!task_to_prefetch->prefetched) /* FIXME: it seems we are prefetching several times?? */
				{
					starpu_prefetch_task_input_for(task_to_prefetch, workerid);
				}
				nb_added_tasks -= 1;
			}
		}
	}

	return task;
}

static void pre_exec_hook_heteroprio_policy(struct starpu_task *task, unsigned sched_ctx_id)
{
	(void) task;
	const unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	if(hp->freeze_data_gathering || !hp->use_auto_calibration)
		return;

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->policy_mutex);
	starpu_worker_relax_off();

	struct timespec tsnow;
	_starpu_clock_gettime(&tsnow);
	const double now = starpu_timing_timespec_to_us(&tsnow);

	// Register free time between the post and pre hook
	hp->current_arch_free_time[starpu_worker_get_type(workerid)] += now - hp->last_hook_exec_time[workerid];

	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->policy_mutex);

	hp->last_hook_exec_time[workerid] = now;
}

static void post_exec_hook_heteroprio_policy(struct starpu_task *task, unsigned sched_ctx_id)
{
	const unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_heteroprio_data *hp = (struct _starpu_heteroprio_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	if(hp->freeze_data_gathering || !hp->use_auto_calibration)
		return;

	struct timespec tsnow;
	_starpu_clock_gettime(&tsnow);
	const double now = starpu_timing_timespec_to_us(&tsnow);
	const double busy_time = now - hp->last_hook_exec_time[workerid];

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&hp->policy_mutex);
	starpu_worker_relax_off();

	// Register the busy time between the pre and post hook
	hp->current_arch_busy_time[starpu_worker_get_type(workerid)] += busy_time;

	// Register task execution
	const int prio = get_task_auto_priority(hp, task);
	if(prio != -1)
	{
		register_task_arch_execution(hp, prio, starpu_worker_get_type(workerid));
		register_execution_time(hp, starpu_worker_get_type(workerid), prio, busy_time);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&hp->policy_mutex);

	hp->last_hook_exec_time[workerid] = now;
}

struct starpu_sched_policy _starpu_sched_heteroprio_policy =
{
	.init_sched = initialize_heteroprio_policy,
	.deinit_sched = deinitialize_heteroprio_policy,
	.add_workers = add_workers_heteroprio_policy,
	.remove_workers = remove_workers_heteroprio_policy,
	.push_task = push_task_heteroprio_policy,
	.simulate_push_task = NULL,
	.push_task_notify = NULL,
	.pop_task = pop_task_heteroprio_policy,
	.pre_exec_hook = pre_exec_hook_heteroprio_policy,
	.post_exec_hook = post_exec_hook_heteroprio_policy,
	.policy_name = "heteroprio",
	.policy_description = "heteroprio",
	.worker_type = STARPU_WORKER_LIST,
	.prefetches = 1,
};
