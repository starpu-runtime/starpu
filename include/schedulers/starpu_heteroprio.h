/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SCHEDULER_HETEROPRIO_H__
#define __STARPU_SCHEDULER_HETEROPRIO_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_HeteroPrio Heteroprio Scheduler
   @brief This is the interface for the heteroprio scheduler
   @{
 */

#define STARPU_HETEROPRIO_MAX_PREFETCH 2
#if STARPU_HETEROPRIO_MAX_PREFETCH <= 0
#error STARPU_HETEROPRIO_MAX_PREFETCH == 1 means no prefetch so STARPU_HETEROPRIO_MAX_PREFETCH must >= 1
#endif

#define STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY_COUNT 28

enum starpu_autoheteroprio_priority_ordering_policy
{
	STARPU_HETEROPRIO_NOD_TIME_COMBINATION, // 0
	STARPU_HETEROPRIO_BEST_NODS_SCORE,
	STARPU_HETEROPRIO_BEST_NODS,
	STARPU_HETEROPRIO_URT_PURE,
	STARPU_HETEROPRIO_URT,
	STARPU_HETEROPRIO_URT_2, // 5
	STARPU_HETEROPRIO_URT_DOT_DIFF_PURE,
	STARPU_HETEROPRIO_URT_DOT_DIFF_PURE_2,
	STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE,
	STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE_2,
	STARPU_HETEROPRIO_URT_DOT_DIFF_2, // 10
	STARPU_HETEROPRIO_URT_DOT_DIFF_3,
	STARPU_HETEROPRIO_URT_DOT_DIFF_4,
	STARPU_HETEROPRIO_URT_DOT_DIFF_5,
	STARPU_HETEROPRIO_URT_DOT_DIFF_6,
	STARPU_HETEROPRIO_URT_DOT_DIFF_7, // 15
	STARPU_HETEROPRIO_URT_DOT_DIFF_8,
	STARPU_HETEROPRIO_URT_DOT_DIFF_9,
	STARPU_HETEROPRIO_URT_DOT_DIFF_10,
	STARPU_HETEROPRIO_URT_DOT_DIFF_11,
	STARPU_HETEROPRIO_URTS_PER_SECONDS, // 20
	STARPU_HETEROPRIO_URTS_PER_SECONDS_2,
	STARPU_HETEROPRIO_URTS_PER_SECONDS_DIFF,
	STARPU_HETEROPRIO_URTS_TIME_RELEASED_DIFF,
	STARPU_HETEROPRIO_URTS_TIME_COMBINATION,
	STARPU_HETEROPRIO_NODS_PER_SECOND,
	STARPU_HETEROPRIO_NODS_TIME_RELEASED,
	STARPU_HETEROPRIO_NODS_TIME_RELEASED_DIFF
};

static const char starpu_autoheteroprio_priority_ordering_policy_names[STARPU_AUTOHETEROPRIO_PRIORITY_ORDERING_POLICY_COUNT][64] = {
	"STARPU_HETEROPRIO_NOD_TIME_COMBINATION",
	"STARPU_HETEROPRIO_BEST_NODS_SCORE",
	"STARPU_HETEROPRIO_BEST_NODS",
	"STARPU_HETEROPRIO_URT_PURE",
	"STARPU_HETEROPRIO_URT",
	"STARPU_HETEROPRIO_URT_2",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_PURE",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_PURE_2",
	"STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE",
	"STARPU_HETEROPRIO_URT_DOT_REL_DIFF_PURE_2",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_2",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_3",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_4",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_5",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_6",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_7",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_8",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_9",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_10",
	"STARPU_HETEROPRIO_URT_DOT_DIFF_11",
	"STARPU_HETEROPRIO_URTS_PER_SECONDS",
	"STARPU_HETEROPRIO_URTS_PER_SECONDS_2",
	"STARPU_HETEROPRIO_URTS_PER_SECONDS_DIFF",
	"STARPU_HETEROPRIO_URTS_TIME_RELEASED_DIFF",
	"STARPU_HETEROPRIO_URTS_TIME_COMBINATION",
	"STARPU_HETEROPRIO_NODS_PER_SECOND",
	"STARPU_HETEROPRIO_NODS_TIME_RELEASED",
	"STARPU_HETEROPRIO_NODS_TIME_RELEASED_DIFF",
};

/** Set if heteroprio should use data locality or not */
void starpu_heteroprio_set_use_locality(unsigned sched_ctx_id, unsigned use_locality);

/** Tell how many prio there are for a given arch */
void starpu_heteroprio_set_nb_prios(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned max_prio);

/** Set the mapping for a given arch prio=>bucket */
void starpu_heteroprio_set_mapping(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned source_prio, unsigned dest_bucket_id);

/** Tell which arch is the faster for the tasks of a bucket (optional) */
void starpu_heteroprio_set_faster_arch(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned bucket_id);

/** Tell how slow is a arch for the tasks of a bucket (optional) */
void starpu_heteroprio_set_arch_slow_factor(unsigned sched_ctx_id, enum starpu_worker_archtype arch, unsigned bucket_id, float slow_factor);

/** One memory node will be one wgroup */
void starpu_heteroprio_map_wgroup_memory_nodes(unsigned sched_ctx_id);

/** Print the current setup groups */
void starpu_heteroprio_print_wgroups(FILE *stream, unsigned sched_ctx_id);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHEDULER_HETEROPRIO_H__ */
