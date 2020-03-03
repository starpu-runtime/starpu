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

#ifndef __STARPU_SCHED_COMPONENT_H__
#define __STARPU_SCHED_COMPONENT_H__

#include <starpu.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

enum starpu_sched_component_properties
{
	STARPU_SCHED_COMPONENT_HOMOGENEOUS = (1<<0),
	STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE = (1<<1)
};

#define STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component) ((component)->properties & STARPU_SCHED_COMPONENT_HOMOGENEOUS)
#define STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component) ((component)->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE)

struct starpu_sched_component
{
	struct starpu_sched_tree *tree;
	struct starpu_bitmap *workers;
	struct starpu_bitmap *workers_in_ctx;
	void *data;
	char *name;
	int nchildren;
	struct starpu_sched_component **children;
	int nparents;
	struct starpu_sched_component **parents;

	void (*add_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	void (*remove_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	void (*add_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);
	void (*remove_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);

	int (*push_task)(struct starpu_sched_component *, struct starpu_task *);
	struct starpu_task *(*pull_task)(struct starpu_sched_component *);

	int (*can_push)(struct starpu_sched_component *component);
	void (*can_pull)(struct starpu_sched_component *component);

	double (*estimated_load)(struct starpu_sched_component *component);
	double (*estimated_end)(struct starpu_sched_component *component);

	void (*deinit_data)(struct starpu_sched_component *component);
	void (*notify_change_workers)(struct starpu_sched_component *component);
	int properties;

#ifdef STARPU_HAVE_HWLOC
	hwloc_obj_t obj;
#else
	void *obj;
#endif
};

struct starpu_sched_tree
{
	struct starpu_sched_component *root;
	struct starpu_bitmap *workers;
	unsigned sched_ctx_id;
	struct starpu_sched_component *worker_components[STARPU_NMAXWORKERS];
	starpu_pthread_mutex_t lock;
};

struct starpu_sched_tree *starpu_sched_tree_create(unsigned sched_ctx_id) STARPU_ATTRIBUTE_MALLOC;
void starpu_sched_tree_destroy(struct starpu_sched_tree *tree);
struct starpu_sched_tree *starpu_sched_tree_get(unsigned sched_ctx_id);
void starpu_sched_tree_update_workers(struct starpu_sched_tree *t);
void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree *t);
int starpu_sched_tree_push_task(struct starpu_task *task);
int starpu_sched_component_push_task(struct starpu_sched_component *from, struct starpu_sched_component *to, struct starpu_task *task);
struct starpu_task *starpu_sched_tree_pop_task(unsigned sched_ctx);
struct starpu_task *starpu_sched_component_pull_task(struct starpu_sched_component *from, struct starpu_sched_component *to);
void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

struct starpu_sched_component *starpu_sched_component_create(struct starpu_sched_tree *tree, const char *name) STARPU_ATTRIBUTE_MALLOC;
void starpu_sched_component_add_child(struct starpu_sched_component* component, struct starpu_sched_component * child);
void starpu_sched_component_destroy(struct starpu_sched_component *component);
void starpu_sched_component_destroy_rec(struct starpu_sched_component *component);
int starpu_sched_component_can_execute_task(struct starpu_sched_component *component, struct starpu_task *task);
int STARPU_WARN_UNUSED_RESULT starpu_sched_component_execute_preds(struct starpu_sched_component *component, struct starpu_task *task, double *length);
double starpu_sched_component_transfer_length(struct starpu_sched_component *component, struct starpu_task *task);
void starpu_sched_component_prefetch_on_node(struct starpu_sched_component *component, struct starpu_task *task);

void starpu_sched_component_connect(struct starpu_sched_component *parent, struct starpu_sched_component *child);

struct starpu_sched_component *starpu_sched_component_worker_get(unsigned sched_ctx, int workerid);
int starpu_sched_component_worker_get_workerid(struct starpu_sched_component *worker_component);
int starpu_sched_component_is_worker(struct starpu_sched_component *component);
int starpu_sched_component_is_simple_worker(struct starpu_sched_component *component);
int starpu_sched_component_is_combined_worker(struct starpu_sched_component *component);
void starpu_sched_component_worker_pre_exec_hook(struct starpu_task *task);
void starpu_sched_component_worker_post_exec_hook(struct starpu_task *task);

double starpu_sched_component_estimated_load(struct starpu_sched_component * component);
double starpu_sched_component_estimated_end_min(struct starpu_sched_component * component);
double starpu_sched_component_estimated_end_average(struct starpu_sched_component * component);

struct starpu_sched_component_fifo_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

struct starpu_sched_component *starpu_sched_component_fifo_create(struct starpu_sched_tree *tree, struct starpu_sched_component_fifo_data *fifo_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_fifo(struct starpu_sched_component *component);

struct starpu_sched_component_prio_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};
struct starpu_sched_component *starpu_sched_component_prio_create(struct starpu_sched_tree *tree, struct starpu_sched_component_prio_data *prio_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_prio(struct starpu_sched_component *component);

struct starpu_sched_component *starpu_sched_component_work_stealing_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_work_stealing(struct starpu_sched_component *component);
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task);

struct starpu_sched_component *starpu_sched_component_random_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_random(struct starpu_sched_component *);

struct starpu_sched_component *starpu_sched_component_eager_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_eager(struct starpu_sched_component *);

struct starpu_sched_component *starpu_sched_component_eager_calibration_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_eager_calibration(struct starpu_sched_component *);

struct starpu_sched_component_mct_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
};
struct starpu_sched_component *starpu_sched_component_mct_create(struct starpu_sched_tree *tree, struct starpu_sched_component_mct_data *mct_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_mct(struct starpu_sched_component *component);

struct starpu_sched_component *starpu_sched_component_heft_create(struct starpu_sched_tree *tree, struct starpu_sched_component_mct_data *mct_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_heft(struct starpu_sched_component *component);

struct starpu_sched_component *starpu_sched_component_best_implementation_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED) STARPU_ATTRIBUTE_MALLOC;

struct starpu_sched_component_perfmodel_select_data
{
	struct starpu_sched_component *calibrator_component;
	struct starpu_sched_component *no_perfmodel_component;
	struct starpu_sched_component *perfmodel_component;
};
struct starpu_sched_component *starpu_sched_component_perfmodel_select_create(struct starpu_sched_tree *tree, struct starpu_sched_component_perfmodel_select_data *perfmodel_select_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_perfmodel_select(struct starpu_sched_component *component);

void starpu_initialize_prio_center_policy(unsigned sched_ctx_id);

struct starpu_sched_component_composed_recipe;
struct starpu_sched_component_composed_recipe *starpu_sched_component_composed_recipe_create(void) STARPU_ATTRIBUTE_MALLOC;
struct starpu_sched_component_composed_recipe *starpu_sched_component_composed_recipe_create_singleton(struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg) STARPU_ATTRIBUTE_MALLOC;
void starpu_sched_component_composed_recipe_add(struct starpu_sched_component_composed_recipe *recipe, struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg);
void starpu_sched_component_composed_recipe_destroy(struct starpu_sched_component_composed_recipe *);
struct starpu_sched_component *starpu_sched_component_composed_component_create(struct starpu_sched_tree *tree, struct starpu_sched_component_composed_recipe *recipe) STARPU_ATTRIBUTE_MALLOC;

#ifdef STARPU_HAVE_HWLOC
struct starpu_sched_component_specs
{
	struct starpu_sched_component_composed_recipe *hwloc_machine_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_component_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_socket_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_cache_composed_sched_component;

	struct starpu_sched_component_composed_recipe *(*worker_composed_sched_component)(enum starpu_worker_archtype archtype);
	int mix_heterogeneous_workers;
};

struct starpu_sched_tree *starpu_sched_component_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_component_specs s);
#endif /* STARPU_HAVE_HWLOC */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_COMPONENT_H__ */
