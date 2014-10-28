/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

enum starpu_sched_component_properties
{
	STARPU_SCHED_COMPONENT_HOMOGENEOUS = (1<<0),
	STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE = (1<<1)
};

#define STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component) ((component)->properties & STARPU_SCHED_COMPONENT_HOMOGENEOUS)
#define STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component) ((component)->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE)

/* struct starpu_sched_component are scheduler modules, a scheduler is a tree-like
 * structure of them, some parts of scheduler can be shared by several contexes
 * to perform some local optimisations, so, for all components, a list of parent is
 * defined indexed by sched_ctx_id
 *
 * they embed there specialised method in a pseudo object-style, so calls are like component->push_task(component,task)
 *
 */
struct starpu_sched_component
{
	/* The tree containing the component */
	struct starpu_sched_tree *tree;

	/* the set of workers in the component's subtree
	 */
	struct starpu_bitmap *workers;
	/* the workers available in context
	 * this member is set with :
	 * component->workers UNION tree->workers UNION
	 * component->child[i]->workers_in_ctx iff exist x such as component->children[i]->parents[x] == component
	 */
	struct starpu_bitmap *workers_in_ctx;

	/* component's private data, no restriction on use
	 */
	void *data;

	/* the numbers of component's children
	 */
	int nchildren;
	/* the vector of component's children
	 */
	struct starpu_sched_component **children;
	/* the numbers of component's parents
	 */
	int nparents;
	/* may be shared by several contexts
	 * so we need several parents
	 */
	struct starpu_sched_component **parents;

	void (*add_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	void (*remove_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	void (*add_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);
	void (*remove_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);

	/* component->push_task(component, task)
	 * this function is called to push a task on component subtree, this can either
	 * perform a recursive call on a child or store the task in the component, then
	 * it will be returned by a further pull_task call
	 *
	 * the caller must ensure that component is able to execute task
	 */
	int (*push_task)(struct starpu_sched_component *,
			 struct starpu_task *);
	/* this function is called by workers to get a task on them parents
	 * this function should first return a localy stored task or perform
	 * a recursive call on parent
	 *
	 * a default implementation simply do a recursive call on parent
	 */
	struct starpu_task * (*pull_task)(struct starpu_sched_component *);

	/* This function is called by a component which implements a queue, allowing it to
	 * signify to its parents that an empty slot is available in its queue.
	 * The basic implementation of this function is a recursive call to its
	 * parents, the user have to specify a personally-made function to catch those
	 * calls.
	 */
	int (*can_push)(struct starpu_sched_component *component);
	/* This function allow a component to wake up a worker.
	 * It is currently called by component which implements a queue, to signify to
	 * its children that a task have been pushed in its local queue, and is
	 * available to been popped by a worker, for example.
	 * The basic implementation of this function is a recursive call to
	 * its children, until at least one worker have been woken up.
	 */
	void (*can_pull)(struct starpu_sched_component *component);

	/* this function is an heuristic that compute load of subtree, basicaly
	 * it compute
	 * estimated_load(component) = sum(estimated_load(component_children)) +
	 *          nb_local_tasks / average(relative_speedup(underlying_worker))
	 */
	double (*estimated_load)(struct starpu_sched_component *component);
	double (*estimated_end)(struct starpu_sched_component *component);

	/* this function is called by starpu_sched_component_destroy just before freeing component
	 */
	void (*deinit_data)(struct starpu_sched_component *component);
	/* this function is called for each component when workers are added or removed from a context
	 */
	void (*notify_change_workers)(struct starpu_sched_component *component);

	/* is_homogeneous is 0 if workers in the component's subtree are heterogeneous,
	 * this field is set and updated automaticaly, you shouldn't write on it
	 */
	int properties;

#ifdef STARPU_HAVE_HWLOC
	/* in case of a modularized scheduler, this is set to the part of
	 * topology that is binded to this component, eg: a numa node for a ws
	 * component that would balance load between underlying sockets
	 */
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

	/* this array store worker components */
	struct starpu_sched_component *worker_components[STARPU_NMAXWORKERS];

	/* this lock is used to protect the scheduler,
	 * it is taken in read mode pushing a task
	 * and in write mode for adding or removing workers
	 */
	starpu_pthread_mutex_t lock;
};

/*******************************************************************************
 *							Scheduling Tree's Interface 					   *
 ******************************************************************************/

/* create an empty tree
 */
struct starpu_sched_tree *starpu_sched_tree_create(unsigned sched_ctx_id);
void starpu_sched_tree_destroy(struct starpu_sched_tree *tree);
struct starpu_sched_tree *starpu_get_tree(unsigned sched_ctx_id);

/* destroy component and all his child
 * except if they are shared between several contexts
 */
void starpu_sched_component_destroy_rec(struct starpu_sched_component *component);

/* update all the component->workers member recursively
 */
void starpu_sched_tree_update_workers(struct starpu_sched_tree *t);
/* idem for workers_in_ctx
 */
void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree *t);

int starpu_sched_tree_push_task(struct starpu_task *task);
struct starpu_task *starpu_sched_tree_pop_task();

void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

/*******************************************************************************
 *					Generic Scheduling Component's Interface 				   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_create(struct starpu_sched_tree *tree);
void starpu_sched_component_destroy(struct starpu_sched_component *component);

int starpu_sched_component_can_execute_task(struct starpu_sched_component *component, struct starpu_task *task);
int STARPU_WARN_UNUSED_RESULT starpu_sched_component_execute_preds(struct starpu_sched_component *component, struct starpu_task *task, double *length);
double starpu_sched_component_transfer_length(struct starpu_sched_component *component, struct starpu_task *task);
void starpu_sched_component_prefetch_on_node(struct starpu_sched_component *component, struct starpu_task *task);

/*******************************************************************************
 *							Worker Component's Interface 				   	   *
 ******************************************************************************/

/* no public create function for workers because we dont want to have several component_worker for a single workerid */
struct starpu_sched_component *starpu_sched_component_worker_get(unsigned sched_ctx, int workerid);
int starpu_sched_component_worker_get_workerid(struct starpu_sched_component *worker_component);

/* this function compare the available function of the component with the standard available for worker components*/
int starpu_sched_component_is_worker(struct starpu_sched_component *component);
int starpu_sched_component_is_simple_worker(struct starpu_sched_component *component);
int starpu_sched_component_is_combined_worker(struct starpu_sched_component *component);

void starpu_sched_component_worker_pre_exec_hook(struct starpu_task *task);
void starpu_sched_component_worker_post_exec_hook(struct starpu_task *task);

/*******************************************************************************
 *					Flow-control Fifo Component's Interface 				   *
 ******************************************************************************/

struct starpu_fifo_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

struct starpu_sched_component *starpu_sched_component_fifo_create(struct starpu_sched_tree *tree, struct starpu_fifo_data *fifo_data);
int starpu_sched_component_is_fifo(struct starpu_sched_component *component);

/*******************************************************************************
 *					Flow-control Prio Component's Interface 				   *
 ******************************************************************************/

struct starpu_prio_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

struct starpu_sched_component *starpu_sched_component_prio_create(struct starpu_sched_tree *tree, struct starpu_prio_data *prio_data);
int starpu_sched_component_is_prio(struct starpu_sched_component *component);

/*******************************************************************************
 *			Resource-mapping Work-Stealing Component's Interface 			   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_work_stealing_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_component_is_work_stealing(struct starpu_sched_component *component);
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task);

/*******************************************************************************
 *				Resource-mapping Random Component's Interface 			   	   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_random_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_component_is_random(struct starpu_sched_component *);

/*******************************************************************************
 *				Resource-mapping Eager Component's Interface 				   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_eager_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_component_is_eager(struct starpu_sched_component *);

/*******************************************************************************
 *			Resource-mapping Eager-Calibration Component's Interface 		   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_eager_calibration_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_component_is_eager_calibration(struct starpu_sched_component *);

/*******************************************************************************
 *				Resource-mapping MCT Component's Interface 					   *
 ******************************************************************************/

struct starpu_mct_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
};

/* create a component with mct_data paremeters
   a copy the struct starpu_mct_data * given is performed during the init_data call
   the mct component doesnt do anything but pushing tasks on no_perf_model_component and calibrating_component
*/
struct starpu_sched_component *starpu_sched_component_mct_create(struct starpu_sched_tree *tree, struct starpu_mct_data *mct_data);
int starpu_sched_component_is_mct(struct starpu_sched_component *component);

/*******************************************************************************
 *				Resource-mapping HEFT Component's Interface 				   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_heft_create(struct starpu_sched_tree *tree, struct starpu_mct_data *mct_data);
int starpu_sched_component_is_heft(struct starpu_sched_component *component);

/*******************************************************************************
 *		Special-purpose Best_Implementation Component's Interface 			   *
 ******************************************************************************/

/* this component select the best implementation for the first worker in context that can execute task.
 * and fill task->predicted and task->predicted_transfer
 * cannot have several child if push_task is called
 */
struct starpu_sched_component *starpu_sched_component_best_implementation_create(struct starpu_sched_tree *tree, void *arg STARPU_ATTRIBUTE_UNUSED);

struct starpu_perfmodel_select_data
{
	struct starpu_sched_component *calibrator_component;
	struct starpu_sched_component *no_perfmodel_component;
	struct starpu_sched_component *perfmodel_component;
};

/*******************************************************************************
 *			Special-purpose Perfmodel_Select Component's Interface	 		   *
 ******************************************************************************/

struct starpu_sched_component *starpu_sched_component_perfmodel_select_create(struct starpu_sched_tree *tree, struct starpu_perfmodel_select_data *perfmodel_select_data);
int starpu_sched_component_is_perfmodel_select(struct starpu_sched_component *component);

/*******************************************************************************
 *						Recipe Component's Interface	 					   *
 ******************************************************************************/

struct starpu_sched_component_composed_recipe;

/* create empty recipe */
struct starpu_sched_component_composed_recipe *starpu_sched_component_create_recipe(void);
struct starpu_sched_component_composed_recipe *starpu_sched_component_create_recipe_singleton(struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg);

/* add a function creation component to recipe */
void starpu_sched_recipe_add_component(struct starpu_sched_component_composed_recipe *recipe, struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg);

void starpu_destroy_composed_sched_component_recipe(struct starpu_sched_component_composed_recipe *);
struct starpu_sched_component *starpu_sched_component_composed_component_create(struct starpu_sched_tree *tree, struct starpu_sched_component_composed_recipe *recipe);


#ifdef STARPU_HAVE_HWLOC
/* null pointer mean to ignore a level L of hierarchy, then components of levels > L become children of level L - 1 */
struct starpu_sched_specs
{
	/* hw_loc_machine_composed_sched_component must be set as its the root of the topology */
	struct starpu_sched_component_composed_recipe *hwloc_machine_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_component_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_socket_composed_sched_component;
	struct starpu_sched_component_composed_recipe *hwloc_cache_composed_sched_component;

	/* this member should return a new allocated starpu_sched_component_composed_recipe or NULL
	 * the starpu_sched_component_composed_recipe_t must not include the worker component
	 */
	struct starpu_sched_component_composed_recipe *(*worker_composed_sched_component)(enum starpu_worker_archtype archtype);

	/* this flag indicate if heterogenous workers should be brothers or cousins,
	 * as example, if a gpu and a cpu should share or not there numa node
	 */
	int mix_heterogeneous_workers;
};

struct starpu_sched_tree *starpu_sched_component_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_specs);
#endif /* STARPU_HAVE_HWLOC */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_COMPONENT_H__ */
