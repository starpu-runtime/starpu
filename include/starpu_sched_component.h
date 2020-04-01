/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2017       Arthur Chevalier
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

/**
   @defgroup API_Modularized_Scheduler Modularized Scheduler Interface
   @{
*/

/**
   flags for starpu_sched_component::properties
*/
enum starpu_sched_component_properties
{
	/** indicate that all workers have the same starpu_worker_archtype */
	STARPU_SCHED_COMPONENT_HOMOGENEOUS = (1<<0),
	/** indicate that all workers have the same memory component */
	STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE = (1<<1)
};

/**
   indicate if component is homogeneous
*/
#define STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component) ((component)->properties & STARPU_SCHED_COMPONENT_HOMOGENEOUS)

/**
   indicate if all workers have the same memory component
*/
#define STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component) ((component)->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE)

/**
   Structure for a scheduler module.  A scheduler is a
   tree-like structure of them, some parts of scheduler can be shared by
   several contexes to perform some local optimisations, so, for all
   components, a list of parent is defined by \c sched_ctx_id. They
   embed there specialised method in a pseudo object-style, so calls are
   like <c>component->push_task(component,task)</c>
*/
struct starpu_sched_component
{
	/** The tree containing the component*/
	struct starpu_sched_tree *tree;
	/** set of underlying workers */
	struct starpu_bitmap *workers;
	/**
	   subset of starpu_sched_component::workers that is currently available in the context
	   The push method should take this value into account, it is set with:
	   component->workers UNION tree->workers UNION
	   component->child[i]->workers_in_ctx iff exist x such as component->children[i]->parents[x] == component
	*/
	struct starpu_bitmap *workers_in_ctx;
	/** private data */
	void *data;
	char *name;
	/** number of compoments's children */
	unsigned nchildren;
	/** vector of component's children */
	struct starpu_sched_component **children;
	/** number of component's parents */
	unsigned nparents;
	/** vector of component's parents */
	struct starpu_sched_component **parents;

	/** add a child to component */
	void (*add_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	/** remove a child from component */
	void (*remove_child)(struct starpu_sched_component *component, struct starpu_sched_component *child);
	void (*add_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);
	void (*remove_parent)(struct starpu_sched_component *component, struct starpu_sched_component *parent);

	/**
	   push a task in the scheduler module. this function is called to
	   push a task on component subtree, this can either perform a
	   recursive call on a child or store the task in the component,
	   then it will be returned by a further pull_task call.
	   the caller must ensure that component is able to execute task.
	   This method must either return 0 if it the task was properly stored or
	   passed over to a child component, or return a value different from 0 if the
	   task could not be consumed (e.g. the queue is full).
	*/
	int (*push_task)(struct starpu_sched_component *, struct starpu_task *);

	/**
	   pop a task from the scheduler module. this function is called by workers to get a task from their
	   parents. this function should first return a locally stored task
	   or perform a recursive call on the parents.
	   the task returned by this function should be executable by the caller
	*/
	struct starpu_task *(*pull_task)(struct starpu_sched_component *from, struct starpu_sched_component *to);

	/**
	   This function is called by a component which implements a queue,
	   allowing it to signify to its parents that an empty slot is
	   available in its queue. This should return 1 if some tasks could be pushed
	   The basic implementation of this function
	   is a recursive call to its parents, the user has to specify a
	   personally-made function to catch those calls.
	*/
	int (*can_push)(struct starpu_sched_component *from, struct starpu_sched_component *to);

	/**
	   This function allow a component to wake up a worker. It is
	   currently called by component which implements a queue, to
	   signify to its children that a task have been pushed in its local
	   queue, and is available to be popped by a worker, for example.
	   This should return 1 if some some container or worker could (or will) pull
	   some tasks.
	   The basic implementation of this function is a recursive call to
	   its children, until at least one worker have been woken up.
	*/
	int (*can_pull)(struct starpu_sched_component *component);

	int (*notify)(struct starpu_sched_component* component, int message_ID, void* arg);

	/**
	   heuristic to compute load of scheduler module. Basically the number of tasks divided by the sum
	   of relatives speedup of workers available in context.
	   estimated_load(component) = sum(estimated_load(component_children)) + nb_local_tasks / average(relative_speedup(underlying_worker))
	*/
	double (*estimated_load)(struct starpu_sched_component *component);
	/**
	   return the time when a worker will enter in starvation. This function is relevant only if the task->predicted
	   member has been set.
	*/
	double (*estimated_end)(struct starpu_sched_component *component);

	/**
	   called by starpu_sched_component_destroy. Should free data allocated during creation
	*/
	void (*deinit_data)(struct starpu_sched_component *component);

	/**
	   this function is called for each component when workers are added or removed from a context
	*/
	void (*notify_change_workers)(struct starpu_sched_component *component);
	int properties;

#ifdef STARPU_HAVE_HWLOC
	/**
	   the hwloc object associated to scheduler module. points to the
	   part of topology that is binded to this component, eg: a numa
	   node for a ws component that would balance load between
	   underlying sockets
	*/
	hwloc_obj_t obj;
#else
	void *obj;
#endif
};

/**
   The actual scheduler
*/
struct starpu_sched_tree
{
	/**
	   entry module of the scheduler
	*/
	struct starpu_sched_component *root;
	/**
	   set of workers available in this context, this value is used to mask workers in modules
	*/
	struct starpu_bitmap *workers;
	/**
	   context id of the scheduler
	*/
	unsigned sched_ctx_id;
	/**
	   lock used to protect the scheduler, it is taken in read mode pushing a task and in write mode for adding or
	   removing workers
	*/
	starpu_pthread_mutex_t lock;
};

void starpu_initialize_prio_center_policy(unsigned sched_ctx_id);

/**
   @name Scheduling Tree API
   @{
*/

/**
   create a empty initialized starpu_sched_tree
*/
struct starpu_sched_tree *starpu_sched_tree_create(unsigned sched_ctx_id) STARPU_ATTRIBUTE_MALLOC;
/**
   destroy tree and free all non shared component in it.
*/
void starpu_sched_tree_destroy(struct starpu_sched_tree *tree);
struct starpu_sched_tree *starpu_sched_tree_get(unsigned sched_ctx_id);
/**
   recursively set all starpu_sched_component::workers, do not take into account shared parts (except workers).
*/
void starpu_sched_tree_update_workers(struct starpu_sched_tree *t);
/**
   recursively set all starpu_sched_component::workers_in_ctx, do not take into account shared parts (except workers)
*/
void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree *t);
/**
   compatibility with starpu_sched_policy interface
*/
int starpu_sched_tree_push_task(struct starpu_task *task);
/**
   compatibility with starpu_sched_policy interface
*/
struct starpu_task *starpu_sched_tree_pop_task(unsigned sched_ctx);

/**
   Push a task to a component. This is a helper for <c>component->push_task(component, task)</c> plus tracing.
*/
int starpu_sched_component_push_task(struct starpu_sched_component *from, struct starpu_sched_component *to, struct starpu_task *task);

/**
   Pull a task from a component. This is a helper for <c>component->pull_task(component)</c> plus tracing.
*/
struct starpu_task *starpu_sched_component_pull_task(struct starpu_sched_component *from, struct starpu_sched_component *to);

struct starpu_task* starpu_sched_component_pump_to(struct starpu_sched_component *component, struct starpu_sched_component *to, int* success);
struct starpu_task* starpu_sched_component_pump_downstream(struct starpu_sched_component *component, int* success);
int starpu_sched_component_send_can_push_to_parents(struct starpu_sched_component * component);
/**
   compatibility with starpu_sched_policy interface
*/
void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
/**
   compatibility with starpu_sched_policy interface
*/
void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

/**
   Attach component \p child to parent \p parent. Some component may accept only one child, others accept several (e.g. MCT)
*/
void starpu_sched_component_connect(struct starpu_sched_component *parent, struct starpu_sched_component *child);

/** @} */

/**
   @name Generic Scheduling Component API
   @{
*/

typedef struct starpu_sched_component * (*starpu_sched_component_create_t)(struct starpu_sched_tree *tree, void *data);

/**
   allocate and initialize component field with defaults values :
   .pop_task make recursive call on father
   .estimated_load compute relative speedup and tasks in sub tree
   .estimated_end return the minimum of recursive call on children
   .add_child is starpu_sched_component_add_child
   .remove_child is starpu_sched_component_remove_child
   .notify_change_workers does nothing
   .deinit_data does nothing
*/
struct starpu_sched_component *starpu_sched_component_create(struct starpu_sched_tree *tree, const char *name) STARPU_ATTRIBUTE_MALLOC;

/**
   free data allocated by starpu_sched_component_create and call component->deinit_data(component)
   set to <c>NULL</c> the member starpu_sched_component::fathers[sched_ctx_id] of all child if its equal to \p component
*/

void starpu_sched_component_destroy(struct starpu_sched_component *component);
/**
   recursively destroy non shared parts of a \p component 's tree
*/
void starpu_sched_component_destroy_rec(struct starpu_sched_component *component);

void starpu_sched_component_add_child(struct starpu_sched_component* component, struct starpu_sched_component * child);

/**
   return true iff \p component can execute \p task, this function take into account the workers available in the scheduling context
*/
int starpu_sched_component_can_execute_task(struct starpu_sched_component *component, struct starpu_task *task);

/**
   return a non <c>NULL</c> value if \p component can execute \p task.
   write the execution prediction length for the best implementation of the best worker available and write this at \p length address.
   this result is more relevant if starpu_sched_component::is_homogeneous is non <c>NULL</c>.
   if a worker need to be calibrated for an implementation, nan is set to \p length.
*/
int STARPU_WARN_UNUSED_RESULT starpu_sched_component_execute_preds(struct starpu_sched_component *component, struct starpu_task *task, double *length);

/**
   return the average time to transfer \p task data to underlying \p component workers.
*/
double starpu_sched_component_transfer_length(struct starpu_sched_component *component, struct starpu_task *task);

void starpu_sched_component_prefetch_on_node(struct starpu_sched_component *component, struct starpu_task *task);

/** @} */

/**
   @name Worker Component API
   @{
*/

/**
   return the struct starpu_sched_component corresponding to \p workerid. Undefined if \p workerid is not a valid workerid
*/
struct starpu_sched_component *starpu_sched_component_worker_get(unsigned sched_ctx, int workerid);
struct starpu_sched_component *starpu_sched_component_worker_new(unsigned sched_ctx, int workerid);

/**
   Create a combined worker that pushes tasks in parallel to workers \p workers (size \p nworkers).
*/
struct starpu_sched_component *starpu_sched_component_parallel_worker_create(struct starpu_sched_tree *tree, unsigned nworkers, unsigned *workers);

/**
   return the workerid of \p worker_component, undefined if starpu_sched_component_is_worker(worker_component) == 0
*/
int starpu_sched_component_worker_get_workerid(struct starpu_sched_component *worker_component);

/**
   return true iff \p component is a worker component
*/
int starpu_sched_component_is_worker(struct starpu_sched_component *component);

/**
   return true iff \p component is a simple worker component
*/
int starpu_sched_component_is_simple_worker(struct starpu_sched_component *component);

/**
   return true iff \p component is a combined worker component
*/
int starpu_sched_component_is_combined_worker(struct starpu_sched_component *component);

/**
   compatibility with starpu_sched_policy interface
   update predictions for workers
*/
void starpu_sched_component_worker_pre_exec_hook(struct starpu_task *task, unsigned sched_ctx_id);

/**
   compatibility with starpu_sched_policy interface
*/
void starpu_sched_component_worker_post_exec_hook(struct starpu_task *task, unsigned sched_ctx_id);

/** @} */

/**
   @name Flow-control Fifo Component API
   @{
*/

/**
   default function for the pull component method, just call pull of parents until one of them returns a task
*/
struct starpu_task * starpu_sched_component_parents_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to);

/**
   default function for the can_push component method, just call can_push of parents until one of them returns non-zero
*/
int starpu_sched_component_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to);

/**
default function for the can_pull component method, just call can_pull of children until one of them returns non-zero
*/
int starpu_sched_component_can_pull(struct starpu_sched_component * component);

/**
   function for the can_pull component method, call can_pull of all children
*/
int starpu_sched_component_can_pull_all(struct starpu_sched_component * component);

/**
   default function for the estimated_load component method, just sum up the loads
   of the children of the component.
*/
double starpu_sched_component_estimated_load(struct starpu_sched_component * component);

/**
   function that can be used for the estimated_end component method, compute the minimum completion time of the children.
*/
double starpu_sched_component_estimated_end_min(struct starpu_sched_component * component);

/**
   function that can be used for the estimated_end component method, compute
   the minimum completion time of the children, and add to it an estimation of how
   existing queued work, plus the exp_len work, can be completed. This is typically
   used instead of starpu_sched_component_estimated_end_min when the component
   contains a queue of tasks, which thus needs to be added to the estimations.
*/
double starpu_sched_component_estimated_end_min_add(struct starpu_sched_component * component, double exp_len);

/**
   default function for the estimated_end component method, compute the average completion time of the children.
*/
double starpu_sched_component_estimated_end_average(struct starpu_sched_component * component);

struct starpu_sched_component_fifo_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
	int ready;
};

/**
   Return a struct starpu_sched_component with a fifo. A stable sort is performed according to tasks priorities.
   A push_task call on this component does not perform recursive calls, underlying components will have to call pop_task to get it.
   starpu_sched_component::estimated_end function compute the estimated length by dividing the sequential length by the number of underlying workers.
*/
struct starpu_sched_component *starpu_sched_component_fifo_create(struct starpu_sched_tree *tree, struct starpu_sched_component_fifo_data *fifo_data) STARPU_ATTRIBUTE_MALLOC;

/**
   return true iff \p component is a fifo component
*/
int starpu_sched_component_is_fifo(struct starpu_sched_component *component);

/** @} */

/**
   @name Flow-control Prio Component API
   @{
*/

struct starpu_sched_component_prio_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
	int ready;
};
struct starpu_sched_component *starpu_sched_component_prio_create(struct starpu_sched_tree *tree, struct starpu_sched_component_prio_data *prio_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_prio(struct starpu_sched_component *component);

/** @} */

/**
   @name Resource-mapping Work-Stealing Component API
   @{
*/

/**
   return a component that perform a work stealing scheduling. Tasks are pushed in a round robin way. estimated_end return the average of expected length of fifos, starting at the average of the expected_end of his children. When a worker have to steal a task, it steal a task in a round robin way, and get the last pushed task of the higher priority.
*/
struct starpu_sched_component *starpu_sched_component_work_stealing_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;

/**
   return true iff \p component is a work stealing component
 */
int starpu_sched_component_is_work_stealing(struct starpu_sched_component *component);

/**
   undefined if there is no work stealing component in the scheduler. If any, \p task is pushed in a default way if the caller is the application, and in the caller's fifo if its a worker.
*/
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task);

/** @} */

/**
   @name Resource-mapping Random Component API
   @{
*/

/**
   create a component that perform a random scheduling
*/
struct starpu_sched_component *starpu_sched_component_random_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;

/**
   return true iff \p component is a random component
*/
int starpu_sched_component_is_random(struct starpu_sched_component *);

/** @} */

/**
   @name Resource-mapping Eager Component API
   @{
*/

struct starpu_sched_component *starpu_sched_component_eager_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_eager(struct starpu_sched_component *);

/** @} */

/**
   @name Resource-mapping Eager Prio Component API
   @{
*/

struct starpu_sched_component *starpu_sched_component_eager_prio_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_eager_prio(struct starpu_sched_component *);

/** @} */

/**
   @name Resource-mapping Eager-Calibration Component API
   @{
*/

struct starpu_sched_component *starpu_sched_component_eager_calibration_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_eager_calibration(struct starpu_sched_component *);

/** @} */

/**
   @name Resource-mapping MCT Component API
   @{
*/

struct starpu_sched_component_mct_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
};

/**
   create a component with mct_data paremeters. the mct component doesnt
   do anything but pushing tasks on no_perf_model_component and
   calibrating_component
*/
struct starpu_sched_component *starpu_sched_component_mct_create(struct starpu_sched_tree *tree, struct starpu_sched_component_mct_data *mct_data) STARPU_ATTRIBUTE_MALLOC;

int starpu_sched_component_is_mct(struct starpu_sched_component *component);

/** @} */

/**
   @name Resource-mapping Heft Component API
   @{
*/

struct starpu_sched_component *starpu_sched_component_heft_create(struct starpu_sched_tree *tree, struct starpu_sched_component_mct_data *mct_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_heft(struct starpu_sched_component *component);

/** @} */

/**
   @name Resource-mapping Heteroprio Component API
   @{
*/

struct starpu_sched_component_heteroprio_data
{
	struct starpu_sched_component_mct_data *mct;
	unsigned batch;
};

struct starpu_sched_component * starpu_sched_component_heteroprio_create(struct starpu_sched_tree *tree, struct starpu_sched_component_heteroprio_data * params) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_heteroprio(struct starpu_sched_component *component);

/** @} */

/**
   @name Special-purpose Best_Implementation Component API
   @{
*/

/**
   Select the implementation that offer the shortest computation length for the first worker that can execute the task.
   Or an implementation that need to be calibrated.
   Also set starpu_task::predicted and starpu_task::predicted_transfer for memory component of the first suitable workerid.
   If starpu_sched_component::push method is called and starpu_sched_component::nchild > 1 the result is undefined.
*/
struct starpu_sched_component *starpu_sched_component_best_implementation_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;

/** @} */

/**
   @name Special-purpose Perfmodel_Select Component API
   @{
*/

struct starpu_sched_component_perfmodel_select_data
{
	struct starpu_sched_component *calibrator_component;
	struct starpu_sched_component *no_perfmodel_component;
	struct starpu_sched_component *perfmodel_component;
};
struct starpu_sched_component *starpu_sched_component_perfmodel_select_create(struct starpu_sched_tree *tree, struct starpu_sched_component_perfmodel_select_data *perfmodel_select_data) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_perfmodel_select(struct starpu_sched_component *component);

/** @} */

/**
   @name Staged pull Component API
   @{
*/

struct starpu_sched_component * starpu_sched_component_stage_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_stage(struct starpu_sched_component *component);

/** @} */

/**
   @name User-choice push Component API
   @{
*/

struct starpu_sched_component * starpu_sched_component_userchoice_create(struct starpu_sched_tree *tree, void *arg) STARPU_ATTRIBUTE_MALLOC;
int starpu_sched_component_is_userchoice(struct starpu_sched_component *component);

/** @} */

/**
   @name Recipe Component API
   @{
*/

/**
   parameters for starpu_sched_component_composed_component_create
*/
struct starpu_sched_component_composed_recipe;

/**
   return an empty recipe for a composed component, it should not be used without modification
*/
struct starpu_sched_component_composed_recipe *starpu_sched_component_composed_recipe_create(void) STARPU_ATTRIBUTE_MALLOC;

/**
   return a recipe to build a composed component with a \p create_component
*/
struct starpu_sched_component_composed_recipe *starpu_sched_component_composed_recipe_create_singleton(struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg) STARPU_ATTRIBUTE_MALLOC;

/**
   add \p create_component under all previous components in recipe
*/
void starpu_sched_component_composed_recipe_add(struct starpu_sched_component_composed_recipe *recipe, struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void *arg), void *arg);

/**
   destroy composed_sched_component, this should be done after starpu_sched_component_composed_component_create was called
*/
void starpu_sched_component_composed_recipe_destroy(struct starpu_sched_component_composed_recipe *);

/**
   create a component that behave as all component of recipe where linked. Except that you cant use starpu_sched_component_is_foo function
   if recipe contain a single create_foo arg_foo pair, create_foo(arg_foo) is returned instead of a composed component
*/
struct starpu_sched_component *starpu_sched_component_composed_component_create(struct starpu_sched_tree *tree, struct starpu_sched_component_composed_recipe *recipe) STARPU_ATTRIBUTE_MALLOC;

#ifdef STARPU_HAVE_HWLOC
/**
   Define how build a scheduler according to topology. Each level (except for hwloc_machine_composed_sched_component) can be <c>NULL</c>, then
   the level is just skipped. Bugs everywhere, do not rely on.
*/
struct starpu_sched_component_specs
{
	/**
	   the composed component to put on the top of the scheduler
	   this member must not be <c>NULL</c> as it is the root of the topology
	*/
	struct starpu_sched_component_composed_recipe *hwloc_machine_composed_sched_component;
	/**
	   the composed component to put for each memory component
	*/
	struct starpu_sched_component_composed_recipe *hwloc_component_composed_sched_component;
	/**
	   the composed component to put for each socket
	*/
	struct starpu_sched_component_composed_recipe *hwloc_socket_composed_sched_component;
	/**
	   the composed component to put for each cache
	*/
	struct starpu_sched_component_composed_recipe *hwloc_cache_composed_sched_component;

	/**
	   a function that return a starpu_sched_component_composed_recipe to put on top of a worker of type \p archtype.
	   <c>NULL</c> is a valid return value, then no component will be added on top
	*/
	struct starpu_sched_component_composed_recipe *(*worker_composed_sched_component)(enum starpu_worker_archtype archtype);
	/**
	   this flag is a dirty hack because of the poor expressivity of this interface. As example, if you want to build
	   a heft component with a fifo component per numa component, and you also have GPUs, if this flag is set, GPUs will share those fifos.
	   If this flag is not set, a new fifo will be built for each of them (if they have the same starpu_perf_arch and the same
	   numa component it will be shared. it indicates if heterogenous workers should be brothers or cousins, as example, if a gpu and a cpu should share or not there numa node
	*/
	int mix_heterogeneous_workers;
};


/**
   build a scheduler for \p sched_ctx_id according to \p s and the hwloc topology of the machine.
*/
struct starpu_sched_tree *starpu_sched_component_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_component_specs s);
#endif /* STARPU_HAVE_HWLOC */

/**
   @name Basic API
   @{
*/

#define STARPU_SCHED_SIMPLE_DECIDE_MASK		(3<<0)

/**
   Request to create downstream queues per worker, i.e. the scheduling decision-making component will choose exactly which workers tasks should got to.
*/
#define STARPU_SCHED_SIMPLE_DECIDE_WORKERS	(1<<0)

/**
   Request to create downstream queues per memory nodes, i.e. the scheduling decision-making component will choose which memory node tasks will go to.
*/
#define STARPU_SCHED_SIMPLE_DECIDE_MEMNODES	(2<<0)

/**
   Request to create downstream queues per computation arch, i.e. the scheduling decision-making component will choose whether tasks go to CPUs, or CUDA, or OpenCL, etc.
*/
#define STARPU_SCHED_SIMPLE_DECIDE_ARCHS	(3<<0)

/**
   Request to create the scheduling decision-making component even if there is only one available choice. This is useful for instance when the decision-making component will store tasks itself (and not use STARPU_SCHED_SIMPLE_FIFO_ABOVE) to decide in which order tasks should be passed below.
*/
#define STARPU_SCHED_SIMPLE_DECIDE_ALWAYS	(1<<3)

/**
   Request to add a perfmodel selector above the scheduling decision-making component. That way, only tasks with a calibrated performance model will be given to the component, other tasks will go to an eager branch that will distributed tasks so that their performance models will get calibrated.
   In other words, this is needed when using a component which needs performance models for tasks.
*/
#define STARPU_SCHED_SIMPLE_PERFMODEL		(1<<4)

/**
   Request that a component be added just above workers, that chooses the best task implementation.
*/
#define STARPU_SCHED_SIMPLE_IMPL		(1<<5)

/**
   Request to create a fifo above the scheduling decision-making component, otherwise tasks will be pushed directly to the component.

   This is useful to store tasks if there is a fifo below which limits the number of tasks to be scheduld in advance. The scheduling decision-making component can also store tasks itself, in which case this flag is not useful.
*/
#define STARPU_SCHED_SIMPLE_FIFO_ABOVE		(1<<6)

/**
   Request that the fifo above be sorted by priorities
*/
#define STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO	(1<<7)

/**
   Request to create fifos below the scheduling decision-making component, otherwise tasks will be pulled directly from workers.

   This is useful to be able to schedule a (tunable) small number of tasks in advance only.
*/
#define STARPU_SCHED_SIMPLE_FIFOS_BELOW		(1<<8)

/**
   Request that the fifos below be sorted by priorities
*/
#define STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO	(1<<9)

/**
   Request that the fifos below be pulled rather ready tasks
*/
#define STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY	(1<<10)

/**
   Request that work between workers using the same fifo below be distributed using a work stealing component.
*/
#define STARPU_SCHED_SIMPLE_WS_BELOW		(1<<11)

/**
   Request to not only choose between simple workers, but also choose between combined workers.
*/
#define STARPU_SCHED_SIMPLE_COMBINED_WORKERS	(1<<12)

/**
   Create a simple modular scheduler tree around a scheduling decision-making
   component \p component. The details of what should be built around \p component
   is described by \p flags. The different STARPU_SCHED_SIMPL_DECIDE_* flags are
   mutually exclusive. \p data is passed to the \p create_decision_component
   function when creating the decision component.
*/
void starpu_sched_component_initialize_simple_scheduler(starpu_sched_component_create_t create_decision_component, void *data, unsigned flags, unsigned sched_ctx_id);

/**
   Create a simple modular scheduler tree around several scheduling decision-making
   components. The parameters are similar to
   starpu_sched_component_initialize_simple_scheduler, but per scheduling decision, for instance:

   starpu_sched_component_initialize_simple_schedulers(sched_ctx_id, 2,
     create1, data1, flags1,
     create2, data2, flags2);

   The different flags parameters must be coherent: same decision flags. They
   must not include the perfmodel flag (not supported yet).
*/
void starpu_sched_component_initialize_simple_schedulers(unsigned sched_ctx_id, unsigned ndecisions, ...);

/** @} */

#define STARPU_COMPONENT_MUTEX_LOCK(m) \
do \
{ \
	const int _relaxed_state = starpu_worker_get_relax_state(); \
	if (!_relaxed_state) \
		starpu_worker_relax_on(); \
	STARPU_PTHREAD_MUTEX_LOCK((m)); \
	if (!_relaxed_state) \
		starpu_worker_relax_off(); \
} \
while(0)

#define STARPU_COMPONENT_MUTEX_TRYLOCK(m) STARPU_PTHREAD_MUTEX_TRYLOCK((m))

#define STARPU_COMPONENT_MUTEX_UNLOCK(m) STARPU_PTHREAD_MUTEX_UNLOCK((m))

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_COMPONENT_H__ */
