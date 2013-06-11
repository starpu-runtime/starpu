#include <starpu_config.h>
#ifdef STARPU_HAVE_HWLOC
#ifndef __SCHEDULER_MAKER_H__
#define __SCHEDULER_MAKER_H__
#include "node_sched.h"
#include "node_composed.h"
#include <common/list.h>



//null pointer mean to ignore a level L of hierarchy, then nodes of levels > L become childs of level L - 1
struct _starpu_sched_specs
{
	//hw_loc_machine_composed_sched_node must be set as its the root of the topology
	struct _starpu_composed_sched_node_recipe * hwloc_machine_composed_sched_node;
	struct _starpu_composed_sched_node_recipe * hwloc_node_composed_sched_node;
	struct _starpu_composed_sched_node_recipe * hwloc_socket_composed_sched_node;
	struct _starpu_composed_sched_node_recipe * hwloc_cache_composed_sched_node;

	/* this member should return a new allocated _starpu_composed_sched_node_recipe_t or NULL
	 * the _starpu_composed_sched_node_recipe_t must not include the worker node
	 */
	struct _starpu_composed_sched_node_recipe * (*worker_composed_sched_node)(enum starpu_worker_archtype);

	/* this flag indicate if heterogenous workers should be brothers or cousins,
	 * as exemple, if a gpu and a cpu should share or not there numa node
	 */
	int mix_heterogeneous_workers;
};

struct _starpu_sched_tree * _starpu_make_scheduler(unsigned sched_ctx_id, struct _starpu_sched_specs);

#endif//#ifndef __SCHEDULER_MAKER_H__
#endif//#ifdef STARPU_HAVE_HWLOC
