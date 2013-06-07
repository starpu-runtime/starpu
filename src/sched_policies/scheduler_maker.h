#include <starpu_config.h>
#ifdef STARPU_HAVE_HWLOC
#ifndef __SCHEDULER_MAKER_H__
#define __SCHEDULER_MAKER_H__
#include "node_sched.h"
#include <common/list.h>



typedef struct _starpu_composed_sched_node_recipe * _starpu_composed_sched_node_recipe_t;



//variadic args are a null terminated sequence of function struct _starpu_sched_node *(*)(void)
_starpu_composed_sched_node_recipe_t _starpu_create_composed_sched_node_recipe(struct _starpu_sched_node * (*create_sched_node_top)(void), ...);
void _starpu_destroy_composed_sched_node_recipe(_starpu_composed_sched_node_recipe_t);


//null pointer mean to ignore a level L of hierarchy, then nodes of levels > L become childs of level L - 1
struct _starpu_sched_specs
{
	_starpu_composed_sched_node_recipe_t hwloc_machine_composed_sched_node;
	_starpu_composed_sched_node_recipe_t hwloc_node_composed_sched_node;
	_starpu_composed_sched_node_recipe_t hwloc_socket_composed_sched_node;
	_starpu_composed_sched_node_recipe_t hwloc_cache_composed_sched_node;


	/* this member should return a new allocated _starpu_composed_sched_node_recipe_t or NULL
	 * the _starpu_composed_sched_node_recipe_t must not include the worker node
	 */
	_starpu_composed_sched_node_recipe_t (*worker_composed_sched_node)(enum starpu_worker_archtype);
};

struct _starpu_sched_tree * _starpu_make_scheduler(unsigned sched_ctx_id, struct _starpu_sched_specs);

#endif//#ifndef __SCHEDULER_MAKER_H__
#endif//#ifdef STARPU_HAVE_HWLOC
