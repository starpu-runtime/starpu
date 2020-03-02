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

#include <starpu_sched_component.h>
#include <common/list.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#include "sched_component.h"



/* The scheduler is built by a recursive function called on the hwloc topology with a starpu_sched_specs structure,
 * each call return a set of starpu_sched_component, not a single one, because you may have a topology like that :
 * MACHINE -- MEMORY NODE -- SOCKET
 *                        \- SOCKET
 * and you have defined a component for MACHINE, and a component for SOCKET, but not for MEMORY NODE then the recursive call
 * on MEMORY NODE will return 2 starpu_sched_component for those 2 sockets
 *
 *
 */

struct sched_component_list
{
	struct starpu_sched_component ** arr;
	unsigned size;
};

static void init_list(struct sched_component_list * list)
{
	memset(list,0,sizeof(*list));
}
static void destroy_list(struct sched_component_list * list)
{
	free(list->arr);
}
static void add_component(struct sched_component_list *list, struct starpu_sched_component * component)
{
	_STARPU_REALLOC(list->arr, sizeof(*list->arr) * (list->size + 1));
	list->arr[list->size] = component;
	list->size++;
}
/* this is the function that actualy built the scheduler, but without workers */
static struct sched_component_list helper_make_scheduler(struct starpu_sched_tree *tree, hwloc_obj_t obj, struct starpu_sched_component_specs specs, unsigned sched_ctx_id)
{
	STARPU_ASSERT(obj);

	struct starpu_sched_component * component = NULL;

	/*set components for this obj */
#define CASE(ENUM,spec_member)						\
		case ENUM:						\
			if(specs.spec_member)				\
				component = starpu_sched_component_composed_component_create(tree, specs.spec_member); \
			break
	switch(obj->type)
	{
		CASE(HWLOC_OBJ_MACHINE,hwloc_machine_composed_sched_component);
		CASE(HWLOC_OBJ_GROUP,hwloc_component_composed_sched_component);
		CASE(HWLOC_OBJ_NUMANODE,hwloc_component_composed_sched_component);
		CASE(HWLOC_OBJ_SOCKET,hwloc_socket_composed_sched_component);
#ifdef HWLOC_OBJ_CACHE
		CASE(HWLOC_OBJ_CACHE,hwloc_cache_composed_sched_component);
#endif
#ifdef HWLOC_OBJ_L1CACHE
		CASE(HWLOC_OBJ_L1CACHE,hwloc_cache_composed_sched_component);
		CASE(HWLOC_OBJ_L2CACHE,hwloc_cache_composed_sched_component);
		CASE(HWLOC_OBJ_L3CACHE,hwloc_cache_composed_sched_component);
		CASE(HWLOC_OBJ_L4CACHE,hwloc_cache_composed_sched_component);
		CASE(HWLOC_OBJ_L5CACHE,hwloc_cache_composed_sched_component);
#endif
	default:
		break;
	}

	struct sched_component_list l;
	init_list(&l);
	unsigned i;
	/* collect childs component's */
	for(i = 0; i < obj->arity; i++)
	{
		struct sched_component_list lc = helper_make_scheduler(tree, obj->children[i],specs, sched_ctx_id);
		unsigned j;
		for(j = 0; j < lc.size; j++)
			add_component(&l, lc.arr[j]);
		destroy_list(&lc);
	}
	if(!component)
		return l;
	for(i = 0; i < l.size; i++)
		starpu_sched_component_connect(component, l.arr[i]);
	destroy_list(&l);
	init_list(&l);
	component->obj = obj;
	add_component(&l, component);
	return l;
}
/* return the firt component in prefix order such as component->obj == obj, or NULL */
struct starpu_sched_component * _find_sched_component_with_obj(struct starpu_sched_component * component, hwloc_obj_t obj)
{
	if(component == NULL)
		return NULL;
	if(component->obj == obj)
		return component;
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * tmp = _find_sched_component_with_obj(component->children[i], obj);
		if(tmp)
			return tmp;
	}
	return NULL;
}

/* return true if all workers in the tree have the same perf_arch as w_ref,
 * if there is no worker it return true
 */
static int is_same_kind_of_all(struct starpu_sched_component * root, struct _starpu_worker * w_ref)
{
	if(starpu_sched_component_is_worker(root))
	{
		struct _starpu_worker * w = root->data;
		STARPU_ASSERT(w->perf_arch.ndevices == 1);
		return w->perf_arch.devices[0].type == w_ref->perf_arch.devices[0].type;
	}

	unsigned i;
	for(i = 0;i < root->nchildren; i++)
		if(!is_same_kind_of_all(root->children[i], w_ref))
			return 0;
	return 1;
}
/* buggy function
 * return the starpu_sched_component linked to the supposed memory component of worker_component
 */
static struct starpu_sched_component * find_mem_component(struct starpu_sched_component * root, struct starpu_sched_component * worker_component)
{
	struct starpu_sched_component * component = worker_component;
	while(component->obj->type != HWLOC_OBJ_NUMANODE
	      && component->obj->type != HWLOC_OBJ_GROUP
	      && component->obj->type != HWLOC_OBJ_MACHINE)
	{
		hwloc_obj_t tmp = component->obj;
		do
		{
			component = _find_sched_component_with_obj(root,tmp);
			tmp = tmp->parent;
		}
		while(!component);

	}
	return component;
}

static struct starpu_sched_component * where_should_we_plug_this(struct starpu_sched_component *root, struct starpu_sched_component * worker_component, struct starpu_sched_component_specs specs, unsigned sched_ctx_id)
{
	struct starpu_sched_component * mem = find_mem_component(root ,worker_component);
	if(specs.mix_heterogeneous_workers || mem->parents[sched_ctx_id] == NULL)
		return mem;
	hwloc_obj_t obj = mem->obj;
	struct starpu_sched_component * parent = mem->parents[sched_ctx_id];
	unsigned i;
	for(i = 0; i < parent->nchildren; i++)
	{
		if(parent->children[i]->obj == obj
		   && is_same_kind_of_all(parent->children[i], worker_component->data))
			return parent->children[i];
	}
	if(obj->type == HWLOC_OBJ_NUMANODE || obj->type == HWLOC_OBJ_GROUP)
	{
		struct starpu_sched_component * component = starpu_sched_component_composed_component_create(root->tree, specs.hwloc_component_composed_sched_component);
		component->obj = obj;
		starpu_sched_component_connect(parent, component);
		return component;
	}
	return parent;
}

static void set_worker_leaf(struct starpu_sched_component * root, struct starpu_sched_component * worker_component, unsigned sched_ctx_id,
			    struct starpu_sched_component_specs specs)
{
	struct _starpu_worker * worker = worker_component->data;
	struct starpu_sched_component * component = where_should_we_plug_this(root,worker_component,specs, sched_ctx_id);
	struct starpu_sched_component_composed_recipe * recipe = specs.worker_composed_sched_component ?
		specs.worker_composed_sched_component(worker->arch):NULL;
	STARPU_ASSERT(component);
	if(recipe)
	{
		struct starpu_sched_component * tmp = starpu_sched_component_composed_component_create(root->tree, recipe);
#ifdef STARPU_DEVEL
#warning FIXME component->obj is set to worker_component->obj even for accelerators workers
#endif
		tmp->obj = worker_component->obj;
		starpu_sched_component_connect(component, tmp);
		component = tmp;
	}
	starpu_sched_component_composed_recipe_destroy(recipe);
	starpu_sched_component_connect(component, worker_component);
}

#ifdef STARPU_DEVEL
static const char * name_hwloc_component(struct starpu_sched_component * component)
{
	return hwloc_obj_type_string(component->obj->type);
}
static const char * name_sched_component(struct starpu_sched_component * component)
{
	if(starpu_sched_component_is_fifo(component))
		return "fifo component";
	if(starpu_sched_component_is_heft(component))
		return "heft component";
	if(starpu_sched_component_is_random(component))
		return "random component";
	if(starpu_sched_component_is_worker(component))
	{
		struct _starpu_worker * w = _starpu_sched_component_worker_get_worker(component);
#define SIZE 256
		static char output[SIZE];
		snprintf(output, SIZE,"component worker %d %s",w->workerid,w->name);
		return output;
	}
	if(starpu_sched_component_is_work_stealing(component))
		return "work stealing component";

	return "unknown";
}
static void helper_display_scheduler(FILE* out, unsigned depth, struct starpu_sched_component * component)
{
	if(!component)
		return;
	fprintf(out,"%*s-> %s : %s\n", depth * 2 , "", name_sched_component(component), name_hwloc_component(component));
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
		helper_display_scheduler(out, depth + 1, component->children[i]);
}
#endif //STARPU_DEVEL
struct starpu_sched_tree * starpu_sched_component_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_component_specs specs)
{
	struct starpu_sched_tree * tree = starpu_sched_tree_create(sched_ctx_id);

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	hwloc_topology_t topology = config->topology.hwtopology;

	struct sched_component_list list = helper_make_scheduler(tree, hwloc_get_root_obj(topology), specs, sched_ctx_id);
	STARPU_ASSERT(list.size == 1);

	tree->root = list.arr[0];
	destroy_list(&list);

	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(i);
		struct starpu_sched_component *worker_component = starpu_sched_component_worker_new(sched_ctx_id, i);
		STARPU_ASSERT(worker);
		set_worker_leaf(tree->root,worker_component, sched_ctx_id, specs);
	}


	starpu_sched_tree_update_workers(tree);
#ifdef STARPU_DEVEL
	_STARPU_MSG("scheduler created :\n");
	helper_display_scheduler(stderr, 0, tree->root);
#endif

	return tree;

}
