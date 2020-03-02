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


/* a composed component is parametred by a list of pair
 * (create_component_function(arg), arg)
 */
LIST_TYPE(fun_create_component,
	  struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void * arg);
	  void * arg;
);

struct starpu_sched_component_composed_recipe
{
	struct fun_create_component_list list;
};

struct starpu_sched_component_composed_recipe * starpu_sched_component_composed_recipe_create(void)
{
	struct starpu_sched_component_composed_recipe *recipe;
	_STARPU_MALLOC(recipe, sizeof(*recipe));
	fun_create_component_list_init(&recipe->list);
	return recipe;
}

void starpu_sched_component_composed_recipe_add(struct starpu_sched_component_composed_recipe * recipe,
				       struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void * arg),
				       void * arg)
{
	struct fun_create_component * e = fun_create_component_new();
	e->create_component = create_component;
	e->arg = arg;
	fun_create_component_list_push_back(&recipe->list, e);
}

struct starpu_sched_component_composed_recipe *starpu_sched_component_composed_recipe_create_singleton(struct starpu_sched_component *(*create_component)(struct starpu_sched_tree *tree, void * arg),
													void * arg)
{
	struct starpu_sched_component_composed_recipe * r = starpu_sched_component_composed_recipe_create();
	starpu_sched_component_composed_recipe_add(r, create_component, arg);
	return r;
}

void starpu_sched_component_composed_recipe_destroy(struct starpu_sched_component_composed_recipe * recipe)
{
	if(!recipe)
		return;
	while(!fun_create_component_list_empty(&recipe->list))
		fun_create_component_delete(fun_create_component_list_pop_back(&recipe->list));
	free(recipe);
}

struct composed_component
{
	struct starpu_sched_component *top,*bottom;
};

/* this function actualy build the composed component data by changing the list of
 * (component_create_fun, arg_create_fun) into a tree where all components have 1 childs
 */
struct composed_component create_composed_component(struct starpu_sched_tree *tree, struct starpu_sched_component_composed_recipe * recipe
#ifdef STARPU_HAVE_HWLOC
						    ,hwloc_obj_t obj
#endif
						    )
{
	struct composed_component c;
	STARPU_ASSERT(recipe);

	struct fun_create_component_list * list = &recipe->list;
	struct fun_create_component * i = fun_create_component_list_begin(list);
	STARPU_ASSERT(i);
	STARPU_ASSERT(i->create_component);
	c.top = c.bottom = i->create_component(tree, i->arg);
#ifdef STARPU_HAVE_HWLOC
	c.top->obj = obj;
#endif
	for(i  = fun_create_component_list_next(i);
	    i != fun_create_component_list_end(list);
	    i  = fun_create_component_list_next(i))
	{
		STARPU_ASSERT(i->create_component);
		struct starpu_sched_component * component = i->create_component(tree, i->arg);
#ifdef STARPU_HAVE_HWLOC
		component->obj = obj;
#endif
		c.bottom->add_child(c.bottom, component);

		/* we want to be able to traverse scheduler bottom up for all sched ctxs
		 * when a worker call pop()
		 */
		unsigned j;
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			component->add_parent(component, c.bottom);
		c.bottom = component;
	}
	STARPU_ASSERT(!starpu_sched_component_is_worker(c.bottom));
	return c;
}

static int composed_component_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	struct composed_component *c = component->data;
	return starpu_sched_component_push_task(component,c->top,task);
}

struct starpu_task * composed_component_pull_task(struct starpu_sched_component *component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	struct composed_component *c = component->data;
	struct starpu_task *task;

	task = starpu_sched_component_pull_task(c->bottom,component);
	if(task)
		return task;

	unsigned i;
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			task = starpu_sched_component_pull_task(component->parents[i],component);
			if(task)
				break;
		}
	}
	return task;
}

double composed_component_estimated_load(struct starpu_sched_component * component)
{
	struct composed_component * c = component->data;
	return c->top->estimated_load(c->top);
}

static void composed_component_add_child(struct starpu_sched_component * component, struct starpu_sched_component * child)
{
	struct composed_component * c = component->data;
	component->add_child(component, child);
	c->bottom->add_child(c->bottom, child);
}

static void composed_component_remove_child(struct starpu_sched_component * component, struct starpu_sched_component * child)
{
	struct composed_component * c = component->data;
	component->remove_child(component, child);
	c->bottom->remove_child(c->bottom, child);
}

static void composed_component_notify_change_workers(struct starpu_sched_component * component)
{
	struct composed_component * c = component->data;
	struct starpu_bitmap * workers = component->workers;
	struct starpu_bitmap * workers_in_ctx = component->workers_in_ctx;
	struct starpu_sched_component * n;
	for(n = c->top; ;n = n->children[0])
	{
		starpu_bitmap_unset_all(n->workers);
		starpu_bitmap_or(n->workers, workers);

		starpu_bitmap_unset_all(n->workers_in_ctx);
		starpu_bitmap_or(n->workers_in_ctx, workers_in_ctx);

		n->properties = component->properties;
		if(n == c->bottom)
			break;
	}
}

void composed_component_deinit_data(struct starpu_sched_component * _component)
{
	struct composed_component *c = _component->data;
	c->bottom->children = NULL;
	c->bottom->nchildren = 0;
	struct starpu_sched_component * component;
	struct starpu_sched_component * next = c->top;
	do
	{
		component = next;
		component->workers = NULL;
		next = component->children ? component->children[0] : NULL;
		starpu_sched_component_destroy(component);
	}
	while(next);
	free(c);
	_component->data = NULL;
}

struct starpu_sched_component * starpu_sched_component_composed_component_create(struct starpu_sched_tree *tree,
										 struct starpu_sched_component_composed_recipe * recipe)
{
	STARPU_ASSERT(!fun_create_component_list_empty(&recipe->list));
	struct fun_create_component_list * l = &recipe->list;
	if(l->_head == l->_tail)
		return l->_head->create_component(tree, l->_head->arg);

	struct starpu_sched_component * component = starpu_sched_component_create(tree, "composed");
	struct composed_component *c;
	_STARPU_MALLOC(c, sizeof(struct composed_component));
	*c = create_composed_component(tree, recipe
#ifdef STARPU_HAVE_HWLOC
				       ,component->obj
#endif
);
	c->bottom->nchildren = component->nchildren;
	c->bottom->children = component->children;
	c->bottom->nparents = component->nparents;
	c->bottom->parents = component->parents;

	component->data = c;
	component->push_task = composed_component_push_task;
	component->pull_task = composed_component_pull_task;
	component->estimated_load = composed_component_estimated_load;
	component->estimated_end = starpu_sched_component_estimated_end_min;
	component->add_child = composed_component_add_child;
	component->remove_child = composed_component_remove_child;
	component->notify_change_workers = composed_component_notify_change_workers;
	return component;
}
