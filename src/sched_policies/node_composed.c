#include "node_sched.h"
#include <common/list.h>
LIST_TYPE(fun_create_node,
	  struct _starpu_sched_node *(*create_node)(void * arg);
	  void * arg;
);


struct _starpu_composed_sched_node_recipe
{
	struct fun_create_node_list * list;
};


struct _starpu_composed_sched_node_recipe * _starpu_sched_node_create_recipe(void)
{
	struct _starpu_composed_sched_node_recipe * recipe = malloc(sizeof(*recipe));
	recipe->list = fun_create_node_list_new();
	return recipe;
}

void _starpu_sched_recipe_add_node(struct _starpu_composed_sched_node_recipe * recipe, struct _starpu_sched_node *(*create_node)(void * arg), void * arg)
{
	struct fun_create_node * e = fun_create_node_new();
	e->create_node = create_node;
	e->arg = arg;
	fun_create_node_list_push_back(recipe->list, e);
}

void _starpu_destroy_composed_sched_node_recipe(struct _starpu_composed_sched_node_recipe * recipe)
{
	if(!recipe)
		return;
	while(!fun_create_node_list_empty(recipe->list))
		fun_create_node_delete(fun_create_node_list_pop_back(recipe->list));
	fun_create_node_list_delete(recipe->list);
	free(recipe);
}

struct composed_node
{
	struct _starpu_sched_node *top,*bottom;
};
struct composed_node create_composed_node(struct _starpu_sched_node * sched_ctx_ids_father[STARPU_NMAX_SCHED_CTXS], struct _starpu_bitmap * workers, struct _starpu_composed_sched_node_recipe * recipe
#ifdef STARPU_HAVE_HWLOC
					    ,hwloc_obj_t obj
#endif
)
{
	struct composed_node c;
	if(!recipe)
	{
		c.top = c.bottom = NULL;
		return c;
	}
	struct fun_create_node_list * list = recipe->list;
	struct fun_create_node * i = fun_create_node_list_begin(list);
	STARPU_ASSERT(i);
	STARPU_ASSERT(i->create_node(i->arg));
	c.top = c.bottom = i->create_node(i->arg);
#ifdef STARPU_HAVE_HWLOC
	c.top->obj = obj;
#endif
	memcpy(c.top->fathers, sched_ctx_ids_father, sizeof(sched_ctx_ids_father));

	for(i  = fun_create_node_list_next(i);
	    i != fun_create_node_list_end(list);
	    i  = fun_create_node_list_next(i))
	{
		STARPU_ASSERT(i->create_node(i->arg));
		struct _starpu_sched_node * node = i->create_node(i->arg);
#ifdef STARPU_HAVE_HWLOC
		node->obj = obj;
#endif
		_starpu_sched_node_add_child(c.bottom, node);
//we want to be able to to traverse scheduler bottom up for all sched ctxs
		int j;
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(sched_ctx_ids_father[j])
				_starpu_sched_node_set_father(node, c.bottom,(unsigned)j);
		node->workers = workers;
		c.bottom = node;
	}
	STARPU_ASSERT(!_starpu_sched_node_is_worker(c.bottom));
	return c;
}
		

static int composed_node_push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct composed_node *c = node->data;
	return c->top->push_task(c->top,task);
}
struct starpu_task * composed_node_pop_task(struct _starpu_sched_node *node,
					    unsigned sched_ctx_id)
{
	struct composed_node *c = node->data;
	return c->bottom->pop_task(c->bottom, sched_ctx_id);
}
void composed_node_available(struct _starpu_sched_node *node)
{
	struct composed_node * c = node->data;
	c->top->available(c->top);
}
	
double composed_node_estimated_load(struct _starpu_sched_node * node)
{
	struct composed_node * c = node->data;
	return c->top->estimated_load(c->top);
}

struct _starpu_task_execute_preds composed_node_estimated_execute_preds(struct _starpu_sched_node * node,
									struct starpu_task * task)
{
	struct composed_node * c = node->data;
	return c->top->estimated_execute_preds(c->top,task);
}

static void invalid_second_init_data(struct _starpu_sched_node * node STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ABORT();
}

void composed_node_init_data(struct _starpu_sched_node *node)
{
	struct _starpu_composed_sched_node_recipe * recipe = node->data;
	struct composed_node * c = malloc(sizeof(struct composed_node));
	*c = create_composed_node(node->fathers, node->workers, recipe
#ifdef STARPU_HAVE_HWLOC
				   ,node->obj
#endif 
);
	c->bottom->nchilds = node->nchilds;
	c->bottom->childs = node->childs;

	node->init_data = invalid_second_init_data;
}

void composed_node_deinit_data(struct _starpu_sched_node * _node)
{
	struct composed_node *c = _node->data;
	c->bottom->childs = NULL;
	c->bottom->nchilds = 0;
	struct _starpu_sched_node * node = c->top;
	struct _starpu_sched_node * next = NULL;
	do
	{
		node->workers = NULL;
		node->deinit_data(node);
		next = node->childs ? node->childs[0] : NULL;
		_starpu_sched_node_destroy(node);
	}while(next);
	free(c);
	_node->data = NULL;
}

struct _starpu_sched_node * _starpu_sched_node_composed_node_create(struct _starpu_composed_sched_node_recipe * recipe)
{
	STARPU_ASSERT(!fun_create_node_list_empty(recipe->list));
	struct fun_create_node_list * l = recipe->list;
	if(l->_head == l->_tail)
		return l->_head->create_node(l->_head->arg);
	struct _starpu_sched_node * node = _starpu_sched_node_create();

	node->data = recipe;
	node->push_task = composed_node_push_task;
	node->pop_task = composed_node_pop_task;
	node->available = composed_node_available;
	node->estimated_load = composed_node_estimated_load;
	node->estimated_execute_preds = composed_node_estimated_execute_preds;
	node->init_data = composed_node_init_data;
	node->deinit_data = composed_node_deinit_data;

	return node;
}
