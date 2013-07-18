#include <starpu_sched_node.h>
#include <common/list.h>


/* a composed node is parametred by a list of pair
 * (create_node_function(arg), arg)
 */
LIST_TYPE(fun_create_node,
	  struct starpu_sched_node *(*create_node)(void * arg);
	  void * arg;
);


struct _starpu_composed_sched_node_recipe
{
	struct fun_create_node_list * list;
};


struct _starpu_composed_sched_node_recipe * starpu_sched_node_create_recipe(void)
{
	struct _starpu_composed_sched_node_recipe * recipe = malloc(sizeof(*recipe));
	recipe->list = fun_create_node_list_new();
	return recipe;
}

void starpu_sched_recipe_add_node(struct _starpu_composed_sched_node_recipe * recipe,
				  struct starpu_sched_node *(*create_node)(void * arg),
				  void * arg)
{
	struct fun_create_node * e = fun_create_node_new();
	e->create_node = create_node;
	e->arg = arg;
	fun_create_node_list_push_back(recipe->list, e);
}
struct _starpu_composed_sched_node_recipe * starpu_sched_node_create_recipe_singleton(struct starpu_sched_node *(*create_node)(void * arg),
										      void * arg)
{
	struct _starpu_composed_sched_node_recipe * r = starpu_sched_node_create_recipe();
	starpu_sched_recipe_add_node(r, create_node, arg);
	return r;
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
	struct starpu_sched_node *top,*bottom;
};

/* this function actualy build the composed node data by changing the list of
 * (node_create_fun, arg_create_fun) into a tree where all nodes have 1 childs
 */
struct composed_node create_composed_node(struct _starpu_composed_sched_node_recipe * recipe
#ifdef STARPU_HAVE_HWLOC
					  ,hwloc_obj_t obj
#endif
)
{
	struct composed_node c;
	STARPU_ASSERT(recipe);

	struct fun_create_node_list * list = recipe->list;
	struct fun_create_node * i = fun_create_node_list_begin(list);
	STARPU_ASSERT(i);
	STARPU_ASSERT(i->create_node(i->arg));
	c.top = c.bottom = i->create_node(i->arg);
#ifdef STARPU_HAVE_HWLOC
	c.top->obj = obj;
#endif
	for(i  = fun_create_node_list_next(i);
	    i != fun_create_node_list_end(list);
	    i  = fun_create_node_list_next(i))
	{
		STARPU_ASSERT(i->create_node(i->arg));
		struct starpu_sched_node * node = i->create_node(i->arg);
#ifdef STARPU_HAVE_HWLOC
		node->obj = obj;
#endif
		starpu_sched_node_add_child(c.bottom, node);

		/* we want to be able to traverse scheduler bottom up for all sched ctxs
		 * when a worker call pop()
		 */
		unsigned j;
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			starpu_sched_node_set_father(node, c.bottom, j);
		c.bottom = node;
	}
	STARPU_ASSERT(!starpu_sched_node_is_worker(c.bottom));
	return c;
}


static int composed_node_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	struct composed_node *c = node->data;
	return c->top->push_task(c->top,task);
}
struct starpu_task * composed_node_pop_task(struct starpu_sched_node *node, unsigned sched_ctx_id)
{
	struct composed_node *c = node->data;
	struct starpu_task * t = c->bottom->pop_task(c->bottom, sched_ctx_id);
	if(t)
		return t;

	if(node->fathers[sched_ctx_id])
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id], sched_ctx_id);
	return NULL;
}

double composed_node_estimated_load(struct starpu_sched_node * node)
{
	struct composed_node * c = node->data;
	return c->top->estimated_load(c->top);
}

static void composed_node_add_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	struct composed_node * c = node->data;
	starpu_sched_node_add_child(node, child);
	c->bottom->add_child(c->bottom, child);
}
static void composed_node_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	struct composed_node * c = node->data;
	c->bottom->remove_child(c->bottom, child);
}

static void composed_node_notify_change_workers(struct starpu_sched_node * node)
{
	struct composed_node * c = node->data;
	struct starpu_bitmap * workers = node->workers;
	struct starpu_bitmap * workers_in_ctx = node->workers_in_ctx;
	struct starpu_sched_node * n;
	for(n = c->top; ;n = n->childs[0])
	{
		starpu_bitmap_unset_all(n->workers);
		starpu_bitmap_or(n->workers, workers);

		starpu_bitmap_unset_all(n->workers_in_ctx);
		starpu_bitmap_or(n->workers_in_ctx, workers_in_ctx);

		n->properties = node->properties;
		if(n == c->bottom)
			break;
	}
}

void composed_node_deinit_data(struct starpu_sched_node * _node)
{
	struct composed_node *c = _node->data;
	c->bottom->childs = NULL;
	c->bottom->nchilds = 0;
	struct starpu_sched_node * node = c->top;
	struct starpu_sched_node * next = NULL;
	do
	{
		node->workers = NULL;
		next = node->childs ? node->childs[0] : NULL;
		starpu_sched_node_destroy(node);
	}while(next);
	free(c);
	_node->data = NULL;
}

struct starpu_sched_node * starpu_sched_node_composed_node_create(struct _starpu_composed_sched_node_recipe * recipe)
{
	STARPU_ASSERT(!fun_create_node_list_empty(recipe->list));
	struct fun_create_node_list * l = recipe->list;
	if(l->_head == l->_tail)
		return l->_head->create_node(l->_head->arg);
	struct starpu_sched_node * node = starpu_sched_node_create();

	struct composed_node * c = malloc(sizeof(struct composed_node));
	*c = create_composed_node(recipe
#ifdef STARPU_HAVE_HWLOC
				  ,node->obj
#endif
);
	c->bottom->nchilds = node->nchilds;
	c->bottom->childs = node->childs;

	node->data = c;
	node->push_task = composed_node_push_task;
	node->pop_task = composed_node_pop_task;
	node->estimated_load = composed_node_estimated_load;
	node->add_child = composed_node_add_child;
	node->remove_child = composed_node_remove_child;
	node->notify_change_workers = composed_node_notify_change_workers;
	return node;
}
