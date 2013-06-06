#include "scheduler_maker.h"
#include "node_sched.h"
#include <common/list.h>
#include <stdarg.h>
#include <core/workers.h>
LIST_TYPE(fun_create_node,
	  struct _starpu_sched_node *(*create_node)(void);
);


struct _starpu_composed_sched_node_recipe
{
	struct fun_create_node_list * list;
};


//variadic args are a null terminated sequence of function struct _starpu_sched_node *(*)(void)
_starpu_composed_sched_node_recipe_t _starpu_create_composed_sched_node_recipe(struct _starpu_sched_node * (*create_sched_node_top)(void), ...){
	_starpu_composed_sched_node_recipe_t recipe = malloc(sizeof(*recipe));
	recipe->list = fun_create_node_list_new();
	
	va_list ap;
	va_start(ap, create_sched_node_top);
	struct _starpu_sched_node *(*create_node)(void);
	for(create_node = va_arg(ap, struct _starpu_sched_node * (*)(void));
	    create_node != NULL;
	    create_node = va_arg(ap, struct _starpu_sched_node * (*)(void)))
	{
		struct fun_create_node * e = fun_create_node_new();
		e->create_node = create_node;
		fun_create_node_list_push_back(recipe->list, e);
	}
	va_end(ap);
	
	return recipe;
}
	

void _starpu_destroy_composed_sched_node_recipe(_starpu_composed_sched_node_recipe_t recipe)
{
	while(!fun_create_node_list_empty(recipe->list))
		fun_create_node_delete(fun_create_node_list_pop_back(recipe->list));
	fun_create_node_list_delete(recipe->list);
	free(recipe);
}




struct composed_sched
{
	struct _starpu_sched_node *top,*bottom;
};
struct composed_sched create_composed_sched(unsigned sched_ctx_id, hwloc_obj_t obj, _starpu_composed_sched_node_recipe_t recipe)
{
	struct fun_create_node_list * list = recipe->list;

	struct composed_sched c;
	struct fun_create_node * i = fun_create_node_list_begin(list);
	STARPU_ASSERT(i);
	STARPU_ASSERT(i->create_node());
	c.top = c.bottom = i->create_node();

	for(i  = fun_create_node_list_next(i);
	    i != fun_create_node_list_end(list);
	    i  = fun_create_node_list_next(i))
	{
		STARPU_ASSERT(i->create_node());
		struct _starpu_sched_node * node = i->create_node();
		node->obj = obj;
		_starpu_sched_node_add_child(c.bottom, node);
		_starpu_sched_node_set_father(node, c.bottom, sched_ctx_id);
		c.bottom = node;
	}
	return c;
}


struct sched_node_list
{
	struct _starpu_sched_node ** arr;
	unsigned size;
};

static void init_list(struct sched_node_list * list)
{
	memset(list,0,sizeof(*list));
}
static void destroy_list(struct sched_node_list * list)
{
	free(list->arr);
}
static void add_node(struct sched_node_list *list, struct _starpu_sched_node * node)
{
	list->arr = realloc(list->arr,sizeof(*list->arr) * (list->size + 1));
	list->arr[list->size] = node;
	list->size++;
}
static struct sched_node_list helper_make_scheduler(hwloc_obj_t obj, struct _starpu_sched_specs specs, unsigned sched_ctx_id)
{
	STARPU_ASSERT(obj);

	struct composed_sched c;
	memset(&c,0,sizeof(c));

	/*set nodes for this obj */
#define CASE(ENUM,spec_member)						\
		case ENUM:						\
			if(specs.spec_member)				\
				c = create_composed_sched(sched_ctx_id,obj,specs.spec_member); \
			break
	switch(obj->type)
	{
		CASE(HWLOC_OBJ_MACHINE,hwloc_machine_composed_sched_node);
		CASE(HWLOC_OBJ_NODE,hwloc_node_composed_sched_node);
		CASE(HWLOC_OBJ_SOCKET,hwloc_socket_composed_sched_node);
		CASE(HWLOC_OBJ_CACHE,hwloc_cache_composed_sched_node);
	default:
		break;
	}

	struct sched_node_list l;
	init_list(&l);
	unsigned i;
	/* collect childs node's */
	for(i = 0; i < obj->arity; i++)
	{
		struct sched_node_list lc = helper_make_scheduler(obj->children[i],specs, sched_ctx_id);
		unsigned j;
		for(j = 0; j < lc.size; j++)
			add_node(&l, lc.arr[j]);
		destroy_list(&lc);
	}
	if(!c.bottom)
		return l;
	for(i = 0; i < l.size; i++)
	{
		_starpu_sched_node_add_child(c.bottom, l.arr[i]);
		_starpu_sched_node_set_father(l.arr[i],c.bottom,sched_ctx_id);
	}
	destroy_list(&l);
	init_list(&l);
	add_node(&l, c.top);
	return l;
}

struct _starpu_sched_tree * _starpu_make_scheduler(unsigned sched_ctx_id, struct _starpu_sched_specs specs)
{
	struct _starpu_sched_tree * tree = malloc(sizeof(*tree));
	STARPU_PTHREAD_RWLOCK_INIT(&tree->lock,NULL);
	
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	hwloc_topology_t topology = config->topology.hwtopology;

	struct sched_node_list list = helper_make_scheduler(hwloc_get_root_obj(topology), specs, sched_ctx_id);
	STARPU_ASSERT(list.size == 1);

	tree->root = list.arr[0];
	destroy_list(&list);

	return tree;
}
