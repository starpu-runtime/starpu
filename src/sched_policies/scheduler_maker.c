#include "scheduler_maker.h"
#include "node_sched.h"
#include "node_composed.h"
#include <common/list.h>
#include <stdarg.h>
#include <core/workers.h>

static void set_all_data_to_null(struct _starpu_sched_node * node)
{
	if(node)
	{
		node->data = NULL;
		int i;
		for(i = 0; i < node->nchilds; i++)
			set_all_data_to_null(node->childs[i]);
	}
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

	struct _starpu_sched_node * node = NULL;

	/*set nodes for this obj */
#define CASE(ENUM,spec_member)						\
		case ENUM:						\
			if(specs.spec_member)				\
				node = _starpu_sched_node_composed_node_create(specs.spec_member); \
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
	if(!node)
		return l;
	for(i = 0; i < l.size; i++)
	{
		_starpu_sched_node_add_child(node, l.arr[i]);
		_starpu_sched_node_set_father(l.arr[i],node,sched_ctx_id);
	}
	destroy_list(&l);
	init_list(&l);
	node->obj = obj;
	add_node(&l, node);
	return l;
}

struct _starpu_sched_node * _find_sched_node_with_obj(struct _starpu_sched_node * node, hwloc_obj_t obj)
{
	if(node == NULL)
		return NULL;
	if(node->obj == obj)
		return node;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * tmp = _find_sched_node_with_obj(node->childs[i], obj);
		if(tmp)
			return tmp;
	}
	return NULL;
}


static void set_cpu_worker_leaf(struct _starpu_sched_node * root, struct _starpu_sched_node * worker, unsigned sched_ctx_id,
				struct _starpu_composed_sched_node_recipe * cpu_composed_sched_node)
{
	hwloc_obj_t obj = worker->obj;
	STARPU_ASSERT(!_find_sched_node_with_obj(root, obj));
	while(obj)
	{
		obj = obj->parent;
		struct _starpu_sched_node * tmp = _find_sched_node_with_obj(root, obj);
		if(tmp)
		{
			struct _starpu_sched_node * node = _starpu_sched_node_composed_node_create(cpu_composed_sched_node);
			node->obj = worker->obj;
			_starpu_sched_node_set_father(node, tmp, sched_ctx_id);
			_starpu_sched_node_add_child(tmp, node);

			_starpu_sched_node_set_father(worker, node, sched_ctx_id);
			_starpu_sched_node_add_child(node, worker);
			return;
		}
	}
	STARPU_ABORT();
}

static void set_other_worker_leaf(struct _starpu_sched_node * root, struct _starpu_sched_node * worker, unsigned sched_ctx_id,
				  struct _starpu_composed_sched_node_recipe * device_composed_sched_node, int sched_have_numa_node)
{
	hwloc_obj_t obj = worker->obj;
	while(obj)
		if((sched_have_numa_node && obj->type == HWLOC_OBJ_NODE) || obj->type == HWLOC_OBJ_MACHINE)
			break;
		else
			obj = obj->parent;
	STARPU_ASSERT(obj != NULL);

	struct _starpu_sched_node * tmp = _find_sched_node_with_obj(root, obj);
	if(tmp)
	{
#ifdef STARPU_DEVEL
#warning FIXME node->obj is set to worker->obj even for accelerators workers
#endif
		struct _starpu_sched_node * node = _starpu_sched_node_composed_node_create(device_composed_sched_node);
		node->obj = worker->obj;
		_starpu_sched_node_set_father(node, tmp, sched_ctx_id);
		_starpu_sched_node_add_child(tmp, node);
		
		_starpu_sched_node_set_father(worker, node, sched_ctx_id);
		_starpu_sched_node_add_child(node, worker);
		return;
	}
	STARPU_ABORT();
}


#ifdef STARPU_DEVEL
static const char * name_hwloc_node(struct _starpu_sched_node * node)
{
	return hwloc_obj_type_string(node->obj->type);
}
static const char * name_sched_node(struct _starpu_sched_node * node)
{
	if(_starpu_sched_node_is_fifo(node))
		return "fifo node";
	if(_starpu_sched_node_is_heft(node))
		return "heft node";
	if(_starpu_sched_node_is_random(node))
		return "random node";
	if(_starpu_sched_node_is_worker(node))
	{
		struct _starpu_worker * w = _starpu_sched_node_worker_get_worker(node);
#define SIZE 256
		static char output[SIZE];
		snprintf(output, SIZE,"node worker %d %s",w->workerid,w->name);
		return output;
	}
	if(_starpu_sched_node_is_work_stealing(node))
		return "work stealing node";

	return "unknown";
}
static void helper_display_scheduler(FILE* out, unsigned depth, struct _starpu_sched_node * node)
{
	if(!node)
		return;
	fprintf(out,"%*s-> %s : %s\n", depth * 2 , "", name_sched_node(node), name_hwloc_node(node));
	int i;
	for(i = 0; i < node->nchilds; i++)
		helper_display_scheduler(out, depth + 1, node->childs[i]);
}
#endif //STARPU_DEVEL

struct _starpu_sched_tree * _starpu_make_scheduler(unsigned sched_ctx_id, struct _starpu_sched_specs specs)
{
	struct _starpu_sched_tree * tree = malloc(sizeof(*tree));
	STARPU_PTHREAD_RWLOCK_INIT(&tree->lock,NULL);
	tree->workers = _starpu_bitmap_create();
	
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	hwloc_topology_t topology = config->topology.hwtopology;

	struct sched_node_list list = helper_make_scheduler(hwloc_get_root_obj(topology), specs, sched_ctx_id);
	STARPU_ASSERT(list.size == 1);

	tree->root = list.arr[0];
	destroy_list(&list);
	
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct _starpu_worker * worker = _starpu_get_worker_struct(i);
		struct _starpu_sched_node * worker_node = _starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker);
		struct _starpu_composed_sched_node_recipe * recipe = specs.worker_composed_sched_node(worker->arch);
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			set_cpu_worker_leaf(tree->root, worker_node, sched_ctx_id, recipe);
			break;
		default:
			set_other_worker_leaf(tree->root, worker_node, sched_ctx_id, recipe, NULL != specs.hwloc_node_composed_sched_node);
			break;
		}
		_starpu_destroy_composed_sched_node_recipe(recipe);
	}

	_starpu_set_workers_bitmaps();
	_starpu_tree_call_init_data(tree);
#ifdef STARPU_DEVEL
	fprintf(stderr, "scheduler created :\n");
	helper_display_scheduler(stderr, 0, tree->root);
#endif

	return tree;
}
