#include "scheduler_maker.h"
#include <starpu_node_sched.h>
#include "node_composed.h"
#include <common/list.h>
#include <stdarg.h>
#include <core/workers.h>

static void set_all_data_to_null(struct starpu_sched_node * node)
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
	struct starpu_sched_node ** arr;
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
static void add_node(struct sched_node_list *list, struct starpu_sched_node * node)
{
	list->arr = realloc(list->arr,sizeof(*list->arr) * (list->size + 1));
	list->arr[list->size] = node;
	list->size++;
}
static struct sched_node_list helper_make_scheduler(hwloc_obj_t obj, struct starpu_sched_specs specs, unsigned sched_ctx_id)
{
	STARPU_ASSERT(obj);

	struct starpu_sched_node * node = NULL;

	/*set nodes for this obj */
#define CASE(ENUM,spec_member)						\
		case ENUM:						\
			if(specs.spec_member)				\
				node = starpu_sched_node_composed_node_create(specs.spec_member); \
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
		starpu_sched_node_add_child(node, l.arr[i]);
		starpu_sched_node_set_father(l.arr[i],node,sched_ctx_id);
	}
	destroy_list(&l);
	init_list(&l);
	node->obj = obj;
	add_node(&l, node);
	return l;
}

struct starpu_sched_node * _find_sched_node_with_obj(struct starpu_sched_node * node, hwloc_obj_t obj)
{
	if(node == NULL)
		return NULL;
	if(node->obj == obj)
		return node;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * tmp = _find_sched_node_with_obj(node->childs[i], obj);
		if(tmp)
			return tmp;
	}
	return NULL;
}


static int is_same_kind_of_all(struct starpu_sched_node * root, struct _starpu_worker * w)
{
	if(starpu_sched_node_is_worker(root))
	{
		struct _starpu_worker * w_ = root->data;
		return w_->perf_arch == w->perf_arch;
	}
	
	int i;
	for(i = 0;i < root->nchilds; i++)
		if(!is_same_kind_of_all(root->childs[i], w))
			return 0;
	return 1;
}

struct starpu_sched_node * find_mem_node(struct starpu_sched_node * root, struct starpu_sched_node * worker_node, unsigned sched_ctx_id)
{
	struct starpu_sched_node * node = worker_node;
	while(node->obj->type != HWLOC_OBJ_NODE
	      && node->obj->type != HWLOC_OBJ_MACHINE)
	{
		hwloc_obj_t tmp = node->obj;
		do
		{
			node = _find_sched_node_with_obj(root,tmp);
			tmp = tmp->parent;
		}
		while(!node);
		
	}
	return node;
}

static struct starpu_sched_node * where_should_we_plug_this(struct starpu_sched_node *root, struct starpu_sched_node * worker_node, struct starpu_sched_specs specs, unsigned sched_ctx_id)
{
	struct starpu_sched_node * mem = find_mem_node(root ,worker_node, sched_ctx_id);
	if(specs.mix_heterogeneous_workers || mem->fathers[sched_ctx_id] == NULL)
		return mem;
	hwloc_obj_t obj = mem->obj;
	struct starpu_sched_node * father = mem->fathers[sched_ctx_id];
	int i;
	for(i = 0; i < father->nchilds; i++)
	{
		if(father->childs[i]->obj == obj
		   && is_same_kind_of_all(father->childs[i], worker_node->data))
			return father->childs[i];
	}
	if(obj->type == HWLOC_OBJ_NODE)
	{	
		struct starpu_sched_node * node = starpu_sched_node_composed_node_create(specs.hwloc_node_composed_sched_node);
		node->obj = obj;
		starpu_sched_node_add_child(father, node);
		starpu_sched_node_set_father(node, father, sched_ctx_id);
		return node;
	}
	return father;
}

static void set_worker_leaf(struct starpu_sched_node * root, struct starpu_sched_node * worker_node, unsigned sched_ctx_id,
			    struct starpu_sched_specs specs)
{
	struct _starpu_worker * worker = worker_node->data;
	struct starpu_sched_node * node = where_should_we_plug_this(root,worker_node,specs, sched_ctx_id);
	struct _starpu_composed_sched_node_recipe * recipe = specs.worker_composed_sched_node ?
		specs.worker_composed_sched_node(worker->arch):NULL;
	STARPU_ASSERT(node);
	if(recipe)
	{
		struct starpu_sched_node * tmp = starpu_sched_node_composed_node_create(recipe);
#ifdef STARPU_DEVEL
#warning FIXME node->obj is set to worker_node->obj even for accelerators workers
#endif
		tmp->obj = worker_node->obj;
		starpu_sched_node_set_father(tmp, node, sched_ctx_id);
		starpu_sched_node_add_child(node, tmp);
		node = tmp;
		
	}
	_starpu_destroy_composed_sched_node_recipe(recipe);
	starpu_sched_node_set_father(worker_node, node, sched_ctx_id);
	starpu_sched_node_add_child(node, worker_node);
}

#ifdef STARPU_DEVEL
static const char * name_hwloc_node(struct starpu_sched_node * node)
{
	return hwloc_obj_type_string(node->obj->type);
}
static const char * name_sched_node(struct starpu_sched_node * node)
{
	if(starpu_sched_node_is_fifo(node))
		return "fifo node";
	if(starpu_sched_node_is_heft(node))
		return "heft node";
	if(starpu_sched_node_is_random(node))
		return "random node";
	if(starpu_sched_node_is_worker(node))
	{
		struct _starpu_worker * w = starpu_sched_node_worker_get_worker(node);
#define SIZE 256
		static char output[SIZE];
		snprintf(output, SIZE,"node worker %d %s",w->workerid,w->name);
		return output;
	}
	if(starpu_sched_node_is_work_stealing(node))
		return "work stealing node";

	return "unknown";
}
static void helper_display_scheduler(FILE* out, unsigned depth, struct starpu_sched_node * node)
{
	if(!node)
		return;
	fprintf(out,"%*s-> %s : %s\n", depth * 2 , "", name_sched_node(node), name_hwloc_node(node));
	int i;
	for(i = 0; i < node->nchilds; i++)
		helper_display_scheduler(out, depth + 1, node->childs[i]);
}
#endif //STARPU_DEVEL
struct starpu_sched_tree * _starpu_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_specs specs)
{
	struct starpu_sched_tree * tree = starpu_sched_tree_create(sched_ctx_id);
	
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
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker);
		set_worker_leaf(tree->root,worker_node, sched_ctx_id, specs);
	}


	starpu_sched_tree_update_workers(t);
#ifdef STARPU_DEVEL
	fprintf(stderr, "scheduler created :\n");
	helper_display_scheduler(stderr, 0, tree->root);
#endif

	return tree;

}
