#include <core/workers.h>
#include "node_sched.h"

struct _starpu_random_data
{
	double * relative_speedup;
};


static double compute_relative_speedup(struct _starpu_sched_node * node)
{
	if(_starpu_sched_node_is_worker(node))
	{
		int id = _starpu_sched_node_worker_get_workerid(node);
		enum starpu_perfmodel_archtype perf_arch = starpu_worker_get_perf_archtype(id);
		return starpu_worker_get_relative_speedup(perf_arch);
	}
	double sum = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
		sum += compute_relative_speedup(node->childs[i]);
	return sum;
}

static void init_data_random(struct _starpu_sched_node * node)
{
	struct _starpu_random_data * rd = malloc(sizeof(struct _starpu_random_data));
	node->data = rd;
	rd->relative_speedup = malloc(sizeof(double) * node->nchilds);
	int i;
	for(i = 0; i < node->nchilds; i++)
		rd->relative_speedup[i] = compute_relative_speedup(node->childs[i]);
}

static void deinit_data_random(struct _starpu_sched_node * node)
{
	struct _starpu_random_data * rd = node->data;
	free(rd->relative_speedup);
	free(rd);
	
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_random_data * rd = node->data;

	int indexes_nodes[node->nchilds];
	int size=0,i;
	double alpha_sum = 0.0;
	for(i = 0; i < node->nchilds ; i++)
	{
		if(_starpu_sched_node_can_execute_task(node->childs[i],task))
		{
			indexes_nodes[size++] = i;
			alpha_sum += rd->relative_speedup[i];
		}
	}

	double random = starpu_drand48()*alpha_sum;
	double alpha = 0.0;
	struct _starpu_sched_node * select  = NULL;
	
	for(i = 0; i < size ; i++)
	{
		int index = indexes_nodes[i];
		if(alpha + rd->relative_speedup[index] >= random)
		{	
			select = node->childs[index];
			break;
		}
		alpha += rd->relative_speedup[index];
	}
	STARPU_ASSERT(select != NULL);
	int ret_val = select->push_task(select,task);
	node->available(node);

	return ret_val;
}


struct _starpu_sched_node * _starpu_sched_node_random_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = NULL;
	node->init_data = init_data_random;
	node->deinit_data = deinit_data_random;
	node->push_task = push_task;
	starpu_srand48(time(NULL));
	return node;
}

static void initialize_random_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->lock,NULL);
 	data->root = _starpu_sched_node_random_create();
	data->workers = _starpu_bitmap_create();

	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct _starpu_sched_node * node = _starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = data->root;
		_starpu_sched_node_add_child(data->root, node);
	}
	_starpu_set_workers_bitmaps();
	_starpu_call_init_data(data);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}
static void deinitialize_random_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_bitmap_destroy(tree->workers);
	_starpu_tree_destroy(tree, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


struct starpu_sched_policy _starpu_sched_tree_random_policy =
{
	.init_sched = initialize_random_center_policy,
	.deinit_sched = deinitialize_random_center_policy,
	.add_workers = _starpu_tree_add_workers,
	.remove_workers = _starpu_tree_remove_workers,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-random",
	.policy_description = "random tree policy"
};
