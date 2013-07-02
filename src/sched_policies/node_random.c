#include <core/workers.h>
#include <starpu_sched_node.h>

struct _starpu_random_data
{
	double * relative_speedup;
};


static double compute_relative_speedup(struct starpu_sched_node * node)
{
	if(starpu_sched_node_is_simple_worker(node))
	{
		int id = starpu_sched_node_worker_get_workerid(node);
		enum starpu_perfmodel_archtype perf_arch = starpu_worker_get_perf_archtype(id);
		return starpu_worker_get_relative_speedup(perf_arch);
	}
	if(starpu_sched_node_is_combined_worker(node))
	{
		struct _starpu_combined_worker * c = starpu_sched_node_combined_worker_get_combined_worker(node);
		return starpu_worker_get_relative_speedup(c->perf_arch);
		
	}
	double sum = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
		sum += compute_relative_speedup(node->childs[i]);
	return sum;
}

static void init_data_random(struct starpu_sched_node * node)
{
	struct _starpu_random_data * rd = malloc(sizeof(struct _starpu_random_data));
	node->data = rd;
	rd->relative_speedup = malloc(sizeof(double) * node->nchilds);
	int i;
	for(i = 0; i < node->nchilds; i++)
		rd->relative_speedup[i] = compute_relative_speedup(node->childs[i]);
}

static void deinit_data_random(struct starpu_sched_node * node)
{
	struct _starpu_random_data * rd = node->data;
	free(rd->relative_speedup);
	free(rd);
	
}

static int push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_random_data * rd = node->data;

	int indexes_nodes[node->nchilds];
	int size=0,i;
	double alpha_sum = 0.0;
	for(i = 0; i < node->nchilds ; i++)
	{
		if(starpu_sched_node_can_execute_task(node->childs[i],task))
		{
			indexes_nodes[size++] = i;
			alpha_sum += rd->relative_speedup[i];
		}
	}

	double random = starpu_drand48()*alpha_sum;
	double alpha = 0.0;
	struct starpu_sched_node * select  = NULL;
	
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


double random_estimated_end(struct starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
		sum += node->childs[i]->estimated_end(node->childs[i]);
	return sum / node->nchilds;
}
struct starpu_sched_node * starpu_sched_node_random_create(void * arg STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	node->data = NULL;
	node->init_data = init_data_random;
	node->estimated_end = random_estimated_end;
	node->deinit_data = deinit_data_random;
	node->push_task = push_task;
	starpu_srand48(time(NULL));
	return node;
}

int starpu_sched_node_is_random(struct starpu_sched_node *node)
{
	return node->init_data == init_data_random;
}


static void initialize_random_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *data = malloc(sizeof(struct starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->lock,NULL);
 	data->root = starpu_sched_node_random_create(NULL);
	data->workers = starpu_bitmap_create();

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * node = starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = data->root;
		starpu_sched_node_add_child(data->root, node);
	}
	_starpu_set_workers_bitmaps();
	starpu_sched_tree_call_init_data(data);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}
static void deinitialize_random_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


struct starpu_sched_policy _starpu_sched_tree_random_policy =
{
	.init_sched = initialize_random_center_policy,
	.deinit_sched = deinitialize_random_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-random",
	.policy_description = "random tree policy"
};
