#include <core/workers.h>
#include <starpu_sched_node.h>

static double compute_relative_speedup(struct starpu_sched_node * node)
{
	double sum = 0.0;
	int id;
	for(id = starpu_bitmap_first(node->workers_in_ctx);
	    id != -1;
	    id = starpu_bitmap_next(node->workers_in_ctx, id))
	{
		enum starpu_perfmodel_archtype perf_arch = starpu_worker_get_perf_archtype(id);
		sum += starpu_worker_get_relative_speedup(perf_arch);

	}
	STARPU_ASSERT(sum != 0.0);
	return sum;
}


static int random_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node->nchilds > 0);

	/* indexes_nodes and size are used to memoize node that can execute tasks
	 * during the first phase of algorithm, it contain the size indexes of the nodes
	 * that can execute task.
	 */
	int indexes_nodes[node->nchilds];
	int size=0;

	/* speedup[i] is revelant only if i is in the size firsts elements of
	 * indexes_nodes
	 */
	double speedup[node->nchilds];

	double alpha_sum = 0.0;

	int i;
	for(i = 0; i < node->nchilds ; i++)
	{
		if(starpu_sched_node_can_execute_task(node->childs[i],task))
		{
			speedup[size] = compute_relative_speedup(node->childs[i]);
			alpha_sum += speedup[size];
			indexes_nodes[size] = i;
			size++;
		}
	}
	if(size == 0)
		return -ENODEV;

	/* not fully sure that this code is correct
	 * because of bad properties of double arithmetic
	 */
	double random = starpu_drand48()*alpha_sum;
	double alpha = 0.0;
	struct starpu_sched_node * select  = NULL;
	
	for(i = 0; i < size ; i++)
	{
		int index = indexes_nodes[i];
		if(alpha + speedup[i] >= random)
		{	
			select = node->childs[index];
			break;
		}
		alpha += speedup[i];
	}
	STARPU_ASSERT(select != NULL);
	int ret_val = select->push_task(select,task);

	return ret_val;
}
/* taking the min of estimated_end not seems to be a good value to return here
 * as random scheduler balance between childs very poorly
 */
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
	node->estimated_end = random_estimated_end;
	node->push_task = random_push_task;
	return node;
}

int starpu_sched_node_is_random(struct starpu_sched_node *node)
{
	return node->push_task == random_push_task;
}


static void initialize_random_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_node_random_create(NULL);

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * node = starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = t->root;
		t->root->add_child(t->root, node);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}
static void deinitialize_random_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
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
