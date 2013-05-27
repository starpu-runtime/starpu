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

static void update_relative_childs_speedup(struct _starpu_sched_node * node)
{
	struct _starpu_random_data * rd = node->data;
	rd->relative_speedup = realloc(rd->relative_speedup,sizeof(double) * node->nchilds);
	int i;
	for(i = 0; i < node->nchilds; i++)
		rd->relative_speedup[i] = compute_relative_speedup(node->childs[i]);
}

static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id)
{
//	STARPU_PTHREAD_RWLOCK_WRLOCK(&node->mutex);
	_starpu_sched_node_add_child(node, child, sched_ctx_id);
	update_relative_childs_speedup(node);
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
}
static void remove_child(struct _starpu_sched_node *node,
			 struct _starpu_sched_node *child,
			 unsigned sched_ctx_id)
{
//	STARPU_PTHREAD_RWLOCK_WRLOCK(&node->mutex);
	_starpu_sched_node_remove_child(node, child, sched_ctx_id);
	update_relative_childs_speedup(node);
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
}


static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_random_data * rd = node->data;
//	STARPU_PTHREAD_RWLOCK_RDLOCK(&node->mutex);
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
	STARPU_ASSERT(select != NULL);t
	int ret_val = select->push_task(select,task);
	node->available(node);
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
	return ret_val;
}

static void destroy_random_node(struct _starpu_sched_node * node)
{
	struct _starpu_random_data * rd = node->data;
	free(rd->relative_speedup);
	free(rd);
	_starpu_sched_node_destroy(node);
}


struct _starpu_sched_node * _starpu_sched_node_random_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_random_data * rd = malloc(sizeof(struct _starpu_random_data));

	rd->relative_speedup = NULL;
	node->data = rd;
	node->destroy_node = destroy_random_node;
	node->push_task = push_task;
	node->add_child = add_child;
	node->remove_child = remove_child;
	starpu_srand48(time(NULL));
	return node;
}

static void initialize_random_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->mutex,NULL);
 	data->root = _starpu_sched_node_random_create();
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}
static void deinitialize_random_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_tree_destroy(tree, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


 static void add_worker_random(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
//	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->mutex);
		struct _starpu_sched_node * random_node = t->root;
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * worker = _starpu_sched_node_worker_get(workerids[i]);
		t->root->add_child(random_node, _starpu_sched_node_worker_get(workerids[i]), sched_ctx_id);
		_starpu_sched_node_set_father(worker, random_node, sched_ctx_id);
	}
	_starpu_tree_update_after_modification(t);
	update_relative_childs_speedup(random_node);
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->mutex);
}

static void remove_worker_random(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->mutex);
	struct _starpu_sched_node * random_node = t->root;
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * worker = _starpu_sched_node_worker_get(workerids[i]);
		random_node->remove_child(random_node, worker, sched_ctx_id);
		_starpu_sched_node_set_father(worker, NULL, sched_ctx_id);
	}
	_starpu_tree_update_after_modification(t);
	update_relative_childs_speedup(t->root);
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->mutex);
}

struct starpu_sched_policy _starpu_sched_tree_random_policy =
{
	.init_sched = initialize_random_center_policy,
	.deinit_sched = deinitialize_random_center_policy,
	.add_workers = add_worker_random,
	.remove_workers = remove_worker_random,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-random",
	.policy_description = "random tree policy"
};
