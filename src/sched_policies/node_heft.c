#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_perfmodel.h>
#include <starpu_scheduler.h>
#include <float.h>


struct _starpu_dmda_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
	
	struct _starpu_sched_node * no_model_node;
};

static double compute_fitness_calibration(struct _starpu_sched_node * child,
					  struct _starpu_dmda_data * data STARPU_ATTRIBUTE_UNUSED,
					  struct _starpu_task_execute_preds *pred,
					  double best_exp_end STARPU_ATTRIBUTE_UNUSED,
					  double max_exp_end STARPU_ATTRIBUTE_UNUSED)
{
	if(pred->state == CALIBRATING)
		return child->estimated_load(child);
	return DBL_MAX;
}

static double compute_fitness_perf_model(struct _starpu_sched_node * child STARPU_ATTRIBUTE_UNUSED,
					 struct _starpu_dmda_data * data,
					 struct _starpu_task_execute_preds * preds,
					 double best_exp_end,
					 double max_exp_end)
{
	double fitness;
	switch(preds->state)
	{
	case CANNOT_EXECUTE:
	case NO_PERF_MODEL:
		return DBL_MAX;
	case PERF_MODEL:
		fitness = data->alpha * (preds->expected_finish_time - best_exp_end)
			+ data->beta  * preds->expected_transfer_length
			+ data->gamma * preds->expected_power;
		return fitness;
	case CALIBRATING:
		STARPU_ASSERT_MSG(0,"we should have calibrate this task");
	default:
		STARPU_ABORT();
		break;
	}
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_PTHREAD_RWLOCK_RDLOCK(&node->mutex);
	struct _starpu_task_execute_preds preds[node->nchilds];
	int i;
	int calibrating = 0;
	int perf_model = 0;
	int can_execute = 0;
	double best_exp_end = DBL_MAX;
	double max_exp_end = DBL_MIN;
	for(i = 0; i < node->nchilds; i++)
	{
		preds[i] = node->childs[i]->estimated_execute_preds(node->childs[i], task);
		switch(preds[i].state)
		{
		case PERF_MODEL:
			STARPU_ASSERT(!isnan(preds[i].expected_finish_time));
			perf_model = 1;
			can_execute = 1;
			if(preds[i].expected_finish_time < best_exp_end)
				best_exp_end = preds[i].expected_finish_time;
			else if(preds[i].expected_finish_time > max_exp_end)
				max_exp_end = preds[i].expected_finish_time;
			break;
		case CALIBRATING:
			calibrating = 1;
			can_execute = 1;
			break;
		case NO_PERF_MODEL:
			can_execute = 1;
			break;
		case CANNOT_EXECUTE:
			break;
		}
	}
	if(!can_execute)
	{
		STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
		return -ENODEV;
	}
	
	struct _starpu_dmda_data * data = node->data;
	
	if(!calibrating && !perf_model)
	{
		int ret = data->no_model_node->push_task(data->no_model_node, task);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
		return ret;
	}

	double (*fitness_fun)(struct _starpu_sched_node *,
			      struct _starpu_dmda_data *,
			      struct _starpu_task_execute_preds*,
			      double,
			      double) = compute_fitness_perf_model;

	if(calibrating)
		fitness_fun = compute_fitness_calibration;



	double best_fitness = DBL_MAX;
	int index_best_fitness = -1;
	for(i = 0; i < node->nchilds; i++)
	{
		double tmp = fitness_fun(node->childs[i],
					 node->data,
					 preds + i,
					 best_exp_end,
					 max_exp_end);
		if(tmp < best_fitness)
		{
			best_fitness = tmp;
			index_best_fitness = i;
		}
	}
	STARPU_ASSERT(best_fitness != DBL_MAX);
	struct _starpu_sched_node * c = node->childs[index_best_fitness];
	starpu_task_set_implementation(task, preds[index_best_fitness].impl);
	task->predicted = preds[index_best_fitness].expected_length;
	task->predicted_transfer = preds[index_best_fitness].expected_transfer_length;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
	return c->push_task(c, task);
}
/*
static void update_helper_node(struct _starpu_sched_node * heft_node)
{
	struct _starpu_dmda_data * data = heft_node->data;
	struct _starpu_sched_node * node = data->no_model_node;
	node->nchilds = heft_node->nchilds;
	node->childs = realloc(node->childs, sizeof(struct _starpu_sched_node *) * node->nchilds);
	memcpy(node->childs, heft_node->childs, sizeof(struct _starpu_sched_node*) * node->nchilds);
	node->nworkers = heft_node->nworkers;
	memcpy(node->workerids, heft_node->workerids, sizeof(int) * node->nworkers);
}
*/

static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id)
{
	_starpu_sched_node_add_child(node,child, sched_ctx_id);
	struct _starpu_dmda_data * data = node->data;
	data->no_model_node->add_child(data->no_model_node, child, sched_ctx_id);

}
static void remove_child(struct _starpu_sched_node *node,
			 struct _starpu_sched_node *child,
			 unsigned sched_ctx_id)

{

	_starpu_sched_node_remove_child(node, child, sched_ctx_id);
	struct _starpu_dmda_data * data = node->data;
	data->no_model_node->remove_child(data->no_model_node, child, sched_ctx_id);
}





static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->mutex,NULL);
	data->root = _starpu_sched_node_heft_create(1,1,1,1);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *t = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_tree_destroy(t, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


static void add_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * fifo = _starpu_sched_node_fifo_create();
		struct _starpu_sched_node * worker =  _starpu_sched_node_worker_get(workerids[i]);

		fifo->add_child(fifo, worker, sched_ctx_id);
		_starpu_sched_node_set_father(worker, fifo, sched_ctx_id);

		t->root->add_child(t->root, fifo, sched_ctx_id);
		_starpu_sched_node_set_father(fifo, t->root, sched_ctx_id);

	}
	_starpu_tree_update_after_modification(t);
}

static void remove_worker_heft(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		int j;
		for(j = 0; j < t->root->nchilds; j++)
			if(t->root->childs[j]->workerids[0] == workerids[i])
				break;
		STARPU_ASSERT(j < t->root->nchilds);
		struct _starpu_sched_node * fifo = t->root->childs[j];
		_starpu_sched_node_set_father(fifo, NULL, sched_ctx_id);
		t->root->remove_child(t->root, fifo, sched_ctx_id);
	}
}

static void destroy_heft_node(struct _starpu_sched_node * node)
{
	struct _starpu_dmda_data * data = node->data;
	data->no_model_node->destroy_node(data->no_model_node);
	_starpu_sched_node_destroy(node);
	free(data);
}

struct _starpu_sched_node * _starpu_sched_node_heft_create(double alpha, double beta, double gamma, double idle_power)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_dmda_data * data = malloc(sizeof(*data));

	data->alpha = alpha;
	data->beta = beta;
	data->gamma = gamma;
	data->idle_power = idle_power;

	node->data = data;
	node->push_task = push_task;
	data->alpha = data->beta = data->gamma = data->idle_power = 0.0;
	//data->total_task_cnt = data->ready_task_cnt = 0;
	node->add_child = add_child;
	node->remove_child = remove_child;
	node->destroy_node = destroy_heft_node;

	data->no_model_node = _starpu_sched_node_random_create();

	return node;
}


struct starpu_sched_policy _starpu_sched_tree_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = add_worker_heft,
	.remove_workers = remove_worker_heft,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-heft",
	.policy_description = "heft tree policy"
};
