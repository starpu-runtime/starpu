#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_scheduler.h>


#define USE_OVERLOAD
#ifdef USE_OVERLOAD
#include <float.h>

/**
 * Minimum number of task we wait for being processed before we start assuming
 * on which child the computation would be faster.
 */
static unsigned calibration_value = 0;

#endif /* USE_OVERLOAD */

struct _starpu_work_stealing_data
{
/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to child when stealing or deferring work
 */
	
	unsigned performed_total;
	unsigned last_pop_child;
	unsigned last_push_child;
};


/**
 * Return a child from which a task can be stolen.
 * Selecting a worker is done in a round-robin fashion, unless
 * the child previously selected doesn't own any task,
 * then we return the first non-empty worker.
 * and take his mutex
 * if no child have tasks return -1 
 */
static int select_victim_round_robin(struct _starpu_sched_node *node)
{
	struct _starpu_work_stealing_data *ws = node->data;
	unsigned i = ws->last_pop_child;
	
	
/* If the worker's queue is empty, let's try
 * the next ones */
	while (1)
	{
		unsigned ntasks;
		struct _starpu_sched_node * child = node->childs[i];
		struct _starpu_fifo_taskq * fifo = _starpu_sched_node_fifo_get_fifo(child);
		STARPU_PTHREAD_MUTEX_LOCK(&child->mutex);
		ntasks = fifo->ntasks;
		if (ntasks)
			break;
		STARPU_PTHREAD_MUTEX_UNLOCK(&child->mutex);
		i = (i + 1) % node->nchilds;
		if (i == ws->last_pop_child)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			return -1;
		}
	}

	ws->last_pop_child = (i+1)%node->nchilds;

	return i;
}

/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(struct _starpu_sched_node * node)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)node->data;
	unsigned i = (ws->last_push_child + 1) % node->nchilds ;
	ws->last_push_child = i;
	return i;
}

#ifdef USE_OVERLOAD

/**
 * Return a ratio helpful to determine whether a worker is suitable to steal
 * tasks from or to put some tasks in its queue.
 *
 * \return	a ratio with a positive or negative value, describing the current state of the worker :
 * 		a smaller value implies a faster worker with an relatively emptier queue : more suitable to put tasks in
 * 		a bigger value implies a slower worker with an reletively more replete queue : more suitable to steal tasks from
 */
static float overload_metric(struct _starpu_sched_node * fifo_node, unsigned performed_total)
{
	float execution_ratio = 0.0f;
	float current_ratio = 0.0f;
	struct _starpu_fifo_taskq * fifo = _starpu_sched_node_fifo_get_fifo(fifo_node);
	int nprocessed = fifo->nprocessed;
	unsigned ntasks = fifo->ntasks;

	/* Did we get enough information ? */
	if (performed_total > 0 && nprocessed > 0)
	{
/* How fast or slow is the worker compared to the other workers */
execution_ratio = (float) nprocessed / performed_total;
/* How replete is its queue */
current_ratio = (float) ntasks / nprocessed;
}
	else
	{
		return 0.0f;
	}

	return (current_ratio - execution_ratio);
}

/**
 * Return the most suitable worker from which a task can be stolen.
 * The number of previously processed tasks, total and local,
 * and the number of tasks currently awaiting to be processed
 * by the tasks are taken into account to select the most suitable
 * worker to steal task from.
 */
static int select_victim_overload(struct _starpu_sched_node * node)
{
	float  child_ratio;
	int best_child = -1;
	float best_ratio = FLT_MIN;
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)node->data;
	unsigned performed_total = ws->performed_total;

	/* Don't try to play smart until we get
	 * enough informations. */
	if (performed_total < calibration_value)
		return select_victim_round_robin(node);

	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		child_ratio = overload_metric(node->childs[i],performed_total);
		if(child_ratio > best_ratio)
		{
			best_ratio = child_ratio;
			best_child = i;
		}
	}
	
	return best_child;
}

/**
 * Return the most suitable worker to whom add a task.
 * The number of previously processed tasks, total and local,
 * and the number of tasks currently awaiting to be processed
 * by the tasks are taken into account to select the most suitable
 * worker to add a task to.
 */
static unsigned select_worker_overload(struct _starpu_sched_node * node)
{
	float  child_ratio;
	int best_child = -1;
	float best_ratio = FLT_MAX;
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)node->data;
	unsigned performed_total = ws->performed_total;

	/* Don't try to play smart until we get
	 * enough informations. */
	if (performed_total < calibration_value)
		return select_victim_round_robin(node);

	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		child_ratio = overload_metric(node->childs[i],performed_total);
		if(child_ratio < best_ratio)
		{
			best_ratio = child_ratio;
			best_child = i;
		}
	}
	
	return best_child;
}

#endif /* USE_OVERLOAD */


/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline int select_victim(struct _starpu_sched_node * node)
{
#ifdef USE_OVERLOAD
	return select_victim_overload(node);
#else
	return select_victim_round_robin(node);
#endif /* USE_OVERLOAD */
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct _starpu_sched_node * node)
{
#ifdef USE_OVERLOAD
	return select_worker_overload(node);
#else
	return select_worker_round_robin(node);
#endif /* USE_OVERLOAD */
}


static struct starpu_task * pop_task(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	int victim = select_victim(node);
	if(victim < 0)
	{
		if(node->fathers[sched_ctx_id])
			return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id],sched_ctx_id);
		else
			return NULL;
	}
	struct _starpu_sched_node * child = node->childs[victim];
	struct _starpu_fifo_taskq * fifo = _starpu_sched_node_fifo_get_fifo(child);
	struct starpu_task * task = _starpu_fifo_pop_task(fifo,
							  starpu_worker_get_id());
	fifo->nprocessed--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&child->mutex);
	if(task)
		starpu_push_task_end(task);
	return task;
}


static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_work_stealing_data * wsd = node->data;
	int ret = -1;
	int start = wsd->last_push_child;
	
	int i;
	for(i = (start+1)%node->nchilds; i != start; i = (i+1)%node->nchilds)
	{
		struct _starpu_sched_node * child = node->childs[i];
		if(_starpu_sched_node_can_execute_task(child,task))
		{
			ret = child->push_task(child,task);
			break;
		}
	}
	if(i == start)
		ret = -ENODEV;
	wsd->last_push_child = i;
	return ret;
}


static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id);
//compute if le father is a work_stealing node
static int is_my_fifo_node(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node->fathers[sched_ctx_id] == NULL)
		return 0;
	return node->fathers[sched_ctx_id]->add_child == add_child;
}

//this function is special, when a worker call it, we want to push the task in his fifo
int _starpu_ws_push_task(struct starpu_task *task)
{
	int workerid = starpu_worker_get_id();
	if(workerid == -1)
		return _starpu_tree_push_task(task);
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_node * node =_starpu_sched_node_worker_get(workerid);
	while(node->fathers[sched_ctx_id] != NULL)
	{
		node = node->fathers[sched_ctx_id];
		if(is_my_fifo_node(node,sched_ctx_id))
		{
			STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
			int ret_val =  _starpu_fifo_push_sorted_task(node->data, task);
			STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
			return ret_val;
		}
	}
	//there were a problem here, dont know what to do
	STARPU_ASSERT(1);
	return _starpu_tree_push_task(task);
}


static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}
	node->childs = realloc(node->childs,
			       sizeof(struct _starpu_sched_node*)
			       * (node->nchilds + 1));
	struct _starpu_sched_node * fifo_node = _starpu_sched_node_fifo_create();
	_starpu_sched_node_add_child(fifo_node, child, sched_ctx_id);


	_starpu_sched_node_set_father(fifo_node, node, sched_ctx_id);
	node->childs[node->nchilds] = fifo_node;
	node->nchilds++;
	
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);

}
static void remove_child(struct _starpu_sched_node *node,
			 struct _starpu_sched_node *child,
			 unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(*node->childs[pos]->childs == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	struct _starpu_sched_node * fifo_node = node->childs[pos];
	node->childs[pos] = node->childs[--node->nchilds];
	STARPU_ASSERT(fifo_node->fathers[sched_ctx_id] == node);
	fifo_node->fathers[sched_ctx_id] = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}



struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_work_stealing_data * wsd = malloc(sizeof(*wsd));
	wsd->performed_total = 0;
	wsd->last_pop_child = 0;
	wsd->last_push_child = 0;
	node->data = wsd;
	node->pop_task = pop_task;
	node->push_task = push_task;
	node->add_child = add_child;
	node->remove_child = remove_child;
	return node;
}



static void initialize_ws_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
 	data->root = _starpu_sched_node_work_stealing_create();
	
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_ws_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *t = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_tree_destroy(t, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


static void add_worker_ws(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		t->root->add_child(t->root,
				   _starpu_sched_node_worker_get(workerids[i]),
				   sched_ctx_id);
	_starpu_tree_update_after_modification(t);
}

static void remove_worker_ws(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		t->root->remove_child(t->root,
				   _starpu_sched_node_worker_get(workerids[i]),
				   sched_ctx_id);

}


struct starpu_sched_policy _starpu_sched_tree_ws_policy =
{
	.init_sched = initialize_ws_center_policy,
	.deinit_sched = deinitialize_ws_center_policy,
	.add_workers = add_worker_ws,
	.remove_workers = remove_worker_ws,
	.push_task = _starpu_ws_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-ws",
	.policy_description = "work stealing tree policy"
};
