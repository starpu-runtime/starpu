#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_scheduler.h>


struct _starpu_work_stealing_data
{
/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to child when stealing or deferring work
 */
	
	unsigned performed_total;
	unsigned last_pop_child;
	unsigned last_push_child;
	
	struct _starpu_fifo_taskq ** fifos;
	starpu_pthread_mutex_t * mutexes;
};


/**
 * steal a task in a round robin way
 * return NULL if none available
 */
static struct starpu_task *  steal_task_round_robin(struct _starpu_sched_node *node, int workerid)
{
	struct _starpu_work_stealing_data *wsd = node->data;
	unsigned i = wsd->last_pop_child;
	/* If the worker's queue have no suitable tasks, let's try
	 * the next ones */
	struct starpu_task * task = NULL;
	while (1)
	{
		i = (i + 1) % node->nchilds;
		struct _starpu_fifo_taskq * fifo = wsd->fifos[i];
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes + i);
		task = _starpu_fifo_pop_task(fifo, workerid);
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes + i);
		if(task)
		{
			fifo->nprocessed--;
			break;
		}
		if (i == wsd->last_pop_child)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			return NULL;
		}
	}

	wsd->last_pop_child = i;
	return task;
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


/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline struct starpu_task * steal_task(struct _starpu_sched_node * node, int workerid)
{
	return steal_task_round_robin(node, workerid);
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct _starpu_sched_node * node)
{
	return select_worker_round_robin(node);
}


static int is_worker_of_node(struct _starpu_sched_node * node, int workerid)
{
	int j;
	for(j = 0; j < node->nworkers; j++)
	{
		if(node->workerids[j] == workerid)
			return 1;
	}
	return 0;
}



static struct starpu_task * pop_task(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	int workerid = starpu_worker_get_id();
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		if(is_worker_of_node(node->childs[i], workerid))
			break;
	}
	STARPU_ASSERT(i < node->nchilds);
	struct _starpu_work_stealing_data * wsd = node->data;
	STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes + i);
	struct starpu_task * task = _starpu_fifo_pop_local_task(wsd->fifos[i]);
	STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes + i);
	if(task)
		return task;
	task  = steal_task(node, workerid);
	if(task)
		return task;
	if(node->fathers[sched_ctx_id])
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id],sched_ctx_id);
	else
		return NULL;
}



static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_PTHREAD_RWLOCK_RDLOCK(&node->mutex);
	struct _starpu_work_stealing_data * wsd = node->data;
	int ret = -1;
	int start = wsd->last_push_child;
	
	int i;
	for(i = (start+1)%node->nchilds; i != start; i = (i+1)%node->nchilds)
	{
		struct _starpu_sched_node * child = node->childs[i];
		if(_starpu_sched_node_can_execute_task(child,task))
		{
			ret = _starpu_fifo_push_sorted_task(wsd->fifos[i], task);
			break;
		}
	}
	wsd->last_push_child = (wsd->last_push_child + 1) % node->nchilds;
	node->childs[i]->available(node->childs[i]);
	return ret;
}


static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id);



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
		if(_starpu_sched_node_is_work_stealing(node))
		{
			int i;
			for(i = 0; i < node->nchilds; i++)
				if(is_worker_of_node(node->childs[i], workerid))
					break;
			STARPU_ASSERT(i < node->nchilds);
			
			struct _starpu_work_stealing_data * wsd = node->data;
			STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes + i);
			int ret = _starpu_fifo_push_sorted_task(wsd->fifos[i], task);
			STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes + i);
			node->childs[i]->available(node->childs[i]);
			return ret;
		}
	}
	STARPU_ASSERT_MSG(0, "there were a problem here, dont know what to do");
	return _starpu_tree_push_task(task);
}


static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}
	struct _starpu_work_stealing_data * wsd = node->data;
	int new_size = node->nchilds + 1;
	node->childs = realloc(node->childs,
			       sizeof(struct _starpu_sched_node*)
			       * new_size);
	wsd->fifos = realloc(wsd->fifos,
			     sizeof(struct _starpu_fifo_taskq*)
			     * new_size);
	wsd->mutexes = realloc(wsd->mutexes,
			     sizeof(starpu_pthread_rwlock_t)
			     * new_size);
	node->childs[new_size - 1] = child;
	wsd->fifos[new_size - 1] = _starpu_create_fifo();
	STARPU_PTHREAD_MUTEX_INIT(wsd->mutexes + (new_size - 1), NULL);
	node->nchilds = new_size;
}


static void remove_child(struct _starpu_sched_node *node,
			 struct _starpu_sched_node *child,
			 unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	struct _starpu_work_stealing_data * wsd = node->data;
	struct _starpu_fifo_taskq * fifo = wsd->fifos[pos];
	wsd->fifos[pos] = wsd->fifos[node->nchilds - 1];
	node->childs[pos] = node->childs[node->nchilds - 1];
	STARPU_PTHREAD_MUTEX_DESTROY(wsd->mutexes + pos);
	wsd->mutexes[pos] = wsd->mutexes[node->nchilds - 1];
	node->nchilds--;
	int i;
	struct starpu_task * task = fifo->taskq.head;
	_starpu_destroy_fifo(fifo);
	for(i = 0; task; i = (i + 1)%node->nchilds)
	{
		struct starpu_task * next = task->next;
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes + i);
		_starpu_fifo_push_sorted_task(wsd->fifos[i],task);
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes + i);
		task = next;
	}
}



struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_work_stealing_data * wsd = malloc(sizeof(*wsd));
	memset(wsd, 0, sizeof(*wsd));
	node->data = wsd;
	node->pop_task = pop_task;
	node->push_task = push_task;
	node->add_child = add_child;
	node->remove_child = remove_child;
	return node;
}

int _starpu_sched_node_is_work_stealing(struct _starpu_sched_node * node)
{
	return node->add_child == add_child
		|| node->remove_child == remove_child
		|| node->pop_task == pop_task
		|| node->push_task == push_task;//...
}



static void initialize_ws_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->mutex,NULL);
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
	struct _starpu_sched_node * ws_node = t->root;
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * worker =_starpu_sched_node_worker_get(workerids[i]);
		ws_node->add_child(ws_node,
				   worker,
				   sched_ctx_id);
		_starpu_sched_node_set_father(worker, ws_node, sched_ctx_id);
	}
	_starpu_tree_update_after_modification(t);
}

static void remove_worker_ws(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_sched_node * ws_node = t->root;
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		struct _starpu_sched_node * worker =_starpu_sched_node_worker_get(workerids[i]);
		ws_node->remove_child(ws_node, worker, sched_ctx_id);
		_starpu_sched_node_set_father(worker,NULL,sched_ctx_id);
	}
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
