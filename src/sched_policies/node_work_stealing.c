#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_scheduler.h>
#include <starpu.h>

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
	wsd->last_pop_child = (wsd->last_pop_child + 1) % node->nchilds;
	/* If the worker's queue have no suitable tasks, let's try
	 * the next ones */
	struct starpu_task * task = NULL;
	while (1)
	{
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
		i = (i + 1) % node->nchilds;
	}

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
	return _starpu_bitmap_get(node->workers, workerid);
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
	{
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes + i);
		wsd->fifos[i]->nprocessed++;
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes + i);
		return task;
	}
	if(node->fathers[sched_ctx_id])
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id],sched_ctx_id);
	else
		return NULL;
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
			ret = _starpu_fifo_push_sorted_task(wsd->fifos[i], task);
			break;
		}
	}
	wsd->last_push_child = (wsd->last_push_child + 1) % node->nchilds;
	node->childs[i]->available(node->childs[i]);
	return ret;
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
			
			//we need to wake all workers
			int j;
			for(j = 0; j < node->nchilds; j++)
			{
				if(j == i)
					continue;
				node->childs[j]->available(node->childs[j]);
			}

			return ret;
		}
	}

	STARPU_ASSERT_MSG(0, "there were a problem here, dont know what to do");
	return _starpu_tree_push_task(task);
}


static void init_ws_data(struct _starpu_sched_node *node)
{
	struct _starpu_work_stealing_data * wsd = malloc(sizeof(*wsd));
	memset(wsd, 0, sizeof(*wsd));
	node->data = wsd;
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}
	int size = node->nchilds;
	wsd->fifos = malloc(sizeof(struct _starpu_fifo_taskq*) * size);
	wsd->mutexes = malloc(sizeof(starpu_pthread_rwlock_t) * size);

	for(i = 0; i < size; i++)
	{
		wsd->fifos[i] = _starpu_create_fifo();
		STARPU_PTHREAD_MUTEX_INIT(wsd->mutexes + i, NULL);
	}
}

static void deinit_ws_data(struct _starpu_sched_node *node)
{
	struct _starpu_work_stealing_data * wsd = node->data;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(wsd->mutexes + i);
		_starpu_destroy_fifo(wsd->fifos[i]);
	}
	free(wsd->mutexes);
	free(wsd->fifos);
	free(wsd);
	node->data = NULL;
}


struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->pop_task = pop_task;
	node->init_data = init_ws_data;
	node->deinit_data = deinit_ws_data;
	node->push_task = push_task;
	return node;
}

int _starpu_sched_node_is_work_stealing(struct _starpu_sched_node * node)
{
	return node->init_data == init_ws_data;
}



static void initialize_ws_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->lock,NULL);
	struct _starpu_sched_node * ws;
 	data->root = ws = _starpu_sched_node_work_stealing_create();
	data->workers = _starpu_bitmap_create();
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct _starpu_sched_node * node = _starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = ws;
		_starpu_sched_node_add_child(ws, node);
	}
	_starpu_set_workers_bitmaps();
	_starpu_tree_call_init_data(data);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_ws_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *t = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_bitmap_destroy(t->workers);
	_starpu_tree_destroy(t, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


struct starpu_sched_policy _starpu_sched_tree_ws_policy =
{
	.init_sched = initialize_ws_center_policy,
	.deinit_sched = deinitialize_ws_center_policy,
	.add_workers = _starpu_tree_add_workers,
	.remove_workers = _starpu_tree_remove_workers,
	.push_task = _starpu_ws_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-ws",
	.policy_description = "work stealing tree policy"
};
