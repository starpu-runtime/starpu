#include <starpu_sched_node.h>
#include "prio_deque.h"
#include <starpu_scheduler.h>
#include <starpu.h>
#include <float.h>
struct _starpu_work_stealing_data
{
/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to child when stealing or deferring work
 */
	unsigned performed_total, last_pop_child, last_push_child;

	struct _starpu_prio_deque ** fifos;	
	starpu_pthread_mutex_t ** mutexes;
	int size;
};


/**
 * steal a task in a round robin way
 * return NULL if none available
 */
static struct starpu_task *  steal_task_round_robin(struct starpu_sched_node *node, int workerid)
{
	struct _starpu_work_stealing_data *wsd = node->data;
	unsigned i = wsd->last_pop_child;
	wsd->last_pop_child = (wsd->last_pop_child + 1) % node->nchilds;
	/* If the worker's queue have no suitable tasks, let's try
	 * the next ones */
	struct starpu_task * task = NULL;
	while (1)
	{
		struct _starpu_prio_deque * fifo = wsd->fifos[i];

		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
		task = _starpu_prio_deque_deque_task_for_worker(fifo, workerid);
		if(task && !isnan(task->predicted))
		{
			fifo->exp_len -= task->predicted;
			fifo->nprocessed--;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);
		if(task)
			break;

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
static unsigned select_worker_round_robin(struct starpu_sched_node * node)
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
static inline struct starpu_task * steal_task(struct starpu_sched_node * node, int workerid)
{
	return steal_task_round_robin(node, workerid);
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct starpu_sched_node * node)
{
	return select_worker_round_robin(node);
}


static int is_worker_of_node(struct starpu_sched_node * node, int workerid)
{
	return starpu_bitmap_get(node->workers, workerid);
}



static struct starpu_task * pop_task(struct starpu_sched_node * node, unsigned sched_ctx_id)
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
	STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
	struct starpu_task * task = _starpu_prio_deque_pop_task(wsd->fifos[i]);
	if(task)
	{
		if(!isnan(task->predicted))
		{
			wsd->fifos[i]->exp_len -= task->predicted;
			wsd->fifos[i]->exp_start = starpu_timing_now() + task->predicted;
		}
	}
	else
		wsd->fifos[i]->exp_len = 0.0;

	STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);
	if(task)
	{
		return task;
	}
	
	task  = steal_task(node, workerid);
	if(task)
	{
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
		wsd->fifos[i]->nprocessed++;
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);

		return task;
	}
	if(node->fathers[sched_ctx_id])
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id],sched_ctx_id);
	else
		return NULL;
}

double _ws_estimated_end(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_work_stealing(node));
	struct _starpu_work_stealing_data * wsd = node->data;
	double sum_len = 0.0;
	double sum_start = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
		sum_len += wsd->fifos[i]->exp_len;
		wsd->fifos[i]->exp_start = STARPU_MAX(starpu_timing_now(), wsd->fifos[i]->exp_start);
		sum_start += wsd->fifos[i]->exp_start;
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);

	}
	int nb_workers = starpu_bitmap_cardinal(node->workers_in_ctx);

	return (sum_start + sum_len) / nb_workers;
}

double _ws_estimated_load(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_work_stealing(node));
	struct _starpu_work_stealing_data * wsd = node->data;
	int ntasks = 0;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
		ntasks += wsd->fifos[i]->ntasks;
		STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);
	}
	double speedup = 0.0;
	int workerid;
	for(workerid = starpu_bitmap_first(node->workers_in_ctx);
	    -1 != workerid;
	    workerid = starpu_bitmap_next(node->workers_in_ctx, workerid))
	{
		speedup += starpu_worker_get_relative_speedup(workerid);
	}
	
	return ntasks / speedup;
}

static int push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_work_stealing_data * wsd = node->data;
	int ret = -1;
	int i = wsd->last_push_child;
	i = (i+1)%node->nchilds;

	STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
	ret = _starpu_prio_deque_push_task(wsd->fifos[i], task);
	STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);

	wsd->last_push_child = i;
//	node->available(node);
	starpu_sched_node_available(node);
	return ret;
}


//this function is special, when a worker call it, we want to push the task in his fifo
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task)
{
	int workerid = starpu_worker_get_id();
	if(workerid == -1)
		return starpu_sched_tree_push_task(task);

	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_node * node =starpu_sched_node_worker_get(workerid);
	while(node->fathers[sched_ctx_id] != NULL)
	{
		node = node->fathers[sched_ctx_id];
		if(starpu_sched_node_is_work_stealing(node))
		{
			if(!starpu_sched_node_can_execute_task(node, task))
				return starpu_sched_tree_push_task(task);

			int i;
			for(i = 0; i < node->nchilds; i++)
				if(is_worker_of_node(node->childs[i], workerid))
					break;
			STARPU_ASSERT(i < node->nchilds);
			
			struct _starpu_work_stealing_data * wsd = node->data;
			STARPU_PTHREAD_MUTEX_LOCK(wsd->mutexes[i]);
			int ret = _starpu_prio_deque_push_task(wsd->fifos[i] , task);
			if(ret == 0 && !isnan(task->predicted))
				wsd->fifos[i]->exp_len += task->predicted;
			STARPU_PTHREAD_MUTEX_UNLOCK(wsd->mutexes[i]);
			
			//we need to wake all workers
/*			int j;
			for(j = 0; j < node->nchilds; j++)
			{
				if(j == i)
					continue;
				node->childs[j]->available(node->childs[j]);
			}
*/
			starpu_sched_node_available(node);
			return ret;
		}
	}

	/* this should not be reached */
	return starpu_sched_tree_push_task(task);
}




void _ws_add_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	struct _starpu_work_stealing_data * wsd = node->data;
	starpu_sched_node_add_child(node, child);
	if(wsd->size < node->nchilds)
	{
		STARPU_ASSERT(wsd->size == node->nchilds - 1);
		wsd->fifos = realloc(wsd->fifos, node->nchilds * sizeof(*wsd->fifos));
		wsd->mutexes = realloc(wsd->mutexes, node->nchilds * sizeof(*wsd->mutexes));
		wsd->size = node->nchilds;
	}

	struct _starpu_prio_deque * fifo = malloc(sizeof(*fifo));
	_starpu_prio_deque_init(fifo);
	wsd->fifos[node->nchilds - 1] = fifo;

	starpu_pthread_mutex_t * mutex = malloc(sizeof(*mutex));
	STARPU_PTHREAD_MUTEX_INIT(mutex,NULL);
	wsd->mutexes[node->nchilds - 1] = mutex;
}

void _ws_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	struct _starpu_work_stealing_data * wsd = node->data;

	STARPU_PTHREAD_MUTEX_DESTROY(wsd->mutexes[node->nchilds - 1]);
	free(wsd->mutexes[node->nchilds - 1]);

	int i_node;
	for(i_node = 0; i_node < node->nchilds; i_node++)
	{
		if(node->childs[i_node] == child)
			break;
	}
	STARPU_ASSERT(i_node != node->nchilds);
	struct _starpu_prio_deque * tmp_fifo = wsd->fifos[i_node];
	wsd->fifos[i_node] = wsd->fifos[node->nchilds - 1];

	
	node->childs[i_node] = node->childs[node->nchilds - 1];
	node->nchilds--;
	struct starpu_task * task;
	while((task = _starpu_prio_deque_pop_task(tmp_fifo)))
	{
		node->push_task(node, task);
	}
	_starpu_prio_deque_destroy(tmp_fifo);
	free(tmp_fifo);
}

void _work_stealing_node_deinit_data(struct starpu_sched_node * node)
{
	free(node->data);
}

struct starpu_sched_node * starpu_sched_node_work_stealing_create(void * arg STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_work_stealing_data * wsd = malloc(sizeof(*wsd));
	memset(wsd, 0, sizeof(*wsd));
	node->pop_task = pop_task;
	node->push_task = push_task;
	node->add_child = _ws_add_child;
	node->remove_child = _ws_remove_child;
	node->estimated_end = _ws_estimated_end;
	node->estimated_load = _ws_estimated_load;
	node->deinit_data = _work_stealing_node_deinit_data;
	node->data = wsd;
	return  node;
}

int starpu_sched_node_is_work_stealing(struct starpu_sched_node * node)
{
	return node->push_task == push_task;
}



static void initialize_ws_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
	struct starpu_sched_node * ws;
 	t->root = ws = starpu_sched_node_work_stealing_create(NULL);
	t->workers = starpu_bitmap_create();
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct starpu_sched_node * node = starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = ws;
		ws->add_child(ws, node);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_ws_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}


struct starpu_sched_policy _starpu_sched_tree_ws_policy =
{
	.init_sched = initialize_ws_center_policy,
	.deinit_sched = deinitialize_ws_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_work_stealing_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-ws",
	.policy_description = "work stealing tree policy"
};
