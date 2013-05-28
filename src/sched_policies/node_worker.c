#include "node_sched.h"
#include <core/workers.h>
#include <float.h>

static struct _starpu_sched_node * _worker_nodes[STARPU_NMAXWORKERS];
static struct _starpu_sched_node  * _starpu_sched_node_worker_create(int workerid);
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];
	else
		return _worker_nodes[workerid] = _starpu_sched_node_worker_create(workerid);
}



int _starpu_sched_node_worker_push_task(struct _starpu_sched_node * node, struct starpu_task *task)
{
	/*this function take the worker's mutex */
	
	int ret = _starpu_push_local_task(node->data, task, task->priority);
	return ret;
/*	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int ret_val = _starpu_fifo_push_sorted_task(node->fifo, task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	node->available(node);
	return ret_val;
*/
}

struct starpu_task * _starpu_sched_node_worker_pop_task(struct _starpu_sched_node *node,unsigned sched_ctx_id)
{
	struct _starpu_sched_node *father = node->fathers[sched_ctx_id];
	if(father == NULL)
		return NULL;
	else
		return father->pop_task(father,sched_ctx_id);
}
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *node)
{
	struct _starpu_worker * worker = node->data;
	unsigned id = worker->workerid;
	assert(_worker_nodes[id] == node);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS ; i++)
		if(node->fathers[i] != NULL)
			return;//this node is shared between several contexts
	_starpu_sched_node_destroy(node);
	_worker_nodes[id] = NULL;
}

static void available(struct _starpu_sched_node * worker_node)
{
	struct _starpu_worker * w = worker_node->data;
	starpu_pthread_mutex_t *sched_mutex = &w->sched_mutex;
	starpu_pthread_cond_t *sched_cond = &w->sched_cond;

	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	STARPU_PTHREAD_COND_SIGNAL(sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static double estimated_transfer_length(struct _starpu_sched_node * node,
					struct starpu_task * task)
{
	STARPU_ASSERT(_starpu_sched_node_is_worker(node));
	unsigned memory_node = starpu_worker_get_memory_node(node->workerids[0]);
	double d = starpu_task_expected_data_transfer_time(memory_node, task);
	return d;
}
static double estimated_finish_time(struct _starpu_sched_node * node)
{
	struct _starpu_worker * worker = node->data;
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	double sum = 0.0;
	struct starpu_task_list list = worker->local_tasks;
	struct starpu_task * task;
	for(task = starpu_task_list_front(&list);
	    task != starpu_task_list_end(&list);
	    task = starpu_task_list_next(task))
		if(!isnan(task->predicted))
		   sum += task->predicted;
/*	if(worker->current_task) 
	{
	// drôle de bug, t est parfois null, il doit y avoir un problème de mutex quelque part
		struct starpu_task * t = worker->current_task;
		if(!isnan(t->predicted))
			sum += t->predicted/2;
			}*/
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
	return sum + starpu_timing_now();
}

struct _starpu_task_execute_preds estimated_execute_preds(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(_starpu_sched_node_is_worker(node));
	struct _starpu_worker * worker = node->data;
	struct _starpu_task_execute_preds preds =
		{
			.state = CANNOT_EXECUTE,
			.archtype = worker->perf_arch,
			.expected_length = DBL_MAX,
			.expected_finish_time = estimated_finish_time(node),
			.expected_transfer_length = estimated_transfer_length(node, task),
			.expected_power = 0.0
		};

	int nimpl;
	for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
	{
		if(starpu_worker_can_execute_task(worker->workerid,task,nimpl))
		{
			double d = starpu_task_expected_length(task,
							       worker->perf_arch,
							       nimpl);
			if(isnan(d))
			{
				preds.state = CALIBRATING;
				preds.expected_length = d;
				preds.impl = nimpl;
				return preds;
			}
			if(_STARPU_IS_ZERO(d) && preds.state == CANNOT_EXECUTE)
			{
				preds.state = NO_PERF_MODEL;
				preds.impl = nimpl;
				continue;
			}
			if(d < preds.expected_length)
			{
				preds.state = PERF_MODEL;
				preds.expected_length = d;
				preds.impl = nimpl;
			}
		}
	}

	if(preds.state == PERF_MODEL)
		preds.expected_finish_time = _starpu_compute_expected_time(starpu_timing_now(),
									  preds.expected_finish_time,
									  preds.expected_length,
									  preds.expected_transfer_length);
	return preds;
}

static double estimated_load(struct _starpu_sched_node * node)
{
	struct _starpu_worker * worker = node->data;
	int nb_task = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	struct starpu_task_list list = worker->local_tasks;
	struct starpu_task * task;
	for(task = starpu_task_list_front(&list);
	    task != starpu_task_list_end(&list);
	    task = starpu_task_list_next(task))
		nb_task++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
	return (double) nb_task
		/ starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(node->workerids[0]));
}



static struct _starpu_sched_node  * _starpu_sched_node_worker_create(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid <  (int) starpu_worker_get_count());

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_worker * worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = worker;
	node->push_task = _starpu_sched_node_worker_push_task;
	node->pop_task = _starpu_sched_node_worker_pop_task;
	node->estimated_execute_preds = estimated_execute_preds;
	node->estimated_load = estimated_load;
	node->destroy_node = _starpu_sched_node_worker_destroy;
	node->available = available;
	node->workerids[0] = workerid;
	node->nworkers = 1;
	_worker_nodes[workerid] = node;
	return node;
}

int _starpu_sched_node_is_worker(struct _starpu_sched_node * node)
{
	return node->available == available
		|| node->push_task == _starpu_sched_node_worker_push_task
		|| node->pop_task == _starpu_sched_node_worker_pop_task
		|| node->estimated_execute_preds == estimated_execute_preds;
		
}

#ifndef STARPU_NO_ASSERT
static int _worker_consistant(struct _starpu_sched_node * node)
{
	int is_a_worker = 0;
	int i;
	for(i = 0; i<STARPU_NMAXWORKERS; i++)
		if(_worker_nodes[i] == node)
			is_a_worker = 1;
	if(!is_a_worker)
		return 0;
	struct _starpu_worker * worker = node->data;
	int id = worker->workerid;
	return  (_worker_nodes[id] == node)
		&&  node->nchilds == 0;
}
#endif

int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node)
{
#ifndef STARPU_NO_ASSERT
	STARPU_ASSERT(_worker_consistant(worker_node));
#endif
	struct _starpu_worker * worker = worker_node->data;
	return worker->workerid;
}
