#include "node_sched.h"
#include <core/workers.h>
#include <float.h>

static struct _starpu_sched_node * _worker_nodes[STARPU_NMAXWORKERS];

/* data structure for worker's queue look like this :
 * W = worker
 * T = simple task
 * P = parallel task
 *
 *
 *         P--P  T
 *         |  | \|
 *   P--P  T  T  P  T
 *   |  |  |  |  |  |
 *   T  T  P--P--P  T
 *   |  |  |  |  |  |
 *   W  W  W  W  W  W
 *
 *
 *
 * its possible that a _starpu_task_grid wont have task
 *
 * N = no task
 *
 *   T  T  T
 *   |  |  |
 *   P--N--N
 *   |  |  |
 *   W  W  W
 * 
 * 
 * this API is a little asymmetric : _starpu_task_grid are allocated by the caller and freed by the data structure
 *
 * exp_{start,end,len} are filled by the caller
 */

struct _starpu_task_grid
{
	/* this member may be NULL if a worker have poped it but its a
	 * parallel task and we dont want mad pointers
	 */
	struct starpu_task * task;
	struct _starpu_task_grid *up, *down, *left, *right;
	
	/* this is used to count the number of task to be poped by a worker
	 * the leftist _starpu_task_grid maintain the ntasks counter (ie .left == NULL),
	 * all the others use the pntasks that point to it
	 *
	 * when the counter reach 0, all the left and right member are set to NULL,
	 * that mean that we will free that nodes.
	 */
	union
	{
		int ntasks;
		int * pntasks;
	};
};

struct _starpu_worker_task_list
{
	double exp_start, exp_len, exp_end;
	struct _starpu_task_grid *first, *last;
	starpu_pthread_mutex_t mutex;
};

struct _starpu_worker_node_data
{
	struct _starpu_worker * worker;
	struct _starpu_combined_worker * combined_worker;
	struct _starpu_worker_task_list * list;
};


static struct _starpu_worker_task_list * _starpu_worker_task_list_create(void)
{
	struct _starpu_worker_task_list * l = malloc(sizeof(*l));
	memset(l, 0, sizeof(*l));
	l->exp_len = 0.0;
	l->exp_start = l->exp_end = starpu_timing_now();
	STARPU_PTHREAD_MUTEX_INIT(&l->mutex,NULL);
	return l;
}
static struct _starpu_task_grid * _starpu_task_grid_create(void)
{
	struct _starpu_task_grid * t = malloc(sizeof(*t));
	memset(t, 0, sizeof(*t));
	return t;
}
static void _starpu_task_grid_destroy(struct _starpu_task_grid * t)
{
	free(t);
}
static void _starpu_worker_task_list_destroy(struct _starpu_worker_task_list * l)
{
	if(!l)
		return;
	STARPU_PTHREAD_MUTEX_DESTROY(&l->mutex);
	free(l);
}

//the task, ntasks, pntasks, left and right field members are set by the caller
static inline void _starpu_worker_task_list_push(struct _starpu_worker_task_list * l, struct _starpu_task_grid * t)
{
	if(l->first == NULL)
		l->first = l->last = t;
	t->down = l->last;
	l->last->up = t;
	t->up = NULL;
	l->last = t;
}

//recursively set left and right pointers to NULL
static inline void _starpu_task_grid_unset_left_right_member(struct _starpu_task_grid * t)
{
	STARPU_ASSERT(t->task == NULL);
	struct _starpu_task_grid * t_left = t->left;
	struct _starpu_task_grid * t_right = t->right;
	t->left = t->right = NULL;
	while(t_left)
	{
		STARPU_ASSERT(t_left->task == NULL);
		t = t_left;
		t_left = t_left->left;
		t->left = NULL;
		t->right = NULL;
	}
	while(t_right)
	{
		STARPU_ASSERT(t_right->task == NULL);
		t = t_right;
		t_right = t_right->right;
		t->left = NULL;
		t->right = NULL;
	}
}

static inline struct starpu_task * _starpu_worker_task_list_pop(struct _starpu_worker_task_list * l)
{
 	if(!l->first)
	{
		l->exp_start = l->exp_end = starpu_timing_now();
		l->exp_len = 0;
		return NULL;
	}
	struct _starpu_task_grid * t = l->first;

	if(t->task == NULL && t->right == NULL && t->left == NULL)
	{
		l->first = t->up;
		if(l->first)
			l->first->down = NULL;
		if(l->last == t)
			l->last = NULL;
		_starpu_task_grid_destroy(t);
		return _starpu_worker_task_list_pop(l);
	}
	
	while(t)
	{
		if(t->task)
		{
			struct starpu_task * task = t->task;
			t->task = NULL;
			int * p = t->left ? t->pntasks : &t->ntasks;
			STARPU_ATOMIC_ADD(p, -1);
			if(*p == 0)
				_starpu_task_grid_unset_left_right_member(t);
			return task;
		}
		t = t->up;
	}

	return NULL;
}





static struct _starpu_sched_node * _starpu_sched_node_worker_create(int workerid);
static struct _starpu_sched_node * _starpu_sched_node_combined_worker_create(int workerid);
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	/* we may need to take a mutex here */
	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];
	else
		return _worker_nodes[workerid] =
			(workerid < (int) starpu_worker_get_count() ?
			 _starpu_sched_node_worker_create:
			 _starpu_sched_node_combined_worker_create)(workerid);
}

struct _starpu_worker * _starpu_sched_node_worker_get_worker(struct _starpu_sched_node * worker_node)
{
	STARPU_ASSERT(_starpu_sched_node_is_worker(worker_node));
	struct _starpu_worker_node_data * data = worker_node->data;
	return data->worker;
}

int _starpu_sched_node_worker_push_task(struct _starpu_sched_node * node, struct starpu_task *task)
{
	/*this function take the worker's mutex */
	struct _starpu_worker_node_data * data = node->data;
	struct _starpu_task_grid * t = _starpu_task_grid_create();
	t->task = task;
	t->ntasks = 1;
	STARPU_PTHREAD_MUTEX_LOCK(&data->list->mutex);
	_starpu_worker_task_list_push(data->list, t);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->list->mutex);
	return 0;
}

struct starpu_task * _starpu_sched_node_worker_pop_task(struct _starpu_sched_node *node,unsigned sched_ctx_id)
{
	struct _starpu_worker_node_data * data = node->data;
	struct _starpu_worker_task_list * list = data->list;
	STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
	struct starpu_task * task =  _starpu_worker_task_list_pop(list);
	STARPU_PTHREAD_MUTEX_UNLOCK(&list->mutex);
	if(task)
	{
		starpu_push_task_end(task);
		return task;
	}
	
	struct _starpu_sched_node *father = node->fathers[sched_ctx_id];
	if(father == NULL)
		return NULL;
	task = father->pop_task(father,sched_ctx_id);
	if(!task)
		return NULL;
	if(task->cl->type == STARPU_SPMD)
	{
		int combined_workerid = starpu_combined_worker_get_id();
		if(combined_workerid < 0)
		{
			starpu_push_task_end(task);
			return task;
		}
		struct _starpu_sched_node * combined_worker_node = _starpu_sched_node_worker_get(combined_workerid);
		(void)combined_worker_node->push_task(combined_worker_node, task);
		//we have pushed a task in queue, so can make a recursive call
		return _starpu_sched_node_worker_pop_task(node, sched_ctx_id);
		
	}
	if(task)
		starpu_push_task_end(task);
	return task;
}
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *node)
{
	struct _starpu_worker * worker = _starpu_sched_node_worker_get_worker(node);
	unsigned id = worker->workerid;
	assert(_worker_nodes[id] == node);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS ; i++)
		if(node->fathers[i] != NULL)
			return;//this node is shared between several contexts
	_starpu_sched_node_destroy(node);
	_worker_nodes[id] = NULL;
}

static void available_worker(struct _starpu_sched_node * worker_node)
{
	(void) worker_node;
	
#ifndef STARPU_NON_BLOCKING_DRIVERS
	struct _starpu_worker * w = _starpu_sched_node_worker_get_worker(worker_node);
//	if(w->workerid == starpu_worker_get_id())
//		return;
	starpu_pthread_mutex_t *sched_mutex = &w->sched_mutex;
	starpu_pthread_cond_t *sched_cond = &w->sched_cond;

	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	STARPU_PTHREAD_COND_SIGNAL(sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
#endif
}

static void available_combined_worker(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(_starpu_sched_node_is_combined_worker(node));
	struct _starpu_worker_node_data * data = node->data;
	int workerid = starpu_worker_get_id();
	int i;
	for(i = 0; i < data->combined_worker->worker_size; i++)
	{
		if(i == workerid)
			continue;
		int worker = data->combined_worker->combined_workerid[i];
		starpu_pthread_mutex_t *sched_mutex;
		starpu_pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
		STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		STARPU_PTHREAD_COND_SIGNAL(sched_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
	}
}

static double estimated_transfer_length(struct _starpu_sched_node * node,
					struct starpu_task * task)
{
	STARPU_ASSERT(_starpu_sched_node_is_worker(node));
	starpu_task_bundle_t bundle = task->bundle;
	struct _starpu_worker_node_data * data = node->data;
	unsigned memory_node = data->worker ? data->worker->memory_node : data->combined_worker->memory_node;
	if(bundle)
		return starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
	else
		return starpu_task_expected_data_transfer_time(memory_node, task);
}
static double worker_estimated_finish_time(struct _starpu_worker * worker)
{
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	double sum = 0.0;
	struct starpu_task_list list = worker->local_tasks;
	struct starpu_task * task;
	for(task = starpu_task_list_front(&list);
	    task != starpu_task_list_end(&list);
	    task = starpu_task_list_next(task))
		if(!isnan(task->predicted))
		   sum += task->predicted;

	if(worker->current_task) 
	{
		struct starpu_task * t = worker->current_task;
		if(t && !isnan(t->predicted))
			sum += t->predicted/2;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
	return sum + starpu_timing_now();
}

static double combined_worker_expected_finish_time(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(_starpu_sched_node_is_combined_worker(node));
	struct _starpu_worker_node_data * data = node->data;
	struct _starpu_combined_worker * combined_worker = data->combined_worker;
	double max = 0.0;
	int i;
	for(i = 0; i < combined_worker->worker_size; i++)
	{
		data = _worker_nodes[combined_worker->combined_workerid[i]]->data;
		STARPU_PTHREAD_MUTEX_LOCK(&data->list->mutex);
		double tmp = data->list->exp_end;
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->list->mutex);
		max = tmp > max ? tmp : max;
	}
	return max;
}
static double simple_worker_expected_finish_time(struct _starpu_sched_node * node)
{
	struct _starpu_worker_node_data * data = node->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->list->mutex);
	double tmp = data->list->exp_end;
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->list->mutex);
	return tmp;
}

static struct _starpu_task_execute_preds estimated_execute_preds(struct _starpu_sched_node * node, struct starpu_task * task,
								 double (*estimated_finish_time)(struct _starpu_sched_node*))
{
	STARPU_ASSERT(_starpu_sched_node_is_worker(node));
	starpu_task_bundle_t bundle = task->bundle;
	struct _starpu_worker * worker = _starpu_sched_node_worker_get_worker(node);
			
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
			double d;
			if(bundle)
				d = starpu_task_bundle_expected_length(bundle, worker->perf_arch, nimpl);
			else
				d = starpu_task_expected_length(task, worker->perf_arch, nimpl);
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
	{
		preds.expected_finish_time = _starpu_compute_expected_time(starpu_timing_now(),
									  preds.expected_finish_time,
									  preds.expected_length,
									  preds.expected_transfer_length);

		if(bundle)
			preds.expected_power = starpu_task_bundle_expected_power(bundle, worker->perf_arch, preds.impl);
		else
			preds.expected_power = starpu_task_expected_power(task, worker->perf_arch,preds.impl);
	}

	return preds;
}

static struct _starpu_task_execute_preds combined_worker_estimated_execute_preds(struct _starpu_sched_node * node, struct starpu_task * task)
{
	return estimated_execute_preds(node,task,combined_worker_expected_finish_time);
}

static struct _starpu_task_execute_preds simple_worker_estimated_execute_preds(struct _starpu_sched_node * node, struct starpu_task * task)
{
	return estimated_execute_preds(node,task,simple_worker_expected_finish_time);
}


static double estimated_load(struct _starpu_sched_node * node)
{
	struct _starpu_worker * worker = _starpu_sched_node_worker_get_worker(node);
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
		/ starpu_worker_get_relative_speedup(_starpu_bitmap_first(node->workers));
}

static void worker_deinit_data(struct _starpu_sched_node * node)
{
	struct _starpu_worker_node_data * data = node->data;
	if(data->list)
		_starpu_worker_task_list_destroy(data->list);
	free(data);
	node->data = NULL;
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		if(_worker_nodes[i] == node)
			break;
	STARPU_ASSERT(i < STARPU_NMAXWORKERS);
	_worker_nodes[i] = NULL;
}


static int _starpu_sched_node_combined_worker_push_task(struct _starpu_sched_node * node, struct starpu_task *task)
{
	STARPU_ASSERT(_starpu_sched_node_is_combined_worker(node));
	struct _starpu_worker_node_data * data = node->data;
	STARPU_ASSERT(data->combined_worker && !data->worker);
	struct _starpu_combined_worker  * combined_worker = data->combined_worker;
	STARPU_ASSERT(combined_worker->worker_size >= 1);
	struct _starpu_task_grid * task_alias[combined_worker->worker_size];
	starpu_parallel_task_barrier_init(task, _starpu_bitmap_first(node->workers));
	task_alias[0] = _starpu_task_grid_create();
	task_alias[0]->task = task;
	task_alias[0]->left = NULL;
	task_alias[0]->ntasks = combined_worker->worker_size;
	int i;
	for(i = 1; i < combined_worker->worker_size; i++)
	{
		task_alias[i] = _starpu_task_grid_create();
		task_alias[i]->task = starpu_task_dup(task);
		task_alias[i]->left = task_alias[i-1];
		task_alias[i - 1]->right = task_alias[i];
		task_alias[i]->pntasks = &task_alias[0]->ntasks;
	}

	starpu_pthread_mutex_t * mutex_to_unlock = NULL; 
	i = 0;
	do
	{
		struct _starpu_sched_node * worker_node = _starpu_sched_node_worker_get(combined_worker->combined_workerid[i]);
		struct _starpu_worker_node_data * worker_data = worker_node->data;
		struct _starpu_worker_task_list * list = worker_data->list;
		STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
		if(mutex_to_unlock)
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex_to_unlock);
		mutex_to_unlock = &list->mutex;
		
		_starpu_worker_task_list_push(list, task_alias[i]);
		worker_node->available(worker_node);
		i++;
	}
	while(i < combined_worker->worker_size);
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex_to_unlock);
	return 0;
}

static struct _starpu_sched_node * _starpu_sched_node_worker_create(int workerid)
{
	STARPU_ASSERT(0 <=  workerid && workerid < (int) starpu_worker_get_count());

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_worker * worker = _starpu_get_worker_struct(workerid);
	if(worker == NULL)
		return NULL;
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_worker_node_data * data = malloc(sizeof(*data));
	memset(data, 0, sizeof(*data));
	data->worker = worker;
	data->list = _starpu_worker_task_list_create();
	node->data = data;
	node->push_task = _starpu_sched_node_worker_push_task;
	node->pop_task = _starpu_sched_node_worker_pop_task;
	node->estimated_execute_preds = simple_worker_estimated_execute_preds;
	node->estimated_load = estimated_load;
	node->available = available_worker;
	node->deinit_data = worker_deinit_data;
	node->workers = _starpu_bitmap_create();
	_starpu_bitmap_set(node->workers, workerid);
	_worker_nodes[workerid] = node;

#ifdef STARPU_HAVE_HWLOC
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->cpu_depth, worker->bindid);
	STARPU_ASSERT(obj);
	node->obj = obj;
#endif

	return node;
}


static struct _starpu_sched_node  * _starpu_sched_node_combined_worker_create(int workerid)
{
	STARPU_ASSERT(0 <= workerid && workerid <  STARPU_NMAXWORKERS);

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_combined_worker * combined_worker = _starpu_get_combined_worker_struct(workerid);
	if(combined_worker == NULL)
		return NULL;
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_worker_node_data * data = malloc(sizeof(*data));
	memset(data, 0, sizeof(*data));
	data->combined_worker = combined_worker;

	node->data = data;
	node->push_task = _starpu_sched_node_combined_worker_push_task;
	node->pop_task = NULL;
	node->estimated_execute_preds = combined_worker_estimated_execute_preds;
	node->estimated_load = estimated_load;
	node->available = available_combined_worker;
	node->deinit_data = worker_deinit_data;
	node->workers = _starpu_bitmap_create();
	_starpu_bitmap_set(node->workers, workerid);
	_worker_nodes[workerid] = node;

#ifdef STARPU_HAVE_HWLOC
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->cpu_depth, combined_worker->combined_workerid[0]);
	STARPU_ASSERT(obj);
	node->obj = obj;
#endif
	return node;
}

int _starpu_sched_node_is_simple_worker(struct _starpu_sched_node * node)
{
	return node->push_task == _starpu_sched_node_worker_push_task;
}
int _starpu_sched_node_is_combined_worker(struct _starpu_sched_node * node)
{
	return node->push_task == _starpu_sched_node_combined_worker_push_task;
}

int _starpu_sched_node_is_worker(struct _starpu_sched_node * node)
{
	return _starpu_sched_node_is_simple_worker(node)
		|| _starpu_sched_node_is_combined_worker(node);
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
	struct _starpu_worker_node_data * data = node->data;
	if(data->worker)
	{
		int id = data->worker->workerid;
		return  (_worker_nodes[id] == node)
			&&  node->nchilds == 0;
	}
	return 1;
}
#endif

int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node)
{
#ifndef STARPU_NO_ASSERT
	STARPU_ASSERT(_worker_consistant(worker_node));
#endif
	return _starpu_sched_node_worker_get_worker(worker_node)->workerid;
}
