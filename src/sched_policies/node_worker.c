#include <starpu_sched_node.h>
#include <core/workers.h>
#include <float.h>

static struct starpu_sched_node * _worker_nodes[STARPU_NMAXWORKERS];

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
 * its possible that a _starpu_task_grid wont have task, because it have been
 * poped by a worker.
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
 * this API is a little asymmetric : struct _starpu_task_grid are allocated by the caller and freed by the data structure
 *
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


/* list->exp_start, list->exp_len, list-exp_end and list->ntasks
 * are updated by starpu_sched_node_worker_push_task(node, task) and pre_exec_hook
 */

struct _starpu_worker_task_list
{
	double exp_start, exp_len, exp_end;
	struct _starpu_task_grid *first, *last;
	unsigned ntasks;
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
	if(l)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&l->mutex);
		free(l);
	}
}

/* the task, ntasks, pntasks, left and right field members are set by the caller */
static inline void _starpu_worker_task_list_push(struct _starpu_worker_task_list * l, struct _starpu_task_grid * t)
{
	STARPU_ASSERT(t->task);
	if(l->first == NULL)
		l->first = l->last = t;
	t->down = l->last;
	l->last->up = t;
	t->up = NULL;
	l->last = t;
	l->ntasks++;

	double predicted = t->task->predicted;
	double predicted_transfer = t->task->predicted_transfer;

	/* Sometimes workers didn't take the tasks as early as we expected */
	l->exp_start = STARPU_MAX(l->exp_start, starpu_timing_now());
	l->exp_end = l->exp_start + l->exp_len;

	if ((starpu_timing_now() + predicted_transfer) < l->exp_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0.0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer = (starpu_timing_now() + predicted_transfer) - l->exp_end;
	}

	if(!isnan(predicted_transfer))
	{
		l->exp_end += predicted_transfer;
		l->exp_len += predicted_transfer;
	}

	if(!isnan(predicted))
	{
		l->exp_end += predicted;
		l->exp_len += predicted;
	}

	t->task->predicted = predicted;
	t->task->predicted_transfer = predicted_transfer;
}

/* recursively set left and right pointers to NULL */
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
			(void) STARPU_ATOMIC_ADD(p, -1);
			if(*p == 0)
				_starpu_task_grid_unset_left_right_member(t);
			l->ntasks--;
			if(!isnan(task->predicted))
			{
				l->exp_len -= task->predicted;
				l->exp_end = l->exp_start + l->exp_len;
			}

			return task;
		}
		t = t->up;
	}

	return NULL;
}





static struct starpu_sched_node * starpu_sched_node_worker_create(int workerid);
static struct starpu_sched_node * starpu_sched_node_combined_worker_create(int workerid);
struct starpu_sched_node * starpu_sched_node_worker_get(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	/* we may need to take a mutex here */
	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];
	else
	{
		struct starpu_sched_node * node;
		if(workerid < (int) starpu_worker_get_count())
			node = starpu_sched_node_worker_create(workerid);
		else
			node = starpu_sched_node_combined_worker_create(workerid);
		_worker_nodes[workerid] = node;
		return node;
	}
}

struct _starpu_worker * starpu_sched_node_worker_get_worker(struct starpu_sched_node * worker_node)
{
	STARPU_ASSERT(starpu_sched_node_is_simple_worker(worker_node));
	struct _starpu_worker_node_data * data = worker_node->data;
	return data->worker;
}
struct _starpu_combined_worker * starpu_sched_node_combined_worker_get_combined_worker(struct starpu_sched_node * worker_node)
{
	STARPU_ASSERT(starpu_sched_node_is_combined_worker(worker_node));
	struct _starpu_worker_node_data * data = worker_node->data;
	return data->combined_worker;
}

enum starpu_perfmodel_archtype starpu_sched_node_worker_get_perf_arch(struct starpu_sched_node * worker_node)
{
	STARPU_ASSERT(starpu_sched_node_is_worker(worker_node));
	if(starpu_sched_node_is_simple_worker(worker_node))
		return starpu_sched_node_worker_get_worker(worker_node)->perf_arch;
	else
		return starpu_sched_node_combined_worker_get_combined_worker(worker_node)->perf_arch;
}


int starpu_sched_node_worker_push_task(struct starpu_sched_node * node, struct starpu_task *task)
{
	/*this function take the worker's mutex */
	struct _starpu_worker_node_data * data = node->data;
	struct _starpu_task_grid * t = _starpu_task_grid_create();
	t->task = task;
	t->ntasks = 1;

	task->workerid = starpu_bitmap_first(node->workers);
#if 0 /* dead lock problem */
	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(task->workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&data->list->mutex);
	_starpu_worker_task_list_push(data->list, t);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->list->mutex);
	node->available(node);
	return 0;
}

struct starpu_task * starpu_sched_node_worker_pop_task(struct starpu_sched_node *node,unsigned sched_ctx_id)
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

	struct starpu_sched_node *father = node->fathers[sched_ctx_id];
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
		struct starpu_sched_node * combined_worker_node = starpu_sched_node_worker_get(combined_workerid);
		(void)combined_worker_node->push_task(combined_worker_node, task);
		//we have pushed a task in queue, so can make a recursive call
		return starpu_sched_node_worker_pop_task(node, sched_ctx_id);

	}
	if(task)
		starpu_push_task_end(task);
	return task;
}
void starpu_sched_node_worker_destroy(struct starpu_sched_node *node)
{
	struct _starpu_worker * worker = starpu_sched_node_worker_get_worker(node);
	unsigned id = worker->workerid;
	assert(_worker_nodes[id] == node);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS ; i++)
		if(node->fathers[i] != NULL)
			return;//this node is shared between several contexts
	starpu_sched_node_destroy(node);
	_worker_nodes[id] = NULL;
}

static void simple_worker_available(struct starpu_sched_node * worker_node)
{
	(void) worker_node;

#ifndef STARPU_NON_BLOCKING_DRIVERS
	struct _starpu_worker * w = starpu_sched_node_worker_get_worker(worker_node);
	if(w->workerid == starpu_worker_get_id())
		return;
	starpu_pthread_mutex_t *sched_mutex = &w->sched_mutex;
	starpu_pthread_cond_t *sched_cond = &w->sched_cond;
	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	STARPU_PTHREAD_COND_SIGNAL(sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
#endif
}

static void combined_worker_available(struct starpu_sched_node * node)
{
	(void) node;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	STARPU_ASSERT(starpu_sched_node_is_combined_worker(node));
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
#endif
}

static double estimated_transfer_length(struct starpu_sched_node * node,
					struct starpu_task * task)
{
	STARPU_ASSERT(starpu_sched_node_is_worker(node));
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

static double combined_worker_estimated_end(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_combined_worker(node));
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
static double simple_worker_estimated_end(struct starpu_sched_node * node)
{
	struct _starpu_worker_node_data * data = node->data;
	STARPU_PTHREAD_MUTEX_LOCK(&data->list->mutex);
	double tmp = data->list->exp_end;
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->list->mutex);
	return tmp;
}



static double simple_worker_estimated_load(struct starpu_sched_node * node)
{
	struct _starpu_worker * worker = starpu_sched_node_worker_get_worker(node);
	int nb_task = 0;
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	struct starpu_task_list list = worker->local_tasks;
	struct starpu_task * task;
	for(task = starpu_task_list_front(&list);
	    task != starpu_task_list_end(&list);
	    task = starpu_task_list_next(task))
		nb_task++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
	struct _starpu_worker_node_data * d = node->data;
	struct _starpu_worker_task_list * l = d->list;
	int ntasks_in_fifo = l ? l->ntasks : 0;
	return (double) (nb_task + ntasks_in_fifo)
		/ starpu_worker_get_relative_speedup(starpu_bitmap_first(node->workers));
}

static double combined_worker_estimated_load(struct starpu_sched_node * node)
{
	struct _starpu_worker_node_data * d = node->data;
	struct _starpu_combined_worker * c = d->combined_worker;
	double load = 0;
	int i;
	for(i = 0; i < c->worker_size; i++)
	{
		struct starpu_sched_node * n = starpu_sched_node_worker_get(c->combined_workerid[i]);
		load += n->estimated_load(n);
	}
	return load;
}

static int starpu_sched_node_combined_worker_push_task(struct starpu_sched_node * node, struct starpu_task *task)
{
	STARPU_ASSERT(starpu_sched_node_is_combined_worker(node));
	struct _starpu_worker_node_data * data = node->data;
	STARPU_ASSERT(data->combined_worker && !data->worker);
	struct _starpu_combined_worker  * combined_worker = data->combined_worker;
	STARPU_ASSERT(combined_worker->worker_size >= 1);
	struct _starpu_task_grid * task_alias[combined_worker->worker_size];
	starpu_parallel_task_barrier_init(task, starpu_bitmap_first(node->workers));
	task_alias[0] = _starpu_task_grid_create();
	task_alias[0]->task = starpu_task_dup(task);
	task_alias[0]->task->workerid = combined_worker->combined_workerid[0];
	task_alias[0]->left = NULL;
	task_alias[0]->ntasks = combined_worker->worker_size;
	int i;
	for(i = 1; i < combined_worker->worker_size; i++)
	{
		task_alias[i] = _starpu_task_grid_create();
		task_alias[i]->task = starpu_task_dup(task);
		task_alias[i]->task->workerid = combined_worker->combined_workerid[i];
		task_alias[i]->left = task_alias[i-1];
		task_alias[i - 1]->right = task_alias[i];
		task_alias[i]->pntasks = &task_alias[0]->ntasks;
	}

	starpu_pthread_mutex_t * mutex_to_unlock = NULL;
	i = 0;
	do
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(combined_worker->combined_workerid[i]);
		struct _starpu_worker_node_data * worker_data = worker_node->data;
		struct _starpu_worker_task_list * list = worker_data->list;
		STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
		if(mutex_to_unlock)
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex_to_unlock);
		mutex_to_unlock = &list->mutex;

		_starpu_worker_task_list_push(list, task_alias[i]);
		i++;
	}
	while(i < combined_worker->worker_size);
	
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex_to_unlock);

	int workerid = starpu_worker_get_id();
	if(-1 == workerid)
	{
		node->available(node);
	}
	else
	{
		starpu_pthread_mutex_t *worker_sched_mutex;
		starpu_pthread_cond_t *worker_sched_cond;
		starpu_worker_get_sched_condition(workerid, &worker_sched_mutex, &worker_sched_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(worker_sched_mutex);

		/* wake up all other workers of combined worker */
		for(i = 0; i < combined_worker->worker_size; i++)
		{
			struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(combined_worker->combined_workerid[i]);
			worker_node->available(worker_node);
		}

		node->available(node);

		STARPU_PTHREAD_MUTEX_LOCK(worker_sched_mutex);
	}
	return 0;
}

static struct starpu_sched_node * starpu_sched_node_worker_create(int workerid)
{
	STARPU_ASSERT(0 <=  workerid && workerid < (int) starpu_worker_get_count());

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_worker * worker = _starpu_get_worker_struct(workerid);
	if(worker == NULL)
		return NULL;
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_worker_node_data * data = malloc(sizeof(*data));
	memset(data, 0, sizeof(*data));
	data->worker = worker;
	data->list = _starpu_worker_task_list_create();
	node->data = data;
	node->push_task = starpu_sched_node_worker_push_task;
	node->pop_task = starpu_sched_node_worker_pop_task;
	node->estimated_end = simple_worker_estimated_end;
	node->estimated_load = simple_worker_estimated_load;
	node->available = simple_worker_available;
	starpu_bitmap_set(node->workers, workerid);
	starpu_bitmap_or(node->workers_in_ctx, node->workers);
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


static struct starpu_sched_node  * starpu_sched_node_combined_worker_create(int workerid)
{
	STARPU_ASSERT(0 <= workerid && workerid <  STARPU_NMAXWORKERS);

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_combined_worker * combined_worker = _starpu_get_combined_worker_struct(workerid);
	if(combined_worker == NULL)
		return NULL;
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_worker_node_data * data = malloc(sizeof(*data));
	memset(data, 0, sizeof(*data));
	data->combined_worker = combined_worker;

	node->data = data;
	node->push_task = starpu_sched_node_combined_worker_push_task;
	node->pop_task = NULL;
	node->estimated_end = combined_worker_estimated_end;
	node->estimated_load = combined_worker_estimated_load;
	node->available = combined_worker_available;
	starpu_bitmap_set(node->workers, workerid);
	starpu_bitmap_or(node->workers_in_ctx, node->workers);
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

int starpu_sched_node_is_simple_worker(struct starpu_sched_node * node)
{
	return node->push_task == starpu_sched_node_worker_push_task;
}
int starpu_sched_node_is_combined_worker(struct starpu_sched_node * node)
{
	return node->push_task == starpu_sched_node_combined_worker_push_task;
}

int starpu_sched_node_is_worker(struct starpu_sched_node * node)
{
	return starpu_sched_node_is_simple_worker(node)
		|| starpu_sched_node_is_combined_worker(node);
}



#ifndef STARPU_NO_ASSERT
static int _worker_consistant(struct starpu_sched_node * node)
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

int starpu_sched_node_worker_get_workerid(struct starpu_sched_node * worker_node)
{
#ifndef STARPU_NO_ASSERT
	STARPU_ASSERT(_worker_consistant(worker_node));
#endif
	STARPU_ASSERT(1 == starpu_bitmap_cardinal(worker_node->workers));
	return starpu_bitmap_first(worker_node->workers);
}


static struct _starpu_worker_task_list * _worker_get_list(void)
{
	int workerid = starpu_worker_get_id();
	STARPU_ASSERT(0 <= workerid && workerid < (int) starpu_worker_get_count());
	struct _starpu_worker_node_data * d = starpu_sched_node_worker_get(workerid)->data;
	return d->list;
}


void starpu_sched_node_worker_pre_exec_hook(struct starpu_task * task)
{
	if(!isnan(task->predicted))
	{
		struct _starpu_worker_task_list * list = _worker_get_list();
		STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
		STARPU_ASSERT(list->ntasks != 0);
		list->ntasks--;
		if(!task->execute_on_a_specific_worker)
			list->exp_len = STARPU_MAX(list->exp_len - task->predicted, 0.0);

		list->exp_start = starpu_timing_now() + task->predicted;
		if(list->ntasks == 0)
		{
			list->exp_end = list->exp_start;
			list->exp_end = 0.0;
		}
		else
			list->exp_end = list->exp_start + list->exp_len;
		STARPU_PTHREAD_MUTEX_UNLOCK(&list->mutex);
	}
}
void starpu_sched_node_worker_post_exec_hook(struct starpu_task * task)
{
	if(task->execute_on_a_specific_worker)
		return;
	struct _starpu_worker_task_list * list = _worker_get_list();
	STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
	list->exp_start = starpu_timing_now();
	list->exp_end = list->exp_start + list->exp_len;
	STARPU_PTHREAD_MUTEX_UNLOCK(&list->mutex);
}

static void starpu_sched_node_worker_push_task_notify(struct starpu_task *task, int workerid, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{

	struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(workerid);
	/* dont work with parallel tasks */
	if(starpu_sched_node_is_combined_worker(worker_node))
	   return;

	struct _starpu_worker_node_data * d = worker_node->data;
	struct _starpu_worker_task_list * list = d->list;
	/* Compute the expected penality */
	enum starpu_perfmodel_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	unsigned memory_node = starpu_worker_get_memory_node(workerid);

	double predicted = starpu_task_expected_length(task, perf_arch,
						       starpu_task_get_implementation(task));

	double predicted_transfer = starpu_task_expected_data_transfer_time(memory_node, task);

	/* Update the predictions */
	STARPU_PTHREAD_MUTEX_LOCK(&list->mutex);
	/* Sometimes workers didn't take the tasks as early as we expected */
	list->exp_start = STARPU_MAX(list->exp_start, starpu_timing_now());
	list->exp_end = list->exp_start + list->exp_len;

	/* If there is no prediction available, we consider the task has a null length */
	if (!isnan(predicted_transfer))
	{
		if (starpu_timing_now() + predicted_transfer < list->exp_end)
		{
			/* We may hope that the transfer will be finshied by
			 * the start of the task. */
			predicted_transfer = 0;
		}
		else
		{
			/* The transfer will not be finished by then, take the
			 * remainder into account */
			predicted_transfer = (starpu_timing_now() + predicted_transfer) - list->exp_end;
		}
		task->predicted_transfer = predicted_transfer;
		list->exp_end += predicted_transfer;
		list->exp_len += predicted_transfer;
	}

	/* If there is no prediction available, we consider the task has a null length */
	if (!isnan(predicted))
	{
		task->predicted = predicted;
		list->exp_end += predicted;
		list->exp_len += predicted;
	}

	list->ntasks++;

	STARPU_PTHREAD_MUTEX_UNLOCK(&list->mutex);
}
