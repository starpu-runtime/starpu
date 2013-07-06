#include <core/jobs.h>
#include <core/workers.h>
#include <starpu_sched_node.h>
#include <starpu_thread_util.h>
#include <float.h>

double starpu_sched_compute_expected_time(double now, double predicted_end, double predicted_length, double predicted_transfer)
{

	if (now + predicted_transfer < predicted_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer += now;
		predicted_transfer -= predicted_end;
	}
	if(!isnan(predicted_transfer))
	{
		predicted_end += predicted_transfer;
		predicted_length += predicted_transfer;
	}

	if(!isnan(predicted_length))
		predicted_end += predicted_length;
	return predicted_end;
}

static void available(struct starpu_sched_node * node)
{
	(void)node;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	int i;
	for(i = 0; i < node->nchilds; i++)
		node->childs[i]->available(node->childs[i]);
#endif
}
static struct starpu_task * pop_task_node(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node->fathers[sched_ctx_id] == NULL)
		return NULL;
	else
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id], sched_ctx_id);
}


void starpu_sched_node_set_father(struct starpu_sched_node *node,
				   struct starpu_sched_node *father_node,
				   unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	node->fathers[sched_ctx_id] = father_node;
}

struct starpu_task * pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid = starpu_worker_get_id();
	struct starpu_sched_node * wn = starpu_sched_node_worker_get(workerid);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&t->lock);
	struct starpu_task * task = wn->pop_task(wn, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
	return task;
}

int push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&t->lock);
	int ret = t->root->push_task(t->root, task);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
	return ret;
}

void _update_worker_bits(struct starpu_sched_node * node, struct starpu_bitmap * workers_in_ctx)
{
	if(starpu_sched_node_is_worker(node))
		return;
	starpu_bitmap_unset_and(node->workers_in_ctx, node->workers, workers_in_ctx);
	int i;
	for(i = 0; i < node->nchilds; i++)
		_update_worker_bits(node->childs[i], workers_in_ctx);
}


void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_set(t->workers, workerids[i]);
	_update_worker_bits(t->root, t->workers);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}

void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_unset(t->workers, workerids[i]);
	_update_worker_bits(t->root, t->workers);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}


void starpu_sched_node_destroy_rec(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node == NULL)
		return;
	struct starpu_sched_node ** stack = NULL;
	int top = -1;
#define PUSH(n) do{							\
		stack = realloc(stack, sizeof(*stack) * (top + 2));	\
		stack[++top] = n;}while(0)
#define POP() stack[top--]
#define EMPTY() (top == -1)
//we want to delete all subtrees exept if a pointer in fathers point in an other tree
//ie an other context

	node->fathers[sched_ctx_id] = NULL;
	int shared = 0;
	{
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
			if(node->fathers[i] != NULL)
				shared = 1;
	}
	if(!shared)
		PUSH(node);
	while(!EMPTY())
	{
		struct starpu_sched_node * n = POP();
		int i;
		for(i = 0; i < n->nchilds; i++)
		{
			struct starpu_sched_node * child = n->childs[i];
			int j;
			shared = 0;
			STARPU_ASSERT(child->fathers[sched_ctx_id] == n);
			child->fathers[sched_ctx_id] = NULL;
			for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			{
				if(child->fathers[j] != NULL)//child is shared
					shared = 1;
			}
			if(!shared)//if not shared we want to destroy it and his childs
				PUSH(child);
		}
		starpu_sched_node_destroy(n);
	}
	free(stack);
}
struct starpu_sched_tree * starpu_sched_tree_create(void)
{
	struct starpu_sched_tree * t = malloc(sizeof(*t));
	memset(t, 0, sizeof(*t));
	t->workers = starpu_bitmap_create();
	STARPU_PTHREAD_RWLOCK_INIT(&t->lock,NULL);
	return t;
}

void starpu_sched_tree_destroy(struct starpu_sched_tree * tree, unsigned sched_ctx_id)
{
	if(tree->root)
		starpu_sched_node_destroy_rec(tree->root, sched_ctx_id);
	starpu_bitmap_destroy(tree->workers);
	STARPU_PTHREAD_RWLOCK_DESTROY(&tree->lock);
	free(tree);
}
void starpu_sched_node_add_child(struct starpu_sched_node* node, struct starpu_sched_node * child)
{
	STARPU_ASSERT(!starpu_sched_node_is_worker(node));
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}

	node->childs = realloc(node->childs, sizeof(struct starpu_sched_node *) * (node->nchilds + 1));
	node->childs[node->nchilds] = child;
	node->nchilds++;
}
void starpu_sched_node_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	node->childs[pos] = node->childs[--node->nchilds];
}

struct starpu_bitmap * _starpu_get_worker_mask(struct starpu_task * task)
{
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(task->sched_ctx);
	return t->workers;
}

int starpu_sched_tree_push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->lock);
	int ret_val = tree->root->push_task(tree->root,task);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->lock);
	return ret_val;
}
struct starpu_task * starpu_sched_tree_pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->lock);
	int workerid = starpu_worker_get_id();
	struct starpu_sched_node * node = starpu_sched_node_worker_get(workerid);
	struct starpu_task * task = node->pop_task(node, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->lock);
	return task;
}
/*
static double estimated_finish_time(struct starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		double tmp = c->estimated_finish_time(c);
		if( tmp > sum)
			sum = tmp;
	}
	return sum;
}
*/
static double estimated_load(struct starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for( i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		sum += c->estimated_load(c);
	}
	return sum;
}


static double _starpu_sched_node_estimated_end_min(struct starpu_sched_node * node)
{
	double min = DBL_MAX;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		double tmp = node->childs[i]->estimated_end(node->childs[i]);
		if(tmp < min)
			min = tmp;
	}
	return min;
}

int STARPU_WARN_UNUSED_RESULT starpu_sched_node_execute_preds(struct starpu_sched_node * node, struct starpu_task * task, double * length)
{
	int can_execute = 0;
	starpu_task_bundle_t bundle = task->bundle;
	double len = DBL_MAX;
	

	int workerid;
	for(workerid = starpu_bitmap_first(node->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(node->workers_in_ctx, workerid))
	{
		enum starpu_perfmodel_archtype archtype = starpu_worker_get_perf_archtype(workerid);
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				double d;
				can_execute = 1;
				if(bundle)
					d = starpu_task_bundle_expected_length(bundle, archtype, nimpl);
				else
					d = starpu_task_expected_length(task, archtype, nimpl);
				if(isnan(d))
				{
					*length = d;
					return can_execute;
						
				}
				if(_STARPU_IS_ZERO(d) && !can_execute)
				{
					can_execute = 1;
					continue;
				}
				if(d < len)
				{
					len = d;
				}
			}
		}
		if(node->is_homogeneous)
			break;
	}

	if(len == DBL_MAX) /* we dont have perf model */
		len = 0.0; 
	if(length)
		*length = len;
	return can_execute;
}

double starpu_sched_node_transfer_length(struct starpu_sched_node * node, struct starpu_task * task)
{
	int nworkers = starpu_bitmap_cardinal(node->workers_in_ctx);
	double sum = 0.0;
	int worker;
	for(worker = starpu_bitmap_first(node->workers_in_ctx);
	    worker != -1;
	    worker = starpu_bitmap_next(node->workers_in_ctx, worker))
	{
		unsigned memory_node  = starpu_worker_get_memory_node(worker);
		if(task->bundle)
		{
			sum += starpu_task_bundle_expected_data_transfer_time(task->bundle,memory_node);
		}
		else
		{
			sum += starpu_task_expected_data_transfer_time(memory_node, task);
			//sum += starpu_task_expected_conversion_time(task, starpu_worker_get_perf_archtype(worker), impl ?)
		}
	}
	return sum / nworkers;
}


/*
static double estimated_transfer_length(struct starpu_sched_node * node, struct starpu_task * task)
{
	double sum = 0.0;
	int nb = 0, i = 0;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		if(starpu_sched_node_can_execute_task(c, task))
		{
			sum += c->estimated_transfer_length(c, task);
			nb++;
		}
	}
	sum /= nb;
	return sum;
}
*/
int starpu_sched_node_can_execute_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	unsigned nimpl;
	int worker;
	STARPU_ASSERT(task);
	STARPU_ASSERT(node);
	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = starpu_bitmap_first(node->workers_in_ctx);
		    -1 != worker;
		    worker = starpu_bitmap_next(node->workers_in_ctx, worker))
			if (starpu_worker_can_execute_task(worker, task, nimpl)
			     || starpu_combined_worker_can_execute_task(worker, task, nimpl))
			    return 1;
	return 0;
}


void take_node_and_does_nothing(struct starpu_sched_node * node STARPU_ATTRIBUTE_UNUSED)
{
}

struct starpu_sched_node * starpu_sched_node_create(void)
{
	struct starpu_sched_node * node = malloc(sizeof(*node));
	memset(node,0,sizeof(*node));
	node->workers = starpu_bitmap_create();
	node->workers_in_ctx = starpu_bitmap_create();
	node->available = available;
	node->add_child = starpu_sched_node_add_child;
	node->remove_child = starpu_sched_node_remove_child;
	node->pop_task = pop_task_node;
	node->estimated_load = estimated_load;
	node->estimated_end = _starpu_sched_node_estimated_end_min;
	return node;
}
void starpu_sched_node_destroy(struct starpu_sched_node *node)
{
	int i,j;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * child = node->childs[i];
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(child->fathers[i] == node)
				child->fathers[i] = NULL;

	}
	free(node->childs);
	starpu_bitmap_destroy(node->workers);
	starpu_bitmap_destroy(node->workers_in_ctx);
	free(node);
}


static void set_is_homogeneous(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_bitmap_cardinal(node->workers) > 0);
	if(starpu_bitmap_cardinal(node->workers) == 1)
		node->is_homogeneous = 1;
	int worker = starpu_bitmap_first(node->workers);
	uint32_t last_worker = _starpu_get_worker_struct(worker)->worker_mask;
	for(;
	    worker != -1;
	    worker = starpu_bitmap_next(node->workers, worker))
		
	{
		if(last_worker != _starpu_get_worker_struct(worker)->worker_mask)
		{
			node->is_homogeneous = 0;
			return;
		}
		last_worker = _starpu_get_worker_struct(worker)->worker_mask;
	}
	node->is_homogeneous = 1;
}



void starpu_sched_node_init_rec(struct starpu_sched_node * node)
{
	if(starpu_sched_node_is_worker(node))
		return;
	int i;
	for(i = 0; i < node->nchilds; i++)
		starpu_sched_node_init_rec(node->childs[i]);

	for(i = 0; i < node->nchilds; i++)
		starpu_bitmap_or(node->workers, node->childs[i]->workers);
	set_is_homogeneous(node);
}



static void _init_add_worker_bit(struct starpu_sched_node * node, int worker)
{
	STARPU_ASSERT(node);
	starpu_bitmap_set(node->workers, worker);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(node->fathers[i])
		{
			_init_add_worker_bit(node->fathers[i], worker);
			set_is_homogeneous(node->fathers[i]);
		}
}

void _starpu_set_workers_bitmaps(void)
{
	unsigned worker;	
	for(worker = 0; worker < starpu_worker_get_count() + starpu_combined_worker_get_count(); worker++)
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(worker);
		_init_add_worker_bit(worker_node, worker);
	}
}


static int push_task_to_first_suitable_parent(struct starpu_sched_node * node, struct starpu_task * task, int sched_ctx_id)
{
	if(node == NULL || node->fathers[sched_ctx_id] == NULL)
		return 1;

	struct starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(starpu_sched_node_can_execute_task(father,task))
		return father->push_task(father, task);
	else
		return push_task_to_first_suitable_parent(father, task, sched_ctx_id);
}


int starpu_sched_node_push_tasks_to_firsts_suitable_parent(struct starpu_sched_node * node, struct starpu_task_list *list, int sched_ctx_id)
{
	while(!starpu_task_list_empty(list))
	{
		struct starpu_task * task = starpu_task_list_pop_front(list);
		int res = push_task_to_first_suitable_parent(node, task, sched_ctx_id);
		if(res)
		{
			starpu_task_list_push_front(list,task);
			return res;
		}
	}
	return 0;
}

