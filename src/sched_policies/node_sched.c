#include <core/jobs.h>
#include <core/workers.h>
#include "node_sched.h"

double _starpu_compute_expected_time(double now, double predicted_end, double predicted_length, double predicted_transfer)
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

static void available(struct _starpu_sched_node * node)
{
	(void)node;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	int i;
	for(i = 0; i < node->nchilds; i++)
		node->childs[i]->available(node->childs[i]);
#endif
}
static struct starpu_task * pop_task_node(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node->fathers[sched_ctx_id] == NULL)
		return NULL;
	else
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id], sched_ctx_id);
}


void _starpu_sched_node_set_father(struct _starpu_sched_node *node,
				   struct _starpu_sched_node *father_node,
				   unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	node->fathers[sched_ctx_id] = father_node;
}

struct starpu_task * pop_task(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid = starpu_worker_get_id();
	struct _starpu_sched_node * wn = _starpu_sched_node_worker_get(workerid);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&t->lock);
	struct starpu_task * task = wn->pop_task(wn, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
	return task;
}

int push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&t->lock);
	int ret = t->root->push_task(t->root, task);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
	return ret;
}

void _starpu_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		_starpu_bitmap_set(t->workers, workerids[i]);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}

void _starpu_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_WRLOCK(&t->lock);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		_starpu_bitmap_unset(t->workers, workerids[i]);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&t->lock);
}


void _starpu_node_destroy_rec(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node == NULL)
		return;
	struct _starpu_sched_node ** stack = NULL;
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
		struct _starpu_sched_node * n = POP();
		int i;
		for(i = 0; i < n->nchilds; i++)
		{
			struct _starpu_sched_node * child = n->childs[i];
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
		_starpu_sched_node_destroy(n);
	}
	free(stack);
}
void _starpu_tree_destroy(struct _starpu_sched_tree * tree, unsigned sched_ctx_id)
{
	_starpu_node_destroy_rec(tree->root, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_DESTROY(&tree->lock);
	free(tree);
}
void _starpu_sched_node_add_child(struct _starpu_sched_node* node, struct _starpu_sched_node * child)
{
	STARPU_ASSERT(!_starpu_sched_node_is_worker(node));
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}

	node->childs = realloc(node->childs, sizeof(struct _starpu_sched_node *) * (node->nchilds + 1));
	node->childs[node->nchilds] = child;
	node->nchilds++;
}
void _starpu_sched_node_remove_child(struct _starpu_sched_node * node, struct _starpu_sched_node * child)
{
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	node->childs[pos] = node->childs[--node->nchilds];
}

struct _starpu_bitmap * _starpu_get_worker_mask(struct starpu_task * task)
{
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(task->sched_ctx);
	return t->workers;
}

int _starpu_tree_push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->lock);
	int ret_val = tree->root->push_task(tree->root,task);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->lock);
	return ret_val;
}
struct starpu_task * _starpu_tree_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->lock);
	int workerid = starpu_worker_get_id();
	struct _starpu_sched_node * node = _starpu_sched_node_worker_get(workerid);
	struct starpu_task * task = node->pop_task(node, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->lock);
	return task;
}
/*
static double estimated_finish_time(struct _starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * c = node->childs[i];
		double tmp = c->estimated_finish_time(c);
		if( tmp > sum)
			sum = tmp;
	}
	return sum;
}
*/
static double estimated_load(struct _starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for( i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * c = node->childs[i];
		sum += c->estimated_load(c);
	}
	return sum;
}


static struct _starpu_task_execute_preds estimated_execute_preds(struct _starpu_sched_node * node, struct starpu_task * task)
{
	if(node->is_homogeneous)
		return node->childs[0]->estimated_execute_preds(node->childs[0], task);
	struct _starpu_task_execute_preds pred =
		{ 
			.state = CANNOT_EXECUTE,
			.expected_length = 0.0,
			.expected_finish_time = 0.0,
			.expected_transfer_length = 0.0,
			.expected_power = 0.0
			
		};
	int nb = 0;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_task_execute_preds tmp = node->childs[i]->estimated_execute_preds(node->childs[i], task);
		switch(tmp.state)
		{
		case CALIBRATING:
			return tmp;
			break;
		case NO_PERF_MODEL:
			if(pred.state == CANNOT_EXECUTE)
				pred = tmp;
			break;
		case PERF_MODEL:
			nb++;
			pred.expected_length += tmp.expected_length;
			pred.expected_finish_time += tmp.expected_finish_time;
			pred.expected_transfer_length += tmp.expected_transfer_length;
			pred.expected_power += tmp.expected_power;
			pred.state = PERF_MODEL;
			break;
		case CANNOT_EXECUTE:
			break;
		}
	}
	pred.expected_length /= nb;
	pred.expected_finish_time /= nb;
	pred.expected_transfer_length /= nb;
	pred.expected_power /= nb;
	return pred;
}
/*
static double estimated_transfer_length(struct _starpu_sched_node * node, struct starpu_task * task)
{
	double sum = 0.0;
	int nb = 0, i = 0;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * c = node->childs[i];
		if(_starpu_sched_node_can_execute_task(c, task))
		{
			sum += c->estimated_transfer_length(c, task);
			nb++;
		}
	}
	sum /= nb;
	return sum;
}
*/
int _starpu_sched_node_can_execute_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	unsigned nimpl;
	int worker;
	struct _starpu_bitmap * worker_mask = _starpu_get_worker_mask(task);
	STARPU_ASSERT(task);
	STARPU_ASSERT(node);
	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = _starpu_bitmap_first(node->workers);
		    -1 != worker;
		    worker = _starpu_bitmap_next(node->workers, worker))
			if (_starpu_bitmap_get(worker_mask, worker)
			    && starpu_worker_can_execute_task(worker, task, nimpl))
				return 1;

	return 0;
}

int _starpu_sched_node_can_execute_task_with_impl(struct _starpu_sched_node * node, struct starpu_task * task, unsigned nimpl)
{

	struct _starpu_bitmap * worker_mask = _starpu_get_worker_mask(task);
	int worker;
	STARPU_ASSERT(task);
	STARPU_ASSERT(nimpl < STARPU_MAXIMPLEMENTATIONS);
	for(worker = _starpu_bitmap_first(node->workers);
	    worker != -1;
	    worker = _starpu_bitmap_next(node->workers, worker))
		if (_starpu_bitmap_get(worker_mask, worker)
		    && starpu_worker_can_execute_task(worker, task, nimpl))
			return 1;
	return 0;

}

void take_node_and_does_nothing(struct _starpu_sched_node * node STARPU_ATTRIBUTE_UNUSED)
{
}

struct _starpu_sched_node * _starpu_sched_node_create(void)
{
	struct _starpu_sched_node * node = malloc(sizeof(*node));
	memset(node,0,sizeof(*node));
	node->workers = _starpu_bitmap_create();
	node->available = available;
	node->init_data = take_node_and_does_nothing;
	node->deinit_data = take_node_and_does_nothing;
	node->pop_task = pop_task_node;
	node->estimated_load = estimated_load;
	node->estimated_execute_preds = estimated_execute_preds;

	return node;
}
void _starpu_sched_node_destroy(struct _starpu_sched_node *node)
{
	node->deinit_data(node);
	int i,j;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * child = node->childs[i];
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(child->fathers[i] == node)
				child->fathers[i] = NULL;

	}
	free(node->childs);
	_starpu_bitmap_destroy(node->workers);

	free(node);
}


static void set_is_homogeneous(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(_starpu_bitmap_cardinal(node->workers) > 0);
	if(_starpu_bitmap_cardinal(node->workers) == 1)
		node->is_homogeneous = 1;
	int worker = _starpu_bitmap_first(node->workers);
	uint32_t last_worker = _starpu_get_worker_struct(worker)->worker_mask;
	for(;
	    worker != -1;
	    worker = _starpu_bitmap_next(node->workers, worker))
		
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


static void add_worker_bit(struct _starpu_sched_node * node, int worker)
{
	STARPU_ASSERT(node);
	_starpu_bitmap_set(node->workers, worker);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(node->fathers[i])
		{
			add_worker_bit(node->fathers[i], worker);
			set_is_homogeneous(node->fathers[i]);
		}
}

void _starpu_set_workers_bitmaps(void)
{
	unsigned worker;	
	for(worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		struct _starpu_sched_node * worker_node = _starpu_sched_node_worker_get(worker);
		add_worker_bit(worker_node, worker);
	}
}

static void helper_starpu_call_init_data(struct _starpu_sched_node *node)
{
	int i;
	for(i = 0; i < node->nchilds; i++)
		helper_starpu_call_init_data(node->childs[i]);
	if(!node->data)
		node->init_data(node);
}

void _starpu_tree_call_init_data(struct _starpu_sched_tree * t)
{
	helper_starpu_call_init_data(t->root);
}


static int push_task_to_first_suitable_parent(struct _starpu_sched_node * node, struct starpu_task * task, int sched_ctx_id)
{
	if(node == NULL || node->fathers[sched_ctx_id] == NULL)
		return 1;

//	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	
	struct _starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(_starpu_sched_node_can_execute_task(father,task))
		return father->push_task(father, task);
	else
		return push_task_to_first_suitable_parent(father, task, sched_ctx_id);
}


int _starpu_sched_node_push_tasks_to_firsts_suitable_parent(struct _starpu_sched_node * node, struct starpu_task_list *list, int sched_ctx_id)
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

