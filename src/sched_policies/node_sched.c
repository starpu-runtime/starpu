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
	int i;
	for(i = 0; i < node->nchilds; i++)
		node->childs[i]->available(node->childs[i]);
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
	STARPU_ASSERT(father_node == NULL ?
		      node->fathers[sched_ctx_id] != NULL :
		      node->fathers[sched_ctx_id] == NULL);
	node->fathers[sched_ctx_id] = father_node;
}

struct starpu_task * pop_task(unsigned sched_ctx_id)
{
	//struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid = starpu_worker_get_id();
	struct _starpu_sched_node * wn = _starpu_sched_node_worker_get(workerid);
	return wn->pop_task(wn, sched_ctx_id);
}

int push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	return t->root->push_task(t->root, task);
}

void _starpu_node_destroy_rec(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
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
		n->destroy_node(n);
	}
	free(stack);
}
void _starpu_tree_destroy(struct _starpu_sched_tree * tree, unsigned sched_ctx_id)
{
	_starpu_node_destroy_rec(tree->root, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_DESTROY(&tree->mutex);
	free(tree);
}
void _starpu_sched_node_add_child(struct _starpu_sched_node* node, struct _starpu_sched_node * child,unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
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
void _starpu_sched_node_remove_child(struct _starpu_sched_node * node, struct _starpu_sched_node * child,unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
//	STARPU_PTHREAD_RWLOCK_WRLOCK(&node->mutex);
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	node->childs[pos] = node->childs[--node->nchilds];
//	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
}


int _starpu_tree_push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->mutex);
	int ret_val = tree->root->push_task(tree->root,task); 
//	starpu_push_task_end(task);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->mutex);
	return ret_val;
}
struct starpu_task * _starpu_tree_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid = starpu_worker_get_id();
	struct _starpu_sched_node * node = _starpu_sched_node_worker_get(workerid);
	STARPU_PTHREAD_RWLOCK_RDLOCK(&tree->mutex);
	struct starpu_task * task = node->pop_task(node, sched_ctx_id);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&tree->mutex);
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
	STARPU_ASSERT(task);
	STARPU_ASSERT(node);
	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = 0; worker < node->nworkers; worker++)
			if (starpu_worker_can_execute_task(node->workerids[worker], task, nimpl))
				return 1;
	return 0;
}

int _starpu_sched_node_can_execute_task_with_impl(struct _starpu_sched_node * node, struct starpu_task * task, unsigned nimpl)
{

	int worker;
	STARPU_ASSERT(task);
	STARPU_ASSERT(nimpl < STARPU_MAXIMPLEMENTATIONS);
	for(worker = 0; worker < node->nworkers; worker++)
		if (starpu_worker_can_execute_task(worker, task, nimpl))
			return 1;
	return 0;

}

struct _starpu_sched_node * _starpu_sched_node_create(void)
{
	struct _starpu_sched_node * node = malloc(sizeof(*node));
	memset(node,0,sizeof(*node));
	STARPU_PTHREAD_RWLOCK_INIT(&node->mutex,NULL);
	node->available = available;
	node->pop_task = pop_task_node;
	node->estimated_load = estimated_load;
	node->estimated_execute_preds = estimated_execute_preds;
	node->destroy_node = _starpu_sched_node_destroy;
	node->add_child = _starpu_sched_node_add_child;
	node->remove_child = _starpu_sched_node_remove_child;

	return node;
}
void _starpu_sched_node_destroy(struct _starpu_sched_node *node)
{
	int i,j;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * child = node->childs[i];
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(child->fathers[i] == node)
				child->fathers[i] = NULL;

	}
	free(node->childs);
	free(node);
}


static int is_homogeneous(int * workerids, int nworkers)
{
	if(nworkers == 0)
		return 1;
	int i = 0;
	uint32_t last_worker = _starpu_get_worker_struct(workerids[i])->worker_mask;
	for(i = 1; i < nworkers; i++)
	{
		if(last_worker != _starpu_get_worker_struct(workerids[i])->worker_mask)
		   return 0;
		last_worker = _starpu_get_worker_struct(workerids[i])->worker_mask;
	}
	return 1;
}


static int in_tab(int elem, int * tab, int size)
{
	for(size--;size >= 0; size--)
		if(tab[size] == elem)
			return 1;
	return 0;
}
static void _update_workerids_after_tree_modification(struct _starpu_sched_node * node)
{
	if(_starpu_sched_node_is_worker(node))
	{
		node->nworkers = 1;
		node->workerids[0] =  _starpu_sched_node_worker_get_workerid(node);
	}
	else
	{
		int i;
		node->nworkers = 0;
		for(i = 0; i < node->nchilds; i++)
		{
			struct _starpu_sched_node * child = node->childs[i];
			_update_workerids_after_tree_modification(child);
			int j;
			for(j = 0; j < child->nworkers; j++)
			{
				int id = child->workerids[j];
				if(!in_tab(id, node->workerids, node->nworkers))
					node->workerids[node->nworkers++] = id;
			}
		}
	}
	node->is_homogeneous = is_homogeneous(node->workerids, node->nworkers);
}


void _starpu_tree_update_after_modification(struct _starpu_sched_tree * tree)
{
	_update_workerids_after_tree_modification(tree->root);
}
