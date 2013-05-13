#include <core/jobs.h>
#include <core/workers.h>
#include "node_sched.h"

static void available(struct _starpu_sched_node * node)
{
	int i;
	for(i = 0; i < node->nchilds; i++)
		node->childs[i]->available(node->childs[i]);
}
static struct starpu_task * pop_task_null(struct _starpu_sched_node * node STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	return NULL;
}

struct _starpu_sched_node * _starpu_sched_node_create(void)
{
	struct _starpu_sched_node * node = malloc(sizeof(*node));
	memset(node,0,sizeof(*node));
	STARPU_PTHREAD_MUTEX_INIT(&node->mutex,NULL);
	node->available = available;
	node->pop_task = pop_task_null;
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

void _starpu_sched_node_set_father(struct _starpu_sched_node *node,
				   struct _starpu_sched_node *father_node,
				   unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
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
	STARPU_PTHREAD_MUTEX_DESTROY(&tree->mutex);
	free(tree);
}
void _starpu_sched_node_add_child(struct _starpu_sched_node* node, struct _starpu_sched_node * child,unsigned sched_ctx_id)
{
	STARPU_ASSERT(!_starpu_sched_node_is_worker(node));
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	node->childs = realloc(node->childs, sizeof(struct _starpu_sched_node *) * (node->nchilds + 1));
	node->childs[node->nchilds] = child;
	child->fathers[sched_ctx_id] = node;
	node->nchilds++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}
void _starpu_sched_node_remove_child(struct _starpu_sched_node * node, struct _starpu_sched_node * child,unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	node->childs[pos] = node->childs[--node->nchilds];
	STARPU_ASSERT(child->fathers[sched_ctx_id] == node);
	child->fathers[sched_ctx_id] = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}


int _starpu_tree_push_task(struct starpu_task * task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_MUTEX_LOCK(&tree->mutex);
	int ret_val = tree->root->push_task(tree->root,task); 
//	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&tree->mutex);
	return ret_val;
}
struct starpu_task * _starpu_tree_pop_task(unsigned sched_ctx_id)
{
	int workerid = starpu_worker_get_id();
	struct _starpu_sched_node * node = _starpu_sched_node_worker_get(workerid);
	return node->pop_task(node, sched_ctx_id);
}



int _starpu_sched_node_can_execute_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	unsigned nimpl;
	int worker;
	STARPU_ASSERT(task);

	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = 0; worker < node->nworkers; worker++)
			if (starpu_worker_can_execute_task(worker, task, nimpl))
				return 1;
	return 0;
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
		return;
	}
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


void _starpu_tree_update_after_modification(struct _starpu_sched_tree * tree)
{
	_update_workerids_after_tree_modification(tree->root);
}
