#include "node_sched.h"

struct _starpu_work_stealing_data
{
	/* keep track of the work performed from the beginning of the algorithm to make
	 * better decisions about which queue to select when stealing or deferring work
	 */
	
	unsigned performed_total;
	unsigned last_pop_worker;
	unsigned last_push_worker;
};
//
///* little dirty hack here, fifo_nodes under the workstealing node need to know wich child they are in order to push task on themselfs
// */
//struct _starpu_fifo_ws_data {
//	struct _starpu_fifo_taskq *fifo;
//	int rank;
//};
//
//static void destroy_fifo_ws(struct _starpu_sched_node * node)
//{
//	struct _starpu_fifo_ws_data * fwsd = node->data;
//	_starpu_destroy_fifo(fwsd->fifo);
//	free(fwsd);
//	_starpu_sched_node_destroy(node);
//}
//
//static int fifo_ws_push_task(struct _starpu_sched_node * node,
//			     struct starpu_task * task)
//{
//	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
//	struct _starpu_fifo_ws_data * fwsd = node->data;
//	int ret_val =  _starpu_push_sorted_task(node->data, task);
//	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
//	return ret_val;
//}
//
//static starpu_task * fifo_ws_pop_task(struct _starpu_sched_node * node,
//				      unsigned sched_ctx_id)
//{
//	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
//	int ret_val =  _starpu_push_sorted_task(node->data, task);
//	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
//	return ret_val;
//}
//


//this function is special, when a worker call it, we want to push the task in his fifo
//because he deserve it.
int _starpu_ws_push_task(struct starpu_task *task)
{
	int workerid = starpu_get_worker_id();
	if(workerid == -1)
		return _starpu_tree_push_task(task);
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_sched_node * node =_starpu_sched_node_worker_get(workerid);
	while(node->fathers[sched_ctx_id] != NULL)
	{
		node = node->fathers[sched_ctx_id];
		if(is_my_fifo_node(node))
		{
			STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
			int ret_val =  _starpu_push_sorted_task(node->data, task);
			STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
			return ret_val;
		}
	}
	//there were a problem here, dont know what to do
	return _starpu_tree_push_task(task);

}
static struct _starpu_sched_node * fifo_ws_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_fifo_ws_data * fwsd = malloc(sizeof(struct _starpu_fifo_ws_data));
	fwsd->fifo = _starpu_create_fifo();
	node->data = fwsd;
	return node;
}


static void add_child(struct _starpu_sched_node *node,
		      struct _starpu_sched_node *child,
		      unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}
	node->childs = realloc(node->childs,
			       sizeof(struct _starpu_sched_node*)
			       * (node->nchilds + 1));
	struct _starpu_sched_node * fifo_node = _starpu_sched_node_fifo_create();
	_starpu_sched_node_set_father(fifo_node, node, sched_ctx_id);
	node->childs[node->nchilds] = fifo_node;
	node->nchilds++;
	
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);

}
static void remove_child(struct _starpu_sched_node *node,
			 struct _starpu_sched_node *child,
			 unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(*node->childs[pos]->childs == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	struct _starpu_sched_node * fifo_node = node->childs[pos];
	node->childs[pos] = node->childs[--node->nchilds];
	STARPU_ASSERT(fifo_node->fathers[sched_ctx_id] == node;
	fifo_node->fathers[sched_ctx_id] = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}



struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_work_stealing_data * wsd = malloc(sizeof(*wsd));
	wsd->performed_total = 0;
	wsd->last_pop_worker = 0;
	wsd->last_push_worker = 0;
	return node;
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	struct _starpu_work_stealing_data * wsd = node->data;
	int ret = -1;
	int start = wsd->last_pop_worker;
	
	int i;
	for(i = start + 1; i != start; (i+1)%node->nchilds)
	{
		if(!childs[i]->fifo)
			continue;
		ret = _starpu_fifo_push_sorted_task(childs[i]->fifo, task);
	}
	
}
