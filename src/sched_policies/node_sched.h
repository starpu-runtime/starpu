#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__
#include <starpu.h>

struct _starpu_sched_node
{
	int (*push_task)(struct _starpu_sched_node *, struct starpu_task *);
	struct starpu_task * (*pop_task)(struct _starpu_sched_node *, unsigned sched_ctx_id);
	void (*available)(struct _starpu_sched_node *);

	void * data;

	int nchilds;
	struct _starpu_sched_node ** childs;

	starpu_pthread_mutex_t mutex;

	//the list of workers in the node's subtree
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;

	/* may be shared by several contexts
	 * so we need several fathers
	 */
	struct _starpu_sched_node * fathers[STARPU_NMAX_SCHED_CTXS];
	


	void (*add_child)(struct _starpu_sched_node *node,
			  struct _starpu_sched_node *child,
			  unsigned sched_ctx_id);
	void (*remove_child)(struct _starpu_sched_node *node,
			     struct _starpu_sched_node *child,
			     unsigned sched_ctx_id);
	/* this function is called to free node (it must call _starpu_sched_node_destroy(node));
	*/
	void (*destroy_node)(struct _starpu_sched_node *);
};


struct _starpu_sched_tree
{
	struct _starpu_sched_node * root;
	starpu_pthread_mutex_t mutex;
};


/* allocate and initalise node field with defaults values :
 *  .pop_task return NULL
 *  .available make a recursive call on childrens
 *  .destroy_node  call _starpu_sched_node_destroy
 *  .update_nchilds a function that does nothing
 *  .{add,remove}_child functions that simply add/remove the child and update the .fathers field of child
 */
struct _starpu_sched_node * _starpu_sched_node_create(void);

/* free memory allocated by _starpu_sched_node_create, it does not call node->destroy_node(node)*/
void _starpu_sched_node_destroy(struct _starpu_sched_node * node);

void _starpu_sched_node_set_father(struct _starpu_sched_node *node, struct _starpu_sched_node *father_node, unsigned sched_ctx_id);

/* those two function call node->update_nchilds after the child was added or removed */
void _starpu_sched_node_add_child(struct _starpu_sched_node* node, struct _starpu_sched_node * child, unsigned sched_ctx_id);
void _starpu_sched_node_remove_child(struct _starpu_sched_node * node, struct _starpu_sched_node * child, unsigned sched_ctx_id);


int _starpu_sched_node_can_execute_task(struct _starpu_sched_node * node, struct starpu_task * task);


//no public create function for workers because we dont want to have several node_worker for a single workerid
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid);
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *);

/*this function assume that workers are the only leafs */
int _starpu_sched_node_is_worker(struct _starpu_sched_node * node);
int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node);

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void);
struct _starpu_fifo_taskq *  _starpu_sched_node_fifo_get_fifo(struct _starpu_sched_node *);

//struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void);
struct _starpu_sched_node * _starpu_sched_node_random_create(void);

struct _starpu_sched_node * _starpu_sched_node_eager_create(void);



void _starpu_tree_destroy(struct _starpu_sched_tree * tree, unsigned sched_ctx_id);

/* destroy node and all his child
 * except if they are shared between several contexts
 */
void _starpu_node_destroy_rec(struct _starpu_sched_node * node, unsigned sched_ctx_id);

int _starpu_tree_push_task(struct starpu_task * task);
struct starpu_task * _starpu_tree_pop_task(unsigned sched_ctx_id);

//this function must be called after all modification of tree
void _starpu_tree_update_after_modification(struct _starpu_sched_tree * tree);
;
//extern struct starpu_sched_policy _starpu_sched_tree_eager_policy;
//extern struct starpu_sched_policy _starpu_sched_tree_random_policy;
#endif
