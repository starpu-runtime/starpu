#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__
#include <starpu.h>

struct _starpu_sched_node
{
	int (*push_task)(struct _starpu_sched_node *, struct starpu_task *);
	struct starpu_task * (*pop_task)(struct _starpu_sched_node *,
					 unsigned sched_ctx_id);
	void (*available)(struct _starpu_sched_node *);
	
	/*this function only consider tasks that have a pref model, others does not count
	 * note that pushing a task not necessarily increase estimated finish time
	 */
	double (*estimated_finish_time)(struct _starpu_sched_node * node);
	/* this function is an heuritic compute subtree's load.
	 * the computation is based on number of tasks and relative speedup of processing units
	 * more revelant than estimated_finish_time() when no perf model are available
	 */
	double (*estimated_load)(struct _starpu_sched_node * node);

	//return the average of transfer length for all subtree workers
	double (*estimated_transfer_length)(struct _starpu_sched_node * node,
					    struct starpu_task * task);
	/* return data on expected length of computation, if node is heterogeneous, its an average
	 * if a calibration is not done, the arch and implementation are returned
	 */
	struct _starpu_execute_pred (*estimated_execute_length)(struct _starpu_sched_node * node,
					   struct starpu_task * task);

	int nchilds;
	struct _starpu_sched_node ** childs;

	starpu_pthread_mutex_t mutex;

	//the list of workers in the node's subtree
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;

	//is_homogeneous is 0 iff workers in the node's subtree are heterogeneous,
	//this field is set and updated automaticaly, you shouldn't write on it
	int is_homogeneous;

	void * data;
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

struct _starpu_execute_pred {
	enum {CANNOT_EXECUTE = 0, CALIBRATING , NO_PERF_MODEL, PERF_MODEL} state;
	enum starpu_perf_archtype archtype;
	int impl;
	double expected_length;
};


struct _starpu_sched_tree
{
	struct _starpu_sched_node * root;
	starpu_pthread_mutex_t mutex;
};


/* allocate and initalise node field with defaults values :
 *  .pop_task make recursive call on father
 *  .estimated_finish_time  max of the recursives calls on childrens
 *  .estimated_load compute relative speedup and tasks in subtree
 *  .estimated_transfer_length  average transfer cost for all workers in the subtree
 *  .estimated_execution_length average execution cost for all workers in the subtree
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
int _starpu_sched_node_can_execute_task_with_impl(struct _starpu_sched_node * node, struct starpu_task * task, unsigned nimpl);

/* no public create function for workers because we dont want to have several node_worker for a single workerid */
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid);
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *);

/* this function compare the available function of the node with the standard available for worker nodes*/
int _starpu_sched_node_is_worker(struct _starpu_sched_node * node);
int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node);

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void);
int _starpu_sched_node_is_fifo(struct _starpu_sched_node * node);
struct _starpu_fifo_taskq *  _starpu_sched_node_fifo_get_fifo(struct _starpu_sched_node *);

/* struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void); */
struct _starpu_sched_node * _starpu_sched_node_random_create(void);

struct _starpu_sched_node * _starpu_sched_node_eager_create(void);

struct _starpu_sched_node * _starpu_sched_node_heft_create(double alpha, double beta, double gamma, double idle_power);




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


#endif
