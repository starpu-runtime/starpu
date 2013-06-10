#ifndef __SCHED_NODE_H__
#define __SCHED_NODE_H__
#include <starpu.h>
#include <common/starpu_spinlock.h>
#include "bitmap.h"

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

struct _starpu_sched_node
{
	int (*push_task)(struct _starpu_sched_node *,
			 struct starpu_task *);
	struct starpu_task * (*pop_task)(struct _starpu_sched_node *,
					 unsigned sched_ctx_id);
	void (*available)(struct _starpu_sched_node *);
	
	/* this function is an heuristic that compute load of subtree */
	double (*estimated_load)(struct _starpu_sched_node * node);

	struct _starpu_task_execute_preds (*estimated_execute_preds)(struct _starpu_sched_node * node,
								     struct starpu_task * task);
	
	int nchilds;
	struct _starpu_sched_node ** childs;

	//the set of workers in the node's subtree
	struct _starpu_bitmap * workers;
	//is_homogeneous is 0 iff workers in the node's subtree are heterogeneous,
	//this field is set and updated automaticaly, you shouldn't write on it
	int is_homogeneous;

	void * data;
	/* may be shared by several contexts
	 * so we need several fathers
	 */
	struct _starpu_sched_node * fathers[STARPU_NMAX_SCHED_CTXS];

	/* this function is called after all childs has been set
	 */
	void (*init_data)(struct _starpu_sched_node *);
	/* this function is called to free data allocated by init_data 
	 */
	void (*deinit_data)(struct _starpu_sched_node *);

#ifdef STARPU_HAVE_HWLOC
	hwloc_obj_t obj;
#endif
};

struct _starpu_task_execute_preds
{
	enum {CANNOT_EXECUTE = 0, CALIBRATING , NO_PERF_MODEL, PERF_MODEL} state;

	/* archtype and nimpl is set to
	 * best values if state is PERF_MODEL
	 * values that needs to be calibrated if state is CALIBRATING
	 * suitable values if NO_PERF_MODEL
	 * irrevelant if CANNOT_EXECUTE
	 */
	enum starpu_perfmodel_archtype archtype;
	int impl;

	double expected_finish_time;
	double expected_length;
	double expected_transfer_length;
	double expected_power;
};


struct _starpu_sched_tree
{
	struct _starpu_sched_node * root;
	struct _starpu_bitmap * workers;
	//this lock is used to protect the scheduler during modifications of his structure
	starpu_pthread_rwlock_t lock;
};


/* allocate and initalise node field with defaults values :
 *  .pop_task make recursive call on father
 *  .estimated_finish_time  max of the recursives calls on childrens
 *  .estimated_load compute relative speedup and tasks in subtree
 *  .estimated_transfer_length  average transfer cost for all workers in the subtree
 *  .estimated_execution_length average execution cost for all workers in the subtree
 *  .available make a recursive call on childrens
 */
struct _starpu_sched_node * _starpu_sched_node_create(void);

/* free memory allocated by _starpu_sched_node_create, it does not call node->destroy_node(node)*/
void _starpu_sched_node_destroy(struct _starpu_sched_node * node);
void _starpu_sched_node_set_father(struct _starpu_sched_node *node, struct _starpu_sched_node *father_node, unsigned sched_ctx_id);
void _starpu_sched_node_add_child(struct _starpu_sched_node* node, struct _starpu_sched_node * child);
void _starpu_sched_node_remove_child(struct _starpu_sched_node * node, struct _starpu_sched_node * child);


int _starpu_sched_node_can_execute_task(struct _starpu_sched_node * node, struct starpu_task * task);
int _starpu_sched_node_can_execute_task_with_impl(struct _starpu_sched_node * node, struct starpu_task * task, unsigned nimpl);

/* no public create function for workers because we dont want to have several node_worker for a single workerid */
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid);
struct _starpu_worker * _starpu_sched_node_worker_get_worker(struct _starpu_sched_node * worker_node);
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *);

/* this function compare the available function of the node with the standard available for worker nodes*/
int _starpu_sched_node_is_worker(struct _starpu_sched_node * node);
int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node);

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void * arg STARPU_ATTRIBUTE_UNUSED);
int _starpu_sched_node_is_fifo(struct _starpu_sched_node * node);
//struct starpu_task_list  _starpu_sched_node_fifo_get_non_executable_tasks(struct _starpu_sched_node * fifo_node);

/* struct _starpu_sched_node * _starpu_sched_node_work_stealing_create(void); */
int _starpu_sched_node_is_work_stealing(struct _starpu_sched_node * node);

struct _starpu_sched_node * _starpu_sched_node_random_create(void * arg STARPU_ATTRIBUTE_UNUSED);
int _starpu_sched_node_is_random(struct _starpu_sched_node *);
struct _starpu_sched_node * _starpu_sched_node_eager_create(void * arg STARPU_ATTRIBUTE_UNUSED);

struct _starpu_sched_node * _starpu_sched_node_heft_create(void * arg STARPU_ATTRIBUTE_UNUSED);
/* this function is called to create the node wich will be used to push task when no perf model are available
 * by default, a random node is created
 */
void _starpu_sched_node_heft_set_no_model_node(struct _starpu_sched_node * heft_node,
					       struct _starpu_sched_node * (*create_no_model_node)(void * arg), void * arg);

int _starpu_sched_node_is_heft(struct _starpu_sched_node * node);

/* compute predicted_end by taking in account the case of the predicted transfer and the predicted_end overlap
 */
double _starpu_compute_expected_time(double now, double predicted_end, double predicted_length, double predicted_transfer);

void _starpu_tree_destroy(struct _starpu_sched_tree * tree, unsigned sched_ctx_id);

/* destroy node and all his child
 * except if they are shared between several contexts
 */
void _starpu_node_destroy_rec(struct _starpu_sched_node * node, unsigned sched_ctx_id);

int _starpu_tree_push_task(struct starpu_task * task);
struct starpu_task * _starpu_tree_pop_task(unsigned sched_ctx_id);
void _starpu_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void _starpu_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

void _starpu_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void _starpu_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);

/* return the bitmap of worker that are allowed to use in this scheduling context
 */

struct _starpu_bitmap * _starpu_get_worker_mask(struct starpu_task * task);

/* this function fill all the node->workers member
 */
void _starpu_set_workers_bitmaps(void);
/* this function call init data on all nodes in postfix order
 */
void _starpu_tree_call_init_data(struct _starpu_sched_tree * t);

/* push task of list lower as possible in the tree, a non null value is returned if some task couldn't be pushed
 */
int _starpu_sched_node_push_tasks_to_firsts_suitable_parent(struct _starpu_sched_node * node, struct starpu_task_list * list, int sched_ctx_id);

#endif
