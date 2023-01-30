/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __COHERENCY__H__
#define __COHERENCY__H__

/** @file */

#include <starpu.h>
#include <common/config.h>

#include <common/starpu_spinlock.h>
#include <common/rwlock.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <common/list.h>

#include <datawizard/interfaces/data_interface.h>
#include <datawizard/datastats.h>
#include <datawizard/memstats.h>
#include <datawizard/data_request.h>

#pragma GCC visibility push(hidden)

enum _starpu_cache_state
{
	STARPU_OWNER,
	STARPU_SHARED,
	STARPU_INVALID
};

/** this should contain the information relative to a given data replicate  */
struct _starpu_data_replicate
{
	starpu_data_handle_t handle;

	/** describe the actual data layout, as manipulated by data interfaces in *_interface.c */
	void *data_interface;

	/** How many requests or tasks are currently working with this replicate */
	int refcnt;

	char memory_node;

	/** describes the state of the local data in term of coherency */
	enum _starpu_cache_state	state: 2;

	/** A buffer that is used for SCRATCH or reduction cannnot be used with
	 * filters. */
	unsigned relaxed_coherency:2;

	/** We may need to initialize the replicate with some value before using it. */
	unsigned initialized:1;

	/** is the data locally allocated ? */
	unsigned allocated:1;
	/** was it automatically allocated ? (else it's the application-provided
	 * buffer, don't ever try to free it!) */
	/** perhaps the allocation was perform higher in the hiearchy
	 * for now this is just translated into !automatically_allocated
	 * */
	unsigned automatically_allocated:1;

	/** is the write side enabled on the mapping?
	 * This is important for drivers which may actually make a copy instead
	 * of a map.
	 *
	 * Only meaningful when mapped != STARPU_UNMAPPED */
	unsigned map_write:1;

#define STARPU_UNMAPPED -1
	/** >= 0 when the data just a mapping of a replicate from that memory node,
	 * otherwise STARPU_UNMAPPED */
	int mapped;

	/** To help the scheduling policies to make some decision, we
	   may keep a track of the tasks that are likely to request
	   this data on the current node.
	   It is the responsability of the scheduling _policy_ to set that
	   flag when it assigns a task to a queue, policies which do not
	   use this hint can simply ignore it.
	 */
	uint32_t requested;

	/** This tracks the list of requests to provide the value */
	struct _starpu_data_request *request[STARPU_MAXNODES];
	/** This points to the last entry of request, to easily append to the list */
	struct _starpu_data_request *last_request[STARPU_MAXNODES];

	/* Which request is loading data here */
	struct _starpu_data_request *load_request;

	/** The number of prefetches that we made for this replicate for various tasks
	 * This is also the number of tasks that we will wait to see use the mc before
	 * we attempt to evict it.
	 */
	unsigned nb_tasks_prefetch;

	/** Pointer to memchunk for LRU strategy */
	struct _starpu_mem_chunk * mc;
};

struct _starpu_data_requester_prio_list;

struct _starpu_jobid_list
{
	unsigned long id;
	struct _starpu_jobid_list *next;
};

/** This structure describes a simply-linked list of task */
struct _starpu_task_wrapper_list
{
	struct starpu_task *task;
	struct _starpu_task_wrapper_list *next;
};

/** This structure describes a doubly-linked list of task */
struct _starpu_task_wrapper_dlist
{
	struct starpu_task *task;
	struct _starpu_task_wrapper_dlist *next;
	struct _starpu_task_wrapper_dlist *prev;
};

extern int _starpu_has_not_important_data;

typedef void (*_starpu_data_handle_unregister_hook)(starpu_data_handle_t);

/** This is initialized in both _starpu_register_new_data and _starpu_data_partition */
struct _starpu_data_state
{
	int magic;
	struct _starpu_data_requester_prio_list req_list;
	/** the number of requests currently in the scheduling engine (not in
	 * the req_list anymore), i.e. the number of holders of the
	 * current_mode rwlock */
	unsigned refcnt;
	/** whether we are already unlocking data requests */
	unsigned unlocking_reqs;
	/** Current access mode. Is always either STARPU_R, STARPU_W,
	 * STARPU_SCRATCH or STARPU_REDUX, but never a combination such as
	 * STARPU_RW. */
	enum starpu_data_access_mode current_mode;
	/** protect meta data */
	struct _starpu_spinlock header_lock;

	/** Condition to make application wait for all transfers before freeing handle */
	/** busy_count is the number of handle->refcnt, handle->per_node[*]->refcnt, number of starpu_data_requesters, and number of tasks that have released it but are still registered on the implicit data dependency lists. */
	/** Core code which releases busy_count has to call
	 * _starpu_data_check_not_busy to let starpu_data_unregister proceed */
	unsigned busy_count;
	/** Is starpu_data_unregister waiting for busy_count? */
	unsigned busy_waiting;
	starpu_pthread_mutex_t busy_mutex;
	starpu_pthread_cond_t busy_cond;

	/** In case we user filters, the handle may describe a sub-data */
	struct _starpu_data_state *root_handle; /** root of the tree */
	struct _starpu_data_state *father_handle; /** father of the node, NULL if the current node is the root */
	starpu_data_handle_t *active_children; /** The currently active set of read-write children */
	unsigned active_nchildren;
	starpu_data_handle_t **active_readonly_children; /** The currently active set of read-only children */
	unsigned *active_readonly_nchildren; /** Size of active_readonly_children[i] array */
	unsigned nactive_readonly_children; /** Size of active_readonly_children and active_readonly_nchildren arrays. Actual use is given by 'partitioned' */
	/** Our siblings in the father partitioning */
	unsigned nsiblings; /** How many siblings */
	starpu_data_handle_t *siblings;
	unsigned sibling_index; /** indicate which child this node is from the father's perpsective (if any) */
	unsigned depth; /** what's the depth of the tree ? */

#ifdef STARPU_BUBBLE
	starpu_pthread_mutex_t unpartition_mutex;
#endif

	/** Synchronous partitioning */
	starpu_data_handle_t children;
	unsigned nchildren;
	/** How many partition plans this handle has */
	unsigned nplans;
	/** Switch codelet for asynchronous partitioning */
	struct starpu_codelet *switch_cl;
	/** size of dyn_nodes recorded in switch_cl */
	unsigned switch_cl_nparts;
	/** Whether a partition plan is currently submitted and the
	 * corresponding unpartition has not been yet
	 *
	 * Or the number of partition plans currently submitted in readonly
	 * mode.
	 */
	unsigned partitioned;
	/** Whether a partition plan is currently submitted in readonly mode */
	unsigned part_readonly:1;

	/** Whether our father is currently partitioned into ourself */
	unsigned active:1;
	unsigned active_ro:1;

	/** describe the state of the data in term of coherency
	 * This is execution-time state. */
	struct _starpu_data_replicate per_node[STARPU_MAXNODES];
	struct _starpu_data_replicate *per_worker;

	struct starpu_data_interface_ops *ops;

	/** Footprint which identifies data layout */
	uint32_t footprint;

	/** where is the data home, i.e. which node it was registered from ? -1 if none yet */
	int home_node;

	/** what is the default write-through mask for that data ? */
	uint32_t wt_mask;

	/** for a readonly handle, the number of times that we have returned again the
	    same handle and thus the number of times we have to ignore unregistration requests */
	unsigned aliases;
	/** for a non-readonly handle, a readonly-only duplicate, that we can
	    return from starpu_data_dup_ro */
	starpu_data_handle_t readonly_dup;
	/** for a readonly handle, the non-readonly handle that is referencing
	    is in its readonly_dup field. */
	starpu_data_handle_t readonly_dup_of;

	/* The following bitfields are set from the application submission thread */

	/** in some case, the application may explicitly tell StarPU that a
	 * piece of data is not likely to be used soon again */
	unsigned is_not_important:1;

	/** Does StarPU have to enforce some implicit data-dependencies ? */
	unsigned sequential_consistency:1;
	/** Is the data initialized, or a task is already submitted to initialize it
	 * This is submission-time initialization state. */
	unsigned initialized:1;
	/** Whether we shall not ever write to this handle, thus allowing various optimizations */
	unsigned readonly:1;
	/** Can the data be pushed to the disk? */
	unsigned ooc:1;

#ifdef STARPU_OPENMP
	unsigned removed_from_context_hash:1;
#endif

	/* The following field is set by StarPU at execution time */

	/** Whether lazy unregistration was requested throught starpu_data_unregister_submit */
	unsigned char lazy_unregister;

	/** This lock should protect any operation to enforce
	 * sequential_consistency */
	starpu_pthread_mutex_t sequential_consistency_mutex;

	/** The last submitted task (or application data request) that declared
	 * it would modify the piece of data ? Any task accessing the data in a
	 * read-only mode should depend on that task implicitely if the
	 * sequential_consistency flag is enabled. */
	enum starpu_data_access_mode last_submitted_mode;
	struct starpu_task *last_sync_task;
	struct _starpu_task_wrapper_dlist last_submitted_accessors;

	/** If FxT is enabled, we keep track of "ghost dependencies": that is to
	 * say the dependencies that are not needed anymore, but that should
	 * appear in the post-mortem DAG. For instance if we have the sequence
	 * f(Aw) g(Aw), and that g is submitted after the termination of f, we
	 * want to have f->g appear in the DAG even if StarPU does not need to
	 * enforce this dependency anymore.*/
	unsigned last_submitted_ghost_sync_id_is_valid;
	unsigned long last_submitted_ghost_sync_id;
	struct _starpu_jobid_list *last_submitted_ghost_accessors_id;

	/** protected by sequential_consistency_mutex */
	struct _starpu_task_wrapper_list *post_sync_tasks;
	unsigned post_sync_tasks_cnt;

	/*
	 *	Reductions
	 */

	/** During reduction we need some specific methods: redux_func performs
	 * the reduction of an interface into another one (eg. "+="), and init_func
	 * initializes the data interface to a default value that is stable by
	 * reduction (eg. 0 for +=). */
	struct starpu_codelet *redux_cl;
	struct starpu_codelet *init_cl;
	void *redux_cl_arg;
	void *init_cl_arg;

	/** Are we currently performing a reduction on that handle ? If so the
	 * reduction_refcnt should be non null until there are pending tasks
	 * that are performing the reduction. */
	unsigned reduction_refcnt;

	/** List of requesters that are specific to the pending reduction. This
	 * list is used when the requests in the req_list list are frozen until
	 * the end of the reduction. */
	struct _starpu_data_requester_prio_list reduction_req_list;

	starpu_data_handle_t *reduction_tmp_handles;

	/** Final request for write invalidation */
	struct _starpu_data_request *write_invalidation_req;

	/** Used for MPI */
	void *mpi_data;

	_starpu_memory_stats_t memory_stats;

	unsigned int mf_node; //XXX

	/** hook to be called when unregistering the data */
	_starpu_data_handle_unregister_hook unregister_hook;

	struct starpu_arbiter *arbiter;
	/** This is protected by the arbiter mutex */
	struct _starpu_data_requester_prio_list arbitered_req_list;

	/** Data maintained by schedulers themselves */
	/** Last worker that took this data in locality mode, or -1 if nobody
	 * took it yet */
	int last_locality;

	/** Application-provided coordinates. The maximum dimension (5) is
	  * relatively arbitrary. */
	unsigned dimensions;
	int coordinates[5];

	/** A generic pointer to data in the user land (could be anything and this
	 * is not manage by StarPU) */
	void *user_data;

	/** A generic pointer to data in the scheduler (could be anything and this
	 * is managed by the scheduler) */
	void *sched_data;
};

/** This does not take a reference on the handle, the caller has to do it,
 * e.g. through _starpu_attempt_to_submit_data_request_from_apps()
 * detached means that the core is allowed to drop the request. The caller
 * should thus *not* take a reference since it can not know whether the request will complete
 * async means that _starpu_fetch_data_on_node will wait for completion of the request
 */
int _starpu_fetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate,
			       enum starpu_data_access_mode mode, unsigned detached,
			       struct starpu_task *task, enum starpu_is_prefetch is_prefetch, unsigned async,
			       void (*callback_func)(void *), void *callback_arg, int prio, const char *origin);
/** This releases a reference on the handle */
void _starpu_release_data_on_node(struct _starpu_data_state *state, uint32_t default_wt_mask,
				  enum starpu_data_access_mode down_to_mode,
				  struct _starpu_data_replicate *replicate);

void _starpu_update_data_state(starpu_data_handle_t handle,
			       struct _starpu_data_replicate *requesting_replicate,
			       enum starpu_data_access_mode mode);

uint32_t _starpu_get_data_refcnt(struct _starpu_data_state *state, unsigned node);

size_t _starpu_data_get_size(starpu_data_handle_t handle);
size_t _starpu_data_get_alloc_size(starpu_data_handle_t handle);
starpu_ssize_t _starpu_data_get_max_size(starpu_data_handle_t handle);

uint32_t _starpu_data_get_footprint(starpu_data_handle_t handle);

void __starpu_push_task_output(struct _starpu_job *j);
/** Version with driver trace */
void _starpu_push_task_output(struct _starpu_job *j);

struct _starpu_worker;
STARPU_ATTRIBUTE_WARN_UNUSED_RESULT
/** Fetch the data parameters for task \p task
 * Setting \p async to 1 allows to only start the fetches, and call
 * \p _starpu_fetch_task_input_tail later when the transfers are finished */
int _starpu_fetch_task_input(struct starpu_task *task, struct _starpu_job *j, int async);
void _starpu_fetch_task_input_tail(struct starpu_task *task, struct _starpu_job *j, struct _starpu_worker *worker);
void _starpu_fetch_nowhere_task_input(struct _starpu_job *j);

int _starpu_select_src_node(struct _starpu_data_state *state, unsigned destination);
int _starpu_determine_request_path(starpu_data_handle_t handle,
				  int src_node, int dst_node,
				  enum starpu_data_access_mode mode, int max_len,
				  unsigned *src_nodes, unsigned *dst_nodes,
				  unsigned *handling_nodes, unsigned write_invalidation);

/** is_prefetch is whether the DSM may drop the request (when there is not enough memory for instance
 * async is whether the caller wants a reference on the last request, to be
 * able to wait for it (which will release that reference).
 */
struct _starpu_data_request *_starpu_create_request_to_fetch_data(starpu_data_handle_t handle,
								  struct _starpu_data_replicate *dst_replicate,
								  enum starpu_data_access_mode mode,
								  struct starpu_task *task, enum starpu_is_prefetch is_prefetch,
								  unsigned async,
								  void (*callback_func)(void *), void *callback_arg, int prio, const char *origin);

void _starpu_redux_init_data_replicate(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, int workerid);
void _starpu_data_start_reduction_mode(starpu_data_handle_t handle);
void _starpu_data_end_reduction_mode(starpu_data_handle_t handle, int priority);
void _starpu_data_end_reduction_mode_terminate(starpu_data_handle_t handle);

void _starpu_data_unmap(starpu_data_handle_t handle, unsigned node);

void _starpu_data_set_unregister_hook(starpu_data_handle_t handle, _starpu_data_handle_unregister_hook func) STARPU_ATTRIBUTE_VISIBILITY_DEFAULT;

#pragma GCC visibility pop

#endif // __COHERENCY__H__
