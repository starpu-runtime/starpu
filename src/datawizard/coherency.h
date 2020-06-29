/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	/** To help the scheduling policies to make some decision, we
	   may keep a track of the tasks that are likely to request
	   this data on the current node.
	   It is the responsability of the scheduling _policy_ to set that
	   flag when it assigns a task to a queue, policies which do not
	   use this hint can simply ignore it.
	 */
	uint32_t requested;
	struct _starpu_data_request *request[STARPU_MAXNODES];

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
	starpu_data_handle_t **active_readonly_children; /** The currently active set of read-only children */
	unsigned nactive_readonly_children; /** Size of active_readonly_children array */
	/** Our siblings in the father partitioning */
	unsigned nsiblings; /** How many siblings */
	starpu_data_handle_t *siblings;
	unsigned sibling_index; /** indicate which child this node is from the father's perpsective (if any) */
	unsigned depth; /** what's the depth of the tree ? */

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
	unsigned readonly:1;

	/** Whether our father is currently partitioned into ourself */
	unsigned active:1;
	unsigned active_ro:1;

	/** describe the state of the data in term of coherency */
	struct _starpu_data_replicate per_node[STARPU_MAXNODES];
	struct _starpu_data_replicate *per_worker;

	struct starpu_data_interface_ops *ops;

	/** Footprint which identifies data layout */
	uint32_t footprint;

	/** where is the data home, i.e. which node it was registered from ? -1 if none yet */
	int home_node;

	/** what is the default write-through mask for that data ? */
	uint32_t wt_mask;

	/** in some case, the application may explicitly tell StarPU that a
 	 * piece of data is not likely to be used soon again */
	unsigned is_not_important;

	/** Does StarPU have to enforce some implicit data-dependencies ? */
	unsigned sequential_consistency;
	/** Is the data initialized, or a task is already submitted to initialize it */
	unsigned initialized;
	/** Can the data be pushed to the disk? */
	unsigned ooc;

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

	unsigned lazy_unregister;

#ifdef STARPU_OPENMP
	unsigned removed_from_context_hash;
#endif

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

	int partition_automatic_disabled;

	/** Application-provided coordinates. The maximum dimension (5) is
	  * relatively arbitrary. */
	unsigned dimensions;
	int coordinates[5];

	/** A generic pointer to data in the user land (could be anything and this
	 * is not manage by StarPU) */
	void *user_data;
};

/** This does not take a reference on the handle, the caller has to do it,
 * e.g. through _starpu_attempt_to_submit_data_request_from_apps()
 * detached means that the core is allowed to drop the request. The caller
 * should thus *not* take a reference since it can not know whether the request will complete
 * async means that _starpu_fetch_data_on_node will wait for completion of the request
 */
int _starpu_fetch_data_on_node(starpu_data_handle_t handle, int node, struct _starpu_data_replicate *replicate,
			       enum starpu_data_access_mode mode, unsigned detached, enum _starpu_is_prefetch is_prefetch, unsigned async,
			       void (*callback_func)(void *), void *callback_arg, int prio, const char *origin);
/** This releases a reference on the handle */
void _starpu_release_data_on_node(struct _starpu_data_state *state, uint32_t default_wt_mask,
				  struct _starpu_data_replicate *replicate);

void _starpu_update_data_state(starpu_data_handle_t handle,
			       struct _starpu_data_replicate *requesting_replicate,
			       enum starpu_data_access_mode mode);

uint32_t _starpu_get_data_refcnt(struct _starpu_data_state *state, unsigned node);

size_t _starpu_data_get_size(starpu_data_handle_t handle);
size_t _starpu_data_get_alloc_size(starpu_data_handle_t handle);

uint32_t _starpu_data_get_footprint(starpu_data_handle_t handle);

void __starpu_push_task_output(struct _starpu_job *j);
/** Version with driver trace */
void _starpu_push_task_output(struct _starpu_job *j);

void _starpu_release_nowhere_task_output(struct _starpu_job *j);

struct _starpu_worker;
STARPU_ATTRIBUTE_WARN_UNUSED_RESULT
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
								  enum starpu_data_access_mode mode, enum _starpu_is_prefetch is_prefetch,
								  unsigned async,
								  void (*callback_func)(void *), void *callback_arg, int prio, const char *origin);

void _starpu_redux_init_data_replicate(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, int workerid);
void _starpu_data_start_reduction_mode(starpu_data_handle_t handle);
void _starpu_data_end_reduction_mode(starpu_data_handle_t handle);
void _starpu_data_end_reduction_mode_terminate(starpu_data_handle_t handle);

void _starpu_data_set_unregister_hook(starpu_data_handle_t handle, _starpu_data_handle_unregister_hook func);
struct _starpu_data_replicate *get_replicate(starpu_data_handle_t handle, enum starpu_data_access_mode mode, int workerid, unsigned node);

#endif // __COHERENCY__H__
