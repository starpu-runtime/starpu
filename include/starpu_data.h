/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021-2021  Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu_util.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Data_Management Data Management
   @brief Data management facilities provided by StarPU. We show how
   to use existing data interfaces in \ref API_Data_Interfaces, but
   developers can design their own data interfaces if required.
   @{
*/

struct _starpu_data_state;

/**
   StarPU uses ::starpu_data_handle_t as an opaque handle to manage a
   piece of data. Once a piece of data has been registered to StarPU,
   it is associated to a ::starpu_data_handle_t which keeps track of
   the state of the piece of data over the entire machine, so that we
   can maintain data consistency and locate data replicates for
   instance. See \ref DataInterface for more details.
*/
typedef struct _starpu_data_state *starpu_data_handle_t;

/**
   Describe a StarPU data access mode

   Note: when adding a flag here, update
   _starpu_detect_implicit_data_deps_with_handle

   Note: other STARPU_* values in include/starpu_task_util.h
*/
enum starpu_data_access_mode
{
	STARPU_R	 = (1 << 0),		  /**< read-only mode */
	STARPU_W	 = (1 << 1),		  /**< write-only mode */
	STARPU_RW	 = (STARPU_R | STARPU_W), /**< read-write mode. Equivalent to ::STARPU_R|::STARPU_W  */
	STARPU_SCRATCH	 = (1 << 2),		  /**< A temporary buffer is allocated
						     for the task, but StarPU does not
						     enforce data consistency---i.e. each
						     device has its own buffer,
						     independently from each other (even
						     for CPUs), and no data transfer is
						     ever performed. This is useful for
						     temporary variables to avoid
						     allocating/freeing buffers inside
						     each task. Currently, no behavior is
						     defined concerning the relation with
						     the ::STARPU_R and ::STARPU_W modes
						     and the value provided at
						     registration --- i.e., the value of
						     the scratch buffer is undefined at
						     entry of the codelet function.  It
						     is being considered for future
						     extensions at least to define the
						     initial value.  For now, data to be
						     used in ::STARPU_SCRATCH mode should
						     be registered with node -1 and a
						     <c>NULL</c> pointer, since the value
						     of the provided buffer is simply
						     ignored for now.

						     See \ref ScratchData for more details.
						  */
	STARPU_REDUX	 = (1 << 3),		  /**< Reduction mode.
						     StarPU will allocate on the fly a per-worker
						     buffer, so that various tasks that access the
						     same data in ::STARPU_REDUX mode can execute
						     in parallel. When a task accesses the
						     data without ::STARPU_REDUX, StarPU will
						     automatically reduce the different contributions.

						     Codelets contributing to these reductions
						     with ::STARPU_REDUX must be registered with
						     ::STARPU_RW | ::STARPU_COMMUTE access modes.

						     See \ref DataReduction for more details.
						  */
	STARPU_COMMUTE	 = (1 << 4),		  /**<  ::STARPU_COMMUTE can be passed
						     along ::STARPU_W or ::STARPU_RW to
						     express that StarPU can let tasks
						     commute, which is useful e.g. when
						     bringing a contribution into some
						     data, which can be done in any order
						     (but still require sequential
						     consistency against reads or
						     non-commutative writes).

						     See \ref DataCommute for more details.
						  */
	STARPU_SSEND	 = (1 << 5),		  /**< used in starpu_mpi_task_insert() to
						     specify the data has to be sent using
						     a synchronous and non-blocking mode
						     (see starpu_mpi_issend())
						  */
	STARPU_LOCALITY	 = (1 << 6),		  /**< used to tell the scheduler which
						     data is the most important for the
						     task, and should thus be used to
						     try to group tasks on the same core
						     or cache, etc. For now only the ws
						     and lws schedulers take this flag
						     into account, and only when rebuild
						     with \c USE_LOCALITY flag defined in
						     the
						     src/sched_policies/work_stealing_policy.c
						     source code.

						     TODO add extended description in documentation.
						  */
	STARPU_MPI_REDUX = (1 << 7),		  /**< Inter-node reduction only.
						     This is similar to ::STARPU_REDUX, except that
						     StarPU will allocate a per-node buffer only,
						     i.e. parallelism will be achieved between
						     nodes, but not within each node. This is
						     useful when the per-worker buffers allocated
						     with ::STARPU_REDUX consume too much memory.

						     See \ref MPIMpiRedux for more details.
						  */
	STARPU_NOPLAN = (1 << 8),                 /**< Disable automatic submission of asynchronous
						     partitioning/unpartitioning, only use internally by StarPU
						  */
	STARPU_UNMAP = (1 << 9),                  /**< Request unmapping the destination replicate, only use internally by StarPU
						   */
	STARPU_NOFOOTPRINT = (1 << 10),           /**< Ignore this data for the footprint computation. See \ref ScratchData
						   */
	STARPU_NONE	 = (1 << 11),		  /**< todo */
	STARPU_ACCESS_MODE_MAX = (1 << 12)        /**< The purpose of ::STARPU_ACCESS_MODE_MAX is to
						     be the maximum of this enum.
						  */
};

struct starpu_data_interface_ops;

/**
   Set the name of the data, to be shown in various profiling tools.
   See \ref CreatingAGanttDiagram for more details.
*/
void starpu_data_set_name(starpu_data_handle_t handle, const char *name);

/**
   Set the coordinates of the data, to be shown in various profiling
   tools. \p dimensions is the size of the \p dims array. This can be
   for instance the tile coordinates within a big matrix. See \ref CreatingAGanttDiagram for more details.
*/
void starpu_data_set_coordinates_array(starpu_data_handle_t handle, unsigned dimensions, int dims[]);

/**
   Set the coordinates of the data, to be shown in various profiling
   tools. \p dimensions is the number of subsequent \c int parameters.
   This can be for instance the tile coordinates within a big matrix. See \ref CreatingAGanttDiagram for more details.
*/
void starpu_data_set_coordinates(starpu_data_handle_t handle, unsigned dimensions, ...);

/**
   Get the coordinates of the data, as set by a previous call to
   starpu_data_set_coordinates_array() or starpu_data_set_coordinates()
   \p dimensions is the size of the \p dims array.
   This returns the actual number of returned coordinates.
   See \ref CreatingAGanttDiagram for more details.
*/
unsigned starpu_data_get_coordinates_array(starpu_data_handle_t handle, unsigned dimensions, int dims[]);

/**
   Unregister a data \p handle from StarPU. If the data was
   automatically allocated by StarPU because the home node was -1, all
   automatically allocated buffers are freed. Otherwise, a valid copy
   of the data is put back into the home node in the buffer that was
   initially registered. Using a data handle that has been
   unregistered from StarPU results in an undefined behaviour. In case
   we do not need to update the value of the data in the home node, we
   can use the function starpu_data_unregister_no_coherency() instead.
   See \ref TaskSubmission for more details.
*/
void starpu_data_unregister(starpu_data_handle_t handle);

/**
   Similar to starpu_data_unregister(), except that StarPU does not
   put back a valid copy into the home node, in the buffer that was
   initially registered. See \ref DataManagementAllocation for more details.
*/
void starpu_data_unregister_no_coherency(starpu_data_handle_t handle);

/**
   Destroy the data \p handle once it is no longer needed by any
   submitted task. No coherency is provided.

   This is not safe to call starpu_data_unregister_submit() on a handle that
   comes from the registration of a non-NULL application home buffer, since the
   moment when the unregistration will happen is unknown to the
   application. Only calling starpu_shutdown() allows to be sure that the data
   was really unregistered. See \ref TemporaryData for more details.
*/
void starpu_data_unregister_submit(starpu_data_handle_t handle);

/**
   Deinitialize all replicates of the data \p handle immediately. After
   data deinitialization, the first access to \p handle must be performed
   in ::STARPU_W mode. Accessing an deinitialized data in ::STARPU_R
   mode results in undefined behaviour. See \ref DataManagementAllocation for more details.
*/
void starpu_data_deinitialize(starpu_data_handle_t handle);

/**
   Submit deinitialization of the data \p handle after completion of
   previously submitted tasks. See \ref DataManagementAllocation for more details.
*/
void starpu_data_deinitialize_submit(starpu_data_handle_t handle);

/**
   Destroy all replicates of the data \p handle immediately. After
   data invalidation, the first access to \p handle must be performed
   in ::STARPU_W mode. Accessing an invalidated data in ::STARPU_R
   mode results in undefined behaviour. See \ref DataManagementAllocation for more details.

   This is the same as starpu_data_deinitialize(), plus explicitly releasing the buffers.
*/
void starpu_data_invalidate(starpu_data_handle_t handle);

/**
   Submit invalidation of the data \p handle after completion of
   previously submitted tasks. See \ref DataManagementAllocation for more details.

   This is the same as starpu_data_deinitialize_submit(), plus explicitly releasing the buffers.
*/
void starpu_data_invalidate_submit(starpu_data_handle_t handle);

/**
   Specify that the data \p handle can be discarded without impacting
   the application.
*/
void starpu_data_advise_as_important(starpu_data_handle_t handle, unsigned is_important);

/**
   @name Access registered data from the application
   @{
*/

/**
   This macro can be used to acquire data, but not require it to be
   available on a given node, only enforce R/W dependencies. This can
   for instance be used to wait for tasks which produce the data, but
   without requesting a fetch to the main memory.
*/
#define STARPU_ACQUIRE_NO_NODE -1

/**
   Similar to ::STARPU_ACQUIRE_NO_NODE, but will lock the data on all
   nodes, preventing them from being evicted for instance. This is
   mostly useful inside StarPU only.
*/
#define STARPU_ACQUIRE_NO_NODE_LOCK_ALL -2

/**
   The application must call this function prior to accessing
   registered data from main memory outside tasks. StarPU ensures that
   the application will get an up-to-date copy of \p handle in main
   memory located where the data was originally registered, and that
   all concurrent accesses (e.g. from tasks) will be consistent with
   the access mode specified with \p mode. starpu_data_release() must
   be called once the application no longer needs to access the piece
   of data. Note that implicit data dependencies are also enforced by
   starpu_data_acquire(), i.e. starpu_data_acquire() will wait for all
   tasks scheduled to work on the data, unless they have been disabled
   explicitly by calling
   starpu_data_set_default_sequential_consistency_flag() or
   starpu_data_set_sequential_consistency_flag().
   starpu_data_acquire() is a blocking call, so that it cannot be
   called from tasks or from their callbacks (in that case,
   starpu_data_acquire() returns <c>-EDEADLK</c>). Upon successful
   completion, this function returns 0. See \ref DataAccess for more details.
*/
int starpu_data_acquire(starpu_data_handle_t handle, enum starpu_data_access_mode mode);

/**
   Similar to starpu_data_acquire(), except that the data will be
   available on the given memory node instead of main memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can
   be used instead of an explicit node number. See \ref DataAccess for more details.
*/
int starpu_data_acquire_on_node(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode);

/**
   Asynchronous equivalent of starpu_data_acquire(). When the data
   specified in \p handle is available in the access \p mode, the \p
   callback function is executed. The application may access
   the requested data during the execution of \p callback. The \p callback
   function must call starpu_data_release() once the application no longer
   needs to access the piece of data. Note that implicit data
   dependencies are also enforced by starpu_data_acquire_cb() in case they
   are not disabled. Contrary to starpu_data_acquire(), this function is
   non-blocking and may be called from task callbacks. Upon successful
   completion, this function returns 0. See \ref DataAccess for more details.
*/
int starpu_data_acquire_cb(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);

/**
   Similar to starpu_data_acquire_cb(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be
   used instead of an explicit node number. See \ref DataAccess for more details.
*/
int starpu_data_acquire_on_node_cb(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);

/**
   Similar to starpu_data_acquire_cb() with the possibility of
   enabling or disabling data dependencies.
   When the data specified in \p handle is available in the access
   \p mode, the \p callback function is executed. The application may access
   the requested data during the execution of this \p callback. The \p callback
   function must call starpu_data_release() once the application no longer
   needs to access the piece of data. Note that implicit data
   dependencies are also enforced by starpu_data_acquire_cb_sequential_consistency() in case they
   are not disabled specifically for the given \p handle or by the parameter \p sequential_consistency.
   Similarly to starpu_data_acquire_cb(), this function is
   non-blocking and may be called from task callbacks. Upon successful
   completion, this function returns 0. See \ref DataAccess for more details.
*/
int starpu_data_acquire_cb_sequential_consistency(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);

/**
   Similar to starpu_data_acquire_cb_sequential_consistency(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be used instead of an
   explicit node number. See \ref DataAccess for more details.
*/
int starpu_data_acquire_on_node_cb_sequential_consistency(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);

/**
   Similar to starpu_data_acquire_on_node_cb_sequential_consistency(),
   except that the \e pre_sync_jobid and \e post_sync_jobid parameters can be used
   to retrieve the jobid of the synchronization tasks. \e pre_sync_jobid happens
   just before the acquisition, and \e post_sync_jobid happens just after the
   release.

   \p callback_soon is called when it is determined when the acquisition of the
   data will be made an estimated amount of time from now, because the last
   dependency has just started and we know how long it will take.

   \p callback_acquired is called when the data is acquired in terms of semantic,
   but the data is not fetched yet. It is given a pointer to the node, which it
   can modify if it wishes so.

   This is a very internal interface, subject to changes, do not use this.
*/
int starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback_soon)(void *arg, double delay), void (*callback_acquired)(void *arg, int *node, enum starpu_data_access_mode mode), void (*callback)(void *arg), void *arg, int sequential_consistency, int quick, long *pre_sync_jobid, long *post_sync_jobid, int prio);

/**
   The application can call this function instead of starpu_data_acquire() so as to
   acquire the data like starpu_data_acquire(), but only if all
   previously-submitted tasks have completed, in which case starpu_data_acquire_try()
   returns 0. StarPU will have ensured that the application will get an up-to-date
   copy of \p handle in main memory located where the data was originally
   registered. starpu_data_release() must be called once the application no longer
   needs to access the piece of data. See \ref DataAccess for more details.
*/
int starpu_data_acquire_try(starpu_data_handle_t handle, enum starpu_data_access_mode mode);

/**
   Similar to starpu_data_acquire_try(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be used instead of an
   explicit node number. See \ref DataAccess for more details.
*/
int starpu_data_acquire_on_node_try(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode);

#ifdef __GCC__

/**
   STARPU_DATA_ACQUIRE_CB() is the same as starpu_data_acquire_cb(),
   except that the code to be executed in a callback is directly provided
   as a macro parameter, and the data \p handle is automatically released
   after it. This permits to easily execute code which depends on the
   value of some registered data. This is non-blocking too and may be
   called from task callbacks.
*/
#define STARPU_DATA_ACQUIRE_CB(handle, mode, code)                    \
	do                                                            \
	{                                                             \
		void callback(void *arg)                              \
		{                                                     \
			code;                                         \
			starpu_data_release(handle);                  \
		}                                                     \
		starpu_data_acquire_cb(handle, mode, callback, NULL); \
	}                                                             \
	while (0)
#endif

/**
   Release the piece of data acquired by the
   application either by starpu_data_acquire() or by
   starpu_data_acquire_cb(). See \ref DataAccess for more details.
*/
void starpu_data_release(starpu_data_handle_t handle);

/**
   Similar to starpu_data_release(), except that the data
   was made available on the given memory \p node instead of main memory.
   The \p node parameter must be exactly the same as the corresponding \c
   starpu_data_acquire_on_node* call. See \ref DataAccess for more details.
*/
void starpu_data_release_on_node(starpu_data_handle_t handle, int node);

/**
   Partly release the piece of data acquired by the application either by
   starpu_data_acquire() or by starpu_data_acquire_cb(), switching the
   acquisition down to \p down_to_mode. For now, only releasing from ::STARPU_RW
   or ::STARPU_W acquisition down to ::STARPU_R is supported, or down to the same
   acquisition. ::STARPU_NONE can also be passed as \p down_to_mode, in which
   case this is equivalent to calling starpu_data_release(). See \ref DataAccess for more details.
*/
void starpu_data_release_to(starpu_data_handle_t handle, enum starpu_data_access_mode down_to_mode);

/**
   Similar to starpu_data_release_to(), except that the data
   was made available on the given memory \p node instead of main memory.
   The \p node parameter must be exactly the same as the corresponding \c
   starpu_data_acquire_on_node* call. See \ref DataAccess for more details.
*/
void starpu_data_release_to_on_node(starpu_data_handle_t handle, enum starpu_data_access_mode down_to_mode, int node);

/** @} */

/**
   This is an arbiter, which implements an advanced but centralized
   management of concurrent data accesses, see \ref
   ConcurrentDataAccess for the details.
*/
typedef struct starpu_arbiter *starpu_arbiter_t;

/**
   Create a data access arbiter, see \ref ConcurrentDataAccess for the
   details
*/
starpu_arbiter_t starpu_arbiter_create(void) STARPU_ATTRIBUTE_MALLOC;

/**
   Make access to \p handle managed by \p arbiter, see \ref
   ConcurrentDataAccess for the details.
*/
void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter);

/**
   Destroy the \p arbiter. This must only be called after all data
   assigned to it have been unregistered. See \ref
   ConcurrentDataAccess for the details.
*/
void starpu_arbiter_destroy(starpu_arbiter_t arbiter);

/**
   Explicitly ask StarPU to allocate room for a piece of data on
   the specified memory \p node. See \ref DataPrefetch for more details.
*/
int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node);

/**
   Prefetch levels

   Data requests are ordered by priorities, but also by prefetching level,
   between data that a task wants now, and data that we will probably want
   "soon".
*/
enum starpu_is_prefetch
{
	/** A task really needs it now! */
	STARPU_FETCH = 0,
	/** A task will need it soon */
	STARPU_TASK_PREFETCH = 1,
	/** It is a good idea to have it asap */
	STARPU_PREFETCH = 2,
	/** Get this here when you have time to */
	STARPU_IDLEFETCH = 3,
	STARPU_NFETCH
};

/**
   Issue a fetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to the given node as soon as possible, so that it is
   available there for tasks. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the
   data. See \ref DataPrefetch for more details.
*/
int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);

/**
   Issue a prefetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to \p node when there is room for it, so that it is
   available there for tasks. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the
   data. See \ref DataPrefetch for more details.
*/
int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);

/**
   See \ref DataPrefetch for more details.
 */
int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);

/**
   Issue an idle prefetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to \p node, so that it is
   available there for tasks, but only when the bus is really idle. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the data. See \ref DataPrefetch for more details.
*/
int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);

/**
   See \ref DataPrefetch for more details.
 */
int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);

/**
   Check whether a valid copy of \p handle is currently available on
   memory node \p node (or a transfer request for getting so is ongoing). See \ref SchedulingHelpers for more details.
*/
unsigned starpu_data_is_on_node(starpu_data_handle_t handle, unsigned node);

/**
   Check whether a valid copy of \p handle is currently available on
   memory node \p node (excluding prefetch).
*/
unsigned starpu_data_is_on_node_excluding_prefetch(starpu_data_handle_t handle, unsigned node);

/**
   Advise StarPU that \p handle will not be used in the close future, and is
   thus a good candidate for eviction from GPUs. StarPU will thus write its value
   back to its home node when the bus is idle, and select this data in priority
   for eviction when memory gets low. See \ref DataPrefetch for more details.
*/
void starpu_data_wont_use(starpu_data_handle_t handle);

/**
   Advise StarPU to evict \p handle from the memory node \p node
   StarPU will thus write its value back to its home node, before evicting it.
   This may however fail if e.g. some task is still working on it.

   If the eviction was successful, 0 is returned ; -1 is returned otherwise.

   See \ref DataPrefetch for more details.
*/
int starpu_data_evict_from_node(starpu_data_handle_t handle, unsigned node);

/**
   Set the write-through mask of the data \p handle (and
   its children), i.e. a bitmask of nodes where the data should be always
   replicated after modification. It also prevents the data from being
   evicted from these nodes when memory gets scarse. When the data is
   modified, it is automatically transferred into those memory nodes. For
   instance a <c>1<<0</c> write-through mask means that the CUDA workers
   will commit their changes in main memory (node 0). See \ref DataManagementAllocation for more details.
*/
void starpu_data_set_wt_mask(starpu_data_handle_t handle, uint32_t wt_mask);

/**
   Set the gathering node of the data, i.e. where the pieces from children of
   partitioning will be collected to obtain data coherency.
*/
void starpu_data_set_gathering_node(starpu_data_handle_t handle, unsigned node);

/**
   @name Implicit Data Dependencies
   In this section, we describe how StarPU makes it possible to
   insert implicit task dependencies in order to enforce sequential data
   consistency. When this data consistency is enabled on a specific data
   handle, any data access will appear as sequentially consistent from
   the application. For instance, if the application submits two tasks
   that access the same piece of data in read-only mode, and then a third
   task that access it in write mode, dependencies will be added between
   the two first tasks and the third one. Implicit data dependencies are
   also inserted in the case of data accesses from the application.
   @{
*/

/**
   Set the data consistency mode associated to a data handle. The
   consistency mode set using this function has the priority over the
   default mode which can be set with
   starpu_data_set_default_sequential_consistency_flag().
   See \ref SequentialConsistency and \ref DataManagementAllocation for more details.
*/
void starpu_data_set_sequential_consistency_flag(starpu_data_handle_t handle, unsigned flag);

/**
   Get the data consistency mode associated to the data handle \p handle. See \ref SequentialConsistency for more details.
*/
unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle);

/**
   Return the default sequential consistency flag. See \ref SequentialConsistency for more details.
*/
unsigned starpu_data_get_default_sequential_consistency_flag(void);

/**
   Set the default sequential consistency flag. If a non-zero
   value is passed, a sequential data consistency will be enforced for
   all handles registered after this function call, otherwise it is
   disabled. By default, StarPU enables sequential data consistency. It
   is also possible to select the data consistency mode of a specific
   data handle with the function
   starpu_data_set_sequential_consistency_flag(). See \ref SequentialConsistency for more details.
*/
void starpu_data_set_default_sequential_consistency_flag(unsigned flag);

/** @} */

/**
   Set whether this data should be elligible to be evicted to disk
   storage (1) or not (0). The default is 1. See \ref OOCDataRegistration for more details.
*/
void starpu_data_set_ooc_flag(starpu_data_handle_t handle, unsigned flag);

/**
   Get whether this data was set to be elligible to be evicted to disk
   storage (1) or not (0). See \ref OOCDataRegistration for more details.
*/
unsigned starpu_data_get_ooc_flag(starpu_data_handle_t handle);

/**
   Query the status of \p handle on the specified \p memory_node.

   \p is_allocated tells whether memory was allocated there for the data.
   \p is_valid tells whether the actual value is available there.
   \p is_loading tells whether the actual value is getting loaded there.
   \p is_requested tells whether the actual value is requested to be loaded
   there by some fetch/prefetch/idlefetch request.
   See \ref DataPrefetch for more details.
*/
void starpu_data_query_status2(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_loading, int *is_requested);

/**
   Same as starpu_data_query_status2(), but without the is_loading parameter. See \ref DataPrefetch for more details.
*/
void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested);

struct starpu_codelet;

/**
   Set the codelets to be used for \p handle when it is accessed in the
   mode ::STARPU_REDUX. Per-worker buffers will be initialized with
   the codelet \p init_cl (which has to take one handle with ::STARPU_W), and
   reduction between per-worker buffers will be done with the codelet \p
   redux_cl (which has to take a first accumulation handle with
   ::STARPU_RW|::STARPU_COMMUTE, and a second contribution handle with ::STARPU_R).
   See \ref DataReduction and \ref TemporaryData for more details.
*/
void starpu_data_set_reduction_methods(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, struct starpu_codelet *init_cl);

/**
   Same as starpu_data_set_reduction_methods() but allows to pass
   arguments to the reduction and init tasks
*/
void starpu_data_set_reduction_methods_with_args(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, void *redux_cl_arg, struct starpu_codelet *init_cl, void *init_cl_arg);

struct starpu_data_interface_ops *starpu_data_get_interface_ops(starpu_data_handle_t handle);

/**
   See \ref DataPrefetch for more details.
*/
unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle_t handle, unsigned memory_node);

/**
   See \ref DataPrefetch for more details.
*/
unsigned starpu_data_test_if_mapped_on_node(starpu_data_handle_t handle, unsigned memory_node);

/**
   See \ref DataPrefetch for more details.
*/
void starpu_memchunk_tidy(unsigned memory_node);

/**
   Set the field \c user_data for the \p handle to \p user_data . It can
   then be retrieved with starpu_data_get_user_data(). \p user_data can be any
   application-defined value, for instance a pointer to an object-oriented
   container for the data.
   See \ref DataHandlesHelpers for more details.
*/
void starpu_data_set_user_data(starpu_data_handle_t handle, void *user_data);

/**
   Retrieve the field \c user_data previously set for the \p handle.
   See \ref DataHandlesHelpers for more details.
*/
void *starpu_data_get_user_data(starpu_data_handle_t handle);

/**
   Set the field \c sched_data for the \p handle to \p sched_data . It can
   then be retrieved with starpu_data_get_sched_data(). \p sched_data can be any
   scheduler-defined value.
   See \ref DataHandlesHelpers for more details.
*/
void starpu_data_set_sched_data(starpu_data_handle_t handle, void *sched_data);

/**
   Retrieve the field \c sched_data previously set for the \p handle.
   See \ref DataHandlesHelpers for more details.
*/
void *starpu_data_get_sched_data(starpu_data_handle_t handle);

/**
   Check whether data \p handle can be evicted now from node \p node. See \ref DataPrefetch for more details.
*/
int starpu_data_can_evict(starpu_data_handle_t handle, unsigned node, enum starpu_is_prefetch is_prefetch);

/**
   Type for a data victim selector

   This is the type of function to be registered with
   starpu_data_register_victim_selector().

   \p toload, when different from NULL, specifies that we are looking for a
   victim with the same shape as this data, so that the buffer can simply be
   reused without any free/alloc operations (that are very costly with CUDA).

   \p node is the target node in which the victim should be evicted

   \p is_prefetch tells why the StarPU core is looking for an eviction
   victim. If it is beyond ::STARPU_FETCH, the selector should not be very
   aggressive: it should really not evict some data that is known to be reused
   soon, only for prefetching some other data.

   \p data is the same data as passed in the starpu_data_register_victim_selector() call.

   The selector returns the handle which should preferrably be evicted from the
   memory node.

   The selector can for instance use starpu_data_is_on_node() to determine which
   handles are on the memory node.

   It *must* use starpu_data_can_evict() to check whether the data can be
   evicted. Otherwise eviction will fail, the selector called again, only to
   fail again, etc. without any possible progress.
*/
typedef starpu_data_handle_t starpu_data_victim_selector(starpu_data_handle_t toload, unsigned node, enum starpu_is_prefetch is_prefetch, void *data);

/**
   Type for a victim eviction failure

   This is the type of function to be registered with
   starpu_data_register_victim_selector() for the failure case.

   \p victim is the data that was supposed to be evicted, but failed to be.

   \p node is the node on which the failure happened.

   \p data is the same data as passed in the starpu_data_register_victim_selector() call.
 */
typedef void starpu_data_victim_eviction_failed(starpu_data_handle_t victim, unsigned node, void *data);

/**
   Register a data victim selector.

   This register function \p selector to be called when StarPU needs to make
   room on a given memory node.

   See starpu_data_victim_selector() for more details.
*/
void starpu_data_register_victim_selector(starpu_data_victim_selector selector, starpu_data_victim_eviction_failed evicted, void *data);

/**
   To be returned by a starpu_data_victim_selector() when no victim was found,
   e.g. because all data is to be used by pending tasks.
*/
#define STARPU_DATA_NO_VICTIM ((starpu_data_handle_t) -1)

/**
   Return the set of data stored on a node

   This returns an array of the data handles that currently have a copy on node
   \p node. The array is returned in \p handles, whether they contain valid data
   is returned in \p states, and the number of handles is returned in \p n. The
   arrays must be freed by the caller with free().
*/
void starpu_data_get_node_data(unsigned node, starpu_data_handle_t **handles, int **valid, unsigned *n);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_H__ */
