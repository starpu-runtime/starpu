/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
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
   instance.
*/
typedef struct _starpu_data_state* starpu_data_handle_t;

/**
   Describe a StarPU data access mode

   Note: when adding a flag here, update
   _starpu_detect_implicit_data_deps_with_handle

   Note: other STARPU_* values in include/starpu_task_util.h
 */
enum starpu_data_access_mode
{
	STARPU_NONE=0, /**< todo */
	STARPU_R=(1<<0), /**< read-only mode */
	STARPU_W=(1<<1), /**< write-only mode */
	STARPU_RW=(STARPU_R|STARPU_W), /**< read-write mode. Equivalent to ::STARPU_R|::STARPU_W  */
	STARPU_SCRATCH=(1<<2), /**< A temporary buffer is allocated
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
			       */
	STARPU_REDUX=(1<<3), /**< todo */
	STARPU_COMMUTE=(1<<4), /**<  ::STARPU_COMMUTE can be passed
				  along ::STARPU_W or ::STARPU_RW to
				  express that StarPU can let tasks
				  commute, which is useful e.g. when
				  bringing a contribution into some
				  data, which can be done in any order
				  (but still require sequential
				  consistency against reads or
				  non-commutative writes).
			       */
	STARPU_SSEND=(1<<5), /**< used in starpu_mpi_insert_task() to
				specify the data has to be sent using
				a synchronous and non-blocking mode
				(see starpu_mpi_issend())
			     */
	STARPU_LOCALITY=(1<<6), /**< used to tell the scheduler which
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
				*/
	STARPU_ACCESS_MODE_MAX=(1<<7) /**< todo */
};

struct starpu_data_interface_ops;

/**
   Set the name of the data, to be shown in various profiling tools.
*/
void starpu_data_set_name(starpu_data_handle_t handle, const char *name);

/**
   Set the coordinates of the data, to be shown in various profiling
   tools. \p dimensions is the size of the \p dims array. This can be
   for instance the tile coordinates within a big matrix.
*/
void starpu_data_set_coordinates_array(starpu_data_handle_t handle, unsigned dimensions, int dims[]);

/**
   Set the coordinates of the data, to be shown in various profiling
   tools. \p dimensions is the number of subsequent \c int parameters.
   This can be for instance the tile coordinates within a big matrix.
*/
void starpu_data_set_coordinates(starpu_data_handle_t handle, unsigned dimensions, ...);

/**
   Get the coordinates of the data, as set by a previous call to
   starpu_data_set_coordinates_array() or starpu_data_set_coordinates()
   \p dimensions is the size of the \p dims array.
   This returns the actual number of returned coordinates.
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
*/
void starpu_data_unregister(starpu_data_handle_t handle);

/**
    Similar to starpu_data_unregister(), except that StarPU does not
    put back a valid copy into the home node, in the buffer that was
    initially registered.
*/
void starpu_data_unregister_no_coherency(starpu_data_handle_t handle);

/**
   Destroy the data \p handle once it is no longer needed by any
   submitted task. No coherency is provided.

   This is not safe to call starpu_data_unregister_submit() on a handle that
   comes from the registration of a non-NULL application home buffer, since the
   moment when the unregistration will happen is unknown to the
   application. Only calling starpu_shutdown() allows to be sure that the data
   was really unregistered.
*/
void starpu_data_unregister_submit(starpu_data_handle_t handle);

/**
   Destroy all replicates of the data \p handle immediately. After
   data invalidation, the first access to \p handle must be performed
   in ::STARPU_W mode. Accessing an invalidated data in ::STARPU_R
   mode results in undefined behaviour.
*/
void starpu_data_invalidate(starpu_data_handle_t handle);

/**
   Submit invalidation of the data \p handle after completion of
   previously submitted tasks.
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
   explictly by calling
   starpu_data_set_default_sequential_consistency_flag() or
   starpu_data_set_sequential_consistency_flag().
   starpu_data_acquire() is a blocking call, so that it cannot be
   called from tasks or from their callbacks (in that case,
   starpu_data_acquire() returns <c>-EDEADLK</c>). Upon successful
   completion, this function returns 0.
*/
int starpu_data_acquire(starpu_data_handle_t handle, enum starpu_data_access_mode mode);

/**
   Similar to starpu_data_acquire(), except that the data will be
   available on the given memory node instead of main memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can
   be used instead of an explicit node number.
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
   completion, this function returns 0.
*/
int starpu_data_acquire_cb(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);

/**
   Similar to starpu_data_acquire_cb(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be
   used instead of an explicit node number.
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
   completion, this function returns 0.
*/
int starpu_data_acquire_cb_sequential_consistency(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);

/**
   Similar to starpu_data_acquire_cb_sequential_consistency(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be used instead of an
   explicit node number.
*/
int starpu_data_acquire_on_node_cb_sequential_consistency(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);

int starpu_data_acquire_on_node_cb_sequential_consistency_quick(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency, int quick);

/**
   Similar to starpu_data_acquire_on_node_cb_sequential_consistency(),
   except that the \e pre_sync_jobid and \e post_sync_jobid parameters can be used
   to retrieve the jobid of the synchronization tasks. \e pre_sync_jobid happens
   just before the acquisition, and \e post_sync_jobid happens just after the
   release.
*/
int starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency, int quick, long *pre_sync_jobid, long *post_sync_jobid);

/**
   The application can call this function instead of starpu_data_acquire() so as to
   acquire the data like starpu_data_acquire(), but only if all
   previously-submitted tasks have completed, in which case starpu_data_acquire_try()
   returns 0. StarPU will have ensured that the application will get an up-to-date
   copy of \p handle in main memory located where the data was originally
   registered. starpu_data_release() must be called once the application no longer
   needs to access the piece of data.
*/
int starpu_data_acquire_try(starpu_data_handle_t handle, enum starpu_data_access_mode mode);

/**
   Similar to starpu_data_acquire_try(), except that the
   data will be available on the given memory node instead of main
   memory.
   ::STARPU_ACQUIRE_NO_NODE and ::STARPU_ACQUIRE_NO_NODE_LOCK_ALL can be used instead of an
   explicit node number.
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
#  define STARPU_DATA_ACQUIRE_CB(handle, mode, code) do \
	{ \						\
		void callback(void *arg)		\
		{					\
			code;				\
			starpu_data_release(handle);  	\
		}			      		\
		starpu_data_acquire_cb(handle, mode, callback, NULL);	\
	}						\
	while(0)
#endif

/**
   Release the piece of data acquired by the
   application either by starpu_data_acquire() or by
   starpu_data_acquire_cb().
*/
void starpu_data_release(starpu_data_handle_t handle);

/**
   Similar to starpu_data_release(), except that the data
   will be available on the given memory \p node instead of main memory.
   The \p node parameter must be exactly the same as the corresponding \c
   starpu_data_acquire_on_node* call.
*/
void starpu_data_release_on_node(starpu_data_handle_t handle, int node);

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
   Make access to \p handle managed by \p arbiter
*/
void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter);

/**
   Destroy the \p arbiter . This must only be called after all data
   assigned to it have been unregistered.
*/
void starpu_arbiter_destroy(starpu_arbiter_t arbiter);

/**
   Explicitly ask StarPU to allocate room for a piece of data on
   the specified memory \p node.
*/
int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node);

/**
   Issue a fetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to the given node as soon as possible, so that it is
   available there for tasks. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the
   data.
*/
int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);

/**
   Issue a prefetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to \p node when there is room for it, so that it is
   available there for tasks. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the
   data.
*/
int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);

int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);

/**
   Issue an idle prefetch request for the data \p handle to \p node, i.e.
   requests that the data be replicated to \p node, so that it is
   available there for tasks, but only when the bus is really idle. If \p async is 0, the call will
   block until the transfer is achieved, else the call will return immediately,
   after having just queued the request. In the latter case, the request will
   asynchronously wait for the completion of any task writing on the data.
*/
int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);

/**
   Check whether a valid copy of \p handle is currently available on
   memory node \p node.
*/
unsigned starpu_data_is_on_node(starpu_data_handle_t handle, unsigned node);

/**
   Advise StarPU that \p handle will not be used in the close future, and is
   thus a good candidate for eviction from GPUs. StarPU will thus write its value
   back to its home node when the bus is idle, and select this data in priority
   for eviction when memory gets low.
*/
void starpu_data_wont_use(starpu_data_handle_t handle);

/**
   Set the write-through mask of the data \p handle (and
   its children), i.e. a bitmask of nodes where the data should be always
   replicated after modification. It also prevents the data from being
   evicted from these nodes when memory gets scarse. When the data is
   modified, it is automatically transfered into those memory nodes. For
   instance a <c>1<<0</c> write-through mask means that the CUDA workers
   will commit their changes in main memory (node 0).
*/
void starpu_data_set_wt_mask(starpu_data_handle_t handle, uint32_t wt_mask);

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
*/
void starpu_data_set_sequential_consistency_flag(starpu_data_handle_t handle, unsigned flag);

/**
   Get the data consistency mode associated to the data handle \p handle
*/
unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle);

/**
   Return the default sequential consistency flag
*/
unsigned starpu_data_get_default_sequential_consistency_flag(void);

/**
   Set the default sequential consistency flag. If a non-zero
   value is passed, a sequential data consistency will be enforced for
   all handles registered after this function call, otherwise it is
   disabled. By default, StarPU enables sequential data consistency. It
   is also possible to select the data consistency mode of a specific
   data handle with the function
   starpu_data_set_sequential_consistency_flag().
*/
void starpu_data_set_default_sequential_consistency_flag(unsigned flag);

/** @} */

/**
   Set whether this data should be elligible to be evicted to disk
   storage (1) or not (0). The default is 1.
*/
void starpu_data_set_ooc_flag(starpu_data_handle_t handle, unsigned flag);
/**
   Get whether this data was set to be elligible to be evicted to disk
   storage (1) or not (0).
*/
unsigned starpu_data_get_ooc_flag(starpu_data_handle_t handle);

/**
   Query the status of \p handle on the specified \p memory_node.
*/
void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested);

struct starpu_codelet;

/**
   Set the codelets to be used for \p handle when it is accessed in the
   mode ::STARPU_REDUX. Per-worker buffers will be initialized with
   the codelet \p init_cl, and reduction between per-worker buffers will be
   done with the codelet \p redux_cl.
*/
void starpu_data_set_reduction_methods(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, struct starpu_codelet *init_cl);

struct starpu_data_interface_ops* starpu_data_get_interface_ops(starpu_data_handle_t handle);

unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle_t handle, unsigned memory_node);

void starpu_memchunk_tidy(unsigned memory_node);

/**
   Set the field \c user_data for the \p handle to \p user_data . It can
   then be retrieved with starpu_data_get_user_data(). \p user_data can be any
   application-defined value, for instance a pointer to an object-oriented
   container for the data.
*/
void starpu_data_set_user_data(starpu_data_handle_t handle, void* user_data);

/**
   Retrieve the field \c user_data previously set for the \p handle.
*/
void *starpu_data_get_user_data(starpu_data_handle_t handle);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_H__ */
