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

#ifndef __STARPU_MPI_H__
#define __STARPU_MPI_H__

#include <starpu.h>

#if defined(STARPU_USE_MPI)

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_MPI_Support MPI Support
   @{
*/

/**
   @name Initialisation
   @{
*/

/**
   Initialize the StarPU library with the given \p conf, and
   initialize the StarPU-MPI library with the given MPI communicator
   \p comm. \p initialize_mpi indicates if MPI should be initialized
   or not by StarPU. StarPU-MPI takes the opportunity to modify \p
   conf to either reserve a core for its MPI thread (by default), or
   execute MPI calls on the CPU driver 0 between tasks.
*/
int starpu_mpi_init_conf(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm, struct starpu_conf *conf);

/**
   Same as starpu_mpi_init_conf(), except that this does not initialize the
   StarPU library. The caller thus has to call starpu_init() before this, and it
   can not reserve a core for the MPI communications.
*/
int starpu_mpi_init_comm(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm);

/**
   Call starpu_mpi_init_comm() with the MPI communicator \c MPI_COMM_WORLD.
*/
int starpu_mpi_init(int *argc, char ***argv, int initialize_mpi);

/**
   @deprecated
   This function has been made deprecated. One should use instead the
   function starpu_mpi_init(). This function does not call \c
   MPI_Init(), it should be called beforehand.
*/
int starpu_mpi_initialize(void) STARPU_DEPRECATED;

/**
   @deprecated
   This function has been made deprecated. One should use instead the
   function starpu_mpi_init(). MPI will be initialized by starpumpi by
   calling <c>MPI_Init_Thread(argc, argv, MPI_THREAD_SERIALIZED,
   ...)</c>.
*/
int starpu_mpi_initialize_extended(int *rank, int *world_size) STARPU_DEPRECATED;

/**
   Clean the starpumpi library. This must be called after calling any
   \c starpu_mpi functions and before the call to starpu_shutdown(),
   if any. \c MPI_Finalize() will be called if StarPU-MPI has been
   initialized by starpu_mpi_init().
*/
int starpu_mpi_shutdown(void);

/**
   Retrieve the current amount of communications from the current node
   in the array \p comm_amounts which must have a size greater or
   equal to the world size. Communications statistics must be enabled
   (see \ref STARPU_COMM_STATS).
*/
void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts);

/**
   Return in \p size the size of the communicator \p comm
*/
int starpu_mpi_comm_size(MPI_Comm comm, int *size);

/**
   Return in \p rank the rank of the calling process in the
   communicator \p comm
*/
int starpu_mpi_comm_rank(MPI_Comm comm, int *rank);

/**
   Return the rank of the calling process in the communicator \c
   MPI_COMM_WORLD
*/
int starpu_mpi_world_rank(void);

/**
   Return the size of the communicator \c MPI_COMM_WORLD
*/
int starpu_mpi_world_size(void);

/**
   When given to the function starpu_mpi_comm_get_attr(), retrieve the
   value for the upper bound for tag value.
*/
#define STARPU_MPI_TAG_UB MPI_TAG_UB

/**
   Retrieve an attribute value by key, similarly to the MPI function
   \c MPI_comm_get_attr(), except that the value is a pointer to
   int64_t instead of int. If an attribute is attached on \p comm to
   \p keyval, then the call returns \p flag equal to \c 1, and the
   attribute value in \p attribute_val. Otherwise, \p flag is set to
   \0.
*/
int starpu_mpi_comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag);

int starpu_mpi_get_communication_tag(void);
void starpu_mpi_set_communication_tag(int tag);

/** @} */

/**
   @name Communication
   \anchor MPIPtpCommunication
   @{
*/

/**
   Opaque type for communication request
*/
typedef void *starpu_mpi_req;

/**
   Type of the message tag.
*/
typedef int64_t starpu_mpi_tag_t;

/**
   Post a standard-mode, non blocking send of \p data_handle to the
   node \p dest using the message tag \p data_tag within the
   communicator \p comm. After the call, the pointer to the request \p
   req can be used to test or to wait for the completion of the
   communication.
*/
int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);

/**
   Similar to starpu_mpi_isend(), but take a priority \p prio.
*/
int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);

/**
   Post a nonblocking receive in \p data_handle from the node \p
   source using the message tag \p data_tag within the communicator \p
   comm. After the call, the pointer to the request \p req can be used
   to test or to wait for the completion of the communication.
*/
int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *req, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm);

/**
   Perform a standard-mode, blocking send of \p data_handle to the
   node \p dest using the message tag \p data_tag within the
   communicator \p comm.
*/
int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);

/**
   Similar to starpu_mpi_send(), but take a priority \p prio.
*/
int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);

/**
   Perform a standard-mode, blocking receive in \p data_handle from
   the node \p source using the message tag \p data_tag within the
   communicator \p comm.
*/
int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, MPI_Status *status);

/**
   Post a standard-mode, non blocking send of \p data_handle to the
   node \p dest using the message tag \p data_tag within the
   communicator \p comm. On completion, the \p callback function is
   called with the argument \p arg.
   Similarly to the pthread detached functionality, when a detached
   communication completes, its resources are automatically released
   back to the system, there is no need to test or to wait for the
   completion of the request.
*/
int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);

/**
   Similar to starpu_mpi_isend_detached, but take a priority \p prio.
*/
int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);

/**
   Post a nonblocking receive in \p data_handle from the node \p
   source using the message tag \p data_tag within the communicator \p
   comm. On completion, the \p callback function is called with the
   argument \p arg.
   Similarly to the pthread detached functionality, when a detached
   communication completes, its resources are automatically released
   back to the system, there is no need to test or to wait for the
   completion of the request.
*/
int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);

/**
   Post a nonblocking receive in \p data_handle from the node \p
   source using the message tag \p data_tag within the communicator \p
   comm. On completion, the \p callback function is called with the
   argument \p arg.
   The parameter \p sequential_consistency allows to enable or disable
   the sequential consistency for \p data handle (sequential
   consistency will be enabled or disabled based on the value of the
   parameter \p sequential_consistency and the value of the sequential
   consistency defined for \p data_handle).
   Similarly to the pthread detached functionality, when a detached
   communication completes, its resources are automatically released
   back to the system, there is no need to test or to wait for the
   completion of the request.
*/
int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency);

/**
   Perform a synchronous-mode, non-blocking send of \p data_handle to
   the node \p dest using the message tag \p data_tag within the
   communicator \p comm.
*/
int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);

/**
   Similar to starpu_mpi_issend(), but take a priority \p prio.
*/
int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);

/**
   Perform a synchronous-mode, non-blocking send of \p data_handle to
   the node \p dest using the message tag \p data_tag within the
   communicator \p comm. On completion, the \p callback function is
   called with the argument \p arg.
   Similarly to the pthread detached functionality, when a detached
   communication completes, its resources are automatically released
   back to the system, there is no need to test or to wait for the
   completion of the request.
*/
int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);

/**
   Similar to starpu_mpi_issend_detached(), but take a priority \p prio.
*/
int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);

/**
   Return when the operation identified by request \p req is complete.
*/
int starpu_mpi_wait(starpu_mpi_req *req, MPI_Status *status);

/**
   If the operation identified by \p req is complete, set \p flag to
   1. The \p status object is set to contain information on the
   completed operation.
*/
int starpu_mpi_test(starpu_mpi_req *req, int *flag, MPI_Status *status);

/**
   Block the caller until all group members of the communicator \p
   comm have called it.
*/
int starpu_mpi_barrier(MPI_Comm comm);

/**
   Wait until all StarPU tasks and communications for the given
   communicator are completed.
*/
int starpu_mpi_wait_for_all(MPI_Comm comm);

/**
   Post a standard-mode, non blocking send of \p data_handle to the
   node \p dest using the message tag \p data_tag within the
   communicator \p comm. On completion, \p tag is unlocked.
*/
int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);

/**
   Similar to starpu_mpi_isend_detached_unlock_tag(), but take a
   priority \p prio.
*/
int starpu_mpi_isend_detached_unlock_tag_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, starpu_tag_t tag);

/**
   Post a nonblocking receive in \p data_handle from the node \p
   source using the message tag \p data_tag within the communicator \p
   comm. On completion, \p tag is unlocked.
*/
int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);

/**
   Post \p array_size standard-mode, non blocking send. Each post
   sends the n-th data of the array \p data_handle to the n-th node of
   the array \p dest using the n-th message tag of the array \p
   data_tag within the n-th communicator of the array \p comm. On
   completion of the all the requests, \p tag is unlocked.
*/
int starpu_mpi_isend_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag);

/**
   Similar to starpu_mpi_isend_array_detached_unlock_tag(), but take a
   priority \p prio.
*/
int starpu_mpi_isend_array_detached_unlock_tag_prio(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, int *prio, MPI_Comm *comm, starpu_tag_t tag);

/**
   Post \p array_size nonblocking receive. Each post receives in the
   n-th data of the array \p data_handle from the n-th node of the
   array \p source using the n-th message tag of the array \p data_tag
   within the n-th communicator of the array \p comm. On completion of
   the all the requests, \p tag is unlocked.
*/
int starpu_mpi_irecv_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *source, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag);

typedef int (*starpu_mpi_datatype_allocate_func_t)(starpu_data_handle_t, MPI_Datatype *);
typedef void (*starpu_mpi_datatype_free_func_t)(MPI_Datatype *);

/**
   Register functions to create and free a MPI datatype for the given
   handle.
   Similar to starpu_mpi_interface_datatype_register().
   It is important that the function is called before any
   communication can take place for a data with the given handle. See
   \ref ExchangingUserDefinedDataInterface for an example.
*/
int starpu_mpi_datatype_register(starpu_data_handle_t handle, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func);

/**
   Register functions to create and free a MPI datatype for the given
   interface id.
   Similar to starpu_mpi_datatype_register().
   It is important that the function is called before any
   communication can take place for a data with the given handle. See
   \ref ExchangingUserDefinedDataInterface for an example.
*/
int starpu_mpi_interface_datatype_register(enum starpu_data_interface_id id, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func);

/**
   Unregister the MPI datatype functions stored for the interface of
   the given handle.
*/
int starpu_mpi_datatype_unregister(starpu_data_handle_t handle);

/**
   Unregister the MPI datatype functions stored for the interface of
   the given interface id. Similar to starpu_mpi_datatype_unregister().
*/
int starpu_mpi_interface_datatype_unregister(enum starpu_data_interface_id id);

/** @} */

/**
   @name Communication Cache
   @{
*/

/**
   Return 1 if the communication cache is enabled, 0 otherwise
*/
int starpu_mpi_cache_is_enabled();

/**
   If \p enabled is 1, enable the communication cache. Otherwise,
   clean the cache if it was enabled and disable it.
*/
int starpu_mpi_cache_set(int enabled);

/**
   Clear the send and receive communication cache for the data \p
   data_handle and invalidate the value. The function has to be called
   at the same point of task graph submission by all the MPI nodes on
   which the handle was registered. The function does nothing if the
   cache mechanism is disabled (see \ref STARPU_MPI_CACHE).
*/
void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle);

/**
   Clear the send and receive communication cache for all data and
   invalidate their values. The function has to be called at the same
   point of task graph submission by all the MPI nodes. The function
   does nothing if the cache mechanism is disabled (see \ref
   STARPU_MPI_CACHE).
*/
void starpu_mpi_cache_flush_all_data(MPI_Comm comm);

/**
   Test whether \p data_handle is cached for reception, i.e. the value
   was previously received from the owner node, and not flushed since
   then.
*/
int starpu_mpi_cached_receive(starpu_data_handle_t data_handle);

/**
 * If \p data is already available in the reception cache, return 1
 * If \p data is NOT available in the reception cache, add it to the
 * cache and return 0
 * Return 0 if the communication cache is not enabled
 */
int starpu_mpi_cached_receive_set(starpu_data_handle_t data);

/**
 * Remove \p data from the reception cache
 */
void starpu_mpi_cached_receive_clear(starpu_data_handle_t data);

/**
   Test whether \p data_handle is cached for emission to node \p dest,
   i.e. the value was previously sent to \p dest, and not flushed
   since then.
*/
int starpu_mpi_cached_send(starpu_data_handle_t data_handle, int dest);

/**
 * If \p data is already available in the emission cache for node
 * \p dest, return 1
 * If \p data is NOT available in the emission cache for node \p dest,
 * add it to the cache and return 0
 * Return 0 if the communication cache is not enabled
 */
int starpu_mpi_cached_send_set(starpu_data_handle_t data, int dest);

/**
 * Remove \p data from the emission cache
 */
void starpu_mpi_cached_send_clear(starpu_data_handle_t data);

/** @} */

/**
   @name MPI Insert Task
   \anchor MPIInsertTask
   @{
*/

/**
   Can be used as rank when calling starpu_mpi_data_register() and
   alike, to specify that the data is per-node: each node will have
   its own value. Tasks writing to such data will be replicated on all
   nodes (and all parameters then have to be per-node). Tasks not
   writing to such data will just take the node-local value without
   any MPI communication.
*/
#define STARPU_MPI_PER_NODE -2

/**
   Register to MPI a StarPU data handle with the given tag, rank and
   MPI communicator. It also automatically clears the MPI
   communication cache when unregistering the data.
*/
void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag, int rank, MPI_Comm comm);

/**
   Register to MPI a StarPU data handle with the given tag, rank and
   the MPI communicator \c MPI_COMM_WORLD.
   It also automatically clears the MPI communication cache when
   unregistering the data.
*/
#define starpu_mpi_data_register(data_handle, data_tag, rank) starpu_mpi_data_register_comm(data_handle, data_tag, rank, MPI_COMM_WORLD)

/**
   Register to MPI a StarPU data handle with the given tag. No rank
   will be defined.
   It also automatically clears the MPI communication cache when
   unregistering the data.
*/
void starpu_mpi_data_set_tag(starpu_data_handle_t handle, starpu_mpi_tag_t data_tag);

/**
   Symbol kept for backward compatibility. Call function starpu_mpi_data_set_tag()
*/
#define starpu_data_set_tag starpu_mpi_data_set_tag

/**
   Register to MPI a StarPU data handle with the given rank and given
   communicator. No tag will be defined.
   It also automatically clears the MPI communication cache when
   unregistering the data.
*/
void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm);

/**
   Register to MPI a StarPU data handle with the given rank and the
   MPI communicator \c MPI_COMM_WORLD. No tag will be defined.
   It also automatically clears the MPI communication cache when
   unregistering the data.
*/
#define starpu_mpi_data_set_rank(handle, rank) starpu_mpi_data_set_rank_comm(handle, rank, MPI_COMM_WORLD)

/**
   Symbol kept for backward compatibility. Call function starpu_mpi_data_set_rank()
*/
#define starpu_data_set_rank starpu_mpi_data_set_rank

/**
   Return the rank of the given data.
*/
int starpu_mpi_data_get_rank(starpu_data_handle_t handle);

/**
   Symbol kept for backward compatibility. Call function starpu_mpi_data_get_rank()
*/
#define starpu_data_get_rank starpu_mpi_data_get_rank

/**
   Return the tag of the given data.
*/
starpu_mpi_tag_t starpu_mpi_data_get_tag(starpu_data_handle_t handle);

/**
   Symbol kept for backward compatibility. Call function starpu_mpi_data_get_tag()
*/
#define starpu_data_get_tag starpu_mpi_data_get_tag

/**
   Create and submit a task corresponding to codelet with the
   following arguments. The argument list must be zero-terminated.
   The arguments following the codelet are the same types as for the
   function starpu_task_insert().
   Access modes for data can also be
   set with ::STARPU_SSEND to specify the data has to be sent using a
   synchronous and non-blocking mode (see starpu_mpi_issend()).
   The extra argument ::STARPU_EXECUTE_ON_NODE followed by an integer
   allows to specify the MPI node to execute the codelet. It is also
   possible to specify that the node owning a specific data will
   execute the codelet, by using ::STARPU_EXECUTE_ON_DATA followed by
   a data handle.

   The internal algorithm is as follows:
   <ol>
   <li>
   Find out which MPI node is going to execute the codelet.
   	<ul>
	<li>
	If there is only one node owning data in ::STARPU_W mode, it
	will be selected;
	<li>
	If there is several nodes owning data in ::STARPU_W mode, a
	node will be selected according to a given node selection
	policy (see ::STARPU_NODE_SELECTION_POLICY or
	starpu_mpi_node_selection_set_current_policy())
	<li>
	The argument ::STARPU_EXECUTE_ON_NODE followed by an integer
	can be used to specify the node;
	<li>
	The argument ::STARPU_EXECUTE_ON_DATA followed by a data handle can be used to specify that the node owing the given data will execute the codelet.
	</ul>
   </li>
   <li>
   Send and receive data as requested. Nodes owning data which need to
   be read by the task are sending them to the MPI node which will
   execute it. The latter receives them.
   </li>
   <li>
   Execute the codelet. This is done by the MPI node selected in the
   1st step of the algorithm.
   </li>
   <li>
   If several MPI nodes own data to be written to, send written data
   back to their owners.
   </li>
   </ol>

   The algorithm also includes a communication cache mechanism that
   allows not to send data twice to the same MPI node, unless the data
   has been modified. The cache can be disabled (see \ref
   STARPU_MPI_CACHE).
*/
int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...);

/**
   Call starpu_mpi_task_insert(). Symbol kept for backward compatibility.
*/
int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...);

/**
   Create a task corresponding to \p codelet with the following given
   arguments. The argument list must be zero-terminated. The function
   performs the first two steps of the function
   starpu_mpi_task_insert(), i.e. submitting the MPI communications
   needed before the execution of the task, and the creation of the
   task on one node. Only the MPI node selected in the first step of
   the algorithm will return a valid task structure which can then be
   submitted, others will return <c>NULL</c>. The function
   starpu_mpi_task_post_build() MUST be called after that on all
   nodes, and after the submission of the task on the node which
   creates it, with the SAME list of arguments.
*/
struct starpu_task *starpu_mpi_task_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);

/**
   MUST be called after a call to starpu_mpi_task_build(),
   with the SAME list of arguments. Perform the fourth -- last -- step of
   the algorithm described in starpu_mpi_task_insert().
*/
int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);

/**
   Transfer data \p data_handle to MPI node \p node, sending it from
   its owner if needed. At least the target node and the owner have to
   call the function.
*/
void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node);

/**
   Transfer data \p data_handle to MPI node \p node, sending it from
   its owner if needed. At least the target node and the owner have to
   call the function. On reception, the \p callback function is called
   with the argument \p arg.
*/
void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg);

/**
   Transfer data \p data_handle to all MPI nodes, sending it from its
   owner if needed. All nodes have to call the function.
*/
void starpu_mpi_get_data_on_all_nodes_detached(MPI_Comm comm, starpu_data_handle_t data_handle);

/**
   Submit migration of the data onto the \p new_rank MPI node. This
   means both submitting the transfer of the data to node \p new_rank
   if it hasn't been submitted already, and setting the home node of
   the data to the new node. Further data transfers submitted by
   starpu_mpi_task_insert() will be done from that new node. This
   function thus needs to be called on all nodes which have registered
   the data at the same point of tasks submissions. This also flushes
   the cache for this data to avoid incoherencies.
*/
void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t handle, int new_rank);

/** @} */

/**
   @name Node Selection Policy
   \anchor MPINodeSelectionPolicy
   @{
*/


/**
   Define the current policy
 */
#define STARPU_MPI_NODE_SELECTION_CURRENT_POLICY -1
/**
   Define the policy in which the selected node is the one having the
   most data in ::STARPU_R mode
*/
#define STARPU_MPI_NODE_SELECTION_MOST_R_DATA    0

typedef int (*starpu_mpi_select_node_policy_func_t)(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data);

/**
   Register a new policy which can then be used when there is several
   nodes owning data in ::STARPU_W mode.
   Here an example of function defining a node selection policy.
   The codelet will be executed on the node owing the first data with
   a size bigger than 1M, or on the node 0 if no data fits the given
   size.
   \code{.c}
   int my_node_selection_policy(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
   {
	// me is the current MPI rank
	// nb_nodes is the number of MPI nodes
	// descr is the description of the data specified when calling starpu_mpi_task_insert
	// nb_data is the number of data in descr
	int i;
	for(i= 0 ; i<nb_data ; i++)
	{
		starpu_data_handle_t data = descr[i].handle;
		enum starpu_data_access_mode mode = descr[i].mode;
		if (mode & STARPU_R)
		{
			int rank = starpu_data_get_rank(data);
			size_t size = starpu_data_get_size(data);
			if (size > 1024*1024) return rank;
		}
	}
	return 0;
	}
	\endcode
*/
int starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_policy_func_t policy_func);

/**
   Unregister a previously registered policy.
*/
int starpu_mpi_node_selection_unregister_policy(int policy);

/**
   Return the current policy used to select the node which will
   execute the codelet
*/
int starpu_mpi_node_selection_get_current_policy();

/**
   Set the current policy used to select the node which will execute
   the codelet. The policy ::STARPU_MPI_NODE_SELECTION_MOST_R_DATA
   selects the node having the most data in ::STARPU_R mode so as to
   minimize the amount of data to be transfered.
*/
int starpu_mpi_node_selection_set_current_policy(int policy);

/** @} */

/**
   @name Collective Operations
   \anchor MPICollectiveOperations
   @{
*/

/**
   Perform a reduction on the given data \p handle. All nodes send the
   data to its owner node which will perform a reduction.
*/
void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle);

/**
   Similar to starpu_mpi_redux_data, but take a priority \p prio.
*/
void starpu_mpi_redux_data_prio(MPI_Comm comm, starpu_data_handle_t data_handle, int prio);

/**
   Scatter data among processes of the communicator based on the
   ownership of the data. For each data of the array \p data_handles,
   the process \p root sends the data to the process owning this data.
   Processes receiving data must have valid data handles to receive
   them. On completion of the collective communication, the \p
   scallback function is called with the argument \p sarg on the
   process \p root, the \p rcallback function is called with the
   argument \p rarg on any other process.
*/
int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);

/**
   Gather data from the different processes of the communicator onto
   the process \p root. Each process owning data handle in the array
   \p data_handles will send them to the process \p root. The process
   \p root must have valid data handles to receive the data. On
   completion of the collective communication, the \p rcallback
   function is called with the argument \p rarg on the process root,
   the \p scallback function is called with the argument \p sarg on
   any other process.
*/
int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);

/** @} */

int starpu_mpi_pre_submit_hook_register(void (*f)(struct starpu_task *));
int starpu_mpi_pre_submit_hook_unregister();

/** @} */

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI
#endif // __STARPU_MPI_H__
