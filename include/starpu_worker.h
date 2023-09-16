/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
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

#ifndef __STARPU_WORKER_H__
#define __STARPU_WORKER_H__

#include <stdlib.h>
#include <starpu_config.h>
#include <starpu_data_interfaces.h>
#include <starpu_thread.h>
#include <starpu_task.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Workers Workers
   @{
*/

/**
   Worker Architecture Type

   The value 3 which was used by the driver SCC is no longer used as
   renumbering workers would make unusable old performance model
   files.
*/
enum starpu_worker_archtype
{
	STARPU_CPU_WORKER      = 0,  /**< CPU core */
	STARPU_CUDA_WORKER     = 1,  /**< NVIDIA CUDA device */
	STARPU_OPENCL_WORKER   = 2,  /**< OpenCL device */
	STARPU_MAX_FPGA_WORKER = 4,  /**< Maxeler FPGA device */
	STARPU_MPI_MS_WORKER   = 5,  /**< MPI Slave device */
	STARPU_TCPIP_MS_WORKER = 6,  /**< TCPIP Slave device */
	STARPU_HIP_WORKER      = 7,  /**< NVIDIA/AMD HIP device */
	STARPU_NARCH	       = 8,  /**< Number of arch types */
	STARPU_ANY_WORKER      = 255 /**< any worker, used in the hypervisor */
};

#define STARPU_UNKNOWN_WORKER      ((enum starpu_worker_archtype)-1) /**< Invalid worker value */

/**
   Structure needed to iterate on the collection
*/
struct starpu_sched_ctx_iterator
{
	/**
	   The index of the current worker in the collection, needed
	   when iterating on the collection.
	*/
	int cursor;
	void *value;
	void *possible_value;
	char visited[STARPU_NMAXWORKERS];
	int possibly_parallel;
};

/**
   Types of structures the worker collection can implement
*/
enum starpu_worker_collection_type
{
	STARPU_WORKER_TREE, /**< The collection is a tree */
	STARPU_WORKER_LIST  /**< The collection is an array */
};

/**
   A scheduling context manages a collection of workers that can be
   memorized using different data structures. Thus, a generic
   structure is available in order to simplify the choice of its type.
   Only the list data structure is available but further data
   structures(like tree) implementations are foreseen.
*/
struct starpu_worker_collection
{
	/**
	   The workerids managed by the collection
	*/
	int *workerids;
	void *collection_private;
	/**
	   The number of workers in the collection
	*/
	unsigned nworkers;
	void *unblocked_workers;
	unsigned nunblocked_workers;
	void *masters;
	unsigned nmasters;
	char present[STARPU_NMAXWORKERS];
	char is_unblocked[STARPU_NMAXWORKERS];
	char is_master[STARPU_NMAXWORKERS];
	/**
	   The type of structure
	*/
	enum starpu_worker_collection_type type;
	/**
	   Check if there is another element in collection
	*/
	unsigned (*has_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	/**
	   Return the next element in the collection
	*/
	int (*get_next)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	/**
	   Add a new element in the collection
	*/
	int (*add)(struct starpu_worker_collection *workers, int worker);
	/**
	   Remove an element from the collection
	*/
	int (*remove)(struct starpu_worker_collection *workers, int worker);
	/**
	   Initialize the collection
	*/
	void (*init)(struct starpu_worker_collection *workers);
	/**
	   Deinitialize the collection
	*/
	void (*deinit)(struct starpu_worker_collection *workers);
	/**
	   Initialize the cursor if there is one
	*/
	void (*init_iterator)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	void (*init_iterator_for_parallel_tasks)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it, struct starpu_task *task);
};

extern struct starpu_worker_collection starpu_worker_list;
extern struct starpu_worker_collection starpu_worker_tree;

/**
   Wait for all workers to be initialised. Calling this function is
   normally not necessary. It is called for example in
   <c>tools/starpu_machine_display</c> to make sure all workers
   information are correctly set before printing their information.
   See \ref PauseResume for more details.
*/
void starpu_worker_wait_for_initialisation(void);

/**
   Return true if type matches one of StarPU's defined worker architectures.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_archtype_is_valid(enum starpu_worker_archtype type);

/**
   Convert a mask of architectures to a worker archtype.
   See \ref TopologyWorkers for more details.
*/
enum starpu_worker_archtype starpu_arch_mask_to_worker_archtype(unsigned mask);

/**
   Return the number of workers (i.e. processing units executing
   StarPU tasks). The return value should be at most \ref
   STARPU_NMAXWORKERS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_get_count(void);

/**
   Return the number of CPUs controlled by StarPU. The return value
   should be at most \ref STARPU_MAXCPUS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_cpu_worker_get_count(void);

/**
   Return the number of CUDA devices controlled by StarPU. The return
   value should be at most \ref STARPU_MAXCUDADEVS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_cuda_worker_get_count(void);

/**
   Return the number of HIP devices controlled by StarPU. The return
   value should be at most \ref STARPU_MAXHIPDEVS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_hip_worker_get_count(void);

/**
   Return the number of OpenCL devices controlled by StarPU. The
   return value should be at most \ref STARPU_MAXOPENCLDEVS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_opencl_worker_get_count(void);

/**
   Return the number of MPI Master Slave workers controlled by StarPU.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_mpi_ms_worker_get_count(void);

/**
   Return the number of TCPIP Master Slave workers controlled by StarPU.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_tcpip_ms_worker_get_count(void);

/**
   Return the identifier of the current worker, i.e the one associated
   to the calling thread. The return value is either \c -1 if the
   current context is not a StarPU worker (i.e. when called from the
   application outside a task or a callback), or an integer between \c
   0 and starpu_worker_get_count() - \c 1.
   See \ref HowToInitializeAComputationLibraryOnceForEachWorker for more details.
*/
int starpu_worker_get_id(void);

unsigned _starpu_worker_get_id_check(const char *f, int l);

/**
   Similar to starpu_worker_get_id(), but abort when called from
   outside a worker (i.e. when starpu_worker_get_id() would return \c
   -1).
   See \ref HowToInitializeAComputationLibraryOnceForEachWorker for more details.
*/
unsigned starpu_worker_get_id_check(void);

#define starpu_worker_get_id_check() _starpu_worker_get_id_check(__FILE__, __LINE__)

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_bindid(int workerid);

/**
   See \ref SchedulingHelpers for more details.
*/
void starpu_sched_find_all_worker_combinations(void);

/**
   Return the type of processing unit associated to the worker \p id.
   The worker identifier is a value returned by the function
   starpu_worker_get_id()). The return value indicates the
   architecture of the worker: ::STARPU_CPU_WORKER for a CPU core,
   ::STARPU_CUDA_WORKER for a CUDA device, and ::STARPU_OPENCL_WORKER
   for a OpenCL device. The return value for an invalid identifier is
   unspecified.
   See \ref TopologyWorkers for more details.
*/
enum starpu_worker_archtype starpu_worker_get_type(int id);

/**
   Return the number of workers of \p type. A positive (or
   <c>NULL</c>) value is returned in case of success, <c>-EINVAL</c>
   indicates that \p type is not valid otherwise.
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);

/**
   Get the list of identifiers of workers of \p type. Fill the array
   \p workerids with the identifiers of the \p workers. The argument
   \p maxsize indicates the size of the array \p workerids. The return
   value gives the number of identifiers that were put in the array.
   <c>-ERANGE</c> is returned is \p maxsize is lower than the number
   of workers with the appropriate type: in that case, the array is
   filled with the \p maxsize first elements. To avoid such overflows,
   the value of maxsize can be chosen by the means of the function
   starpu_worker_get_count_by_type(), or by passing a value greater or
   equal to \ref STARPU_NMAXWORKERS.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, unsigned maxsize);

/**
   Return the identifier of the \p num -th worker that has the
   specified \p type. If there is no such worker, -1 is returned.
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

/**
   Return the identifier of the worker that has the specified \p type
   and device id \p devid (which may not be the n-th, if some devices
   are skipped for instance). If there is no such worker, \c -1 is
   returned.
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

/**
   Return true if worker type can execute this task.
   See \ref SchedulingHelpers for more details.
*/
unsigned starpu_worker_type_can_execute_task(enum starpu_worker_archtype worker_type, const struct starpu_task *task);

/**
   Get the name of the worker \p id. StarPU associates a unique human
   readable string to each processing unit. This function copies at
   most the \p maxlen first bytes of the unique string associated to
   the worker \p id into the \p dst buffer. The caller is responsible
   for ensuring that \p dst is a valid pointer to a buffer of \p
   maxlen bytes at least. Calling this function on an invalid
   identifier results in an unspecified behaviour.
   See \ref TopologyWorkers for more details.
*/
void starpu_worker_get_name(int id, char *dst, size_t maxlen);

/**
   Display on \p output the list (if any) of all workers.
   See \ref TopologyWorkers for more details.
*/
void starpu_worker_display_all(FILE *output);

/**
   Display on \p output the list (if any) of all the workers of the
   given \p type.
   See \ref TopologyWorkers for more details.
*/
void starpu_worker_display_names(FILE *output, enum starpu_worker_archtype type);

/**
   Display on \p output the number of workers of the given \p type.
   See \ref TopologyWorkers for more details.
*/
void starpu_worker_display_count(FILE *output, enum starpu_worker_archtype type);

/**
   Return the device id of the worker \p id. The worker should be
   identified with the value returned by the starpu_worker_get_id()
   function. In the case of a CUDA worker, this device identifier is
   the logical device identifier exposed by CUDA (used by the function
   \c cudaGetDevice() for instance). The device identifier of a CPU
   worker is the logical identifier of the core on which the worker
   was bound; this identifier is either provided by the OS or by the
   library <c>hwloc</c> in case it is available.
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_devid(int id);

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_devnum(int id);

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_subworkerid(int id);

/**
   See \ref TopologyWorkers for more details.
*/
struct starpu_tree *starpu_workers_get_tree(void);

/**
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_get_sched_ctx_list(int worker, unsigned **sched_ctx);

/**
   Return when the current task is expected to be finished.

   Note: the returned date should be used with caution since the task might very
   well end just after this function returns.

   See \ref Per-taskFeedback for more details.
 */
void starpu_worker_get_current_task_exp_end(unsigned workerid, struct timespec *date);

/**
   Return whether worker \p workerid is currently blocked in a parallel task.
   See \ref SchedulingHelpers for more details.
 */
unsigned starpu_worker_is_blocked_in_parallel(int workerid);

/**
   See \ref SchedulingHelpers for more details.
 */
unsigned starpu_worker_is_slave_somewhere(int workerid);

/**
   Return worker \p type as a string.
   See \ref TopologyWorkers for more details.
*/
const char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type);

/**
   Return worker \p type from a string.
   Returns STARPU_UNKNOWN_WORKER if the string doesn't match a worker type.
   See \ref TopologyWorkers for more details.
*/
enum starpu_worker_archtype starpu_worker_get_type_from_string(const char *type);

/**
   Return worker \p type as a string suitable for environment variable names (CPU, CUDA, etc.).
   See \ref TopologyWorkers for more details.
*/
const char *starpu_worker_get_type_as_env_var(enum starpu_worker_archtype type);

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_bindid_get_workerids(int bindid, int **workerids);

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_devids(enum starpu_worker_archtype type, int *devids, int num);

/**
   See \ref TopologyWorkers for more details.
*/
int starpu_worker_get_stream_workerids(unsigned devid, int *workerids, enum starpu_worker_archtype type);

#ifdef STARPU_HAVE_HWLOC
/**
   If StarPU was compiled with \c hwloc support, return a duplicate of
   the \c hwloc cpuset associated with the worker \p workerid. The
   returned cpuset is obtained from a \c hwloc_bitmap_dup() function
   call. It must be freed by the caller using \c hwloc_bitmap_free().
   See \ref InteroperabilityHWLOC for more details.
*/
hwloc_cpuset_t starpu_worker_get_hwloc_cpuset(int workerid);
/**
   If StarPU was compiled with \c hwloc support, return the \c hwloc
   object corresponding to  the worker \p workerid.
   See \ref SchedulingHelpers for more details.
*/
hwloc_obj_t starpu_worker_get_hwloc_obj(int workerid);
#endif

/**
   See \ref TopologyMemory for more details.
*/
int starpu_memory_node_get_devid(unsigned node);

/**
   Find the memory node associated to the device identified by /p devid and
   /p kind
*/
unsigned starpu_memory_devid_find_node(int devid, enum starpu_node_kind kind);

/**
   Return the memory node associated to the current worker
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_get_local_memory_node(void);

/**
   Return the identifier of the memory node associated to the worker
   identified by \p workerid.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_worker_get_memory_node(unsigned workerid);

/**
   Return the number of memory nodes.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_memory_nodes_get_count(void);

/**
   Return the number of memory nodes of a given \p kind.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_memory_nodes_get_count_by_kind(enum starpu_node_kind kind);

/**
   Get the list of memory nodes of kind \p kind.
   Fill the array \p memory_nodes_ids with the memory nodes numbers.
   The argument \p maxsize indicates the size of the array
   \p memory_nodes_ids. The return value gives the number of node numbers
   that were put in the array. <c>-ERANGE</c> is returned if \p maxsize
   is lower than the number of memory nodes with the appropriate kind: in that
   case, the array is filled with the \p maxsize first elements. To avoid such
   overflows, the value of maxsize can be chosen by the means of function
   starpu_memory_nodes_get_count_by_kind(), or by passing a value greater or
   equal to \ref STARPU_MAXNODES.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_memory_node_get_ids_by_type(enum starpu_node_kind kind, unsigned *memory_nodes_ids, unsigned maxsize);

/**
   Return in \p name the name of a memory node (NUMA 0, CUDA 0, etc.)
   \p size is the size of the \p name array.
   See \ref TopologyWorkers for more details.
*/
int starpu_memory_node_get_name(unsigned node, char *name, size_t size);

/**
   Return the number of NUMA nodes used by StarPU.
   See \ref TopologyWorkers for more details.
*/
unsigned starpu_memory_nodes_get_numa_count(void);

/**
   Return the identifier of the memory node associated to the NUMA
   node identified by \p osid by the Operating System.
   See \ref TopologyWorkers for more details.
*/
int starpu_memory_nodes_numa_id_to_devid(int osid);

/**
   Return the Operating System identifier of the memory node whose
   StarPU identifier is \p id.
   See \ref TopologyWorkers for more details.
*/
int starpu_memory_nodes_numa_devid_to_id(unsigned id);

/**
   Return the type of \p node as defined by ::starpu_node_kind. For
   example, when defining a new data interface, this function should
   be used in the allocation function to determine on which device the
   memory needs to be allocated.
   See \ref TopologyWorkers for more details.
*/
enum starpu_node_kind starpu_node_get_kind(unsigned node);

/**
   Return whether \p node needs the use a memory offset, i.e. the value returned
   by the allocation method is not a pointer, but some driver handle, and an
   offset needs to be used to access within the allocated area. This is for
   instance the case with OpenCL.
*/
int starpu_node_needs_offset(unsigned node);

/**
   Return the type of worker which operates on memory node kind \p node_kind.
   See \ref TopologyWorkers for more details.
  */
enum starpu_worker_archtype starpu_memory_node_get_worker_archtype(enum starpu_node_kind node_kind);

/**
   Return the type of memory node that arch type \p type operates on.
   See \ref TopologyWorkers for more details.
  */
enum starpu_node_kind starpu_worker_get_memory_node_kind(enum starpu_worker_archtype type);

/**
   @name Scheduling operations
   @{
*/

/**
   Return \c !0 if current worker has a scheduling operation in
   progress, and \c 0 otherwise.
*/
int starpu_worker_sched_op_pending(void);

/**
   Allow other threads and workers to temporarily observe the current
   worker state, even though it is performing a scheduling operation.
   Must be called by a worker before performing a potentially blocking
   call such as acquiring a mutex other than its own sched_mutex. This
   function increases \c state_relax_refcnt from the current worker.
   No more than <c>UINT_MAX-1</c> nested starpu_worker_relax_on()
   calls should performed on the same worker. This function is
   automatically called by  starpu_worker_lock() to relax the caller
   worker state while attempting to lock the target worker.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_relax_on(void);

/**
   Must be called after a potentially blocking call is complete, to
   restore the relax state in place before the corresponding
   starpu_worker_relax_on(). Decreases \c state_relax_refcnt. Calls to
   starpu_worker_relax_on() and starpu_worker_relax_off() must be
   properly paired. This function is automatically called by
   starpu_worker_unlock() after the target worker has been unlocked.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_relax_off(void);

/**
   Return \c !0 if the current worker \c state_relax_refcnt!=0 and \c
   0 otherwise.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
int starpu_worker_get_relax_state(void);

/**
   Acquire the sched mutex of \p workerid. If the caller is a worker,
   distinct from \p workerid, the caller worker automatically enters a
   relax state while acquiring the target worker lock.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_lock(int workerid);

/**
   Attempt to acquire the sched mutex of \p workerid. Returns \c 0 if
   successful, \c !0 if \p workerid sched mutex is held or the
   corresponding worker is not in a relax state. If the caller is a
   worker, distinct from \p workerid, the caller worker automatically
   enters relax state if successfully acquiring the target worker lock.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
int starpu_worker_trylock(int workerid);

/**
   Release the previously acquired sched mutex of \p workerid. Restore
   the relax state of the caller worker if needed.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_unlock(int workerid);

/**
   Acquire the current worker sched mutex.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_lock_self(void);

/**
   Release the current worker sched mutex.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
void starpu_worker_unlock_self(void);

#ifdef STARPU_WORKER_CALLBACKS
/**
   If StarPU was compiled with blocking drivers support and worker
   callbacks support enabled, allow to specify an external resource
   manager callback to be notified about workers going to sleep.
   See \ref SchedulingHelpers for more details.
*/
void starpu_worker_set_going_to_sleep_callback(void (*callback)(unsigned workerid));

/**
   If StarPU was compiled with blocking drivers support and worker
   callbacks support enabled, allow to specify an external resource
   manager callback to be notified about workers waking-up.
   See \ref SchedulingHelpers for more details.
*/
void starpu_worker_set_waking_up_callback(void (*callback)(unsigned workerid));
#endif

/** @} */

/** @} */

/**
   @defgroup API_Parallel_Tasks Parallel Tasks
   @{
*/

/**
   Return the number of different combined workers.
   See \ref SchedulingHelpers for more details.
*/
unsigned starpu_combined_worker_get_count(void);
/**
   See \ref SchedulingHelpers for more details.
*/
unsigned starpu_worker_is_combined_worker(int id);

/**
   Return the identifier of the current combined worker.
   See \ref SchedulingHelpers for more details.
*/
int starpu_combined_worker_get_id(void);

/**
   Return the size of the current combined worker, i.e. the total
   number of CPUS running the same task in the case of ::STARPU_SPMD
   parallel tasks, or the total number of threads that the task is
   allowed to start in the case of ::STARPU_FORKJOIN parallel tasks.
   See \ref Fork-modeParallelTasks and \ref SPMD-modeParallelTasks for more details.
*/
int starpu_combined_worker_get_size(void);

/**
   Return the rank of the current thread within the combined worker.
   Can only be used in ::STARPU_SPMD parallel tasks, to know which
   part of the task to work on.
   See \ref SPMD-modeParallelTasks for more details.
*/
int starpu_combined_worker_get_rank(void);

/**
   Register a new combined worker and get its identifier.
   See \ref SchedulingHelpers for more details.
*/
int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[]);

/**
   Get the description of a combined worker.
   See \ref SchedulingHelpers for more details.
*/
int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid);

/**
   Variant of starpu_worker_can_execute_task() compatible with
   combined workers.
   See \ref DefiningANewBasicSchedulingPolicy for more details.
*/
int starpu_combined_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

/**
   Initialise the barrier for the parallel task, and dispatch the task
   between the different workers of the given combined worker.
   See \ref SchedulingHelpers for more details.
 */
void starpu_parallel_task_barrier_init(struct starpu_task *task, int workerid);

/**
   Initialise the barrier for the parallel task, to be pushed to \p
   worker_size workers (without having to explicit a given combined
   worker).
   See \ref SchedulingHelpers for more details.
*/
void starpu_parallel_task_barrier_init_n(struct starpu_task *task, int worker_size);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_WORKER_H__ */
