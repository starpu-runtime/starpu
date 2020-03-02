/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_thread.h>
#include <starpu_task.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Workers_Properties Workers’ Properties
   @{
*/

/**
  Memory node Type
*/
enum starpu_node_kind
{
	STARPU_UNUSED=0,
	STARPU_CPU_RAM=1,
	STARPU_CUDA_RAM=2,
	STARPU_OPENCL_RAM=3,
	STARPU_DISK_RAM=4,
	STARPU_MIC_RAM=5,
	STARPU_MPI_MS_RAM=6
};

/**
   Worker Architecture Type

   The value 4 which was used by the driver SCC is no longer used as
   renumbering workers would make unusable old performance model
   files.
*/
enum starpu_worker_archtype
{
	STARPU_CPU_WORKER=0,        /**< CPU core */
	STARPU_CUDA_WORKER=1,       /**< NVIDIA CUDA device */
	STARPU_OPENCL_WORKER=2,     /**< OpenCL device */
	STARPU_MIC_WORKER=3,        /**< Intel MIC device */
	STARPU_MPI_MS_WORKER=5,     /**< MPI Slave device */
	STARPU_ANY_WORKER=6         /**< any worker, used in the hypervisor */
};

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
	STARPU_WORKER_TREE,  /**< The collection is a tree */
	STARPU_WORKER_LIST   /**< The collection is an array */
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
	   Deinitialize the colection
	*/
	void (*deinit)(struct starpu_worker_collection *workers);
	/**
	   Initialize the cursor if there is one
	*/
	void (*init_iterator)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it);
	void (*init_iterator_for_parallel_tasks)(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it, struct starpu_task *task);
};

extern struct starpu_worker_collection worker_list;
extern struct starpu_worker_collection worker_tree;

/**
   Return the number of workers (i.e. processing units executing
   StarPU tasks). The return value should be at most \ref
   STARPU_NMAXWORKERS.
*/
unsigned starpu_worker_get_count(void);

/**
   Return the number of CPUs controlled by StarPU. The return value
   should be at most \ref STARPU_MAXCPUS.
*/
unsigned starpu_cpu_worker_get_count(void);

/**
   Return the number of CUDA devices controlled by StarPU. The return
   value should be at most \ref STARPU_MAXCUDADEVS.
*/
unsigned starpu_cuda_worker_get_count(void);

/**
   Return the number of OpenCL devices controlled by StarPU. The
   return value should be at most \ref STARPU_MAXOPENCLDEVS.
*/
unsigned starpu_opencl_worker_get_count(void);

/**
   Return the number of MIC workers controlled by StarPU.
*/
unsigned starpu_mic_worker_get_count(void);

/**
   Return the number of MPI Master Slave workers controlled by StarPU.
*/
unsigned starpu_mpi_ms_worker_get_count(void);

/**
   Return the number of MIC devices controlled by StarPU. The return
   value should be at most \ref STARPU_MAXMICDEVS.
*/
unsigned starpu_mic_device_get_count(void);

/**
   Return the identifier of the current worker, i.e the one associated
   to the calling thread. The return value is either \c -1 if the
   current context is not a StarPU worker (i.e. when called from the
   application outside a task or a callback), or an integer between \c
   0 and starpu_worker_get_count() - \c 1.
*/
int starpu_worker_get_id(void);

unsigned _starpu_worker_get_id_check(const char *f, int l);

/**
   Similar to starpu_worker_get_id(), but abort when called from
   outside a worker (i.e. when starpu_worker_get_id() would return \c
   -1).
*/
unsigned starpu_worker_get_id_check(void);

#define starpu_worker_get_id_check() _starpu_worker_get_id_check(__FILE__, __LINE__)
int starpu_worker_get_bindid(int workerid);

void starpu_sched_find_all_worker_combinations(void);

/**
   Return the type of processing unit associated to the worker \p id.
   The worker identifier is a value returned by the function
   starpu_worker_get_id()). The return value indicates the
   architecture of the worker: ::STARPU_CPU_WORKER for a CPU core,
   ::STARPU_CUDA_WORKER for a CUDA device, and ::STARPU_OPENCL_WORKER
   for a OpenCL device. The return value for an invalid identifier is
   unspecified.
*/
enum starpu_worker_archtype starpu_worker_get_type(int id);

/**
   Return the number of workers of \p type. A positive (or
   <c>NULL</c>) value is returned in case of success, <c>-EINVAL</c>
   indicates that \p type is not valid otherwise.
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
*/
unsigned starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, unsigned maxsize);

/**
   Return the identifier of the \p num -th worker that has the
   specified \p type. If there is no such worker, -1 is returned.
*/
int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);

/**
   Return the identifier of the worker that has the specified \p type
   and device id \p devid (which may not be the n-th, if some devices
   are skipped for instance). If there is no such worker, \c -1 is
   returned.
*/
int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);

/**
   Get the name of the worker \p id. StarPU associates a unique human
   readable string to each processing unit. This function copies at
   most the \p maxlen first bytes of the unique string associated to
   the worker \p id into the \p dst buffer. The caller is responsible
   for ensuring that \p dst is a valid pointer to a buffer of \p
   maxlen bytes at least. Calling this function on an invalid
   identifier results in an unspecified behaviour.
*/
void starpu_worker_get_name(int id, char *dst, size_t maxlen);

/**
   Display on \p output the list (if any) of all the workers of the
   given \p type.
*/
void starpu_worker_display_names(FILE *output, enum starpu_worker_archtype type);

/**
   Return the device id of the worker \p id. The worker should be
   identified with the value returned by the starpu_worker_get_id()
   function. In the case of a CUDA worker, this device identifier is
   the logical device identifier exposed by CUDA (used by the function
   \c cudaGetDevice() for instance). The device identifier of a CPU
   worker is the logical identifier of the core on which the worker
   was bound; this identifier is either provided by the OS or by the
   library <c>hwloc</c> in case it is available.
*/
int starpu_worker_get_devid(int id);

int starpu_worker_get_mp_nodeid(int id);

struct starpu_tree* starpu_workers_get_tree(void);

unsigned starpu_worker_get_sched_ctx_list(int worker, unsigned **sched_ctx);

unsigned starpu_worker_is_blocked_in_parallel(int workerid);

unsigned starpu_worker_is_slave_somewhere(int workerid);

/**
   Return worker \p type as a string.
*/
char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type);

int starpu_bindid_get_workerids(int bindid, int **workerids);

int starpu_worker_get_devids(enum starpu_worker_archtype type, int *devids, int num);

int starpu_worker_get_stream_workerids(unsigned devid, int *workerids, enum starpu_worker_archtype type);

unsigned starpu_worker_get_sched_ctx_id_stream(unsigned stream_workerid);

#ifdef STARPU_HAVE_HWLOC
/**
   If StarPU was compiled with \c hwloc support, return a duplicate of
   the \c hwloc cpuset associated with the worker \p workerid. The
   returned cpuset is obtained from a \c hwloc_bitmap_dup() function
   call. It must be freed by the caller using \c hwloc_bitmap_free().
*/
hwloc_cpuset_t starpu_worker_get_hwloc_cpuset(int workerid);
/**
   If StarPU was compiled with \c hwloc support, return the \c hwloc
   object corresponding to  the worker \p workerid.
*/
hwloc_obj_t starpu_worker_get_hwloc_obj(int workerid);
#endif

int starpu_memory_node_get_devid(unsigned node);

/**
   Return the memory node associated to the current worker
*/
unsigned starpu_worker_get_local_memory_node(void);

/**
   Return the identifier of the memory node associated to the worker
   identified by \p workerid.
*/
unsigned starpu_worker_get_memory_node(unsigned workerid);

unsigned starpu_memory_nodes_get_count(void);
int starpu_memory_node_get_name(unsigned node, char *name, size_t size);
int starpu_memory_nodes_get_numa_count(void);

/**
   Return the identifier of the memory node associated to the NUMA
   node identified by \p osid by the Operating System.
*/
int starpu_memory_nodes_numa_id_to_devid(int osid);

/**
   Return the Operating System identifier of the memory node whose
   StarPU identifier is \p id.
*/
int starpu_memory_nodes_numa_devid_to_id(unsigned id);

/**
   Return the type of \p node as defined by ::starpu_node_kind. For
   example, when defining a new data interface, this function should
   be used in the allocation function to determine on which device the
   memory needs to be allocated.
*/
enum starpu_node_kind starpu_node_get_kind(unsigned node);

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
*/
void starpu_worker_relax_on(void);

/**
   Must be called after a potentially blocking call is complete, to
   restore the relax state in place before the corresponding
   starpu_worker_relax_on(). Decreases \c state_relax_refcnt. Calls to
   starpu_worker_relax_on() and starpu_worker_relax_off() must be
   properly paired. This function is automatically called by
   starpu_worker_unlock() after the target worker has been unlocked.
*/
void starpu_worker_relax_off(void);

/**
   Return \c !0 if the current worker \c state_relax_refcnt!=0 and \c
   0 otherwise.
*/
int starpu_worker_get_relax_state(void);

/**
   Acquire the sched mutex of \p workerid. If the caller is a worker,
   distinct from \p workerid, the caller worker automatically enters a
   relax state while acquiring the target worker lock.
*/
void starpu_worker_lock(int workerid);

/**
   Attempt to acquire the sched mutex of \p workerid. Returns \c 0 if
   successful, \c !0 if \p workerid sched mutex is held or the
   corresponding worker is not in a relax state. If the caller is a
   worker, distinct from \p workerid, the caller worker automatically
   enters relax state if successfully acquiring the target worker lock.
*/
int starpu_worker_trylock(int workerid);

/**
   Release the previously acquired sched mutex of \p workerid. Restore
   the relax state of the caller worker if needed.
*/
void starpu_worker_unlock(int workerid);

/**
   Acquire the current worker sched mutex.
*/
void starpu_worker_lock_self(void);

/**
   Release the current worker sched mutex.
*/
void starpu_worker_unlock_self(void);

#ifdef STARPU_WORKER_CALLBACKS
/**
   If StarPU was compiled with blocking drivers support and worker
   callbacks support enabled, allow to specify an external resource
   manager callback to be notified about workers going to sleep.
*/
void starpu_worker_set_going_to_sleep_callback(void (*callback)(unsigned workerid));

/**
   If StarPU was compiled with blocking drivers support and worker
   callbacks support enabled, allow to specify an external resource
   manager callback to be notified about workers waking-up.
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
*/
unsigned starpu_combined_worker_get_count(void);
unsigned starpu_worker_is_combined_worker(int id);

/**
   Return the identifier of the current combined worker.
*/
int starpu_combined_worker_get_id(void);

/**
   Return the size of the current combined worker, i.e. the total
   number of CPUS running the same task in the case of ::STARPU_SPMD
   parallel tasks, or the total number of threads that the task is
   allowed to start in the case of ::STARPU_FORKJOIN parallel tasks.
*/
int starpu_combined_worker_get_size(void);

/**
   Return the rank of the current thread within the combined worker.
   Can only be used in ::STARPU_SPMD parallel tasks, to know which
   part of the task to work on.
*/
int starpu_combined_worker_get_rank(void);

/**
   Register a new combined worker and get its identifier
*/
int starpu_combined_worker_assign_workerid(int nworkers, int workerid_array[]);

/**
   Get the description of a combined worker
*/
int starpu_combined_worker_get_description(int workerid, int *worker_size, int **combined_workerid);

/**
   Variant of starpu_worker_can_execute_task() compatible with
   combined workers
*/
int starpu_combined_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl);

/**
   Initialise the barrier for the parallel task, and dispatch the task
   between the different workers of the given combined worker.
 */
void starpu_parallel_task_barrier_init(struct starpu_task *task, int workerid);

/**
   Initialise the barrier for the parallel task, to be pushed to \p
   worker_size workers (without having to explicit a given combined
   worker).
*/
void starpu_parallel_task_barrier_init_n(struct starpu_task *task, int worker_size);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_WORKER_H__ */
