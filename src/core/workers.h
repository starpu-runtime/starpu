/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __WORKERS_H__
#define __WORKERS_H__

/** \addtogroup workers */
/* @{ */

#include <limits.h>

#include <starpu.h>
#include <common/config.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <common/thread.h>
#include <common/utils.h>
#include <core/jobs.h>
#include <core/perfmodel/perfmodel.h>
#include <core/sched_policy.h>
#include <core/topology.h>
#include <core/errorcheck.h>
#include <core/sched_ctx.h>
#include <core/sched_ctx_list.h>
#include <core/simgrid.h>
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#include <core/drivers.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>

#ifdef STARPU_USE_MIC
#include <drivers/mic/driver_mic_source.h>
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_MPI_MASTER_SLAVE
#include <drivers/mpi/driver_mpi_source.h>
#endif

#include <drivers/cpu/driver_cpu.h>

#include <datawizard/datawizard.h>

#include <starpu_parameters.h>

#define STARPU_MAX_PIPELINE 4

enum initialization { UNINITIALIZED = 0, CHANGING, INITIALIZED };

struct _starpu_ctx_change_list;

/** This is initialized by _starpu_worker_init() */
LIST_TYPE(_starpu_worker,
	struct _starpu_machine_config *config;
        starpu_pthread_mutex_t mutex;
	enum starpu_worker_archtype arch; /**< what is the type of worker ? */
	uint32_t worker_mask; /**< what is the type of worker ? */
	struct starpu_perfmodel_arch perf_arch; /**< in case there are different models of the same arch */
	starpu_pthread_t worker_thread; /**< the thread which runs the worker */
	unsigned devid; /**< which cpu/gpu/etc is controlled by the worker ? */
	unsigned subworkerid; /**< which sub-worker this one is for the cpu/gpu */
	int bindid; /**< which cpu is the driver bound to ? (logical index) */
	int workerid; /**< uniquely identify the worker among all processing units types */
	int combined_workerid; /**< combined worker currently using this worker */
	int current_rank; /**< current rank in case the worker is used in a parallel fashion */
	int worker_size; /**< size of the worker in case we use a combined worker */
	starpu_pthread_cond_t started_cond; /**< indicate when the worker is ready */
	starpu_pthread_cond_t ready_cond; /**< indicate when the worker is ready */
	unsigned memory_node; /**< which memory node is the worker associated with ? */
	unsigned numa_memory_node; /**< which numa memory node is the worker associated with? (logical index) */
	  /**
	   * condition variable used for passive waiting operations on worker
	   * STARPU_PTHREAD_COND_BROADCAST must be used instead of STARPU_PTHREAD_COND_SIGNAL,
	   * since the condition is shared for multiple purpose */
	starpu_pthread_cond_t sched_cond;
        starpu_pthread_mutex_t sched_mutex; /**< mutex protecting sched_cond */
	unsigned state_relax_refcnt; /**< mark scheduling sections where other workers can safely access the worker state */
#ifdef STARPU_SPINLOCK_CHECK
	const char *relax_on_file;
	int relax_on_line;
	const char *relax_on_func;
	const char *relax_off_file;
	int relax_off_line;
	const char *relax_off_func;
#endif
	unsigned state_sched_op_pending; /**< a task pop is ongoing even though sched_mutex may temporarily be unlocked */
	unsigned state_changing_ctx_waiting; /**< a thread is waiting for operations such as pop to complete before acquiring sched_mutex and modifying the worker ctx*/
	unsigned state_changing_ctx_notice; /**< the worker ctx is about to change or being changed, wait for flag to be cleared before starting new scheduling operations */
	unsigned state_blocked_in_parallel; /**< worker is currently blocked on a parallel section */
	unsigned state_blocked_in_parallel_observed; /**< the blocked state of the worker has been observed by another worker during a relaxed section */
	unsigned state_block_in_parallel_req; /**< a request for state transition from unblocked to blocked is pending */
	unsigned state_block_in_parallel_ack; /**< a block request has been honored */
	unsigned state_unblock_in_parallel_req; /**< a request for state transition from blocked to unblocked is pending */
	unsigned state_unblock_in_parallel_ack; /**< an unblock request has been honored */
	  /**
	   * cumulative blocking depth
	   * - =0  worker unblocked
	   * - >0  worker blocked
	   * - transition from 0 to 1 triggers a block_req
	   * - transition from 1 to 0 triggers a unblock_req
	   */
	unsigned block_in_parallel_ref_count;
	starpu_pthread_t thread_changing_ctx; /**< thread currently changing a sched_ctx containing the worker */
	  /**
	     list of deferred context changes
	     *
	     * when the current thread is a worker, _and_ this worker is in a
	     * scheduling operation, new ctx changes are queued to this list for
	     * subsequent processing once worker completes the ongoing scheduling
	     * operation */
	struct _starpu_ctx_change_list ctx_change_list;
	struct starpu_task_list local_tasks; /**< this queue contains tasks that have been explicitely submitted to that queue */
	struct starpu_task **local_ordered_tasks; /**< this queue contains tasks that have been explicitely submitted to that queue with an explicit order */
	unsigned local_ordered_tasks_size; /**< this records the size of local_ordered_tasks */
	unsigned current_ordered_task; /**< this records the index (within local_ordered_tasks) of the next ordered task to be executed */
	unsigned current_ordered_task_order; /**< this records the order of the next ordered task to be executed */
	struct starpu_task *current_task; /**< task currently executed by this worker (non-pipelined version) */
	struct starpu_task *current_tasks[STARPU_MAX_PIPELINE]; /**< tasks currently executed by this worker (pipelined version) */
#ifdef STARPU_SIMGRID
	starpu_pthread_wait_t wait;
#endif

	struct timespec cl_start; /**< Codelet start time of the task currently running */
	struct timespec cl_end; /**< Codelet end time of the last task running */
	unsigned char first_task; /**< Index of first task in the pipeline */
	unsigned char ntasks; /**< number of tasks in the pipeline */
	unsigned char pipeline_length; /**< number of tasks to be put in the pipeline */
	unsigned char pipeline_stuck; /**< whether a task prevents us from pipelining */
	struct _starpu_worker_set *set; /**< in case this worker belongs to a set */
	unsigned worker_is_running;
	unsigned worker_is_initialized;
	enum _starpu_worker_status status; /**< what is the worker doing now ? (eg. CALLBACK) */
	unsigned state_keep_awake; /**< !0 if a task has been pushed to the worker and the task has not yet been seen by the worker, the worker should no go to sleep before processing this task*/
	char name[128];
	char short_name[32];
	unsigned run_by_starpu; /**< Is this run by StarPU or directly by the application ? */
	struct _starpu_driver_ops *driver_ops;

	struct _starpu_sched_ctx_list *sched_ctx_list;
	int tmp_sched_ctx;
	unsigned nsched_ctxs; /**< the no of contexts a worker belongs to*/
	struct _starpu_barrier_counter tasks_barrier; /**< wait for the tasks submitted */

	unsigned has_prev_init; /**< had already been inited in another ctx */

	unsigned removed_from_ctx[STARPU_NMAX_SCHED_CTXS+1];

	unsigned spinning_backoff ; /**< number of cycles to pause when spinning  */

	unsigned nb_buffers_transferred; /**< number of piece of data already send to worker */
	unsigned nb_buffers_totransfer; /**< number of piece of data already send to worker */
	struct starpu_task *task_transferring; /**< The buffers of this task are being sent */

	  /**
	   * indicate whether the workers shares tasks lists with other workers
	   * in this case when removing him from a context it disapears instantly
	   */
	unsigned shares_tasks_lists[STARPU_NMAX_SCHED_CTXS+1];

	unsigned poped_in_ctx[STARPU_NMAX_SCHED_CTXS+1]; 	  /**< boolean to chose the next ctx a worker will pop into */

	  /**
	   * boolean indicating at which moment we checked all ctxs and change phase for the booleab poped_in_ctx
	   * one for each of the 2 priorities
	   */
	unsigned reverse_phase[2];

	unsigned pop_ctx_priority; 	  /**< indicate which priority of ctx is currently active: the values are 0 or 1*/
	unsigned is_slave_somewhere; 	  /**< bool to indicate if the worker is slave in a ctx */

	struct _starpu_sched_ctx *stream_ctx;

#ifdef __GLIBC__
	cpu_set_t cpu_set;
#endif /* __GLIBC__ */
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t hwloc_cpu_set;
	hwloc_obj_t hwloc_obj;
#endif

	/** Keep this last, to make sure to separate worker data in separate cache lines. */
	char padding[STARPU_CACHELINE_SIZE];
);

struct _starpu_combined_worker
{
	struct starpu_perfmodel_arch perf_arch; 	 /**< in case there are different models of the same arch */
	uint32_t worker_mask; /**< what is the type of workers ? */
	int worker_size;
	unsigned memory_node; 	 /**< which memory node is associated that worker to ? */
	int combined_workerid[STARPU_NMAXWORKERS];
#ifdef STARPU_USE_MP
	int count;
	starpu_pthread_mutex_t count_mutex;
#endif

#ifdef __GLIBC__
	cpu_set_t cpu_set;
#endif /* __GLIBC__ */
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t hwloc_cpu_set;
#endif

	/** Keep this last, to make sure to separate worker data in separate
	  cache lines. */
	char padding[STARPU_CACHELINE_SIZE];
};

/**
 * in case a single CPU worker may control multiple
 * accelerators
*/
struct _starpu_worker_set
{
        starpu_pthread_mutex_t mutex;
	starpu_pthread_t worker_thread; /**< the thread which runs the worker */
	unsigned nworkers;
	unsigned started; /**< Only one thread for the whole set */
	void *retval;
	struct _starpu_worker *workers;
        starpu_pthread_cond_t ready_cond; /**< indicate when the set is ready */
	unsigned set_is_initialized;
};

#ifdef STARPU_USE_MPI_MASTER_SLAVE
extern struct _starpu_worker_set mpi_worker_set[STARPU_MAXMPIDEVS];
#endif

struct _starpu_machine_topology
{
	/** Total number of workers. */
	unsigned nworkers;

	/** Total number of combined workers. */
	unsigned ncombinedworkers;

	unsigned nsched_ctxs;

#ifdef STARPU_HAVE_HWLOC
	/** Topology as detected by hwloc. */
	hwloc_topology_t hwtopology;
#endif
	/** custom hwloc tree*/
	struct starpu_tree *tree;

	/** Total number of CPU cores, as detected by the topology code. May
	 * be different from the actual number of CPU workers.
	 */
	unsigned nhwcpus;

	/** Total number of PUs (i.e. threads), as detected by the topology code. May
	 * be different from the actual number of PU workers.
	 */
	unsigned nhwpus;

	/** Total number of CUDA devices, as detected. May be different
	 * from the actual number of CUDA workers.
	 */
	unsigned nhwcudagpus;

	/** Total number of OpenCL devices, as detected. May be
	 * different from the actual number of OpenCL workers.
	 */
	unsigned nhwopenclgpus;

	/** Total number of MPI nodes, as detected. May be different
	 * from the actual number of node workers.
	 */
	unsigned nhwmpi;

	/** Actual number of CPU workers used by StarPU. */
	unsigned ncpus;

	/** Actual number of CUDA GPUs used by StarPU. */
	unsigned ncudagpus;
	unsigned nworkerpercuda;
	int cuda_th_per_stream;
	int cuda_th_per_dev;

	/** Actual number of OpenCL workers used by StarPU. */
	unsigned nopenclgpus;

	/** Actual number of MPI workers used by StarPU. */
	unsigned nmpidevices;
        unsigned nhwmpidevices;

	unsigned nhwmpicores[STARPU_MAXMPIDEVS]; /**< Each MPI node has its set of cores. */
	unsigned nmpicores[STARPU_MAXMPIDEVS];

	/** Topology of MP nodes (MIC) as well as necessary
	 * objects to communicate with them. */
	unsigned nhwmicdevices;
	unsigned nmicdevices;

	unsigned nhwmiccores[STARPU_MAXMICDEVS]; /**< Each MIC node has its set of cores. */
	unsigned nmiccores[STARPU_MAXMICDEVS];

	/** Indicates the successive logical PU identifier that should be used
	 * to bind the workers. It is either filled according to the
	 * user's explicit parameters (from starpu_conf) or according
	 * to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over
	 * the cores.
	 */
	unsigned workers_bindid[STARPU_NMAXWORKERS];

	/** Indicates the successive CUDA identifier that should be
	 * used by the CUDA driver.  It is either filled according to
	 * the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CUDAID env. variable.
	 * Otherwise, they are taken in ID order.
	 */
	unsigned workers_cuda_gpuid[STARPU_NMAXWORKERS];

	/** Indicates the successive OpenCL identifier that should be
	 * used by the OpenCL driver.  It is either filled according
	 * to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_OPENCLID env. variable.
	 * Otherwise, they are taken in ID order.
	 */
	unsigned workers_opencl_gpuid[STARPU_NMAXWORKERS];

	/*** Indicates the successive MIC devices that should be used
	 * by the MIC driver.  It is either filled according to the
	 * user's explicit parameters (from starpu_conf) or according
	 * to the STARPU_WORKERS_MICID env. variable. Otherwise, they
	 * are taken in ID order. */
	/** TODO */
	/** unsigned workers_mic_deviceid[STARPU_NMAXWORKERS]; */

	unsigned workers_mpi_ms_deviceid[STARPU_NMAXWORKERS];
};

struct _starpu_machine_config
{
	struct _starpu_machine_topology topology;

#ifdef STARPU_HAVE_HWLOC
	int cpu_depth;
	int pu_depth;
#endif

	/** Where to bind next worker ? */
	int current_bindid;
	char currently_bound[STARPU_NMAXWORKERS];
	char currently_shared[STARPU_NMAXWORKERS];

	/** Which GPU(s) do we use for CUDA ? */
	int current_cuda_gpuid;

	/** Which GPU(s) do we use for OpenCL ? */
	int current_opencl_gpuid;

	/** Which MIC do we use? */
	int current_mic_deviceid;

	/** Which MPI do we use? */
	int current_mpi_deviceid;

	/** Memory node for cpus, if only one */
	int cpus_nodeid;
	/** Memory node for CUDA, if only one */
	int cuda_nodeid;
	/** Memory node for OpenCL, if only one */
	int opencl_nodeid;
	/** Memory node for MIC, if only one */
	int mic_nodeid;
	/** Memory node for MPI, if only one */
	int mpi_nodeid;

	/** Separate out previous variables from per-worker data. */
	char padding1[STARPU_CACHELINE_SIZE];

	/** Basic workers : each of this worker is running its own driver and
	 * can be combined with other basic workers. */
	struct _starpu_worker workers[STARPU_NMAXWORKERS];

	/** Combined workers: these worker are a combination of basic workers
	 * that can run parallel tasks together. */
	struct _starpu_combined_worker combined_workers[STARPU_NMAX_COMBINEDWORKERS];

	starpu_pthread_mutex_t submitted_mutex;

	/** Separate out previous mutex from the rest of the data. */
	char padding2[STARPU_CACHELINE_SIZE];

	/** Translation table from bindid to worker IDs */
	struct
	{
		int *workerids;
		unsigned nworkers; /**< size of workerids */
	} *bindid_workers;
	unsigned nbindid; /**< size of bindid_workers */

	/** This bitmask indicates which kinds of worker are available. For
	 * instance it is possible to test if there is a CUDA worker with
	 * the result of (worker_mask & STARPU_CUDA). */
	uint32_t worker_mask;

        /** either the user given configuration passed to starpu_init or a default configuration */
	struct starpu_conf conf;

	/** this flag is set until the runtime is stopped */
	unsigned running;

	int disable_kernels;

	/** Number of calls to starpu_pause() - calls to starpu_resume(). When >0,
	 * StarPU should pause. */
	int pause_depth;

	/** all the sched ctx of the current instance of starpu */
	struct _starpu_sched_ctx sched_ctxs[STARPU_NMAX_SCHED_CTXS+1];

	/** this flag is set until the application is finished submitting tasks */
	unsigned submitting;

	int watchdog_ok;
};

extern int _starpu_worker_parallel_blocks;

extern struct _starpu_machine_config _starpu_config STARPU_ATTRIBUTE_INTERNAL;
extern int _starpu_keys_initialized STARPU_ATTRIBUTE_INTERNAL;
extern starpu_pthread_key_t _starpu_worker_key STARPU_ATTRIBUTE_INTERNAL;
extern starpu_pthread_key_t _starpu_worker_set_key STARPU_ATTRIBUTE_INTERNAL;

/** Three functions to manage argv, argc */
void _starpu_set_argc_argv(int *argc, char ***argv);
int *_starpu_get_argc();
char ***_starpu_get_argv();

/** Fill conf with environment variables */
void _starpu_conf_check_environment(struct starpu_conf *conf);

/** Called by the driver when it is ready to pause  */
void _starpu_may_pause(void);

/** Has starpu_shutdown already been called ? */
static inline unsigned _starpu_machine_is_running(void)
{
	unsigned ret;
	/* running is just protected by a memory barrier */
	STARPU_RMB();

	ANNOTATE_HAPPENS_AFTER(&_starpu_config.running);
	ret = _starpu_config.running;
	ANNOTATE_HAPPENS_BEFORE(&_starpu_config.running);
	return ret;
}


/** initialise a worker */
void _starpu_worker_init(struct _starpu_worker *workerarg, struct _starpu_machine_config *pconfig);

/** Check if there is a worker that may execute the task. */
uint32_t _starpu_worker_exists(struct starpu_task *);

/** Is there a worker that can execute CUDA code ? */
uint32_t _starpu_can_submit_cuda_task(void);

/** Is there a worker that can execute CPU code ? */
uint32_t _starpu_can_submit_cpu_task(void);

/** Is there a worker that can execute OpenCL code ? */
uint32_t _starpu_can_submit_opencl_task(void);

/** Check whether there is anything that the worker should do instead of
 * sleeping (waiting on something to happen). */
unsigned _starpu_worker_can_block(unsigned memnode, struct _starpu_worker *worker);

/** This function must be called to block a worker. It puts the worker in a
 * sleeping state until there is some event that forces the worker to wake up.
 * */
void _starpu_block_worker(int workerid, starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex);

/** This function initializes the current driver for the given worker */
void _starpu_driver_start(struct _starpu_worker *worker, unsigned fut_key, unsigned sync);
/** This function initializes the current thread for the given worker */
void _starpu_worker_start(struct _starpu_worker *worker, unsigned fut_key, unsigned sync);

static inline unsigned _starpu_worker_get_count(void)
{
	return _starpu_config.topology.nworkers;
}
#define starpu_worker_get_count _starpu_worker_get_count

/** The _starpu_worker structure describes all the state of a StarPU worker.
 * This function sets the pthread key which stores a pointer to this structure.
 * */
static inline void _starpu_set_local_worker_key(struct _starpu_worker *worker)
{
	STARPU_ASSERT(_starpu_keys_initialized);
	STARPU_PTHREAD_SETSPECIFIC(_starpu_worker_key, worker);
}

/** Returns the _starpu_worker structure that describes the state of the
 * current worker. */
static inline struct _starpu_worker *_starpu_get_local_worker_key(void)
{
	if (!_starpu_keys_initialized)
		return NULL;
	return (struct _starpu_worker *) STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_key);
}

/** The _starpu_worker_set structure describes all the state of a StarPU worker_set.
 * This function sets the pthread key which stores a pointer to this structure.
 * */
static inline void _starpu_set_local_worker_set_key(struct _starpu_worker_set *worker)
{
	STARPU_ASSERT(_starpu_keys_initialized);
	STARPU_PTHREAD_SETSPECIFIC(_starpu_worker_set_key, worker);
}

/** Returns the _starpu_worker_set structure that describes the state of the
 * current worker_set. */
static inline struct _starpu_worker_set *_starpu_get_local_worker_set_key(void)
{
	if (!_starpu_keys_initialized)
		return NULL;
	return (struct _starpu_worker_set *) STARPU_PTHREAD_GETSPECIFIC(_starpu_worker_set_key);
}

/** Returns the _starpu_worker structure that describes the state of the
 * specified worker. */
static inline struct _starpu_worker *_starpu_get_worker_struct(unsigned id)
{
	STARPU_ASSERT(id < starpu_worker_get_count());
	return &_starpu_config.workers[id];
}

/** Returns the starpu_sched_ctx structure that describes the state of the
 * specified ctx */
static inline struct _starpu_sched_ctx *_starpu_get_sched_ctx_struct(unsigned id)
{
	return (id > STARPU_NMAX_SCHED_CTXS) ? NULL : &_starpu_config.sched_ctxs[id];
}

struct _starpu_combined_worker *_starpu_get_combined_worker_struct(unsigned id);

/** Returns the structure that describes the overall machine configuration (eg.
 * all workers and topology). */
static inline struct _starpu_machine_config *_starpu_get_machine_config(void)
{
	return &_starpu_config;
}

/** Return whether kernels should be run (<=0) or not (>0) */
static inline int _starpu_get_disable_kernels(void)
{
	return _starpu_config.disable_kernels;
}

/** Retrieve the status which indicates what the worker is currently doing. */
static inline enum _starpu_worker_status _starpu_worker_get_status(int workerid)
{
	return _starpu_config.workers[workerid].status;
}

/** Change the status of the worker which indicates what the worker is currently
 * doing (eg. executing a callback). */
static inline void _starpu_worker_set_status(int workerid, enum _starpu_worker_status status)
{
	_starpu_config.workers[workerid].status = status;
}

/** We keep an initial sched ctx which might be used in case no other ctx is available */
static inline struct _starpu_sched_ctx* _starpu_get_initial_sched_ctx(void)
{
	return &_starpu_config.sched_ctxs[STARPU_GLOBAL_SCHED_CTX];
}

int starpu_worker_get_nids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

/** returns workers not belonging to any context, be careful no mutex is used,
   the list might not be updated */
int starpu_worker_get_nids_ctx_free_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

static inline unsigned _starpu_worker_mutex_is_sched_mutex(int workerid, starpu_pthread_mutex_t *mutex)
{
	struct _starpu_worker *w = _starpu_get_worker_struct(workerid);
	return &w->sched_mutex == mutex;
}

static inline int _starpu_worker_get_nsched_ctxs(int workerid)
{
	return _starpu_config.workers[workerid].nsched_ctxs;
}

/** Get the total number of sched_ctxs created till now */
static inline unsigned _starpu_get_nsched_ctxs(void)
{
	/* topology.nsched_ctxs may be increased asynchronously in sched_ctx_create */
	STARPU_RMB();
	return _starpu_config.topology.nsched_ctxs;
}

/** Inlined version when building the core.  */
static inline int _starpu_worker_get_id(void)
{
	struct _starpu_worker * worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->workerid;
	}
	else
	{
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}
#define starpu_worker_get_id _starpu_worker_get_id

/** Similar behaviour to starpu_worker_get_id() but fails when called from outside a worker */
/** This returns an unsigned object on purpose, so that the caller is sure to get a positive value */
static inline unsigned __starpu_worker_get_id_check(const char *f, int l)
{
	(void) l;
	(void) f;
	int id = starpu_worker_get_id();
	STARPU_ASSERT_MSG(id>=0, "%s:%d Cannot be called from outside a worker\n", f, l);
	return id;
}
#define _starpu_worker_get_id_check(f,l) __starpu_worker_get_id_check(f,l)

enum starpu_node_kind _starpu_worker_get_node_kind(enum starpu_worker_archtype type);

void _starpu_worker_set_stream_ctx(unsigned workerid, struct _starpu_sched_ctx *sched_ctx);

struct _starpu_sched_ctx* _starpu_worker_get_ctx_stream(unsigned stream_workerid);

/** Send a request to the worker to block, before a parallel task is about to
 * begin.
 *
 * Must be called with worker's sched_mutex held.
 */
static inline void _starpu_worker_request_blocking_in_parallel(struct _starpu_worker * const worker)
{
	_starpu_worker_parallel_blocks = 1;
	/* flush pending requests to start on a fresh transaction epoch */
	while (worker->state_unblock_in_parallel_req)
		STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

	/* announce blocking intent */
	STARPU_ASSERT(worker->block_in_parallel_ref_count < UINT_MAX);
	worker->block_in_parallel_ref_count++;

	if (worker->block_in_parallel_ref_count == 1)
	{
		/* only the transition from 0 to 1 triggers the block_in_parallel_req */

		STARPU_ASSERT(!worker->state_blocked_in_parallel);
		STARPU_ASSERT(!worker->state_block_in_parallel_req);
		STARPU_ASSERT(!worker->state_block_in_parallel_ack);
		STARPU_ASSERT(!worker->state_unblock_in_parallel_req);
		STARPU_ASSERT(!worker->state_unblock_in_parallel_ack);

		/* trigger the block_in_parallel_req */
		worker->state_block_in_parallel_req = 1;
		STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
#ifdef STARPU_SIMGRID
		starpu_pthread_queue_broadcast(&_starpu_simgrid_task_queue[worker->workerid]);
#endif

		/* wait for block_in_parallel_req to be processed */
		while (!worker->state_block_in_parallel_ack)
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

		STARPU_ASSERT(worker->block_in_parallel_ref_count >= 1);
		STARPU_ASSERT(worker->state_block_in_parallel_req);
		STARPU_ASSERT(worker->state_blocked_in_parallel);

		/* reset block_in_parallel_req state flags */
		worker->state_block_in_parallel_req = 0;
		worker->state_block_in_parallel_ack = 0;

		/* broadcast block_in_parallel_req state flags reset */
		STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
	}
}

/** Send a request to the worker to unblock, after a parallel task is complete.
 *
 * Must be called with worker's sched_mutex held.
 */
static inline void _starpu_worker_request_unblocking_in_parallel(struct _starpu_worker * const worker)
{
	/* flush pending requests to start on a fresh transaction epoch */
	while (worker->state_block_in_parallel_req)
		STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

	/* unblocking may be requested unconditionnally
	 * thus, check is unblocking is really needed */
	if (worker->state_blocked_in_parallel)
	{
		if (worker->block_in_parallel_ref_count == 1)
		{
			/* only the transition from 1 to 0 triggers the unblock_in_parallel_req */

			STARPU_ASSERT(!worker->state_block_in_parallel_req);
			STARPU_ASSERT(!worker->state_block_in_parallel_ack);
			STARPU_ASSERT(!worker->state_unblock_in_parallel_req);
			STARPU_ASSERT(!worker->state_unblock_in_parallel_ack);

			/* trigger the unblock_in_parallel_req */
			worker->state_unblock_in_parallel_req = 1;
			STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);

			/* wait for the unblock_in_parallel_req to be processed */
			while (!worker->state_unblock_in_parallel_ack)
				STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

			STARPU_ASSERT(worker->state_unblock_in_parallel_req);
			STARPU_ASSERT(!worker->state_blocked_in_parallel);

			/* reset unblock_in_parallel_req state flags */
			worker->state_unblock_in_parallel_req = 0;
			worker->state_unblock_in_parallel_ack = 0;

			/* broadcast unblock_in_parallel_req state flags reset */
			STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
		}

		/* announce unblocking complete */
		STARPU_ASSERT(worker->block_in_parallel_ref_count > 0);
		worker->block_in_parallel_ref_count--;
	}
}

/** Called by the the worker to process incoming requests to block or unblock on
 * parallel task boundaries.
 *
 * Must be called with worker's sched_mutex held.
 */
static inline void _starpu_worker_process_block_in_parallel_requests(struct _starpu_worker * const worker)
{
	while (worker->state_block_in_parallel_req)
	{
		STARPU_ASSERT(!worker->state_blocked_in_parallel);
		STARPU_ASSERT(!worker->state_block_in_parallel_ack);
		STARPU_ASSERT(!worker->state_unblock_in_parallel_req);
		STARPU_ASSERT(!worker->state_unblock_in_parallel_ack);
		STARPU_ASSERT(worker->block_in_parallel_ref_count > 0);

		/* enter effective blocked state */
		worker->state_blocked_in_parallel = 1;

		/* notify block_in_parallel_req processing */
		worker->state_block_in_parallel_ack = 1;
		STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);

		/* block */
		while (!worker->state_unblock_in_parallel_req)
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

		STARPU_ASSERT(worker->state_blocked_in_parallel);
		STARPU_ASSERT(!worker->state_block_in_parallel_req);
		STARPU_ASSERT(!worker->state_block_in_parallel_ack);
		STARPU_ASSERT(!worker->state_unblock_in_parallel_ack);
		STARPU_ASSERT(worker->block_in_parallel_ref_count > 0);

		/* leave effective blocked state */
		worker->state_blocked_in_parallel = 0;

		/* notify unblock_in_parallel_req processing */
		worker->state_unblock_in_parallel_ack = 1;
		STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
	}
}

/** Mark the beginning of a scheduling operation by the worker. No worker
 * blocking operations on parallel tasks and no scheduling context change
 * operations must be performed on contexts containing the worker, on
 * contexts about to add the worker and on contexts about to remove the
 * worker, while the scheduling operation is in process. The sched mutex
 * of the worker may only be acquired permanently by another thread when
 * no scheduling operation is in process, or when a scheduling operation
 * is in process _and_ worker->state_relax_refcnt!=0. If a
 * scheduling operation is in process _and_
 * worker->state_relax_refcnt==0, a thread other than the worker
 * must wait on condition worker->sched_cond for
 * worker->state_relax_refcnt!=0 to become true, before acquiring
 * the worker sched mutex permanently.
 *
 * Must be called with worker's sched_mutex held.
 */
#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_enter_sched_op(struct _starpu_worker * const worker, const char*file, int line, const char* func)
#else
static inline void _starpu_worker_enter_sched_op(struct _starpu_worker * const worker)
#endif
{
	STARPU_ASSERT(!worker->state_sched_op_pending);
	if (!worker->state_blocked_in_parallel_observed)
	{
		/* process pending block requests before entering a sched_op region */
		_starpu_worker_process_block_in_parallel_requests(worker);
		while (worker->state_changing_ctx_notice)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

			/* new block requests may have been triggered during the wait,
			 * need to check again */
			_starpu_worker_process_block_in_parallel_requests(worker);
		}
	}
	else
	{
		/* if someone observed the worker state since the last call, postpone block request
		 * processing for one sched_op turn more, because the observer will not have seen
		 * new block requests between its observation and now.
		 *
		 * however, the worker still has to wait for context change operations to complete
		 * before entering sched_op again*/
		while (worker->state_changing_ctx_notice)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
		}
	}

	/* no block request and no ctx change ahead,
	 * enter sched_op */
	worker->state_sched_op_pending = 1;
	worker->state_blocked_in_parallel_observed = 0;
	worker->state_relax_refcnt = 0;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_on_file = file;
	worker->relax_on_line = line;
	worker->relax_on_func = func;
#endif
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_enter_sched_op(worker) __starpu_worker_enter_sched_op((worker), __FILE__, __LINE__, __starpu_func__)
#endif

/** Mark the end of a scheduling operation by the worker.
 *
 * Must be called with worker's sched_mutex held.
 */
void _starpu_worker_apply_deferred_ctx_changes(void);
#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_leave_sched_op(struct _starpu_worker * const worker, const char*file, int line, const char* func)
#else
static inline void _starpu_worker_leave_sched_op(struct _starpu_worker * const worker)
#endif
{
	STARPU_ASSERT(worker->state_sched_op_pending);
	worker->state_relax_refcnt = 1;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_off_file = file;
	worker->relax_off_line = line;
	worker->relax_off_func = func;
#endif
	worker->state_sched_op_pending = 0;
	STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
	_starpu_worker_apply_deferred_ctx_changes();
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_leave_sched_op(worker) __starpu_worker_leave_sched_op((worker), __FILE__, __LINE__, __starpu_func__)
#endif

static inline int _starpu_worker_sched_op_pending(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid == -1)
		return 0;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	return worker->state_sched_op_pending;
}

/** Must be called before altering a context related to the worker
 * whether about adding the worker to a context, removing it from a
 * context or modifying the set of workers of a context of which the
 * worker is a member, to mark the beginning of a context change
 * operation. The sched mutex of the worker must be held before calling
 * this function.
 *
 * Must be called with worker's sched_mutex held.
 */
static inline void _starpu_worker_enter_changing_ctx_op(struct _starpu_worker * const worker)
{
	STARPU_ASSERT(!starpu_pthread_equal(worker->thread_changing_ctx, starpu_pthread_self()));
	/* flush pending requests to start on a fresh transaction epoch */
	while (worker->state_changing_ctx_notice)
		STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);

	/* announce changing_ctx intent
	 *
	 * - an already started sched_op is allowed to complete
	 * - no new sched_op may be started
	 */
	worker->state_changing_ctx_notice = 1;

	worker->thread_changing_ctx = starpu_pthread_self();

	/* allow for an already started sched_op to complete */
	if (worker->state_sched_op_pending)
	{
		/* request sched_op to broadcast when way is cleared */
		worker->state_changing_ctx_waiting = 1;

		/* wait for sched_op completion */
		STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
#ifdef STARPU_SIMGRID
		starpu_pthread_queue_broadcast(&_starpu_simgrid_task_queue[worker->workerid]);
#endif
		do
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
		}
		while (worker->state_sched_op_pending);

		/* reset flag so other sched_ops wont have to broadcast state */
		worker->state_changing_ctx_waiting = 0;
	}
}

/** Mark the end of a context change operation.
 *
 * Must be called with worker's sched_mutex held.
 */
static inline void _starpu_worker_leave_changing_ctx_op(struct _starpu_worker * const worker)
{
	worker->thread_changing_ctx = (starpu_pthread_t)0;
	worker->state_changing_ctx_notice = 0;
	STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
}

/** Temporarily allow other worker to access current worker state, when still scheduling,
 * but the scheduling has not yet been made or is already done */
#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_relax_on(const char*file, int line, const char* func)
#else
static inline void _starpu_worker_relax_on(void)
#endif
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	if (worker == NULL)
		return;
	if (!worker->state_sched_op_pending)
		return;
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
#ifdef STARPU_SPINLOCK_CHECK
	STARPU_ASSERT_MSG(worker->state_relax_refcnt<UINT_MAX, "relax last turn on in %s (%s:%d)\n", worker->relax_on_func, worker->relax_on_file, worker->relax_on_line);
#else
	STARPU_ASSERT(worker->state_relax_refcnt<UINT_MAX);
#endif
	worker->state_relax_refcnt++;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_on_file = file;
	worker->relax_on_line = line;
	worker->relax_on_func = func;
#endif
	STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_relax_on() __starpu_worker_relax_on(__FILE__, __LINE__, __starpu_func__)
#endif
#define starpu_worker_relax_on _starpu_worker_relax_on

/** Same, but with current worker mutex already held */
#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_relax_on_locked(struct _starpu_worker *worker, const char*file, int line, const char* func)
#else
static inline void _starpu_worker_relax_on_locked(struct _starpu_worker *worker)
#endif
{
	if (!worker->state_sched_op_pending)
		return;
#ifdef STARPU_SPINLOCK_CHECK
	STARPU_ASSERT_MSG(worker->state_relax_refcnt<UINT_MAX, "relax last turn on in %s (%s:%d)\n", worker->relax_on_func, worker->relax_on_file, worker->relax_on_line);
#else
	STARPU_ASSERT(worker->state_relax_refcnt<UINT_MAX);
#endif
	worker->state_relax_refcnt++;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_on_file = file;
	worker->relax_on_line = line;
	worker->relax_on_func = func;
#endif
	STARPU_PTHREAD_COND_BROADCAST(&worker->sched_cond);
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_relax_on_locked(worker) __starpu_worker_relax_on_locked(worker,__FILE__, __LINE__, __starpu_func__)
#endif

#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_relax_off(const char*file, int line, const char* func)
#else
static inline void _starpu_worker_relax_off(void)
#endif
{
	int workerid = starpu_worker_get_id();
	if (workerid == -1)
		return;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	if (!worker->state_sched_op_pending)
		return;
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
#ifdef STARPU_SPINLOCK_CHECK
	STARPU_ASSERT_MSG(worker->state_relax_refcnt>0, "relax last turn off in %s (%s:%d)\n", worker->relax_on_func, worker->relax_on_file, worker->relax_on_line);
#else
	STARPU_ASSERT(worker->state_relax_refcnt>0);
#endif
	worker->state_relax_refcnt--;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_off_file = file;
	worker->relax_off_line = line;
	worker->relax_off_func = func;
#endif
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_relax_off() __starpu_worker_relax_off(__FILE__, __LINE__, __starpu_func__)
#endif
#define starpu_worker_relax_off _starpu_worker_relax_off

#ifdef STARPU_SPINLOCK_CHECK
static inline void __starpu_worker_relax_off_locked(const char*file, int line, const char* func)
#else
static inline void _starpu_worker_relax_off_locked(void)
#endif
{
	int workerid = starpu_worker_get_id();
	if (workerid == -1)
		return;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	if (!worker->state_sched_op_pending)
		return;
#ifdef STARPU_SPINLOCK_CHECK
	STARPU_ASSERT_MSG(worker->state_relax_refcnt>0, "relax last turn off in %s (%s:%d)\n", worker->relax_on_func, worker->relax_on_file, worker->relax_on_line);
#else
	STARPU_ASSERT(worker->state_relax_refcnt>0);
#endif
	worker->state_relax_refcnt--;
#ifdef STARPU_SPINLOCK_CHECK
	worker->relax_off_file = file;
	worker->relax_off_line = line;
	worker->relax_off_func = func;
#endif
}
#ifdef STARPU_SPINLOCK_CHECK
#define _starpu_worker_relax_off_locked() __starpu_worker_relax_off_locked(__FILE__, __LINE__, __starpu_func__)
#endif

static inline int _starpu_worker_get_relax_state(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid < 0)
		return 1;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	return worker->state_relax_refcnt != 0;
}
#define starpu_worker_get_relax_state _starpu_worker_get_relax_state

/** lock a worker for observing contents
 *
 * notes:
 * - if the observed worker is not in state_relax_refcnt, the function block until the state is reached */
static inline void _starpu_worker_lock(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	int cur_workerid = starpu_worker_get_id();
	if (workerid != cur_workerid)
	{
		starpu_worker_relax_on();

		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
		while (!worker->state_relax_refcnt)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
		}
	}
	else
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
	}
}

static inline int _starpu_worker_trylock(int workerid)
{
	struct _starpu_worker *cur_worker = _starpu_get_local_worker_key();
	int cur_workerid = cur_worker->workerid;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);

	/* Start with ourself */
	int ret = STARPU_PTHREAD_MUTEX_TRYLOCK_SCHED(&cur_worker->sched_mutex);
	if (ret)
		return ret;
	if (workerid == cur_workerid)
		/* We only needed to lock ourself */
		return 0;

	/* Now try to lock the other worker */
	ret = STARPU_PTHREAD_MUTEX_TRYLOCK_SCHED(&worker->sched_mutex);
	if (!ret)
	{
		/* Good, check that it is relaxed */
		ret = !worker->state_relax_refcnt;
		if (ret)
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	}
	if (!ret)
		_starpu_worker_relax_on_locked(cur_worker);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&cur_worker->sched_mutex);
	return ret;
}

static inline void _starpu_worker_unlock(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	int cur_workerid = starpu_worker_get_id();
	if (workerid != cur_workerid)
	{
		starpu_worker_relax_off();
	}
}

static inline void _starpu_worker_lock_self(void)
{
	int workerid = starpu_worker_get_id_check();
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
}

static inline void _starpu_worker_unlock_self(void)
{
	int workerid = starpu_worker_get_id_check();
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
}

static inline int _starpu_wake_worker_relax(int workerid)
{
	_starpu_worker_lock(workerid);
	int ret = starpu_wake_worker_locked(workerid);
	_starpu_worker_unlock(workerid);
	return ret;
}

int starpu_wake_worker_relax_light(int workerid);

/**
 * Allow a worker pulling a task it cannot execute to properly refuse it and
 *  send it back to the scheduler.
 */
void _starpu_worker_refuse_task(struct _starpu_worker *worker, struct starpu_task *task);

/* @}*/

#endif // __WORKERS_H__
