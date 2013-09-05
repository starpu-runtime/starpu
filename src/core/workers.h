/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  INRIA
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

#include <starpu.h>
#include <common/config.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <core/jobs.h>
#include <core/perfmodel/perfmodel.h>
#include <core/sched_policy.h>
#include <core/topology.h>
#include <core/errorcheck.h>
#include <core/sched_ctx.h>
#include <core/sched_ctx_list.h>
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#include <drivers/cuda/driver_cuda.h>
#include <drivers/opencl/driver_opencl.h>

#ifdef STARPU_USE_MIC
#include <drivers/mic/driver_mic_source.h>
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_SCC
#include <drivers/scc/driver_scc_source.h>
#endif


#include <drivers/cpu/driver_cpu.h>

#include <datawizard/datawizard.h>

#include <starpu_parameters.h>

struct _starpu_worker
{
	struct _starpu_machine_config *config;
        starpu_pthread_mutex_t mutex;
	enum starpu_worker_archtype arch; /* what is the type of worker ? */
	uint32_t worker_mask; /* what is the type of worker ? */
	enum starpu_perfmodel_archtype perf_arch; /* in case there are different models of the same arch */
	starpu_pthread_t worker_thread; /* the thread which runs the worker */
	int mp_nodeid; /* which mp node hold the cpu/gpu/etc (-1 for this
			* node) */
	unsigned devid; /* which cpu/gpu/etc is controlled by the worker ? */
	int bindid; /* which cpu is the driver bound to ? (logical index) */
	int workerid; /* uniquely identify the worker among all processing units types */
	int combined_workerid; /* combined worker currently using this worker */
	int current_rank; /* current rank in case the worker is used in a parallel fashion */
	int worker_size; /* size of the worker in case we use a combined worker */
	starpu_pthread_cond_t started_cond; /* indicate when the worker is ready */
	starpu_pthread_cond_t ready_cond; /* indicate when the worker is ready */
	unsigned memory_node; /* which memory node is the worker associated with ? */
	starpu_pthread_cond_t sched_cond; /* condition variable used when the worker waits for tasks. */
        starpu_pthread_mutex_t sched_mutex; /* mutex protecting sched_cond */
	struct starpu_task_list local_tasks; /* this queue contains tasks that have been explicitely submitted to that queue */
	struct starpu_task *current_task; /* task currently executed by this worker */
	struct _starpu_worker_set *set; /* in case this worker belongs to a set */
	struct _starpu_job_list *terminated_jobs; /* list of pending jobs which were executed */
	unsigned worker_is_running;
	unsigned worker_is_initialized;
	enum _starpu_worker_status status; /* what is the worker doing now ? (eg. CALLBACK) */
	char name[64];
	char short_name[10];
	unsigned run_by_starpu; /* Is this run by StarPU or directly by the application ? */

	struct _starpu_sched_ctx_list *sched_ctx_list;
	unsigned nsched_ctxs; /* the no of contexts a worker belongs to*/
	struct _starpu_barrier_counter tasks_barrier; /* wait for the tasks submitted */

	unsigned has_prev_init; /* had already been inited in another ctx */

	unsigned removed_from_ctx[STARPU_NMAX_SCHED_CTXS];

	unsigned spinning_backoff ; /* number of cycles to pause when spinning  */

	/* conditions variables used when parallel sections are executed in contexts */
	starpu_pthread_cond_t parallel_sect_cond;
	starpu_pthread_mutex_t parallel_sect_mutex;

	/* boolean indicating that workers should block in order to allow
	   parallel sections to be executed on their allocated resources */
	unsigned parallel_sect;

	/* indicate whether the workers shares tasks lists with other workers*/
	/* in this case when removing him from a context it disapears instantly */
	unsigned shares_tasks_lists[STARPU_NMAX_SCHED_CTXS];

#ifdef __GLIBC__
	cpu_set_t cpu_set;
#endif /* __GLIBC__ */
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t hwloc_cpu_set;
#endif
};

struct _starpu_combined_worker
{
	enum starpu_perfmodel_archtype perf_arch; /* in case there are different models of the same arch */
	uint32_t worker_mask; /* what is the type of workers ? */
	int worker_size;
	unsigned memory_node; /* which memory node is associated that worker to ? */
	int combined_workerid[STARPU_NMAXWORKERS];
#ifdef STARPU_USE_MP
	int count;
	pthread_mutex_t count_mutex;
#endif

#ifdef __GLIBC__
	cpu_set_t cpu_set;
#endif /* __GLIBC__ */
#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t hwloc_cpu_set;
#endif
};

/* in case a single CPU worker may control multiple
 * accelerators (eg. Gordon for n SPUs) */
struct _starpu_worker_set
{
        starpu_pthread_mutex_t mutex;
	starpu_pthread_t worker_thread; /* the thread which runs the worker */
	unsigned nworkers;
	unsigned started; /* Only one thread for the whole set */
	void *retval;
	struct _starpu_worker *workers;
        starpu_pthread_cond_t ready_cond; /* indicate when the set is ready */
	unsigned set_is_initialized;
};

struct _starpu_machine_topology
{
	/* Total number of workers. */
	unsigned nworkers;

	/* Total number of combined workers. */
	unsigned ncombinedworkers;

	unsigned nsched_ctxs;
#ifdef STARPU_HAVE_HWLOC
	/* Topology as detected by hwloc. */
	hwloc_topology_t hwtopology;
#else
	/* We maintain ABI compatibility with and without hwloc */
	void *dummy;
#endif

	/* Total number of CPUs, as detected by the topology code. May
	 * be different from the actual number of CPU workers.
	 */
	unsigned nhwcpus;

	/* Total number of CUDA devices, as detected. May be different
	 * from the actual number of CUDA workers.
	 */
	unsigned nhwcudagpus;

	/* Total number of OpenCL devices, as detected. May be
	 * different from the actual number of OpenCL workers.
	 */
	unsigned nhwopenclgpus;

	/* Total number of SCC cores, as detected. May be different
	 * from the actual number of core workers.
	 */
	unsigned nhwscc;

	/* Actual number of CPU workers used by StarPU. */
	unsigned ncpus;

	/* Actual number of CUDA workers used by StarPU. */
	unsigned ncudagpus;

	/* Actual number of OpenCL workers used by StarPU. */
	unsigned nopenclgpus;

	/* Actual number of SCC workers used by StarPU. */
	unsigned nsccdevices;

	/* Topology of MP nodes (mainly MIC and SCC) as well as necessary
	 * objects to communicate with them. */
	unsigned nhwmicdevices;
	unsigned nmicdevices;

	unsigned nhwmiccores[STARPU_MAXMICDEVS]; // Each MIC node has its set of cores.
	unsigned nmiccores[STARPU_MAXMICDEVS];

	/* Indicates the successive cpu identifier that should be used
	 * to bind the workers. It is either filled according to the
	 * user's explicit parameters (from starpu_conf) or according
	 * to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over
	 * the cpus.
	 */
	unsigned workers_bindid[STARPU_NMAXWORKERS];

	/* Indicates the successive CUDA identifier that should be
	 * used by the CUDA driver.  It is either filled according to
	 * the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CUDAID env. variable.
	 * Otherwise, they are taken in ID order.
	 */
	unsigned workers_cuda_gpuid[STARPU_NMAXWORKERS];

	/* Indicates the successive OpenCL identifier that should be
	 * used by the OpenCL driver.  It is either filled according
	 * to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_OPENCLID env. variable.
	 * Otherwise, they are taken in ID order.
	 */
	unsigned workers_opencl_gpuid[STARPU_NMAXWORKERS];

	/** Indicates the successive MIC devices that should be used
	 * by the MIC driver.  It is either filled according to the
	 * user's explicit parameters (from starpu_conf) or according
	 * to the STARPU_WORKERS_MICID env. variable. Otherwise, they
	 * are taken in ID order. */
	/* unsigned workers_mic_deviceid[STARPU_NMAXWORKERS]; */

	/* Which SCC(s) do we use ? */
	/* Indicates the successive SCC devices that should be used by
	 * the SCC driver.  It is either filled according to the
	 * user's explicit parameters (from starpu_conf) or according
	 * to the STARPU_WORKERS_SCCID env. variable. Otherwise, they
	 * are taken in ID order.
	 */
	unsigned workers_scc_deviceid[STARPU_NMAXWORKERS];
};

struct _starpu_machine_config
{
	struct _starpu_machine_topology topology;

#ifdef STARPU_HAVE_HWLOC
	int cpu_depth;
#endif

	/* Where to bind workers ? */
	int current_bindid;

	/* Which GPU(s) do we use for CUDA ? */
	int current_cuda_gpuid;

	/* Which GPU(s) do we use for OpenCL ? */
	int current_opencl_gpuid;

	/* Which MIC do we use? */
	int current_mic_deviceid;

	/* Which SCC do we use? */
	int current_scc_deviceid;

	/* Basic workers : each of this worker is running its own driver and
	 * can be combined with other basic workers. */
	struct _starpu_worker workers[STARPU_NMAXWORKERS];

	/* Combined workers: these worker are a combination of basic workers
	 * that can run parallel tasks together. */
	struct _starpu_combined_worker combined_workers[STARPU_NMAX_COMBINEDWORKERS];

	/* This bitmask indicates which kinds of worker are available. For
	 * instance it is possible to test if there is a CUDA worker with
	 * the result of (worker_mask & STARPU_CUDA). */
	uint32_t worker_mask;

        /* either the user given configuration passed to starpu_init or a default configuration */
	struct starpu_conf *conf;
	/* set to 1 if no conf has been given by the user, it
	 * indicates the memory allocated for the default
	 * configuration should be freed on shutdown */
	int default_conf;

	/* this flag is set until the runtime is stopped */
	unsigned running;

	/* all the sched ctx of the current instance of starpu */
	struct _starpu_sched_ctx sched_ctxs[STARPU_NMAX_SCHED_CTXS];

	/* this flag is set until the application is finished submitting tasks */
	unsigned submitting;
};

/* Three functions to manage argv, argc */
void _starpu_set_argc_argv(int *argc, char ***argv);
int *_starpu_get_argc();
char ***_starpu_get_argv();

/* Fill conf with environment variables */
void _starpu_conf_check_environment(struct starpu_conf *conf);

/* Has starpu_shutdown already been called ? */
unsigned _starpu_machine_is_running(void);

/* Check if there is a worker that may execute the task. */
uint32_t _starpu_worker_exists(struct starpu_task *);

/* Is there a worker that can execute CUDA code ? */
uint32_t _starpu_can_submit_cuda_task(void);

/* Is there a worker that can execute CPU code ? */
uint32_t _starpu_can_submit_cpu_task(void);

/* Is there a worker that can execute OpenCL code ? */
uint32_t _starpu_can_submit_opencl_task(void);

/* Is there a worker that can execute OpenCL code ? */
uint32_t _starpu_can_submit_scc_task(void);

/* Check whether there is anything that the worker should do instead of
 * sleeping (waiting on something to happen). */
unsigned _starpu_worker_can_block(unsigned memnode);

/* This function must be called to block a worker. It puts the worker in a
 * sleeping state until there is some event that forces the worker to wake up.
 * */
void _starpu_block_worker(int workerid, starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex);

/* The _starpu_worker structure describes all the state of a StarPU worker.
 * This function sets the pthread key which stores a pointer to this structure.
 * */
void _starpu_set_local_worker_key(struct _starpu_worker *worker);

/* This function initializes the current thread for the given worker */
void _starpu_worker_init(struct _starpu_worker *worker, unsigned fut_key);

/* Returns the _starpu_worker structure that describes the state of the
 * current worker. */
struct _starpu_worker *_starpu_get_local_worker_key(void);

/* Returns the _starpu_worker structure that describes the state of the
 * specified worker. */
struct _starpu_worker *_starpu_get_worker_struct(unsigned id);

/* Returns the starpu_sched_ctx structure that descriebes the state of the 
 * specified ctx */
struct _starpu_sched_ctx *_starpu_get_sched_ctx_struct(unsigned id);

struct _starpu_combined_worker *_starpu_get_combined_worker_struct(unsigned id);

int _starpu_is_initialized(void);

/* Returns the structure that describes the overall machine configuration (eg.
 * all workers and topology). */
struct _starpu_machine_config *_starpu_get_machine_config(void);

/* Retrieve the status which indicates what the worker is currently doing. */
enum _starpu_worker_status _starpu_worker_get_status(int workerid);

/* Change the status of the worker which indicates what the worker is currently
 * doing (eg. executing a callback). */
void _starpu_worker_set_status(int workerid, enum _starpu_worker_status status);

/* We keep an initial sched ctx which might be used in case no other ctx is available */
struct _starpu_sched_ctx* _starpu_get_initial_sched_ctx(void);

int starpu_worker_get_nids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

/* returns workers not belonging to any context, be careful no mutex is used, 
   the list might not be updated */
int starpu_worker_get_nids_ctx_free_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);

#endif // __WORKERS_H__
