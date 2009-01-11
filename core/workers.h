#ifndef __WORKERS_H__
#define __WORKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <common/parameters.h>
#include <pthread.h>
#include <common/util.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <core/jobs.h>
#include <core/perfmodel/perfmodel.h>
#include <core/policies/sched_policy.h>

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

#include <drivers/core/driver_core.h>

#include <datawizard/datawizard.h>

#define CORE_ALPHA	1.0f
#define CUDA_ALPHA	13.33f
#define GORDON_ALPHA	6.0f /* XXX this is a random value ... */

#define NMAXWORKERS	16

enum archtype {
	CORE_WORKER,
	CUDA_WORKER,
	GORDON_WORKER
};

struct worker_s {
	enum archtype arch; /* what is the type of worker ? */
	enum perf_archtype perf_arch; /* in case there are different models of the same arch */
	pthread_t worker_thread; /* the thread which runs the worker */
	int id; /* which core/gpu/etc is controlled by the workker ? */
        sem_t ready_sem; /* indicate when the worker is ready */
	int bindid; /* which core is the driver bound to ? */
	unsigned memory_node; /* which memory node is associated that worker to ? */
	struct jobq_s *jobq; /* in which queue will that worker get/put tasks ? */
	struct worker_set_s *set; /* in case this worker belongs to a set */
	struct job_list_s *terminated_jobs; /* list of pending jobs which were executed */
	unsigned worker_is_running;
};

/* in case a single CPU worker may control multiple 
 * accelerators (eg. Gordon for n SPUs) */
struct worker_set_s {
	pthread_t worker_thread; /* the thread which runs the worker */
	unsigned nworkers;
	unsigned joined; /* only one thread may call pthread_join*/
	void *retval;
	struct worker_s *workers;
        sem_t ready_sem; /* indicate when the worker is ready */
};

struct machine_config_s {
	unsigned nworkers;

	struct worker_s workers[NMAXWORKERS];

	/* this flag is set until the runtime is stopped */
	unsigned running;
};

void init_machine(void);
void terminate_workers(struct machine_config_s *config);
void kill_all_workers(struct machine_config_s *config);
void display_general_stats(void);
void terminate_machine(void);

unsigned machine_is_running(void);

inline uint32_t worker_exists(uint32_t task_mask);
inline uint32_t may_submit_cuda_task(void);
inline uint32_t may_submit_core_task(void);


#endif // __WORKERS_H__
