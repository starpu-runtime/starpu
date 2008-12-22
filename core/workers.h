#ifndef __WORKERS_H__
#define __WORKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <common/parameters.h>
#include <common/threads.h>
#include <common/util.h>
#include <common/timing.h>
#include <common/fxt.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_SPU
#include <drivers/spu/ppu/driver_spu.h>
#endif

#ifdef USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

#include <drivers/core/driver_core.h>

#include <datawizard/datawizard.h>

#define CORE_ALPHA	1.0f
#define CUDA_ALPHA	13.33f

#define NMAXWORKERS	16

enum archtype {
	CORE_WORKER,
	CUDA_WORKER
};

struct worker_s {
	enum archtype arch; /* what is the type of worker ? */
	thread_t worker_thread; /* the thread which runs the worker */
	int id; /* which core/gpu/etc is controlled by the workker ? */
        sem_t ready_sem; /* indicate when the worker is ready */
	int bindid; /* which core is the driver bound to ? */
	unsigned memory_node; /* which memory node is associated that worker to ? */
	struct jobq_s *jobq; /* in which queue will that worker get/put tasks ? */
};

struct machine_config_s {
	unsigned nworkers;

	struct worker_s workers[NMAXWORKERS];

	/* this flag is set until the runtime is stopped */
	volatile unsigned running;
};

void init_machine(void);
void terminate_workers(struct machine_config_s *config);
void kill_all_workers(struct machine_config_s *config);
void display_general_stats(void);
void terminate_machine(void);

unsigned machine_is_running(void);

#endif // __WORKERS_H__
