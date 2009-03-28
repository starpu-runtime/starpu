#ifndef __JOBS_H__
#define __JOBS_H__

#include <starpu.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <common/config.h>
#include <common/timing.h>
#include <common/list.h>
#include <common/fxt.h>

#include <core/dependencies/tags.h>

#include <datawizard/datawizard.h>

#include <core/perfmodel/perfmodel.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

/* codelet function */
typedef void (*cl_func)(data_interface_t *, void *);
typedef void (*callback)(void *);

#define CORE_MAY_PERFORM(j)	((j)->task->cl->where & CORE)
#define CUDA_MAY_PERFORM(j)     ((j)->task->cl->where & CUDA)
#define CUBLAS_MAY_PERFORM(j)   ((j)->task->cl->where & CUBLAS)
#define SPU_MAY_PERFORM(j)	((j)->task->cl->where & SPU)
#define GORDON_MAY_PERFORM(j)	((j)->task->cl->where & GORDON)

/* a job is the internal representation of a task */
LIST_TYPE(job,
	struct starpu_task *task;

	sem_t sync_sem;

	struct tag_s *tag;

	double predicted;
	double penality;

	unsigned footprint_is_computed;
	uint32_t footprint;

	unsigned terminated;
);

//#warning this must not be exported anymore ... 
//job_t job_create(struct starpu_task *task);
void handle_job_termination(job_t j);
size_t job_get_data_size(job_t j);

//int submit_job(job_t j);
//int submit_prio_job(job_t j);
//int submit_job_sync(job_t j);

#endif // __JOBS_H__
