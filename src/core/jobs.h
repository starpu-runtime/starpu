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
#include <common/timing.h>
#include <common/list.h>
#include <common/fxt.h>

#include <core/dependencies/tags.h>

#include <datawizard/datawizard.h>

#include <core/perfmodel/perfmodel.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#define NMAXBUFS	8

#define MIN_PRIO        (-4)
#define MAX_PRIO        5
#define DEFAULT_PRIO	0

#define ANY	(~0)
#define CORE	((1ULL)<<1)
#define CUBLAS	((1ULL)<<2)
#define CUDA	((1ULL)<<3)
#define SPU	((1ULL)<<4)
#define GORDON	((1ULL)<<5)

/* codelet function */
typedef void (*cl_func)(data_interface_t *, void *);
typedef void (*callback)(void *);

#define CORE_MAY_PERFORM(j)	((j)->cl->where & CORE)
#define CUDA_MAY_PERFORM(j)     ((j)->cl->where & CUDA)
#define CUBLAS_MAY_PERFORM(j)   ((j)->cl->where & CUBLAS)
#define SPU_MAY_PERFORM(j)	((j)->cl->where & SPU)
#define GORDON_MAY_PERFORM(j)	((j)->cl->where & GORDON)

LIST_TYPE(job,
	codelet *cl;

	callback cb;	/* do "cb(argcb)" when finished */
	void *argcb;

	unsigned synchronous; /* if set, a call to push is blocking */
	sem_t sync_sem;

	struct tag_s *tag;
	unsigned use_tag;

	int priority; /* MAX_PRIO = most important 
			: MIN_PRIO = least important */

	unsigned nbuffers;
	buffer_descr buffers[NMAXBUFS];
	data_interface_t interface[NMAXBUFS];

	double predicted;
	double penality;

	unsigned footprint_is_computed;
	uint32_t footprint;

	unsigned terminated;
);

job_t job_create(void);
void handle_job_termination(job_t j);
size_t job_get_data_size(job_t j);

int submit_job(job_t j);
int submit_prio_job(job_t j);
int submit_job_sync(job_t j);

#endif // __JOBS_H__
