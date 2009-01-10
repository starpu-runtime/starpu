#ifndef __JOBS_H__
#define __JOBS_H__

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

typedef enum {CODELET} jobtype;

/* codelet function */
typedef void (*cl_func)(data_interface_t *, void *);
typedef void (*callback)(void *);

#define CORE_MAY_PERFORM(j)	( (j)->where & CORE	)
#define CUDA_MAY_PERFORM(j)     ( (j)->where & CUDA	)
#define CUBLAS_MAY_PERFORM(j)   ( (j)->where & CUBLAS	)
#define SPU_MAY_PERFORM(j)	( (j)->where & SPU	)
#define GORDON_MAY_PERFORM(j)	( (j)->where & GORDON	)

/*
 * A codelet describes the various function 
 * that may be called from a worker
 */
typedef struct codelet_t {
	/* the different implementations of the codelet */
	void *cuda_func;
	void *cublas_func;
	void *core_func;
	void *spu_func;
	uint8_t gordon_func;

	/* the arguments are given as a buffer */
	void *cl_arg;
	/* in case the argument buffer has to be uploaded explicitely */
	size_t cl_arg_size;
} codelet;

LIST_TYPE(job,
	jobtype type;	/* what kind of job ? */
	uint32_t where;	/* where can it be performed ? */

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

	struct perfmodel_t *model;
	double predicted;

	unsigned footprint_is_computed;
	uint32_t footprint;
);

job_t job_create(void);
void handle_job_termination(job_t j);
size_t job_get_data_size(job_t j);

#endif // __JOBS_H__
