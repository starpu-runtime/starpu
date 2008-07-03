#ifndef __JOBS_H__
#define __JOBS_H__

#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <semaphore.h>
#include <common/timing.h>
#include <common/list.h>
#include <common/threads.h>
#include <common/fxt.h>
#include <core/tags.h>
#include <stdarg.h>

#include <datawizard/coherency.h>

#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif

typedef enum {GPU, CUDA, CUBLAS, SPU, CORE, GORDON, ANY} cap;
typedef enum {ADD, SUB, MUL, PART, PRECOND, CLEAN, ABORT, SGEMM, SAXPY, SGEMV, STRSM, STRSV, SGER, SSYR, SCOPY, CODELET} jobtype;

typedef void (*callback)(void *);
/* codelet function */
typedef void (*cl_func)(buffer_descr *, void *);

#define CORE_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == CORE )
#define CUDA_MAY_PERFORM(j)     ( (j)->where == ANY || (j)->where == GPU || (j)->where == CUDA )
#define CUBLAS_MAY_PERFORM(j)     ( (j)->where == ANY || (j)->where == GPU || (j)->where == CUBLAS )
#define SPU_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == SPU )
#define GORDON_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == GORDON )

/*
 * A codelet describes the various function 
 * that may be called from a worker ... XXX
 */
typedef struct codelet_t {
	void *cuda_func;
	void *cublas_func;
	void *core_func;
	void *spu_func;
	void *gordon_func;
	void *cl_arg;
} codelet;

#define NMAXBUFS	8


LIST_TYPE(job,
	jobtype type;	/* what kind of job ? */
	cap where;	/* where can it be performed ? */
	callback cb;	/* do "cb(argcb)" when finished */
	codelet *cl;
	void *argcb;
	int counter;	/* when this reaches 0 the callback can be executed */
	struct _tag_s *tag;
	unsigned use_tag;
	unsigned nbuffers;
	buffer_descr buffers[NMAXBUFS];
);

typedef struct job_descr_t {
	int debug;
	int counter;
	callback f;
	void *argf;
	tick_t job_submission;
	tick_t job_preconditionned;
	tick_t job_computed;
	tick_t job_finished;
	tick_t job_refstart;
	tick_t job_refstop;
} job_descr;

void init_work_queue(void);
void push_task(job_t task);
void push_prio_task(job_t task);
job_t pop_task(void);
void push_task(job_t task);

#endif // __JOBS_H__
