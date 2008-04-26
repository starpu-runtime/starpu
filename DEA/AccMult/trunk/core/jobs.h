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

#ifdef USE_CUDA
#include <cuda.h>
#endif

typedef enum {GPU, CUDA, CUBLAS, SPU, CORE, GORDON, ANY} cap;

typedef enum {ADD, SUB, MUL, PART, PRECOND, CLEAN, ABORT, SGEMM, SAXPY, SGEMV, STRSM, STRSV, SGER, SSYR, SCOPY, CODELET} jobtype;

typedef void (*callback)(void *);
/* codelet function */
typedef void (*cl_func)(void *);

#define CORE_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == CORE )
#define CUDA_MAY_PERFORM(j)     ( (j)->where == ANY || (j)->where == GPU || (j)->where == CUDA )
#define CUBLAS_MAY_PERFORM(j)     ( (j)->where == ANY || (j)->where == GPU || (j)->where == CUBLAS )
#define SPU_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == SPU )
#define GORDON_MAY_PERFORM(j)	( (j)->where == ANY || (j)->where == GORDON )

#ifdef USE_CUDA
typedef struct cuda_matrix_t {
	CUdeviceptr matdata;
	CUdeviceptr matheader;
} cuda_matrix;
#endif

#ifdef USE_CUBLAS
typedef struct cublas_matrix_t {
	float *dev_data;
} cublas_matrix;
#endif

typedef struct matrix_t {
	float *data;
	unsigned width;
	unsigned heigth;
	/* XXX put a flag to tell which copy are available */
#ifdef USE_CUDA
	cuda_matrix cuda_data;
#endif
#ifdef USE_CUBLAS
	cublas_matrix cublas_data;
#endif
} matrix;

typedef struct submatrix_t {
	matrix *mat;
	unsigned xa;
	unsigned xb;
	unsigned ya;
	unsigned yb;
} submatrix;

/*
 * A codelet describes the various function 
 * that may be called from a worker ... XXX
 */
typedef struct codelet_t {
	cl_func cuda_func;
	cl_func cublas_func;
	cl_func core_func;
	cl_func spu_func;
	cl_func gordon_func;
	void *cl_arg;
} codelet;


LIST_TYPE(job,
//	/* don't move that structure ! (cf opaque pointers ..) */
//	struct {
//		submatrix matA; /* inputs */
//		submatrix matB;
//	} input;
//	union {
//		matrix matC;    /* output */
//		submatrix matC_sub;
//		matrix *matC_existing; /* when we just need a reference .. */
//	} output;
//	struct {
//		matrix *mat1;
//		matrix *mat2;
//		matrix *mat3;
//	} args;
	jobtype type;	/* what kind of job ? */
	cap where;	/* where can it be performed ? */
	callback cb;	/* do "cb(argcb)" when finished */
	codelet *cl;
	void *argcb;
	int counter;	/* when this reaches 0 the callback can be executed */
#ifdef USE_CUDA
	CUdeviceptr device_job ;
	CUdeviceptr toto;
#endif
);

typedef struct job_descr_t {
//	matrix *matA;
//	matrix *matB;
//	matrix *matC;
//	matrix *matD;
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

#endif // __JOBS_H__
