#ifndef __JOBS_H__
#define __JOBS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include "list.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda.h>
#endif

/* XXX find a better name ;) */
typedef enum {GPU, CELL, CORE, ANY} cap;

typedef enum {ADD, SUB, MUL, PART, ABORT} jobtype;

typedef void (*callback)(void *);

#ifdef USE_CUDA
typedef struct cuda_matrix_t {
	CUdeviceptr matdata;
	CUdeviceptr matheader;
	float *dev_data;
} cuda_matrix;
#endif

typedef struct matrix_t {
	float *data;
	unsigned width;
	unsigned heigth;
	/* XXX put a flag to tell which copy are available */
#ifdef USE_CUDA
	cuda_matrix cuda_data;
#endif
} matrix;

typedef struct submatrix_t {
	matrix *mat;
	unsigned xa;
	unsigned xb;
	unsigned ya;
	unsigned yb;
} submatrix;

LIST_TYPE(job,
	/* don't move that structure ! (cf opaque pointers ..) */
	struct {
		submatrix matA; /* inputs */
		submatrix matB;
	} input;
	union {
		matrix matC;    /* output */
		submatrix matC_sub;
		matrix *matC_existing; /* when we just need a reference .. */
	} output;
	jobtype type;	/* what kind of job ? */
	cap where;	/* where can it be performed ? */
	callback cb;	/* do "cb(argcb)" when finished */
	void *argcb;
	int counter;	/* when this reaches 0 the callback can be executed */
#ifdef USE_CUDA
	CUdeviceptr device_job ;
	CUdeviceptr toto;
#endif
);

void init_work_queue(void);
void push_task(job_t task);
job_t pop_task(void);

#endif // __JOBS_H__
