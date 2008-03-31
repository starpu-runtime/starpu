#ifndef __MULT_CUDA_H__
#define __MULT_CUDA_H__

#define _GNU_SOURCE
#include <sched.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <common/util.h>
#include <common/parameters.h>
#include <core/jobs.h>

/* this is a randomly choosen value ... */
#ifndef MAXCUDADEVS
#define MAXCUDADEVS	4
#endif

#ifndef SHMEMSIZE
#define SHMEMSIZE	3160
#endif

typedef struct cuda_worker_arg_t {
	int deviceid;
	int bindid;
	volatile int ready_flag;
	unsigned memory_node;
#if 0
	matrix *A;
	matrix *B;
	matrix *C;
#endif
} cuda_worker_arg;

void init_cuda(void);
int precondition_cuda(matrix *, matrix *, matrix *);
void *cuda_worker(void *);

#define OK              0
#define TRYAGAIN        1
#define FATAL           2

#endif //  __MULT_CUDA_H__
