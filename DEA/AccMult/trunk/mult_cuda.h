#ifndef __MULT_CUDA_H__
#define __MULT_CUDA_H__

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "util.h"
#include "jobs.h"
#include "parameters.h"

/* this is a randomly choosen value ... */
#ifndef MAXCUDADEVS
#define MAXCUDADEVS	4
#endif

#ifndef SHMEMSIZE
#define SHMEMSIZE	3160
#endif

typedef struct cuda_worker_arg_t {
	int deviceid;
	matrix *A;
	matrix *B;
	matrix *C;
} cuda_worker_arg;

void init_cuda(void);
void precondition_cuda(matrix *, matrix *, matrix *);
void *cuda_worker(void *);
void *cublas_worker(void *);

#endif //  __MULT_CUDA_H__
