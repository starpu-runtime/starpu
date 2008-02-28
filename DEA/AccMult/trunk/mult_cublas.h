#ifndef __MULT_CUBLAS_H__
#define __MULT_CUBLAS_H__

#include <assert.h>
#include <stdio.h>


#include "threads.h"
#include "util.h"
#include "jobs.h"
#include "parameters.h"
// don't forget that one or you will regret it !
#include <cublas.h>

#define MAXCUBLASDEVS	4

typedef struct cublas_worker_arg_t {
	int deviceid;
	matrix *A;
	matrix *B;
	matrix *C;
} cublas_worker_arg;

void *cublas_worker(void *);

#endif //  __MULT_CUBLAS_H__
