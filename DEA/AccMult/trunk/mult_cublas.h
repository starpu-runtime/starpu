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

#define START_POS(_mat)         \
		((_mat)->xa + (_mat)->ya*(_mat)->mat->width)

#define DEV_DATA(_mat)  ((_mat)->mat->cublas_data.dev_data)


typedef struct cublas_worker_arg_t {
	int deviceid;
	volatile int ready_flag;
	matrix *A;
	matrix *B;
	matrix *C;
} cublas_worker_arg;

void *cublas_worker(void *);

#endif //  __MULT_CUBLAS_H__
