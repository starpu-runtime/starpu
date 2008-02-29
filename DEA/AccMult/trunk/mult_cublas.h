#ifndef __MULT_CUBLAS_H__
#define __MULT_CUBLAS_H__

#define _GNU_SOURCE
#include <sched.h>

#include <assert.h>
#include <stdio.h>
#include <math.h>

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

#define SAFE_CUBLAS_CALL(ops)		 				\
	do {								\
	cublasStatus _status;						\
	_status = (ops);								\
	if (_status) {							\
	switch (_status) {						\
		case CUBLAS_STATUS_NOT_INITIALIZED:			\
			printf("CUBLAS_STATUS_NOT_INITIALIZED\n");	\
			break;						\
                case CUBLAS_STATUS_ALLOC_FAILED:			\
                            printf("CUBLAS_STATUS_ALLOC_FAILED\n");	\
                            break;					\
                case CUBLAS_STATUS_INTERNAL_ERROR:			\
                        printf("CUBLAS_STATUS_INTERNAL_ERROR\n");	\
                        break;						\
                case CUBLAS_STATUS_INVALID_VALUE:			\
                        printf("CUBLAS_STATUS_INVALID_VALUE\n");	\
                        break;						\
                case CUBLAS_STATUS_EXECUTION_FAILED:			\
                        printf("CUBLAS_STATUS_EXECUTION_FAILED\n");	\
                        break;						\
                case CUBLAS_STATUS_MAPPING_ERROR:			\
                        printf("CUBLAS_STATUS_MAPPING_ERROR\n");	\
                        break;						\
                default:						\
                        printf("UNKNOWN REASON\n");			\
                        break;						\
	}								\
	}								\
	} while (0);

#define OK              0
#define TRYAGAIN        1
#define FATAL           2

typedef struct cublas_worker_arg_t {
	int deviceid;
	int bindid;
	volatile int ready_flag;
	matrix *A;
	matrix *B;
	matrix *C;
} cublas_worker_arg;

void *cublas_worker(void *);

#endif //  __MULT_CUBLAS_H__
