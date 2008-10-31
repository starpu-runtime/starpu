#ifndef __DRIVER_CUBLAS_H__
#define __DRIVER_CUBLAS_H__

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <assert.h>
#include <stdio.h>
#include <math.h>

//#include <common/threads.h>
#include <common/util.h>
#include <core/jobs.h>
#include <common/parameters.h>
#include <cublas.h>

#include <common/fxt.h>

#include <datawizard/copy-driver.h>



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

typedef struct cublas_worker_arg_t {
	int deviceid;
	int bindid;
	volatile int ready_flag;
	unsigned memory_node;
	struct jobq_s *jobq;
} cublas_worker_arg;

void *cublas_worker(void *);

unsigned get_cublas_device_count(void);

#ifndef MAXCUBLASDEVS
#define MAXCUBLASDEVS	4
#endif


#endif //  __DRIVER_CUBLAS_H__
