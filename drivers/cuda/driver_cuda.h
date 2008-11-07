#ifndef __DRIVER_CUDA_H__
#define __DRIVER_CUDA_H__

#define _GNU_SOURCE
#include <sched.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <common/util.h>
#include <common/parameters.h>
#include <core/jobs.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/perfmodel.h>

#include <common/fxt.h>

/* this is a randomly choosen value ... */
#ifndef MAXCUDADEVS
#define MAXCUDADEVS	4
#endif

#ifndef SHMEMSIZE
#define SHMEMSIZE	3160
#endif

typedef struct cuda_module_s {
	CUmodule module;
	char *module_path;
	unsigned is_loaded[MAXCUDADEVS];
} cuda_module_t;

typedef struct cuda_function_s {
	struct cuda_module_s *module;
	CUfunction function;
	char *symbol;
	unsigned is_loaded[MAXCUDADEVS];
} cuda_function_t;

typedef struct cuda_codelet_s {
	/* which function to execute on the card ? */
	struct cuda_function_s *func;

	/* grid and block shapes */
	unsigned gridx;
	unsigned gridy;
	unsigned blockx;
	unsigned blocky;

	unsigned shmemsize;

	void *stack; /* arguments */
	size_t stack_size;
} cuda_codelet_t;

typedef struct cuda_worker_arg_t {
	int deviceid;
	int bindid;
	volatile int ready_flag;
	unsigned memory_node;
	struct jobq_s *jobq;
} cuda_worker_arg;

void init_cuda(void);
//int precondition_cuda(matrix *, matrix *, matrix *);
void *cuda_worker(void *);

void init_cuda_module(struct cuda_module_s *module, char *path);
void load_cuda_module(int devid, struct cuda_module_s *module);
void init_cuda_function(struct cuda_function_s *func,
                        struct cuda_module_s *module,
                        char *symbol);
void load_cuda_function(int devid, struct cuda_function_s *function);

#define CUBLAS_REPORT_ERROR(status) 					\
	do {								\
		char *errormsg;						\
		switch (status) {					\
			case CUBLAS_STATUS_SUCCESS:			\
				errormsg = "success";			\
				break;					\
			case CUBLAS_STATUS_NOT_INITIALIZED:		\
				errormsg = "not initialized";		\
				break;					\
			case CUBLAS_STATUS_ALLOC_FAILED:		\
				errormsg = "alloc failed";		\
				break;					\
			case CUBLAS_STATUS_INVALID_VALUE:		\
				errormsg = "invalid value";		\
				break;					\
			case CUBLAS_STATUS_ARCH_MISMATCH:		\
				errormsg = "arch mismatch";		\
				break;					\
			case CUBLAS_STATUS_EXECUTION_FAILED:		\
				errormsg = "execution failed";		\
				break;					\
			case CUBLAS_STATUS_INTERNAL_ERROR:		\
				errormsg = "internal error";		\
				break;					\
			default:					\
				errormsg = "unknown error";		\
				break;					\
		}							\
		printf("oops  in %s ... %s \n", __func__, errormsg);	\
		assert(0);						\
	} while (0)  



#define CUDA_REPORT_ERROR(status) 					\
	do {								\
		char *errormsg;						\
		switch (status) {					\
			case CUDA_SUCCESS:				\
				errormsg = "success";			\
				break;					\
			case CUDA_ERROR_INVALID_VALUE:			\
				errormsg = "invalid value";		\
				break;					\
			case CUDA_ERROR_OUT_OF_MEMORY:			\
				errormsg = "out of memory";		\
				break;					\
			case CUDA_ERROR_NOT_INITIALIZED:		\
				errormsg = "not initialized";		\
				break;					\
			case CUDA_ERROR_DEINITIALIZED:			\
				errormsg = "deinitialized";		\
				break;					\
			case CUDA_ERROR_NO_DEVICE:			\
				errormsg = "no device";			\
				break;					\
			case CUDA_ERROR_INVALID_DEVICE:			\
				errormsg = "invalid device";		\
				break;					\
			case CUDA_ERROR_INVALID_IMAGE:			\
				errormsg = "invalid image";		\
				break;					\
			case CUDA_ERROR_INVALID_CONTEXT:		\
				errormsg = "invalid context";		\
				break;					\
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:	\
				errormsg = "context already current";	\
				break;					\
			case CUDA_ERROR_MAP_FAILED:			\
				errormsg = "map failed";		\
				break;					\
			case CUDA_ERROR_UNMAP_FAILED:			\
				errormsg = "unmap failed";		\
				break;					\
			case CUDA_ERROR_ARRAY_IS_MAPPED:		\
				errormsg = "array is mapped";		\
				break;					\
			case CUDA_ERROR_ALREADY_MAPPED:			\
				errormsg = "already mapped";		\
				break;					\
			case CUDA_ERROR_NO_BINARY_FOR_GPU:		\
				errormsg = "no binary for gpu";		\
				break;					\
			case CUDA_ERROR_ALREADY_ACQUIRED:		\
				errormsg = "already acquired";		\
				break;					\
			case CUDA_ERROR_NOT_MAPPED:			\
				errormsg = "not mapped";		\
				break;					\
			case CUDA_ERROR_INVALID_SOURCE:			\
				errormsg = "invalid source";		\
				break;					\
			case CUDA_ERROR_FILE_NOT_FOUND:			\
				errormsg = "file not found";		\
				break;					\
			case CUDA_ERROR_INVALID_HANDLE:			\
				errormsg = "invalid handle";		\
				break;					\
			case CUDA_ERROR_NOT_FOUND:			\
				errormsg = "not found";			\
				break;					\
			case CUDA_ERROR_NOT_READY:			\
				errormsg = "not ready";			\
				break;					\
			case CUDA_ERROR_LAUNCH_FAILED:			\
				errormsg = "launch failed";		\
				break;					\
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:	\
				errormsg = "launch out of resources";	\
				break;					\
			case CUDA_ERROR_LAUNCH_TIMEOUT:			\
				errormsg = "launch timeout";		\
				break;					\
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:	\
				errormsg = "launch incompatible texturing";\
				break;					\
			case CUDA_ERROR_UNKNOWN:			\
			default:					\
				errormsg = "unknown error";		\
				break;					\
		}							\
		printf("oops  in %s ... %s \n", __func__, errormsg);	\
		assert(0);						\
	} while (0)  

#endif //  __DRIVER_CUDA_H__

