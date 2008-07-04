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
	struct cuda_module_s *module;
	struct cuda_function_s *function;

	/* grid and block shapes */
	unsigned gridx;
	unsigned gridy;
	unsigned blockx;
	unsigned blocky;

	unsigned shmemsize;
} cuda_codelet_t;

typedef struct cuda_worker_arg_t {
	int deviceid;
	int bindid;
	volatile int ready_flag;
	unsigned memory_node;
} cuda_worker_arg;

void init_cuda(void);
//int precondition_cuda(matrix *, matrix *, matrix *);
void *cuda_worker(void *);

#define OK              0
#define TRYAGAIN        1
#define FATAL           2

#endif //  __DRIVER_CUDA_H__
