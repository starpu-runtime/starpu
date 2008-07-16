#ifndef __WORKERS_H__
#define __WORKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <common/parameters.h>
#include <common/threads.h>
#include <common/util.h>
#include <common/timing.h>
#include <common/fxt.h>
#include "jobs.h"

#ifdef USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_CUBLAS
#include <drivers/cublas/driver_cublas.h>
#endif

#ifdef USE_SPU
#include <drivers/spu/ppu/driver_spu.h>
#endif

#ifdef USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

#include <drivers/core/driver_core.h>

#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>

struct machine_config_s {
	#ifdef USE_CPUS
	unsigned ncores;
	thread_t corethreads[NMAXCORES];
	core_worker_arg coreargs[NMAXCORES];
	#endif
	
	#ifdef USE_CUDA
	thread_t cudathreads[MAXCUDADEVS];
	cuda_worker_arg cudaargs[MAXCUDADEVS];
	extern int ncudagpus;
	#endif
	
	#ifdef USE_CUBLAS
	thread_t cublasthreads[MAXCUBLASDEVS];
	cublas_worker_arg cublasargs[MAXCUBLASDEVS];
	unsigned ncublasgpus;
	#endif
	
	#ifdef USE_SPU
	thread_t sputhreads[MAXSPUS];
	unsigned nspus;
	spu_worker_arg spuargs[MAXSPUS];
	#endif
	
	#ifdef USE_GORDON
	thread_t gordonthread;
	/* only the threads managed by gordon */
	unsigned ngordonspus;
	gordon_worker_arg gordonargs;
	#endif
};

void init_machine(void);
void terminate_workers(struct machine_config_s *config);
void kill_all_workers(struct machine_config_s *config);
void display_general_stats(void);

#endif // __WORKERS_H__
