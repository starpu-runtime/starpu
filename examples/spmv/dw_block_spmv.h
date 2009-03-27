#ifndef __DW_BLOCK_SPMV_H__
#define __DW_BLOCK_SPMV_H__

#include <semaphore.h>
#include <common/timing.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cublas.h>
#endif

void core_block_spmv(data_interface_t *descr, void *_args);

#ifdef USE_CUDA
void cublas_block_spmv(data_interface_t *descr, void *_args);
#endif // USE_CUDA

#endif // __DW_BLOCK_SPMV_H__
