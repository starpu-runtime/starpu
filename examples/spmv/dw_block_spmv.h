#ifndef __DW_BLOCK_SPMV_H__
#define __DW_BLOCK_SPMV_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <datawizard/datawizard.h>

void core_block_spmv(data_interface_t *descr, void *_args);

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void cublas_block_spmv(data_interface_t *descr, void *_args);
#endif // USE_CUBLAS

#endif // __DW_BLOCK_SPMV_H__
