#ifndef __DW_CHOLESKY_H__
#define __DW_CHOLESKY_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <common/blas.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif


#include <datawizard/datawizard.h>
#include <core/dependencies/tags.h>
#include <common/malloc.h>
//#include "lu_kernels_model.h"

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

typedef struct {
	data_state *dataA;
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned nblocks;
	unsigned *remaining;
	sem_t *sem;
} cl_args;

void chol_core_codelet_update_u11(data_interface_t *, void *);
void chol_core_codelet_update_u21(data_interface_t *, void *);
void chol_core_codelet_update_u22(data_interface_t *, void *);

#ifdef USE_CUDA
void chol_cublas_codelet_update_u11(data_interface_t *descr, void *_args);
void chol_cublas_codelet_update_u21(data_interface_t *descr, void *_args);
void chol_cublas_codelet_update_u22(data_interface_t *descr, void *_args);
#endif

void initialize_system(float **A, unsigned dim, unsigned pinned);
void dw_cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks);

extern struct perfmodel_t chol_model_11;
extern struct perfmodel_t chol_model_21;
extern struct perfmodel_t chol_model_22;

#endif // __DW_CHOLESKY_H__
