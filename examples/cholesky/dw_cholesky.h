#ifndef __DW_CHOLESKY_H__
#define __DW_CHOLESKY_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif


#include <datawizard/datawizard.h>
#include <core/dependencies/tags.h>
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

#ifdef USE_CUDA

static float **ptrA;
static unsigned __dim;

sem_t sem_malloc;

static void malloc_pinned_codelet(data_interface_t *buffers __attribute__((unused)),
					void *addr  __attribute__((unused)))
{
	cuMemAllocHost((void **)ptrA, __dim*__dim*sizeof(float));
}

static void malloc_pinned_callback(void *arg  __attribute__((unused)))
{
	sem_post(&sem_malloc);
}

#endif

static inline void malloc_pinned(float **A, unsigned _dim)
{
#ifdef USE_CUDA
	codelet *cl = malloc(sizeof(codelet));
	cl->cl_arg = NULL;
	cl->cublas_func = malloc_pinned_codelet; 
	
	ptrA = A;
	__dim = _dim;

	job_t j = job_create();
	j->where = CUBLAS;
	j->cb = malloc_pinned_callback; 
	j->cl = cl;

	sem_init(&sem_malloc, 0, 0U);

	push_task(j);

	sem_wait(&sem_malloc);
	sem_destroy(&sem_malloc);
#else
	*A = malloc(_dim*_dim*sizeof(float));
#endif	
}

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
