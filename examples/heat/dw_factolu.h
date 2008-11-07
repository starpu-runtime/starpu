#ifndef __DW_FACTO_LU_H__
#define __DW_FACTO_LU_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif


#include <datawizard/datawizard.h>
#include <core/dependencies/tags.h>
#include "lu_kernels_model.h"

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

#if defined(USE_CUBLAS) || defined(USE_CUDA)

static float **ptrA;
static float **ptrB;
static unsigned __dim;

sem_t sem_malloc;

static void malloc_pinned_codelet(data_interface_t *buffers __attribute__((unused)),
					void *addr  __attribute__((unused)))
{
	cuMemAllocHost((void **)ptrA, __dim*__dim*sizeof(float));
	cuMemAllocHost((void **)ptrB, __dim*sizeof(float));
}

static void malloc_pinned_callback(void *arg  __attribute__((unused)))
{
	sem_post(&sem_malloc);
}

#endif

static inline void malloc_pinned(float **A, float **B, unsigned _dim)
{
#if defined(USE_CUBLAS) || defined(USE_CUDA)
	codelet *cl = malloc(sizeof(codelet));
	cl->cl_arg = NULL;
	cl->cublas_func = malloc_pinned_codelet; 
	
	ptrA = A;
	ptrB = B;
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
	*B = malloc(_dim*sizeof(float));
#endif	
}

#ifdef CHECK_RESULTS
static void __attribute__ ((unused)) compare_A_LU(float *A, float *LU,
				unsigned size, unsigned ld)
{
	unsigned i,j;
	float *L;
	float *U;

	L = malloc(size*size*sizeof(float));
	U = malloc(size*size*sizeof(float));

	memset(L, 0, size*size*sizeof(float));
	memset(U, 0, size*size*sizeof(float));

	/* only keep the lower part */
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[i+j*size] = LU[i+j*ld];
		}

		/* diag i = j */
		L[j+j*size] = LU[j+j*ld];
		U[j+j*size] = 1.0f;

		for (i = j+1; i < size; i++)
		{
			U[i+j*size] = LU[i+j*ld];
		}
	}

        /* now A_err = L, compute L*U */
	cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
			CblasUnit, size, size, 1.0f, U, size, L, size);

	float max_err = 0.0f;
	for (i = 0; i < size ; i++)
	{
		for (j = 0; j < size; j++) 
		{
			max_err = MAX(max_err, fabs(  L[i+j*size] - A[i+j*ld]  ));
		}
	}

	printf("max error between A and L*U = %f \n", max_err);
}
#endif // CHECK_RESULTS

void dw_core_codelet_update_u11(data_interface_t *, void *);
void dw_core_codelet_update_u12(data_interface_t *, void *);
void dw_core_codelet_update_u21(data_interface_t *, void *);
void dw_core_codelet_update_u22(data_interface_t *, void *);

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void dw_cublas_codelet_update_u11(data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u12(data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u21(data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u22(data_interface_t *descr, void *_args);
#endif

void dw_callback_codelet_update_u11(void *);
void dw_callback_codelet_update_u12_21(void *);
void dw_callback_codelet_update_u22(void *);

void dw_callback_v2_codelet_update_u11(void *);
void dw_callback_v2_codelet_update_u12(void *);
void dw_callback_v2_codelet_update_u21(void *);
void dw_callback_v2_codelet_update_u22(void *);

extern struct perfmodel_t model_11;
extern struct perfmodel_t model_12;
extern struct perfmodel_t model_21;
extern struct perfmodel_t model_22;

#endif // __DW_FACTO_LU_H__
