#ifndef __DW_FACTO_LU_H__
#define __DW_FACTO_LU_H__

#include <semaphore.h>

#include <cblas.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/filters.h>

typedef struct {
	data_state *dataA;
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned nblocks;
	unsigned *remaining;
	sem_t *sem;
} cl_args;


void dw_callback_codelet_update_u11(void *);
void dw_callback_codelet_update_u12_21(void *);
void dw_callback_codelet_update_u22(void *);

void dw_core_codelet_update_u11(void *);
void dw_core_codelet_update_u12(void *);
void dw_core_codelet_update_u21(void *);
void dw_core_codelet_update_u22(void *);

#ifdef CHECK_RESULTS
static void __attribute__ ((unused)) compare_A_LU(float *A, float *LU,
				unsigned size)
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
			L[i+j*size] = LU[i+j*size];
		}

		/* diag i = j */
		L[j+j*size] = LU[j+j*size];
		U[j+j*size] = 1.0f;

		for (i = j+1; i < size; i++)
		{
			U[i+j*size] = LU[i+j*size];
		}
	}

        /* now A_err = L, compute L*U */
	cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
			CblasUnit, size, size, 1.0f, U, size, L, size);

	float max_err = 0.0f;
	for (i = 0; i < size*size ; i++)
	{
		max_err = MAX(max_err, fabs(  L[i] - A[i]  ));
	}

	printf("max error between A and L*U = %f \n", max_err);
}
#endif // CHECK_RESULTS

#endif // __DW_FACTO_LU_H__
