#include "comp.h"
#include "mult.h"
#include "jobs.h"
#include <cblas.h>

extern void precondition_cuda(matrix *A, matrix *B, matrix *C);
extern int ncudagpus;
extern int ncublasgpus;

extern void copy_job_on_device(job_t j);

extern unsigned ncores;

void ref_mult(matrix *A, matrix *B, matrix *C)
{
	submatrix sA;
	submatrix sB;
	submatrix sC;
	
	sA.mat = A;
	sA.xa = 0;
	sA.ya = 0;
	sA.xb = A->width;
	sA.yb = A->heigth;

	sB.mat = B;
	sB.xa = 0;
	sB.ya = 0;
	sB.xb = B->width;
	sB.yb = B->heigth;

	sC.mat = C;
	sC.xa = 0;
	sC.ya = 0;
	sC.xb = C->width;
	sC.yb = C->heigth;

	cblas_mult(&sA, &sB, &sC);

}

void dummy_mult(submatrix *A, submatrix *B, submatrix *C)
{
	float sum;
	unsigned x,y, z;

	unsigned sizexa;
	unsigned sizexb;

	ASSERT(A->xb - A->xa == B->yb - B->ya);

	float *matA = A->mat->data;
	float *matB = B->mat->data;
	float *matC = C->mat->data;

	sizexa = A->mat->width;
	sizexb = B->mat->width;

	for (y = A->ya; y < A->yb ; y++)
	{
		for (x = B->xa; x < B->xb; x++)
		{
			sum = 0;

			for (z = A->xa; z < A->xb ; z++)
			{
				sum += matA[z+sizexa*y]*matB[x+sizexb*z];
			}

			matC[x+y*sizexb] = sum;
		}
	}
}

void cblas_mult(submatrix *A, submatrix *B, submatrix *C)
{
	/* 
	 * void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
	 */
	printf("SGEMM\n");

	int M = C->yb - C->ya;
	int N = C->xb - C->xa;
	int K = A->xb - A->xa;

	int lda = A->mat->width;
	int ldb = B->mat->width;
	int ldc = C->mat->width;

	float * dataA =  &A->mat->data[A->xa+A->ya*A->mat->width];
	float * dataB =  &B->mat->data[B->xa+B->ya*B->mat->width];
	float * dataC =  &C->mat->data[C->xa+C->ya*C->mat->width];

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
		 1.0, dataA, lda, dataB, ldb, 0.0, dataC, ldc);
}

void *dummy_mult_wrap(job_t j)
{
	return NULL;
}

/*
 *  * A x B = C
 *   */
void mult(matrix *A, matrix *B, matrix *C)
{
	unsigned nx,ny;
	unsigned x,y;

	nx = (B->width)/GRAIN;
	ny = (A->heigth)/GRAIN;

	for (x = 0; x < nx; x++)
	{
		for (y = 0; y < ny; y++)
		{
			job_t j = job_new();

			j->input.matA.mat = A;
			j->input.matA.xa = 0;
			j->input.matA.xb = A->width;
			j->input.matA.ya = y*GRAIN;
			j->input.matA.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->input.matB.mat = B;
			j->input.matB.xa = x*GRAIN;
			j->input.matB.xb = MIN( (x+1)*GRAIN, B->width);
			j->input.matB.ya = 0;
			j->input.matB.yb = B->heigth;

			j->output.matC_sub.mat  = C;
			j->output.matC_sub.xa = x*GRAIN;
			j->output.matC_sub.xb = MIN( (x+1)*GRAIN, B->width);
			j->output.matC_sub.ya = y*GRAIN;
			j->output.matC_sub.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->type = MUL;

			j->cb = NULL;

			push_task(j);
		}
	}

	/* terminate all threads */
	unsigned nworkers = 0;

#ifdef USE_CPUS
	nworkers += ncores;
#endif
#ifdef USE_CUDA
	nworkers += ncudagpus;
#endif
#ifdef USE_CUBLAS
	nworkers += ncublasgpus;
#endif

	int worker;
	for (worker = 0; worker < nworkers ; worker++) {
		job_t j = job_new();
		j->output.matC_existing = C;
		j->type = ABORT;
		push_task(j);
	}

	if (nworkers == 0) {
		fprintf(stderr, "Warning there is no worker ... \n");
	}

	return;
}
