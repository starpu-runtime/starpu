#include "mult_core.h"

extern int corecounters[NMAXCORES];
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

	/* we use the CBLAS not to die before 
	 * the end of speedup measurements ! */
	cblas_mult(&sA, &sB, &sC);
}

static void __attribute__ ((unused)) dummy_mult(submatrix *A, submatrix *B, submatrix *C)
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
		 ALPHA, dataA, lda, dataB, ldb, BETA, dataC, ldc);
}

void execute_job_on_core(job_t j)
{
        switch (j->type) {
		case SGEMM:
		case SAXPY:
		case SGEMV:
		case STRSM:
		case STRSV:
		case SGER:
		case SSYR:
		case SCOPY:
			/* TODO */
			assert(0);
			break;
                case MUL:
			printf("core mult task %d\n", ((job_descr *)j->argcb)->debug);
#ifdef USE_CPU_BLAS
                        cblas_mult(&j->input.matA, &j->input.matB, &j->output.matC_sub);
#else
                        dummy_mult(&j->input.matA, &j->input.matB, &j->output.matC_sub);
#endif
                        break;
                case ABORT:
                        printf("core abort\n");
                        thread_exit(NULL);
                        break;
                default:
			printf("don't know what to do with that task on a core ! ... \n");
                        break;
        }
}

void *core_worker(void *arg)
{
        int core = ((core_worker_arg *)arg)->coreid;

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(((core_worker_arg *)arg)->bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

        printf("core worker %d is ready on logical core %d\n", core, ((core_worker_arg *)arg)->bindid);

        /* tell the main thread that we are ready */
        ((core_worker_arg *)arg)->ready_flag = 1;

        job_t j;

        do {
                j = pop_task();
                if (j == NULL) continue;

		/* can a core perform that task ? */
		if (!CORE_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			push_task(j);
			continue;
		}

                execute_job_on_core(j);

                if (j->cb)
                        j->cb(j->argcb);

                corecounters[core]++;
        } while(1);

        return NULL;
}
