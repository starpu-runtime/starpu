#ifdef USE_CUBLAS

#include <assert.h>
#include <math.h>

#include <cublas.h>
#include "jobs.h"
#include "mult_cublas.h"

#define START_POS(_mat)		\
		((_mat)->xa + (_mat)->ya*(_mat)->mat->width)

#define DEV_DATA(_mat)	((_mat)->mat->cublas_data.dev_data)

extern int cublascounters[MAXCUBLASDEVS];

int ncublasgpus = 1;

static void precondition_cublas(matrix *A, matrix *B, matrix *C)
{

	unsigned sizeA, sizeB, sizeC;

	sizeA = A->width*A->heigth;
	sizeB = B->width*B->heigth;
	sizeC = C->width*C->heigth;

	cublasAlloc(sizeA, sizeof(float), (void **)&A->cublas_data.dev_data);
	cublasAlloc(sizeB, sizeof(float), (void **)&B->cublas_data.dev_data);
	cublasAlloc(sizeC, sizeof(float), (void **)&C->cublas_data.dev_data);

	cublasSetMatrix(A->width,  A->heigth, sizeof(float), A->data, A->width, A->cublas_data.dev_data, A->width);
	cublasSetMatrix(B->width,  B->heigth, sizeof(float), B->data, B->width, B->cublas_data.dev_data, B->width);
}



static void cublas_mult(job_t j)
{
	/* Since we have a row major CUBLAS implementation,
	 * we take the transposed submatrices ... lda will also be the height,
	 * instead of width
	 * From CUBLAS point of view, matX stores Xt ...
	 *  Ct = Bt At (C = AB)
	 */

	submatrix *matA = &j->input.matA;
	submatrix *matB = &j->input.matB;
	submatrix *matC = &j->output.matC_sub;

	float *d_A = &(DEV_DATA(matA))[START_POS(matA)];
	int lda = matA->mat->width;

	float *d_B = &(DEV_DATA(matB))[START_POS(matB)];
	int ldb = matB->mat->width;

	float *d_C = &(DEV_DATA(matC))[START_POS(matC)];
	int ldc = matC->mat->width;

	float *h_C = &((matC->mat->data)[START_POS(matC)]);

	int nrowC = matC->yb - matC->ya;
	int ncolC = matC->xb - matC->xa;
	int ncolA = matA->xb - matA->xa;

	//printf("cubla_mult \n");
	cublasSgemm('n', 'n', nrowC, ncolC, ncolA, 1.0, d_B, ldb, d_A, lda, 0.0, d_C, ldc);
	if (cublasGetError()) {
		printf("sgemm failed \n");
	}
	
	/* XXX fetch data from the device */	
	cublasGetMatrix(nrowC, ncolC, sizeof(float), d_C, ldc, h_C, ldc);
	if (cublasGetError()) {
		printf("getmatrix failed \n");
	}
}

static void execute_job_on_cublas(job_t j)
{
	switch (j->type) {
		case MUL:
			//printf("cublas mult\n");
			cublas_mult(j);
			break;
		case ABORT:
			printf("CUBLAS abort\n");
			cublasShutdown();
			pthread_exit(NULL);
			break;
		default:
			break;
	}
}

void *cublas_worker(void *arg)
{
	struct cublas_worker_arg_t* args = (struct cublas_worker_arg_t*)arg;

	int devid = args->deviceid;

	cublasInit();

	precondition_cublas(args->A, args->B, args->C);

	job_t j;
	do {
		j = pop_task();
		if (j == NULL) continue;
		execute_job_on_cublas(j);

		if (j->cb)
			j->cb(j->argcb);
		
		cublascounters[devid]++;		

	} while(1);

	return NULL;
}


#endif // USE_CUBLAS
