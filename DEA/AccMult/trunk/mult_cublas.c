#ifdef USE_CUBLAS

#include "mult_cublas.h"

static cublasStatus status;

extern int cublascounters[MAXCUBLASDEVS];

int ncublasgpus = 1;

void clean_cublas_problem(void *cbarg)
{
	/* simply free the device memory */
	job_descr *jd = (job_descr *)cbarg;

	cublasFree(jd->matA->cublas_data.dev_data);
	cublasFree(jd->matB->cublas_data.dev_data);
	cublasFree(jd->matC->cublas_data.dev_data);
}

void precondition_cublas(matrix *A, matrix *B, matrix *C)
{

	unsigned sizeA, sizeB, sizeC;

	sizeA = A->width*A->heigth;
	sizeB = B->width*B->heigth;
	sizeC = C->width*C->heigth;

	SAFE_CUBLAS_CALL(cublasAlloc(sizeA, sizeof(float), (void **)&A->cublas_data.dev_data));
	SAFE_CUBLAS_CALL(cublasAlloc(sizeB, sizeof(float), (void **)&B->cublas_data.dev_data));
	SAFE_CUBLAS_CALL(cublasAlloc(sizeC, sizeof(float), (void **)&C->cublas_data.dev_data));

	cublasSetMatrix(A->width,  A->heigth, sizeof(float), A->data, A->width, A->cublas_data.dev_data, A->width);
	cublasSetMatrix(B->width,  B->heigth, sizeof(float), B->data, B->width, B->cublas_data.dev_data, B->width);
	cublasSetMatrix(C->width,  C->heigth, sizeof(float), C->data, C->width, C->cublas_data.dev_data, C->width);
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
	cublasSgemm('n', 'n', nrowC, ncolC, ncolA, ALPHA, d_B, ldb, d_A, lda, BETA, d_C, ldc);
	status = cublasGetError();
	if (status) {
		printf("sgemm failed :");
		switch (status) {
			case CUBLAS_STATUS_NOT_INITIALIZED:
				printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
				break;
			case CUBLAS_STATUS_ALLOC_FAILED:
				printf("CUBLAS_STATUS_ALLOC_FAILED\n");
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR:
				printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
				break;
			case CUBLAS_STATUS_INVALID_VALUE:
				printf("CUBLAS_STATUS_INVALID_VALUE\n");
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED:
				printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
				break;
			case CUBLAS_STATUS_MAPPING_ERROR:
				printf("CUBLAS_STATUS_MAPPING_ERROR\n");
				break;
			default:
				printf("UNKNOWN REASON\n");
				break;
		}
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
		case PRECOND:
			printf("preconditionning ... \n");
			job_descr *jd = j->argcb;
			precondition_cublas(jd->matA, jd->matB, jd->matC);
			printf("preconditionned ok ... \n");
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

#ifndef DONTBIND
        /* fix the thread on the correct cpu */
        cpu_set_t aff_mask;
        CPU_ZERO(&aff_mask);
        CPU_SET(args->bindid, &aff_mask);
        sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	cublasInit();

	//precondition_cublas(args->A, args->B, args->C);

	printf("cublas thread is ready to run on CPU %d !\n", args->bindid);
	/* tell the main thread that this one is ready to work */
	args->ready_flag = 1;

	job_t j;
	do {
		j = pop_task();
		if (j == NULL) continue;

		/* can cublas do that task ? */
		if (!CUBLAS_MAY_PERFORM(j))
		{
			push_task(j);
			continue;
		}

		execute_job_on_cublas(j);

		if (j->cb)
			j->cb(j->argcb);
		
		cublascounters[devid]++;		

	} while(1);

	return NULL;
}


#endif // USE_CUBLAS
