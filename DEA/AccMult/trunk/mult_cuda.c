#ifdef USE_CUDA

#include <assert.h>
#include <math.h>

//#ifdef USE_CUBLAS
#include <cublas.h>
//#endif

#include "jobs.h"
#include "mult_cuda.h"

//#define DEBUG

/* the number of CUDA devices */
int ncudagpus;
/* the respective properties of all devices */
static struct cudaDeviceProp cudadevprops[MAXCUDADEVS];

static CUdevice cuDevice;
static CUcontext cuContext[MAXCUDADEVS];
static CUmodule cuModule;
static char* module_path = "./comp_cuda.cubin";

CUresult status;
cublasStatus cb_status;

static CUfunction dummyMatrixMul = NULL;

extern int cudacounters[MAXCUDADEVS];

extern char *execpath;

void init_context(int devid)
{
	status = cuCtxCreate( &cuContext[devid], 0, 0);
	if ( CUDA_SUCCESS != status )
		goto error;

	status = cuCtxAttach(&cuContext[devid], 0);
	if ( CUDA_SUCCESS != status )
		goto error;

	status = cuModuleLoad(&cuModule, module_path);
	if ( CUDA_SUCCESS != status )
		goto error;
	
	status = cuModuleGetFunction( &dummyMatrixMul, cuModule, "cuda_mult" );
	if ( CUDA_SUCCESS != status )
		goto error;

	/* launch the kernel */
	status = cuFuncSetBlockShape( dummyMatrixMul, BLOCKDIMX, BLOCKDIMY, 1);
	if ( CUDA_SUCCESS != status )
		goto error;
	
 	status = cuFuncSetSharedSize(dummyMatrixMul, SHMEMSIZE);
	if ( CUDA_SUCCESS != status )
		goto error;


	return;
error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}

void init_cuda(void)
{
	CUresult status;

	status = cuInit(0);
	if ( CUDA_SUCCESS != status )
		goto error;

	cudaGetDeviceCount(&ncudagpus);
	assert(ncudagpus <= MAXCUDADEVS);

	int dev;
	for (dev = 0; dev < ncudagpus; dev++)
	{
		cudaGetDeviceProperties(&cudadevprops[dev], dev);
	}

	return;
error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}


/*
 * The flags determines whether the data are input/output or both
 *  so that we can avoid memory copies if possible
 *
 *	INOUT = IN | OUT
 */
#define OUTBUFF 0x2
#define	INBUFF	0x1

void copy_matrix_on_device(matrix *M, unsigned devid, unsigned flags)
{

	/* first allocate the corresponding memory on device */
	unsigned datasize = M->width*M->heigth*sizeof(float);
	status = cuMemAlloc(&M->cuda_data.matdata, datasize);
	if (status != CUDA_SUCCESS) 
		goto error;


	/* copy data in the case of input */
	if (flags & INBUFF) {
		status = cuMemcpyHtoD(M->cuda_data.matdata, M->data, datasize);
		if (status != CUDA_SUCCESS) 
			goto error;
	}
	return;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}

void precondition_cuda(matrix *A, matrix *B, matrix *C)
{
	/* a copy of the various matrices is created on the device */
	copy_matrix_on_device(A, 0, INBUFF);
	copy_matrix_on_device(B, 0, INBUFF);
	copy_matrix_on_device(C, 0, OUTBUFF);

}

void precondition_cublas(matrix *A, matrix *B, matrix *C)
{

	unsigned sizeA, sizeB, sizeC;

	sizeA = A->width*A->heigth;
	sizeB = B->width*B->heigth;
	sizeC = C->width*C->heigth;

	cublasAlloc(sizeA, sizeof(float), (void **)&A->cuda_data.dev_data);
	cublasAlloc(sizeB, sizeof(float), (void **)&B->cuda_data.dev_data);
	cublasAlloc(sizeC, sizeof(float), (void **)&C->cuda_data.dev_data);

	cublasSetMatrix(A->width,  A->heigth, sizeof(float), A->data, A->width, A->cuda_data.dev_data, A->width);
	cublasSetMatrix(B->width,  B->heigth, sizeof(float), B->data, B->width, B->cuda_data.dev_data, B->width);
}



void copy_matrix(matrix *C)
{
	unsigned datasize = C->width*C->heigth*sizeof(float);

	status = cuMemcpyDtoH((void *)C->data, C->cuda_data.matdata, datasize);
	if (status != CUDA_SUCCESS) 
		goto error;
	
	return;
error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}

/*
 * This is NOT the efficient way of course
 */
void copy_submatrix(submatrix *C)
{
        /* of course this is stupid but ... simple */
        /* copy each line one by one :) */
        int line;

        CUdeviceptr matrixstart;
        CUdeviceptr linestart;

        matrixstart = C->mat->cuda_data.matdata;

        unsigned int linesize = (C->xb - C->xa)*sizeof(float);

        for (line = C->ya; line < C->yb; line++)
        {
                linestart = matrixstart +
                        (C->xa + C->mat->width*line)*sizeof(float);
                status = cuMemcpyDtoH(&C->mat->data[C->xa + C->mat->width*line], linestart, linesize);
                if (status != CUDA_SUCCESS)
                        goto error;
        }

        return;

error:
        printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
        assert(0);
        pthread_exit(NULL);

}

void set_args(job_t j)
{
	unsigned int offset = 0;

	submatrix *matA = &j->input.matA;
	submatrix *matB = &j->input.matB;
	submatrix *matC = &j->output.matC_sub;

	/* datamatA */
	status = cuParamSetv(dummyMatrixMul,offset,&matA->mat->cuda_data.matdata,sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(CUdeviceptr);

	status = cuParamSetv( dummyMatrixMul, offset, &matA->mat->width, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matA->xa, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	/* datamatB */
	status = cuParamSetv( dummyMatrixMul, offset, &matB->mat->cuda_data.matdata, sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(CUdeviceptr);

	status = cuParamSetv( dummyMatrixMul, offset, &matB->mat->width, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matB->ya, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matB->yb, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);


	/* datamatC */
	status = cuParamSetv( dummyMatrixMul, offset, &matC->mat->cuda_data.matdata, sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(CUdeviceptr);

	status = cuParamSetv( dummyMatrixMul, offset, &matC->mat->width, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matC->xa, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matC->xb, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matC->ya, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

	status = cuParamSetv( dummyMatrixMul, offset, &matC->yb, sizeof(unsigned));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(unsigned);

#ifdef DEBUG
	status = cuParamSetv( dummyMatrixMul, offset, &j->toto, sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
		goto error;
	offset += sizeof(CUdeviceptr);
#endif


	status = cuParamSetSize(dummyMatrixMul, offset);
	if ( CUDA_SUCCESS != status )
		goto error;


	return;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
}

void cuda_mult(job_t j)
{
#ifdef DEBUG
	status = cuMemAlloc(&j->toto, sizeof(int));
	if (status != CUDA_SUCCESS) 
		goto error;
#endif

	set_args(j);

	status = cuLaunchGrid( dummyMatrixMul, GRIDDIMX, GRIDDIMY); 
	if ( CUDA_SUCCESS != status )
		goto error;

	/* wait for its termination */
	status = cuCtxSynchronize();
	if ( CUDA_SUCCESS != status )
		goto error;


	/* copy data back into memory */
// XXX optimize !
#ifdef USE_CPUS
	copy_submatrix(&j->output.matC_sub);
#endif
	
	return;
error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}

void remove_job_from_device(job_t j)
{
#ifdef DEBUG
	int toto2 = -1; 

	status = cuMemcpyDtoH(&toto2, j->toto, sizeof(int));
	if (status != CUDA_SUCCESS) {
		printf("could not cuMemcpyDtoH !\n");
		goto error;
	}
	printf("AFTER toto2 = %p\n", toto2);
#endif // DEBUG

	//status = cuMemcpyDtoH(&j2, j->device_job, sizeof(int));
	//if (status != CUDA_SUCCESS) {45.45//	printf("could not cuMemcpyDtoH !\n");
	//	goto error;
	//}

//	printf("AFTER toto2 = %d\n", toto2);

//	status = cuMemFree(j->toto);
//	if (status != CUDA_SUCCESS) {
//		printf("could not cuMemcpyHtoD !\n");
//		goto error;
//	}

//	status = cuMemFree(j->device_job);
//	if (status != CUDA_SUCCESS) {
//		printf("could not cuMemcpyHtoD !\n");
//		goto error;
//	}
	
	return;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}

void execute_job_on_cuda(job_t j)
{
	switch (j->type) {
		case MUL:
			cuda_mult(j);
			remove_job_from_device(j);
			break;
		case ABORT:
			printf("CUDA abort\n");
			pthread_exit(NULL);
			break;
		default:
			break;
	}
}

void *cuda_worker(void *arg)
{
	struct cuda_worker_arg_t* args = (struct cuda_worker_arg_t*)arg;

	int devid = args->deviceid;

	init_context(devid);

	precondition_cuda(args->A, args->B, args->C);

	job_t j;
	
	do {
		j = pop_task();
		if (j == NULL) continue;
		execute_job_on_cuda(j);

		if (j->cb)
			j->cb(j->argcb);
		
		cudacounters[devid]++;		

	} while(1);

	return NULL;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);

}

#define START_POS(_mat)		\
		((_mat)->xa + (_mat)->ya*(_mat)->mat->width)

#define DEV_DATA(_mat)	((_mat)->mat->cuda_data.dev_data)

void cublas_mult(job_t j)
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
	float *h_A = &((matA->mat->data)[START_POS(matA)]);
	float *h_B = &((matB->mat->data)[START_POS(matB)]);

	int nrowC = matC->yb - matC->ya;
	int ncolC = matC->xb - matC->xa;
	int ncolA = matA->xb - matA->xa;

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

void execute_job_on_cublas(job_t j)
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
	struct cuda_worker_arg_t* args = (struct cuda_worker_arg_t*)arg;

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
		
		cudacounters[devid]++;		

	} while(1);

	return NULL;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	assert(0);
}


#endif // USE_CUDA
