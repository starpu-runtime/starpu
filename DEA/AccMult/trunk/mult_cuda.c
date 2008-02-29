#ifdef USE_CUDA

#include "mult_cuda.h"

/* the number of CUDA devices */
int ncudagpus;
/* the respective properties of all devices */
static struct cudaDeviceProp cudadevprops[MAXCUDADEVS];

static CUdevice cuDevice;
static CUcontext cuContext[MAXCUDADEVS];
static CUmodule cuModule;
static char* module_path = "./comp_cuda.cubin";

CUresult status;

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
	thread_exit(NULL);
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
	thread_exit(NULL);
}


/*
 * The flags determines whether the data are input/output or both
 *  so that we can avoid memory copies if possible
 *
 *	INOUT = IN | OUT
 */
#define OUTBUFF 0x2
#define	INBUFF	0x1

int copy_matrix_on_device(matrix *M, unsigned devid, unsigned flags)
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

	return OK;

error:
	printf("oops  in %s ... %s \n", __func__, cudaGetErrorString(status));
	return TRYAGAIN;
}

int precondition_cuda(matrix *A, matrix *B, matrix *C)
{
	int res;

	/* a copy of the various matrices is created on the device */
	res = copy_matrix_on_device(A, 0, INBUFF);
		if (res) goto errorA;
	res = copy_matrix_on_device(B, 0, INBUFF);
		if (res) goto errorB;
	res = copy_matrix_on_device(C, 0, OUTBUFF);
		if (res) goto errorC;
	/* TODO check output */

	return OK;

errorC:
	cuMemFree(B->cuda_data.matdata);
errorB:
	cuMemFree(C->cuda_data.matdata);
errorA:
	return TRYAGAIN;
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
	thread_exit(NULL);
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
        thread_exit(NULL);

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
	thread_exit(NULL);
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
	thread_exit(NULL);
}

int execute_job_on_cuda(job_t j)
{
	int res;
	job_descr * jd;

	switch (j->type) {
		case MUL:
			cuda_mult(j);
			remove_job_from_device(j);
			break;
		case PRECOND:
			jd = j->argcb;
			res = precondition_cuda(jd->matA, jd->matB, jd->matC);
			if (res != OK) {
				printf("precondition failed ... trying later\n");
				return TRYAGAIN;
			}
			printf("preconditionned ok ... \n");
			break;
		case ABORT:
			printf("CUDA abort\n");
			thread_exit(NULL);
			break;
		default:
			break;
	}

	return OK;
}

void *cuda_worker(void *arg)
{
	struct cuda_worker_arg_t* args = (struct cuda_worker_arg_t*)arg;

	int devid = args->deviceid;

#ifndef DONTBIND
        /* fix the thread on the correct cpu */
        cpu_set_t aff_mask;
        CPU_ZERO(&aff_mask);
        CPU_SET(args->bindid, &aff_mask);
        sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	init_context(devid);

	//precondition_cuda(args->A, args->B, args->C);

	/* tell the main thread that this one is ready */
	args->ready_flag = 1;

	job_t j;
	int res;
	
	do {
		j = pop_task();
		if (j == NULL) continue;

		/* can CUDA do that task ? */
		if (!CUDA_MAY_PERFORM(j))
		{
			push_task(j);
			continue;
		}

		res = execute_job_on_cuda(j);
		if (res != OK) {
			switch (res) {
				case OK:
				case FATAL:
					assert(0);
				case TRYAGAIN:
					push_task(j);
					continue;
				default:
					assert(0);
			}
		}

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
