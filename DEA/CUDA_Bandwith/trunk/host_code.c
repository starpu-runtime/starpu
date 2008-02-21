#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

static CUdevice cuDevice;
static CUcontext cuContext;
static CUmodule cuModule;
CUresult status;
extern char *execpath;

#define BLOCKDIMX	4
#define BLOCKDIMY	4
#define GRIDDIMX	16
#define GRIDDIMY	16

#define SHMEMSIZE	1024

#define DATASIZE	2048

static CUfunction dummyMatrixMul = NULL;

unsigned offset = 0;
CUdeviceptr srcdevptr; 
CUdeviceptr dstdevptr; 

unsigned errline = -1;

int main(int argc, char **argv)
{

	status = cuInit(0);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuCtxCreate( &cuContext, 0, 0);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuCtxAttach(&cuContext, 0);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuModuleLoad(&cuModule, "./kernel_code.cubin");
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuModuleGetFunction( &dummyMatrixMul, cuModule, "bandwith_test");
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	/* allocate buffers on the device (slow memory) */
	status = cuMemAlloc(&srcdevptr, DATASIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuMemAlloc(&dstdevptr, DATASIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	/* launch the kernel */
	status = cuFuncSetBlockShape( dummyMatrixMul, BLOCKDIMX, BLOCKDIMY, 1);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

 	status = cuFuncSetSharedSize(dummyMatrixMul, SHMEMSIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuParamSetv(dummyMatrixMul,offset,&srcdevptr,sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);

	status = cuParamSetv(dummyMatrixMul,offset,&srcdevptr,sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);

	status = cuParamSeti(dummyMatrixMul,offset,DATASIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);



	status = cuLaunchGrid( dummyMatrixMul, GRIDDIMX, GRIDDIMY); 
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	/* wait for its termination */
	status = cuCtxSynchronize();
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	return 0;
error:
	printf("oops  in %s line %d... %s \n", __func__, errline,cudaGetErrorString(status));
	assert(0);
	pthread_exit(NULL);
}
