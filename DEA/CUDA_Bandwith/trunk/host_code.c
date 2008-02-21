#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "param.h"

static CUdevice cuDevice;
static CUcontext cuContext;
static CUmodule cuModule;
CUresult status;
extern char *execpath;

static CUfunction benchKernel = NULL;

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

	status = cuModuleGetFunction( &benchKernel, cuModule, "bandwith_test");
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
	status = cuFuncSetBlockShape( benchKernel, BLOCKDIMX, BLOCKDIMY, 1);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

 	status = cuFuncSetSharedSize(benchKernel, SHMEMSIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	status = cuParamSetv(benchKernel,offset,&srcdevptr,sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);

	status = cuParamSetv(benchKernel,offset,&srcdevptr,sizeof(CUdeviceptr));
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);

	status = cuParamSeti(benchKernel,offset,DATASIZE);
	if ( CUDA_SUCCESS != status )
	{
		errline = __LINE__;
		goto error;
	}

	offset += sizeof(CUdeviceptr);



	status = cuLaunchGrid( benchKernel, GRIDDIMX, GRIDDIMY); 
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
