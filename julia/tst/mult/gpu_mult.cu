#include <starpu.h>
#include <stdint.h>
#include <stdio.h>




__global__ void gpuMultKernel
(
		uint32_t nxC, uint32_t nyC, uint32_t nyA,
		uint32_t ldA, uint32_t ldB, uint32_t ldC,
		float * subA, float * subB, float * subC
)
{
	uint32_t id, i, j, k;
	float sum;

	id = blockIdx.x * blockDim.x + threadIdx.x;
	i = id % nxC;
	j = id / nxC;

	if (j >= nyC){
		return;
	}

	sum = 0.;

	for (k = 0 ; k < nyA ; k++){
		sum += subA[i + k*ldA] * subB[k + j*ldB];
	}

	subC[i + j*ldC] = sum;

}



#define THREADS_PER_BLOCK 64

extern "C" void gpu_mult(void * descr[], void * args)
{

	float * d_subA, * d_subB, * d_subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;
	uint32_t nblocks;

	d_subA = (float *) STARPU_MATRIX_GET_PTR(descr[0]);
	d_subB = (float *) STARPU_MATRIX_GET_PTR(descr[1]);
	d_subC = (float *) STARPU_MATRIX_GET_PTR(descr[2]);

	nxC = STARPU_MATRIX_GET_NX(descr[2]);
	nyC = STARPU_MATRIX_GET_NY(descr[2]);
	nyA = STARPU_MATRIX_GET_NY(descr[0]);

	ldA = STARPU_MATRIX_GET_LD(descr[0]);
	ldB = STARPU_MATRIX_GET_LD(descr[1]);
	ldC = STARPU_MATRIX_GET_LD(descr[2]);

	nblocks = (nxC * nyC + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

	gpuMultKernel
		<<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
		>>> (nxC, nyC, nyA, ldA, ldB, ldC, d_subA, d_subB, d_subC);

	cudaStreamSynchronize(starpu_cuda_get_local_stream());

}
