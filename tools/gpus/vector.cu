#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n)
{
	int i;
	for(i = 0; i < n; i ++)
	{
		out[i] = a[i] + b[i];
	}
}

int main()
{
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	int cnt;
	cudaError_t cures;

	cures = cudaGetDeviceCount(&cnt);
	fprintf(stderr, "count %d\n", cnt);
	if (cures != cudaSuccess)
	{
		fprintf(stderr, "error");
		exit(1);
	}

	// Allocate host memory
	a   = (float*)malloc(sizeof(float) * N);
	b   = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	// Initialize host arrays
	int i;
	for(i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}

	// Allocate device memory
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	// Transfer data from host to device memory
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Executing kernel
	vector_add<<<1,1>>>(d_out, d_a, d_b, N);

	// Transfer data back to host memory
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	// Verification
	for(i = 0; i < N; i++)
	{
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("out[0] = %f\n", out[0]);
	printf("PASSED\n");

	// Deallocate device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	// Deallocate host memory
	free(a);
	free(b);
	free(out);
}
