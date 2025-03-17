#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

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
	hipError_t cures;

	cures = hipGetDeviceCount(&cnt);
	fprintf(stderr, "count %d\n", cnt);
	if (cures != hipSuccess)
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
		a[i] = 21.0f;
		b[i] = 21.0f;
	}

	// Allocate device memory
	hipMalloc((void**)&d_a, sizeof(float) * N);
	hipMalloc((void**)&d_b, sizeof(float) * N);
	hipMalloc((void**)&d_out, sizeof(float) * N);

	// Transfer data from host to device memory
	hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice);
	hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice);

	// Executing kernel
	vector_add<<<1,1>>>(d_out, d_a, d_b, N);

	// Transfer data back to host memory
	hipMemcpy(out, d_out, sizeof(float) * N, hipMemcpyDeviceToHost);

	// Verification
	for(i = 0; i < N; i++)
	{
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("out[0] = %f\n", out[0]);
	printf("PASSED\n");

	// Deallocate device memory
	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_out);

	// Deallocate host memory
	free(a);
	free(b);
	free(out);
}
