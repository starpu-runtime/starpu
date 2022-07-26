#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(void)
{
	int n, i, version;
	cudaError_t err;
	err = cudaGetDeviceCount(&n);
	if (err)
	{
		fprintf(stderr,"cuda error %d\n", err);
		exit(1);
	}
	cudaDriverGetVersion(&version);
	printf("driver version %d\n", version);
	cudaRuntimeGetVersion(&version);
	printf("runtime version %d\n", version);
	printf("\n");
	for (i = 0; i < n; i++)
	{
		struct cudaDeviceProp props;
		printf("CUDA%d\n", i);
		err = cudaGetDeviceProperties(&props, i);
		if (err)
		{
			fprintf(stderr,"cuda error %d\n", err);
			continue;
		}
		printf("%s\n", props.name);
		printf("%0.3f GB\n", (float) props.totalGlobalMem / (1<<30));
		printf("%u MP\n", props.multiProcessorCount);
		printf("\n");
	}
	return 0;
}
