#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int main(void)
{
	int i, cnt;
	hipError_t hipres;
	hipres = hipGetDeviceCount(&cnt);
	if (hipres)
	{
		fprintf(stderr,"hip error: <%s>\n", hipGetErrorString(hipres));
		exit(1);
	}
	printf("number of hip devices: %d\n", cnt);
	for (i = 0; i < cnt; i++)
	{
		struct hipDeviceProp_t props;
		printf("HIP%d\n", i);
		hipres = hipGetDeviceProperties(&props, i);
		if (hipres)
		{
			fprintf(stderr,"hip error: <%s>\n", hipGetErrorString(hipres));
			continue;
		}
		printf("%s\n", props.name);
		printf("%0.3f GB\n", (float) props.totalGlobalMem / (1<<30));
		printf("%u MP\n", props.multiProcessorCount);
		printf("\n");
	}
	return 0;
}
