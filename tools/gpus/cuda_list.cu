#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <nvml.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv)
{
	struct cudaDeviceProp prop;
	cudaError_t cures;
	int cnt;
	int dev;
	int version;

	cures = cudaGetDeviceCount(&cnt);
	if (cures)
	{
		const char *msg =cudaGetErrorString(cures);
		printf("cudaGetDeviceCount failed: %s (%d)\n", msg, cures);
		exit(1);
	}

	cudaDriverGetVersion(&version);
	printf("driver version %d\n", version);
	cudaRuntimeGetVersion(&version);
	printf("runtime version %d\n", version);

	for (dev = 0; dev < cnt; dev++)
	{
		printf("GPU%d\n", dev);
		cures = cudaGetDeviceProperties(&prop, dev);
		if (cures)
			exit(1);
		printf("\t%s\n", prop.name);
		printf("\tglobal: %0.3f GiB\n", (float) prop.totalGlobalMem / (1 << 30));
		printf("\tshared: %lu KiB\n", (unsigned long) prop.sharedMemPerBlock >> 10);
		printf("\tconst:  %u KiB\n", prop.totalConstMem >> 10);
		printf("\tClock %0.3fGHz\n", (float)(float)  prop.clockRate / (1 << 20));
		printf("\t%u MP\n", prop.multiProcessorCount);
		printf("\tcapability %u.%u\n", prop.major, prop.minor);
		char busid[16];
		snprintf(busid, sizeof(busid), "%04x:%02x:%02x.%x", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
		printf("\tPCI %s\n", busid);
		printf("\t%u multiProcessorCount\n", prop.multiProcessorCount);
		printf("\tasync engine %d\n", prop.asyncEngineCount);
		printf("\tconcurrentKernels %d\n", prop.concurrentKernels);
#if CUDART_VERSION >= 5050
		printf("\tstreamPriorities %d\n", prop.streamPrioritiesSupported);
		printf("\tECC %s\n", prop.ECCEnabled?"on":"off");
#endif
		printf("\n");
	}

	return 0;
}
