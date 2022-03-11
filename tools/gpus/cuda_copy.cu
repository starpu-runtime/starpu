#include <stdio.h>
#include <math.h>

#define SRC_DEV 0
#define DST_DEV 1

#define DSIZE (8*1048576)

#define cudaCheckErrors(msg) do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) \
	{ \
		fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
		fprintf(stderr, "*** FAILED - ABORTING\n"); \
		exit(1); \
        } \
} while (0)

int main(int argc, char *argv[])
{
	int devcount;
	int srcdev = SRC_DEV;
	int dstdev = DST_DEV;
	int *d_s, *d_d, *h;
	int canAccessPeer = 0;
	int version;

	cudaDriverGetVersion(&version);
	printf("driver version %d\n", version);
	cudaRuntimeGetVersion(&version);
	printf("runtime version %d\n", version);

	cudaGetDeviceCount(&devcount);
	cudaCheckErrors("cuda failure");

	if (devcount < 2)
	{
		printf("not enough cuda devices for the requested operation\n");
		return 1;
	}

	h = (int *)malloc(DSIZE*sizeof(int));
	if (h == NULL)
	{
		printf("malloc fail\n");
		return 1;
	}
	for (int i = 0; i < DSIZE; i++)
		h[i] = i;

	cudaDeviceCanAccessPeer(&canAccessPeer, srcdev, dstdev);
	cudaCheckErrors("cudaDeviceCanAccessPeer");

	printf("%s of %d bytes\n", canAccessPeer ? "Doing P2P transfer" : "Doing ordinary transfer", DSIZE*sizeof(int));

	cudaSetDevice(srcdev);
	cudaMalloc(&d_s, DSIZE*sizeof(int));
	cudaMemcpy(d_s, h, DSIZE*sizeof(int), cudaMemcpyHostToDevice);

	if (canAccessPeer)
		cudaDeviceEnablePeerAccess(dstdev,0);
	cudaSetDevice(dstdev);

	cudaMalloc(&d_d, DSIZE*sizeof(int));
	cudaCheckErrors("cudaMalloc fail");
	cudaMemset(d_d, 0, DSIZE*sizeof(int));
	cudaCheckErrors("cudaMemset fail");

	if (canAccessPeer)
		cudaDeviceEnablePeerAccess(srcdev,0);

	cudaMemcpyPeer(d_d, dstdev, d_s, srcdev, DSIZE*sizeof(int));
	cudaCheckErrors("cudaMemcpyPeer fail");

	cudaSetDevice(dstdev);
	cudaMemcpy(h, d_d, DSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");

	for (int i = 0; i < DSIZE; i++)
		if (h[i] != i)
		{
			printf("transfer failure\n");
			return 1;
		}
	printf("transfer ok\n");
	return 0;
}
