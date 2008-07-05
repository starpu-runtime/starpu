#include "incrementer_cuda.h"

extern "C" __global__ 
void cuda_incrementer(CUdeviceptr ptr, uint32_t nx, uint32_t ny, uint32_t ld, int foo1, int foo2)
{
	int *tab;
	tab = (int *)ptr;

	tab[0] = ptr;
	tab[1] = nx;
	tab[2] = ny;
	tab[3] = ld;
	tab[4] = foo1*foo2*foo2;
	tab[5] = foo1;
	tab[6] = foo2;
	tab[7] = 0;
	tab[8] = sizeof(CUdeviceptr);
 	tab[9] = 0;

	__syncthreads();

	return;
}
