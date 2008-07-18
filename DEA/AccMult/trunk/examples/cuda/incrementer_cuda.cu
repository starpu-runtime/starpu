#include "incrementer_cuda.h"

extern "C" __global__ 
void cuda_incrementer(uint64_t *foo, float * tab, uint32_t nx, uint32_t pad1, float *unity, uint32_t nx2, uint32_t pad2)
{
//	tab[0] = tab[0] + 1.0f;
//	tab[1] = tab[1] + 0.0f;
//	tab[2] = tab[2] + 1.0f;

	tab[0] = tab[0] + unity[0];
	tab[1] = tab[1] + unity[1];
	tab[2] = tab[2] + unity[2];
	
	*foo = (uint64_t)nx2;

	return;
}
